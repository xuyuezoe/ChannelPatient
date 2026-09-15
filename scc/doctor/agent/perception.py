"""感知器：把患者的一句话转成各槽上的软观测（SemanticObs）。

RegexPerceiver（stub）：用簇配置的显示词、模糊词、程度词、功能影响词、withheld 短语反查报告值。
  模板患者的话是这些词拼出来的，所以 stub 模式下反解是可逆的、确定的。
LLMPerceiver：让 LLM 给出每个字段的分布 + 方式信号；失败回退到 RegexPerceiver 并计数。
两者接口相同：perceive(utterance, asked, lang) -> list[SemanticObs]。asked 是本轮问的候选（可为 None）。
"""
from __future__ import annotations
import json
from typing import Protocol
from scc.env.cases import ClusterConfig
from scc.doctor.agent.state import SemanticObs, Candidate, UNKNOWN, NOT_MENTIONED
from scc.types import QUESTION_FORMS

BOOL_SLOTS = ("trigger", "relief", "associated", "family_history", "medical_history")
NEG_MARKERS = {"en": ["no ", "no,", "no.", "not ", "haven't", "don't", "doesn't", "never", "nothing", "isn't", "hasn't"], "zh": ["没有", "不", "没"]}


class Perceiver(Protocol):
    def perceive(self, utterance: str, asked: Candidate | None, lang: str) -> list[SemanticObs]: ...


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase) and phrase.lower() in text.lower()


class RegexPerceiver:
    """基于显示词反查的确定性感知器（零 LLM）。"""
    name = "regex"

    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg
        self._tables = {lang: self._build(lang) for lang in ("en", "zh")}

    def _build(self, lang: str) -> dict:
        """(slot, tag) -> [(phrase, value)]，phrase 越长越优先。"""
        c = self.cfg; t: dict[tuple, list] = {}
        for slot, sd in c.slots.items():
            if slot == "chief_complaint":
                continue
            if sd["type"] in ("categorical", "ordinal_3"):
                opts = sd.get("options") or []
                tag_vals = sd.get("tag_values")
                if tag_vals:
                    for tag, vals in tag_vals.items():
                        for v in vals:
                            for ph in [c.display(slot, v, lang)] + (sd.get("downplay", {}).get(lang, {}).get(v, "") and [sd["downplay"][lang][v]] or []):
                                t.setdefault((slot, tag), []).append((ph, v))
                else:
                    for v in opts:
                        phrases = [c.display(slot, v, lang)]
                        if slot == "severity":
                            phrases = c.severity_words(v, lang) + phrases
                        if slot == "functional_impact":
                            phrases = c.functional_words(v, lang) + phrases
                        for ph in phrases:
                            t.setdefault((slot, None), []).append((ph, v))
                    for v in opts:
                        tok = c.vague_token(slot, v)
                        if tok:
                            t.setdefault((slot, None), []).append((c.display(slot, tok, lang), tok))
            elif sd["type"] == "bool":
                for tag in sd.get("tag_options", []):
                    t.setdefault((slot, tag), []).append((c.tag_display(slot, tag, lang), None))   # value decided by negation
        for k in t:
            t[k].sort(key=lambda pv: -len(pv[0]))
        return t

    def _bool_value(self, utterance: str, phrase: str, lang: str) -> bool:
        """在 phrase 前后一小段里找否定词。"""
        low = utterance.lower(); i = low.find(phrase.lower())
        window = low[max(0, i - 25): i + len(phrase) + 16]
        return not any(m in window for m in NEG_MARKERS[lang])

    def perceive(self, utterance: str, asked: Candidate | None, lang: str = "en") -> list[SemanticObs]:
        c = self.cfg; table = self._tables[lang]; out = []
        withheld_hit = any(_contains(utterance, w) for w in c.phrases("withheld_phrases", lang))
        flooding = 1.0 if any(_contains(utterance, w) for w in c.phrases("flooding_phrases", lang)) else 0.0
        intensity = 0.0
        for lvl, val in (("mild", 0.2), ("moderate", 0.6), ("severe", 1.0)):
            if any(_contains(utterance, w) for w in c.severity_words(lvl, lang)):
                intensity = max(intensity, val)
        form = asked.form if asked and asked.form else "open"
        seen: set[tuple] = set()
        for (slot, tag), pairs in table.items():
            hits = []
            for ph, v in pairs:
                if _contains(utterance, ph):
                    if c.slots[slot]["type"] == "bool":
                        v = self._bool_value(utterance, ph, lang)
                    if v not in [h for h in hits]:
                        hits.append(v)
                    break                                  # 最长匹配优先，每槽取一个
            if hits:
                dist = {h: 1.0 / len(hits) for h in hits}
                out.append(SemanticObs(slot, tag, form, dist, {"intensity": intensity, "flooding": flooding, "hedging": 1.0 if len(hits) > 1 else 0.0, "self_attribution": 0.0}, raw="regex"))
                seen.add((slot, tag))
        # 问了但没答：只对被问的槽记 not_mentioned / unknown
        if asked and asked.slot and (asked.slot, asked.tag) not in seen and asked.kind.value == "ASK":
            dist = {NOT_MENTIONED: 1.0} if withheld_hit else {UNKNOWN: 1.0}
            out.append(SemanticObs(asked.slot, asked.tag, form, dist, {"intensity": intensity, "flooding": flooding, "hedging": 0.0, "self_attribution": 0.0}, raw="regex:missing"))
        return out


class LLMPerceiver:
    """LLM 感知器：给出各字段分布 + 方式信号；解析失败回退正则。"""
    name = "llm"

    def __init__(self, cfg: ClusterConfig, client, fallback: RegexPerceiver | None = None):
        self.cfg, self.client = cfg, client
        self.fallback = fallback or RegexPerceiver(cfg)
        self.n_fallback = 0; self.n_calls = 0

    def _fields_text(self, asked: Candidate | None, lang: str) -> tuple[list[tuple[str, str | None, list]], str]:
        c = self.cfg; fields = []
        slots = [(asked.slot, asked.tag)] if asked and asked.slot else []
        # 顺带可能听到的槽：伴随症状与触发（患者常一句带出）
        for slot, tag in slots:
            sd = c.slots[slot]
            if sd["type"] == "bool":
                vals = ["yes", "no"]
            else:
                vals = list(sd.get("options") or c.slot_options(slot, tag) or [])
                vals += sorted({c.vague_token(slot, v) for v in vals} - {None})
            fields.append((slot, tag, vals))
        text = "\n".join(f"- {s}{'/' + t if t else ''}: {', '.join(map(str, v))}, unknown, not_mentioned" for s, t, v in fields)
        return fields, text

    def perceive(self, utterance: str, asked: Candidate | None, lang: str = "en") -> list[SemanticObs]:
        from scc.doctor import prompts
        fields, ftext = self._fields_text(asked, lang)
        if not fields:
            return self.fallback.perceive(utterance, asked, lang)
        schema = {"type": "object", "properties": {"fields": {"type": "object"}, "manner": {"type": "object"}}, "required": ["fields", "manner"], "additionalProperties": True}
        asked_txt = f"{asked.slot}{'/' + asked.tag if asked.tag else ''} ({asked.form})" if asked else "(open)"
        try:
            out = self.client.chat([{"role": "system", "content": prompts.get("PERCEIVE_SYS", lang)},
                                    {"role": "user", "content": prompts.get("PERCEIVE_USER", lang).format(asked=asked_txt, fields=ftext, utterance=utterance)}],
                                   max_tokens=300, temperature=0.0, json_schema=schema)
            self.n_calls += 1
            obs = []
            manner = {k: float(out.get("manner", {}).get(k, 0.0)) for k in ("intensity", "flooding", "self_attribution", "hedging")}
            for slot, tag, vals in fields:
                key = f"{slot}/{tag}" if tag else slot
                raw = out["fields"].get(key) or out["fields"].get(slot) or {}
                dist = {}
                for k, v in raw.items():
                    kk = {"yes": True, "no": False, "true": True, "false": False}.get(str(k).lower(), k)
                    dist[kk] = max(0.0, float(v))
                s = sum(dist.values())
                if s <= 0:
                    dist = {UNKNOWN: 1.0}
                else:
                    dist = {k: v / s for k, v in dist.items()}
                obs.append(SemanticObs(slot, tag, asked.form if asked and asked.form else "open", dist, manner, raw="llm"))
            return obs
        except Exception:
            self.n_fallback += 1
            r = self.fallback.perceive(utterance, asked, lang)
            for o in r:
                o.raw = "llm_fallback"
            return r
