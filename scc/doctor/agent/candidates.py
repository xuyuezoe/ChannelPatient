"""候选动作生成。

MenuCandidates（stub，零 LLM）：从槽位表枚举 (槽位 × 适用问法) + 锚定动作 + 下诊断，文本用模板句。
LLMCandidates：让 LLM 针对当前 top-2 假设生成问句，用关键词分类器抽象成 (槽位, 问法)，再合并菜单里的锚定与红旗必问句。
两者接口相同：propose(state_view) -> list[Candidate]。state_view 是 agent 传来的只读字典：
  {"top2": [z1, z2], "channel_guess": str, "asked": {slot|tag: n}, "transcript": [...], "lang": str, "red_flag": str}
"""
from __future__ import annotations
import json
from typing import Protocol
from scc.env.cases import ClusterConfig
from scc.types import ActionKind
from scc.doctor.agent.state import Candidate

CALIBRATION_FORMS = ("forced_choice", "point_to")


def _q(text: str) -> str:
    """把患者视角的短语改成医生提问视角（I -> you, my -> your）。"""
    import re
    text = re.sub(r"\bI\b", "you", text); text = re.sub(r"\bmy\b", "your", text); text = re.sub(r"\bme\b", "you", text)
    return text


def _opts_text(cfg: ClusterConfig, slot: str, lang: str) -> str:
    opts = cfg.slots[slot]["options"]
    ch = cfg.slots[slot].get("choice", {}).get(lang, {})
    disp = [ch.get(o, cfg.display(slot, o, lang)) for o in opts]
    if lang == "zh":
        return "、".join(disp[:-1]) + "，还是" + disp[-1]
    return ", ".join(disp[:-1]) + ", or " + disp[-1]


class CandidateGenerator(Protocol):
    def propose(self, view: dict) -> list[Candidate]: ...


class MenuCandidates:
    """确定性菜单。"""
    name = "menu"

    def __init__(self, cfg: ClusterConfig, anchoring: bool = True, lang: str = "en"):
        self.cfg, self.anchoring, self.lang = cfg, anchoring, lang
        self._menu = self._build(lang)

    def _build(self, lang: str) -> list[Candidate]:
        c = self.cfg; out: list[Candidate] = []
        for slot, sd in c.slots.items():
            if slot == "chief_complaint":
                continue
            reask = c.reask(slot, lang)
            t = sd["type"]
            if t == "text":
                out.append(Candidate(ActionKind.ASK, reask[0] if reask else ("Have you had any tests done before? What did they show?" if lang == "en" else "以前做过检查吗？结果怎么说？"), slot, None, "recall_test"))
            elif t == "ordinal_3":
                q = reask[0] if reask else ("How bad is the pain?" if lang == "en" else "疼得厉害吗？")
                out.append(Candidate(ActionKind.ASK, q, slot, None, "severity_open"))
            elif t == "categorical" and sd.get("options"):
                q_open = reask[0] if reask else (f"Tell me about the {slot.replace('_', ' ')}." if lang == "en" else f"说说{slot}。")
                out.append(Candidate(ActionKind.ASK, q_open, slot, None, "open"))
                q_fc = (f"Is it {_opts_text(c, slot, lang)}?" if lang == "en" else f"是{_opts_text(c, slot, lang)}？")
                out.append(Candidate(ActionKind.ASK, q_fc, slot, None, "forced_choice", is_calibration=True))
                if slot == "location":
                    out.append(Candidate(ActionKind.ASK, "Can you point to exactly where it hurts?" if lang == "en" else "你用手指一下具体疼的地方。", slot, None, "point_to", is_calibration=True))
            elif t == "categorical" and sd.get("tag_values"):
                for tag in sd["tag_options"]:
                    td = c.tag_display(slot, tag, lang) if sd.get("tag_display") else tag
                    q = {"smoking": "Do you smoke?", "alcohol": "Do you drink alcohol?"}.get(tag, f"Tell me about your {tag}.") if lang == "en" else {"smoking": "抽烟吗？", "alcohol": "喝酒吗？"}.get(tag, f"说说{tag}。")
                    out.append(Candidate(ActionKind.ASK, q, slot, tag, "yes_no"))
            elif t == "bool":
                for tag in sd.get("tag_options", []):
                    td = _q(c.tag_display(slot, tag, lang)) if lang == "en" else c.tag_display(slot, tag, lang)
                    if slot == "trigger":
                        q = f"Does the pain get worse {td}?" if lang == "en" else f"{td}会更疼吗？"
                    elif slot == "relief":
                        q = f"Does {td} make it better?" if lang == "en" else f"{td}会好一点吗？"
                    elif slot == "associated":
                        q = f"Have you had any {td} with it?" if lang == "en" else f"有没有{td}？"
                    elif slot == "family_history":
                        q = f"Is there any {td}?" if lang == "en" else f"{td}吗？"
                    else:
                        q = f"Do you have {td}?" if lang == "en" else f"有{td}吗？"
                    out.append(Candidate(ActionKind.ASK, q, slot, tag, "yes_no"))
        if self.anchoring:
            out.append(Candidate(ActionKind.VERIFY_RECORD, "VERIFY RECORD: prior_severity", "prior_severity", None, "verify", target="prior_severity", is_calibration=True))
            out.append(Candidate(ActionKind.VERIFY_RECORD, "VERIFY RECORD: prior_test", "prior_test", None, "verify", target="prior_test", is_calibration=True))
            out.append(Candidate(ActionKind.ASK_FAMILY, "ASK FAMILY: heart disease", "family_history", "cad", "verify", target="heart disease", is_calibration=True))
        return out

    def propose(self, view: dict) -> list[Candidate]:
        return [Candidate(**{**c.__dict__}) for c in self._menu]


class LLMCandidates(MenuCandidates):
    """LLM 生成自然问句，再抽象到菜单坐标；锚定动作与红旗问句从菜单补齐。"""
    name = "llm"

    def __init__(self, cfg: ClusterConfig, client, classifier, anchoring: bool = True, lang: str = "en", n: int = 6, n_clar: int = 2):
        super().__init__(cfg, anchoring, lang)
        self.client, self.clf, self.n, self.n_clar = client, classifier, n, n_clar
        self.n_calls = 0; self.n_fallback = 0
        from scc.env.likelihood import Likelihood  # noqa: F401  (保持依赖显式)

    def propose(self, view: dict) -> list[Candidate]:
        from scc.doctor import prompts
        from scc.doctor.agent.likelihood import pseudo_atom
        lang = view.get("lang", self.lang)
        tr = "\n".join(f"{r}: {t}" for r, t in view.get("transcript", [])[-8:]) or "(start)"
        asked = ", ".join(f"{k}({v})" for k, v in view.get("asked", {}).items()) or "none"
        top2 = ", ".join(self.cfg.dx_display(z, lang) for z in view.get("top2", [])[:2])
        msgs = [{"role": "system", "content": prompts.get("CANDIDATES_SYS", lang)},
                {"role": "user", "content": prompts.get("CANDIDATES_USER", lang).format(transcript=tr, top2=top2, channel_guess=view.get("channel_guess", "cooperative"), asked=asked, n=self.n, n_clar=self.n_clar, red_flag=self.cfg.dx_display(view.get("red_flag", self.cfg.red_flags[0] if self.cfg.red_flags else ""), lang))}]
        schema = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}}, "required": ["questions"], "additionalProperties": False}
        try:
            out = self.client.chat(msgs, max_tokens=400, temperature=0.3, json_schema=schema); self.n_calls += 1
            qs = [q for q in out.get("questions", []) if isinstance(q, str) and q.strip()]
        except Exception:
            self.n_fallback += 1; qs = []
        # 抽象：用关键词分类器把问句映射到 (槽, 标签, 问法)；用一个只含槽位的伪 atom 列表
        atoms = []
        for slot, sd in self.cfg.slots.items():
            if slot == "chief_complaint":
                continue
            for tag in (sd.get("tag_options") or [None]):
                atoms.append(pseudo_atom(self.cfg, slot, tag))
        cands: list[Candidate] = []
        for q in qs:
            co = self.clf.classify(q, atoms, [], lang)
            if not co.atom_ids:
                continue
            aid = co.atom_ids[0]; slot, tag = aid.split(":")[0], (aid.split(":")[1] or None)
            form = co.question_form if co.question_form != "chat" else "open"
            cands.append(Candidate(ActionKind.ASK, q.strip(), slot, tag, form, is_calibration=form in CALIBRATION_FORMS, source="llm"))
        menu = super().propose(view)
        seen = {c.key() for c in cands}
        for m in menu:
            if m.kind != ActionKind.ASK or m.key() not in seen:
                cands.append(m)
        return cands
