"""Classifier: doctor utterance -> (atom ids hit, question form). Keyword stub + LLM implementation."""
from __future__ import annotations
import json, re
from typing import Protocol
from scc.types import Atom, ClassifierOutput, QUESTION_FORMS
from scc.env.cases import ClusterConfig

FORM_RULES = {
    "en": {"point_to": [r"\bpoint\b", r"show me", r"put your (hand|finger)"],
           "recall_test": [r"last time", r"previous", r"\bbefore\b.*(test|check|ecg|ekg|scan|result)", r"\b(test|ecg|ekg|scan|endoscopy|x-ray|result)s?\b.*(say|show|find|result|before|had)", r"had .* (done|checked)", r"what did .* (say|show|find)"],
           "forced_choice": [r"\bor\b", r"which of"],
           "scale": [r"\b0 to 10\b", r"\bzero to ten\b", r"\bscale\b", r"out of 10", r"\brate\b"],
           "severity_open": [r"how bad", r"how severe", r"how strong", r"how intense", r"bearable", r"put up with", r"how much does it hurt", r"how painful", r"keep you up", r"wake you"],
           "yes_no": [r"^(do|does|did|have|has|is|are|were|was|can|could|any)\b", r"\bany\b", r"ever\b"]},
    "zh": {"point_to": [r"指一下", r"指给", r"用手"],
           "recall_test": [r"上次", r"以前", r"之前.*(查|检查|结果)", r"(检查|心电图|胃镜|化验).*(怎么说|结果|说什么|查过)", r"查过"],
           "forced_choice": [r"还是"],
           "scale": [r"几分", r"0到10", r"零到十", r"打分"],
           "severity_open": [r"厉害", r"严重", r"能忍", r"多疼", r"疼得", r"影响", r"睡"],
           "yes_no": [r"有没有", r"是否", r"吗[？?]?$", r"有.*吗", r"会不会"]},
}
QUESTION_MARK = re.compile(r"[?？]")
WH_EN = re.compile(r"\b(what|where|when|how|which|why|tell me|describe)\b", re.I)
WH_ZH = re.compile(r"(什么|哪|怎么|多久|多长|几|如何|说说|描述)")


class Classifier(Protocol):
    def classify(self, question: str, atoms: list[Atom], transcript: list, lang: str) -> ClassifierOutput: ...


class KeywordClassifier:
    name = "keyword"

    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg

    def _form(self, q: str, lang: str) -> str:
        ql = q.lower()
        for form in ("point_to", "recall_test", "scale", "forced_choice", "severity_open", "yes_no"):
            for pat in FORM_RULES[lang][form]:
                if re.search(pat, ql, re.I):
                    return form
        return "open"

    @staticmethod
    def _has(word: str, ql: str, lang: str) -> bool:
        w = str(word).lower()
        if lang == "en" and re.fullmatch(r"[a-z0-9 '\-]+", w):
            return re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", ql) is not None
        return w in ql

    def classify(self, question: str, atoms: list[Atom], transcript: list, lang: str) -> ClassifierOutput:
        ql = question.lower()
        hits: list[str] = []
        for a in atoms:
            kw = self.cfg.keywords(a.slot, lang)
            words = []
            if isinstance(kw, dict):
                for t in (a.tags or []):
                    words += list(kw.get(t, []))
            else:
                words = list(kw)
            if any(self._has(w, ql, lang) for w in words):
                hits.append(a.id)
        # chief_complaint keywords ("what brings you in") mean a general open question, not a chief-complaint atom hit
        cc = [a.id for a in atoms if a.slot == "chief_complaint"]
        general = any(h in cc for h in hits)
        hits = [h for h in hits if h not in cc]
        if general:
            hits = []
        by_id = {a.id: a for a in atoms}
        if any(by_id[h].slot == "family_history" for h in hits):        # "family ... heart disease" is about the family
            hits = [h for h in hits if by_id[h].slot != "medical_history"]
        form = self._form(question, lang)
        if form == "forced_choice" and hits and all(by_id[h].type == "bool" for h in hits):
            form = "yes_no"                                              # "any sweating or nausea?" is a yes/no question
        if general and not hits:
            form = "open"
        is_chat = not hits and not general and not QUESTION_MARK.search(question) and not (WH_EN.search(question) if lang == "en" else WH_ZH.search(question))
        if is_chat:
            form = "chat"
        return ClassifierOutput(hits, form, is_chat, raw="keyword")


class LLMClassifier:
    name = "llm"

    def __init__(self, cfg: ClusterConfig, client, fallback: KeywordClassifier | None = None):
        self.cfg, self.client = cfg, client
        self.fallback = fallback or KeywordClassifier(cfg)

    def _atom_menu(self, atoms: list[Atom], lang: str) -> str:
        lines = []
        for a in atoms:
            desc = self.cfg.tag_display(a.slot, a.tag, lang) if a.tag else a.slot.replace("_", " ")
            lines.append(f"- {a.id}: {a.slot} / {desc}")
        return "\n".join(lines)

    def classify(self, question: str, atoms: list[Atom], transcript: list, lang: str) -> ClassifierOutput:
        from scc.sim import prompts
        schema = {"type": "object", "properties": {"atom_ids": {"type": "array", "items": {"type": "string"}},
                                                   "question_form": {"type": "string", "enum": QUESTION_FORMS},
                                                   "is_chat": {"type": "boolean"}}, "required": ["atom_ids", "question_form", "is_chat"], "additionalProperties": False}
        msgs = [{"role": "system", "content": prompts.get("CLASSIFIER_SYS", lang)},
                {"role": "user", "content": prompts.get("CLASSIFIER_USER", lang).format(menu=self._atom_menu(atoms, lang), question=question)}]
        try:
            out = self.client.chat(msgs, max_tokens=200, temperature=0.0, json_schema=schema)
            valid = {a.id for a in atoms}
            ids = [i for i in out.get("atom_ids", []) if i in valid]
            form = out.get("question_form", "open")
            if form not in QUESTION_FORMS:
                form = "open"
            return ClassifierOutput(ids, form, bool(out.get("is_chat", False)), raw=json.dumps(out, ensure_ascii=False))
        except Exception as e:  # network / parse failure -> keyword fallback, flagged in raw
            r = self.fallback.classify(question, atoms, transcript, lang)
            r.raw = f"fallback:{type(e).__name__}"
            return r
