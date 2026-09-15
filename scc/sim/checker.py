"""Checker: is the utterance faithful to the TurnPlan? RegexChecker (no LLM) and NLIChecker (LLM)."""
from __future__ import annotations
import json, re
from typing import Protocol
from scc.types import TurnPlan, CheckResult, Case
from scc.env.cases import ClusterConfig


class Checker(Protocol):
    def check(self, utterance: str, plan: TurnPlan, case: Case, lang: str) -> CheckResult: ...


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase) and phrase.lower() in text.lower()


class RegexChecker:
    name = "regex"

    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg

    def _slot_words(self, slot: str, value, lang: str) -> list[str]:
        d = self.cfg.display(slot, value, lang)
        if slot == "severity":
            return self.cfg.severity_words(value, lang)
        if slot == "functional_impact":
            return self.cfg.functional_words(value, lang)
        return [d]

    def check(self, utterance: str, plan: TurnPlan, case: Case, lang: str = "en") -> CheckResult:
        v: list[str] = []
        c = self.cfg
        allowed_label = c.dx_display(plan.self_dx[0], lang).lower() if plan.self_dx else None
        for term in c.vocab.get("red_flag_terms", {}).get(lang, []) + c.vocab.get("diagnosis_terms", {}).get(lang, []):
            if _contains(utterance, term) and not (allowed_label and term.lower() in allowed_label):
                v.append(f"diagnosis_term:{term}")
        for a in plan.withheld:
            words = []
            if a.tag:
                words.append(c.tag_display(a.slot, a.tag, lang))
            if a.type in ("categorical", "ordinal_3") and a.options:
                for o in a.options:
                    words += self._slot_words(a.slot, o, lang)
            for w in words:
                if _contains(utterance, w):
                    v.append(f"withheld_mentioned:{a.id}:{w}")
        for a, val in plan.say:
            if a.type in ("categorical", "ordinal_3") and a.options and val in a.options:
                for o in a.options:
                    if o == val:
                        continue
                    for w in self._slot_words(a.slot, o, lang):
                        if _contains(utterance, w) and not any(_contains(w2, w) for w2 in self._slot_words(a.slot, val, lang)):
                            v.append(f"wrong_value:{a.id}:{w}")
        return CheckResult(passed=not v, violations=v, retry_hint="; ".join(v[:3]))


class NLIChecker(RegexChecker):
    """Regex first; then one LLM call listing medical facts not covered by the allowed content."""
    name = "nli"

    def __init__(self, cfg: ClusterConfig, client, phraser_for_allowed=None):
        super().__init__(cfg)
        self.client = client
        self.allowed_fn = phraser_for_allowed

    def check(self, utterance: str, plan: TurnPlan, case: Case, lang: str = "en") -> CheckResult:
        res = super().check(utterance, plan, case, lang)
        if not res.passed:
            return res
        from scc.sim import prompts
        from scc.sim.phraser import TemplatePhraser
        allowed = "\n".join("- " + TemplatePhraser(self.cfg)._one(a, v, lang) for a, v in plan.say) or "- (nothing)"
        if plan.self_dx:
            allowed += f"\n- (belief) {self.cfg.dx_display(plan.self_dx[0], lang)}"
        schema = {"type": "object", "properties": {"new_medical_facts": {"type": "array", "items": {"type": "string"}}}, "required": ["new_medical_facts"], "additionalProperties": False}
        try:
            out = self.client.chat([{"role": "user", "content": prompts.get("ATOMIC_FACTS", lang).format(allowed=allowed, utterance=utterance)}], max_tokens=200, temperature=0.0, json_schema=schema)
            facts = [f for f in out.get("new_medical_facts", []) if str(f).strip()]
        except Exception as e:
            return CheckResult(True, [], retry_hint=f"nli_unavailable:{type(e).__name__}")
        if facts:
            return CheckResult(False, [f"new_fact:{f}" for f in facts], retry_hint="remove: " + "; ".join(facts[:3]))
        return res
