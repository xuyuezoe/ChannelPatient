"""Pure-code routing of anchoring / test / diagnose actions (never through the patient)."""
from __future__ import annotations
import re
from typing import Any
from scc.types import ActionKind, Case, DoctorAction, Observation, ObsKind, Atom
from scc.env.cases import ClusterConfig


def _match(target: str | None, atom: Atom, cfg: ClusterConfig) -> bool:
    if not target:
        return True
    t = target.lower().replace("_", " ")
    cand = [atom.id, atom.slot, *(atom.tags or [])]
    for c in cand:
        if c.lower().replace("_", " ") in t or t in c.lower().replace("_", " "):
            return True
    kws: list[str] = []
    for lang in ("en", "zh"):
        k = cfg.keywords(atom.slot, lang)
        if isinstance(k, dict):
            for tag, words in k.items():
                if not atom.tags or tag in atom.tags:
                    kws += list(words)
        else:
            kws += list(k)
    return any(str(k).lower() in t for k in kws)


def route(action: DoctorAction, case: Case, rules, cfg: ClusterConfig, lang: str, turn: int) -> tuple[Observation, list[dict]]:
    """Returns (observation, disclosed-entries-for-log)."""
    if action.kind == ActionKind.VERIFY_RECORD:
        atoms = [a for a in case.verifiable("record") if _match(action.target, a, cfg)]
        exact = [a for a in atoms if action.target and a.id.lower() == action.target.strip().lower()]
        atoms = exact or atoms
        if not atoms:
            msg = "无相关记录。" if lang == "zh" else f"No record available for {action.target or 'that'}."
            return Observation(ObsKind.RECORD, msg, turn), []
        a = atoms[0]; rules.on_verified(a.id)
        if a.type == "text":
            text = ("记录原文：" if lang == "zh" else "Record: ") + str(a.true_value)
        else:
            label = cfg.tag_display(a.slot, a.tag, lang) if a.tag else a.slot.replace("_", " ")
            text = (f"就诊记录：{label} 记为 {a.true_value}" if lang == "zh" else f"Prior visit note: {label} recorded as '{a.true_value}'.")
        return Observation(ObsKind.RECORD, text, turn), [{"atom": a.id, "true_value": a.true_value, "report_value": a.true_value, "rule": "verified:record", "n_asked": rules.n_asked[a.id]}]
    if action.kind == ActionKind.ASK_FAMILY:
        atoms = [a for a in case.verifiable("family") if _match(action.target, a, cfg)]
        if not atoms:
            msg = "家属：这个我不清楚。" if lang == "zh" else "Family member: I'm not sure about that."
            return Observation(ObsKind.FAMILY, msg, turn), []
        a = atoms[0]; rules.on_verified(a.id)
        val = cfg.display(a.slot, a.true_value, lang) if a.type != "text" else str(a.true_value)
        tagd = cfg.tag_display(a.slot, a.tag, lang) if a.tag else a.slot
        body = f"{tagd}: {val}" if lang == "en" else f"{tagd}：{val}"
        tmpl = cfg.vocab.get("family_template", {}).get(lang, "Family member: {text}")
        return Observation(ObsKind.FAMILY, tmpl.format(text=body), turn), [{"atom": a.id, "true_value": a.true_value, "report_value": a.true_value, "rule": "verified:family", "n_asked": rules.n_asked[a.id]}]
    if action.kind == ActionKind.REQUEST_TEST:
        tests = case.exam_and_tests.get("Test_Results", {}); exam = case.exam_and_tests.get("Physical_Examination_Findings", {})
        t = (action.target or action.text or "").lower()
        for name, val in {**tests, **exam}.items():
            key = name.lower().replace("_", " ")
            if key in t or t in key or any(w in t for w in key.split() if len(w) > 3):
                return Observation(ObsKind.TEST, f"RESULTS: {name}: {val}", turn), []
        return Observation(ObsKind.TEST, "RESULTS: NORMAL READINGS", turn), []
    if action.kind == ActionKind.DIAGNOSE:
        return Observation(ObsKind.END, action.text, turn), []
    raise ValueError(f"route() does not handle {action.kind}")
