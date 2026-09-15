"""Load and validate cases and cluster configuration."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from scc.config import settings
from scc.types import Case, Atom, disclosure_from

VALID_DISCLOSURE = {"spontaneous", "on_general_ask", "on_specific_ask"}


@dataclass
class ClusterConfig:
    name: str
    ddx_set: list[str]
    red_flags: list[str]
    display_dx: dict
    plausible_wrong: dict
    u_library: dict
    prior_u: Any
    prior_z: Any
    slots: dict
    vocab: dict
    raw: dict = field(default_factory=dict)

    # ---- display helpers
    def display(self, slot: str, value: Any, lang: str = "en") -> str:
        sd = self.slots.get(slot, {})
        disp = sd.get("display", {}).get(lang, {})
        for key in (value, "true" if value is True else "false" if value is False else None, str(value)):
            if key is not None and key in disp:
                return disp[key]
        vd = self.vocab.get("vague_display", {})
        if isinstance(value, str) and value in vd:
            return vd[value].get(lang, value)
        return str(value)

    def tag_display(self, slot: str, tag: str, lang: str = "en") -> str:
        return self.slots.get(slot, {}).get("tag_display", {}).get(lang, {}).get(tag, tag)

    def dx_display(self, dx: str, lang: str = "en") -> str:
        return self.display_dx.get(dx, {}).get(lang, dx)

    def keywords(self, slot: str, lang: str = "en") -> Any:
        return self.slots.get(slot, {}).get("keywords", {}).get(lang, [])

    def reask(self, slot: str, lang: str = "en") -> list[str]:
        return self.slots.get(slot, {}).get("reask", {}).get(lang, [])

    def vague_token(self, slot: str, value: Any) -> str | None:
        return self.vocab.get("vague_map", {}).get(slot, {}).get(value)

    def vague_tokens(self) -> set[str]:
        return set(self.vocab.get("vague_display", {}).keys())

    def severity_words(self, level: str, lang: str = "en") -> list[str]:
        return self.vocab.get("severity_words", {}).get(level, {}).get(lang, [level])

    def functional_words(self, level: str, lang: str = "en") -> list[str]:
        return self.vocab.get("functional_words", {}).get(level, {}).get(lang, [level])

    def phrases(self, key: str, lang: str = "en") -> list[str]:
        v = self.vocab.get(key, {})
        return v.get(lang, []) if isinstance(v, dict) else []

    def slot_options(self, slot: str, tag: str | None = None) -> list | None:
        sd = self.slots.get(slot, {})
        if tag and "tag_values" in sd:
            return sd["tag_values"].get(tag)
        return sd.get("options")


def load_cluster_config(name: str) -> ClusterConfig:
    d = settings.clusters_dir / name
    c = yaml.safe_load(open(d / "cluster.yaml"))
    slots = yaml.safe_load(open(d / "slots.yaml"))
    vocab = yaml.safe_load(open(d / "vocab.yaml"))
    return ClusterConfig(name=c["name"], ddx_set=list(c["ddx_set"]), red_flags=list(c.get("red_flags", [])),
                         display_dx=c.get("display", {}), plausible_wrong=c.get("plausible_wrong", {}),
                         u_library=c.get("u_library", {}), prior_u=c.get("prior_u", "uniform"),
                         prior_z=c.get("prior_z", "uniform"), slots=slots, vocab=vocab, raw=c)


def load_case(path: str | Path) -> Case:
    return Case.from_dict(json.load(open(path)))


def load_cluster(name: str, ids: list[str] | None = None) -> list[Case]:
    d = settings.clusters_dir / name / "cases"
    cases = [load_case(p) for p in sorted(d.glob("*.json"))]
    if ids:
        cases = [c for c in cases if c.case_id in ids]
    return cases


def validate(case: Case, cfg: ClusterConfig) -> list[str]:
    errs: list[str] = []
    ids = [a.id for a in case.atoms]
    if len(ids) != len(set(ids)):
        errs.append("duplicate atom ids: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    if case.labels.primary_dx not in case.labels.ddx_set:
        errs.append(f"primary_dx {case.labels.primary_dx} not in ddx_set")
    for r in case.labels.red_flags:
        if r not in case.labels.ddx_set:
            errs.append(f"red flag {r} not in ddx_set")
    if not any(a.disclosure == "spontaneous" for a in case.atoms):
        errs.append("no spontaneous atom")
    if not any(a.verifiable_by for a in case.atoms):
        errs.append("no verifiable atom")
    for a in case.atoms:
        if a.slot not in cfg.slots:
            errs.append(f"{a.id}: unknown slot {a.slot}"); continue
        sd = cfg.slots[a.slot]
        if a.type != sd["type"]:
            errs.append(f"{a.id}: type {a.type} != slot type {sd['type']}")
        if a.type in ("categorical", "ordinal_3"):
            opts = a.options or cfg.slot_options(a.slot, a.tag)
            if not opts:
                errs.append(f"{a.id}: no options")
            elif a.true_value not in opts:
                errs.append(f"{a.id}: true_value {a.true_value!r} not in options {opts}")
        if a.type == "bool" and not isinstance(a.true_value, bool):
            errs.append(f"{a.id}: bool atom with non-bool value {a.true_value!r}")
        if "tag_options" in sd:
            if not a.tags:
                errs.append(f"{a.id}: slot {a.slot} requires a tag")
            for t in a.tags:
                if t not in sd["tag_options"]:
                    errs.append(f"{a.id}: tag {t} not in {sd['tag_options']}")
        disc = a.disclosure
        if not (disc in VALID_DISCLOSURE or (isinstance(disc, tuple) and disc[0] == "after_n_asks" and disc[1] >= 1)):
            errs.append(f"{a.id}: bad disclosure {disc!r}")
        if a.verifiable_by not in (None, "record", "family", "exam"):
            errs.append(f"{a.id}: bad verifiable_by {a.verifiable_by}")
    return errs


def wrap_agentclinic(case: Case, cfg: ClusterConfig, lang: str = "en") -> dict:
    """Export in AgentClinic OSCE JSON shape (for comparison experiments). Patient_Actor is built from atoms."""
    pos = [f"{cfg.tag_display(a.slot, a.tag, lang) if a.tag else a.slot}: {cfg.display(a.slot, a.true_value, lang)}"
           for a in case.atoms if a.true_value not in (False, "none", None) and a.slot != "chief_complaint"]
    neg = [f"{cfg.tag_display(a.slot, a.tag, lang) if a.tag else a.slot}" for a in case.atoms if a.true_value is False]
    return {"OSCE_Examination": {
        "Objective_for_Doctor": "Assess and diagnose the patient presenting with chest pain.",
        "Patient_Actor": {"Demographics": case.brief(lang), "History": "; ".join(pos), "Review_of_Systems": "Denies: " + ", ".join(neg)},
        **case.exam_and_tests,
        "Correct_Diagnosis": cfg.dx_display(case.labels.primary_dx, "en")}}
