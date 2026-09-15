import pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.channels import build_report_table
from scc.sim.rules import RuleEngine
from scc.sim.phraser import TemplatePhraser
from scc.sim.checker import RegexChecker
from scc.sim.persona import TemplateRole
from scc.types import *

CFG = load_cluster_config("chest_pain"); CASE = load_cluster("chest_pain", ["cp_001"])[0]


def engine(specs=()):
    cfgp = PatientConfig(channels=[ChannelSpec(k, p) for k, p in specs])
    t, sd, _ = build_report_table(CASE, cfgp, CFG)
    return RuleEngine(CASE, t, sd, cfgp, CFG)


def test_template_every_slot_value_has_text():
    ph = TemplatePhraser(CFG)
    for lang in ("en", "zh"):
        for a in CASE.atoms:
            vals = [True, False] if a.type == "bool" else (a.options or [a.true_value])
            for v in vals:
                s = ph._one(a, v, lang)
                assert s and s != str(v), (a.id, v, s)
        assert ph._one(CASE.atom("cp_location"), "vague_center", lang)


def test_template_plans():
    ph = TemplatePhraser(CFG)
    r = engine([("exaggerate", {"k": 2, "flooding_rate": 1.0}), ("self_dx", {"label": "stable_angina", "conviction": "insist"})])
    op = r.opening(); s = ph.phrase(op, "", [], "en")
    assert "angina" in s and "chest pain" in s
    p = r.plan(["cp_severity", "cp_functional"], "severity_open", 1); s = ph.phrase(p, "", [], "en")
    assert any(w in s for w in CFG.severity_words("severe", "en")) and "night" in s
    p = r.plan(["assoc_sweating"], "yes_no", 2); s = ph.phrase(p, "", [], "en")
    assert "sweating" in s and p.flooding
    r2 = engine([("omit", {"q": 1.0, "soften": {3: 1.0}})])
    p = r2.plan(["assoc_dyspnea"], "yes_no", 3); s = ph.phrase(p, "", [], "en")
    assert any(w in s for w in CFG.phrases("withheld_phrases", "en"))
    p = r.plan([], "chat", 4, is_chat=True); assert ph.phrase(p, "", [], "zh")


def test_regex_checker():
    ck = RegexChecker(CFG); r = engine([("vague", {"p": 1.0})])
    p = r.plan(["cp_location"], "open", 1)
    assert ck.check("It's somewhere in the middle of my chest, kind of near the stomach.", p, CASE, "en").passed
    res = ck.check("It's just above the stomach, below the chest. I think it's angina.", p, CASE, "en")
    assert not res.passed and any(v.startswith("diagnosis_term") for v in res.violations)
    r3 = engine([("omit", {"q": 1.0, "soften": {3: 1.0}})]); p = r3.plan(["assoc_dyspnea"], "yes_no", 2)      # withheld by omit
    res = ck.check("I get shortness of breath too.", p, CASE, "en")
    assert any(v.startswith("withheld_mentioned") for v in res.violations)
    r2 = engine([("exaggerate", {"k": 2})]); p = r2.plan(["cp_severity"], "severity_open", 1)
    assert not ck.check("It's a bit sore, bearable.", p, CASE, "en").passed        # mild words for a severe report
    assert ck.check("It's unbearable.", p, CASE, "en").passed
    r3 = engine([("self_dx", {"label": "stable_angina"})]); p = r3.opening()
    assert ck.check("Doctor, I'm pretty sure this is angina (heart-related chest pain). I've been having chest pain.", p, CASE, "en").passed


def test_template_role_no_inducing_sentence():
    for c in (1, 2, 3):
        role = TemplateRole().build(Persona(c=c), CASE, "en")
        assert "medications you take" not in role and "Never invent" in role
    assert "不要编造" in TemplateRole().build(Persona(), CASE, "zh")
