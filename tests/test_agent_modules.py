"""阶段 2 各模块的单元测试（全部零 API）。"""
import json, math, pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle
from scc.env.channels import build_report_table
from scc.sim.rules import RuleEngine
from scc.sim.phraser import TemplatePhraser
from scc.doctor.agent.state import *
from scc.doctor.agent.perception import RegexPerceiver
from scc.doctor.agent.likelihood import TableLikelihood, LLMLikelihood, LLMFullLikelihood, default_u_library, support, ETA_UNKNOWN as ETA
from scc.doctor.agent.belief import BeliefEngine
from scc.doctor.agent.candidates import MenuCandidates, LLMCandidates
from scc.doctor.agent.value import ca_ig, misspecified_ig, joint_mi, lookahead2, outcomes
from scc.doctor.agent.monitor import SurprisalMonitor
from scc.doctor.agent.policy import choose, readout
from scc.doctor.api_doctor import FakeClient
from scc.types import *

CFG = load_cluster_config("chest_pain"); CASE = load_cluster("chest_pain", ["cp_001"])[0]
ULIB = default_u_library(CFG); P = TableLikelihood(CFG, ULIB)


def fresh():
    return BeliefEngine(list(CFG.ddx_set), list(ULIB), P)


# ----------------------------------------------------------------------------- state
def test_state_json():
    o = SemanticObs("severity", None, "severity_open", {"severe": 0.7, "moderate": 0.3}, {"intensity": 1.0})
    c = Candidate(ActionKind.ASK, "How bad?", "severity", None, "severity_open", is_calibration=False)
    e = LedgerEntry(1, "patient", "bad", "How bad?", [o])
    for obj in (o.to_dict(), c.to_dict(), e.to_dict()):
        json.dumps(obj)
    assert AgentConfig.from_dict({"z_only": True, "bogus": 1}).z_only is True


# ----------------------------------------------------------------------------- perception
def test_regex_perceiver_inverts_templates():
    ph = TemplatePhraser(CFG, seed=0); per = RegexPerceiver(CFG); n = ok = 0
    for a in CASE.atoms:
        if a.slot in ("chief_complaint", "prior_test"):
            continue
        vals = [True, False] if a.type == "bool" else (a.options or [a.true_value])
        for v in vals:
            plan = TurnPlan([(a, v)], [], None, False, "yes_no")
            s = ph.phrase(plan, "", [], "en")
            obs = per.perceive(s, Candidate(ActionKind.ASK, "?", a.slot, a.tag, "yes_no"), "en")
            hit = [o for o in obs if (o.slot, o.tag) == (a.slot, a.tag)]
            n += 1; ok += bool(hit) and max(hit[0].dist, key=hit[0].dist.get) == v
    assert ok == n, (ok, n)


def test_regex_perceiver_vague_and_withheld():
    per = RegexPerceiver(CFG)
    o = per.perceive("It's somewhere in the middle of my chest, kind of near the stomach.", Candidate(ActionKind.ASK, "?", "location", None, "open"), "en")
    assert o[0].dist == {"vague_center": 1.0}
    o = per.perceive("I haven't really noticed.", Candidate(ActionKind.ASK, "?", "associated", "dyspnea", "yes_no"), "en")
    assert o[0].dist == {NOT_MENTIONED: 1.0}
    o = per.perceive("Hmm.", Candidate(ActionKind.ASK, "?", "associated", "dyspnea", "yes_no"), "en")
    assert o[0].dist == {UNKNOWN: 1.0}
    o = per.perceive("It's unbearable. my whole chest feels off.", Candidate(ActionKind.ASK, "?", "severity", None, "severity_open"), "en")
    assert o[0].manner["intensity"] == 1.0 and o[0].manner["flooding"] == 1.0


# ----------------------------------------------------------------------------- likelihood provider
def test_table_likelihood_normalised_and_matches_oracle():
    for z in CFG.ddx_set:
        for u in ULIB:
            for slot in ("location", "severity", "associated", "prior_severity", "pattern"):
                tag = "dyspnea" if slot == "associated" else None
                for form in ("open", "forced_choice", "yes_no", "severity_open"):
                    d = P.dist(z, u, slot, tag, form)
                    assert abs(sum(d.values()) - 1) < 1e-9 and set(d) == set(support(CFG, slot, tag))
    # 与 oracle 的报告似然一致（去掉 unknown / not_mentioned 的质量后）
    o = Oracle(CASE, CFG, Likelihood(CFG))
    for z in CFG.ddx_set:
        for u in ULIB:
            a = CASE.atom("cp_severity")
            lo = o._lik_report(z, u, a, "severe", "severity_open")
            d = P.dist(z, u, "severity", None, "severity_open")
            assert d["severe"] / ((1 - ETA) * (1 - d[NOT_MENTIONED])) == pytest.approx(lo, abs=1e-9)


def test_llm_medical_likelihood_cache_and_fallback(tmp_path):
    rows = {z: {"mild": 0.6, "moderate": 0.3, "severe": 0.1} for z in CFG.ddx_set}; rows["stable_angina"] = {"mild": 0.2, "moderate": 0.5, "severe": 0.3}
    fc = FakeClient([json.dumps(rows), "not json"])
    L = LLMLikelihood(CFG, ULIB, fc, "en", cache_dir=tmp_path)
    d = L.dist("stable_angina", "cooperative", "severity", None, "severity_open")
    assert d["moderate"] > d["mild"] and fc.calls == 1
    L2 = LLMLikelihood(CFG, ULIB, FakeClient(["x"]), "en", cache_dir=tmp_path)
    assert L2.p_true("GERD", "severity", None)["mild"] == pytest.approx(0.6) and L2.client.calls == 0 and L2.n_cache == 1
    L3 = LLMLikelihood(CFG, ULIB, FakeClient(["not json"]), "en", cache_dir=tmp_path / "empty")
    d3 = L3.dist("GERD", "exaggerate_k2", "severity", None, "severity_open"); assert L3.n_fallback == 1 and abs(sum(d3.values()) - 1) < 1e-9
    # U 无关性由构造保证：家族史在两种信道下相同
    assert L.dist("GERD", "cooperative", "family_history", "cad", "yes_no")[True] == pytest.approx(L.dist("GERD", "exaggerate_k2", "family_history", "cad", "yes_no")[True])


def test_llm_full_likelihood_cache_and_fallback(tmp_path):
    fc = FakeClient(['{"dist": {"mild": 0.2, "moderate": 0.3, "severe": 0.5}}', "not json at all"])
    L = LLMFullLikelihood(CFG, ULIB, fc, "en", cache_dir=tmp_path)
    d1 = L.dist("GERD", "cooperative", "severity", None, "severity_open")
    assert d1["severe"] > d1["mild"] and fc.calls == 1
    L._cache.clear()
    d2 = L.dist("GERD", "cooperative", "severity", None, "severity_open")
    assert d2 == d1 and fc.calls == 1 and L.n_cache == 1                       # 磁盘缓存命中，零调用
    d3 = L.dist("GERD", "exaggerate_k2", "severity", None, "severity_open")     # 坏 JSON -> 回退查表
    assert L.n_fallback == 1 and abs(sum(d3.values()) - 1) < 1e-9


# ----------------------------------------------------------------------------- belief
def test_belief_hard_obs_equals_oracle():
    b = fresh(); o = Oracle(CASE, CFG, Likelihood(CFG)); t = 0
    for aid, v, form in [("cp_quality", "burning", "forced_choice"), ("cp_severity", "severe", "severity_open"), ("assoc_dyspnea", True, "yes_no"), ("cp_location", "vague_center", "open")]:
        a = CASE.atom(aid); t += 1
        b.update_soft(SemanticObs(a.slot, a.tag, form, {v: 1.0}), 1)
        o.update(TurnLog("e", t, DoctorAction(ActionKind.ASK, "q"), form, [aid], [{"atom": aid, "true_value": a.true_value, "report_value": v, "rule": "x", "n_asked": 1}], [], None, Observation(ObsKind.PATIENT, "", t), {"pass": True}))
        for z in CFG.ddx_set:
            assert b.posterior_z()[z] == pytest.approx(o.posterior_z()[z], abs=1e-6), (aid, z)


def test_belief_soft_between_hard_and_verify_redecodes():
    hard_mild = fresh(); hard_mild.update_soft(SemanticObs("severity", None, "severity_open", {"mild": 1.0}))
    hard_sev = fresh(); hard_sev.update_soft(SemanticObs("severity", None, "severity_open", {"severe": 1.0}))
    soft = fresh(); soft.update_soft(SemanticObs("severity", None, "severity_open", {"mild": 0.5, "severe": 0.5}))
    pu = lambda b: b.posterior_u()["exaggerate_k2"]
    assert min(pu(hard_mild), pu(hard_sev)) <= pu(soft) <= max(pu(hard_mild), pu(hard_sev))
    b = fresh(); b.update_soft(SemanticObs("severity", None, "severity_open", {"severe": 1.0}))
    before = b.posterior_u()["exaggerate_k2"]; b.observe_true("severity", None, "mild")
    assert b.posterior_u()["exaggerate_k2"] > before and b.posterior_u()["cooperative"] < 1e-6
    hist = list(b.history); b2 = fresh(); b2.replay(hist)
    for k in b.log_post:
        assert b2.posterior_joint()[k] == pytest.approx(b.posterior_joint()[k], abs=1e-9)


def test_belief_temper_and_lock():
    b = fresh(); b.update_soft(SemanticObs("severity", None, "severity_open", {"severe": 1.0}))
    h0 = -sum(v * math.log(v) for v in b.posterior_u().values() if v > 0); b.temper_u(0.5)
    h1 = -sum(v * math.log(v) for v in b.posterior_u().values() if v > 0); assert h1 > h0
    assert b.update_soft(SemanticObs("severity", None, "severity_open", {"severe": 1.0})) is False


# ----------------------------------------------------------------------------- candidates / value
def test_menu_candidates_cover_slots():
    m = MenuCandidates(CFG).propose({})
    slots = {c.slot for c in m}
    assert slots >= {s for s in CFG.slots if s != "chief_complaint"}
    assert any(c.kind == ActionKind.VERIFY_RECORD for c in m) and any(c.is_calibration for c in m)
    assert all(c.form for c in m if c.kind == ActionKind.ASK)


def test_llm_candidates_abstracted():
    from scc.sim.classifier import KeywordClassifier
    fc = FakeClient(['{"questions": ["Where exactly is the pain?", "Is it behind the breastbone, above the stomach, or on the left side?", "Any sweating with it?"]}'])
    g = LLMCandidates(CFG, fc, KeywordClassifier(CFG))
    cs = g.propose({"top2": ["GERD", "stable_angina"], "transcript": [], "asked": {}, "lang": "en"})
    llm = [c for c in cs if c.source == "llm"]
    assert {(c.slot, c.form) for c in llm} >= {("location", "open"), ("location", "forced_choice"), ("associated", "yes_no")}
    assert any(c.is_calibration for c in llm) and any(c.kind == ActionKind.VERIFY_RECORD for c in cs)


def test_value_toy_and_predictions():
    b = fresh(); m = MenuCandidates(CFG).propose({})
    by = {(c.slot, c.form): c for c in m if c.kind == ActionKind.ASK}
    # 手算：IG = H(z) - E H(z|r)
    c = by[("quality", "forced_choice")]
    outs = outcomes(b, c); h0 = b.entropy_z()
    assert ca_ig(b, c) == pytest.approx(max(0.0, h0 - sum(p * nb.entropy_z() for _, p, nb in outs)), abs=1e-12)
    assert abs(sum(p for _, p, _ in outs) - 1) < 1e-6
    # P1：U 无关的槽（家族史）上 CA-IG == misspecified-IG
    fam = next(c for c in m if c.slot == "family_history" and c.tag == "cad")
    assert ca_ig(b, fam) == pytest.approx(misspecified_ig(b, fam), abs=1e-6)
    # P2（静态）：aliasing 槽上 forced_choice 的 CA-IG 高于 open；误设 IG 看不出差别
    assert ca_ig(b, by[("location", "forced_choice")]) > ca_ig(b, by[("location", "open")])
    assert misspecified_ig(b, by[("location", "forced_choice")]) == pytest.approx(misspecified_ig(b, by[("location", "open")]), abs=1e-6)
    # 联合 MI ≥ CA-IG；前瞻 ≥ 单步
    assert joint_mi(b, c) >= ca_ig(b, c) - 1e-9
    v2, det = lookahead2(b, by[("location", "forced_choice")], sorted(m, key=lambda x: -ca_ig(b, x))[:8])
    assert v2 >= det["ig1"] - 1e-12


# ----------------------------------------------------------------------------- monitor / policy
def test_monitor_alarms_on_misspecification():
    m = SurprisalMonitor(threshold=3.0); b = fresh()
    for _ in range(20):
        pred = b.predictive("associated", "dyspnea", "yes_no"); alarm = m.update(pred, SemanticObs("associated", "dyspnea", "yes_no", {False: 1.0}))
        b.update_soft(SemanticObs("associated", "dyspnea", "yes_no", {False: 1.0}))
    assert m.alarms == 0
    m2 = SurprisalMonitor(threshold=1.0); b2 = fresh()
    for (z, u) in b2.log_post:
        if u != "cooperative":
            b2.log_post[(z, u)] = -1e9         # 模型写死为合作
    for (z, u) in b2.log_post:
        if z != "GERD":
            b2.log_post[(z, u)] -= 6.0         # 且已相当确定是 GERD：真 GERD 患者很少说"重"
    fired = False
    for slot, v in (("severity", "severe"), ("prior_severity", "severe"), ("functional_impact", "wakes_at_night"), ("pattern", "constant")):
        pred = b2.predictive(slot, None, "severity_open")
        fired |= m2.update(pred, SemanticObs(slot, None, "severity_open", {v: 1.0}))
    assert fired, m2.snapshot()


def test_policy_branches():
    cfg = AgentConfig(); b = fresh(); m = MenuCandidates(CFG).propose({})
    for c in m: c.value = 0.1
    a, info = choose(m, b, cfg, ["stable_angina"], {}); assert a.kind == ActionKind.ASK and info["decision"] == "ask"
    b2 = fresh()
    for (z, u) in b2.log_post: b2.log_post[(z, u)] = 0.0 if z == "GERD" else -20.0
    a, info = choose(m, b2, cfg, ["stable_angina"], {}); assert a.kind == ActionKind.DIAGNOSE and a.text == "GERD"
    for c in m: c.value = 0.0
    a, info = choose(m, fresh(), cfg, ["stable_angina"], {}); assert a.kind == ActionKind.DIAGNOSE and info["decision"] == "undetermined"
    r = readout(fresh()); assert abs(sum(r.topk.values()) - 1) < 1e-6 and r.confidence == pytest.approx(0.0, abs=1e-6)
