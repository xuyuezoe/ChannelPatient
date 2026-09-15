import pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.channels import build_report_table
from scc.sim.rules import RuleEngine
from scc.types import ChannelSpec, PatientConfig

CFG = load_cluster_config("chest_pain")
CASE = load_cluster("chest_pain", ["cp_001"])[0]


def engine(specs=(), seed=0, **kw):
    cfgp = PatientConfig(channels=[ChannelSpec(k, p) for k, p in specs], seed=seed, **kw)
    t, sd, _ = build_report_table(CASE, cfgp, CFG)
    return RuleEngine(CASE, t, sd, cfgp, CFG)


def test_opening_and_general_ask():
    r = engine()
    op = r.opening()
    assert [a.id for a, _ in op.say] == ["cp_present"] and op.self_dx is None
    p = r.plan([], "open", 1)
    ids = [a.id for a, _ in p.say]
    assert set(ids) == {"cp_location", "cp_severity", "cp_duration"}     # on_general_ask atoms, not yet disclosed
    p2 = r.plan([], "open", 2)
    assert p2.say == []                                                  # already disclosed


def test_specific_atoms_answered_when_targeted():
    r = engine()
    p = r.plan(["assoc_dyspnea"], "open", 1)          # "how about your breathing?" is open in form but targets the atom
    assert any(a.id == "assoc_dyspnea" for a, _ in p.say) and not p.withheld
    r = engine([("omit", {"q": 1.0, "soften": {2: 1.0}})])
    p = r.plan(["assoc_dyspnea"], "open", 1)
    assert p.withheld and p.withheld[0].id == "assoc_dyspnea"    # omit channel still withholds until asked twice
    r = engine()
    p = r.plan([a.id for a in CASE.by_slot("trigger")], "open", 1)   # broad sweep: only positives are voiced
    assert all(v is True for a, v in p.say if a.slot == "trigger") and not p.withheld


def test_consistency_lock_and_forms():
    r = engine([("vague", {"p": 1.0}), ("exaggerate", {"k": 2})])
    p1 = r.plan(["cp_severity", "cp_location"], "severity_open", 1)
    v = dict((a.id, val) for a, val in p1.say)
    assert v["cp_severity"] == "severe" and v["cp_location"] == "vague_center"
    p2 = r.plan(["cp_severity"], "severity_open", 3)
    assert dict((a.id, val) for a, val in p2.say)["cp_severity"] == "severe"
    p3 = r.plan(["cp_location"], "forced_choice", 4)
    assert dict((a.id, val) for a, val in p3.say)["cp_location"] == "epigastric"


def test_after_n_asks():
    r = engine([("omit", {"q": 1.0, "soften": {3: 1.0}})])
    assert r.plan(["assoc_acid"], "yes_no", 1).withheld[0].id == "assoc_acid"
    assert r.plan(["assoc_acid"], "yes_no", 2).withheld[0].id == "assoc_acid"
    p = r.plan(["assoc_acid"], "yes_no", 3)
    assert p.say[0][0].id == "assoc_acid" and p.say[0][1] is True and not p.withheld


def test_verified_truth_and_post_verify_shift():
    r = engine([("exaggerate", {"k": 2})], post_verify_shift=-1)
    assert r.table["cp_functional"].report_value == "wakes_at_night"
    r.on_verified("cp_severity")
    assert r.plan(["cp_severity"], "severity_open", 1).say[0][1] == "mild"          # verified -> truth
    assert r.table["cp_functional"].report_value == "limits_activity"                # undisclosed atoms softened by one level
    r2 = engine([("exaggerate", {"k": 2})], post_verify_shift=0)
    r2.on_verified("cp_severity")
    assert r2.table["cp_functional"].report_value == "wakes_at_night"


def test_self_dx_directives():
    r = engine([("self_dx", {"label": "stable_angina", "conviction": "insist"})])
    assert r.opening().self_dx == ("stable_angina", "open_mention")
    d = [r.plan([], "open", t).self_dx for t in range(1, 7)]
    assert [x[1] if x else None for x in d] == ["insist", None, "insist", None, "insist", None]
    r = engine([("self_dx", {"label": "stable_angina", "conviction": "open_mention"})])
    assert r.opening().self_dx[1] == "open_mention" and r.plan([], "open", 1).self_dx is None
    r = engine([("self_dx", {"label": "stable_angina", "conviction": "reframe"})])
    assert r.plan([], "open", 1).self_dx[1] == "reframe"


def test_flooding_marked():
    r = engine([("exaggerate", {"k": 0, "flooding_rate": 1.0})])
    p = r.plan(["assoc_sweating"], "yes_no", 1)
    assert p.say[0][1] is True and p.flooding and p.flooding[0].id == "assoc_sweating"
