import math, sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "pilot"))
P = pytest.importorskip("f4_pilot", reason="pilot/f4_pilot.py (2026-09-01 pilot oracle) not shipped in this repo")
from scc.types import *
from scc.env.cases import load_cluster, load_cluster_config, ClusterConfig
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle

CFG = load_cluster_config("chest_pain")
L = Likelihood(CFG)


def tl(turn, form, disclosed=(), withheld=(), hits=None, obs=ObsKind.PATIENT):
    return TurnLog("e", turn, DoctorAction(ActionKind.ASK, "q"), form, list(hits if hits is not None else [d["atom"] for d in disclosed] + list(withheld)),
                   list(disclosed), list(withheld), None, Observation(obs, "", turn), {"pass": True})


# ----------------------------------------------------------------------------- pilot alignment
def _pilot_setup(k, current_band):
    """Translate pilot/f4_pilot.py's 20-parameter scenario into a Case + custom kernels."""
    bands = ["low", "mid", "high"]
    atoms = [Atom(f"hist_{i}", "history_pain", "ordinal_3", "mid", bands, "on_specific_ask", "record") for i in range(k)]
    atoms.append(Atom("cur", "severity", "ordinal_3", current_band, bands, "on_general_ask"))
    case = Case("pilot", "pilot", {}, Labels("severe", ["severe", "non_severe"]), atoms)
    slots = {"history_pain": {"type": "ordinal_3", "options": bands}, "severity": {"type": "ordinal_3", "options": bands}}
    cfg = ClusterConfig("pilot", ["severe", "non_severe"], [], {}, {}, {}, "uniform", "uniform", slots, {})
    table = {z: {"severity": dict(P.TRUE_BAND[z])} for z in P.Z}           # history slot missing -> uniform (unrelated to Z)
    lik = Likelihood(cfg, table=table)
    kernel = lambda u, atom, t, r, form: P.kernel(u, t, r)
    return case, cfg, lik, kernel


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("cur", ["high", "mid"])
def test_matches_pilot_oracle(k, cur):
    case, cfg, lik, kernel = _pilot_setup(k, cur)
    o = Oracle(case, cfg, lik, u_library={"accurate": [], "exaggerate": []}, prior_u=P.PRIOR_U, prior_z=P.PRIOR_Z,
               kernel_fn=kernel, withheld_fn=lambda u, a, n, f: 1.0)
    pairs = []
    for i in range(k):                                    # history: patient reports 'high', record says 'mid'
        o.update(tl(2 * i + 1, "severity_open", [{"atom": f"hist_{i}", "true_value": "mid", "report_value": "high", "rule": "exaggerate:step+1", "n_asked": 1}]))
        o.update(tl(2 * i + 2, "recall_test", [{"atom": f"hist_{i}", "true_value": "mid", "report_value": "mid", "rule": "verified:record", "n_asked": 1}], obs=ObsKind.RECORD))
        pairs.append(("high", "mid"))
    p_exg_hist = o.posterior_u()["exaggerate"]                 # pilot's P(exag) is computed from history only
    o.update(tl(99, "severity_open", [{"atom": "cur", "true_value": cur, "report_value": cur, "rule": "identity", "n_asked": 1}]))
    p_sev, p_exg = P.oracle(pairs, cur)
    assert o.posterior_z()["severe"] == pytest.approx(p_sev, abs=1e-6)
    assert p_exg_hist == pytest.approx(p_exg, abs=1e-6)


def test_pilot_accurate_history_direction():
    case, cfg, lik, kernel = _pilot_setup(3, "high")
    o = Oracle(case, cfg, lik, u_library={"accurate": [], "exaggerate": []}, prior_u=P.PRIOR_U, prior_z=P.PRIOR_Z, kernel_fn=kernel, withheld_fn=lambda *a: 1.0)
    for i in range(3):
        o.update(tl(i, "severity_open", [{"atom": f"hist_{i}", "true_value": "mid", "report_value": "mid", "rule": "identity", "n_asked": 1}]))
        o.update(tl(i, "recall_test", [{"atom": f"hist_{i}", "true_value": "mid", "report_value": "mid", "rule": "verified:record", "n_asked": 1}], obs=ObsKind.RECORD))
    o.update(tl(9, "severity_open", [{"atom": "cur", "true_value": "high", "report_value": "high", "rule": "identity", "n_asked": 1}]))
    p_sev, p_exg = P.oracle([("mid", "mid")] * 3, "high")
    assert o.posterior_z()["severe"] == pytest.approx(p_sev, abs=1e-6) and p_exg < 0.1


# ----------------------------------------------------------------------------- real cluster behaviour
CASE = load_cluster("chest_pain", ["cp_001"])[0]


def _run(reports, withheld_turns=()):
    o = Oracle(CASE, CFG, L)
    t = 0
    for aid, val, form in reports:
        t += 1
        o.update(tl(t, form, [{"atom": aid, "true_value": CASE.atom(aid).true_value, "report_value": val, "rule": "x", "n_asked": 1}]))
    return o


def test_exaggerated_reports_raise_p_exaggerate():
    o = _run([("cp_severity", "severe", "severity_open"), ("cp_functional", "wakes_at_night", "yes_no"), ("cp_pattern", "constant", "yes_no")])
    pu = o.posterior_u()
    assert pu["exaggerate_k2"] + pu["exaggerate_k1"] > pu["cooperative"]


def test_cooperative_many_reports_converge():
    o = Oracle(CASE, CFG, L); t = 0
    for a in CASE.atoms:
        if a.type == "text":
            continue
        t += 1
        form = "yes_no" if a.type == "bool" else "open"        # open: vague would have produced a vague token
        o.update(tl(t, form, [{"atom": a.id, "true_value": a.true_value, "report_value": a.true_value, "rule": "identity", "n_asked": 1}]))
    pu = o.posterior_u(); pz = o.posterior_z()
    assert pu["cooperative"] > 0.6, pu
    assert max(pz, key=pz.get) == "GERD", pz


def test_verification_redecodes_and_flags_exaggeration():
    o = Oracle(CASE, CFG, L)
    o.update(tl(1, "severity_open", [{"atom": "cp_severity", "true_value": "mild", "report_value": "severe", "rule": "x", "n_asked": 1}]))
    before = o.posterior_u()
    # a record reveals the true severity was mild
    o.update(tl(2, "recall_test", [{"atom": "cp_severity", "true_value": "mild", "report_value": "mild", "rule": "verified:record", "n_asked": 1}], obs=ObsKind.RECORD))
    after = o.posterior_u()
    assert after["exaggerate_k2"] > before["exaggerate_k2"] and after["cooperative"] < 1e-6, (before, after)
    assert o.known_true["cp_severity"] == "mild"


def test_silence_supports_omit():
    o = Oracle(CASE, CFG, L)
    o.update(tl(1, "yes_no", [], ["assoc_dyspnea"], hits=["assoc_dyspnea"]))
    pu = o.posterior_u()
    assert pu["omit_q05"] > pu["cooperative"]
    o2 = Oracle(CASE, CFG, L)
    o2.update(tl(1, "yes_no", [{"atom": "assoc_dyspnea", "true_value": False, "report_value": False, "rule": "identity", "n_asked": 1}]))
    assert o2.posterior_u()["omit_q05"] < o2.posterior_u()["cooperative"]


def test_repeat_report_no_double_count_and_prior_scan():
    o1 = _run([("cp_quality", "burning", "forced_choice")])
    o2 = _run([("cp_quality", "burning", "forced_choice"), ("cp_quality", "burning", "forced_choice")])
    assert o1.posterior_z() == pytest.approx(o2.posterior_z())
    o3 = Oracle(CASE, CFG, L, prior_u={"cooperative": 0.8, "exaggerate_k1": 0.05, "exaggerate_k2": 0.05, "vague_p08": 0.05, "omit_q05": 0.05})
    assert o3.posterior_u()["cooperative"] == pytest.approx(0.8)
    s = o3.snapshot(); assert set(s) == {"posterior_z", "posterior_u", "entropy_z", "known_true"}


def test_text_atom_verification_does_not_collapse_u():
    o = Oracle(CASE, CFG, L)
    o.update(tl(1, "severity_open", [{"atom": "cp_severity", "true_value": "mild", "report_value": "severe", "rule": "x", "n_asked": 1}]))
    o.update(tl(2, "recall_test", [{"atom": "prior_ecg", "true_value": CASE.atom("prior_ecg").true_value, "report_value": CASE.atom("prior_ecg").true_value, "rule": "identity", "n_asked": 1}]))
    before = o.posterior_u()
    o.update(tl(3, None, [{"atom": "prior_ecg", "true_value": CASE.atom("prior_ecg").true_value, "report_value": CASE.atom("prior_ecg").true_value, "rule": "verified:record", "n_asked": 1}], obs=ObsKind.RECORD))
    after = o.posterior_u()
    assert after == pytest.approx(before, abs=1e-9)
