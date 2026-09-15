import math, pandas as pd, pytest
from scc.analysis import metrics as M

PRIM = {"c1": "GERD"}


def row(turn, oz, tk=None, conf=None, kind="ASK", form="open", obs="PATIENT", text=""):
    return {"case_id": "c1", "channel": "x", "doctor": "d", "seed": 0, "turn": turn, "obs_kind": obs, "question_form": form,
            "n_disclosed": 0, "n_withheld": 0, "checker_pass": True, "checker_fallback": False, "oracle_z": oz, "oracle_u": {},
            "entropy_z": 0.0, "readout_topk": tk, "readout_conf": conf, "action_kind": kind, "doctor_text": text}


def test_kl_and_parse_rate():
    df = pd.DataFrame([row(0, {"GERD": 0.5, "x": 0.5}, None, None, kind=None), row(1, {"GERD": 0.5, "x": 0.5}, {"GERD": 0.5, "x": 0.5}, 0.5), row(2, {"GERD": 0.9, "x": 0.1}, {"GERD": 0.5, "x": 0.5}, 0.5), row(3, {"GERD": 0.9, "x": 0.1}, None, None)])
    kl = M.kl_per_turn(df)
    assert kl.kl.iloc[0] == pytest.approx(0.0) and kl.kl.iloc[1] == pytest.approx(0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1))
    assert M.readout_parse_rate(df).readout_parse_rate.iloc[0] == pytest.approx(2 / 3)


def test_mirage_gap():
    # turn 1->2: oracle p_true down, confidence up 0.3 -> counts; 2->3: oracle up, conf up -> does not count; 3->4: oracle flat, conf up 0.1 -> counts
    df = pd.DataFrame([row(0, {"GERD": 0.5}, None, None, kind=None), row(1, {"GERD": 0.6}, {"GERD": 1}, 0.4), row(2, {"GERD": 0.5}, {"GERD": 1}, 0.7),
                       row(3, {"GERD": 0.8}, {"GERD": 1}, 0.8), row(4, {"GERD": 0.8}, {"GERD": 1}, 0.9)])
    g = M.mirage_gap(df, PRIM)
    assert g.mirage_gap.iloc[0] == pytest.approx(0.3 + 0.1) and g.n_pairs.iloc[0] == 3


def test_post_verify_ratio_and_accuracy_and_clar():
    df = pd.DataFrame([row(0, {"GERD": 0.5, "x": 0.5}, None, None, kind=None),
                       row(1, {"GERD": 0.5, "x": 0.5}, {"GERD": 0.5, "x": 0.5}, 0.5, form="forced_choice"),
                       row(2, {"GERD": 0.9, "x": 0.1}, {"GERD": 0.5, "x": 0.5}, 0.5, kind="VERIFY_RECORD", form=None, obs="RECORD"),
                       row(3, {"GERD": 0.9, "x": 0.1}, {"GERD": 0.7, "x": 0.3}, 0.6, form="open"),
                       row(4, {"GERD": 0.9, "x": 0.1}, {"GERD": 0.7, "x": 0.3}, 0.6, kind="DIAGNOSE", form=None, obs="END", text="GERD")])
    pv = M.post_verify_discount_ratio(df, PRIM)
    assert pv.delta_oracle.iloc[0] == pytest.approx(0.4) and pv.delta_doctor.iloc[0] == pytest.approx(0.2) and pv.ratio.iloc[0] == pytest.approx(0.5)
    assert M.accuracy_final(df, PRIM).accuracy_final.iloc[0] == 1.0
    assert M.clarification_rate(df).clarification_rate.iloc[0] == pytest.approx(0.5)
    assert M.verify_usage(df).verify_usage.iloc[0] == 1.0


def test_caig_rank_on_stub_runs(tmp_path):
    from scc.analysis.caig_rank import caig_rank
    from scc.env.cases import load_cluster, load_cluster_config
    from scc.env.likelihood import Likelihood
    from scc.env.oracle import Oracle
    from scc.sim.components import stub_components
    from scc.sim.episode import run_episode
    from scc.sim.scripted_doctor import FixedListDoctor
    from scc.doctor.agent.agent_doctor import build_agent_doctor
    from scc.types import PatientConfig
    cfg = load_cluster_config("chest_pain"); case = load_cluster("chest_pain", ["cp_001"])[0]
    run_episode(case, PatientConfig(name="cooperative"), build_agent_doctor({"components": "stub"}, cfg, "en"), stub_components(cfg), cfg, Oracle(case, cfg, Likelihood(cfg)), 14, "en", tmp_path / "a.jsonl", "cp_001|cooperative|222222B|seed=0|doctor=agent")
    run_episode(case, PatientConfig(name="cooperative"), FixedListDoctor(lang="en"), stub_components(cfg), cfg, Oracle(case, cfg, Likelihood(cfg)), 14, "en", tmp_path / "s.jsonl", "cp_001|cooperative|222222B|seed=0|doctor=scripted_fixed")
    df = caig_rank(tmp_path)
    g = df.groupby(df.episode_id.str.split("|").str[4]).rank_pct.mean()
    assert g["doctor=agent"] > g["doctor=scripted_fixed"] and g["doctor=agent"] > 0.8
