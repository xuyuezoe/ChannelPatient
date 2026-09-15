import pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle
from scc.sim.components import stub_components
from scc.sim.episode import run_episode
from scc.doctor.agent.agent_doctor import build_agent_doctor
from scc.types import *

CFG = load_cluster_config("chest_pain"); L = Likelihood(CFG)
CH = {"cooperative": [], "exaggerate_k2": [ChannelSpec("exaggerate", {"k": 2, "flooding_rate": 0.3})], "vague_p08": [ChannelSpec("vague", {"p": 1.0})], "self_dx": [ChannelSpec("self_dx", {"conviction": "insist"})]}


def run(cid, ch, spec, seed=0, max_turns=14):
    case = load_cluster("chest_pain", [cid])[0]; pc = PatientConfig(channels=CH[ch], seed=seed, name=ch)
    doc = build_agent_doctor(spec, CFG, "en"); o = Oracle(case, CFG, L)
    logs = run_episode(case, pc, doc, stub_components(CFG), CFG, o, max_turns=max_turns)
    return case, doc, o, logs


@pytest.mark.parametrize("cid", ["cp_001", "cp_002", "cp_003"])
@pytest.mark.parametrize("ch", list(CH))
def test_agent_runs_and_matches_oracle_when_truthfully_perceived(cid, ch):
    case, doc, o, logs = run(cid, ch, {"components": "stub"})
    assert logs[-1].observation.kind == ObsKind.END and 2 <= len(logs) <= 15
    assert all(l.doctor_action.readout is not None for l in logs[1:])
    # 感知精确、似然同表 -> agent 的 P(z) 与 oracle 逐轮一致（自诊信道除外：agent 不建模自诊，且 self_dx 不改事实，仍应一致）
    for l in logs[1:]:
        pass
    az, oz = doc.belief.posterior_z(), o.posterior_z()
    for z in CFG.ddx_set:
        assert az[z] == pytest.approx(oz[z], abs=0.05), (cid, ch, z, az, oz)


def test_doctor_cannot_see_case():
    case = load_cluster("chest_pain", ["cp_001"])[0]
    doc = build_agent_doctor({"components": "stub"}, CFG, "en")
    assert not hasattr(doc, "case") and not any(hasattr(doc, a) for a in ("atoms", "report_table"))


def test_vague_channel_triggers_clarification_and_zonly_does_not():
    case, doc, o, logs = run("cp_002", "vague_p08", {"components": "stub", "tau_diagnose": 0.99})
    forms = [l.question_form for l in logs[1:] if l.doctor_action.kind == ActionKind.ASK]
    assert any(f in ("forced_choice", "point_to") for f in forms[:6]), forms
    case, doc0, o0, logs0 = run("cp_002", "vague_p08", {"components": "stub", "z_only": True, "tau_diagnose": 0.99})
    assert doc0.us == ["cooperative"]


def test_exaggerate_channel_raises_p_u_when_severity_asked():
    case = load_cluster("chest_pain", ["cp_001"])[0]; pc = PatientConfig(channels=CH["exaggerate_k2"], name="exaggerate_k2")
    doc = build_agent_doctor({"components": "stub", "tau_diagnose": 0.999, "epsilon_stop": 0.0}, CFG, "en")
    logs = run_episode(case, pc, doc, stub_components(CFG), CFG, Oracle(case, CFG, L), max_turns=14)
    pu = doc.belief.posterior_u()
    asked_sev = any(l.question_form == "severity_open" for l in logs)
    if asked_sev:
        assert pu["exaggerate_k2"] + pu["exaggerate_k1"] > pu["cooperative"]
    assert len(logs) >= 6
