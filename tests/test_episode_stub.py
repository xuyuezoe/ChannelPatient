import json, pytest
from pathlib import Path
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle
from scc.sim.components import stub_components
from scc.sim.episode import ConsultationEnv, run_episode
from scc.sim.scripted_doctor import FixedListDoctor
from scc.doctor.base import parse_doctor_text, format_action_protocol
from scc.types import *

CFG = load_cluster_config("chest_pain"); L = Likelihood(CFG)
CHANNELS = {"cooperative": [], "exaggerate_k2": [ChannelSpec("exaggerate", {"k": 2, "flooding_rate": 0.3})],
            "vague_p08": [ChannelSpec("vague", {"p": 0.8})], "self_dx": [ChannelSpec("self_dx", {"conviction": "insist"})]}
REQUIRED = {"episode_id", "turn", "doctor_action", "question_form", "hit_atoms", "disclosed", "withheld", "self_dx_directive", "observation", "checker", "oracle", "llm_calls", "timing_ms"}


@pytest.mark.parametrize("cid", ["cp_001", "cp_002", "cp_003"])
@pytest.mark.parametrize("ch", list(CHANNELS))
def test_full_episode_stub(tmp_path, cid, ch):
    case = load_cluster("chest_pain", [cid])[0]
    pc = PatientConfig(channels=CHANNELS[ch], seed=0, name=ch)
    logp = tmp_path / f"{cid}_{ch}.jsonl"
    logs = run_episode(case, pc, FixedListDoctor(lang="en"), stub_components(CFG), CFG, Oracle(case, CFG, L), max_turns=14, log_path=logp)
    assert logs[-1].observation.kind == ObsKind.END and len(logs) >= 9
    rows = [json.loads(l) for l in open(logp)]
    assert len(rows) == len(logs)
    for r in rows:
        assert REQUIRED <= set(r), set(r) ^ REQUIRED
        assert r["oracle"] and abs(sum(r["oracle"]["posterior_z"].values()) - 1) < 1e-5
        for d in r["disclosed"]:
            assert {"atom", "true_value", "report_value", "rule", "n_asked"} <= set(d)
    # verification turn shows the record and marks the atom verified
    rec = [r for r in rows if r["observation"]["kind"] == "RECORD"]
    assert rec and rec[0]["disclosed"][0]["rule"] == "verified:record"
    if ch == "exaggerate_k2":
        sev = [d for r in rows for d in r["disclosed"] if d["atom"] == "cp_severity"]
        assert sev and sev[0]["report_value"] != sev[0]["true_value"]
    if ch == "self_dx":
        assert rows[0]["self_dx_directive"] == "open_mention"
        wrong = [CFG.dx_display(d, "en") for d in CFG.plausible_wrong[case.labels.primary_dx]]
        assert any(w in rows[0]["observation"]["text"] for w in wrong), rows[0]["observation"]["text"]


def test_doctor_cannot_see_case():
    case = load_cluster("chest_pain", ["cp_001"])[0]

    class Spy:
        name = "spy"
        def reset(self, ctx): self.ctx = ctx
        def act(self, ctx):
            assert not hasattr(ctx, "atoms") and not hasattr(ctx, "case") and "epigastric" not in ctx.case_brief
            return DoctorAction(ActionKind.DIAGNOSE, "GERD")
    logs = run_episode(case, PatientConfig(), Spy(), stub_components(CFG), CFG, None, log_path=None)
    assert logs[-1].observation.kind == ObsKind.END


def test_parse_doctor_text():
    a = parse_doctor_text("VERIFY RECORD: ECG\nREADOUT: {\"topk\": {\"GERD\": 0.6, \"stable_angina\": 0.4}, \"confidence\": 0.7}")
    assert a.kind == ActionKind.VERIFY_RECORD and a.target == "ECG" and a.readout.confidence == 0.7 and a.readout.topk["GERD"] == 0.6
    assert parse_doctor_text("ASK FAMILY: heart disease").kind == ActionKind.ASK_FAMILY
    assert parse_doctor_text("REQUEST TEST: Troponin").target == "Troponin"
    d = parse_doctor_text("DIAGNOSIS READY: GERD."); assert d.kind == ActionKind.DIAGNOSE and d.text == "GERD"
    q = parse_doctor_text("Where does it hurt?\nREADOUT: not json"); assert q.kind == ActionKind.ASK and q.readout is None and "READOUT" not in q.text
    assert parse_doctor_text("I see, thank you.").kind == ActionKind.CHAT
    assert parse_doctor_text("疼得厉害吗？").kind == ActionKind.ASK
    assert "READOUT" in format_action_protocol("en", ["GERD"]) and "VERIFY RECORD" in format_action_protocol("zh")


def test_env_step_after_done_raises():
    case = load_cluster("chest_pain", ["cp_001"])[0]
    env = ConsultationEnv(case, PatientConfig(), stub_components(CFG), CFG, None, max_turns=2)
    env.reset(); env.step(DoctorAction(ActionKind.ASK, "Where is it?")); _, _, _, done = env.step(DoctorAction(ActionKind.ASK, "How bad?"))
    assert done
    with pytest.raises(RuntimeError):
        env.step(DoctorAction(ActionKind.ASK, "x"))
