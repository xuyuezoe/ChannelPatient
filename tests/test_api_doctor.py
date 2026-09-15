import json, pytest
from scc.doctor.api_doctor import ApiDoctor, FakeClient, scripted_doctor_texts, told_hint
from scc.doctor import prompts
from scc.doctor.base import format_action_protocol
from scc.types import DoctorContext, ActionKind
from scc.env.cases import load_cluster, load_cluster_config

CFG = load_cluster_config("chest_pain")
DDX = list(CFG.ddx_set)
READ = 'READOUT: {"topk": {"GERD": 0.5, "stable_angina": 0.3, "pericarditis": 0.1, "panic_attack": 0.05, "costochondritis": 0.05}, "confidence": 0.6}'


def ctx(transcript=None):
    return DoctorContext("54-year-old man, truck driver, presenting with chest pain.", transcript or [("patient", "I've been having chest pain.")], 1, 12, DDX, list(ActionKind), "en")


def test_replay_texts_parse():
    texts = ["Where exactly is the pain?\n" + READ, "VERIFY RECORD: prior_severity\n" + READ, "DIAGNOSIS READY: GERD\n" + READ, "How bad is it?\nREADOUT: not json"]
    d = ApiDoctor(FakeClient(texts), "vanilla"); d.reset(ctx())
    a = d.act(ctx()); assert a.kind == ActionKind.ASK and a.readout.topk["GERD"] == 0.5 and "READOUT" not in a.text
    a = d.act(ctx()); assert a.kind == ActionKind.VERIFY_RECORD and a.target == "prior_severity"
    a = d.act(ctx()); assert a.kind == ActionKind.DIAGNOSE and a.text == "GERD"
    a = d.act(ctx()); assert a.kind == ActionKind.ASK and a.readout is None and d.n_parse_fail == 1


def test_reminder_after_two_failures():
    d = ApiDoctor(FakeClient(["How bad is it?", "Where?", "What kind?"]), "vanilla"); d.reset(ctx())
    d.act(ctx()); d.act(ctx())
    assert "READOUT" not in d.client.prompts[1][1]["content"] or "Reminder" not in d.client.prompts[1][1]["content"]
    d.act(ctx())
    assert "Reminder" in d.client.prompts[2][1]["content"]


def test_prompt_has_no_leak_and_has_protocol():
    d = ApiDoctor(FakeClient([READ]), "cot"); d.reset(ctx())
    d.act(ctx([("patient", "I've been having chest pain."), ("doctor", "Where?"), ("patient", "Just above my stomach.")]))
    sys_txt, user_txt = d.client.prompts[0][0]["content"], d.client.prompts[0][1]["content"]
    assert "VERIFY RECORD" in sys_txt and "READOUT" in sys_txt and "REASONING" in sys_txt
    for leak in ("epigastric", "true_value", "atoms", "burning", "mild"):
        assert leak not in sys_txt and leak not in user_txt
    assert "Patient: Just above my stomach." in user_txt and "Doctor: Where?" in user_txt


@pytest.mark.parametrize("ch,needle", [("exaggerate_k2", "overstate"), ("vague_p08", "vague"), ("omit_q05", "volunteer"), ("self_dx", "belief"), ("cooperative", "")])
def test_told_hint(ch, needle):
    h = told_hint(ch, "en"); assert needle in h
    d = ApiDoctor(FakeClient([READ]), "told", told_channel=ch); d.reset(ctx()); d.act(ctx())
    assert (needle in d.client.prompts[0][0]["content"]) if needle else ("Note from the chart" not in d.client.prompts[0][0]["content"])


def test_prompts_bilingual():
    for name in ("DOCTOR_SYS", "FORMAT_REMINDER", "PERCEIVE_USER", "CANDIDATES_USER", "LIKELIHOOD_USER"):
        assert prompts.get(name, "en") and prompts.get(name, "zh")
    for b in ("vanilla", "cot", "uncertainty", "told"):
        assert b in prompts.get("BASELINE_EXTRA", "zh")
    assert "VERIFY RECORD" in format_action_protocol("zh", DDX)


def test_scripted_texts_cover_default_script():
    t = scripted_doctor_texts(DDX, "en")
    assert len(t) == 13 and all("READOUT" in x for x in t) and t[-1].startswith("DIAGNOSIS READY: GERD")
    d = ApiDoctor(FakeClient(t), "vanilla"); d.reset(ctx())
    kinds = [d.act(ctx()).kind for _ in t]
    assert kinds[-2] == ActionKind.VERIFY_RECORD and kinds[-1] == ActionKind.DIAGNOSE and d.n_parse_fail == 0
