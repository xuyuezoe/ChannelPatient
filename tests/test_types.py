import json
from scc.types import *


def test_atom_roundtrip():
    a = Atom("x", "severity", "ordinal_3", "mild", ["mild", "moderate", "severe"], "after_n_asks:2", None, ["t"])
    d = a.to_dict(); s = json.dumps(d)
    b = Atom.from_dict(json.loads(s))
    assert b.disclosure == ("after_n_asks", 2) and b.tags == ["t"] and b.options == a.options


def test_case_roundtrip():
    c = Case("c1", "chest_pain", {"age": 50, "sex": "M"}, Labels("GERD", ["GERD", "x"], ["x"]),
             [Atom("a", "chief_complaint", "bool", True, disclosure="spontaneous")], {"t": 1}, {"dataset": "hw"})
    d = json.loads(json.dumps(c.to_dict()))
    c2 = Case.from_dict(d)
    assert c2.labels.red_flags == ["x"] and c2.atoms[0].disclosure == "spontaneous" and c2.exam_and_tests == {"t": 1}
    assert "GERD" not in c2.brief() and "50-year-old man" in c2.brief()


def test_patient_config_and_label():
    cfg = PatientConfig.from_dict({"persona": {"e": 3}, "channels": [{"kind": "exaggerate", "params": {"k": 2, "flooding_rate": 0.3}}], "seed": 4})
    assert cfg.persona.e == 3 and cfg.persona.code() == "232222B"
    assert cfg.label() == "exaggerate(flooding_rate=0.3,k=2)" and cfg.kinds() == {"exaggerate"}
    assert PatientConfig().label() == "cooperative"


def test_action_and_readout_json():
    a = DoctorAction(ActionKind.VERIFY_RECORD, "", target="prior_test", readout=DoctorReadout({"GERD": 2, "x": 1}, 0.5))
    d = json.loads(json.dumps(a.to_dict()))
    assert d["kind"] == "VERIFY_RECORD" and d["readout"]["topk"]["GERD"] == 2
    assert abs(sum(a.readout.normalised().values()) - 1) < 1e-9


def test_turnlog_json():
    t = TurnLog("e", 1, DoctorAction(ActionKind.ASK, "hi"), "open", ["a"], [{"atom": "a"}], [], None,
                Observation(ObsKind.PATIENT, "ok", 1), {"pass": True}, {"posterior_z": {"GERD": 1.0}})
    d = json.loads(json.dumps(t.to_dict()))
    assert d["observation"]["kind"] == "PATIENT" and d["oracle"]["posterior_z"]["GERD"] == 1.0
