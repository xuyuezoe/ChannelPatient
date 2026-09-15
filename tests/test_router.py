import pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.channels import build_report_table
from scc.sim.rules import RuleEngine
from scc.sim.router import route
from scc.types import *

CFG = load_cluster_config("chest_pain")
CASE = load_cluster("chest_pain", ["cp_001"])[0]


def engine():
    cfgp = PatientConfig(channels=[ChannelSpec("exaggerate", {"k": 2})])
    t, sd, _ = build_report_table(CASE, cfgp, CFG)
    return RuleEngine(CASE, t, sd, cfgp, CFG)


def test_verify_record():
    r = engine()
    obs, disc = route(DoctorAction(ActionKind.VERIFY_RECORD, target="ecg"), CASE, r, CFG, "en", 3)
    assert obs.kind == ObsKind.RECORD and "flattened T waves" in obs.text and disc[0]["rule"] == "verified:record" and "prior_ecg" in r.verified
    obs, disc = route(DoctorAction(ActionKind.VERIFY_RECORD, target="colonoscopy"), CASE, r, CFG, "en", 4)
    assert obs.kind == ObsKind.RECORD and "No record" in obs.text and disc == []


def test_ask_family():
    r = engine()
    obs, disc = route(DoctorAction(ActionKind.ASK_FAMILY, target="heart disease"), CASE, r, CFG, "en", 3)
    assert obs.kind == ObsKind.FAMILY and obs.text.startswith("Family member:") and disc[0]["atom"] == "fam_cad"
    obs, _ = route(DoctorAction(ActionKind.ASK_FAMILY, target="心脏病"), CASE, r, CFG, "zh", 3)
    assert "家属" in obs.text


def test_request_test_and_normal():
    r = engine()
    obs, _ = route(DoctorAction(ActionKind.REQUEST_TEST, target="Troponin"), CASE, r, CFG, "en", 3)
    assert obs.kind == ObsKind.TEST and "Negative" in obs.text
    obs, _ = route(DoctorAction(ActionKind.REQUEST_TEST, target="MRI brain"), CASE, r, CFG, "en", 3)
    assert obs.text.endswith("NORMAL READINGS")


def test_diagnose_ends():
    r = engine()
    obs, _ = route(DoctorAction(ActionKind.DIAGNOSE, "GERD"), CASE, r, CFG, "en", 9)
    assert obs.kind == ObsKind.END and obs.text == "GERD"
