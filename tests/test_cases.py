import copy, pytest
from scc.env.cases import load_cluster, load_cluster_config, validate, wrap_agentclinic
from scc.types import Case

CFG = load_cluster_config("chest_pain")


def test_load_three_cases_valid():
    cases = load_cluster("chest_pain", ["cp_001", "cp_002", "cp_003"])
    assert [c.case_id for c in cases] == ["cp_001", "cp_002", "cp_003"]
    for c in cases:
        assert validate(c, CFG) == [], (c.case_id, validate(c, CFG))
        assert c.verifiable("record") and c.verifiable("family")


def _broken(mutate):
    c = load_cluster("chest_pain", ["cp_001"])[0]
    d = c.to_dict(); mutate(d); return Case.from_dict(d)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["atoms"].append(dict(d["atoms"][1])), "duplicate"),
    (lambda d: d["labels"].update(primary_dx="nope"), "primary_dx"),
    (lambda d: d["labels"]["red_flags"].append("zzz"), "red flag"),
    (lambda d: [a.update(disclosure="on_specific_ask") for a in d["atoms"]], "spontaneous"),
    (lambda d: [a.update(verifiable_by=None) for a in d["atoms"]], "verifiable"),
    (lambda d: d["atoms"][1].update(true_value="nowhere"), "not in options"),
    (lambda d: d["atoms"][1].update(slot="bogus"), "unknown slot"),
    (lambda d: d["atoms"][8].update(tags=["moon"]), "tag moon"),
])
def test_validation_catches(mutate, needle):
    errs = validate(_broken(mutate), CFG)
    assert any(needle in e for e in errs), errs


def test_brief_has_no_leak():
    c = load_cluster("chest_pain", ["cp_002"])[0]
    b = c.brief("en"); bz = c.brief("zh")
    assert "angina" not in b.lower() and "retrosternal" not in b and "心绞痛" not in bz


def test_display_and_wrap():
    assert CFG.display("location", "epigastric", "zh").startswith("胃")
    assert CFG.display("associated", True, "en") == "yes, I've had that"
    assert CFG.vague_token("location", "epigastric") == "vague_center"
    c = load_cluster("chest_pain", ["cp_001"])[0]
    w = wrap_agentclinic(c, CFG)
    assert w["OSCE_Examination"]["Correct_Diagnosis"].startswith("acid reflux") and "Test_Results" in w["OSCE_Examination"]
