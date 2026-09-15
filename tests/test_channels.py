import numpy as np, pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.channels import *
from scc.types import ChannelSpec, PatientConfig, Persona

CFG = load_cluster_config("chest_pain")
CASE = load_cluster("chest_pain", ["cp_001"])[0]


def cfgp(*specs, seed=0, **kw):
    return PatientConfig(channels=[ChannelSpec(k, p) for k, p in specs], seed=seed, **kw)


def test_seed_reproducible_and_cooperative_identity():
    t1, _, _ = build_report_table(CASE, cfgp(("exaggerate", {"k": 1, "flooding_rate": 0.5}), ("vague", {"p": 0.8}), seed=7), CFG)
    t2, _, _ = build_report_table(CASE, cfgp(("exaggerate", {"k": 1, "flooding_rate": 0.5}), ("vague", {"p": 0.8}), seed=7), CFG)
    assert {k: (v.report_value, v.rule) for k, v in t1.items()} == {k: (v.report_value, v.rule) for k, v in t2.items()}
    t0, sd, _ = build_report_table(CASE, cfgp(), CFG)
    assert all(v.report_value == v.true_value and v.rule == "identity" for v in t0.values()) and sd is None


@pytest.mark.parametrize("k,expect", [(1, "moderate"), (2, "severe"), (3, "severe")])
def test_exaggerate_severity_steps(k, expect):
    t, _, _ = build_report_table(CASE, cfgp(("exaggerate", {"k": k})), CFG)
    assert t["cp_severity"].report_value == expect and t["cp_severity"].rule == f"exaggerate:step+{k}"
    assert t["cp_functional"].report_value == LEVELS["functional_impact"][min(k, 2)]
    assert t["cp_pattern"].report_value == "constant"
    for aid in ("cp_location", "cp_quality", "prior_ecg", "hx_htn", "life_smoking"):
        assert t[aid].report_value == t[aid].true_value


def test_flooding_only_false_associated():
    t, _, _ = build_report_table(CASE, cfgp(("exaggerate", {"k": 0, "flooding_rate": 1.0})), CFG)
    for a in CASE.by_slot("associated"):
        assert t[a.id].report_value is True
        assert (t[a.id].rule == "exaggerate:flooding") == (a.true_value is False)
    t, _, _ = build_report_table(CASE, cfgp(("exaggerate", {"k": 0, "flooding_rate": 0.0})), CFG)
    assert all(t[a.id].report_value == a.true_value for a in CASE.by_slot("associated"))


def test_vague_only_open_forms():
    t, _, _ = build_report_table(CASE, cfgp(("vague", {"p": 1.0})), CFG)
    e = t["cp_location"]
    assert e.report_value == "vague_center" and e.value_for("open") == "vague_center"
    assert e.value_for("forced_choice") == "epigastric" and e.value_for("point_to") == "epigastric"
    assert t["cp_quality"].report_value == "vague_quality"
    assert t["cp_severity"].report_value == "mild" and t["cp_duration"].report_value == "weeks"   # no map -> untouched
    t, _, _ = build_report_table(CASE, cfgp(("vague", {"p": 0.0})), CFG)
    assert t["cp_location"].report_value == "epigastric" and t["cp_location"].by_form is None


def test_omit_changes_disclosure_not_value():
    t, _, _ = build_report_table(CASE, cfgp(("omit", {"q": 1.0, "soften": {2: 0.5, 3: 0.5}})), CFG)
    for a in CASE.by_slot("associated"):
        assert t[a.id].report_value == a.true_value
        assert isinstance(t[a.id].disclosure, tuple) and t[a.id].disclosure[0] == "after_n_asks" and t[a.id].disclosure[1] in (2, 3)
    assert t["cp_location"].disclosure == "on_general_ask"


def test_self_dx_label_is_wrong_and_plausible():
    for seed in range(5):
        _, sd, _ = build_report_table(CASE, cfgp(("self_dx", {"conviction": "insist"}), seed=seed), CFG)
        assert sd[0] != CASE.labels.primary_dx and sd[0] in CFG.plausible_wrong["GERD"] and sd[1] == "insist"
    _, sd, _ = build_report_table(CASE, cfgp(("self_dx", {"label": "panic_attack"})), CFG)
    assert sd[0] == "panic_attack"


def _specs(*specs):
    return [ChannelSpec(k, p) for k, p in specs]


@pytest.mark.parametrize("specs", [_specs(), _specs(("exaggerate", {"k": 1, "flooding_rate": 0.3})),
                                   _specs(("vague", {"p": 0.8})), _specs(("vague", {"p": 0.8}), ("exaggerate", {"k": 2, "flooding_rate": 0.2})),
                                   _specs(("omit", {"q": 0.5}))])
@pytest.mark.parametrize("form", ["open", "forced_choice", "yes_no", "severity_open"])
def test_likelihood_kernel_sums_to_one(specs, form):
    for a in CASE.atoms:
        if a.type == "text":
            continue
        support = report_support(a, CFG)
        for tv in ([True, False] if a.type == "bool" else (a.options or CFG.slot_options(a.slot, a.tag))):
            s = sum(likelihood_kernel(specs, a, tv, rv, form, CFG) for rv in support)
            assert abs(s - 1) < 1e-9, (a.id, tv, form, s)


def test_likelihood_matches_forward():
    specs = _specs(("vague", {"p": 1.0}), ("exaggerate", {"k": 2, "flooding_rate": 1.0}))
    t, _, _ = build_report_table(CASE, PatientConfig(channels=specs, seed=1), CFG)
    for a in CASE.atoms:
        if a.type == "text":
            continue
        e = t[a.id]
        assert likelihood_kernel(specs, a, a.true_value, e.value_for("open"), "open", CFG) == pytest.approx(1.0), a.id
        assert likelihood_kernel(specs, a, a.true_value, e.value_for("forced_choice"), "forced_choice", CFG) == pytest.approx(1.0), a.id
    loc = CASE.atom("cp_location")
    assert likelihood_kernel(_specs(("vague", {"p": 0.8})), loc, "epigastric", "vague_center", "open", CFG) == pytest.approx(0.8)
    assert likelihood_kernel(_specs(("vague", {"p": 0.8})), loc, "retrosternal", "vague_center", "open", CFG) == pytest.approx(0.8)   # aliasing


def test_sample_params_from_persona():
    rng = np.random.default_rng(0)
    assert sample_params(Persona(e=1), "exaggerate", rng)["k"] == 0
    ks = {sample_params(Persona(e=3), "exaggerate", np.random.default_rng(i))["k"] for i in range(30)}
    assert ks <= {1, 2, 3} and len(ks) >= 2
    assert sample_params(Persona(c=3, cefr="A"), "vague", rng)["p"] == pytest.approx(0.9)
    t, _, specs = build_report_table(CASE, PatientConfig(persona=Persona(e=3), channels=_specs(("exaggerate", {})), seed=3, sample_params_from_persona=True), CFG)
    assert specs[0].params["k"] in (1, 2, 3)


def test_p_withheld():
    a = CASE.atom("assoc_dyspnea")
    assert p_withheld(_specs(), a, 1, "yes_no") == pytest.approx(0.02)
    assert p_withheld(_specs(), a, 1, "open") == 1.0
    om = _specs(("omit", {"q": 1.0, "soften": {2: 1.0}}))
    assert p_withheld(om, a, 1, "yes_no") > 0.9 and p_withheld(om, a, 2, "yes_no") < 0.1


def test_second_batch_not_implemented():
    with pytest.raises(NotImplementedError):
        build_report_table(CASE, cfgp(("understate", {"k": 1})), CFG)
