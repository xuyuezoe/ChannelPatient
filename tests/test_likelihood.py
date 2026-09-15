import pytest
from scc.env.cases import load_cluster_config
from scc.env.likelihood import Likelihood

CFG = load_cluster_config("chest_pain")
L = Likelihood(CFG)


def test_all_rows_normalised():
    for dx in CFG.ddx_set:
        for slot, sd in CFG.slots.items():
            if sd["type"] == "bool":
                for tag in sd.get("tag_options", [None]):
                    d = L.dist(dx, slot, tag); assert abs(sum(d.values()) - 1) < 1e-9 and 0 <= d[True] <= 1
            elif sd["type"] == "text":
                assert L.dist(dx, slot) == {}
            elif "tag_values" in sd:
                for tag, opts in sd["tag_values"].items():
                    d = L.dist(dx, slot, tag); assert set(d) == set(opts) and abs(sum(d.values()) - 1) < 1e-9
            else:
                d = L.dist(dx, slot); assert set(d) == set(sd["options"]) and abs(sum(d.values()) - 1) < 1e-9, (dx, slot)


def test_direction_sanity():
    assert L.p_true("GERD", "quality", "burning") > L.p_true("stable_angina", "quality", "burning")
    assert L.p_true("stable_angina", "trigger", True, "exertional") > 0.8 > L.p_true("GERD", "trigger", True, "exertional")
    assert L.p_true("costochondritis", "trigger", True, "movement") > L.p_true("panic_attack", "trigger", True, "movement")
    assert L.p_true("GERD", "severity", "severe") < L.p_true("pericarditis", "severity", "severe")


def test_missing_uniform_and_noise():
    L2 = Likelihood(CFG, table={"GERD": {}})
    assert L2.dist("GERD", "location") == pytest.approx({o: 0.25 for o in CFG.slots["location"]["options"]})
    assert L2.dist("GERD", "trigger", "exertional") == {True: 0.5, False: 0.5}
    Ln = L.noised(0.5, seed=1)
    d = Ln.dist("GERD", "location"); assert abs(sum(d.values()) - 1) < 1e-9 and d != L.dist("GERD", "location")
