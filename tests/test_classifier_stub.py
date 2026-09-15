import pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.sim.classifier import KeywordClassifier

CFG = load_cluster_config("chest_pain"); CASE = load_cluster("chest_pain", ["cp_001"])[0]; K = KeywordClassifier(CFG)

EN = [
    ("Hi, what brings you in today?", set(), "open"),
    ("Where exactly is the pain?", {"cp_location"}, "open"),
    ("Is it behind the breastbone, above the stomach, or on the left side?", {"cp_location"}, "forced_choice"),
    ("Can you point to where it hurts?", {"cp_location"}, "point_to"),
    ("What does the pain feel like?", {"cp_quality"}, "open"),
    ("How bad is it? Does it keep you up at night?", {"cp_severity", "cp_functional"}, "severity_open"),
    ("On a scale of 0 to 10, how bad?", {"cp_severity"}, "scale"),
    ("How long has this been going on?", {"cp_duration"}, "open"),
    ("Is it there all the time or does it come and go?", {"cp_pattern"}, "forced_choice"),
    ("Does it get worse after eating?", {"trig_meal"}, "yes_no"),
    ("Any sweating or shortness of breath with it?", {"assoc_sweating", "assoc_dyspnea"}, "yes_no"),
    ("Have you had any tests done before, like an ECG? What did they show?", {"prior_ecg"}, "recall_test"),
    ("Does anyone in your family have heart disease?", {"fam_cad"}, "yes_no"),
    ("Do you smoke?", {"life_smoking"}, "yes_no"),
    ("Are you taking any medication?", {"meds"}, "yes_no"),
    ("I understand, that must be worrying.", set(), "chat"),
]
ZH = [
    ("你好，哪里不舒服？", set(), "open"),
    ("具体疼在哪里？", {"cp_location"}, "open"),
    ("是胸骨后面、胃上面，还是左胸口？", {"cp_location"}, "forced_choice"),
    ("疼得厉害吗？影响睡觉吗？", {"cp_severity", "cp_functional"}, "severity_open"),
    ("吃完饭会加重吗？", {"trig_meal"}, "yes_no"),
    ("有没有出汗、气短？", {"assoc_sweating", "assoc_dyspnea"}, "yes_no"),
    ("以前做过心电图吗？结果怎么说？", {"prior_ecg"}, "recall_test"),
    ("家里有人心脏病吗？", {"fam_cad"}, "yes_no"),
    ("抽烟吗？", {"life_smoking"}, "yes_no"),
    ("别担心，我们一步步来。", set(), "chat"),
]


@pytest.mark.parametrize("q,ids,form", EN)
def test_en(q, ids, form):
    out = K.classify(q, CASE.atoms, [], "en")
    assert set(out.atom_ids) == ids, (q, out)
    assert out.question_form == form, (q, out)


@pytest.mark.parametrize("q,ids,form", ZH)
def test_zh(q, ids, form):
    out = K.classify(q, CASE.atoms, [], "zh")
    assert set(out.atom_ids) == ids, (q, out)
    assert out.question_form == form, (q, out)
