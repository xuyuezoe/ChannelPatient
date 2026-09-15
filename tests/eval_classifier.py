"""J2 (post-integration): evaluate Keyword vs LLM classifier on 50 hand-labelled questions. ~50 API calls.
  python tests/eval_classifier.py [--llm]
"""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scc.env.cases import load_cluster, load_cluster_config
from scc.sim.classifier import KeywordClassifier, LLMClassifier
from test_classifier_stub import EN, ZH

EXTRA_EN = [("Tell me more about the pain.", set(), "open"), ("Is the pain sharp or more like a pressure?", {"cp_quality"}, "forced_choice"),
            ("Did it start suddenly?", {"cp_onset"}, "yes_no"), ("Does lying down make it worse?", {"trig_lying_down", "trig_lying"}, "yes_no"),
            ("Does it spread to your arm or jaw?", {"cp_radiation"}, "yes_no"), ("Do you have high blood pressure or diabetes?", {"hx_htn", "hx_dm", "hx_hypertension", "hx_diabetes"}, "yes_no"),
            ("What did the doctor say last time?", {"prior_ecg", "prior_test", "prior_severity"}, "recall_test"), ("Okay, let's move on.", set(), "chat"),
            ("Does resting help?", {"rel_rest"}, "yes_no"), ("Do you drink alcohol?", {"life_alcohol"}, "yes_no")]
EXTRA_ZH = [("再说说这个疼。", set(), "open"), ("是刺痛还是闷痛？", {"cp_quality"}, "forced_choice"), ("是突然开始的吗？", {"cp_onset"}, "yes_no"),
            ("躺下会加重吗？", {"trig_lying_down", "trig_lying"}, "yes_no"), ("会串到胳膊或者下巴吗？", {"cp_radiation"}, "yes_no"), ("有高血压或糖尿病吗？", {"hx_htn", "hx_dm", "hx_hypertension", "hx_diabetes"}, "yes_no"),
            ("上次医生怎么说的？", {"prior_ecg", "prior_test", "prior_severity"}, "recall_test"), ("好，我们继续。", set(), "chat"), ("休息会好点吗？", {"rel_rest"}, "yes_no"), ("喝酒吗？", {"life_alcohol"}, "yes_no")]


def run(clf, items, lang, atoms):
    tp = fp = fn = 0; form_ok = 0
    for q, ids, form in items:
        out = clf.classify(q, atoms, [], lang)
        got = set(out.atom_ids); want = {i for i in ids if i in {a.id for a in atoms}}
        tp += len(got & want); fp += len(got - want); fn += len(want - got); form_ok += (out.question_form == form)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"f1": 2 * p * r / max(1e-9, p + r), "form_acc": form_ok / len(items), "n": len(items)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", action="store_true"); args = ap.parse_args()
    cfg = load_cluster_config("chest_pain"); case = load_cluster("chest_pain", ["cp_001"])[0]
    clfs = {"keyword": KeywordClassifier(cfg)}
    if args.llm:
        from scc.llm import ChatClient
        clfs["llm"] = LLMClassifier(cfg, ChatClient("qwen3.7-max"))
    res = {name: {"en": run(c, EN + EXTRA_EN, "en", case.atoms), "zh": run(c, ZH + EXTRA_ZH, "zh", case.atoms)} for name, c in clfs.items()}
    print(json.dumps(res, indent=1))
