"""Fidelity gates over run logs (J2, run after the doctor side is connected):
leakage_rate (LLM), reask_consistency, induce_resistance, channel_identifiability, rule_adherence; report() -> Markdown."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from scc.env.cases import load_cluster, load_cluster_config, ClusterConfig
from scc.sim.audit import audit


def _episodes(run_dir: Path) -> Iterable[tuple[str, list[dict]]]:
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name == "failed.jsonl":
            continue
        rows = [json.loads(l) for l in open(f)]
        if rows:
            yield rows[0]["episode_id"], rows


def _patient_rows(rows):
    return [r for r in rows if r["observation"]["kind"] == "PATIENT"]


def rule_adherence(run_dir: str | Path, cluster: str = "chest_pain") -> dict:
    run_dir = Path(run_dir); a = audit(run_dir, cluster)
    n_pass = n_all = n_fb = 0
    for _, rows in _episodes(run_dir):
        for r in _patient_rows(rows):
            n_all += 1; n_pass += bool(r["checker"].get("pass")); n_fb += bool(r["checker"].get("fallback"))
    return {"checker_pass_rate": n_pass / max(1, n_all), "fallback_rate": n_fb / max(1, n_all), "audit_mismatch": a["disclosed_mismatch"], "audit_ok": a["disclosed_ok"]}


def reask_consistency(run_dir: str | Path) -> dict:
    """Same atom disclosed twice in an episode -> same report_value (or same level)."""
    same = diff = 0
    for _, rows in _episodes(Path(run_dir)):
        seen: dict[str, object] = {}
        for r in rows:
            for d in r["disclosed"]:
                if str(d["rule"]).startswith("verified"):
                    continue
                a = d["atom"]
                if a in seen:
                    if seen[a] == d["report_value"]:
                        same += 1
                    else:
                        diff += 1
                else:
                    seen[a] = d["report_value"]
    return {"pairs": same + diff, "consistency": same / max(1, same + diff)}


def induce_resistance(run_dir: str | Path, cluster: str = "chest_pain") -> dict:
    cfg = load_cluster_config(cluster)
    n = bad = 0
    for eid, rows in _episodes(Path(run_dir)):
        cid = eid.split("|")[0]; case = load_cluster(cluster, [cid])[0]
        lang = "zh" if any("一" <= ch <= "鿿" for ch in rows[0]["observation"]["text"]) else "en"
        terms = cfg.vocab.get("red_flag_terms", {}).get(lang, []) + cfg.vocab.get("diagnosis_terms", {}).get(lang, [])
        self_dx = None
        for r in rows:
            if r.get("self_dx_directive"):
                self_dx = True
        for r in _patient_rows(rows):
            n += 1
            t = r["observation"]["text"].lower()
            if any(term.lower() in t for term in terms) and not self_dx:
                bad += 1
    return {"patient_turns": n, "resistance": 1 - bad / max(1, n)}


def channel_identifiability(run_dir: str | Path, folds: int = 5) -> dict:
    """TF-IDF + logistic regression, K-fold macro-F1 on patient text -> channel name."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score
    X, y = [], []
    for eid, rows in _episodes(Path(run_dir)):
        X.append(" ".join(r["observation"]["text"] for r in _patient_rows(rows))); y.append(eid.split("|")[1])
    if len(set(y)) < 2 or len(y) < folds:
        return {"n": len(y), "macro_f1": float("nan"), "note": "not enough episodes/classes"}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="char_wb" if any("一" <= c <= "鿿" for c in X[0]) else "word")
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    classes = sorted(set(y)); counts = np.bincount([classes.index(v) for v in y])
    k = int(min(folds, counts.min()))
    pred = cross_val_predict(clf, Xv, y, cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=0))
    return {"n": len(y), "classes": sorted(set(y)), "macro_f1": float(f1_score(y, pred, average="macro"))}


def leakage_rate(run_dir: str | Path, client, cluster: str = "chest_pain", max_episodes: int | None = None) -> dict:
    """LLM-judged: patient sentences stating medical facts not in the allowed content (one call per patient turn)."""
    from scc.sim import prompts
    from scc.sim.phraser import TemplatePhraser
    cfg = load_cluster_config(cluster); tp = TemplatePhraser(cfg)
    n = leaks = calls = 0
    schema = {"type": "object", "properties": {"new_medical_facts": {"type": "array", "items": {"type": "string"}}}, "required": ["new_medical_facts"], "additionalProperties": False}
    for k, (eid, rows) in enumerate(_episodes(Path(run_dir))):
        if max_episodes and k >= max_episodes:
            break
        cid = eid.split("|")[0]; case = load_cluster(cluster, [cid])[0]
        lang = "zh" if any("一" <= ch <= "鿿" for ch in rows[0]["observation"]["text"]) else "en"
        for r in _patient_rows(rows):
            allowed = "\n".join("- " + tp._one(case.atom(d["atom"]), d["report_value"], lang) for d in r["disclosed"]) or "- (nothing)"
            out = client.chat([{"role": "user", "content": prompts.get("ATOMIC_FACTS", lang).format(allowed=allowed, utterance=r["observation"]["text"])}], max_tokens=200, temperature=0.0, json_schema=schema)
            calls += 1; n += 1; leaks += bool(out.get("new_medical_facts"))
    return {"patient_turns": n, "leak_rate": leaks / max(1, n), "llm_calls": calls}


def report(main_run: str | Path, reask_run: str | Path | None = None, induce_run: str | Path | None = None, leak: dict | None = None, cluster: str = "chest_pain") -> str:
    ra = rule_adherence(main_run, cluster); ci = channel_identifiability(main_run)
    rc = reask_consistency(reask_run or main_run); ir = induce_resistance(induce_run or main_run, cluster)
    lines = ["# 保真度门槛实测", "", "| 指标 | 实测 | 第一版阈值 |", "|---|---|---|",
             f"| 泄露率 | {leak['leak_rate']:.3f} (n={leak['patient_turns']}) | < 0.05 |" if leak else "| 泄露率 | 未测（需 LLM） | < 0.05 |",
             f"| 重问一致性 | {rc['consistency']:.3f} (pairs={rc['pairs']}) | > 0.90 |",
             f"| 抗诱导 | {ir['resistance']:.3f} (turns={ir['patient_turns']}) | > 0.85 |",
             f"| 分型可辨性 宏F1 | {ci['macro_f1']:.3f} (n={ci['n']}) | > 0.60 |",
             f"| 规则遵从（checker pass / audit 不一致） | {ra['checker_pass_rate']:.3f} / {ra['audit_mismatch']} | > 0.95 / 0 |"]
    return "\n".join(lines) + "\n"
