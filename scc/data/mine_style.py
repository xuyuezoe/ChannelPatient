"""Mine real-corpus style material (no API calls):
  1. IMCS-21 patient sentences containing intensity words, bucketed by severity level (vocab.yaml severity_words)
  2. parent self-diagnosis sentences (data/imcs21_mined.json guesses + regex)
  3. ask-then-disclose pairs: doctor asks the same symptom >= 2 times before the patient informs it -> n distribution
  4. CPB-Bench self-diagnosis sentences (zh: IMCS/MedDG, en: ACI/MediTOD)
Outputs data/style/{zh,en}/*.jsonl and data/style/stats.md
"""
from __future__ import annotations
import json, re, collections
from pathlib import Path
import yaml
from scc.config import ROOT

IMCS = ROOT / "data" / "imcs21" / "dataset"
MINED = ROOT / "data" / "imcs21_mined.json"
CPB = ROOT / "external" / "cpb-bench" / "cpb-bench_data" / "Pos_behavior_692"
OUT = ROOT / "data" / "style"
SELFDX = re.compile(r"(是不是|会不会是|是否是|像是|我觉得是|怀疑是|应该是|可能是|有没有可能是|肯定是)")
EXTRA_SEVERE = ["要死", "受不了", "特别厉害", "疼得不行", "非常严重", "吓死", "整晚", "一晚上没睡", "很严重"]
EXTRA_MODERATE = ["挺厉害", "比较严重", "有点厉害", "很不舒服", "老是", "一直"]
EXTRA_MILD = ["有点", "一点点", "不太", "轻微", "还好", "偶尔", "稍微"]


def load_imcs():
    d = {}
    for sp in ("train", "dev", "test"):
        d.update(json.load(open(IMCS / f"{sp}.json")))
    return d


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    vocab = yaml.safe_load(open(ROOT / "scc/env/clusters/chest_pain/vocab.yaml"))
    words = {lvl: list(vocab["severity_words"][lvl]["zh"]) for lvl in ("mild", "moderate", "severe")}
    words["severe"] += EXTRA_SEVERE; words["moderate"] += EXTRA_MODERATE; words["mild"] += EXTRA_MILD
    D = load_imcs(); mined = {m["id"]: m for m in json.load(open(MINED))}
    sev_rows, selfdx_rows, soften_rows = [], [], []
    n_dist = collections.Counter()
    for did, v in D.items():
        turns = v["dialogue"]
        asked: dict[str, list[int]] = collections.defaultdict(list)
        for i, t in enumerate(turns):
            s = t["sentence"]
            if t["speaker"] == "患者":
                for lvl in ("severe", "moderate", "mild"):
                    if any(w in s for w in words[lvl]) and 4 <= len(s) <= 60:
                        sev_rows.append({"text": s, "level": lvl, "source": f"imcs21:{did}:{t['sentence_id']}"}); break
                if SELFDX.search(s) and 4 <= len(s) <= 80:
                    selfdx_rows.append({"text": s, "guess": mined.get(did, {}).get("guesses"), "source": f"imcs21:{did}:{t['sentence_id']}"})
                if t["dialogue_act"] == "Inform-Symptom":
                    for sym in t.get("symptom_norm", []):
                        if len(asked[sym]) >= 2:
                            n = len(asked[sym]); n_dist[min(n, 4)] += 1
                            soften_rows.append({"symptom": sym, "n_asked_before_inform": n, "source": f"imcs21:{did}", "doctor_turns": asked[sym][:4], "patient_turn": t["sentence_id"]})
                        asked[sym] = []   # reset once informed
            elif t["speaker"] == "医生" and t["dialogue_act"] == "Request-Symptom":
                for sym in t.get("symptom_norm", []):
                    asked[sym].append(t["sentence_id"])
    # CPB self-dx
    cpb_zh, cpb_en = [], []
    for f in sorted(CPB.glob("*_safety_benchmark.json")):
        d = json.load(open(f))
        for c in d["cases"]:
            if c["behavior_category"] == "Self-diagnosis":
                row = {"text": c["patient_behavior_text"], "source": f"cpb:{c['case_id']}"}
                (cpb_zh if d["dataset_name"] in ("IMCS", "MedDG") else cpb_en).append(row)
    write_jsonl(OUT / "zh" / "severity.jsonl", sev_rows)
    write_jsonl(OUT / "zh" / "self_dx.jsonl", selfdx_rows + cpb_zh)
    write_jsonl(OUT / "zh" / "soften.jsonl", soften_rows)
    write_jsonl(OUT / "en" / "self_dx.jsonl", cpb_en)
    lv = collections.Counter(r["level"] for r in sev_rows)
    tot = sum(n_dist.values()) or 1
    stats = ["# 语料挖掘统计（mine_style.py，无 API）", "",
             f"- IMCS-21 患者句含程度词：{len(sev_rows)}（severe {lv['severe']}, moderate {lv['moderate']}, mild {lv['mild']}）",
             f"- IMCS-21 家长自诊句（正则）：{len(selfdx_rows)}；CPB-Bench 自诊句 zh {len(cpb_zh)} / en {len(cpb_en)}",
             f"- 追问-松口对（医生对同一症状 Request ≥2 次后患者才 Inform）：{len(soften_rows)}", "",
             "| 追问次数 n | 次数 | 占比 |", "|---|---|---|"] + [f"| {n} | {c} | {c/tot:.2f} |" for n, c in sorted(n_dist.items())] + [
             "", "用途：severity.jsonl → 措辞器 few-shot（按档）；self_dx.jsonl → 自诊句 few-shot；soften 分布 → omit 信道 soften 初值。"]
    (OUT / "stats.md").write_text("\n".join(stats) + "\n")
    print("\n".join(stats))


if __name__ == "__main__":
    main()
