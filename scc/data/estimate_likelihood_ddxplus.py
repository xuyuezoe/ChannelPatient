"""Estimate P(slot value | condition) from the DDXPlus test split for slots that have a ddxplus mapping in slots.yaml.

Usage: python -m scc.data.estimate_likelihood_ddxplus --cluster chest_pain [--out results/ddxplus_likelihood_stats.yaml]
Output is a YAML with the same shape as likelihood.yaml (dx -> slot -> value -> prob), to be hand-merged.
"""
from __future__ import annotations
import argparse, ast, csv, collections, json
from pathlib import Path
import yaml
from scc.config import settings, ROOT

DDX = ROOT / "data" / "external" / "ddxplus"


def load_cluster(cluster: str):
    d = settings.clusters_dir / cluster
    return yaml.safe_load(open(d / "cluster.yaml")), yaml.safe_load(open(d / "slots.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="chest_pain")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cluster, slots = load_cluster(args.cluster)
    cond_map = {v: k for k, v in cluster["ddxplus_conditions"].items() if v}   # ddxplus name -> our dx
    # accumulators
    n = collections.Counter()
    cat = {dx: collections.defaultdict(collections.Counter) for dx in cond_map.values()}     # dx -> slot -> value -> count
    tagged = {dx: collections.defaultdict(int) for dx in cond_map.values()}                  # dx -> (slot, tag) -> count of True
    with open(DDX / "release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if row["PATHOLOGY"] not in cond_map:
                continue
            dx = cond_map[row["PATHOLOGY"]]
            n[dx] += 1
            evs = ast.literal_eval(row["EVIDENCES"])
            present = set(evs)
            for slot, sd in slots.items():
                dd = sd.get("ddxplus")
                if not isinstance(dd, dict):
                    continue
                if "evidence" in dd and "map" in dd:
                    ev = dd["evidence"]; vals = [e.split("@_")[1] for e in evs if e.startswith(ev + "_@_")]
                    mapped = [dd["map"][v] for v in vals if v in dd["map"]]
                    if slot == "location":
                        # multi-valued in DDXPlus: pick the first chest value in our option order
                        for opt in sd["options"]:
                            if opt in mapped:
                                cat[dx][slot][opt] += 1; break
                    elif slot == "radiation":
                        if not mapped or mapped == ["none"]:
                            cat[dx][slot]["none"] += 1
                        else:
                            for opt in sd["options"][1:]:
                                if opt in mapped:
                                    cat[dx][slot][opt] += 1; break
                    else:
                        for m in set(mapped):
                            cat[dx][slot][m] += 1
                elif "evidence" in dd and "map_scale" in dd:
                    ev = dd["evidence"]; vals = [int(e.split("@_")[1]) for e in evs if e.startswith(ev + "_@_")]
                    for v in vals:
                        for band, (lo, hi) in dd["map_scale"].items():
                            if lo <= v <= hi:
                                cat[dx][slot][band] += 1
                elif "tags" in dd:
                    for tag, ev in dd["tags"].items():
                        if ev in present:
                            tagged[dx][(slot, tag)] += 1
    out = {}
    for dx in cond_map.values():
        out[dx] = {"_n": n[dx]}
        for slot, counter in cat[dx].items():
            tot = sum(counter.values())
            if tot:
                out[dx][slot] = {k: round(v / tot, 3) for k, v in sorted(counter.items())}
        for (slot, tag), c in tagged[dx].items():
            out[dx].setdefault(slot, {})[tag] = round(c / n[dx], 3)
    text = yaml.safe_dump(out, allow_unicode=True, sort_keys=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
