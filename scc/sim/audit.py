"""Audit a run: recompute every disclosed report_value from the channel rules and compare with the log."""
from __future__ import annotations
import json, sys
from pathlib import Path
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.channels import build_report_table
from scc.types import PatientConfig, ChannelSpec, Persona


def audit(run_dir: str | Path, cluster: str = "chest_pain") -> dict:
    run_dir = Path(run_dir); cfg = load_cluster_config(cluster)
    manifest = json.load(open(run_dir / "manifest.json"))
    chan_specs = {c["name"]: [ChannelSpec.from_dict(s) for s in c.get("specs", [])] for c in manifest["config"].get("channels", [])}
    persona = Persona.from_dict(manifest["config"].get("persona", {}))
    n_ok = n_bad = n_turns = 0; bad = []
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name == "failed.jsonl":
            continue
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        cid, ch, _, seed, _ = rows[0]["episode_id"].split("|")
        seed = int(seed.split("=")[1])
        case = load_cluster(cluster, [cid])[0]
        pc = PatientConfig(persona=persona, channels=chan_specs.get(ch, []), seed=seed, post_verify_shift=int(manifest["config"].get("post_verify_shift", 0)), name=ch)
        table, sd, _ = build_report_table(case, pc, cfg)
        verified = set()
        for r in rows:
            n_turns += 1
            for d in r["disclosed"]:
                if str(d["rule"]).startswith("verified"):
                    verified.add(d["atom"]); expect = case.atom(d["atom"]).true_value
                elif d["atom"] in verified:
                    expect = case.atom(d["atom"]).true_value
                else:
                    e = table[d["atom"]]; expect = e.value_for(r["question_form"] or "open")
                if d["report_value"] == expect:
                    n_ok += 1
                else:
                    n_bad += 1; bad.append((f.name, r["turn"], d["atom"], d["report_value"], expect))
    return {"episodes": len(list(run_dir.glob("*.jsonl"))), "turns": n_turns, "disclosed_ok": n_ok, "disclosed_mismatch": n_bad, "examples": bad[:10]}


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=1, ensure_ascii=False))
