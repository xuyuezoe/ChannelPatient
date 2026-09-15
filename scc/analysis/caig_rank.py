"""CA-IG 排名：事后给任何医生每一轮实际问的问题打分——它在当时信念下的 CA-IG 在菜单里排第几（分位数，1 = 最好）。

信念用查表似然从日志里的披露值重放（与 oracle 同表），所以对被测医生 / RL 医生同样适用。
输出每轮：{episode, turn, slot, form, caig, rank_pct, n_menu, misspecified_rank_pct}。
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scc.env.cases import load_cluster, load_cluster_config
from scc.doctor.agent.likelihood import TableLikelihood, default_u_library
from scc.doctor.agent.belief import BeliefEngine
from scc.doctor.agent.candidates import MenuCandidates
from scc.doctor.agent.value import ca_ig, misspecified_ig
from scc.doctor.agent.state import SemanticObs
from scc.types import ActionKind


def caig_rank(run_dir: str | Path, cluster: str = "chest_pain") -> pd.DataFrame:
    run_dir = Path(run_dir); cfg = load_cluster_config(cluster); cases = {c.case_id: c for c in load_cluster(cluster)}
    ulib = default_u_library(cfg); P = TableLikelihood(cfg, ulib); menu = MenuCandidates(cfg).propose({})
    rows = []
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name == "failed.jsonl":
            continue
        logs = [json.loads(l) for l in open(f)]
        if not logs:
            continue
        cid = logs[0]["episode_id"].split("|")[0]; case = cases[cid]
        b = BeliefEngine(list(cfg.ddx_set), list(ulib), P); n_asked: dict = {}
        for r in logs:
            d = r["doctor_action"]
            if d and d["kind"] == "ASK" and r["hit_atoms"]:
                a = case.atom(r["hit_atoms"][0]); form = r["question_form"] or "open"
                asked = [c for c in menu if c.slot == a.slot and c.tag == a.tag and c.form == form]
                if not asked:
                    asked = [c for c in menu if c.slot == a.slot and c.tag == a.tag]
                if asked:
                    scores = sorted(((ca_ig(b, c, n_asked.get((c.slot, c.tag), 0) + 1), c.key()) for c in menu if c.kind == ActionKind.ASK), reverse=True)
                    ms = sorted(((misspecified_ig(b, c, n_asked=n_asked.get((c.slot, c.tag), 0) + 1), c.key()) for c in menu if c.kind == ActionKind.ASK), reverse=True)
                    k = asked[0].key(); n = len(scores)
                    rank = next(i for i, (_, kk) in enumerate(scores) if kk == k); mrank = next(i for i, (_, kk) in enumerate(ms) if kk == k)
                    rows.append({"episode_id": r["episode_id"], "turn": r["turn"], "slot": a.slot, "tag": a.tag, "form": form, "caig": scores[rank][0],
                                 "rank_pct": 1 - rank / max(1, n - 1), "misspecified_rank_pct": 1 - mrank / max(1, n - 1), "n_menu": n})
                n_asked[(a.slot, a.tag)] = n_asked.get((a.slot, a.tag), 0) + 1
            for dd in r["disclosed"]:
                a = case.atom(dd["atom"])
                if str(dd["rule"]).startswith("verified"):
                    b.observe_true(a.slot, a.tag, dd["true_value"])
                else:
                    b.update_soft(SemanticObs(a.slot, a.tag, r["question_form"] or "open", {dd["report_value"]: 1.0}), max(1, n_asked.get((a.slot, a.tag), 1)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = caig_rank(sys.argv[1])
    print(df.groupby(df.episode_id.str.split("|").str[4]).agg(rank_pct=("rank_pct", "mean"), mis=("misspecified_rank_pct", "mean"), n=("turn", "count")).round(3))
