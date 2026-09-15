"""多医生 / 多信道对比表：读若干 run 目录，汇总 KL、澄清率、核实率、解析率、准确率、Mirage Gap、轮数。

  python -m scc.analysis.compare results/episodes/d_m1_stub [more_run_dirs...] [--out results/x.md]
"""
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from scc.analysis.trajectory import load_run
from scc.analysis import metrics as M
from scc.env.cases import load_cluster


def _primary(cluster: str = "chest_pain") -> dict[str, str]:
    return {c.case_id: c.labels.primary_dx for c in load_cluster(cluster)}


def _with_doctor_text(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """load_run 不带医生原话；这里补一列 doctor_text 供 accuracy_final 用。"""
    import json
    texts = []
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name == "failed.jsonl":
            continue
        for line in open(f):
            r = json.loads(line)
            texts.append((r["doctor_action"] or {}).get("text", ""))
    if len(texts) == len(df):
        df = df.copy(); df["doctor_text"] = texts
    return df


def compare(run_dirs: list[str | Path], cluster: str = "chest_pain") -> pd.DataFrame:
    prim = _primary(cluster)
    frames = []
    for rd in run_dirs:
        rd = Path(rd); df = _with_doctor_text(load_run(rd), rd)
        if df.empty:
            continue
        df["run"] = rd.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    keys = ["channel", "doctor"]
    kl = M.kl_per_turn(df).groupby(keys).kl.mean().reset_index(name="kl_mean")
    kl_last = M.kl_per_turn(df).sort_values("turn").groupby(M.EPISODE_KEYS).kl.last().reset_index().groupby(keys).kl.mean().reset_index(name="kl_final")
    parts = [kl, kl_last, M.clarification_rate(df), M.verify_usage(df), M.readout_parse_rate(df), M.accuracy_final(df, prim),
             M.mirage_gap(df, prim).groupby(keys).mirage_gap.mean().reset_index(),
             M.episode_turns(df).groupby(keys).turns.mean().reset_index(name="turns_mean")]
    pv = M.post_verify_discount_ratio(df, prim)
    if not pv.empty:
        parts.append(pv.groupby(keys).ratio.mean().reset_index(name="post_verify_ratio"))
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=keys, how="outer")
    return out.round(3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", default=None); ap.add_argument("--cluster", default="chest_pain")
    a = ap.parse_args()
    t = compare(a.runs, a.cluster)
    md = t.to_markdown(index=False)
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n")


if __name__ == "__main__":
    main()
