"""Belief trajectories: oracle P(primary_dx) and doctor readout per turn, faceted by channel; plus P(U=true channel)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd


def load_run(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    rows = []
    for f in sorted(run_dir.glob("*.jsonl")):
        if f.name == "failed.jsonl":
            continue
        for line in open(f):
            r = json.loads(line)
            parts = r["episode_id"].split("|")
            case_id, channel, persona, seed, doctor = parts[0], parts[1], parts[2], parts[3].split("=")[1], parts[4].split("=")[1]
            oz = (r.get("oracle") or {}).get("posterior_z", {}); ou = (r.get("oracle") or {}).get("posterior_u", {})
            ro = (r.get("doctor_action") or {}).get("readout") if r.get("doctor_action") else None
            rows.append({"case_id": case_id, "channel": channel, "persona": persona, "seed": int(seed), "doctor": doctor, "turn": r["turn"],
                         "obs_kind": r["observation"]["kind"], "question_form": r.get("question_form"), "n_disclosed": len(r["disclosed"]),
                         "n_withheld": len(r["withheld"]), "checker_pass": r["checker"].get("pass"), "checker_fallback": r["checker"].get("fallback"),
                         "oracle_z": oz, "oracle_u": ou, "entropy_z": (r.get("oracle") or {}).get("entropy_z"),
                         "readout_topk": (ro or {}).get("topk"), "readout_conf": (ro or {}).get("confidence"), "action_kind": (r.get("doctor_action") or {}).get("kind")})
    return pd.DataFrame(rows)


def _primary_of(case_id: str, cluster: str = "chest_pain") -> str:
    from scc.env.cases import load_cluster
    return load_cluster(cluster, [case_id])[0].labels.primary_dx


def plot_trajectories(run_dir: str | Path, out_png: str | Path | None = None, cluster: str = "chest_pain") -> Path:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir); df = load_run(run_dir)
    if df.empty:
        raise SystemExit("no logs")
    prim = {c: _primary_of(c, cluster) for c in df.case_id.unique()}
    df["p_primary"] = [oz.get(prim[c], float("nan")) for oz, c in zip(df.oracle_z, df.case_id)]
    df["p_readout"] = [ (tk or {}).get(prim[c]) if tk else float("nan") for tk, c in zip(df.readout_topk, df.case_id)]
    def true_u(ch):
        return {"cooperative": "cooperative", "exaggerate_k2": "exaggerate_k2", "exaggerate_k1": "exaggerate_k1", "vague_p08": "vague_p08", "omit_q05": "omit_q05"}.get(ch, None)
    df["p_true_u"] = [ (ou or {}).get(true_u(ch), float("nan")) for ou, ch in zip(df.oracle_u, df.channel)]
    chans = list(df.channel.unique())
    fig, axes = plt.subplots(2, len(chans), figsize=(4 * len(chans), 6.5), sharex=True, squeeze=False)
    for j, ch in enumerate(chans):
        sub = df[df.channel == ch]
        ax = axes[0, j]
        for (cid, seed), g in sub.groupby(["case_id", "seed"]):
            ax.plot(g.turn, g.p_primary, color="tab:blue", alpha=0.25, lw=1)
            if g.p_readout.notna().any():
                ax.plot(g.turn, g.p_readout, color="tab:red", alpha=0.25, lw=1, ls="--")
        m = sub.groupby("turn").p_primary.mean(); ax.plot(m.index, m.values, color="tab:blue", lw=2.5, label="oracle P(true dx)")
        if sub.p_readout.notna().any():
            m2 = sub.groupby("turn").p_readout.mean(); ax.plot(m2.index, m2.values, color="tab:red", lw=2.5, ls="--", label="doctor readout")
        ax.set_title(ch); ax.set_ylim(0, 1); ax.grid(alpha=.3)
        if j == 0: ax.set_ylabel("P(true diagnosis)"); ax.legend(fontsize=8)
        ax2 = axes[1, j]
        for (cid, seed), g in sub.groupby(["case_id", "seed"]):
            ax2.plot(g.turn, g.p_true_u, color="tab:green", alpha=0.25, lw=1)
        m3 = sub.groupby("turn").p_true_u.mean(); ax2.plot(m3.index, m3.values, color="tab:green", lw=2.5)
        ax2.set_ylim(0, 1); ax2.grid(alpha=.3); ax2.set_xlabel("turn")
        if j == 0: ax2.set_ylabel("oracle P(U = true channel)")
    fig.suptitle(f"{run_dir.name}: oracle belief trajectories (thin = episodes, thick = mean)")
    fig.tight_layout()
    out = Path(out_png) if out_png else run_dir / "trajectories.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_trajectories(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
