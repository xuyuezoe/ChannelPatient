"""Write results/M1_REPORT.md from a stub run: counts, audit, offline gates, oracle summary, sample dialogues."""
from __future__ import annotations
import json, sys
from pathlib import Path
from scc.analysis.trajectory import load_run, plot_trajectories
from scc.env.cases import load_cluster
from scc.sim.audit import audit
from scc.sim.fidelity import rule_adherence, reask_consistency, channel_identifiability, induce_resistance


def sample_dialogue(run_dir: Path, channel: str, case_id: str = "cp_001", seed: int = 0, max_turns: int = 14) -> str:
    f = next(run_dir.glob(f"{case_id}__{channel}__*seed-{seed}__*.jsonl"))
    out = []
    for r in [json.loads(l) for l in open(f)][:max_turns]:
        d = r["doctor_action"]["text"] if r["doctor_action"] else "(opening)"
        disc = ", ".join(f"{x['atom']}={x['report_value']}" + ("" if x["report_value"] == x["true_value"] else f"(true {x['true_value']})") for x in r["disclosed"])
        oz = r["oracle"]["posterior_z"]; top = max(oz, key=oz.get)
        out.append(f"| {r['turn']} | {d[:60]} | {r['observation']['text'][:90]} | {disc[:80]} | {top} {oz[top]:.2f} |")
    return "\n".join(["| 轮 | 医生 | 患者 / 环境 | 披露 (报告值, 真值若不同) | oracle top |", "|---|---|---|---|---|"] + out)


def main(run_dir: str):
    run_dir = Path(run_dir); df = load_run(run_dir); man = json.load(open(run_dir / "manifest.json"))
    prim = {c.case_id: c.labels.primary_dx for c in load_cluster("chest_pain")}
    df["p_true"] = [oz.get(prim[c]) for oz, c in zip(df.oracle_z, df.case_id)]
    a = audit(run_dir); ra = rule_adherence(run_dir); rc = reask_consistency(run_dir); ci = channel_identifiability(run_dir); ir = induce_resistance(run_dir)
    fin = df[df.obs_kind == "END"].groupby("channel").p_true.agg(["mean", "min", "max", "count"]).round(3)
    turns = df.groupby(["case_id", "channel", "seed"]).turn.max().describe().round(1)
    png = plot_trajectories(run_dir)
    L = [f"# M1 验收报告（stub 模式，{man['started']}）", "",
         f"- run: `{run_dir.name}`，git `{man['git']}`，components = {man['components']}，doctor = {man['config'].get('doctor')}，API 调用 = 0",
         f"- 病例 {df.case_id.nunique()} × 信道 {df.channel.nunique()} × 种子 {df.seed.nunique()} = {man['n_episodes']} 段；失败 {len(man['failed'])}；总轮数 {len(df)}",
         f"- 每段轮数：均值 {turns['mean']}，最小 {turns['min']}，最大 {turns['max']}",
         f"- audit（按规则重算每条报告值）：{a['disclosed_ok']} 条一致，{a['disclosed_mismatch']} 条不一致",
         "", "## 离线门槛（模板句上界，不作为通过证据）", "",
         "| 指标 | 实测 |", "|---|---|",
         f"| 规则遵从 checker pass / fallback | {ra['checker_pass_rate']:.3f} / {ra['fallback_rate']:.3f} |",
         f"| 重问一致性 | {rc['consistency']:.3f}（{rc['pairs']} 对） |",
         f"| 抗诱导（FixedList 医生，仅词表） | {ir['resistance']:.3f} |",
         f"| 分型可辨性 宏 F1（TF-IDF+LR 5 折） | {ci['macro_f1']:.3f}（n={ci['n']}） |",
         "", "## oracle 终局 P(真实诊断) 按信道", "", fin.to_markdown(), "",
         "读法：夸大信道下 oracle 终局后验更低、更分散（模型看到的证据被扭曲，oracle 知道信道不可靠所以打折）；模糊信道靠强制选项问法恢复。",
         "", f"## 轨迹图", "", f"![trajectories]({png.name})", "",
         "上排：oracle P(真实诊断) 随轮次；下排：oracle P(U=真实信道)。细线 = 单段，粗线 = 均值。", ""]
    for ch in df.channel.unique():
        L += [f"## 样例对话：cp_001 / {ch} / seed 0", "", sample_dialogue(run_dir, ch), ""]
    (Path("results") / "M1_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/episodes/m1_stub")
