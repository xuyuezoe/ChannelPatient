"""3.3 似然估计抽查：20 个 (z, u, 槽, 问法) 组合，LLM 估计 vs 查表（T1 相关、T2 方向、T3 单调、T4 U 无关不变）。约 20 次 API。

  python scripts/check_llm_likelihood.py [--model qwen3.7-max] [--n 20]
输出 results/D_M4_LIK_CHECK.md 与调用数。
"""
from __future__ import annotations
import argparse, itertools, math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from scc.env.cases import load_cluster_config
from scc.doctor.agent.likelihood import TableLikelihood, LLMFullLikelihood as LLMLikelihood, default_u_library, UNKNOWN, NOT_MENTIONED
from scc.llm import ChatClient

COMBOS = [  # (z, u, slot, tag, form)
    ("GERD", "cooperative", "quality", None, "forced_choice"), ("stable_angina", "cooperative", "quality", None, "forced_choice"),
    ("GERD", "cooperative", "severity", None, "severity_open"), ("GERD", "exaggerate_k1", "severity", None, "severity_open"), ("GERD", "exaggerate_k2", "severity", None, "severity_open"),
    ("stable_angina", "cooperative", "trigger", "exertional", "yes_no"), ("GERD", "cooperative", "trigger", "exertional", "yes_no"), ("GERD", "cooperative", "trigger", "postprandial", "yes_no"),
    ("costochondritis", "cooperative", "trigger", "movement", "yes_no"), ("panic_attack", "cooperative", "associated", "palpitations", "yes_no"),
    ("GERD", "cooperative", "associated", "palpitations", "yes_no"), ("pericarditis", "cooperative", "trigger", "breathing", "yes_no"),
    ("GERD", "cooperative", "location", None, "open"), ("GERD", "vague_p08", "location", None, "open"), ("GERD", "vague_p08", "location", None, "forced_choice"),
    ("stable_angina", "cooperative", "radiation", None, "forced_choice"), ("stable_angina", "cooperative", "family_history", "cad", "yes_no"), ("stable_angina", "exaggerate_k2", "family_history", "cad", "yes_no"),
    ("panic_attack", "cooperative", "onset", None, "forced_choice"), ("costochondritis", "cooperative", "severity", None, "severity_open"),
]


def core(d: dict) -> dict:
    d = {k: v for k, v in d.items() if k not in (UNKNOWN, NOT_MENTIONED)}; s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3.7-max"); ap.add_argument("--n", type=int, default=20); a = ap.parse_args()
    cfg = load_cluster_config("chest_pain"); ulib = default_u_library(cfg)
    T = TableLikelihood(cfg, ulib); client = ChatClient(a.model, temperature=0.0); Lm = LLMLikelihood(cfg, ulib, client, "en")
    rows = []; xs, ys = [], []
    for z, u, slot, tag, form in COMBOS[: a.n]:
        dt = core(T.dist(z, u, slot, tag, form)); dl = core(Lm.dist(z, u, slot, tag, form))
        for k in dt:
            xs.append(dt[k]); ys.append(dl.get(k, 0.0))
        rows.append((z, u, slot, tag, form, dt, dl))
    rho = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
    # T2 方向：成对比较（同槽同问法、不同 z）argmax 是否一致
    by = {}
    for r in rows:
        by.setdefault((r[2], r[3], r[4], r[1]), []).append(r)
    t2_ok = t2_n = 0
    for k, grp in by.items():
        for r1, r2 in itertools.combinations(grp, 2):
            for v in r1[5]:
                sign_t = np.sign(r1[5][v] - r2[5][v]); sign_l = np.sign(r1[6].get(v, 0) - r2[6].get(v, 0))
                if sign_t != 0:
                    t2_n += 1; t2_ok += (sign_t == sign_l)
    # T3 单调：GERD severity 在 cooperative / k1 / k2 下 P(severe) 递增
    sev = {r[1]: r[6].get("severe", 0.0) for r in rows if r[0] == "GERD" and r[2] == "severity"}
    t3 = sev.get("cooperative", 0) <= sev.get("exaggerate_k1", 0) <= sev.get("exaggerate_k2", 0)
    # T4 U 无关：family_history 在 cooperative vs exaggerate_k2 下 LLM 估计差 < 0.1
    fam = {r[1]: r[6].get(True, 0.0) for r in rows if r[2] == "family_history"}
    t4 = abs(fam.get("cooperative", 0) - fam.get("exaggerate_k2", 0)) < 0.1 if len(fam) == 2 else None
    L = ["# D-M4 似然估计抽查（LLM vs 查表）", "", f"模型 {a.model}；组合 {len(rows)}；API 调用 {client.calls}（缓存命中 {Lm.n_cache}，回退 {Lm.n_fallback}）", "",
         f"- T1 相关：ρ = {rho:.3f}（门槛 ≥ 0.6）", f"- T2 方向一致：{t2_ok}/{t2_n} = {t2_ok / max(1, t2_n):.2f}（门槛 ≥ 0.8）",
         f"- T3 单调（GERD 严重度 P(severe) coop ≤ k1 ≤ k2）：{sev} → {'通过' if t3 else '不通过'}",
         f"- T4 U 无关（家族史在 coop vs k2）：{fam} → {'通过' if t4 else '不通过'}", "", "| z | u | slot/tag | form | 查表 | LLM |", "|---|---|---|---|---|---|"]
    for z, u, slot, tag, form, dt, dl in rows:
        L.append(f"| {z} | {u} | {slot}{'/' + tag if tag else ''} | {form} | {', '.join(f'{k}:{v:.2f}' for k, v in dt.items())} | {', '.join(f'{k}:{v:.2f}' for k, v in dl.items())} |")
    pathlib.Path("results/D_M4_LIK_CHECK.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:8])); print("api calls:", client.calls)


if __name__ == "__main__":
    main()
