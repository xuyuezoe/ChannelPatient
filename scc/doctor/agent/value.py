"""动作价值：CA-IG、误设 IG、联合互信息、两步前瞻、锚定动作价值。单位 nat。

记号：b 为 BeliefEngine；c 为 Candidate；P(r | c) = Σ_{z,u} P(z,u) P(r | z, u, slot, form)。
CA-IG(c) = H[P(z)] − Σ_r P(r|c) H[P(z | r, c)]。
两步前瞻只对标定型候选（澄清、核实）展开：value2(c) = IG1(c) + [ E_r max_b IG1(b | r, c) − max_b IG1(b) ]⁺。
"""
from __future__ import annotations
import math
from typing import Any
from scc.doctor.agent.belief import BeliefEngine
from scc.doctor.agent.state import Candidate, SemanticObs, UNKNOWN, NOT_MENTIONED
from scc.types import ActionKind


def _H(p: dict) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def outcomes(b: BeliefEngine, c: Candidate, n_asked: int = 1) -> list[tuple[Any, float, BeliefEngine]]:
    """枚举候选的可能结果：(结果值, 概率, 更新后的信念副本)。"""
    if c.kind == ActionKind.ASK:
        pred = b.predictive(c.slot, c.tag, c.form or "open", n_asked)
        res = []
        for r, p in pred.items():
            if p <= 1e-9:
                continue
            nb = b.clone()
            nb.update_soft(SemanticObs(c.slot, c.tag, c.form or "open", {r: 1.0}), n_asked)
            res.append((r, p, nb))
        return res
    if c.kind in (ActionKind.VERIFY_RECORD, ActionKind.ASK_FAMILY):
        pz = b.posterior_z(); pt_all = {z: b.P.p_true(z, c.slot, c.tag) for z in b.zs}
        vals = set().union(*[set(d) for d in pt_all.values()]) if any(pt_all.values()) else set()
        res = []
        for v in vals:
            p = sum(pz[z] * pt_all[z].get(v, 0.0) for z in b.zs)
            if p <= 1e-9:
                continue
            nb = b.clone(); nb.observe_true(c.slot, c.tag, v)
            res.append((v, p, nb))
        return res
    return []


def ca_ig(b: BeliefEngine, c: Candidate, n_asked: int = 1) -> float:
    h0 = b.entropy_z()
    outs = outcomes(b, c, n_asked)
    if not outs:
        return 0.0
    return max(0.0, h0 - sum(p * nb.entropy_z() for _, p, nb in outs))


def joint_mi(b: BeliefEngine, c: Candidate, n_asked: int = 1) -> float:
    h0 = b.entropy_joint()
    outs = outcomes(b, c, n_asked)
    return max(0.0, h0 - sum(p * nb.entropy_joint() for _, p, nb in outs)) if outs else 0.0


def misspecified_ig(b: BeliefEngine, c: Candidate, coop: str = "cooperative", n_asked: int = 1) -> float:
    """把 U 写死成合作后的信息增益：现有主动问诊方法的形式化。"""
    mb = b.clone()
    for (z, u) in mb.log_post:
        if u != coop:
            mb.log_post[(z, u)] = -1e9
    return ca_ig(mb, c, n_asked)


def lookahead2(b: BeliefEngine, c: Candidate, others: list[Candidate], n_asked: dict | None = None, top: int = 10) -> tuple[float, dict]:
    """两步前瞻：value2 = IG1(c) + [E_r max_o IG1(o | r,c) − max_o IG1(o)]⁺。others 已按单步价值排序取前 top。"""
    n_asked = n_asked or {}
    ig1 = ca_ig(b, c, n_asked.get((c.slot, c.tag), 0) + 1)
    pool = [o for o in others if o.key() != c.key()][:top]
    if not pool:
        return ig1, {"ig1": ig1, "gain2": 0.0}
    best_now = max(ca_ig(b, o, n_asked.get((o.slot, o.tag), 0) + 1) for o in pool)
    exp_best = 0.0
    for r, p, nb in outcomes(b, c, n_asked.get((c.slot, c.tag), 0) + 1):
        exp_best += p * max(ca_ig(nb, o, n_asked.get((o.slot, o.tag), 0) + 1) for o in pool)
    gain2 = max(0.0, exp_best - best_now)
    return ig1 + gain2, {"ig1": ig1, "gain2": gain2, "best_now": best_now, "exp_best": exp_best}


def channel_discriminability(b: BeliefEngine, c: Candidate) -> float:
    """κ(a) = min_{u≠u'} KL(P(r|u,a) ‖ P(r|u',a))，按当前 P(z) 对 z 边际化。锚定动作定价用。"""
    pz = b.posterior_z(); dists = {}
    for u in b.us:
        d: dict = {}
        for z, w in pz.items():
            for r, q in b.P.dist(z, u, c.slot, c.tag, c.form or "open").items():
                d[r] = d.get(r, 0.0) + w * q
        dists[u] = d
    best = float("inf")
    for u in b.us:
        for v in b.us:
            if u == v:
                continue
            kl = sum(p * math.log(p / max(dists[v].get(r, 1e-12), 1e-12)) for r, p in dists[u].items() if p > 0)
            best = min(best, kl)
    return 0.0 if best == float("inf") else best


# ----------------------------------------------------------------------------- 风险加权（RVOI）
def loss_matrix(zs: list[str], red_flags: list[str], L: dict) -> dict[tuple[str, str], float]:
    """L[(d, z)]：决定 d（诊断名或 'handoff'）在真值 z 下的损失。"""
    out: dict[tuple[str, str], float] = {}
    red = set(red_flags)
    for d in list(zs) + ["handoff"]:
        for z in zs:
            if d == "handoff":
                out[(d, z)] = float(L.get("handoff", 1.5))
            elif d == z:
                out[(d, z)] = 0.0
            elif z in red and d not in red:
                out[(d, z)] = float(L.get("miss_red_flag", 10.0))
            elif d in red and z not in red:
                out[(d, z)] = float(L.get("false_alarm", 2.0))
            elif d in red and z in red:
                out[(d, z)] = float(L.get("wrong_red_red", 1.0))
            else:
                out[(d, z)] = float(L.get("wrong_benign", 1.0))
    return out


def bayes_risk(b: BeliefEngine, Lm: dict) -> tuple[float, str]:
    """当前信念下的最小期望损失及对应决定。"""
    pz = b.posterior_z(); best, best_d = float("inf"), "handoff"
    for d in list(b.zs) + ["handoff"]:
        r = sum(pz[z] * Lm[(d, z)] for z in b.zs)
        if r < best - 1e-12:
            best, best_d = r, d
    return best, best_d


def rvoi(b: BeliefEngine, c: Candidate, Lm: dict, n_asked: int = 1) -> float:
    """风险加权价值：问 c 之后期望损失能降多少（不含问句成本）。"""
    r0, _ = bayes_risk(b, Lm)
    outs = outcomes(b, c, n_asked)
    if not outs:
        return 0.0
    return max(0.0, r0 - sum(p * bayes_risk(nb, Lm)[0] for _, p, nb in outs))


def lookahead2_rvoi(b: BeliefEngine, c: Candidate, others: list[Candidate], Lm: dict, n_asked: dict | None = None, top: int = 10) -> tuple[float, dict]:
    """两步前瞻的 RVOI 版：value2 = V1(c) + [E_r max_o V1(o | r,c) − max_o V1(o)]⁺。"""
    n_asked = n_asked or {}
    v1 = rvoi(b, c, Lm, n_asked.get((c.slot, c.tag), 0) + 1)
    pool = [o for o in others if o.key() != c.key()][:top]
    if not pool:
        return v1, {"v1": v1, "gain2": 0.0}
    best_now = max(rvoi(b, o, Lm, n_asked.get((o.slot, o.tag), 0) + 1) for o in pool)
    exp_best = sum(p * max(rvoi(nb, o, Lm, n_asked.get((o.slot, o.tag), 0) + 1) for o in pool) for _, p, nb in outcomes(b, c, n_asked.get((c.slot, c.tag), 0) + 1))
    gain2 = max(0.0, exp_best - best_now)
    return v1 + gain2, {"v1": v1, "gain2": gain2, "best_now": best_now, "exp_best": exp_best}
