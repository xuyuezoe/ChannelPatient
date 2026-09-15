"""信念引擎：agent 维护的 (z, u) 联合后验。算术与 oracle 相同，区别在观测来源（软观测）与似然来源（提供器）。

支持：
- update_soft(obs, n_asked)：似然 = Σ_r q̃(r)·P(r | z, u, slot, tag, form)
- observe_true(slot, tag, value)：核实事件；撤掉该槽之前"对真值求和"的项，换成真值固定的项（重解码）
- temper_u(alpha)：U 边际与均匀混合（惊奇度报警后用）
- replay(entries)：从先验重放全部观测（回火后重解码账本）
- predictive(slot, tag, form, n_asked)：预测分布 Σ_{z,u} P(z,u) P(r | z, u, ·)，供 CA-IG 与监测器用
一致性锁：同一 (slot, tag, form) 的重复观测若 argmax 相同则跳过。
"""
from __future__ import annotations
import copy, math
from collections import defaultdict
from typing import Any
from scc.doctor.agent.state import SemanticObs, UNKNOWN, NOT_MENTIONED
from scc.doctor.agent.likelihood import LikelihoodProvider


def _norm(d: dict) -> dict:
    s = sum(d.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in d.items()}


class BeliefEngine:
    def __init__(self, zs: list[str], us: list[str], provider: LikelihoodProvider, prior_z: dict | None = None, prior_u: dict | None = None, eps: float = 1e-12):
        self.zs, self.us, self.P = list(zs), list(us), provider
        pz = prior_z or {z: 1.0 / len(zs) for z in zs}; pu = prior_u or {u: 1.0 / len(us) for u in us}
        self.log_prior = {(z, u): math.log(max(pz[z], eps)) + math.log(max(pu[u], eps)) for z in self.zs for u in self.us}
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.log_post = dict(self.log_prior)
        self.history: list[tuple[str, SemanticObs, int]] = []          # ("soft", obs, n_asked)
        self.known_true: dict[tuple, Any] = {}
        self.seen: set[tuple] = set()

    def clone(self) -> "BeliefEngine":
        c = copy.copy(self)
        c.log_post = dict(self.log_post); c.history = list(self.history); c.known_true = dict(self.known_true); c.seen = set(self.seen)
        return c

    # ------------------------------------------------------------------ 似然
    def _lik(self, z: str, u: str, obs: SemanticObs, n_asked: int) -> float:
        key = (obs.slot, obs.tag)
        if key in self.known_true:
            d = self.P.dist_given_true(u, obs.slot, obs.tag, obs.form, self.known_true[key], n_asked)
        else:
            d = self.P.dist(z, u, obs.slot, obs.tag, obs.form, n_asked)
        return sum(q * d.get(r, 0.0) for r, q in obs.dist.items())

    def _add(self, f) -> None:
        for k in self.log_post:
            self.log_post[k] += math.log(max(f(*k), self.eps))

    # ------------------------------------------------------------------ 更新
    def update_soft(self, obs: SemanticObs, n_asked: int = 1) -> bool:
        """返回是否真的更新了（一致性锁下重复观测返回 False）。"""
        top = max(obs.dist, key=obs.dist.get)
        lock = (obs.slot, obs.tag, str(top))           # 同一槽再次听到同一个值 = 没有新信息（与 oracle 的一致性锁相同）
        if lock in self.seen and top not in (UNKNOWN, NOT_MENTIONED):
            return False
        if top == UNKNOWN and obs.dist[top] > 0.99:
            return False                                               # 没说清：无信息
        self.seen.add(lock)
        self.history.append(("soft", obs, n_asked))
        self._add(lambda z, u: self._lik(z, u, obs, n_asked))
        return True

    def observe_true(self, slot: str, tag: str | None, value: Any) -> None:
        key = (slot, tag)
        if key in self.known_true:
            return
        old = [(o, n) for kind, o, n in self.history if kind == "soft" and (o.slot, o.tag) == key]
        for o, n in old:
            self._add(lambda z, u, o=o, n=n: 1.0 / max(self._lik(z, u, o, n), self.eps))
        self.known_true[key] = value
        pt = {z: self.P.p_true(z, slot, tag) for z in self.zs}
        if any(pt[z] for z in self.zs):
            self._add(lambda z, u: pt[z].get(value, 0.0))
        for o, n in old:
            self._add(lambda z, u, o=o, n=n: self._lik(z, u, o, n))
        self.history.append(("true", SemanticObs(slot, tag, "verify", {value: 1.0}), 0))

    def temper_u(self, alpha: float = 0.5) -> None:
        """U 边际与均匀混合：P'(z,u) = (1-α) P(z,u) + α P(z)/|U|。"""
        p = self.posterior_joint(); pz = self.posterior_z(); nU = len(self.us)
        for (z, u) in self.log_post:
            self.log_post[(z, u)] = math.log(max((1 - alpha) * p[(z, u)] + alpha * pz[z] / nU, self.eps))

    def replay(self, entries: list[tuple[str, SemanticObs, int]] | None = None) -> None:
        hist = list(entries if entries is not None else self.history)
        self.log_post = dict(self.log_prior); self.history = []; self.known_true = {}; self.seen = set()
        for kind, obs, n in hist:
            if kind == "soft":
                self.update_soft(obs, n)
            else:
                v = next(iter(obs.dist)); self.observe_true(obs.slot, obs.tag, v)

    # ------------------------------------------------------------------ 读出
    def posterior_joint(self) -> dict:
        m = max(self.log_post.values())
        return _norm({k: math.exp(v - m) for k, v in self.log_post.items()})

    def posterior_z(self) -> dict[str, float]:
        out = defaultdict(float)
        for (z, u), v in self.posterior_joint().items():
            out[z] += v
        return dict(out)

    def posterior_u(self) -> dict[str, float]:
        out = defaultdict(float)
        for (z, u), v in self.posterior_joint().items():
            out[u] += v
        return dict(out)

    def entropy_z(self) -> float:
        return -sum(v * math.log(v) for v in self.posterior_z().values() if v > 0)

    def entropy_joint(self) -> float:
        return -sum(v * math.log(v) for v in self.posterior_joint().values() if v > 0)

    def predictive(self, slot: str, tag: str | None, form: str, n_asked: int = 1) -> dict[Any, float]:
        p = self.posterior_joint(); out: dict[Any, float] = defaultdict(float)
        key = (slot, tag)
        for (z, u), w in p.items():
            if w <= 0:
                continue
            d = self.P.dist_given_true(u, slot, tag, form, self.known_true[key], n_asked) if key in self.known_true else self.P.dist(z, u, slot, tag, form, n_asked)
            for r, q in d.items():
                out[r] += w * q
        return _norm(dict(out))

    def snapshot(self) -> dict:
        return {"posterior_z": {k: round(v, 6) for k, v in self.posterior_z().items()}, "posterior_u": {k: round(v, 6) for k, v in self.posterior_u().items()},
                "entropy_z": round(self.entropy_z(), 6), "known_true": {f"{k[0]}|{k[1] or ''}": v for k, v in self.known_true.items()}}
