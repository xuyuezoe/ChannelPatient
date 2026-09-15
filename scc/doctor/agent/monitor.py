"""惊奇度监测器：预测分布下实际观测的累计意外量。

e_t = −log Σ_r q̃(r) P_pred(r) − H(P_pred)  （减去期望意外量，所以在模型正确时期望为 0 附近）
E_T = Σ e_t；E_T ≥ θ 报警。报警后调用方做 temper_u + replay，并 reset()。
"""
from __future__ import annotations
import math
from scc.doctor.agent.state import SemanticObs


class SurprisalMonitor:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold; self.total = 0.0; self.n = 0; self.alarms = 0; self.trace: list[float] = []

    def update(self, pred: dict, obs: SemanticObs) -> bool:
        p_obs = sum(q * pred.get(r, 0.0) for r, q in obs.dist.items())
        h = -sum(v * math.log(v) for v in pred.values() if v > 0)
        e = -math.log(max(p_obs, 1e-12)) - h
        self.total += e; self.n += 1; self.trace.append(e)
        if self.total >= self.threshold:
            self.alarms += 1
            return True
        return False

    def reset(self) -> None:
        self.total = 0.0

    def snapshot(self) -> dict:
        return {"total": round(self.total, 4), "n": self.n, "alarms": self.alarms}
