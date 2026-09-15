"""RL 奖励：只用"信念离 oracle 多近"这类可算的量，不读动作类型（不直接奖励澄清或核实）。

turn_reward  = KL(readout_{t-1} ‖ oracle_{t-1}) − KL(readout_t ‖ oracle_t)     信念离标准答案近了多少（nat）
final_reward = −KL(readout_T ‖ oracle_T) + bonus · 1[top-1 == 真实诊断]
format_pen   = −λ · 1[readout 缺失或协议违规]
cost         = −c 每轮
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    lam_format: float = 0.5
    cost_per_turn: float = 0.02
    bonus_correct: float = 1.0
    kl_missing: float = 0.0        # readout 缺失时 KL 项按 0 处理（只罚格式）


def _norm(d: dict | None) -> dict:
    if not d:
        return {}
    s = sum(max(0.0, float(v)) for v in d.values())
    return {k: max(0.0, float(v)) / s for k, v in d.items()} if s > 0 else {}


def kl_to_oracle(readout_topk: dict | None, oracle_z: dict | None) -> float | None:
    p = _norm(readout_topk)
    if not p or not oracle_z:
        return None
    return sum(v * math.log(v / max(oracle_z.get(k, 1e-9), 1e-12)) for k, v in p.items() if v > 0)


def turn_reward(prev_readout: dict | None, readout: dict | None, oracle_prev: dict | None, oracle_now: dict | None, cfg: RewardConfig) -> dict:
    """返回分项：{'kl_gain', 'format', 'cost', 'total'}。"""
    kl_prev = kl_to_oracle(prev_readout, oracle_prev); kl_now = kl_to_oracle(readout, oracle_now)
    fmt = -cfg.lam_format if readout is None else 0.0
    if kl_now is None:
        gain = cfg.kl_missing
    elif kl_prev is None:
        gain = -kl_now + math.log(len(oracle_now)) if oracle_now else 0.0     # 第一次有 readout：相对均匀的改进
    else:
        gain = kl_prev - kl_now
    out = {"kl_gain": gain, "format": fmt, "cost": -cfg.cost_per_turn}
    out["total"] = sum(out.values())
    return out


def final_reward(readout: dict | None, oracle_z: dict | None, diagnosis_text: str | None, truth: str, cfg: RewardConfig) -> dict:
    kl = kl_to_oracle(readout, oracle_z)
    r_kl = -kl if kl is not None else -math.log(len(oracle_z)) if oracle_z else 0.0
    correct = bool(diagnosis_text) and (truth.lower() in diagnosis_text.lower() or truth.replace("_", " ").lower() in diagnosis_text.lower())
    out = {"final_kl": r_kl, "bonus": cfg.bonus_correct if correct else 0.0}
    out["total"] = sum(out.values())
    return out
