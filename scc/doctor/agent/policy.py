"""选择与停止：红旗刚性、不可辨识移交、argmax；readout 生成。"""
from __future__ import annotations
import math
from scc.doctor.agent.belief import BeliefEngine
from scc.doctor.agent.state import Candidate, AgentConfig
from scc.doctor.agent.value import bayes_risk
from scc.types import ActionKind, DoctorAction, DoctorReadout


def readout(b: BeliefEngine) -> DoctorReadout:
    pz = b.posterior_z(); hmax = math.log(len(pz)) if len(pz) > 1 else 1.0
    conf = 1.0 - b.entropy_z() / hmax
    return DoctorReadout({k: round(v, 4) for k, v in pz.items()}, round(max(0.0, min(1.0, conf)), 4), raw="agent")


def choose_rvoi(cands: list[Candidate], b: BeliefEngine, cfg: AgentConfig, Lm: dict, question_cost: float, n_asked: dict, lang: str = "en") -> tuple[DoctorAction, dict]:
    """风险加权决策：若没有候选能把期望损失降得超过一句的成本，就做当前风险最小的决定（诊断或移交）；否则问价值最高的。
    红旗刚性不再是阈值，而是损失矩阵里"漏红旗代价大"的结果。"""
    risk, d_star = bayes_risk(b, Lm); pz = b.posterior_z()
    def adj(c: Candidate) -> float:
        return c.value * (0.1 if n_asked.get((c.slot, c.tag), 0) >= 2 else 1.0)
    best = max(cands, key=adj) if cands else None
    info = {"risk": risk, "d_star": d_star, "p_top": max(pz.values()), "best_value": best.value if best else 0.0, "question_cost": question_cost}
    if best is None or adj(best) <= question_cost or not cfg.stopping and False:
        if d_star == "handoff":
            info["decision"] = "undetermined"
            txt = "undetermined; handoff for examination" if lang == "en" else "无法辨识；移交体检"
            return DoctorAction(ActionKind.DIAGNOSE, txt, readout=readout(b)), info
        info["decision"] = "diagnose"
        return DoctorAction(ActionKind.DIAGNOSE, d_star, readout=readout(b)), info
    info["decision"] = "ask" if best.kind == ActionKind.ASK else best.kind.value.lower(); info["chosen"] = best.to_dict()
    return DoctorAction(best.kind, best.text, target=best.target, readout=readout(b)), info


def choose(cands: list[Candidate], b: BeliefEngine, cfg: AgentConfig, red_flags: list[str], n_asked: dict, lang: str = "en") -> tuple[DoctorAction, dict]:
    """信息增益目标下的决策（caig）：阈值式停止与红旗刚性。cands 的 value 已由 agent 填好。"""
    pz = b.posterior_z(); top = max(pz, key=pz.get)
    red_mass = sum(pz.get(r, 0.0) for r in red_flags)
    best_val = max((c.value for c in cands), default=0.0)
    info = {"top": top, "p_top": pz[top], "red_mass": red_mass, "best_value": best_val}
    # 下诊断：最大后验够高且红旗已排除（或红旗本身就是 top）
    if pz[top] >= cfg.tau_diagnose and (red_mass <= cfg.tau_red or top in red_flags):
        info["decision"] = "diagnose"
        return DoctorAction(ActionKind.DIAGNOSE, top, readout=readout(b)), info
    # 无信息可得
    if cfg.stopping and best_val < cfg.epsilon_stop:
        if pz[top] >= cfg.tau_stop and (red_mass <= cfg.tau_red or top in red_flags):
            info["decision"] = "diagnose_exhausted"
            return DoctorAction(ActionKind.DIAGNOSE, top, readout=readout(b)), info
        info["decision"] = "undetermined"
        txt = "undetermined; handoff for examination" if lang == "en" else "无法辨识；移交体检"
        return DoctorAction(ActionKind.DIAGNOSE, txt, readout=readout(b)), info
    # 已问 ≥2 次的槽降权（不删，留给前瞻）
    def adj(c: Candidate) -> float:
        n = n_asked.get((c.slot, c.tag), 0)
        return c.value * (0.1 if n >= 2 else 1.0)
    best = max(cands, key=adj)
    info["decision"] = "ask" if best.kind == ActionKind.ASK else best.kind.value.lower()
    info["chosen"] = best.to_dict()
    return DoctorAction(best.kind, best.text, target=best.target, readout=readout(b)), info
