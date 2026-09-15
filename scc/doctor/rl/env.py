"""DoctorRLEnv：把患者环境包成"文本进、文本出"的 RL 环境。

reset() -> prompt(messages)；step(completion) -> (prompt, reward, done, info)。
prompt 由 ApiDoctor.build_prompt() 生成，保证训练与评测同构；奖励从 TurnLog.oracle 与解析出的 readout 计算。
"""
from __future__ import annotations
from scc.types import Case, PatientConfig, SimComponents, DoctorAction, ActionKind
from scc.env.cases import ClusterConfig
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle
from scc.sim.episode import ConsultationEnv
from scc.doctor.api_doctor import ApiDoctor
from scc.doctor.base import parse_doctor_text
from scc.doctor.rl.reward import RewardConfig, turn_reward, final_reward
from scc.doctor.agent.value import loss_matrix


class _NullClient:
    def chat(self, *a, **k):
        raise RuntimeError("DoctorRLEnv 只用 ApiDoctor.build_prompt，不应调用 client")


class DoctorRLEnv:
    def __init__(self, case: Case, patient_cfg: PatientConfig, components: SimComponents, cfg: ClusterConfig, lang: str = "en",
                 max_turns: int = 12, reward_cfg: RewardConfig | None = None, log_path=None, baseline: str = "vanilla"):
        self.case, self.pc, self.comp, self.cfg, self.lang, self.max_turns = case, patient_cfg, components, cfg, lang, max_turns
        self.rcfg = reward_cfg or RewardConfig()
        self.doctor = ApiDoctor(_NullClient(), baseline, lang, told_channel=patient_cfg.name, name="rl")
        self.env = ConsultationEnv(case, patient_cfg, components, cfg, Oracle(case, cfg, Likelihood(cfg)), max_turns, lang, log_path, doctor_name="rl")
        self.truth = case.labels.primary_dx
        self.Lm = loss_matrix(list(cfg.ddx_set), list(cfg.red_flags), cfg.loss)
        self.prev_readout = None; self.prev_oracle = None; self.ctx = None; self.done = True; self.turn_rewards: list[dict] = []

    def reset(self) -> list[dict]:
        obs, self.ctx = self.env.reset(); self.doctor.reset(self.ctx)
        self.prev_readout = None; self.prev_oracle = self.env.turn_logs[-1].oracle["posterior_z"] if self.env.turn_logs[-1].oracle else None
        self.done = False; self.turn_rewards = []
        return self.doctor.build_prompt(self.ctx)

    def step(self, completion: str) -> tuple[list[dict] | None, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        action = parse_doctor_text(completion or "")
        if action.kind == ActionKind.CHAT and not action.text.strip():
            action = DoctorAction(ActionKind.ASK, "Could you tell me more about the pain?", readout=action.readout)
        if action.readout is None:
            self.doctor.fail_streak += 1
        else:
            self.doctor.fail_streak = 0
        obs, self.ctx, log, done = self.env.step(action)
        oracle_now = log.oracle["posterior_z"] if log.oracle else None
        ro = action.readout.topk if action.readout else None
        r = turn_reward(self.prev_readout, ro, self.prev_oracle, oracle_now, self.rcfg)
        info = {"turn": log.turn, "action": action.kind.value, "readout_ok": action.readout is not None, "reward_parts": dict(r)}
        if done:
            dec = None
            if action.kind == ActionKind.DIAGNOSE:
                dec = next((z for z in self.cfg.ddx_set if z.lower() in action.text.lower() or z.replace("_", " ").lower() in action.text.lower()), "handoff")
            loss = self.Lm[(dec, self.truth)] if dec is not None else self.Lm[("handoff", self.truth)]
            fr = final_reward(ro, oracle_now, action.text if action.kind == ActionKind.DIAGNOSE else None, self.truth, self.rcfg, loss_of_decision=loss)
            r["total"] += fr["total"]; info["final_parts"] = fr
        self.prev_readout, self.prev_oracle = ro, oracle_now
        self.done = done; self.turn_rewards.append(r)
        return (None if done else self.doctor.build_prompt(self.ctx)), float(r["total"]), done, info

    def episode_return(self) -> float:
        return sum(r["total"] for r in self.turn_rewards)
