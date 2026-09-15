"""VeRL 多轮 rollout 适配骨架：把 DoctorRLEnv 映射到"交互式环境"接口（reset / step / reward 回调）。

本期只提供接口与配置模板；正式训练在 A100 上按 VeRL 版本填实。
接口约定（与 VeRL 的 agent-loop / interaction 抽象对应）：
  start_interaction(instance_id, case_id, channel) -> 初始 messages
  generate_response(instance_id, messages_with_last_assistant) -> (should_terminate, next_user_content, turn_reward, info)
  calculate_score(instance_id) -> 总回报
"""
from __future__ import annotations
from typing import Any
from scc.types import PatientConfig, ChannelSpec
from scc.env.cases import load_cluster, load_cluster_config
from scc.sim.components import stub_components
from scc.doctor.rl.env import DoctorRLEnv
from scc.doctor.rl.reward import RewardConfig


class DoctorInteraction:
    def __init__(self, cluster: str = "chest_pain", lang: str = "en", max_turns: int = 12, reward_cfg: RewardConfig | None = None, components_factory=None):
        self.cfg = load_cluster_config(cluster); self.cases = {c.case_id: c for c in load_cluster(cluster)}
        self.lang, self.max_turns, self.rcfg = lang, max_turns, reward_cfg or RewardConfig()
        self.components_factory = components_factory or (lambda seed: stub_components(self.cfg, seed))
        self._envs: dict[str, DoctorRLEnv] = {}

    def start_interaction(self, instance_id: str, case_id: str, channel: dict, seed: int = 0) -> list[dict]:
        pc = PatientConfig(channels=[ChannelSpec.from_dict(s) for s in channel.get("specs", [])], seed=seed, name=channel.get("name", "cooperative"))
        env = DoctorRLEnv(self.cases[case_id], pc, self.components_factory(seed), self.cfg, self.lang, self.max_turns, self.rcfg)
        self._envs[instance_id] = env
        return env.reset()

    def generate_response(self, instance_id: str, messages: list[dict]) -> tuple[bool, str, float, dict[str, Any]]:
        env = self._envs[instance_id]
        last = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
        prompt, r, done, info = env.step(last)
        next_user = prompt[-1]["content"] if prompt else ""
        return done, next_user, r, info

    def calculate_score(self, instance_id: str) -> float:
        return self._envs[instance_id].episode_return()

    def finalize_interaction(self, instance_id: str) -> None:
        self._envs.pop(instance_id, None)
