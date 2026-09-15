"""用任意 policy(messages -> completion 文本) 跑一段或多段对话，返回 (prompt, completion, reward, info) 序列。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from scc.doctor.rl.env import DoctorRLEnv

PolicyFn = Callable[[list[dict]], str]


def rollout(policy: PolicyFn, env: DoctorRLEnv) -> list[dict]:
    steps = []; prompt = env.reset()
    while prompt is not None:
        completion = policy(prompt)
        next_prompt, r, done, info = env.step(completion)
        steps.append({"prompt": prompt, "completion": completion, "reward": r, "info": info})
        prompt = next_prompt if not done else None
    return steps


def rollout_many(policy: PolicyFn, envs: list[DoctorRLEnv], workers: int = 1) -> list[list[dict]]:
    if workers <= 1:
        return [rollout(policy, e) for e in envs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda e: rollout(policy, e), envs))
