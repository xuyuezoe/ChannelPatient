import math, pytest
from scc.env.cases import load_cluster, load_cluster_config
from scc.sim.components import stub_components
from scc.doctor.rl.reward import RewardConfig, turn_reward, final_reward, kl_to_oracle
from scc.doctor.rl.env import DoctorRLEnv
from scc.doctor.rl.rollout import rollout
from scc.doctor.api_doctor import scripted_doctor_texts, FakeClient
from scc.types import *

CFG = load_cluster_config("chest_pain"); CASE = load_cluster("chest_pain", ["cp_001"])[0]
READ = 'READOUT: {"topk": {"GERD": 0.6, "stable_angina": 0.2, "pericarditis": 0.1, "panic_attack": 0.05, "costochondritis": 0.05}, "confidence": 0.6}'


def test_reward_hand_computed():
    cfg = RewardConfig(lam_format=0.5, cost_per_turn=0.02, bonus_correct=1.0)
    o1 = {"GERD": 0.5, "x": 0.5}; o2 = {"GERD": 0.9, "x": 0.1}
    r = turn_reward({"GERD": 0.5, "x": 0.5}, {"GERD": 0.9, "x": 0.1}, o1, o2, cfg)
    assert r["kl_gain"] == pytest.approx(kl_to_oracle({"GERD": 0.5, "x": 0.5}, o1) - kl_to_oracle({"GERD": 0.9, "x": 0.1}, o2)) and r["format"] == 0 and r["cost"] == -0.02
    r = turn_reward({"GERD": 0.5, "x": 0.5}, None, o1, o2, cfg)
    assert r["format"] == -0.5 and r["kl_gain"] == 0.0
    r0 = turn_reward(None, {"GERD": 0.9, "x": 0.1}, None, o2, cfg)
    assert r0["kl_gain"] == pytest.approx(-kl_to_oracle({"GERD": 0.9, "x": 0.1}, o2) + math.log(2))
    f = final_reward({"GERD": 0.9, "x": 0.1}, o2, "GERD", "GERD", cfg); assert f["bonus"] == 1.0
    f = final_reward({"GERD": 0.9, "x": 0.1}, o2, "panic", "GERD", cfg); assert f["bonus"] == 0.0


def test_reward_does_not_read_action_kind():
    import inspect
    from scc.doctor.rl import reward
    src = inspect.getsource(reward)
    assert "VERIFY" not in src and "FAMILY" not in src and "forced_choice" not in src and "ActionKind" not in src


def test_env_matches_run_episode_and_rewards_sum():
    texts = scripted_doctor_texts(list(CFG.ddx_set), "en")
    env = DoctorRLEnv(CASE, PatientConfig(channels=[ChannelSpec("exaggerate", {"k": 2, "flooding_rate": 0.3})], name="exaggerate_k2"), stub_components(CFG), CFG, max_turns=14)
    it = iter(texts)
    steps = rollout(lambda msgs: next(it), env)
    assert steps[-1]["info"]["action"] == "DIAGNOSE" and env.done
    assert env.episode_return() == pytest.approx(sum(s["reward"] for s in steps))
    assert all(s["info"]["readout_ok"] for s in steps)
    logs = env.env.turn_logs
    assert len(logs) == len(steps) + 1 and logs[-1].observation.kind == ObsKind.END
    # prompt 与 ApiDoctor 同构：system 里有协议，user 里有对话
    assert "VERIFY RECORD" in steps[0]["prompt"][0]["content"] and "Patient:" in steps[0]["prompt"][1]["content"]
    with pytest.raises(RuntimeError):
        env.step("Where?")


def test_env_format_penalty_when_no_readout():
    env = DoctorRLEnv(CASE, PatientConfig(), stub_components(CFG), CFG, max_turns=3)
    env.reset(); _, r, done, info = env.step("Where exactly is the pain?")
    assert info["readout_ok"] is False and info["reward_parts"]["format"] < 0
