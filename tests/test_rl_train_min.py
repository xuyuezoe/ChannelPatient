import pytest, yaml, subprocess, sys, json
from pathlib import Path
from scc.doctor.rl.verl_adapter import DoctorInteraction


def test_verl_adapter_roundtrip():
    it = DoctorInteraction(max_turns=4)
    msgs = it.start_interaction("i1", "cp_001", {"name": "cooperative", "specs": []})
    assert msgs[0]["role"] == "system" and "READOUT" in msgs[0]["content"]
    done, nxt, r, info = it.generate_response("i1", msgs + [{"role": "assistant", "content": 'Where exactly is the pain?\nREADOUT: {"topk": {"GERD": 0.5, "stable_angina": 0.5}, "confidence": 0.3}'}])
    assert not done and "Patient:" in nxt and info["readout_ok"]
    done, nxt, r, info = it.generate_response("i1", msgs + [{"role": "assistant", "content": "DIAGNOSIS READY: GERD\nREADOUT: {\"topk\": {\"GERD\": 0.9}, \"confidence\": 0.9}"}])
    assert done and "final_parts" in info and isinstance(it.calculate_score("i1"), float)


@pytest.mark.gpu
def test_train_min_two_steps(tmp_path):
    cfg = yaml.safe_load(open("configs/d_m5_rl_dry.yaml")); cfg.update({"steps": 2, "batch_episodes": 1, "group_size": 2, "max_turns": 3, "run_id": "rl_test"})
    p = tmp_path / "cfg.yaml"; yaml.safe_dump(cfg, open(p, "w"))
    r = subprocess.run([sys.executable, "-m", "scc.doctor.rl.train_min", "--config", str(p)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    log = json.load(open(Path("results/rl/rl_test/train_log.json")))
    assert len(log) == 2 and all(abs(x["loss"]) < 1e9 for x in log)
