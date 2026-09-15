"""最小 GRPO 训练循环（单卡、HF transformers），只用于验证 RL 流水线；正式训练用 VeRL 在 A100。

流程：每步抽 B 段（病例 × 信道），每段 G 次采样整段对话（多轮，每轮一次生成），
按段内相对优势（组内标准化）做 token 级策略梯度 + KL 到参考模型；stub 患者，零 API。

  python -m scc.doctor.rl.train_min --config configs/d_m5_rl_dry.yaml
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import yaml
import numpy as np
from scc.config import settings
from scc.types import PatientConfig, ChannelSpec, Persona
from scc.env.cases import load_cluster, load_cluster_config
from scc.sim.components import stub_components
from scc.doctor.rl.env import DoctorRLEnv
from scc.doctor.rl.reward import RewardConfig


def build_envs(cfg: dict, cluster_cfg, cases, channels, rng: random.Random, n: int) -> list[DoctorRLEnv]:
    envs = []
    for _ in range(n):
        case = rng.choice(cases); ch = rng.choice(channels)
        pc = PatientConfig(persona=Persona.from_dict(cfg.get("persona", {})), channels=[ChannelSpec.from_dict(s) for s in ch.get("specs", [])], seed=rng.randint(0, 10**6), name=ch["name"])
        envs.append(DoctorRLEnv(case, pc, stub_components(cluster_cfg, pc.seed), cluster_cfg, cfg.get("lang", "en"), int(cfg.get("max_turns", 8)), RewardConfig(**cfg.get("reward", {}))))
    return envs


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); rng = random.Random(int(cfg.get("seed", 0))); torch.manual_seed(int(cfg.get("seed", 0)))
    cluster_cfg = load_cluster_config(cfg.get("cluster", "chest_pain")); cases = load_cluster(cfg.get("cluster", "chest_pain"), cfg.get("cases") if cfg.get("cases") != "all" else None)
    channels = cfg.get("channels", [{"name": "cooperative", "specs": []}])
    model_name = cfg["policy_model"]; device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if cfg.get("lora", False):
        from peft import LoraConfig, get_peft_model
        policy = get_peft_model(policy, LoraConfig(r=int(cfg.get("lora_r", 8)), lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=float(cfg.get("lr", 1e-6)))
    B, G, steps = int(cfg.get("batch_episodes", 2)), int(cfg.get("group_size", 4)), int(cfg.get("steps", 20))
    max_new, beta = int(cfg.get("max_new_tokens", 96)), float(cfg.get("kl_beta", 0.02))
    out_dir = settings.results_dir / "rl" / cfg.get("run_id", "rl_dry"); out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def generate(msgs: list[dict]) -> tuple[str, torch.Tensor, int]:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=int(cfg.get("max_prompt_tokens", 1536))).to(device)
        with torch.no_grad():
            out = policy.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=float(cfg.get("temperature", 1.0)), top_p=1.0, pad_token_id=tok.pad_token_id)
        comp_ids = out[0, enc["input_ids"].shape[1]:]
        return tok.decode(comp_ids, skip_special_tokens=True), out[0], enc["input_ids"].shape[1]

    def logprob_sum(model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        logits = model(full_ids.unsqueeze(0)).logits[0, :-1]
        lp = torch.log_softmax(logits.float(), -1)
        tgt = full_ids[1:]
        tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return tok_lp[prompt_len - 1:].sum()

    t0 = time.time()
    for step in range(steps):
        envs = build_envs(cfg, cluster_cfg, cases, channels, rng, B)
        samples = []          # (group_id, [(full_ids, prompt_len)], return)
        for gi, env in enumerate(envs):
            for g in range(G):
                env_g = build_envs(cfg, cluster_cfg, [env.case], [{"name": env.pc.name, "specs": [{"kind": s.kind, "params": s.params} for s in env.pc.channels]}], random.Random(env.pc.seed), 1)[0]
                prompt = env_g.reset(); turns = []
                while prompt is not None:
                    comp, full, plen = generate(prompt); turns.append((full, plen))
                    prompt, r, done, info = env_g.step(comp)
                samples.append((gi, turns, env_g.episode_return(), None, env_g))
        # 组内标准化优势
        rets = np.array([s[2] for s in samples], dtype=float)
        adv = np.zeros_like(rets)
        for gi in range(B):
            idx = [i for i, s in enumerate(samples) if s[0] == gi]
            r = rets[idx]; adv[idx] = (r - r.mean()) / (r.std() + 1e-6)
        opt.zero_grad(); total_loss = 0.0; n_tok = 0
        policy.train()
        for (gi, turns, ret, _, env_g), a_ in zip(samples, adv):
            for full, plen in turns:
                lp = logprob_sum(policy, full, plen)
                with torch.no_grad():
                    lp_ref = logprob_sum(ref, full, plen)
                loss = -(float(a_) * lp) + beta * (lp - lp_ref)
                (loss / (len(samples) * max(1, len(turns)))).backward(); total_loss += loss.item(); n_tok += int(full.shape[0] - plen)
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step(); policy.eval()
        n_turns = sum(len(s[1]) for s in samples)
        n_ok = sum(1 for s in samples for tr in s[4].turn_rewards if tr["format"] == 0.0)
        rec = {"step": step, "return_mean": float(rets.mean()), "return_std": float(rets.std()), "loss": total_loss / max(1, len(samples)), "tokens": n_tok,
               "readout_ok_rate": n_ok / max(1, n_turns), "turns_mean": n_turns / max(1, len(samples)),
               "elapsed_s": round(time.time() - t0, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
        log.append(rec); print(json.dumps(rec), flush=True)
        json.dump(log, open(out_dir / "train_log.json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [r["step"] for r in log]; ys = [r["return_mean"] for r in log]
        plt.figure(figsize=(5, 3)); plt.plot(xs, ys, marker="o"); plt.xlabel("step"); plt.ylabel("mean episode return"); plt.title(cfg.get("run_id", "rl_dry")); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(out_dir / "return_curve.png", dpi=120)
    except Exception as e:
        print("plot skipped:", e)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
