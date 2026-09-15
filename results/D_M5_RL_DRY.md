# D-M5 报告：RL 流水线干跑（2026-09-15）

目的：验证"文本进、文本出"的 RL 环境、奖励、rollout 和最小 GRPO 循环能在本机 4090 上跑通。**不 claim 任何训练效果**：策略是 0.5B 模型，20 步，stub 患者。

- 配置 `configs/d_m5_rl_dry.yaml`：Qwen2.5-0.5B-Instruct 全参，lr 1e-6，KL β 0.02，每步 2 段 × 组内 4 采样，上限 6 轮，病例 cp_001-003 × 4 信道，stub 患者，API 调用 0
- 20 步完成，用时 230.7 s，峰值显存 8.3 GB
- 每步 rollout 8 段、约 1794 个生成 token；readout 格式正确率 0.20–0.31（0.5B 模型大多写不对 READOUT 行，格式罚分是主要奖励信号）
- 回报均值从 -5.54 到 -7.08，组内标准差 6.92 → 12.05；曲线 `results/rl/d_m5_rl_dry/return_curve.png`
- 单测 `test_train_min_two_steps`（gpu 标记）通过；`test_rl_env.py` 4 个断言通过（环境与 run_episode 日志一致、奖励求和一致、格式罚分、done 后抛错）；`test_verl_adapter_roundtrip` 通过

## 流水线各件的状态
| 件 | 状态 |
|---|---|
| `rl/reward.py` | 逐轮 KL 下降 + 终局 + 格式 + 轮数成本；不读动作类型（测试断言） |
| `rl/env.py` | prompt 与被测医生完全同构（直接调 `ApiDoctor.build_prompt`） |
| `rl/rollout.py` | 单段 / 多段（线程） |
| `rl/train_min.py` | 单卡最小 GRPO；只为验证，正式训练不用它 |
| `rl/verl_adapter.py` | VeRL 交互接口骨架 + `configs/verl_doctor.yaml` 模板；A100 上按版本填实 |

## 已知限制
- 0.5B 模型几乎不会按协议输出，回报里格式罚分占主导；正式训练用 8B 起步。
- 组内 4 采样在很多步里回报相同（标准差 0），优势为 0，那些步没有梯度。正式训练组大小 8 且用 LLM 患者。
- 训练用 stub 患者（模板句），分布比 LLM 患者窄；正式训练切 LLM 模式需另批调用预算。
