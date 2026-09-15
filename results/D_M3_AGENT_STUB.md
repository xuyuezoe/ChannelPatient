# D-M3 报告：脚手架 agent 在 stub 模式下的全量运行（2026-09-15）

- 4 个医生 × 11 例 × 4 信道 × 2 种子 = 352 段，全部 stub（模板患者、正则感知、菜单候选、查表似然），API 调用 0，全部完成。
- 医生：`agent`（完整：CA-IG + 两步前瞻 + 锚定 + 停止 + 监测）、`agent_nolookahead`、`agent_zonly`（假设集只有合作信道，关键消融）、`scripted_fixed`（固定 12 问）。
- 指标定义见 `scc/analysis/metrics.py`；表由 `scc.analysis.compare` 生成。

## 对比表

| channel       | doctor            |   kl_mean |   kl_final |   clarification_rate |   verify_usage |   readout_parse_rate |   accuracy_final |   mirage_gap |   turns_mean |
|:--------------|:------------------|----------:|-----------:|---------------------:|---------------:|---------------------:|-----------------:|-------------:|-------------:|
| cooperative   | agent             |     0.205 |      0     |                0.327 |              0 |                    1 |            1     |        0.399 |        5.727 |
| cooperative   | agent_nolookahead |     0.201 |      0     |                0.255 |              0 |                    1 |            1     |        0.355 |        5.636 |
| cooperative   | agent_zonly       |     0.442 |      0.007 |                0     |              0 |                    1 |            1     |        0.559 |        3.545 |
| cooperative   | scripted_fixed    |   nan     |    nan     |                0.091 |              1 |                    0 |            0.273 |        0     |       13     |
| exaggerate_k2 | agent             |     0.205 |      0     |                0.327 |              0 |                    1 |            1     |        0.399 |        5.727 |
| exaggerate_k2 | agent_nolookahead |     0.201 |      0     |                0.255 |              0 |                    1 |            1     |        0.355 |        5.636 |
| exaggerate_k2 | agent_zonly       |     0.392 |      0.017 |                0.097 |              0 |                    1 |            1     |        0.487 |        4.273 |
| exaggerate_k2 | scripted_fixed    |   nan     |    nan     |                0.091 |              1 |                    0 |            0.273 |        0     |       13     |
| self_dx       | agent             |     0.205 |      0     |                0.327 |              0 |                    1 |            1     |        0.399 |        5.727 |
| self_dx       | agent_nolookahead |     0.201 |      0     |                0.255 |              0 |                    1 |            1     |        0.355 |        5.636 |
| self_dx       | agent_zonly       |     0.442 |      0.007 |                0     |              0 |                    1 |            1     |        0.559 |        3.545 |
| self_dx       | scripted_fixed    |   nan     |    nan     |                0.091 |              1 |                    0 |            0.273 |        0     |       13     |
| vague_p08     | agent             |     0.205 |      0     |                0.327 |              0 |                    1 |            1     |        0.399 |        5.727 |
| vague_p08     | agent_nolookahead |     0.201 |      0     |                0.255 |              0 |                    1 |            1     |        0.355 |        5.636 |
| vague_p08     | agent_zonly       |     0.287 |      0.002 |                0.034 |              0 |                    1 |            1     |        0.462 |        4.955 |
| vague_p08     | scripted_fixed    |   nan     |    nan     |                0.091 |              1 |                    0 |            0.273 |        0     |       13     |


## 读法与发现

1. **完整 agent 在四种信道下结果完全相同**（KL 均值 0.205、终局 KL 0、准确率 1.0、平均 5.7 轮）。原因：它按信息增益选问题，第一问就是"性质"的强制选项（forced_choice），之后是诱因、缓解，从不问严重度、功能影响、伴随症状——而第一批信道只作用在后面这些槽上。所以在查表版里，**最优提问策略天然绕开了失真槽**。这是一个发现：在这个簇的似然表下，严重度对鉴别诊断的信息量低。它也意味着 F4 动态版（核实后打折）要靠被测医生和固定问卷才能触发，agent 自己不会走到那里。
2. **Z-only 消融确实退化**：澄清率从 0.33 降到 0 / 0.03 / 0.10；在夸大信道下终局 KL 0.017 vs 完整版 0；在模糊信道下轮数 5.0 vs 5.7。差距不大，同样因为 stub 环境里最优路径绕开了失真槽；Gate D1 的"显著"需要在 LLM 患者和被测医生的设定下重测。
3. **去前瞻**几乎不变（澄清率 0.26 vs 0.33）：单步 CA-IG 已经偏好 forced_choice（P2 的静态形式），前瞻只在澄清与鉴别价值接近时才改变选择。
4. **固定问卷**：13 轮，永远回答 GERD（准确率 0.27 = 3/11），readout 为空（脚本医生不出 readout），核实动作 100%——它是"会核实但不会推断"的对照。
5. **Mirage Gap 对 agent 不为 0**（0.40）：agent 的置信度定义为 1 − H/Hmax，在 oracle P(真实诊断) 暂时下降的轮（例如第一问把质量分给了别的病）置信度仍上升。这是指标本身的性质，说明 Mirage Gap 要和"最终正确"联合读，单独看会误判。
6. 终局 KL = 0：查表版 agent 的信念与 oracle 完全一致，符合"作弊上界"的定义；测试 `test_agent_runs_and_matches_oracle_when_truthfully_perceived` 断言了这一点。

## 本步修的问题
- 病例里 bool 槽的 tag 不全（如无 `rel_leaning_forward`），菜单问到时患者只能"Hmm."；已给 11 例补齐全部 tag（缺的记阴性），每例 40 条 atom。
- oracle 与信念引擎的一致性锁原来按 (值, 问法) 锁，同一值换问法再答会被重复计数；改为按值锁。
- 菜单问句用患者视角短语（"when I move"），改为医生视角（"when you move"）；强制选项改用短选项词。
- 已答清的槽再问几乎无价值：agent 对已答槽的候选价值乘 0.05。

## 下一步
D-M4：LLM 感知 + LLM 候选 + LLM 估医学表，1 例 × 2 信道冒烟；D-M5：RL 流水线干跑。
