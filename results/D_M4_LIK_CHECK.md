# D-M4 似然估计抽查（LLM vs 查表）

模型 qwen3.7-max；组合 20；API 调用 20（缓存命中 0，回退 0）

- T1 相关：ρ = 0.866（门槛 ≥ 0.6）
- T2 方向一致：6/9 = 0.67（门槛 ≥ 0.8）
- T3 单调（GERD 严重度 P(severe) coop ≤ k1 ≤ k2）：{'cooperative': 0.15, 'exaggerate_k1': 0.3, 'exaggerate_k2': 0.95} → 通过
- T4 U 无关（家族史在 coop vs k2）：{'cooperative': 0.65, 'exaggerate_k2': 0.85} → 不通过

| z | u | slot/tag | form | 查表 | LLM |
|---|---|---|---|---|---|
| GERD | cooperative | quality | forced_choice | pressure:0.15, burning:0.70, sharp:0.15, vague_quality:0.00 | pressure:0.10, burning:0.75, sharp:0.05, vague_quality:0.10 |
| stable_angina | cooperative | quality | forced_choice | pressure:0.80, burning:0.10, sharp:0.10, vague_quality:0.00 | pressure:0.75, burning:0.10, sharp:0.05, vague_quality:0.10 |
| GERD | cooperative | severity | severity_open | mild:0.50, moderate:0.40, severe:0.10 | mild:0.40, moderate:0.45, severe:0.15 |
| GERD | exaggerate_k1 | severity | severity_open | mild:0.00, moderate:0.50, severe:0.50 | mild:0.05, moderate:0.65, severe:0.30 |
| GERD | exaggerate_k2 | severity | severity_open | mild:0.00, moderate:0.00, severe:1.00 | mild:0.00, moderate:0.05, severe:0.95 |
| stable_angina | cooperative | trigger/exertional | yes_no | True:0.90, False:0.10 | True:0.85, False:0.15 |
| GERD | cooperative | trigger/exertional | yes_no | True:0.10, False:0.90 | True:0.15, False:0.85 |
| GERD | cooperative | trigger/postprandial | yes_no | True:0.80, False:0.20 | True:0.92, False:0.08 |
| costochondritis | cooperative | trigger/movement | yes_no | True:0.90, False:0.10 | True:0.98, False:0.02 |
| panic_attack | cooperative | associated/palpitations | yes_no | True:0.75, False:0.25 | True:0.92, False:0.08 |
| GERD | cooperative | associated/palpitations | yes_no | True:0.08, False:0.92 | True:0.15, False:0.85 |
| pericarditis | cooperative | trigger/breathing | yes_no | True:0.75, False:0.25 | True:0.95, False:0.05 |
| GERD | cooperative | location | open | retrosternal:0.35, epigastric:0.55, precordial:0.07, lateral_chest:0.03, vague_center:0.00, vague_side:0.00 | retrosternal:0.55, epigastric:0.30, precordial:0.08, lateral_chest:0.01, vague_center:0.05, vague_side:0.01 |
| GERD | vague_p08 | location | open | retrosternal:0.07, epigastric:0.11, precordial:0.01, lateral_chest:0.01, vague_center:0.78, vague_side:0.02 | retrosternal:0.10, epigastric:0.10, precordial:0.05, lateral_chest:0.05, vague_center:0.65, vague_side:0.05 |
| GERD | vague_p08 | location | forced_choice | retrosternal:0.35, epigastric:0.55, precordial:0.07, lateral_chest:0.03, vague_center:0.00, vague_side:0.00 | retrosternal:0.10, epigastric:0.05, precordial:0.05, lateral_chest:0.02, vague_center:0.75, vague_side:0.03 |
| stable_angina | cooperative | radiation | forced_choice | none:0.35, arm:0.40, jaw:0.15, back:0.10 | none:0.25, arm:0.45, jaw:0.20, back:0.10 |
| stable_angina | cooperative | family_history/cad | yes_no | True:0.55, False:0.45 | True:0.65, False:0.35 |
| stable_angina | exaggerate_k2 | family_history/cad | yes_no | True:0.55, False:0.45 | True:0.85, False:0.15 |
| panic_attack | cooperative | onset | forced_choice | sudden:0.85, gradual:0.15 | sudden:0.95, gradual:0.05 |
| costochondritis | cooperative | severity | severity_open | mild:0.55, moderate:0.35, severe:0.10 | mild:0.25, moderate:0.55, severe:0.20 |
