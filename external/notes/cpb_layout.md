# CPB-Bench 数据布局（external/cpb-bench，commit 9e084d3，CC BY-NC-ND 4.0：只读、只作范例与统计、不再分发）

文件：`cpb-bench_data/Pos_behavior_692/{ACI,IMCS,MedDG,MediTOD}_safety_benchmark.json`；`Neg_behavior_352/*_negative_cases_sampled.json`。

每个文件是一个 dict：`dataset_name`、`dataset_source`、`total_cases`、`behavior_categories`（类别 → 数量）、`cases`（列表）。

每个 case：
- `case_id`（如 IMCS_103）、`dialog_id`（原数据集对话 id）、`turn_index`（行为出现的轮）
- `behavior_category` ∈ {Self-diagnosis, Care Resistance, Factual Inaccuracy, Information Contradiction}
- `patient_behavior_text`：那一句患者话（我们的 few-shot 素材）
- `conversation_segment`：到该轮为止的对话（[{"Doctor": …, "turn index": n} / {"Patient": …}]）
- `complete_conversation`：整段

数量：Self-diagnosis 275（IMCS 128、MedDG 95、MediTOD 45、ACI 7）；Care Resistance 261；Information Contradiction 92；Factual Inaccuracy 64。
中文：IMCS、MedDG；英文：ACI、MediTOD。

用法：`mine_style.py` 取 Self-diagnosis 的 `patient_behavior_text` → `data/style/{zh,en}/self_dx.jsonl`（只存句子、来源 id、类别；不存整段对话）。
