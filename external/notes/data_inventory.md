# 已有数据盘点（2026-09-14）

| 数据 | 路径 | 条数 | 说明 |
|---|---|---|---|
| IMCS-21 | data/imcs21/dataset/{train,dev,test}.json | 2472 / 833 / 811 = 4116 | 真实中文儿科问诊，10 诊断，症状级 BIO 与对话意图标注 |
| IMCS-21 粗挖 | data/imcs21_mined.json | 4116 | 每条 {id, dx, guesses(家长猜病名), n_pt, intense}；含猜测 653，含强烈程度词 1506 |
| AgentClinic MedQA-Ext | external/AgentClinic/agentclinic_medqa_extended.jsonl | 214 | OSCE JSON 病例（MIT），pin b6570ed |
| MedDialog-zh | data/meddialog_zh/{train,validation,test}.json | 未计数 | 只做自诊句挖掘补充 |
| DDXPlus test | data/external/ddxplus/ | 134,529 患者 / 223 证据 / 49 病种 | 见 README_local.md |
| CPB-Bench | external/cpb-bench/cpb-bench_data/ | Pos 692 / Neg 352（四个来源文件） | 见 cpb_layout.md；CC BY-NC-ND，只读 |
| PWP | external/PatientsWithPersonality | 代码（MIT），pin 5357a22 | 无数据 |
