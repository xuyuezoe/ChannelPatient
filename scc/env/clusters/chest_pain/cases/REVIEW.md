# 病例审核记录（cases/REVIEW.md）

审核：自动一致性规则检查已过；人工逐例复核待办。每例一行：来源、关键 atom、改动 / 备注。

| case | dx | 来源 | 年龄/性别/职业 | 部位/性质/严重度 | 触发(True) | 伴随(True) | 锚定 | 备注 |
|---|---|---|---|---|---|---|---|---|
| cp_001 | GERD | handwritten | 54/M/truck driver | epigastric/burning/mild | postprandial,lying_down | acid_taste | prior_severity:record, prior_ecg:record, fam_cad:family | ok |
| cp_002 | stable_angina | handwritten | 63/F/retired teacher | retrosternal/pressure/moderate | exertional | sweating,dyspnea | prior_severity:record, prior_stress:record, fam_cad:family | ok |
| cp_003 | costochondritis | handwritten | 34/M/warehouse worker | precordial/sharp/mild | breathing,movement |  | prior_severity:record, prior_ecg:record, fam_cad:family | ok |
| cp_004 | GERD | ddxplus | 40/F/software engineer | epigastric/sharp/moderate | postprandial,lying_down | acid_taste | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_005 | GERD | ddxplus | 50/F/office clerk | epigastric/burning/moderate | postprandial,lying_down | acid_taste,cough | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_006 | stable_angina | ddxplus | 46/M/office clerk | retrosternal/pressure/moderate | exertional,stress | dyspnea | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_007 | stable_angina | ddxplus | 64/F/nurse | precordial/pressure/severe | exertional,stress |  | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_008 | pericarditis | ddxplus | 60/M/nurse | precordial/sharp/severe | lying_down,breathing |  | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_009 | pericarditis | ddxplus | 37/F/farmer | retrosternal/sharp/severe | lying_down,breathing | dyspnea | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_010 | panic_attack | ddxplus | 26/M/taxi driver | precordial/pressure/mild | stress | sweating,dyspnea,nausea,palpitations,dizziness | prior_severity:record, prior_test:record, fam_cad:family | ok |
| cp_011 | panic_attack | ddxplus | 34/F/taxi driver | precordial/sharp/moderate | stress | dyspnea,nausea,dizziness | prior_severity:record, prior_test:record, fam_cad:family | 惊恐无心悸 |

已知取舍：
- cp_004（GERD）性质为 sharp：DDXPlus 该患者 quality 多选里含 knife/sharp，保留作为非典型 GERD。
- DDXPlus 强度整体偏高；GERD 的 severe 已降为 moderate（builder 规则）。
- 职业按年龄自动生成（<22 学生，≥65 退休）。
- prior_severity（上次就诊记录的严重度）为可核对结构化 atom，是核实事件的载体；prior_test 为文本，不受第一批信道影响。
