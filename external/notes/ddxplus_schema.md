# DDXPlus 结构摸底（2026-09-14，test split 134,529 患者）

## 文件
- `release_evidences.json`：223 条证据，键 `E_xx`；字段 `question_en`、`data_type`（B 二值 208 / C 分类 10 / M 多选 5）、`possible-values`、`value_meaning`（只有 9 条证据有，值键 `V_xx`）。
- `release_conditions.json`：49 病种，键为法文名；字段 `cond-name-eng`、`icd10-id`、`symptoms`、`antecedents`、`severity`。
- `release_test_patients.csv`：列 AGE, DIFFERENTIAL_DIAGNOSIS, SEX, PATHOLOGY, EVIDENCES（列表，`E_xx` 或 `E_xx_@_V_yy` / `E_xx_@_数字`）, INITIAL_EVIDENCE。最常见起始证据 E_53（"有和主诉相关的疼痛吗"）25,721 人。

## 疼痛相关证据（我们的 slot 来源）
| 证据 | 类型 | 问题 | 我们的 slot |
|---|---|---|---|
| E_53 | B | 有相关疼痛吗 | chief_complaint |
| E_55 | M | 疼在哪（约 180 个部位值，胸部相关：lower chest V_29, upper chest V_101, breast L/R V_160/V_159, side of the chest L/R V_56/V_55, epigastric V_197, posterior chest wall V_170/171, scapula V_127/128） | location |
| E_54 | M | 疼的性质（burning V_181, heavy V_183, tedious V_154, exhausting V_198, a knife stroke V_179, sharp V_192, violent V_191, a cramp V_182, sickening V_193, haunting V_112, tugging V_180, sensitive V_161, heartbreaking V_71, scary V_196） | quality（映射：burning→burning；heavy/tedious/exhausting→pressure；knife/sharp/violent→sharp；其余忽略） |
| E_56 | C | 强度 0-10 | severity（0-3 mild, 4-6 moderate, 7-10 severe） |
| E_57 | M | 放射到哪 | radiation |
| E_59 | C | 出现多快 0-10 | onset |
| E_58 | C | 定位精确度 0-10 | （不用；可作 vague 信道的现实依据） |
| E_14 | B | 静息时也胸痛 | trigger:rest |
| E_216 | B | 活动时加重 | trigger:movement |
| E_220 | B | 深吸气加重 | trigger:breathing |
| E_33 | B | 前倾缓解 | relief:leaning_forward |
| E_105 | B | 曾有心梗或心绞痛 | medical_history:cad |

## 候选病种（test 集人数；胸部定位比例）
| 病种 | n | 主要部位 | 性质 | 强度档 | 备注 |
|---|---|---|---|---|---|
| GERD | 3543 | epigastric .76, lower chest .48, upper chest .45 | burning .74 | mild .15 / mod .43 / sev .42 | 消化性 |
| Stable angina | 2386 | lower chest .70, breast .65, upper chest .62 | exhausting/tedious/heavy ≈ .7 | .15/.43/.42 | E_105 .81；放射肩/臂/颈 .69 |
| Unstable angina | 2880 | 同上 | 同上 | 0/.44/.56 | E_14 .67 |
| Possible NSTEMI/STEMI | 2911 | 同上 | heavy/sickening/scary .75 | 0/.33/.67 | |
| Pericarditis | 3095 | lower chest .76, breast .71 | knife .82, sharp .75 | 0/.43/.57 | E_220 .70；放射背 .75 |
| Panic attack | 3387 | side of chest .57/.45（其余散在腹部） | cramp .79, sharp .69 | .29/.42/.29 | 起病快 7.5 |
| Spontaneous rib fracture | 778 | side of chest .72, posterior wall .72 | sharp/knife .72 | 0/.21/.79 | E_216 1.0, E_220 1.0（胸壁痛的替身） |
| Pulmonary embolism | 3679 | scapula/side of chest ≈ .72 | knife .77 | .29/.30/.41 | E_220 .73 |
| Spontaneous pneumothorax | 1343 | breast .75 | violent/knife .75 | 0/.19/.81 | E_220 .91 |
| Myocarditis | 1478 | breast/lower chest .58 | knife .76 | .28/.43/.28 | |

**没有的**：肋软骨炎 / 胸壁肌骨痛（最近的是 Spontaneous rib fracture）。

## 簇的决定（写进 cluster.yaml）
ddx_set = [GERD, stable_angina, pericarditis, panic_attack, costochondritis]；红旗 = stable_angina（心源性）。
costochondritis 为 `manual`（似然手写，参考 rib fracture 的 E_216/E_220 模式但强度降到 mild/moderate）。
DDXPlus 的强度整体偏高（GERD 也 42% severe），似然表里 severity 一行按临床常识手工下调并在文件头标注。
