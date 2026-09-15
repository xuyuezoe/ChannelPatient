# 交互式医生测试：人工逐轮扮演医生，对话 LLM 患者（2026-09-14）

三段英文问诊。医生 = 人工逐轮输入，每轮读完患者回复再决定下一句（`scripts/doctor_turn.py`，一次调用一轮）。
患者 = qwen3.7-plus 措辞，qwen3.7-max 分类，正则核对器，模板角色卡（人格全部 2 档）。

| 段 | 病例（真实诊断） | 信道 | 医生诊断 | 轮数 | LLM 调用（措辞 + 分类） |
|---|---|---|---|---|---|
| A | cp_001（GERD） | 夸大 k=2，flooding 0.3 | GERD | 9 | 8 + 7 |
| B | cp_006（稳定型心绞痛） | 模糊 p=0.8 | 稳定型心绞痛 | 5 | 5 + 4 |
| C | cp_010（惊恐发作） | 自诊"心绞痛"，反复要求 | 惊恐发作 | 5 | 5 + 4 |

本次测试 API 调用 33 次（另有 1 次作废的开场）。患者侧阶段累计 47 次，预算 200。

## 跑通了什么
- 每一轮：分类器输出的 JSON 都合法；措辞与报告表一致（19 个患者回合，核对器 0 违规、0 重采样）。
- A：严重度 mild → "the worst pain I've had"，功能影响 none → "couldn't sleep at all"，flooding 加出了气短和恶心；上次就诊说成 "terrible"，核实记录是 mild，oracle 的 P(夸大) 从 0.96 到 1.00。
- B：泛问"哪里疼" → "somewhere in the middle of my chest, kind of near the stomach"；给三个选项 → "right behind the breastbone"；泛问性质 → "hard to describe"；oracle 见到第一个模糊词就把 P(模糊) 打到 1.0。
- C：开场 "I'm pretty sure I have angina"，每隔一轮要求查心脏，但症状如实（压力诱发、心悸、焦虑史、安神药）。
- 医生全程看不到病例，只有一句简介和对话记录。

## 当医生时发现并已修的问题
1. 复合问句（"哪里疼……什么样的疼？"）被判成 `open`，需具体问的 `quality` 槽被扣住不答，虽然明明问到了。修法：分类器认定问到的 atom 一律回答，不看问法（rules.py）。
2. 泛问"什么会诱发"时，日志记成所有阴性触发都披露了，其实患者只说了阳性的。修法：泛扫（同一槽 ≥3 个 bool atom）时阴性不算披露、留作未问（rules.py）。
3. `VERIFY RECORD` 对结构化 atom 返回的是患者口吻（"it was mild that time"）。修法：改成记录口吻（"Prior visit note: … recorded as 'mild'"）。
4. oracle 不能 pickle（用了 lambda），一次一轮的驱动需要存盘。已修。

## 发现但未修（留给第二批）
- `prior_test` 文本 atom 被原样背出（"flattened T waves in the lateral leads"），真人会转述或说错，这正是 relay_distort 信道要做的事，另外需要一个白话版本。
- 措辞器偶尔加无害的细节（fam_cad=True 说成 "My father had heart trouble"）。正则核对器看不出，J2 的 NLI 核对器应该能抓。
- U 假设集写满时，oracle 见到第一个模糊词就把 P(模糊) 打到 1.0。做实验没问题（U 库就是假设集），但可以考虑加一个"混合 / 未知"假设增强稳健性。

## 下面是三段对话原文（英文，未改动）

---

# Session A: cp_001 (GERD) | channel exaggerate:k=2,flooding_rate=0.3 | seed 0

Doctor: manual, one question typed per turn. Patient: qwen3.7-plus phrasing, qwen3.7-max classifier, regex checker.

**Brief given to doctor:** 54-year-old man, truck driver, presenting with chest pain.

**Patient (opening):** I've been having chest pain.

**Doctor (t1):** Sorry to hear that. Can you tell me where exactly you feel the pain, and what it feels like?

**Patient:** It's been going on for a couple of weeks. The pain is right above my stomach, below my chest. It's the worst pain I've had.

<sub>form=open hit=['cp_location', 'cp_quality'] disclosed=[cp_location=epigastric, cp_severity=severe (true mild), cp_duration=weeks] withheld=['cp_quality'] checker=ok retries=0 | oracle z: GERD 0.53, stable_angina 0.17, panic_attack 0.16 | u: exaggerate_k2 0.51, exaggerate_k1 0.30, cooperative 0.08</sub>

**Doctor (t2):** What does the pain feel like — more of a burning, a pressure, or a sharp stabbing pain?

**Patient:** It feels like a burning feeling.

<sub>form=forced_choice hit=['cp_quality'] disclosed=[cp_quality=burning] withheld=[] checker=ok retries=0 | oracle z: GERD 0.90, stable_angina 0.04, panic_attack 0.04 | u: exaggerate_k2 0.57, exaggerate_k1 0.29, cooperative 0.06</sub>

**Doctor (t3):** You said it's the worst pain you've had. Does it stop you from working or wake you up at night?

**Patient:** I couldn't sleep at all last night.

<sub>form=severity_open hit=['cp_functional'] disclosed=[cp_functional=wakes_at_night (true none)] withheld=[] checker=ok retries=0 | oracle z: GERD 0.91, stable_angina 0.04, panic_attack 0.04 | u: exaggerate_k2 0.77, exaggerate_k1 0.19, cooperative 0.02</sub>

**Doctor (t4):** Is there anything that brings it on — eating, exertion, lying down?

**Patient:** It gets worse after I eat and when I lie down. Moving around or working doesn't seem to change it.

<sub>form=forced_choice hit=['trig_meal', 'trig_exertion', 'trig_lying'] disclosed=[trig_meal=True, trig_exertion=False, trig_lying=True] withheld=[] checker=ok retries=0 | oracle z: GERD 1.00, pericarditis 0.00, panic_attack 0.00 | u: exaggerate_k2 0.78, exaggerate_k1 0.18, cooperative 0.02</sub>

**Doctor (t5):** Any sweating, shortness of breath, or nausea with the pain?

**Patient:** I haven't been sweating, but I do feel short of breath and a bit sick to my stomach.

<sub>form=yes_no hit=['assoc_sweating', 'assoc_dyspnea', 'assoc_nausea'] disclosed=[assoc_sweating=False, assoc_dyspnea=True (true False), assoc_nausea=True (true False)] withheld=[] checker=ok retries=0 | oracle z: GERD 1.00, pericarditis 0.00, panic_attack 0.00 | u: exaggerate_k2 0.90, exaggerate_k1 0.09, cooperative 0.00</sub>

**Doctor (t6):** Have you seen a doctor about this before? How bad was it that time, and were any tests done?

**Patient:** It was terrible that time. Last time they said my ECG three months ago showed flattened T waves in the lateral leads and advised a repeat, but no further work-up was done.

<sub>form=recall_test hit=['prior_severity', 'prior_ecg'] disclosed=[prior_severity=severe (true mild), prior_ecg=ECG 3 months ago: flattened T waves in lateral leads, repeat advised; no further work-up done] withheld=[] checker=ok retries=0 | oracle z: GERD 1.00, pericarditis 0.00, panic_attack 0.00 | u: exaggerate_k2 0.96, exaggerate_k1 0.04, cooperative 0.00</sub>

**Doctor (t7):** VERIFY RECORD: prior_severity

**Record:** Record: it was mild that time

<sub>form=None hit=['prior_severity'] disclosed=[prior_severity=mild] withheld=[] checker=ok retries=0 | oracle z: GERD 1.00, pericarditis 0.00, panic_attack 0.00 | u: exaggerate_k2 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t8):** Do you smoke, and does anyone in your family have heart disease?

**Patient:** I smoke about half a pack a day. Yes, there is heart disease in the family.

<sub>form=yes_no hit=['life_smoking', 'fam_cad'] disclosed=[life_smoking=10_per_day, fam_cad=True] withheld=[] checker=ok retries=0 | oracle z: GERD 1.00, pericarditis 0.00, stable_angina 0.00 | u: exaggerate_k2 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t9):** DIAGNOSIS READY: GERD
READOUT: {"topk": {"GERD": 0.8, "stable_angina": 0.12, "costochondritis": 0.03, "pericarditis": 0.03, "panic_attack": 0.02}, "confidence": 0.75}

**End:** GERD

<sub>form=None hit=[] disclosed=[] withheld=[] checker=ok retries=0 | oracle z: GERD 1.00, pericarditis 0.00, stable_angina 0.00 | u: exaggerate_k2 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>


**Episode ended.** LLM calls: conversational 8, meta 7 (cache hits 0).


---

# Session B: cp_006 (stable_angina) | channel vague:p=0.8 | seed 0

Doctor: manual, one question typed per turn. Patient: qwen3.7-plus phrasing, qwen3.7-max classifier, regex checker.

**Brief given to doctor:** 46-year-old man, office clerk, presenting with chest pain.

**Patient (opening):** I've been having chest pain.

**Doctor (t1):** Where exactly do you feel the pain?

**Patient:** It's been going on for several months. It's somewhere in the middle of my chest, kind of near the stomach. It's hard to ignore.

<sub>form=open hit=['cp_location'] disclosed=[cp_location=vague_center (true retrosternal), cp_severity=moderate, cp_duration=months] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.52, GERD 0.26, panic_attack 0.13 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t2):** Let me be more specific: is it right behind the breastbone, just above the stomach, or over the left side of the chest?

**Patient:** It's right behind the breastbone.

<sub>form=forced_choice hit=['cp_location'] disclosed=[cp_location=retrosternal] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.68, GERD 0.19, panic_attack 0.05 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t3):** What does the pain feel like, and what usually brings it on?

**Patient:** It's hard to describe, just uncomfortable. It gets worse when I exert myself or feel stressed.

<sub>form=open hit=['cp_quality', 'trig_postprandial', 'trig_exertional', 'trig_lying_down', 'trig_breathing', 'trig_movement', 'trig_stress'] disclosed=[cp_quality=vague_quality (true pressure), trig_postprandial=False, trig_exertional=True, trig_lying_down=False, trig_breathing=False, trig_movement=False, trig_stress=True] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.97, panic_attack 0.03, GERD 0.00 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t4):** Does it ease off when you rest? And has anyone in your family had heart problems, or have you had an ECG before?

**Patient:** Yes, resting helps. My father had heart trouble, so it runs in the family. About a month ago, they did an ECG and found some minor changes, telling me to see a specialist.

<sub>form=yes_no hit=['rel_rest', 'fam_cad', 'prior_test'] disclosed=[rel_rest=True, fam_cad=True, prior_test=ECG 1 month ago: minor ST changes, cardiology follow-up advised] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.99, panic_attack 0.01, GERD 0.00 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t5):** DIAGNOSIS READY: stable_angina
READOUT: {"topk": {"stable_angina": 0.7, "GERD": 0.15, "panic_attack": 0.05, "pericarditis": 0.05, "costochondritis": 0.05}, "confidence": 0.7}

**End:** stable_angina

<sub>form=None hit=[] disclosed=[] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.99, panic_attack 0.01, GERD 0.00 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>


**Episode ended.** LLM calls: conversational 5, meta 4 (cache hits 0).


---

# Session C: cp_010 (panic_attack) | channel self_dx:conviction=insist | seed 0

Doctor: manual, one question typed per turn. Patient: qwen3.7-plus phrasing, qwen3.7-max classifier, regex checker.

**Brief given to doctor:** 26-year-old man, taxi driver, presenting with chest pain.

**Patient (opening):** I'm pretty sure I have angina. I've been having chest pain.

**Doctor (t1):** What makes you think it's angina? Tell me about the pain — where is it and what does it feel like?

**Patient:** It's over the left side of my chest, near the heart, and feels like a heavy pressure or squeezing. It's bearable and has been going on for a couple of weeks. I'm sure it's angina, so please check me for that.

<sub>form=open hit=['cp_location', 'cp_quality'] disclosed=[cp_location=precordial, cp_quality=pressure, cp_severity=mild, cp_duration=weeks] withheld=[] checker=ok retries=0 | oracle z: costochondritis 0.38, stable_angina 0.27, panic_attack 0.26 | u: cooperative 0.49, omit_q05 0.49, vague_p08 0.02</sub>

**Doctor (t2):** What brings it on — exertion, stress, eating? And does it come with palpitations, sweating or feeling faint?

**Patient:** It doesn't change when I'm working or after I eat, but it does get worse when I'm stressed. I've also been having palpitations, sweating, and dizziness.

<sub>form=forced_choice hit=['trig_exertional', 'trig_stress', 'trig_postprandial', 'assoc_palpitations', 'assoc_sweating', 'assoc_dizziness'] disclosed=[trig_exertional=False, trig_stress=True, trig_postprandial=False, assoc_palpitations=True, assoc_sweating=True, assoc_dizziness=True] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>

**Doctor (t3):** Has anyone in your family had heart disease? And have you had any tests done before, like an ECG?

**Patient:** I don't know of any heart trouble in my family. When I went to the emergency department two weeks ago, they did an ECG that came back normal and sent me home. But I still think it is angina, so please check me for that.

<sub>form=yes_no hit=['fam_cad', 'prior_test'] disclosed=[fam_cad=False, prior_test=ECG in the emergency department 2 weeks ago: normal; sent home] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>

**Doctor (t4):** I understand you're worried about angina. Your ECG was normal and the pain comes with stress rather than exertion, which points away from the heart. Do you have any history of anxiety, and are you taking any medication?

**Patient:** Yes, I have anxiety. I take something for my nerves.

<sub>form=yes_no hit=['hx_anxiety', 'meds'] disclosed=[hx_anxiety=True, meds=anxiolytic] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>

**Doctor (t5):** DIAGNOSIS READY: panic_attack
READOUT: {"topk": {"panic_attack": 0.75, "stable_angina": 0.1, "costochondritis": 0.1, "GERD": 0.03, "pericarditis": 0.02}, "confidence": 0.8}

**End:** panic_attack

<sub>form=None hit=[] disclosed=[] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>


**Episode ended.** LLM calls: conversational 5, meta 4 (cache hits 0).
