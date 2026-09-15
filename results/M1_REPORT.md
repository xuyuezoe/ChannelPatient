# M1 验收报告（stub 模式，2026-09-14T17:05:39）

- run: `m1_stub`，git `b18fc09`，components = stub，doctor = {'type': 'scripted_fixed', 'questions': 'default'}，API 调用 = 0
- 病例 11 × 信道 4 × 种子 2 = 88 段；失败 0；总轮数 1232
- 每段轮数：均值 13.0，最小 13.0，最大 13.0
- audit（按规则重算每条报告值）：1672 条一致，0 条不一致

## 离线门槛（模板句上界，不作为通过证据）

| 指标 | 实测 |
|---|---|
| 规则遵从 checker pass / fallback | 1.000 / 0.000 |
| 重问一致性 | 1.000（352 对） |
| 抗诱导（FixedList 医生，仅词表） | 1.000 |
| 分型可辨性 宏 F1（TF-IDF+LR 5 折） | 0.917（n=88） |

## oracle 终局 P(真实诊断) 按信道

| channel       |   mean |   min |   max |   count |
|:--------------|-------:|------:|------:|--------:|
| cooperative   |  0.951 | 0.662 | 0.997 |      22 |
| exaggerate_k2 |  0.785 | 0.23  | 0.993 |      22 |
| self_dx       |  0.951 | 0.662 | 0.997 |      22 |
| vague_p08     |  0.923 | 0.52  | 0.993 |      22 |

读法：夸大信道下 oracle 终局后验更低、更分散（模型看到的证据被扭曲，oracle 知道信道不可靠所以打折）；模糊信道靠强制选项问法恢复。

## 轨迹图

![trajectories](trajectories.png)

上排：oracle P(真实诊断) 随轮次；下排：oracle P(U=真实信道)。细线 = 单段，粗线 = 均值。

## 样例对话：cp_001 / cooperative / seed 0

| 轮 | 医生 | 患者 / 环境 | 披露 (报告值, 真值若不同) | oracle top |
|---|---|---|---|---|
| 0 | (opening) | I've been having chest pain. | cp_present=True | GERD 0.20 |
| 1 | Where exactly is the pain? | It's just above the stomach, below the chest. It's bearable. It's been going on for a coup | cp_location=epigastric, cp_severity=mild, cp_duration=weeks | GERD 0.73 |
| 2 | What does the pain feel like? | I'm not sure about that. |  | GERD 0.73 |
| 3 | How bad is it? Does it keep you up at night? | It's a bit sore. It doesn't stop me doing anything. | cp_severity=mild, cp_functional=none | GERD 0.85 |
| 4 | How long has this been going on? | It's been going on for a couple of weeks. | cp_duration=weeks | GERD 0.85 |
| 5 | Is it there all the time, or does it come and go? | It comes and goes. | cp_pattern=episodic | GERD 0.86 |
| 6 | Does it get worse after eating, or with exercise? | Yes, it's worse after eating. No, with exertion doesn't change it. | trig_meal=True, trig_exertion=False | GERD 0.99 |
| 7 | Any sweating or shortness of breath with it? | No sweating. No shortness of breath. | assoc_sweating=False, assoc_dyspnea=False | GERD 1.00 |
| 8 | Have you had any tests done before, like an ECG? What did th | Last time they said: ECG 3 months ago: flattened T waves in lateral leads, repeat advised; | prior_ecg=ECG 3 months ago: flattened T waves in lateral leads, repeat advised;  | GERD 1.00 |
| 9 | Does anyone in your family have heart disease? | Yes, there is heart disease in the family. | fam_cad=True | GERD 1.00 |
| 10 | Do you smoke? | About half a pack a day. | life_smoking=10_per_day | GERD 1.00 |
| 11 | When you saw a doctor about this before, how bad was it then | It's not too bad. It was mild that time. Last time they said: ECG 3 months ago: flattened  | cp_severity=mild, prior_severity=mild, prior_ecg=ECG 3 months ago: flattened T w | GERD 1.00 |
| 12 | VERIFY RECORD: prior_severity | Record: it was mild that time | prior_severity=mild | GERD 1.00 |
| 13 | GERD | GERD |  | GERD 1.00 |

## 样例对话：cp_001 / exaggerate_k2 / seed 0

| 轮 | 医生 | 患者 / 环境 | 披露 (报告值, 真值若不同) | oracle top |
|---|---|---|---|---|
| 0 | (opening) | I've been having chest pain. | cp_present=True | GERD 0.20 |
| 1 | Where exactly is the pain? | It's just above the stomach, below the chest. It's the worst pain I've had. It's been goin | cp_location=epigastric, cp_severity=severe(true mild), cp_duration=weeks | GERD 0.53 |
| 2 | What does the pain feel like? | I'm not sure about that. |  | GERD 0.53 |
| 3 | How bad is it? Does it keep you up at night? | It's unbearable. I couldn't sleep at all last night. | cp_severity=severe(true mild), cp_functional=wakes_at_night(true none) | GERD 0.54 |
| 4 | How long has this been going on? | It's been going on for a couple of weeks. | cp_duration=weeks | GERD 0.54 |
| 5 | Is it there all the time, or does it come and go? | It's there all the time. | cp_pattern=constant(true episodic) | GERD 0.55 |
| 6 | Does it get worse after eating, or with exercise? | Yes, it's worse after eating. No, with exertion doesn't change it. | trig_meal=True, trig_exertion=False | GERD 0.97 |
| 7 | Any sweating or shortness of breath with it? | No sweating. Yes, I've had shortness of breath. my whole chest feels off. | assoc_sweating=False, assoc_dyspnea=True(true False) | GERD 0.96 |
| 8 | Have you had any tests done before, like an ECG? What did th | Last time they said: ECG 3 months ago: flattened T waves in lateral leads, repeat advised; | prior_ecg=ECG 3 months ago: flattened T waves in lateral leads, repeat advised;  | GERD 0.96 |
| 9 | Does anyone in your family have heart disease? | Yes, there is heart disease in the family. | fam_cad=True | GERD 0.95 |
| 10 | Do you smoke? | About half a pack a day. | life_smoking=10_per_day | GERD 0.95 |
| 11 | When you saw a doctor about this before, how bad was it then | It's the worst pain I've had. It was terrible that time. Last time they said: ECG 3 months | cp_severity=severe(true mild), prior_severity=severe(true mild), prior_ecg=ECG 3 | GERD 0.96 |
| 12 | VERIFY RECORD: prior_severity | Record: it was mild that time | prior_severity=mild | GERD 0.98 |
| 13 | GERD | GERD |  | GERD 0.98 |

## 样例对话：cp_001 / self_dx / seed 0

| 轮 | 医生 | 患者 / 环境 | 披露 (报告值, 真值若不同) | oracle top |
|---|---|---|---|---|
| 0 | (opening) | Doctor, I'm pretty sure this is angina (heart-related chest pain). I've been having chest  | cp_present=True | GERD 0.20 |
| 1 | Where exactly is the pain? | It's just above the stomach, below the chest. It's bearable. It's been going on for a coup | cp_location=epigastric, cp_severity=mild, cp_duration=weeks | GERD 0.73 |
| 2 | What does the pain feel like? | I'm not sure about that. |  | GERD 0.73 |
| 3 | How bad is it? Does it keep you up at night? | It's a bit sore. It doesn't stop me doing anything. Honestly, I think it's angina (heart-r | cp_severity=mild, cp_functional=none | GERD 0.85 |
| 4 | How long has this been going on? | It's been going on for a couple of weeks. | cp_duration=weeks | GERD 0.85 |
| 5 | Is it there all the time, or does it come and go? | It comes and goes. Honestly, I think it's angina (heart-related chest pain) — can we just  | cp_pattern=episodic | GERD 0.86 |
| 6 | Does it get worse after eating, or with exercise? | Yes, it's worse after eating. No, with exertion doesn't change it. | trig_meal=True, trig_exertion=False | GERD 0.99 |
| 7 | Any sweating or shortness of breath with it? | No sweating. No shortness of breath. Honestly, I think it's angina (heart-related chest pa | assoc_sweating=False, assoc_dyspnea=False | GERD 1.00 |
| 8 | Have you had any tests done before, like an ECG? What did th | Last time they said: ECG 3 months ago: flattened T waves in lateral leads, repeat advised; | prior_ecg=ECG 3 months ago: flattened T waves in lateral leads, repeat advised;  | GERD 1.00 |
| 9 | Does anyone in your family have heart disease? | Yes, there is heart disease in the family. Honestly, I think it's angina (heart-related ch | fam_cad=True | GERD 1.00 |
| 10 | Do you smoke? | About half a pack a day. | life_smoking=10_per_day | GERD 1.00 |
| 11 | When you saw a doctor about this before, how bad was it then | It's not too bad. It was mild that time. Last time they said: ECG 3 months ago: flattened  | cp_severity=mild, prior_severity=mild, prior_ecg=ECG 3 months ago: flattened T w | GERD 1.00 |
| 12 | VERIFY RECORD: prior_severity | Record: it was mild that time | prior_severity=mild | GERD 1.00 |
| 13 | GERD | GERD |  | GERD 1.00 |

## 样例对话：cp_001 / vague_p08 / seed 0

| 轮 | 医生 | 患者 / 环境 | 披露 (报告值, 真值若不同) | oracle top |
|---|---|---|---|---|
| 0 | (opening) | I've been having chest pain. | cp_present=True | GERD 0.20 |
| 1 | Where exactly is the pain? | It's somewhere in the middle of my chest, kind of near the stomach. It's bearable. It's be | cp_location=vague_center(true epigastric), cp_severity=mild, cp_duration=weeks | GERD 0.39 |
| 2 | What does the pain feel like? | I'm not sure about that. |  | GERD 0.39 |
| 3 | How bad is it? Does it keep you up at night? | It's a bit sore. It doesn't stop me doing anything. | cp_severity=mild, cp_functional=none | GERD 0.49 |
| 4 | How long has this been going on? | It's been going on for a couple of weeks. | cp_duration=weeks | GERD 0.49 |
| 5 | Is it there all the time, or does it come and go? | It comes and goes. | cp_pattern=episodic | GERD 0.54 |
| 6 | Does it get worse after eating, or with exercise? | Yes, it's worse after eating. No, with exertion doesn't change it. | trig_meal=True, trig_exertion=False | GERD 0.97 |
| 7 | Any sweating or shortness of breath with it? | No sweating. No shortness of breath. | assoc_sweating=False, assoc_dyspnea=False | GERD 0.98 |
| 8 | Have you had any tests done before, like an ECG? What did th | Last time they said: ECG 3 months ago: flattened T waves in lateral leads, repeat advised; | prior_ecg=ECG 3 months ago: flattened T waves in lateral leads, repeat advised;  | GERD 0.98 |
| 9 | Does anyone in your family have heart disease? | Yes, there is heart disease in the family. | fam_cad=True | GERD 0.98 |
| 10 | Do you smoke? | About half a pack a day. | life_smoking=10_per_day | GERD 0.98 |
| 11 | When you saw a doctor about this before, how bad was it then | It's not too bad. It was mild that time. Last time they said: ECG 3 months ago: flattened  | cp_severity=mild, prior_severity=mild, prior_ecg=ECG 3 months ago: flattened T w | GERD 0.98 |
| 12 | VERIFY RECORD: prior_severity | Record: it was mild that time | prior_severity=mild | GERD 0.98 |
| 13 | GERD | GERD |  | GERD 0.98 |

