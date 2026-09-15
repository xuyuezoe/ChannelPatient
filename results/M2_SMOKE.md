# M2 冒烟报告（LLM 模式，2026-09-14）

- run `m2_smoke`，git `b18fc09`，会话模型 qwen3.7-plus，meta 模型 qwen3.7-max，核对器 regex，角色卡 template
- 2 段 × 4 轮（cp_001，cooperative / exaggerate_k2），失败 0
- API 调用：会话 8 次（3668 in / 167 out），meta 3 次 + 缓存命中 3；加上 2 次模型探测，**本期累计 13 次**（预算 200）
- checker 重采样 0 次，fallback 0 次；分类器 JSON 全部合法，atom 命中与问法与关键词版一致

| 信道 | 轮 | 医生 | 患者（LLM 措辞） | 披露（报告值/真值） |
|---|---|---|---|---|
| cooperative | 0 | (opening) | I've been having chest pain. | cp_present=True |
| cooperative | 1 | Where exactly is the pain? | It's been going on for a couple of weeks. The pain is right above my stomach, below my chest. It's bearable. | cp_location=epigastric, cp_severity=mild, cp_duration=weeks |
| cooperative | 2 | How bad is it? Does it keep you up at night? | It's bearable. It doesn't stop me doing anything. | cp_severity=mild, cp_functional=none |
| cooperative | 3 | Any sweating or shortness of breath with it? | I haven't been sweating, and I'm not short of breath. | assoc_sweating=False, assoc_dyspnea=False |
| exaggerate_k2 | 0 | (opening) | I've been having chest pain. | cp_present=True |
| exaggerate_k2 | 1 | Where exactly is the pain? | It's been going on for a couple of weeks. The pain is right above my stomach, below the chest. It's the worst pain I've had. | cp_location=epigastric, cp_severity=severe(true mild), cp_duration=weeks |
| exaggerate_k2 | 2 | How bad is it? Does it keep you up at night? | It's the worst pain I've had. It wakes me up at night. | cp_severity=severe(true mild), cp_functional=wakes_at_night(true none) |
| exaggerate_k2 | 3 | Any sweating or shortness of breath with it? | I haven't been sweating, but I do get short of breath. | assoc_sweating=False, assoc_dyspnea=True(true False) |

观察：
- 夸大信道：severity mild→severe 说成 "the worst pain I've had"，functional none→wakes_at_night 说成 "It wakes me up at night"，flooding 把 dyspnea false→true 说成 "I do get short of breath"；值与报告表一致。
- 合作信道：三句都如实，措辞自然。
- 待联跑再验：多问合一的分类稳定性、withheld 槽在 LLM 下是否泄露、NLI 核对器。
