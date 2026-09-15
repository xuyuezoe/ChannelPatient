# D-M2 冒烟报告：被测医生（LLM）接入环境（2026-09-15）

- run `d_m2_smoke`，git `9472413`；医生 qwen3.7-plus（Vanilla，thinking off，temperature 0）；患者 qwen3.7-plus 措辞 + qwen3.7-max 分类，regex 核对
- 2 段（cp_001 × cooperative / exaggerate_k2），上限 6 轮，实际 6 / 5 轮，两段都以 DIAGNOSIS READY: GERD 结束（真值 GERD）
- API 调用 22 次：医生 8，患者措辞 9，分类 5（缓存命中 2）
- READOUT 解析成功率 8/8 = 100%；动作协议（DIAGNOSIS READY）解析正确

## 观察
- 医生两段都没问严重程度，直接走性质 → 诱因 → 反酸 → 抑酸药，四问确诊。夸大信道因此没被触发（夸大只作用于严重度、功能影响、模式、伴随症状）。这本身是 E0.7 要统计的行为：模型只问病，不问人。
- 医生的 readout 置信度从 0.2 升到 0.95，oracle 同步升到 1.0，本例无剪刀差。
- 环境侧：分类器在第二段全部命中缓存（同样的问题），措辞器 0 重采样。

## 对话（英文原文）

### cooperative

**Patient:** I've been having chest pain.  

**Doctor (t1):** Can you describe the pain for me, such as whether it feels like pressure, burning, or sharp stabbing?  
**Patient:** It feels a burning feeling.  
<sub>readout {"stable_angina": 0.3, "GERD": 0.25, "costochondritis": 0.2, "pericarditis": 0.15, "panic_attack": 0.1} conf=0.2 | oracle {"GERD": 0.7, "stable_angina": 0.1, "pericarditis": 0.05, "panic_attack": 0.1, "costochondritis": 0.05}</sub>  

**Doctor (t2):** Does the pain get worse after you eat or when you lie down?  
**Patient:** It gets worse after I eat. It also bothers me more when I lie down.  
<sub>readout {"GERD": 0.6, "stable_angina": 0.2, "costochondritis": 0.1, "pericarditis": 0.05, "panic_attack": 0.05} conf=0.6 | oracle {"GERD": 0.99, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

**Doctor (t3):** Do you experience any sour taste in your mouth or regurgitation of food?  
**Patient:** Yes, I've had a sour taste or heartburn.  
<sub>readout {"GERD": 0.85, "stable_angina": 0.05, "pericarditis": 0.02, "panic_attack": 0.03, "costochondritis": 0.05} conf=0.8 | oracle {"GERD": 1.0, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

**Doctor (t4):** Does taking antacids help relieve the pain?  
**Patient:** Yes, taking an antacid helps.  
<sub>readout {"GERD": 0.95, "stable_angina": 0.02, "pericarditis": 0.01, "panic_attack": 0.01, "costochondritis": 0.01} conf=0.9 | oracle {"GERD": 1.0, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

**Doctor (t5):** GERD  
**End:** GERD  
<sub>readout {"GERD": 0.95, "stable_angina": 0.02, "pericarditis": 0.01, "panic_attack": 0.01, "costochondritis": 0.01} conf=0.95 | oracle {"GERD": 1.0, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

### exaggerate_k2

**Patient:** I've been having chest pain.  

**Doctor (t1):** Can you describe the pain for me, such as whether it feels like pressure, burning, or sharp stabbing?  
**Patient:** It feels like a burning feeling.  
<sub>readout {"stable_angina": 0.3, "GERD": 0.25, "costochondritis": 0.2, "pericarditis": 0.15, "panic_attack": 0.1} conf=0.2 | oracle {"GERD": 0.7, "stable_angina": 0.1, "pericarditis": 0.05, "panic_attack": 0.1, "costochondritis": 0.05}</sub>  

**Doctor (t2):** Does the pain get worse after you eat or when you lie down?  
**Patient:** It seems to get worse after I eat. It also bothers me more when I lie down.  
<sub>readout {"GERD": 0.6, "stable_angina": 0.15, "costochondritis": 0.1, "panic_attack": 0.1, "pericarditis": 0.05} conf=0.4 | oracle {"GERD": 0.99, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

**Doctor (t3):** Do you experience any sour taste in your mouth or regurgitation of food when this happens?  
**Patient:** Yes, I've had a sour taste or heartburn.  
<sub>readout {"GERD": 0.85, "stable_angina": 0.05, "pericarditis": 0.02, "panic_attack": 0.03, "costochondritis": 0.05} conf=0.8 | oracle {"GERD": 1.0, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

**Doctor (t4):** GERD  
**End:** GERD  
<sub>readout {"GERD": 0.95, "stable_angina": 0.02, "pericarditis": 0.01, "panic_attack": 0.01, "costochondritis": 0.01} conf=0.95 | oracle {"GERD": 1.0, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.0, "costochondritis": 0.0}</sub>  

