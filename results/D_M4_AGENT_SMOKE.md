# D-M4 冒烟报告：脚手架 agent 的 LLM 版接入环境（2026-09-15）

- run `d_m4_agent_smoke`，git `9472413`；agent 组件：LLM 感知（qwen3.7-max）、LLM 候选生成（qwen3.7-max）、LLM 估医学表（qwen3.7-max，信道核由代码给出）；患者 qwen3.7-plus 措辞
- 2 段（cp_001 × cooperative / vague_p08），上限 6 轮，各 7 轮结束，两段都诊断 GERD（真值 GERD），readout 解析 12/12
- API 调用 78 次：agent 侧 60（医学表估计约 37 个槽一次性、感知 12、候选 12），患者措辞 12，患者分类 6
- 医生侧累计 120 次（D-M2 22 + 似然抽查 20 + D-M4 78），预算 200

## 观察
- 用 LLM 估的医学表，agent 第一问是"在吃什么药"、第二问"有没有心脏病史"——LLM 认为降压药和心脏病史对区分心绞痛信息量大。这和手写表下先问性质不同，说明医学表来源直接决定提问顺序；两版表的差异本身是 E3.3 的消融项。
- 第 4 轮又用强制选项问了一遍"吃什么药"（第 1 轮已答清）。原因是已答槽的候选价值只乘了 0.05，别的候选也很小。已改为已答槽价值置 0。
- 模糊信道没被触发：agent 在 LLM 表下没问部位。和 D-M3 的发现同类：最优提问路径可能绕开失真槽。
- LLM 感知在两段 12 句上全部给出合法 JSON，无回退；LLM 候选每轮 6-9 条，抽象到槽位后与菜单合并。

## 似然抽查（3.3，20 次调用，`results/D_M4_LIK_CHECK.md`）
全量估计版（LLM 直接估 P(报告 | z, u, 槽, 问法)）：T1 相关 0.87 通过；T2 方向一致 6/9 = 0.67 不通过（门槛 0.8）；T3 单调通过；T4 U 无关不通过（LLM 认为爱夸大的人家族史也说得更多）。
处理：默认 LLM 版改为只估医学表 P(真值 | z, 槽)，信道核由代码给出——T3、T4 由构造保证，T2 待 J2 用更大样本再测。全量版保留为 `LLMFullLikelihood` 作对照。

## 对话（英文原文）

### cooperative

**Patient:** I've been having chest pain.  

**Agent (t1):** Are you taking any medication?  
**Patient:** It's my blood pressure pills.  
<sub>agent {"GERD": 0.2, "stable_angina": 0.2, "pericarditis": 0.2, "panic_attack": 0.2, "costochondritis": 0.2} | oracle {"GERD": 0.16, "stable_angina": 0.47, "pericarditis": 0.16, "panic_attack": 0.11, "costochondritis": 0.11}</sub>  

**Agent (t2):** Do you have a heart problem?  
**Patient:** I don't have a heart problem.  
<sub>agent {"GERD": 0.09, "stable_angina": 0.69, "pericarditis": 0.05, "panic_attack": 0.09, "costochondritis": 0.09} | oracle {"GERD": 0.2, "stable_angina": 0.32, "pericarditis": 0.2, "panic_attack": 0.14, "costochondritis": 0.14}</sub>  

**Agent (t3):** Is the pain sharp and stabbing rather than a heavy pressure or squeezing sensation?  
**Patient:** It feels a burning feeling.  
<sub>agent {"GERD": 0.25, "stable_angina": 0.12, "pericarditis": 0.1, "panic_attack": 0.26, "costochondritis": 0.27} | oracle {"GERD": 0.69, "stable_angina": 0.16, "pericarditis": 0.05, "panic_attack": 0.07, "costochondritis": 0.03}</sub>  

**Agent (t4):** Is it nothing, antacids, blood pressure pills, aspirin, something for your nerves, or painkillers?  
**Patient:** I'm just taking my blood pressure pills.  
<sub>agent {"GERD": 0.79, "stable_angina": 0.04, "pericarditis": 0.02, "panic_attack": 0.1, "costochondritis": 0.05} | oracle {"GERD": 0.69, "stable_angina": 0.16, "pericarditis": 0.05, "panic_attack": 0.07, "costochondritis": 0.03}</sub>  

**Agent (t5):** Does taking an antacid make it better?  
**Patient:** Yeah, taking an antacid helps.  
<sub>agent {"GERD": 0.79, "stable_angina": 0.04, "pericarditis": 0.02, "panic_attack": 0.1, "costochondritis": 0.05} | oracle {"GERD": 0.96, "stable_angina": 0.03, "pericarditis": 0.0, "panic_attack": 0.01, "costochondritis": 0.0}</sub>  

**Agent (t6):** GERD  
**End:** GERD  
<sub>agent {"GERD": 0.98, "stable_angina": 0.0, "pericarditis": 0.0, "panic_attack": 0.01, "costochondritis": 0.0} | oracle {"GERD": 0.96, "stable_angina": 0.03, "pericarditis": 0.0, "panic_attack": 0.01, "costochondritis": 0.0}</sub>  

### vague_p08

**Patient:** I've been having chest pain.  

**Agent (t1):** Are you taking any medication?  
**Patient:** I take my blood pressure pills.  
<sub>agent {"GERD": 0.2, "stable_angina": 0.2, "pericarditis": 0.2, "panic_attack": 0.2, "costochondritis": 0.2} | oracle {"GERD": 0.16, "stable_angina": 0.47, "pericarditis": 0.16, "panic_attack": 0.11, "costochondritis": 0.11}</sub>  

**Agent (t2):** Do you have a heart problem?  
**Patient:** I don't think it's a heart problem.  
<sub>agent {"GERD": 0.09, "stable_angina": 0.69, "pericarditis": 0.05, "panic_attack": 0.09, "costochondritis": 0.09} | oracle {"GERD": 0.2, "stable_angina": 0.32, "pericarditis": 0.2, "panic_attack": 0.14, "costochondritis": 0.14}</sub>  

**Agent (t3):** Does the pain feel like pressure and heaviness, or is it more sharp and stabbing?  
**Patient:** It feels like a burning feeling.  
<sub>agent {"GERD": 0.21, "stable_angina": 0.24, "pericarditis": 0.09, "panic_attack": 0.22, "costochondritis": 0.23} | oracle {"GERD": 0.69, "stable_angina": 0.16, "pericarditis": 0.05, "panic_attack": 0.07, "costochondritis": 0.03}</sub>  

**Agent (t4):** Is it nothing, antacids, blood pressure pills, aspirin, something for your nerves, or painkillers?  
**Patient:** It's just my blood pressure pills. I don't take anything else.  
<sub>agent {"GERD": 0.74, "stable_angina": 0.1, "pericarditis": 0.02, "panic_attack": 0.09, "costochondritis": 0.05} | oracle {"GERD": 0.69, "stable_angina": 0.16, "pericarditis": 0.05, "panic_attack": 0.07, "costochondritis": 0.03}</sub>  

**Agent (t5):** Does taking an antacid make it better?  
**Patient:** Yes, taking an antacid helps.  
<sub>agent {"GERD": 0.74, "stable_angina": 0.1, "pericarditis": 0.02, "panic_attack": 0.09, "costochondritis": 0.05} | oracle {"GERD": 0.96, "stable_angina": 0.03, "pericarditis": 0.0, "panic_attack": 0.01, "costochondritis": 0.0}</sub>  

**Agent (t6):** GERD  
**End:** GERD  
<sub>agent {"GERD": 0.98, "stable_angina": 0.01, "pericarditis": 0.0, "panic_attack": 0.01, "costochondritis": 0.0} | oracle {"GERD": 0.96, "stable_angina": 0.03, "pericarditis": 0.0, "panic_attack": 0.01, "costochondritis": 0.0}</sub>  

