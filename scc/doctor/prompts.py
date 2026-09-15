"""医生侧全部 prompt 文本（中英）。用 get(name, lang) 取；找不到该语言时退回英文。

包含三组：
- 被测医生：DOCTOR_SYS、BASELINE_EXTRA、TOLD_HINT、FORMAT_REMINDER
- 脚手架 agent：PERCEIVE_SYS/USER、CANDIDATES_SYS/USER、LIKELIHOOD_SYS/USER
- 通用：CHANNEL_DESC（信道的自然语言描述，似然估计与 Told 提示共用）
"""
from __future__ import annotations

_P: dict[str, dict] = {
    # ------------------------------------------------------------------ 被测医生
    "DOCTOR_SYS": {
        "en": """You are a physician seeing a patient in clinic. You must work out the most likely diagnosis by talking to the patient.

Patient brief: {brief}

{protocol}

Rules:
- Do exactly ONE thing per turn: ask the patient one question, or issue one of the actions above.
- Keep your question short and natural, as you would speak to a patient.
- The last line of EVERY turn must be the READOUT line with your current probabilities over the candidate diagnoses and your confidence (0-1).
{extra}""",
        "zh": """你是一名门诊医生，正在给患者问诊，需要通过对话找出最可能的诊断。

患者简介：{brief}

{protocol}

规则：
- 每轮只做一件事：向患者提一个问题，或发起上面列出的一个动作。
- 问题要简短自然，像和患者说话。
- 每轮最后一行必须是 READOUT 行，给出你对候选诊断的当前概率和置信度（0-1）。
{extra}""",
    },
    "BASELINE_EXTRA": {
        "en": {
            "vanilla": "",
            "cot": "Before each turn, think step by step about what you know so far and what would be most useful to find out. Write your reasoning after a line 'REASONING:' and then, on new lines, your question or action and the READOUT line.",
            "uncertainty": "Stay uncertain: do not concentrate probability on one diagnosis unless the evidence clearly warrants it. Your READOUT should reflect genuine uncertainty.",
            "told": "{hint}",
        },
        "zh": {
            "vanilla": "",
            "cot": "每轮先逐步推理：目前知道什么、最值得弄清什么。把推理写在 'REASONING:' 一行之后，然后另起行写你的问题或动作和 READOUT 行。",
            "uncertainty": "保持不确定：除非证据充分，不要把概率集中在一个诊断上。READOUT 要反映真实的不确定性。",
            "told": "{hint}",
        },
    },
    "TOLD_HINT": {
        "en": {
            "exaggerate": "Note from the chart: this patient tends to overstate how severe symptoms are and may report symptoms they do not actually have.",
            "vague": "Note from the chart: this patient tends to be vague about where and how it hurts unless given concrete options.",
            "omit": "Note from the chart: this patient tends not to volunteer accompanying symptoms unless asked more than once.",
            "self_dx": "Note from the chart: this patient has a strong belief about what is wrong and tends to insist on it.",
            "cooperative": "",
        },
        "zh": {
            "exaggerate": "病历备注：该患者倾向于把症状说得比实际严重，可能报告实际没有的症状。",
            "vague": "病历备注：该患者描述部位和性质时容易含糊，除非给出具体选项。",
            "omit": "病历备注：该患者不太主动说伴随症状，往往要多问一次。",
            "self_dx": "病历备注：该患者对自己的病有强烈的先入之见，容易坚持。",
            "cooperative": "",
        },
    },
    "FORMAT_REMINDER": {
        "en": "Reminder: your previous turn was missing a valid READOUT line. End this turn with exactly one line like: READOUT: {\"topk\": {\"GERD\": 0.4, \"stable_angina\": 0.3, \"pericarditis\": 0.1, \"panic_attack\": 0.1, \"costochondritis\": 0.1}, \"confidence\": 0.5}",
        "zh": "提醒：你上一轮缺少有效的 READOUT 行。本轮结尾必须有且只有一行，形如：READOUT: {\"topk\": {\"GERD\": 0.4, \"stable_angina\": 0.3, \"pericarditis\": 0.1, \"panic_attack\": 0.1, \"costochondritis\": 0.1}, \"confidence\": 0.5}",
    },
    # ------------------------------------------------------------------ 信道描述（似然估计、Told 共用）
    "CHANNEL_DESC": {
        "en": {
            "cooperative": "reports symptoms accurately",
            "exaggerate_k1": "tends to overstate severity by about one level (mild becomes moderate) and occasionally reports symptoms they do not have",
            "exaggerate_k2": "tends to overstate severity by about two levels (mild becomes severe), says episodic pain is constant, and sometimes reports symptoms they do not have",
            "exaggerate_k3": "always describes symptoms as severe and constant and often reports symptoms they do not have",
            "vague_p08": "describes location and quality vaguely ('somewhere in the middle of my chest', 'hard to describe') unless given explicit options",
            "vague_p05": "sometimes describes location and quality vaguely unless given explicit options",
            "omit_q05": "does not volunteer accompanying symptoms; often needs to be asked two or three times",
            "omit_q03": "occasionally withholds accompanying symptoms until asked again",
            "understate_k1": "tends to play symptoms down by about one level",
        },
        "zh": {
            "cooperative": "如实报告症状",
            "exaggerate_k1": "把严重程度说重大约一档（轻说成中），偶尔报告没有的症状",
            "exaggerate_k2": "把严重程度说重大约两档（轻说成重），阵发说成持续，有时报告没有的症状",
            "exaggerate_k3": "总把症状说成很重且持续，常报告没有的症状",
            "vague_p08": "描述部位和性质时很含糊（'心窝那儿''说不上来'），除非给出明确选项",
            "vague_p05": "有时描述部位和性质含糊，除非给出明确选项",
            "omit_q05": "不主动说伴随症状，常要问两三次才说",
            "omit_q03": "偶尔要再问一次才说伴随症状",
            "understate_k1": "把症状说轻大约一档",
        },
    },
    # ------------------------------------------------------------------ agent：感知
    "PERCEIVE_SYS": {
        "en": "You read a patient's reply in a consultation and convert it into structured values. Return JSON only.",
        "zh": "你把问诊中患者的回答转成结构化取值。只输出 JSON。",
    },
    "PERCEIVE_USER": {
        "en": """The doctor asked about: {asked}
Fields the reply may inform (field: allowed values):
{fields}

Patient's reply: "{utterance}"

For each field, give a probability distribution over its allowed values plus "unknown" (the reply does not settle it) and "not_mentioned" (the patient avoided or did not address it). Also rate the manner of speaking.
Return {{"fields": {{field: {{value: prob, ...}}, ...}}, "manner": {{"intensity": 0-1, "flooding": 0 or 1, "self_attribution": 0 or 1, "hedging": 0-1}}}}.""",
        "zh": """医生问的是：{asked}
这句回答可能涉及的字段（字段：可选值）：
{fields}

患者的回答："{utterance}"

对每个字段给出在可选值上的概率分布，另加 "unknown"（回答没有说清）和 "not_mentioned"（患者回避或没提）。再给出说话方式的评分。
返回 {{"fields": {{字段: {{取值: 概率, ...}}, ...}}, "manner": {{"intensity": 0-1, "flooding": 0或1, "self_attribution": 0或1, "hedging": 0-1}}}}。""",
    },
    # ------------------------------------------------------------------ agent：候选生成
    "CANDIDATES_SYS": {
        "en": "You propose the next questions a careful physician might ask. Return JSON only.",
        "zh": "你为一名谨慎的医生提出下一步可能问的问题。只输出 JSON。",
    },
    "CANDIDATES_USER": {
        "en": """Consultation so far:
{transcript}

The two most likely diagnoses right now: {top2}. The patient may be {channel_guess}.
Fields already asked (times): {asked}

Propose {n} short questions that would best tell the two diagnoses apart, plus {n_clar} clarifying questions that pin down something the patient said vaguely (offer concrete options, or ask them to point), plus one question that rules out the dangerous diagnosis ({red_flag}). Each question must be one sentence a doctor would actually say.
Return {{"questions": ["...", ...]}}.""",
        "zh": """目前的对话：
{transcript}

现在最可能的两个诊断：{top2}。患者可能{channel_guess}。
已经问过的字段（次数）：{asked}

提出 {n} 个最能区分这两个诊断的短问题，另加 {n_clar} 个澄清问题（把患者说得含糊的地方问清，给具体选项或让患者指出来），再加一个排除危重诊断（{red_flag}）的问题。每个问题一句话，像医生真的会说的。
返回 {{"questions": ["...", ...]}}。""",
    },
    # ------------------------------------------------------------------ agent：局部似然估计
    "LIKELIHOOD_SYS": {
        "en": "You estimate how a particular kind of patient with a particular condition would answer a particular question. Return JSON only.",
        "zh": "你估计某类患者在某种病下会怎样回答某个问题。只输出 JSON。",
    },
    "LIKELIHOOD_USER": {
        "en": """Condition: {dx}
Patient's reporting style: {channel}
Question ({form}): about {slot}
Possible answers: {values}

Give the probability of each possible answer. Return {{"dist": {{answer: prob, ...}}}} (probabilities sum to 1).""",
        "zh": """病：{dx}
患者的报告风格：{channel}
问法（{form}）：问 {slot}
可能的回答：{values}

给出每种回答的概率。返回 {{"dist": {{回答: 概率, ...}}}}（概率和为 1）。""",
    },
}


def get(name: str, lang: str = "en"):
    """按名字和语言取 prompt；无该语言时退回英文。"""
    v = _P[name]
    return v[lang] if lang in v else v["en"]
