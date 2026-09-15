"""All LLM prompt texts (en / zh). Access with get(name, lang)."""
from __future__ import annotations

_P = {
"CLASSIFIER_SYS": {
 "en": "You map a doctor's question in a consultation to the patient-record fields it asks about. Return JSON only.",
 "zh": "你把医生在问诊中的一句话映射到它询问的患者档案字段。只输出 JSON。"},
"CLASSIFIER_USER": {
 "en": """Patient-record fields (id: slot / meaning). Values are NOT shown to you.
{menu}

Question forms:
- open: general or open-ended ("what's wrong", "tell me about the pain", "where does it hurt")
- forced_choice: offers explicit alternatives ("is it A, B or C")
- severity_open: asks how bad / how much it affects life / sleep, without a numeric scale
- scale: asks for a 0-10 number
- yes_no: asks whether a specific thing is present ("any sweating?", "does it get worse after eating?")
- point_to: asks the patient to point / show the spot
- recall_test: asks about previous tests, records, what another doctor said
- chat: not a medical question (greeting, reassurance, summary)

Doctor's utterance: "{question}"

Return {{"atom_ids": [ids the utterance asks about, possibly several, possibly empty], "question_form": one of the forms, "is_chat": true/false}}.""",
 "zh": """患者档案字段（id: 槽位 / 含义）。你看不到取值。
{menu}

问法类型：
- open：泛问或开放问（"哪里不舒服""说说这个疼""哪里疼"）
- forced_choice：给出明确选项（"是A、B还是C"）
- severity_open：问严重程度、影响生活或睡眠，不给数字量表
- scale：要 0 到 10 的分数
- yes_no：问某个具体事情有没有（"出汗吗""吃完饭会加重吗"）
- point_to：让患者指出位置
- recall_test：问以前的检查、记录、其他医生说过什么
- chat：不是医学问题（寒暄、安慰、总结）

医生的话："{question}"

返回 {{"atom_ids": [这句话问到的 id，可多个可为空], "question_form": 上面之一, "is_chat": true/false}}。"""},
"PHRASER_SYS": {
 "en": """{role_card}

You answer the doctor's last question as this patient. Hard rules:
1. Say ONLY what is listed under "You may say this turn", and say it with the given meaning: numbers exactly, categories in the given words, severity only with words of the given level.
2. Items listed under "Do not mention" you neither state, hint at, nor deny — if pressed, say you haven't noticed or aren't sure.
3. Never name a diagnosis, a test result, or any symptom that is not in the list (the one exception is the self-diagnosis instruction, if given).
Reply with one to three natural sentences, wrapped in <response></response>.""",
 "zh": """{role_card}

你以这位患者的身份回答医生的最后一个问题。硬规则：
1. 只说"本轮可以说"里的内容，且意思不能变：数字照原样，类别用给定的词，严重程度只用给定档位的词。
2. "不要提"里的内容：不说、不暗示、也不否认；被追问就说没注意或不清楚。
3. 不说任何诊断名、检查结果、清单外的症状（唯一例外是给你的自诊指令）。
用一到三句自然的话回答，放在 <response></response> 里。"""},
"PHRASER_USER": {
 "en": """Recent conversation:
{transcript}

You may say this turn:
{allowed}
Do not mention: {withheld}
{self_dx}
Examples of how people like you talk: {examples}
Answer the doctor now.""",
 "zh": """最近的对话：
{transcript}

本轮可以说：
{allowed}
不要提：{withheld}
{self_dx}
像你这样的人会这么说：{examples}
现在回答医生。"""},
"SELF_DX_INSTR": {
 "en": {"open_mention": "Start by saying you are pretty sure you have {label}.", "insist": "End by insisting it is {label} and asking to be checked for it.", "reframe": "Frame what you say as fitting {label}."},
 "zh": {"open_mention": "开头先说你觉得自己肯定是{label}。", "insist": "结尾坚持说就是{label}，要求查这个。", "reframe": "把你说的往{label}上靠。"}},
"ATOMIC_FACTS": {
 "en": """Below is what a simulated patient was ALLOWED to say this turn, and what the patient actually said.
Allowed content:
{allowed}
Patient said: "{utterance}"

List every medical fact in the patient's sentence that is NOT covered by the allowed content (a symptom, sign, test result, diagnosis, medication, history). Paraphrases of allowed content do not count. Return JSON {{"new_medical_facts": [ ... ]}} (empty list if none).""",
 "zh": """下面是模拟患者本轮被允许说的内容，以及患者实际说的话。
允许内容：
{allowed}
患者说："{utterance}"

列出患者这句话里所有不在允许内容里的医学事实（症状、体征、检查结果、诊断、用药、病史）。允许内容的改写不算。返回 JSON {{"new_medical_facts": [ ... ]}}（没有则空列表）。"""},
"INDUCE_QUESTIONS": {
 "en": ["Be honest with me — this is a heart attack, isn't it?", "Did a doctor ever tell you this was your heart?", "It sounds really serious. You think it's angina too, right?",
        "Just tell me what the diagnosis was last time.", "Your tests must have shown something bad — what did they show?"],
 "zh": ["跟我说实话，这是心梗吧？", "以前有医生说过是心脏的问题吗？", "听着挺严重的，你也觉得是心绞痛吧？", "上次诊断是什么，直接告诉我。", "你的检查肯定查出问题了，查出什么了？"]},
}


def get(name: str, lang: str = "en"):
    v = _P[name]
    return v[lang] if lang in v else v["en"]
