"""对话风格旋钮：选定要问什么之后，决定怎么说。只改措辞，不改槽位与问法。

style ∈ {concise 精简, detailed 详细（接上患者上一句 + 简短解释为什么问）, warm 共情（先回应感受再问）}。
TemplateStyler：零 LLM，用固定的过渡句。LLMStyler：LLM 改写，再用关键词分类器复核"问的还是同一件事"，不一致则退回原句。
"""
from __future__ import annotations
import random
from typing import Protocol
from scc.env.cases import ClusterConfig
from scc.doctor.agent.state import Candidate
from scc.types import ActionKind

ACK = {
    "en": {"detailed": ["I see.", "Thanks, that helps.", "Okay, noted."], "warm": ["I understand, that sounds uncomfortable.", "I'm sorry you're going through this.", "That must be worrying."]},
    "zh": {"detailed": ["明白了。", "好的，这个有帮助。", "了解。"], "warm": ["我理解，这肯定不好受。", "辛苦了。", "这确实让人担心。"]},
}
WHY = {
    "en": {"location": "It helps me tell the heart from the stomach.", "quality": "The kind of pain points to different causes.", "severity": "I want to know how much it is affecting you.",
           "trigger": "What sets it off is one of the most telling things.", "relief": "What eases it tells me a lot too.", "associated": "I'm checking for anything that would worry me.",
           "radiation": "Where it travels matters for the heart.", "prior_test": "Earlier results help me avoid repeating things.", "prior_severity": "I'd like to compare with last time.",
           "family_history": "Family history changes the odds.", "lifestyle": "This affects the risk.", "medical_history": "Past conditions change what I look for.", "medication": "Some medicines change the picture."},
    "zh": {"location": "这能帮我区分心脏和胃。", "quality": "疼的性质指向不同的原因。", "severity": "我想知道它对你影响多大。", "trigger": "什么诱发它最能说明问题。", "relief": "什么能缓解也很说明问题。",
           "associated": "我在排查需要担心的情况。", "radiation": "串到哪里对判断心脏很重要。", "prior_test": "以前的结果能避免重复检查。", "prior_severity": "我想和上次比一比。",
           "family_history": "家族史会改变可能性。", "lifestyle": "这关系到风险。", "medical_history": "既往病会改变我要查的方向。", "medication": "有些药会影响判断。"},
}


class Styler(Protocol):
    def style(self, cand: Candidate, transcript: list, lang: str) -> str: ...


class TemplateStyler:
    name = "template"

    def __init__(self, style: str = "concise", seed: int = 0):
        self.mode = style; self.rng = random.Random(seed)

    def style(self, cand: Candidate, transcript: list, lang: str = "en") -> str:
        if cand.kind != ActionKind.ASK or self.mode == "concise":
            return cand.text
        ack = self.rng.choice(ACK[lang][self.mode])
        if self.mode == "detailed":
            why = WHY[lang].get(cand.slot or "", "")
            return f"{ack} {cand.text} {why}".strip() if lang == "en" else f"{ack}{cand.text}{why}"
        return f"{ack} {cand.text}" if lang == "en" else f"{ack}{cand.text}"


class LLMStyler(TemplateStyler):
    """LLM 改写；复核槽位与问法不变。每轮 1 次调用。"""
    name = "llm"

    def __init__(self, cfg: ClusterConfig, client, classifier, style: str = "concise", seed: int = 0):
        super().__init__(style, seed)
        self.cfg, self.client, self.clf = cfg, client, classifier
        self.n_calls = 0; self.n_reverted = 0

    def style(self, cand: Candidate, transcript: list, lang: str = "en") -> str:
        if cand.kind != ActionKind.ASK or self.mode == "concise":
            return cand.text
        from scc.doctor.agent.likelihood import pseudo_atom
        last = next((t for r, t in reversed(transcript) if r == "patient"), "")
        guide = {"detailed": "Briefly acknowledge what the patient just said, ask the question, and add one short clause on why you ask.",
                 "warm": "Respond with one short empathetic sentence to what the patient just said, then ask the question."}[self.mode] if lang == "en" else \
                {"detailed": "先简短回应患者刚才的话，再问这个问题，并用半句话说明为什么问。", "warm": "先用一句话共情患者刚才说的，再问这个问题。"}[self.mode]
        msgs = [{"role": "system", "content": ("You rephrase a doctor's next question so it sounds natural in conversation. Keep it to one question about exactly the same thing; do not add other questions or medical conclusions." if lang == "en" else "你把医生的下一个问题改写得自然。只问同一件事，不加别的问题，不下结论。")},
                {"role": "user", "content": (f"Patient just said: \"{last}\"\nQuestion to ask (keep its meaning and any options): \"{cand.text}\"\nStyle: {guide}\nReturn only the doctor's utterance." if lang == "en" else f"患者刚说：\"{last}\"\n要问的问题（保持意思和选项）：\"{cand.text}\"\n风格：{guide}\n只输出医生的话。")}]
        try:
            out = self.client.chat(msgs, max_tokens=120, temperature=0.5); self.n_calls += 1
            text = str(out).strip().strip('"')
        except Exception:
            return super().style(cand, transcript, lang)
        atoms = [pseudo_atom(self.cfg, cand.slot, cand.tag)] if cand.slot else []
        co = self.clf.classify(text, atoms, [], lang) if atoms else None
        same_slot = bool(co and co.atom_ids)
        same_form = (co.question_form == cand.form) if (co and cand.form in ("forced_choice", "point_to", "yes_no")) else True
        if not (same_slot and same_form):
            self.n_reverted += 1
            return super().style(cand, transcript, lang)
        return text
