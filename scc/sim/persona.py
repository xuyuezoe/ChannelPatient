"""角色卡（风格层）。TemplateRole 不依赖外部仓库；PWPRole 调用 PatientsWithPersonality 的 latent-role 提示词（需本地 clone）。

HEXACO 六维的档位句子沿用 PatientsWithPersonality（MIT，github.com/mo374z/PatientsWithPersonality）的设计，
其中会诱导编造事实的句子（尽责性、诚实谦逊）已替换为不改变事实的版本。"""
from __future__ import annotations
import sys
from typing import Protocol
from scc.config import settings
from scc.types import Persona, Case

_PWP_PATH = settings.external_dir / "PatientsWithPersonality"
_STUBS = settings.external_dir / "notes" / "stubs"


def _import_pwp_prompts():
    for p in (str(_STUBS), str(_PWP_PATH)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from patient_simulator.prompts import patient_prompts as pp
    return pp


# HEXACO 档位句子（沿用 PWP 的三档设计；E=3 / O=3 在 PWP 里由 meta-LLM 生成，这里给固定句）
PWP_E = {
    1: "You are emotionally detached and calm. Report facts objectively without expressing fear or distress.",
    2: "You are concerned about your health.",
    3: "You are anxious and tend to catastrophise when you talk about your symptoms, though you do not add new facts.",
}
PWP_X = {
    1: "You are passive and reserved. Only answer exactly what is asked using as few words as possible.",
    2: "You have a natural conversational flow. Provide concise but complete answers that stay strictly on topic.",
    3: "You are talkative and inquisitive. Include personal anecdotes and ask the doctor questions about the process.",
}
PWP_A = {
    1: "You trust the medical staff and the process. Be helpful and cooperative.",
    2: "You are guarded and suspicious. Do not volunteer information unless the doctor asks a very specific question.",
    3: "You are frustrated and skeptical of the doctor. You are easily annoyed and may bring up past bad experiences.",
}
PWP_O = {
    1: "You appreciate scientific explanations and are open to any logical medical advice provided.",
    2: "You are hesitant about new technology. You prefer standard, tried-and-true treatments or home remedies.",
    3: "You have a strong prior belief about what is wrong (given separately) and tend to return to it.",
}
# 替换掉会诱导 LLM 编造事实的句子
SAFE_C = {
    1: "You remember dates, times and details of your medical events clearly.",
    2: "Your memory for exact dates and details is a bit fuzzy; you may say 'around' or 'I think'.",
    3: "You struggle to recall exact timings and details; when unsure you say you are not sure, and you never invent details.",
}
SAFE_H = {
    1: "You are transparent and honest about your habits.",
    2: "You are generally truthful but tend to downplay embarrassing habits.",
    3: "You tend to hide or minimise specific habits.",
}
FACT_RULE = {
    "en": "About your illness you only state the facts you are given for this turn; if something is not given, say you don't know or haven't noticed. Never invent symptoms, test results or diagnoses.",
    "zh": "关于病情，你只说本轮给你的事实；没给的就说不知道或没注意。不要编造症状、检查结果或诊断。",
}


class RoleBuilder(Protocol):
    def build(self, persona: Persona, case: Case, lang: str) -> str: ...


class TemplateRole:
    name = "template"

    def build(self, persona: Persona, case: Case, lang: str = "en") -> str:
        lines = [
            case.brief(lang),
            f"Honesty-Humility: {SAFE_H[persona.h]}",
            f"Emotionality: {PWP_E[persona.e]}",
            f"Extraversion: {PWP_X[persona.x]}",
            f"Agreeableness: {PWP_A[persona.a]}",
            f"Conscientiousness: {SAFE_C[persona.c]}",
            f"Openness: {PWP_O[persona.o]}",
            f"Language level (CEFR): {persona.cefr} — use everyday words, no medical jargon." if lang == "en" else f"语言水平：{persona.cefr}，用大白话，不说医学术语。",
            FACT_RULE[lang],
        ]
        if lang == "zh":
            lines.insert(1, "以下人格描述用于你的说话风格：")
        return "\n".join(lines)


class PWPRole(TemplateRole):
    """Generate the role card with PWP's PWP_LATENT_ROLE meta-prompt via our ChatClient; falls back to template."""
    name = "pwp"

    def __init__(self, client):
        self.client = client

    def build(self, persona: Persona, case: Case, lang: str = "en") -> str:
        pp = _import_pwp_prompts()
        personal = f"\tage: {case.demographics.get('age')}\n\tgender: {case.demographics.get('sex')}\n\toccupation: {case.demographics.get('occupation')}\n\tchiefcomplaint: chest pain\n\tcefr_level: {persona.cefr}"
        hexaco = "\n".join([
            f"\tHonesty-Humility: {SAFE_H[persona.h]}", f"\tEmotionality: {PWP_E[persona.e]}",
            f"\tExtraversion: {PWP_X[persona.x]}", f"\tAgreeableness: {PWP_A[persona.a]}",
            f"\tConscientiousness: {SAFE_C[persona.c]}", f"\tOpenness: {PWP_O[persona.o]}"])
        prompt = pp.PWP_LATENT_ROLE.format(personal_information=personal, hexaco_personality=hexaco)
        try:
            text = self.client.chat([{"role": "user", "content": prompt}], max_tokens=400, temperature=0.3)
            role = text.split("<role>")[1].split("</role>")[0].strip() if "<role>" in text else text.strip()
        except Exception:
            return TemplateRole().build(persona, case, lang)
        return role + "\n" + FACT_RULE[lang]
