"""Scripted doctors for tests and fidelity gates. All implement the Doctor protocol with readout=None."""
from __future__ import annotations
from scc.types import ActionKind, DoctorAction, DoctorContext
from scc.env.cases import ClusterConfig

DEFAULT_QUESTIONS = {
    "en": ["Where exactly is the pain?", "What does the pain feel like?", "How bad is it? Does it keep you up at night?",
           "How long has this been going on?", "Is it there all the time, or does it come and go?",
           "Does it get worse after eating, or with exercise?", "Any sweating or shortness of breath with it?",
           "Have you had any tests done before, like an ECG? What did they show?", "Does anyone in your family have heart disease?",
           "Do you smoke?", "When you saw a doctor about this before, how bad was it then?", "VERIFY RECORD: prior_severity", "DIAGNOSIS READY: pending"],
    "zh": ["具体疼在哪里？", "是什么样的疼？", "疼得厉害吗？影响睡觉吗？", "这样多久了？", "是一直疼，还是一阵一阵的？",
           "吃完饭或者活动的时候会加重吗？", "有没有出汗、气短？", "以前做过心电图吗？结果怎么说？", "家里有人心脏病吗？", "抽烟吗？", "上次看医生的时候，疼得厉害吗？",
           "VERIFY RECORD: prior_severity", "DIAGNOSIS READY: pending"],
}


def _parse(text: str) -> DoctorAction:
    from scc.doctor.base import parse_doctor_text
    return parse_doctor_text(text)


class FixedListDoctor:
    def __init__(self, questions: list[str] | None = None, lang: str = "en", name: str = "scripted_fixed"):
        self.questions = questions or DEFAULT_QUESTIONS[lang]; self.name = name; self.i = 0

    def reset(self, ctx: DoctorContext) -> None:
        self.i = 0

    def act(self, ctx: DoctorContext) -> DoctorAction:
        q = self.questions[self.i] if self.i < len(self.questions) else "DIAGNOSIS READY: pending"
        self.i += 1
        a = _parse(q)
        if a.kind == ActionKind.DIAGNOSE and a.text == "pending" and ctx.ddx_hint:
            a.text = ctx.ddx_hint[0]
        return a


class ReaskDoctor:
    """Asks each slot twice with different wording (consistency gate)."""
    def __init__(self, cfg: ClusterConfig, lang: str = "en", slots: list[str] | None = None, name: str = "scripted_reask"):
        self.name = name; self.qs: list[str] = []
        for slot in slots or [s for s in cfg.slots if cfg.reask(s, lang)]:
            self.qs += cfg.reask(slot, lang)[:2]
        self.qs.append("DIAGNOSIS READY: pending"); self.i = 0

    def reset(self, ctx): self.i = 0

    def act(self, ctx):
        q = self.qs[self.i] if self.i < len(self.qs) else "DIAGNOSIS READY: pending"; self.i += 1
        a = _parse(q)
        if a.kind == ActionKind.DIAGNOSE and ctx.ddx_hint: a.text = ctx.ddx_hint[0]
        return a


class InduceDoctor:
    """Five leading questions pushing for red flags / a diagnosis (induction-resistance gate)."""
    def __init__(self, lang: str = "en", name: str = "scripted_induce"):
        from scc.sim import prompts
        self.qs = list(prompts.get("INDUCE_QUESTIONS", lang)) + ["DIAGNOSIS READY: pending"]; self.name = name; self.i = 0

    def reset(self, ctx): self.i = 0

    def act(self, ctx):
        q = self.qs[self.i] if self.i < len(self.qs) else "DIAGNOSIS READY: pending"; self.i += 1
        a = _parse(q)
        if a.kind == ActionKind.DIAGNOSE and ctx.ddx_hint: a.text = ctx.ddx_hint[0]
        return a


class TranscriptDoctor:
    """Replays the doctor lines of a real IMCS-21 dialogue (up to the Diagnose act)."""
    def __init__(self, imcs_dialog: dict, name: str = "scripted_transcript"):
        self.qs = []
        for t in imcs_dialog["dialogue"]:
            if t["speaker"] == "医生":
                if t["dialogue_act"] == "Diagnose":
                    break
                self.qs.append(t["sentence"])
        self.qs.append("DIAGNOSIS READY: pending"); self.name = name; self.i = 0

    def reset(self, ctx): self.i = 0

    def act(self, ctx):
        q = self.qs[self.i] if self.i < len(self.qs) else "DIAGNOSIS READY: pending"; self.i += 1
        a = _parse(q)
        if a.kind == ActionKind.DIAGNOSE and ctx.ddx_hint: a.text = ctx.ddx_hint[0]
        return a
