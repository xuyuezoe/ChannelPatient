"""被测医生 ApiDoctor：现成 LLM 加 prompt 基线（vanilla / cot / uncertainty / told），自由文本 + READOUT。

设计要点：
- 只读 DoctorContext（简介 + 对话记录），拿不到病例事实。
- 每轮一次 LLM 调用；READOUT 解析失败即记缺失，不修补数字；连续失败两次后在 prompt 末尾追加格式提醒。
- Told 基线的提示由实验层按患者的真实信道名注入（told_channel），医生对象本身不看 Case。
- build_prompt() 独立出来，RL 环境复用，保证训练与评测的 prompt 一字不差。
"""
from __future__ import annotations
from typing import Protocol
from scc.types import DoctorAction, DoctorContext, ActionKind
from scc.doctor.base import parse_doctor_text, format_action_protocol
from scc.doctor import prompts

ROLE_LABEL = {
    "en": {"doctor": "Doctor", "patient": "Patient", "record": "Record", "family": "Family member", "test": "Test result", "end": "End"},
    "zh": {"doctor": "医生", "patient": "患者", "record": "记录", "family": "家属", "test": "检查结果", "end": "结束"},
}
FAIL_STREAK_FOR_REMINDER = 2


class ChatClientLike(Protocol):
    """医生用到的 LLM 客户端接口：chat(messages, max_tokens, temperature) -> str。"""

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float | None = None, json_schema: dict | None = None, stop=None) -> str | dict: ...


def told_hint(channel_name: str, lang: str = "en") -> str:
    """把患者配置的信道名（如 exaggerate_k2 / vague_p08 / self_dx）映射成 Told 基线的病历备注。"""
    kind = channel_name.split("_")[0] if channel_name else "cooperative"
    if channel_name.startswith("self_dx"):
        kind = "self_dx"
    table = prompts.get("TOLD_HINT", lang)
    return table.get(kind, "")


class ApiDoctor:
    """prompt 基线医生。

    参数：
        client: LLM 客户端（真实 ChatClient 或测试用 FakeClient）
        baseline: vanilla | cot | uncertainty | told
        lang: en | zh
        told_channel: baseline=told 时注入的信道名；其他基线忽略
        max_tokens / temperature: 生成参数
        history_turns: 放进 prompt 的最近轮数（None = 全部）
    """

    def __init__(self, client: ChatClientLike, baseline: str = "vanilla", lang: str = "en", told_channel: str | None = None,
                 max_tokens: int = 400, temperature: float = 0.0, history_turns: int | None = None, name: str | None = None):
        if baseline not in ("vanilla", "cot", "uncertainty", "told"):
            raise ValueError(f"未知基线 {baseline}")
        self.client, self.baseline, self.lang = client, baseline, lang
        self.told_channel, self.max_tokens, self.temperature, self.history_turns = told_channel, max_tokens, temperature, history_turns
        self.name = name or f"api_{baseline}"
        self.fail_streak = 0
        self.n_parse_fail = 0
        self.n_turns = 0
        self.last_raw = ""

    # ------------------------------------------------------------------ prompt
    def system_text(self, ctx: DoctorContext) -> str:
        extra_tbl = prompts.get("BASELINE_EXTRA", self.lang)
        extra = extra_tbl[self.baseline]
        if self.baseline == "told":
            extra = extra.format(hint=told_hint(self.told_channel or "cooperative", self.lang))
        protocol = format_action_protocol(self.lang, ctx.ddx_hint)
        return prompts.get("DOCTOR_SYS", self.lang).format(brief=ctx.case_brief, protocol=protocol, extra=extra).strip()

    def transcript_text(self, ctx: DoctorContext) -> str:
        labels = ROLE_LABEL[self.lang]
        turns = ctx.transcript if self.history_turns is None else ctx.transcript[-2 * self.history_turns:]
        sep = ": " if self.lang == "en" else "："
        return "\n".join(f"{labels.get(role, role)}{sep}{text}" for role, text in turns)

    def build_prompt(self, ctx: DoctorContext) -> list[dict]:
        """拼装本轮的 messages。RL 环境直接调用此函数，保证训练与评测同构。"""
        user = self.transcript_text(ctx)
        tail = "\n\n" + ("Your turn." if self.lang == "en" else "轮到你了。")
        if self.fail_streak >= FAIL_STREAK_FOR_REMINDER:
            tail += "\n" + prompts.get("FORMAT_REMINDER", self.lang)
        return [{"role": "system", "content": self.system_text(ctx)}, {"role": "user", "content": user + tail}]

    # ------------------------------------------------------------------ Doctor 协议
    def reset(self, ctx: DoctorContext) -> None:
        self.fail_streak = 0; self.n_parse_fail = 0; self.n_turns = 0; self.last_raw = ""

    def act(self, ctx: DoctorContext) -> DoctorAction:
        text = self.client.chat(self.build_prompt(ctx), max_tokens=self.max_tokens, temperature=self.temperature)
        self.last_raw = text if isinstance(text, str) else str(text)
        action = parse_doctor_text(self.last_raw)
        self.n_turns += 1
        if action.readout is None:
            self.fail_streak += 1; self.n_parse_fail += 1
        else:
            self.fail_streak = 0
        if action.kind == ActionKind.CHAT and not action.text.strip():
            action = DoctorAction(ActionKind.ASK, "Could you tell me more about the pain?" if self.lang == "en" else "能再说说这个疼吗？", readout=action.readout)
        return action

    def stats(self) -> dict:
        return {"turns": self.n_turns, "parse_fail": self.n_parse_fail}


class FakeClient:
    """测试与 stub 用的假 LLM：按顺序回放给定文本，记录收到的 prompt，计数调用。"""

    def __init__(self, texts: list[str], loop_last: bool = True):
        self.texts, self.loop_last = list(texts), loop_last
        self.i = 0; self.calls = 0; self.prompts: list[list[dict]] = []

    def chat(self, messages, max_tokens=256, temperature=None, json_schema=None, stop=None):
        self.calls += 1; self.prompts.append(messages)
        if self.i < len(self.texts):
            t = self.texts[self.i]; self.i += 1
        else:
            t = self.texts[-1] if self.loop_last else ""
        if json_schema is not None:
            import json
            try:
                return json.loads(t)
            except Exception:
                from scc.llm import ChatClient
                return ChatClient._parse_json(t)
        return t

    def stats(self) -> dict:
        return {"model": "fake", "calls": self.calls, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0}


def scripted_doctor_texts(ddx: list[str], lang: str = "en") -> list[str]:
    """D-M1 用：默认 12 问 + READOUT 的回放脚本，概率逐轮向第一个候选集中。"""
    from scc.sim.scripted_doctor import DEFAULT_QUESTIONS
    import json
    qs = DEFAULT_QUESTIONS[lang]
    out = []
    n = len(ddx)
    for i, q in enumerate(qs):
        p0 = min(0.9, 0.2 + 0.06 * i)
        rest = (1 - p0) / max(1, n - 1)
        topk = {d: (p0 if k == 0 else rest) for k, d in enumerate(ddx)}
        line = f"READOUT: {json.dumps({'topk': {k: round(v, 3) for k, v in topk.items()}, 'confidence': round(min(0.95, 0.3 + 0.05 * i), 2)})}"
        text = q.replace("DIAGNOSIS READY: pending", f"DIAGNOSIS READY: {ddx[0]}")
        out.append(text + "\n" + line)
    return out
