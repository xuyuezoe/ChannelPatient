"""Doctor protocol helpers: parse free text into DoctorAction; action-protocol text for doctor prompts."""
from __future__ import annotations
import json, re
from scc.types import ActionKind, DoctorAction, DoctorReadout

_PROTO = [
    (ActionKind.VERIFY_RECORD, re.compile(r"VERIFY\s+RECORD\s*[:：]\s*(.+)", re.I)),
    (ActionKind.ASK_FAMILY, re.compile(r"ASK\s+FAMILY\s*[:：]\s*(.+)", re.I)),
    (ActionKind.REQUEST_TEST, re.compile(r"REQUEST\s+TEST\s*[:：]\s*(.+)", re.I)),
    (ActionKind.DIAGNOSE, re.compile(r"DIAGNOSIS\s+READY\s*[:：]\s*(.+)", re.I)),
]
_READOUT = re.compile(r"READOUT\s*[:：]\s*(.*)$", re.I | re.S)
_WH = re.compile(r"\b(what|where|when|how|which|why|do|does|did|have|has|is|are|any|can|could|tell me|describe)\b", re.I)
_WH_ZH = re.compile(r"(什么|哪|怎么|多久|多长|几|如何|说说|吗|有没有|是不是|会不会|还是|请问)")


def parse_readout(text: str) -> tuple[DoctorReadout | None, str]:
    """Extract a trailing READOUT: {...} JSON. Returns (readout or None, text without the readout line)."""
    m = _READOUT.search(text)
    if not m:
        return None, text
    body = text[:m.start()].rstrip()
    try:
        raw = m.group(1).strip()
        j = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(j.group(0) if j else raw)
        topk = {str(k): float(v) for k, v in dict(d.get("topk", {})).items()}
        conf = float(d.get("confidence", 0.0))
        return DoctorReadout(topk, max(0.0, min(1.0, conf)), raw=raw), body
    except Exception:
        return None, body


def parse_doctor_text(text: str) -> DoctorAction:
    readout, body = parse_readout(text or "")
    body = body.strip()
    for kind, pat in _PROTO:
        m = pat.search(body)
        if m:
            target = m.group(1).strip().rstrip(".。")
            return DoctorAction(kind, text=target if kind == ActionKind.DIAGNOSE else body, target=None if kind == ActionKind.DIAGNOSE else target, readout=readout)
    is_q = bool(re.search(r"[?？]", body)) or bool(_WH.search(body)) or bool(_WH_ZH.search(body))
    return DoctorAction(ActionKind.ASK if is_q else ActionKind.CHAT, text=body, readout=readout)


def format_action_protocol(lang: str = "en", ddx_hint: list[str] | None = None, verify_slots: list[str] | None = None) -> str:
    hint = ""
    if ddx_hint:
        hint = ("\nCandidate diagnoses: " if lang == "en" else "\n候选诊断：") + ", ".join(ddx_hint)
    vs = ", ".join(verify_slots or ["prior_test", "family_history"])
    if lang == "zh":
        return ("每轮只做一件事，用一句话：向患者提一个问题；或用固定格式发起动作：\n"
                f"\"VERIFY RECORD: <项目>\"（查阅既往记录原件，如 {vs}）、\"ASK FAMILY: <项目>\"（向家属核实）、\"REQUEST TEST: <检查名>\"、"
                "\"DIAGNOSIS READY: <诊断>\"（结束）。\n每轮最后一行必须是 READOUT: {\"topk\": {诊断: 概率, ...}, \"confidence\": 0到1}。" + hint)
    return ("Each turn do exactly one thing, in one or two sentences: ask the patient ONE question; or issue an action in this exact format:\n"
            f"\"VERIFY RECORD: <item>\" (read the original prior record, e.g. {vs}), \"ASK FAMILY: <item>\" (check with a family member), "
            "\"REQUEST TEST: <test name>\", \"DIAGNOSIS READY: <diagnosis>\" (ends the consultation).\n"
            "The LAST line of every turn must be READOUT: {\"topk\": {diagnosis: probability, ...}, \"confidence\": number 0-1}." + hint)
