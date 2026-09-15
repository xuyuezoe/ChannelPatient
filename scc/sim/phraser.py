"""Phraser: TurnPlan -> patient utterance. TemplatePhraser (no LLM) and LLMPhraser."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Protocol
from scc.config import ROOT
from scc.types import TurnPlan, Atom
from scc.env.cases import ClusterConfig

BOOL_SLOTS = {"trigger", "relief", "associated", "family_history", "medical_history"}


class Phraser(Protocol):
    def phrase(self, plan: TurnPlan, role_card: str, transcript: list, lang: str) -> str: ...


def _label_display(cfg: ClusterConfig, label: str, lang: str) -> str:
    return cfg.dx_display(label, lang)


class TemplatePhraser:
    name = "template"

    def __init__(self, cfg: ClusterConfig, seed: int = 0):
        self.cfg, self.rng = cfg, random.Random(seed)

    def _one(self, atom: Atom, value, lang: str) -> str:
        c = self.cfg
        if atom.slot == "chief_complaint":
            return "I've been having chest pain." if lang == "en" else "我胸口疼。"
        if isinstance(value, str) and value in c.vague_tokens():
            d = c.display(atom.slot, value, lang)
            return (f"It's {d}." if lang == "en" else f"{d}。")
        if atom.slot == "severity":
            w = self.rng.choice(c.severity_words(value, lang))
            return (f"It's {w}." if lang == "en" else f"{w}。")
        if atom.slot == "functional_impact":
            w = self.rng.choice(c.functional_words(value, lang))
            return (f"{w[0].upper() + w[1:]}." if lang == "en" else f"{w}。")
        if atom.type == "bool" or atom.slot in BOOL_SLOTS:
            tagd = c.tag_display(atom.slot, atom.tag, lang) if atom.tag else atom.slot
            if atom.slot == "associated":
                return (f"Yes, I've had {tagd}." if value else f"No {tagd}.") if lang == "en" else (f"有{tagd}。" if value else f"没有{tagd}。")
            if atom.slot in ("trigger",):
                return (f"Yes, it's worse {tagd}." if value else f"No, {tagd} doesn't change it.") if lang == "en" else (f"对，{tagd}会更疼。" if value else f"{tagd}没什么关系。")
            if atom.slot == "relief":
                return (f"Yes, {tagd} helps." if value else f"No, {tagd} doesn't help.") if lang == "en" else (f"{tagd}会好一点。" if value else f"{tagd}也不管用。")
            if atom.slot == "family_history":
                return (f"Yes, there is {tagd}." if value else f"No {tagd} that I know of.") if lang == "en" else (f"有，{tagd}。" if value else f"没有{tagd}。")
            if atom.slot == "medical_history":
                return (f"Yes, I have {tagd}." if value else f"No {tagd}.") if lang == "en" else (f"有{tagd}。" if value else f"没有{tagd}。")
            return (f"{tagd}: {'yes' if value else 'no'}.")
        if atom.type == "text":
            return (f"Last time they said: {value}." if lang == "en" else f"上次说的是：{value}。")
        d = c.display(atom.slot, value, lang)
        if atom.slot == "location":
            return f"It's {d}." if lang == "en" else f"就在{d}。"
        if atom.slot == "quality":
            return f"It feels {d}." if lang == "en" else f"是{d}的感觉。"
        if atom.slot == "duration":
            return f"It's been going on for {d}." if lang == "en" else f"{d}。"
        if atom.slot == "lifestyle":
            tagd = c.tag_display(atom.slot, atom.tag, lang) if atom.tag else ""
            return f"{d.capitalize()}." if lang == "en" else f"{d}。"
        return f"{d[0].upper() + d[1:]}." if lang == "en" else f"{d}。"

    def phrase(self, plan: TurnPlan, role_card: str, transcript: list, lang: str = "en") -> str:
        c = self.cfg
        parts = []
        if plan.self_dx and plan.self_dx[1] == "open_mention":
            parts.append(c.vocab["self_dx_phrases"]["open_mention"][lang].format(label=_label_display(c, plan.self_dx[0], lang)))
        if plan.is_chat and not plan.say:
            parts.append("Okay." if lang == "en" else "好的。")
        for atom, value in plan.say:
            parts.append(self._one(atom, value, lang))
        if plan.withheld:
            w = self.rng.choice(c.phrases("withheld_phrases", lang))
            parts.append((w + (" about anything else." if plan.say else ".")) if lang == "en" else (("别的" + w) if plan.say else w) + "。")
        if plan.flooding:
            parts.append(self.rng.choice(c.phrases("flooding_phrases", lang)) + ("." if lang == "en" else "。"))
        if plan.self_dx and plan.self_dx[1] in ("insist", "reframe"):
            parts.append(c.vocab["self_dx_phrases"][plan.self_dx[1]][lang].format(label=_label_display(c, plan.self_dx[0], lang)).strip())
        if not parts:
            parts.append("Hmm." if lang == "en" else "嗯。")
        sep = " " if lang == "en" else ""
        return sep.join(p.strip() for p in parts if p.strip())


class LLMPhraser:
    name = "llm"

    def __init__(self, cfg: ClusterConfig, client, style_dir: Path | None = None, n_examples: int = 3, temperature: float = 0.7):
        self.cfg, self.client, self.n, self.temperature = cfg, client, n_examples, temperature
        self.style_dir = style_dir or ROOT / "data" / "style"
        self._cache: dict = {}
        self.template = TemplatePhraser(cfg)

    def _examples(self, plan: TurnPlan, lang: str) -> str:
        key = (lang, "severity")
        ex = []
        sev = [v for a, v in plan.say if a.slot == "severity"]
        f = self.style_dir / lang / "severity.jsonl"
        if sev and f.exists():
            if key not in self._cache:
                rows = [json.loads(l) for l in open(f)]
                self._cache[key] = rows
            rows = [r["text"] for r in self._cache[key] if r.get("level") == sev[0]][:50]
            ex += rows[: self.n]
        if plan.self_dx:
            f2 = self.style_dir / lang / "self_dx.jsonl"
            if f2.exists():
                rows = [json.loads(l)["text"] for l in open(f2)][:50]
                ex += rows[: self.n]
        return " | ".join(ex) if ex else ("(none)" if lang == "en" else "（无）")

    def allowed_text(self, plan: TurnPlan, lang: str) -> str:
        return "\n".join("- " + self.template._one(a, v, lang) for a, v in plan.say) or ("- (nothing new; respond briefly in character)" if lang == "en" else "- （没有新内容；简短回应）")

    def phrase(self, plan: TurnPlan, role_card: str, transcript: list, lang: str = "en") -> str:
        from scc.sim import prompts
        c = self.cfg
        withheld = ", ".join((c.tag_display(a.slot, a.tag, lang) if a.tag else a.slot) for a in plan.withheld) or ("nothing" if lang == "en" else "无")
        sd = ""
        if plan.self_dx:
            sd = prompts.get("SELF_DX_INSTR", lang)[plan.self_dx[1]].format(label=_label_display(c, plan.self_dx[0], lang))
        tr = "\n".join(f"{r}: {t}" for r, t in transcript[-6:]) or "(start of consultation)"
        sys_msg = prompts.get("PHRASER_SYS", lang).format(role_card=role_card)
        user = prompts.get("PHRASER_USER", lang).format(transcript=tr, allowed=self.allowed_text(plan, lang), withheld=withheld, self_dx=sd, examples=self._examples(plan, lang))
        text = self.client.chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], max_tokens=160, temperature=self.temperature)
        if "<response>" in text:
            text = text.split("<response>")[1].split("</response>")[0]
        return text.strip()
