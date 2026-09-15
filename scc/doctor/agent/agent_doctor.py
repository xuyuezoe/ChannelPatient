"""脚手架 agent：BeliefAgentDoctor。组装感知 → 账本 → 信念 → 候选 → 价值 → 监测 → 策略。

每轮 LLM 调用：感知 1（LLM 版）+ 候选 1（LLM 版）+ 似然估计（LLM 版，缓存）；stub 版全部为 0。
消融通过 AgentConfig 开关：z_only / lookahead / anchoring / stopping / monitor / manner_signals / u_library / likelihood。
"""
from __future__ import annotations
import re
from typing import Any
from scc.types import Doctor, DoctorAction, DoctorContext, ActionKind
from scc.env.cases import ClusterConfig
from scc.doctor.agent.state import AgentConfig, Candidate, LedgerEntry, SemanticObs, UNKNOWN, NOT_MENTIONED
from scc.doctor.agent.likelihood import TableLikelihood, LLMLikelihood, default_u_library
from scc.doctor.agent.belief import BeliefEngine
from scc.doctor.agent.perception import RegexPerceiver, LLMPerceiver
from scc.doctor.agent.candidates import MenuCandidates, LLMCandidates
from scc.doctor.agent.value import ca_ig, lookahead2, rvoi, lookahead2_rvoi, loss_matrix
from scc.doctor.agent.monitor import SurprisalMonitor
from scc.doctor.agent.policy import choose, choose_rvoi, readout
from scc.doctor.agent.styler import TemplateStyler, LLMStyler

MANNER_LIK = {"intensity": {"exaggerate": 1.6, "cooperative": 0.7}, "flooding": {"exaggerate": 2.0, "cooperative": 0.5}}


class BeliefAgentDoctor:
    def __init__(self, cluster_cfg: ClusterConfig, cfg: AgentConfig, perceiver, generator, provider, lang: str = "en", name: str | None = None, styler=None):
        self.cfg, self.acfg, self.perceiver, self.generator, self.P, self.lang = cluster_cfg, cfg, perceiver, generator, provider, lang
        self.us = ["cooperative"] if cfg.z_only else list(provider.u_library.keys())
        self.name = name or ("agent_zonly" if cfg.z_only else "agent")
        self.Lm = loss_matrix(list(cluster_cfg.ddx_set), list(cluster_cfg.red_flags), cluster_cfg.loss)
        self.q_cost = cfg.question_cost if cfg.question_cost is not None else float(cluster_cfg.loss.get("question_cost", 0.05))
        self.styler = styler or TemplateStyler(cfg.style, cfg.seed)
        self.reset(None)

    # ------------------------------------------------------------------ 状态
    def reset(self, ctx: DoctorContext | None) -> None:
        self.belief = BeliefEngine(list(self.cfg.ddx_set), self.us, self.P)
        self.monitor = SurprisalMonitor(self.acfg.surprisal_threshold)
        self.ledger: list[LedgerEntry] = []
        self.n_asked: dict[tuple, int] = {}
        self.last: Candidate | None = None
        self.turn = 0
        self.decisions: list[dict] = []
        self.n_seen_transcript = 0
        self.answered: set[tuple] = set()          # 已得到明确（非 unknown / 非模糊）答案的 (slot, tag)

    # ------------------------------------------------------------------ 观测处理
    def _parse_verified(self, kind: str, text: str) -> dict[tuple, Any]:
        """解析路由返回的记录/家属文本（我们自己的格式）为 {(slot, tag): value}。"""
        out: dict[tuple, Any] = {}
        c = self.cfg
        m = re.search(r"recorded as '([^']+)'", text) or re.search(r"记为 (\S+)", text)
        if m and self.last is not None and self.last.slot:
            out[(self.last.slot, self.last.tag)] = m.group(1)
            return out
        if kind == "family":
            for slot in ("family_history",):
                for tag in c.slots[slot].get("tag_options", []):
                    td = c.tag_display(slot, tag, self.lang)
                    if td.lower() in text.lower():
                        yes = c.display(slot, True, self.lang).lower(); no = c.display(slot, False, self.lang).lower()
                        if no and no in text.lower():
                            out[(slot, tag)] = False
                        elif yes and yes in text.lower():
                            out[(slot, tag)] = True
        return out

    def _ingest(self, ctx: DoctorContext) -> None:
        """把 transcript 里新出现的观测（上一轮我们的动作之后）读进账本与信念。"""
        new = ctx.transcript[self.n_seen_transcript:]
        self.n_seen_transcript = len(ctx.transcript)
        for role, text in new:
            if role == "doctor":
                continue
            if role in ("record", "family"):
                tv = self._parse_verified(role, text)
                self.ledger.append(LedgerEntry(self.turn, role, text, self.last.text if self.last else "", [], {f"{k[0]}|{k[1] or ''}": v for k, v in tv.items()}))
                for (slot, tag), v in tv.items():
                    vv = v
                    opts = self.cfg.slot_options(slot, tag)
                    if opts and v not in opts:
                        for o in opts:
                            if str(o) == str(v) or self.cfg.display(slot, o, self.lang).lower() == str(v).lower():
                                vv = o
                    self.belief.observe_true(slot, tag, vv)
                continue
            if role != "patient":
                continue
            obs = self.perceiver.perceive(text, self.last, self.lang)
            entry = LedgerEntry(self.turn, "patient", text, self.last.text if self.last else "", obs, None, {f"{k[0]}|{k[1] or ''}": v for k, v in self.n_asked.items()})
            self.ledger.append(entry)
            for o in obs:
                n = self.n_asked.get((o.slot, o.tag), 0)
                top = max(o.dist, key=o.dist.get)
                if top not in (UNKNOWN, NOT_MENTIONED) and top not in self.cfg.vague_tokens():
                    self.answered.add((o.slot, o.tag))
                if self.acfg.monitor and self.last is not None and (o.slot, o.tag) == (self.last.slot, self.last.tag):
                    pred = self.belief.predictive(o.slot, o.tag, o.form, max(1, n))
                    if self.monitor.update(pred, o):
                        self.belief.temper_u(self.acfg.temper_alpha); self.belief.replay(); self.monitor.reset()
                self.belief.update_soft(o, max(1, n))
                if self.acfg.manner_signals and not self.acfg.z_only:
                    self._manner_update(o)

    def _manner_update(self, o: SemanticObs) -> None:
        """方式信号作为对 U 的附加似然：程度词很强 / flooding → 夸大类假设加权。"""
        inten = float(o.manner.get("intensity", 0.0)); fl = float(o.manner.get("flooding", 0.0))
        if inten < 0.9 and fl < 0.5:
            return
        for (z, u) in self.belief.log_post:
            w = 1.0
            if inten >= 0.9:
                w *= MANNER_LIK["intensity"]["exaggerate" if u.startswith("exaggerate") else "cooperative"] ** 0.5
            if fl >= 0.5:
                w *= MANNER_LIK["flooding"]["exaggerate" if u.startswith("exaggerate") else "cooperative"] ** 0.5
            import math
            self.belief.log_post[(z, u)] += math.log(w)

    # ------------------------------------------------------------------ 决策
    def _view(self, ctx: DoctorContext) -> dict:
        pz = self.belief.posterior_z(); pu = self.belief.posterior_u()
        top2 = sorted(pz, key=pz.get, reverse=True)[:2]
        return {"top2": top2, "channel_guess": max(pu, key=pu.get), "asked": {f"{k[0]}|{k[1] or ''}": v for k, v in self.n_asked.items()},
                "transcript": ctx.transcript, "lang": self.lang, "red_flag": self.cfg.red_flags[0] if self.cfg.red_flags else ""}

    def act(self, ctx: DoctorContext) -> DoctorAction:
        self.turn = ctx.turn
        self._ingest(ctx)
        cands = self.generator.propose(self._view(ctx))
        if not self.acfg.anchoring:
            cands = [c for c in cands if c.kind == ActionKind.ASK]
        use_rvoi = self.acfg.objective == "rvoi"
        for c in cands:
            n1 = self.n_asked.get((c.slot, c.tag), 0) + 1
            c.value = rvoi(self.belief, c, self.Lm, n1) if use_rvoi else ca_ig(self.belief, c, n1)
            if c.kind == ActionKind.ASK and (c.slot, c.tag) in self.answered:
                c.value = 0.0                                  # 已经答清楚的槽，再问没有价值
        cands.sort(key=lambda c: -c.value)
        cands = cands[: self.acfg.max_candidates]
        if self.acfg.lookahead:
            pool = cands[: self.acfg.lookahead_top]
            for c in cands:
                if c.is_calibration:
                    if use_rvoi:
                        c.value, c.value_detail = lookahead2_rvoi(self.belief, c, pool, self.Lm, self.n_asked, self.acfg.lookahead_top)
                    else:
                        c.value, c.value_detail = lookahead2(self.belief, c, pool, self.n_asked, self.acfg.lookahead_top)
        if use_rvoi:
            action, info = choose_rvoi(cands, self.belief, self.acfg, self.Lm, self.q_cost, self.n_asked, self.lang)
        else:
            action, info = choose(cands, self.belief, self.acfg, list(self.cfg.red_flags), self.n_asked, self.lang)
        self.decisions.append({"turn": self.turn, **info, "belief": self.belief.snapshot(), "monitor": self.monitor.snapshot()})
        chosen = next((c for c in cands if c.text == action.text and c.kind == action.kind), None)
        self.last = chosen
        if chosen is not None and chosen.slot:
            self.n_asked[(chosen.slot, chosen.tag)] = self.n_asked.get((chosen.slot, chosen.tag), 0) + 1
        if chosen is not None and action.kind == ActionKind.ASK:
            action = DoctorAction(action.kind, self.styler.style(chosen, ctx.transcript, self.lang), target=action.target, readout=action.readout)
        return action

    def stats(self) -> dict:
        d = {"turns": self.turn, "alarms": self.monitor.alarms, "ledger": len(self.ledger)}
        for comp in (self.perceiver, self.generator, self.P):
            for k in ("n_calls", "n_fallback", "n_cache"):
                if hasattr(comp, k):
                    d[f"{type(comp).__name__}.{k}"] = getattr(comp, k)
        return d


def build_agent_doctor(spec: dict, cluster_cfg: ClusterConfig, lang: str, clients: dict | None = None) -> BeliefAgentDoctor:
    """按配置组装 agent。components: stub | llm；likelihood: table | llm。"""
    clients = clients if clients is not None else {}
    acfg = AgentConfig.from_dict(spec)
    ulib = default_u_library(cluster_cfg, acfg.u_library)
    comp = spec.get("components", "stub")
    meta = None
    if comp == "llm" or acfg.likelihood == "llm":
        from scc.llm import ChatClient
        meta = clients.get("doctor_meta") or ChatClient(spec.get("model", "qwen3.7-max"), temperature=0.0)
        clients["doctor_meta"] = meta
    provider = LLMLikelihood(cluster_cfg, ulib, meta, lang) if acfg.likelihood == "llm" else TableLikelihood(cluster_cfg, ulib)
    styler = None
    if comp == "llm":
        from scc.sim.classifier import KeywordClassifier
        perceiver = LLMPerceiver(cluster_cfg, meta)
        generator = LLMCandidates(cluster_cfg, meta, KeywordClassifier(cluster_cfg), acfg.anchoring, lang, acfg.n_llm_questions, acfg.n_llm_clarify)
        if acfg.style != "concise":
            conv = clients.get("doctor") or ChatClient(spec.get("style_model", "qwen3.7-plus"), temperature=0.5)
            clients["doctor"] = conv
            styler = LLMStyler(cluster_cfg, conv, KeywordClassifier(cluster_cfg), acfg.style, acfg.seed)
    else:
        perceiver = RegexPerceiver(cluster_cfg)
        generator = MenuCandidates(cluster_cfg, acfg.anchoring, lang)
    return BeliefAgentDoctor(cluster_cfg, acfg, perceiver, generator, provider, lang, name=spec.get("name"), styler=styler)
