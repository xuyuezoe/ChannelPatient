"""ChannelPatient: assembles classifier -> rules -> phraser -> checker for one consultation."""
from __future__ import annotations
import time
from collections import Counter
from scc.types import Case, PatientConfig, DoctorAction, TurnPlan, ClassifierOutput, CheckResult, SimComponents
from scc.env.cases import ClusterConfig
from scc.env.channels import build_report_table
from scc.sim.rules import RuleEngine
from scc.sim.phraser import TemplatePhraser

MAX_RETRIES = 3


class ChannelPatient:
    def __init__(self, case: Case, cfg_p: PatientConfig, components: SimComponents, cfg: ClusterConfig, lang: str = "en"):
        self.case, self.cfg_p, self.comp, self.cfg, self.lang = case, cfg_p, components, cfg, lang
        self.report_table, self.self_dx, self.specs = build_report_table(case, cfg_p, cfg)
        self.rules = RuleEngine(case, self.report_table, self.self_dx, cfg_p, cfg)
        self.role_card = components.role.build(cfg_p.persona, case, lang)
        self.calls = Counter()
        self.fallback = TemplatePhraser(cfg, seed=cfg_p.seed)

    def _say(self, plan: TurnPlan, transcript: list) -> tuple[str, CheckResult]:
        res = CheckResult(True)
        utt = ""
        for attempt in range(MAX_RETRIES + 1):
            utt = self.comp.phraser.phrase(plan, self.role_card, transcript, self.lang); self.calls["phraser"] += 1
            res = self.comp.checker.check(utt, plan, self.case, self.lang); self.calls["checker"] += 1
            res.retries = attempt
            if res.passed:
                return utt, res
        utt = self.fallback.phrase(plan, self.role_card, transcript, self.lang)
        res.fallback = True
        return utt, res

    def opening(self, transcript: list) -> tuple[str, TurnPlan, CheckResult]:
        plan = self.rules.opening()
        utt, res = self._say(plan, transcript)
        return utt, plan, res

    def respond(self, action: DoctorAction, transcript: list, turn: int) -> tuple[str, TurnPlan, ClassifierOutput, CheckResult, dict]:
        t0 = time.time()
        co = self.comp.classifier.classify(action.text, self.case.atoms, transcript, self.lang); self.calls["classifier"] += 1
        t1 = time.time()
        plan = self.rules.plan(co.atom_ids, co.question_form, turn, co.is_chat)
        utt, res = self._say(plan, transcript)
        timing = {"classifier_ms": int((t1 - t0) * 1000), "phrase_ms": int((time.time() - t1) * 1000)}
        return utt, plan, co, res, timing
