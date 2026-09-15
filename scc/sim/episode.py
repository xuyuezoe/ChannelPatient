"""ConsultationEnv: the patient side as an environment; run_episode drives a Doctor against it."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
from scc.types import *
from scc.env.cases import ClusterConfig
from scc.env.oracle import Oracle
from scc.sim.patient import ChannelPatient
from scc.sim.router import route

ALL_ACTIONS = [ActionKind.ASK, ActionKind.VERIFY_RECORD, ActionKind.ASK_FAMILY, ActionKind.REQUEST_TEST, ActionKind.DIAGNOSE, ActionKind.CHAT]


class ConsultationEnv:
    def __init__(self, case: Case, patient_cfg: PatientConfig, components: SimComponents, cfg: ClusterConfig,
                 oracle: Oracle | None = None, max_turns: int = 12, lang: str = "en", log_path: str | Path | None = None,
                 episode_id: str | None = None, ddx_hint: bool = True, doctor_name: str = "doctor"):
        self.case, self.cfg_p, self.cfg, self.oracle, self.max_turns, self.lang = case, patient_cfg, cfg, oracle, max_turns, lang
        self.patient = ChannelPatient(case, patient_cfg, components, cfg, lang)
        self.log_path = Path(log_path) if log_path else None
        self.episode_id = episode_id or f"{case.case_id}|{patient_cfg.label()}|{patient_cfg.persona.code()}|seed={patient_cfg.seed}|doctor={doctor_name}"
        self.ddx_hint = list(case.labels.ddx_set) if ddx_hint else None
        self.turn = 0; self.transcript: list[tuple[str, str]] = []; self.turn_logs: list[TurnLog] = []; self.done = False

    # ------------------------------------------------------------------ helpers
    def _ctx(self) -> DoctorContext:
        return DoctorContext(self.case.brief(self.lang), list(self.transcript), self.turn, self.max_turns, self.ddx_hint, list(ALL_ACTIONS), self.lang)

    def _disclosed(self, plan: TurnPlan | None) -> list[dict]:
        if not plan:
            return []
        return [{"atom": a.id, "true_value": a.true_value, "report_value": v, "rule": ("verified" if a.id in self.patient.rules.verified else self.patient.report_table[a.id].rule),
                 "n_asked": self.patient.rules.n_asked[a.id]} for a, v in plan.say]

    def _log(self, log: TurnLog) -> None:
        if self.oracle is not None:
            self.oracle.update(log); log.oracle = self.oracle.snapshot()
        self.turn_logs.append(log)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ API
    def reset(self) -> tuple[Observation, DoctorContext]:
        self.turn = 0; self.transcript = []; self.turn_logs = []; self.done = False
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True); self.log_path.write_text("")
        utt, plan, res = self.patient.opening(self.transcript)
        obs = Observation(ObsKind.PATIENT, utt, 0)
        self.transcript.append(("patient", utt))
        self._log(TurnLog(self.episode_id, 0, None, "open", [a.id for a, _ in plan.say], self._disclosed(plan), [], plan.self_dx[1] if plan.self_dx else None,
                          obs, {"pass": res.passed, "retries": res.retries, "fallback": res.fallback, "violations": res.violations},
                          llm_calls=dict(self.patient.calls), timing_ms={}))
        return obs, self._ctx()

    def step(self, action: DoctorAction) -> tuple[Observation, DoctorContext, TurnLog, bool]:
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        self.turn += 1
        t0 = time.time()
        before = dict(self.patient.calls)
        if action.kind in (ActionKind.VERIFY_RECORD, ActionKind.ASK_FAMILY, ActionKind.REQUEST_TEST, ActionKind.DIAGNOSE):
            obs, disclosed = route(action, self.case, self.patient.rules, self.cfg, self.lang, self.turn)
            self.transcript.append(("doctor", action.text or f"{action.kind.value}: {action.target}"))
            self.transcript.append((obs.kind.value.lower(), obs.text))
            log = TurnLog(self.episode_id, self.turn, action, None, [d["atom"] for d in disclosed], disclosed, [], None, obs,
                          {"pass": True, "retries": 0, "fallback": False, "violations": []}, llm_calls={}, timing_ms={"total_ms": int((time.time() - t0) * 1000)})
        else:
            utt, plan, co, res, timing = self.patient.respond(action, self.transcript, self.turn)
            obs = Observation(ObsKind.PATIENT, utt, self.turn)
            self.transcript.append(("doctor", action.text)); self.transcript.append(("patient", utt))
            calls = {k: self.patient.calls[k] - before.get(k, 0) for k in self.patient.calls}
            log = TurnLog(self.episode_id, self.turn, action, co.question_form, list(co.atom_ids), self._disclosed(plan), [a.id for a in plan.withheld],
                          plan.self_dx[1] if plan.self_dx else None, obs,
                          {"pass": res.passed, "retries": res.retries, "fallback": res.fallback, "violations": res.violations},
                          llm_calls=calls, timing_ms={**timing, "total_ms": int((time.time() - t0) * 1000)}, classifier_raw=co.raw)
        self._log(log)
        self.done = obs.kind == ObsKind.END or self.turn >= self.max_turns
        if self.done and obs.kind != ObsKind.END:
            obs = Observation(ObsKind.END, "max turns reached", self.turn)
        return obs, self._ctx(), log, self.done


def run_episode(case: Case, patient_cfg: PatientConfig, doctor: Doctor, components: SimComponents, cfg: ClusterConfig,
                oracle: Oracle | None = None, max_turns: int = 12, lang: str = "en", log_path: str | Path | None = None,
                episode_id: str | None = None) -> list[TurnLog]:
    env = ConsultationEnv(case, patient_cfg, components, cfg, oracle, max_turns, lang, log_path, episode_id, doctor_name=getattr(doctor, "name", "doctor"))
    obs, ctx = env.reset()
    doctor.reset(ctx)
    while True:
        action = doctor.act(ctx)
        obs, ctx, log, done = env.step(action)
        if done:
            break
    return env.turn_logs
