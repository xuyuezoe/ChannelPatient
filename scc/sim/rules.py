"""RuleEngine: pure code that decides, per turn, which atoms are said, withheld, and with what value.
No randomness here; all randomness lives in the report table built at initialisation."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
from scc.types import Atom, Case, PatientConfig, ReportEntry, TurnPlan
from scc.env.cases import ClusterConfig
from scc.env import channels as CH

GENERAL_FORMS = ("open", "chat")


class RuleEngine:
    def __init__(self, case: Case, report_table: dict[str, ReportEntry], self_dx: tuple[str, str] | None,
                 cfg_p: PatientConfig, cfg: ClusterConfig):
        self.case, self.table, self.self_dx, self.cfg_p, self.cfg = case, dict(report_table), self_dx, cfg_p, cfg
        self.n_asked: dict[str, int] = defaultdict(int)
        self.verified: set[str] = set()
        self.disclosed_once: set[str] = set()
        self.last_directive_turn = -10

    # ------------------------------------------------------------------ disclosure logic
    @staticmethod
    def disclosure_allows(disc: Any, form: str, n_asked: int) -> bool:
        if disc in ("spontaneous", "on_general_ask"):
            return True
        if disc == "on_specific_ask":
            return form not in GENERAL_FORMS
        if isinstance(disc, tuple) and disc[0] == "after_n_asks":
            return form not in GENERAL_FORMS and n_asked >= int(disc[1])
        return False

    def _value(self, aid: str, form: str) -> Any:
        if aid in self.verified:
            return self.case.atom(aid).true_value
        return self.table[aid].value_for(form)

    def _directive(self, turn: int) -> str | None:
        if not self.self_dx:
            return None
        conv = self.self_dx[1]
        if conv == "open_mention":
            return "open_mention" if turn == 0 else None
        if conv == "insist":
            if turn == 0:
                return "open_mention"
            if turn - self.last_directive_turn >= 2:
                self.last_directive_turn = turn
                return "insist"
            return None
        if conv == "reframe":
            return "open_mention" if turn == 0 else "reframe"
        return None

    # ------------------------------------------------------------------ plans
    def opening(self) -> TurnPlan:
        say = []
        for a in self.case.atoms:
            if self.table[a.id].disclosure == "spontaneous":
                say.append((a, self._value(a.id, "open"))); self.disclosed_once.add(a.id)
        d = self._directive(0)
        return TurnPlan(say, [], (self.self_dx[0], d) if (self.self_dx and d) else None, False, "open")

    def plan(self, hits: list[str], form: str, turn: int, is_chat: bool = False) -> TurnPlan:
        say, withheld, flooding = [], [], []
        from collections import Counter
        slot_hits = Counter(self.case.atom(a).slot for a in hits)
        for aid in hits:
            atom = self.case.atom(aid); entry = self.table[aid]
            self.n_asked[aid] += 1
            if aid in self.verified:
                say.append((atom, atom.true_value)); continue
            # an atom the classifier says was asked about counts as specifically asked, whatever the phrasing
            eff_form = form if form not in GENERAL_FORMS else "yes_no"
            if not self.disclosure_allows(entry.disclosure, eff_form, self.n_asked[aid]):
                withheld.append(atom); continue
            v = self._value(aid, form)
            if form == "open" and atom.type == "bool" and v is False and atom.slot in ("trigger", "relief", "associated") and slot_hits[atom.slot] >= 3:
                continue          # broad "what brings it on?" sweep: patients list what applies, not every negative; leave it unasked
            say.append((atom, v)); self.disclosed_once.add(aid)
            if entry.rule == "exaggerate:flooding":
                flooding.append(atom)
        if form in GENERAL_FORMS and not is_chat:
            # a general "what's wrong / tell me more" also brings out the on_general_ask atoms not yet disclosed
            for a in self.case.atoms:
                if a.id in self.disclosed_once or a.id in hits:
                    continue
                if self.table[a.id].disclosure in ("spontaneous", "on_general_ask"):
                    say.append((a, self._value(a.id, form))); self.disclosed_once.add(a.id)
        d = self._directive(turn)
        return TurnPlan(say, withheld, (self.self_dx[0], d) if (self.self_dx and d) else None, is_chat, form, flooding)

    # ------------------------------------------------------------------ verification reaction
    def on_verified(self, aid: str) -> None:
        self.verified.add(aid)
        shift = int(self.cfg_p.post_verify_shift)
        if shift == 0:
            return
        # soften the channel for atoms NOT yet disclosed: recompute exaggeration with k+shift
        for a in self.case.atoms:
            if a.id in self.disclosed_once or a.id == aid:
                continue
            e = self.table[a.id]
            if e.rule.startswith("exaggerate:step+"):
                k = int(e.rule.split("+")[1]); nk = max(k + shift, 0)
                lv = CH._levels_for(a)
                if lv:
                    e.report_value = CH.step_up(a.true_value, nk, lv)
                    e.rule = f"exaggerate:step+{nk}" if nk else "identity"
            elif e.rule == "exaggerate:pattern" and shift < 0:
                e.report_value, e.rule = a.true_value, "identity"
