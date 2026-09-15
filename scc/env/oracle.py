"""Oracle: exact joint posterior P(Z, U | history) at the fact-table level.

Uses the SAME channel rules as the simulator (channels.likelihood_kernel). Handles:
- patient reports (sum over the unknown true value),
- silence after being asked (only the omit hypothesis makes silence likely),
- verification events (record / family): the true value becomes known and earlier reports of that atom are
  re-decoded with the true value fixed (this is where 'catching an exaggerator' happens).
Each (atom, question_form) pair counts once (consistency lock makes repeats non-informative).
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Callable
from scc.types import Atom, Case, ChannelSpec, TurnLog
from scc.env.cases import ClusterConfig
from scc.env.likelihood import Likelihood
from scc.env import channels as CH

KernelFn = Callable[[str, Atom, Any, Any, str], float]          # (u_name, atom, true, report, form) -> P
WithheldFn = Callable[[str, Atom, int, str], float]              # (u_name, atom, n_asked, form) -> P(silent)


def _norm(d: dict) -> dict:
    s = sum(d.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in d.items()}


class Oracle:
    def __init__(self, case: Case, cfg: ClusterConfig, likelihood: Likelihood,
                 u_library: dict[str, list[ChannelSpec]] | None = None, prior_u: Any = None, prior_z: Any = None,
                 kernel_fn: KernelFn | None = None, withheld_fn: WithheldFn | None = None, eps: float = 1e-12):
        self.case, self.cfg, self.L = case, cfg, likelihood
        lib = u_library if u_library is not None else {k: [ChannelSpec.from_dict(s) for s in v] for k, v in cfg.u_library.items()}
        self.u_library = lib
        self.zs = list(case.labels.ddx_set)
        self.us = list(lib.keys())
        pz = self._prior(prior_z if prior_z is not None else cfg.prior_z, self.zs)
        pu = self._prior(prior_u if prior_u is not None else cfg.prior_u, self.us)
        self.log_prior = {(z, u): math.log(pz[z]) + math.log(pu[u]) for z in self.zs for u in self.us}
        self.eps = eps
        self.kernel_fn = kernel_fn or self._default_kernel
        self.withheld_fn = withheld_fn or self._default_withheld
        self.reset()

    def _default_kernel(self, u, atom, t, r, form):
        return CH.likelihood_kernel(self.u_library[u], atom, t, r, form, self.cfg)

    def _default_withheld(self, u, atom, n, form):
        return CH.p_withheld(self.u_library[u], atom, n, form)

    @staticmethod
    def _prior(spec: Any, keys: list) -> dict:
        if spec is None or spec == "uniform":
            return {k: 1.0 / len(keys) for k in keys}
        d = {k: float(spec.get(k, 0.0)) for k in keys}
        return _norm(d)

    def reset(self):
        self.log_post = dict(self.log_prior)
        self.reports: dict[str, list[tuple[Any, str]]] = defaultdict(list)     # atom -> [(report, form)]
        self.known_true: dict[str, Any] = {}
        self.n_asked: dict[str, int] = defaultdict(int)
        self.silences: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ likelihood pieces
    def _lik_report(self, z: str, u: str, atom: Atom, report: Any, form: str) -> float:
        if atom.id in self.known_true:
            return self.kernel_fn(u, atom, self.known_true[atom.id], report, form)
        dist = self.L.dist(z, atom.slot, atom.tag, atom.options)
        if not dist:                               # text atoms: no medical table -> uninformative
            return 1.0
        return sum(pt * self.kernel_fn(u, atom, t, report, form) for t, pt in dist.items())

    def _add(self, f: Callable[[str, str], float]):
        for (z, u) in self.log_post:
            self.log_post[(z, u)] += math.log(max(f(z, u), self.eps))

    # ------------------------------------------------------------------ public update
    def update(self, log: TurnLog) -> None:
        form = log.question_form or "open"
        for aid in log.hit_atoms:
            self.n_asked[aid] += 1
        for d in log.disclosed:
            atom = self.case.atom(d["atom"])
            rule = str(d.get("rule", ""))
            if rule.startswith("verified"):
                self._observe_true(atom, d["true_value"])
                continue
            if any(r == d["report_value"] and f == form for r, f in self.reports[atom.id]):
                continue                           # consistency lock: repeat carries no information
            n = self.n_asked[atom.id] or int(d.get("n_asked", 1))
            self.reports[atom.id].append((d["report_value"], form))
            self._add(lambda z, u, a=atom, r=d["report_value"], fm=form: self._lik_report(z, u, a, r, fm))
            if atom.slot == "associated" and atom.disclosure == "on_specific_ask" and len(self.reports[atom.id]) == 1:
                self._add(lambda z, u, a=atom, nn=n, fm=form: 1.0 - self.withheld_fn(u, a, nn, fm))
        for aid in log.withheld:
            atom = self.case.atom(aid)
            if self.reports[aid] or aid in self.known_true:
                continue
            n = self.n_asked[aid]
            self._add(lambda z, u, a=atom, nn=n, fm=form: self.withheld_fn(u, a, nn, fm))

    def _observe_true(self, atom: Atom, true_value: Any) -> None:
        """Verification: (1) P(true | z); (2) re-decode earlier reports of this atom with true fixed."""
        if atom.id in self.known_true:
            return
        old_reports = list(self.reports[atom.id])
        # remove the marginalised report terms
        for r, f in old_reports:
            self._add(lambda z, u, a=atom, rr=r, ff=f: 1.0 / max(self._lik_report(z, u, a, rr, ff), self.eps))
        self.known_true[atom.id] = true_value
        dist_fn = lambda z: self.L.dist(z, atom.slot, atom.tag, atom.options)
        if dist_fn(self.zs[0]):
            self._add(lambda z, u: dist_fn(z).get(true_value, 0.0))
        for r, f in old_reports:
            self._add(lambda z, u, a=atom, rr=r, ff=f: self.kernel_fn(u, a, true_value, rr, ff))

    # ------------------------------------------------------------------ read-outs
    def _post(self) -> dict:
        m = max(self.log_post.values())
        p = {k: math.exp(v - m) for k, v in self.log_post.items()}
        return _norm(p)

    def posterior_joint(self) -> dict:
        return self._post()

    def posterior_z(self) -> dict[str, float]:
        p = self._post(); out = defaultdict(float)
        for (z, u), v in p.items():
            out[z] += v
        return dict(out)

    def posterior_u(self) -> dict[str, float]:
        p = self._post(); out = defaultdict(float)
        for (z, u), v in p.items():
            out[u] += v
        return dict(out)

    def entropy_z(self) -> float:
        return -sum(v * math.log(v) for v in self.posterior_z().values() if v > 0)

    def snapshot(self) -> dict:
        return {"posterior_z": {k: round(v, 6) for k, v in self.posterior_z().items()},
                "posterior_u": {k: round(v, 6) for k, v in self.posterior_u().items()},
                "entropy_z": round(self.entropy_z(), 6), "known_true": dict(self.known_true)}
