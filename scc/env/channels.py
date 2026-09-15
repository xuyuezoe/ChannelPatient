"""Channel kernels: true value -> reported value. The SAME rules are used forward (simulator report table)
and backward (likelihood_kernel for the oracle), so simulator and oracle can never disagree.

Randomness only at initialisation (build_report_table) via numpy default_rng(seed).
"""
from __future__ import annotations
import copy
from typing import Any
import numpy as np
import yaml
from scc.config import ROOT
from scc.types import Atom, Case, ChannelSpec, PatientConfig, Persona, ReportEntry
from scc.env.cases import ClusterConfig

LEVELS = {"ordinal_3": ["mild", "moderate", "severe"], "functional_impact": ["none", "limits_activity", "wakes_at_night"]}
APPLY_ORDER = ["omit", "vague", "exaggerate", "understate", "relay_distort"]
IMPLEMENTED = {"cooperative", "exaggerate", "vague", "omit", "self_dx"}
PERSONA_TABLE_PATH = ROOT / "scc" / "env" / "clusters" / "_shared" / "persona_to_channel.yaml"


# ----------------------------------------------------------------------------- helpers
def _levels_for(atom: Atom) -> list | None:
    if atom.type == "ordinal_3":
        return atom.options or LEVELS["ordinal_3"]
    if atom.slot == "functional_impact":
        return atom.options or LEVELS["functional_impact"]
    return None


def step_up(value: Any, k: int, options: list) -> Any:
    i = options.index(value)
    return options[min(i + k, len(options) - 1)]


def step_down(value: Any, k: int, options: list) -> Any:
    i = options.index(value)
    return options[max(i - k, 0)]


def _identity(atom: Atom) -> ReportEntry:
    return ReportEntry(atom.id, atom.true_value, atom.true_value, "identity", atom.disclosure)


# ----------------------------------------------------------------------------- forward kernels
def kernel_exaggerate(atom: Atom, entry: ReportEntry, params: dict, rng: np.random.Generator, cfg: ClusterConfig) -> ReportEntry:
    """Exaggeration acts in four places, all as 'jump k levels', never on location/quality/history."""
    k = int(params.get("k", 1)); fl = float(params.get("flooding_rate", 0.0))
    e = copy.copy(entry)
    if (atom.type == "ordinal_3" or atom.slot == "functional_impact") and _levels_for(atom):
        new = step_up(entry.report_value, k, _levels_for(atom))
        if new != entry.report_value:
            e.report_value, e.rule = new, f"exaggerate:step+{k}"
    elif atom.slot == "pattern" and k >= 1 and entry.report_value == "episodic":
        e.report_value, e.rule = "constant", "exaggerate:pattern"
    elif atom.slot == "associated" and atom.true_value is False and fl > 0:
        if rng.random() < fl:
            e.report_value, e.rule = True, "exaggerate:flooding"
    return e


def kernel_vague(atom: Atom, entry: ReportEntry, params: dict, rng: np.random.Generator, cfg: ClusterConfig) -> ReportEntry:
    """Categorical slots with a vague_map entry: under open questions the patient uses the vague token w.p. p;
    forced_choice / point_to questions get the true value (channel degrades to identity)."""
    p = float(params.get("p", 0.8))
    tok = cfg.vague_token(atom.slot, entry.report_value) if atom.type == "categorical" else None
    if tok is None:
        return entry
    e = copy.copy(entry)
    if rng.random() < p:
        e.by_form = {"open": tok, "chat": tok, "forced_choice": entry.report_value, "point_to": entry.report_value,
                     "yes_no": entry.report_value, "severity_open": tok, "scale": entry.report_value, "recall_test": tok}
        e.report_value, e.rule = tok, f"vague:p={p}"
    else:
        e.rule = f"vague:kept(p={p})"
    return e


def kernel_omit(atom: Atom, entry: ReportEntry, params: dict, rng: np.random.Generator, cfg: ClusterConfig) -> ReportEntry:
    """Does not change values. Associated symptoms that are on_specific_ask become after_n_asks w.p. q."""
    q = float(params.get("q", 0.5)); soften = params.get("soften", {2: 0.6, 3: 0.4})
    if atom.slot != "associated" or entry.disclosure != "on_specific_ask":
        return entry
    e = copy.copy(entry)
    if rng.random() < q:
        ns = sorted(int(n) for n in soften); ps = np.array([float(soften[n] if n in soften else soften[str(n)]) for n in ns]); ps = ps / ps.sum()
        n = int(rng.choice(ns, p=ps))
        e.disclosure, e.rule = ("after_n_asks", n), f"omit:after_{n}"
    else:
        e.rule = "omit:kept"
    return e


def kernel_understate(atom, entry, params, rng, cfg):  # second batch
    raise NotImplementedError("understate channel is scheduled for the second batch")


def kernel_relay_distort(atom, entry, params, rng, cfg):  # second batch
    raise NotImplementedError("relay_distort channel is scheduled for the second batch")


KERNELS = {"exaggerate": kernel_exaggerate, "vague": kernel_vague, "omit": kernel_omit,
           "understate": kernel_understate, "relay_distort": kernel_relay_distort}


def pick_self_dx_label(case: Case, params: dict, rng: np.random.Generator, cfg: ClusterConfig) -> tuple[str, str]:
    label = params.get("label")
    if label is None:
        cands = [d for d in cfg.plausible_wrong.get(case.labels.primary_dx, []) if d != case.labels.primary_dx]
        if not cands:
            cands = [d for d in case.labels.ddx_set if d != case.labels.primary_dx]
        label = str(rng.choice(sorted(cands)))
    assert label != case.labels.primary_dx, "self-dx label must be wrong"
    return label, params.get("conviction", "insist")


# ----------------------------------------------------------------------------- persona -> params
def load_persona_table() -> dict:
    return yaml.safe_load(open(PERSONA_TABLE_PATH))


def sample_params(persona: Persona, kind: str, rng: np.random.Generator, table: dict | None = None) -> dict:
    """Persona level decides the SAMPLING DISTRIBUTION of channel parameters (SIM_PLAN_v2 §4.3)."""
    table = table or load_persona_table()
    t = table.get(kind, {})
    out: dict = {}
    for pname, spec in t.items():
        dim = spec["by"]; level = getattr(persona, dim) if dim != "cefr" else persona.cefr
        dist = spec["levels"][level] if level in spec["levels"] else spec["levels"][str(level)]
        if isinstance(dist, dict):                      # {value: prob}
            vals = list(dist.keys()); ps = np.array([float(v) for v in dist.values()]); ps = ps / ps.sum()
            out[pname] = vals[int(rng.choice(len(vals), p=ps))]
        else:
            out[pname] = dist
        if "cefr_bonus" in spec and persona.cefr == "A":
            out[pname] = min(1.0, float(out[pname]) + float(spec["cefr_bonus"]))
    return out


# ----------------------------------------------------------------------------- report table
def resolve_specs(cfg_p: PatientConfig, rng: np.random.Generator) -> list[ChannelSpec]:
    if not cfg_p.sample_params_from_persona:
        return list(cfg_p.channels)
    table = load_persona_table()
    return [ChannelSpec(s.kind, {**sample_params(cfg_p.persona, s.kind, rng, table), **s.params}) for s in cfg_p.channels]


def build_report_table(case: Case, cfg_p: PatientConfig, cfg: ClusterConfig) -> tuple[dict[str, ReportEntry], tuple[str, str] | None, list[ChannelSpec]]:
    """Returns (report_table, self_dx, resolved specs). Deterministic given cfg_p.seed."""
    rng = np.random.default_rng(cfg_p.seed)
    specs = resolve_specs(cfg_p, rng)
    for s in specs:
        if s.kind not in IMPLEMENTED:
            raise NotImplementedError(f"channel {s.kind} not implemented in first batch")
    table = {a.id: _identity(a) for a in case.atoms}
    for kind in APPLY_ORDER:
        for s in [s for s in specs if s.kind == kind]:
            for a in case.atoms:
                table[a.id] = KERNELS[kind](a, table[a.id], s.params, rng, cfg)
    self_dx = None
    for s in specs:
        if s.kind == "self_dx":
            self_dx = pick_self_dx_label(case, s.params, rng, cfg)
    return table, self_dx, specs


# ----------------------------------------------------------------------------- backward: likelihood
def report_support(atom: Atom, cfg: ClusterConfig) -> list:
    """All values a report could take for this atom (true options + vague tokens)."""
    if atom.type == "bool":
        return [True, False]
    opts = list(atom.options or cfg.slot_options(atom.slot, atom.tag) or [])
    toks = {cfg.vague_token(atom.slot, o) for o in opts} - {None}
    return opts + sorted(toks)


def _single_kernel(kind: str, params: dict, atom: Atom, true_value: Any, report_value: Any, form: str, cfg: ClusterConfig) -> float:
    """P(report | true) for ONE channel kind applied to an identity entry (no disclosure effects)."""
    if kind in ("cooperative", "omit", "self_dx"):
        return 1.0 if report_value == true_value else 0.0
    if kind == "exaggerate":
        k = int(params.get("k", 1)); fl = float(params.get("flooding_rate", 0.0))
        if (atom.type == "ordinal_3" or atom.slot == "functional_impact") and _levels_for(atom):
            return 1.0 if report_value == step_up(true_value, k, _levels_for(atom)) else 0.0
        if atom.slot == "pattern" and k >= 1:
            return 1.0 if report_value == ("constant" if true_value == "episodic" else true_value) else 0.0
        if atom.slot == "associated" and true_value is False:
            return fl if report_value is True else 1.0 - fl
        return 1.0 if report_value == true_value else 0.0
    if kind == "vague":
        p = float(params.get("p", 0.8))
        tok = cfg.vague_token(atom.slot, true_value) if atom.type == "categorical" else None
        if tok is None or form in ("forced_choice", "point_to", "yes_no", "scale"):
            return 1.0 if report_value == true_value else 0.0
        if report_value == tok:
            return p
        if report_value == true_value:
            return 1.0 - p
        return 0.0
    raise NotImplementedError(kind)


def likelihood_kernel(specs: list[ChannelSpec], atom: Atom, true_value: Any, report_value: Any, form: str, cfg: ClusterConfig) -> float:
    """P(report_value | true_value, U=specs, question_form). Channels compose in APPLY_ORDER by summing over
    intermediate values, exactly mirroring build_report_table."""
    chain = [s for kind in APPLY_ORDER for s in specs if s.kind == kind]
    if not chain or atom.type == "text":           # text atoms (prior tests) carry no channel in the first batch
        return 1.0 if report_value == true_value else 0.0
    support = report_support(atom, cfg)
    dist = {true_value: 1.0}
    for s in chain:
        nxt: dict = {}
        for mid, pm in dist.items():
            if pm == 0:
                continue
            for out in support:
                pr = _single_kernel(s.kind, s.params, atom, mid, out, form, cfg)
                if pr:
                    nxt[out] = nxt.get(out, 0.0) + pm * pr
        dist = nxt
    return dist.get(report_value, 0.0)


def p_withheld(specs: list[ChannelSpec], atom: Atom, n_asked: int, form: str, eps: float = 0.02) -> float:
    """P(patient did not disclose atom although asked n_asked times | U). Only the omit channel makes silence likely."""
    omit = [s for s in specs if s.kind == "omit"]
    base_allows = atom.disclosure in ("spontaneous", "on_general_ask") or (atom.disclosure == "on_specific_ask" and form != "open")
    if not base_allows:
        return 1.0                                   # silence is expected regardless of U
    if not omit or atom.slot != "associated":
        return eps
    q = float(omit[0].params.get("q", 0.5)); soften = omit[0].params.get("soften", {2: 0.6, 3: 0.4})
    tot = sum(float(v) for v in soften.values())
    p_still = sum(float(v) / tot for n, v in soften.items() if int(n) > n_asked)   # withheld iff n_required > n_asked
    return (1 - q) * eps + q * (p_still + (1 - p_still) * eps)
