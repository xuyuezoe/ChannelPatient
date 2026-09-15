"""局部似然提供器：P(报告值 | 诊断 z, 信道 u, 槽位, 问法)。

支持集 S(slot) = 可选值 ∪ 模糊 token ∪ {unknown, not_mentioned}；bool 槽为 {True, False, unknown, not_mentioned}。
- TableLikelihood：用患者侧的医学小表 × 信道核（与 oracle 同一份代码）。是 agent 的"作弊上界"。
- LLMLikelihood：让 LLM 直接给出分布；键 (model, z, u, slot, tag, form) 磁盘缓存；失败回退 Table 并计数。
两者同一接口：dist(z, u, slot, tag, form, n_asked) 与 dist_given_true(u, slot, tag, form, true_value, n_asked)。
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Protocol
from scc.config import settings
from scc.types import Atom, ChannelSpec
from scc.env.cases import ClusterConfig
from scc.env.likelihood import Likelihood
from scc.env import channels as CH
from scc.doctor.agent.state import UNKNOWN, NOT_MENTIONED

ETA_UNKNOWN = 0.02       # "没说清"的固定小概率


def default_u_library(cfg: ClusterConfig, name: str = "default") -> dict[str, list[ChannelSpec]]:
    """agent 的信道假设库。default = 簇配置里 oracle 用的那份；k9 = 扩到 9 个（更细的参数网格）。"""
    base = {k: [ChannelSpec.from_dict(s) for s in v] for k, v in cfg.u_library.items()}
    if name == "k2":
        return {k: v for k, v in base.items() if k in ("cooperative", "exaggerate_k2")}
    if name == "k9":
        base.update({
            "exaggerate_k3": [ChannelSpec("exaggerate", {"k": 3, "flooding_rate": 0.5})],
            "vague_p05": [ChannelSpec("vague", {"p": 0.5})],
            "omit_q03": [ChannelSpec("omit", {"q": 0.3, "soften": {2: 0.7, 3: 0.3}})],
            "vague_exag": [ChannelSpec("vague", {"p": 0.8}), ChannelSpec("exaggerate", {"k": 1, "flooding_rate": 0.1})],
        })
    return base


def pseudo_atom(cfg: ClusterConfig, slot: str, tag: str | None) -> Atom:
    """按槽位定义造一个没有真值的 Atom，供信道核与 p_withheld 用。"""
    sd = cfg.slots[slot]
    opts = cfg.slot_options(slot, tag)
    return Atom(id=f"{slot}:{tag or ''}", slot=slot, type=sd["type"], true_value=None, options=list(opts) if opts else None,
                disclosure=sd.get("default_disclosure", "on_specific_ask"), tags=[tag] if tag else [])


def support(cfg: ClusterConfig, slot: str, tag: str | None) -> list:
    sd = cfg.slots[slot]
    if sd["type"] == "bool":
        return [True, False, UNKNOWN, NOT_MENTIONED]
    if sd["type"] == "text":
        return [UNKNOWN, NOT_MENTIONED]
    opts = list(cfg.slot_options(slot, tag) or [])
    toks = sorted({cfg.vague_token(slot, o) for o in opts} - {None})
    return opts + toks + [UNKNOWN, NOT_MENTIONED]


class LikelihoodProvider(Protocol):
    def dist(self, z: str, u: str, slot: str, tag: str | None, form: str, n_asked: int = 1) -> dict[Any, float]: ...
    def dist_given_true(self, u: str, slot: str, tag: str | None, form: str, true_value: Any, n_asked: int = 1) -> dict[Any, float]: ...
    def p_true(self, z: str, slot: str, tag: str | None) -> dict[Any, float]: ...


class TableLikelihood:
    """查表版：医学小表 × 信道核。"""
    name = "table"

    def __init__(self, cfg: ClusterConfig, u_library: dict[str, list[ChannelSpec]], likelihood: Likelihood | None = None):
        self.cfg, self.u_library = cfg, u_library
        self.L = likelihood or Likelihood(cfg)
        self._cache: dict = {}

    def p_true(self, z: str, slot: str, tag: str | None) -> dict[Any, float]:
        return self.L.dist(z, slot, tag, self.cfg.slot_options(slot, tag))

    def _report_dist_given_true(self, u: str, atom: Atom, true_value: Any, form: str) -> dict[Any, float]:
        specs = self.u_library[u]
        vals = [v for v in support(self.cfg, atom.slot, atom.tag) if v not in (UNKNOWN, NOT_MENTIONED)]
        d = {v: CH.likelihood_kernel(specs, atom, true_value, v, form, self.cfg) for v in vals}
        s = sum(d.values())
        return {k: (v / s if s > 0 else 0.0) for k, v in d.items()}

    def _wrap(self, base: dict[Any, float], u: str, atom: Atom, form: str, n_asked: int) -> dict[Any, float]:
        p_nm = 0.0
        if atom.slot == "associated":
            p_nm = CH.p_withheld(self.u_library[u], atom, n_asked, form)
        p_nm = min(p_nm, 0.95)
        # 与 oracle 一致：报告值的相对似然只差常数 (1-η)，"问了没说"的质量 p_nm 与 oracle 的 p_withheld 相同
        out = {k: (1.0 - ETA_UNKNOWN) * (1.0 - p_nm) * v for k, v in base.items()}
        out[UNKNOWN] = ETA_UNKNOWN * (1.0 - p_nm); out[NOT_MENTIONED] = p_nm
        return out

    def dist(self, z: str, u: str, slot: str, tag: str | None, form: str, n_asked: int = 1) -> dict[Any, float]:
        key = ("d", z, u, slot, tag, form, min(n_asked, 4))
        if key in self._cache:
            return self._cache[key]
        atom = pseudo_atom(self.cfg, slot, tag)
        if atom.type == "text":
            out = {UNKNOWN: 0.5, NOT_MENTIONED: 0.5}
        else:
            pt = self.p_true(z, slot, tag)
            base: dict[Any, float] = {}
            for t, p in pt.items():
                for r, q in self._report_dist_given_true(u, atom, t, form).items():
                    base[r] = base.get(r, 0.0) + p * q
            out = self._wrap(base, u, atom, form, n_asked)
        self._cache[key] = out
        return out

    def dist_given_true(self, u: str, slot: str, tag: str | None, form: str, true_value: Any, n_asked: int = 1) -> dict[Any, float]:
        atom = pseudo_atom(self.cfg, slot, tag)
        if atom.type == "text":
            return {UNKNOWN: 0.5, NOT_MENTIONED: 0.5}
        return self._wrap(self._report_dist_given_true(u, atom, true_value, form), u, atom, form, n_asked)


class LLMFullLikelihood(TableLikelihood):
    """全量 LLM 估计版：P(报告 | z, u, 槽, 问法) 整个由 LLM 给出（每个 (z,u,槽,问法) 一次调用，缓存）。
    抽查（results/D_M4_LIK_CHECK.md）显示它对 U 无关的槽也会随信道变化，且调用量大，故默认不用；保留作对照。"""
    name = "llm_full"

    def __init__(self, cfg: ClusterConfig, u_library, client, lang: str = "en", cache_dir: str | Path | None = None, likelihood: Likelihood | None = None):
        super().__init__(cfg, u_library, likelihood)
        self.client, self.lang = client, lang
        self.cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir / "likelihood"
        self.n_calls = 0; self.n_fallback = 0; self.n_cache = 0

    def _key(self, z, u, slot, tag, form) -> str:
        return hashlib.sha256(json.dumps([getattr(self.client, "model", "?"), z, u, slot, tag, form, self.lang]).encode()).hexdigest()

    def _llm_dist(self, z: str, u: str, slot: str, tag: str | None, form: str) -> dict[Any, float] | None:
        from scc.doctor import prompts
        key = self._key(z, u, slot, tag, form); p = self.cache_dir / f"{key}.json"
        if p.exists():
            self.n_cache += 1
            return {json.loads(k) if k.startswith(("true", "false")) else k: v for k, v in json.load(open(p)).items()}
        vals = [v for v in support(self.cfg, slot, tag) if v not in (UNKNOWN, NOT_MENTIONED)]
        names = ["yes" if v is True else "no" if v is False else str(v) for v in vals]
        chan = prompts.get("CHANNEL_DESC", self.lang).get(u, u)
        slot_txt = f"{slot}{' (' + self.cfg.tag_display(slot, tag, self.lang) + ')' if tag else ''}"
        msgs = [{"role": "system", "content": prompts.get("LIKELIHOOD_SYS", self.lang)},
                {"role": "user", "content": prompts.get("LIKELIHOOD_USER", self.lang).format(dx=self.cfg.dx_display(z, self.lang), channel=chan, form=form, slot=slot_txt, values=", ".join(names))}]
        schema = {"type": "object", "properties": {"dist": {"type": "object"}}, "required": ["dist"], "additionalProperties": False}
        try:
            out = self.client.chat(msgs, max_tokens=200, temperature=0.0, json_schema=schema)
            self.n_calls += 1
            raw = out.get("dist", {})
            d = {}
            for v, nm in zip(vals, names):
                d[v] = max(0.0, float(raw.get(nm, raw.get(str(v), 0.0))))
            s = sum(d.values())
            if s <= 0:
                return None
            d = {k: x / s for k, x in d.items()}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            json.dump({("true" if k is True else "false" if k is False else k): x for k, x in d.items()}, open(p, "w"))
            return d
        except Exception:
            return None

    def dist(self, z: str, u: str, slot: str, tag: str | None, form: str, n_asked: int = 1) -> dict[Any, float]:
        key = ("l", z, u, slot, tag, form, min(n_asked, 4))
        if key in self._cache:
            return self._cache[key]
        atom = pseudo_atom(self.cfg, slot, tag)
        if atom.type == "text":
            out = {UNKNOWN: 0.5, NOT_MENTIONED: 0.5}
        else:
            base = self._llm_dist(z, u, slot, tag, form)
            if base is None:
                self.n_fallback += 1
                return super().dist(z, u, slot, tag, form, n_asked)
            out = self._wrap(base, u, atom, form, n_asked)
        self._cache[key] = out
        return out


class LLMLikelihood(TableLikelihood):
    """LLM 估计版（默认）：LLM 只估**医学小表** P(真值 | z, 槽)，信道核仍由代码给出。

    理由：信道规则是 agent 的先验知识（"夸大的人会往重说两档"），不是需要估的量；把 U 的作用交给代码，
    U 无关性（T4）和单调性（T3）自动成立，LLM 只需回答医学常识问题。每个 (槽, tag) 一次调用给出全部 z 的分布，
    磁盘缓存；失败回退查表并计数。
    """
    name = "llm"

    def __init__(self, cfg: ClusterConfig, u_library, client, lang: str = "en", cache_dir: str | Path | None = None, likelihood: Likelihood | None = None):
        super().__init__(cfg, u_library, likelihood)
        self.client, self.lang = client, lang
        self.cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir / "likelihood_med"
        self.n_calls = 0; self.n_fallback = 0; self.n_cache = 0
        self._med: dict[tuple, dict[str, dict]] = {}

    def _key(self, slot: str, tag: str | None) -> str:
        return hashlib.sha256(json.dumps([getattr(self.client, "model", "?"), self.cfg.name, slot, tag, self.lang, sorted(self.cfg.ddx_set)]).encode()).hexdigest()

    def _estimate_slot(self, slot: str, tag: str | None) -> dict[str, dict] | None:
        """返回 {z: {value: p}}；bool 槽的 value 为 True/False。"""
        from scc.doctor import prompts
        key = self._key(slot, tag); p = self.cache_dir / f"{key}.json"
        sd = self.cfg.slots[slot]
        vals = [True, False] if sd["type"] == "bool" else list(self.cfg.slot_options(slot, tag) or [])
        names = ["yes" if v is True else "no" if v is False else str(v) for v in vals]
        if p.exists():
            self.n_cache += 1
            raw = json.load(open(p))
            return {z: {vals[names.index(k)]: v for k, v in d.items() if k in names} for z, d in raw.items()}
        slot_txt = f"{slot}{' (' + self.cfg.tag_display(slot, tag, self.lang) + ')' if tag else ''}"
        rows = "\n".join(f"- {z}: {self.cfg.dx_display(z, self.lang)}" for z in self.cfg.ddx_set)
        user = (f"For each candidate condition below, estimate the probability distribution of the TRUE value of the clinical feature '{slot_txt}' in a typical patient with that condition.\n"
                f"Conditions:\n{rows}\nPossible values: {', '.join(names)}\n"
                f"Return JSON {{condition_id: {{value: prob, ...}}, ...}} using the condition ids exactly as given; each row sums to 1.")
        schema = {"type": "object", "additionalProperties": {"type": "object"}}
        try:
            out = self.client.chat([{"role": "system", "content": prompts.get("LIKELIHOOD_SYS", self.lang)}, {"role": "user", "content": user}], max_tokens=600, temperature=0.0, json_schema=schema)
            self.n_calls += 1
            res: dict[str, dict] = {}
            for z in self.cfg.ddx_set:
                d = out.get(z) or {}
                row = {v: max(0.0, float(d.get(nm, 0.0))) for v, nm in zip(vals, names)}
                s_ = sum(row.values())
                if s_ <= 0:
                    return None
                res[z] = {k: x / s_ for k, x in row.items()}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            json.dump({z: {nm: res[z][v] for v, nm in zip(vals, names)} for z in res}, open(p, "w"))
            return res
        except Exception:
            return None

    def p_true(self, z: str, slot: str, tag: str | None) -> dict[Any, float]:
        if self.cfg.slots[slot]["type"] == "text":
            return {}
        key = (slot, tag)
        if key not in self._med:
            est = self._estimate_slot(slot, tag)
            if est is None:
                self.n_fallback += 1; est = {zz: super(LLMLikelihood, self).p_true(zz, slot, tag) for zz in self.cfg.ddx_set}
            self._med[key] = est
        return self._med[key][z]
