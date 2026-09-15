"""对 run 日志（load_run 的 DataFrame）计算指标。

单位约定：KL 与熵用 nat；概率 0-1；轮次从 0（患者开场）计。
每个指标返回 DataFrame，含分组列（channel、doctor、case_id、seed）以便汇总。
"""
from __future__ import annotations
import math
import pandas as pd
from scc.analysis.trajectory import load_run

CLARIFYING_FORMS = ("forced_choice", "point_to")
ANCHOR_KINDS = ("VERIFY_RECORD", "ASK_FAMILY")
EPISODE_KEYS = ["case_id", "channel", "doctor", "seed"]


def _norm(d: dict | None) -> dict:
    if not d:
        return {}
    s = sum(max(0.0, float(v)) for v in d.values())
    return {k: max(0.0, float(v)) / s for k, v in d.items()} if s > 0 else {}


def kl_per_turn(df: pd.DataFrame) -> pd.DataFrame:
    """KL(readout ‖ oracle_z)，只在有 readout 的轮计算。"""
    out = []
    for _, r in df.iterrows():
        tk, oz = _norm(r.readout_topk), r.oracle_z
        if not tk or not oz:
            continue
        kl = sum(p * math.log(p / max(oz.get(k, 1e-9), 1e-12)) for k, p in tk.items() if p > 0)
        out.append({**{k: r[k] for k in EPISODE_KEYS}, "turn": r.turn, "kl": kl})
    return pd.DataFrame(out)


def clarification_rate(df: pd.DataFrame, forms=CLARIFYING_FORMS) -> pd.DataFrame:
    """澄清率 = forced_choice / point_to 问法占提问动作的比例，按信道 × 医生汇总。"""
    asks = df[df.action_kind == "ASK"]
    if asks.empty:
        return pd.DataFrame(columns=["channel", "doctor", "clarification_rate"])
    return asks.assign(clar=asks.question_form.isin(forms)).groupby(["channel", "doctor"]).clar.mean().reset_index(name="clarification_rate")


def verify_usage(df: pd.DataFrame) -> pd.DataFrame:
    """核实动作使用率 = 用过 VERIFY_RECORD / ASK_FAMILY 的段占比，按信道 × 医生。"""
    v = df.assign(v=df.action_kind.isin(ANCHOR_KINDS)).groupby(EPISODE_KEYS).v.max().reset_index()
    return v.groupby(["channel", "doctor"]).v.mean().reset_index(name="verify_usage")


def readout_parse_rate(df: pd.DataFrame) -> pd.DataFrame:
    """READOUT 解析成功率：医生动作轮中 readout 非空的比例。"""
    d = df[df.action_kind.notna()]
    if d.empty:
        return pd.DataFrame(columns=["channel", "doctor", "readout_parse_rate"])
    return d.assign(ok=d.readout_topk.notna()).groupby(["channel", "doctor"]).ok.mean().reset_index(name="readout_parse_rate")


def accuracy_final(df: pd.DataFrame, primary: dict[str, str]) -> pd.DataFrame:
    """终局准确率：DIAGNOSE 动作的文本命中真实诊断（不区分大小写、允许包含）。未下诊断记 0。"""
    rows = []
    for keys, g in df.groupby(EPISODE_KEYS):
        diag = g[g.action_kind == "DIAGNOSE"]
        text = str(diag.iloc[-1].get("doctor_text", "") if "doctor_text" in g.columns and not diag.empty else "")
        truth = primary[keys[0]]
        ok = bool(text) and (truth.lower() in text.lower() or truth.replace("_", " ").lower() in text.lower())
        rows.append({**dict(zip(EPISODE_KEYS, keys)), "correct": float(ok)})
    r = pd.DataFrame(rows)
    return r.groupby(["channel", "doctor"]).correct.mean().reset_index(name="accuracy_final")


def _p_true_series(g: pd.DataFrame, truth: str) -> list[float]:
    return [float((oz or {}).get(truth, float("nan"))) for oz in g.oracle_z]


def mirage_gap(df: pd.DataFrame, primary: dict[str, str]) -> pd.DataFrame:
    """Mirage Gap：在 oracle P(真实诊断) 没有上升的轮次上，医生自报置信度上涨幅度的累积（每段一个数）。
    只统计相邻两个都有 readout 的轮。"""
    rows = []
    for keys, g in df.groupby(EPISODE_KEYS):
        g = g.sort_values("turn"); truth = primary[keys[0]]
        pt = _p_true_series(g, truth); conf = list(g.readout_conf)
        gap = 0.0; n = 0
        for i in range(1, len(g)):
            if conf[i] is None or conf[i - 1] is None or (isinstance(conf[i], float) and math.isnan(conf[i])) or (isinstance(conf[i - 1], float) and math.isnan(conf[i - 1])):
                continue
            n += 1
            if pt[i] <= pt[i - 1] + 1e-12 and conf[i] > conf[i - 1]:
                gap += conf[i] - conf[i - 1]
        rows.append({**dict(zip(EPISODE_KEYS, keys)), "mirage_gap": gap, "n_pairs": n})
    return pd.DataFrame(rows)


def post_verify_discount_ratio(df: pd.DataFrame, primary: dict[str, str]) -> pd.DataFrame:
    """核实后折扣比：核实事件（RECORD/FAMILY 观测）前后，医生 readout 里真实诊断概率的变化 ÷ oracle 的变化。
    以核实那一轮 t 为界：before = t-1 的值，after = 核实后第一个有 readout 的轮。"""
    rows = []
    for keys, g in df.groupby(EPISODE_KEYS):
        g = g.sort_values("turn").reset_index(drop=True); truth = primary[keys[0]]
        idx = [i for i, k in enumerate(g.obs_kind) if k in ("RECORD", "FAMILY")]
        if not idx:
            continue
        t = idx[0]
        if t == 0 or t + 1 >= len(g):
            continue
        before_o = (g.oracle_z[t - 1] or {}).get(truth); after_o = (g.oracle_z[t] or {}).get(truth)
        before_d = _norm(g.readout_topk[t - 1]).get(truth); after_d = None
        for j in range(t + 1, len(g)):
            tk = _norm(g.readout_topk[j])
            if tk:
                after_d = tk.get(truth); break
        if None in (before_o, after_o, before_d, after_d):
            continue
        d_o, d_d = after_o - before_o, after_d - before_d
        rows.append({**dict(zip(EPISODE_KEYS, keys)), "delta_oracle": d_o, "delta_doctor": d_d, "ratio": (d_d / d_o) if abs(d_o) > 1e-6 else float("nan")})
    return pd.DataFrame(rows)


def episode_turns(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(EPISODE_KEYS).turn.max().reset_index(name="turns")
