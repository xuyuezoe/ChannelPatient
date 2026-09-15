"""Metric functions over run logs (first version: kl_per_turn, clarification_rate, verify_usage; others are signatures)."""
from __future__ import annotations
import math
import pandas as pd
from scc.analysis.trajectory import load_run

CLARIFYING_FORMS = ("forced_choice", "point_to")


def kl_per_turn(df: pd.DataFrame) -> pd.DataFrame:
    """KL(readout || oracle_z) per turn where a readout exists."""
    out = []
    for _, r in df.iterrows():
        tk, oz = r.readout_topk, r.oracle_z
        if not tk or not oz:
            continue
        s = sum(max(0.0, v) for v in tk.values()) or 1.0
        kl = 0.0
        for k, v in tk.items():
            p = max(0.0, v) / s; q = oz.get(k, 1e-9)
            if p > 0:
                kl += p * math.log(p / max(q, 1e-12))
        out.append({"case_id": r.case_id, "channel": r.channel, "seed": r.seed, "turn": r.turn, "kl": kl})
    return pd.DataFrame(out)


def clarification_rate(df: pd.DataFrame, forms=CLARIFYING_FORMS) -> pd.DataFrame:
    asks = df[df.action_kind == "ASK"]
    return asks.assign(clar=asks.question_form.isin(forms)).groupby(["channel", "doctor"]).clar.mean().reset_index(name="clarification_rate")


def verify_usage(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(v=df.action_kind.isin(["VERIFY_RECORD", "ASK_FAMILY"])).groupby(["channel", "doctor", "case_id", "seed"]).v.max().groupby(["channel", "doctor"]).mean().reset_index(name="verify_usage")


def mirage_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative confidence increase on turns where oracle P(true dx) did not increase. Needs readouts."""
    raise NotImplementedError("needs doctor readouts; implemented in the doctor-side phase")


def post_verify_discount_ratio(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("needs doctor readouts; implemented in the doctor-side phase")
