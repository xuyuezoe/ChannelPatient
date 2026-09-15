"""Cluster-level medical small table: P(true value | dx, slot[, tag])."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from scc.config import settings
from scc.env.cases import ClusterConfig


class Likelihood:
    def __init__(self, cfg: ClusterConfig, path: str | Path | None = None, table: dict | None = None):
        self.cfg = cfg
        if table is None:
            path = path or settings.clusters_dir / cfg.name / "likelihood.yaml"
            table = yaml.safe_load(open(path))
        self.table = {dx: v for dx, v in table.items() if not str(dx).startswith("_")}

    def dist(self, dx: str, slot: str, tag: str | None = None, options: list | None = None) -> dict[Any, float]:
        """Distribution over true values. For bool tagged slots returns {True: p, False: 1-p}."""
        sd = self.cfg.slots.get(slot, {})
        stype = sd.get("type")
        row = self.table.get(dx, {}).get(slot)
        if stype == "bool":
            p = None
            if isinstance(row, dict) and tag in row:
                p = float(row[tag])
            elif isinstance(row, (int, float)):
                p = float(row)
            if p is None:
                p = 0.5
            return {True: p, False: 1.0 - p}
        if stype == "text":
            return {}
        opts = list(options or self.cfg.slot_options(slot, tag) or [])
        if isinstance(row, dict) and tag and tag in row and isinstance(row[tag], dict):
            row = row[tag]
        if not isinstance(row, dict) or not opts:
            return {o: 1.0 / len(opts) for o in opts} if opts else {}
        vals = {o: float(row.get(o, 0.0)) for o in opts}
        s = sum(vals.values())
        if s <= 0:
            return {o: 1.0 / len(opts) for o in opts}
        return {o: v / s for o, v in vals.items()}

    def p_true(self, dx: str, slot: str, value: Any, tag: str | None = None, options: list | None = None) -> float:
        return self.dist(dx, slot, tag, options).get(value, 0.0)

    def noised(self, sigma: float, seed: int = 0) -> "Likelihood":
        """Multiplicative log-normal noise on every number (E-noise robustness experiment)."""
        rng = np.random.default_rng(seed)
        t = copy.deepcopy(self.table)

        def walk(x):
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, (int, float)):
                return float(x) * float(np.exp(rng.normal(0, sigma)))
            return x
        return Likelihood(self.cfg, table=walk(t))
