"""脚手架 agent 的数据结构：假设、语义观测、账本条目、候选动作、agent 状态。

术语：
- 语义档位 / 报告值支持集 S(slot)：该槽的可选值 ∪ 模糊 token ∪ {"unknown", "not_mentioned"}。
- 软观测 SemanticObs：感知器对一句话在 S 上给出的概率分布，不做硬判决。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
from scc.types import ActionKind, to_dict

UNKNOWN = "unknown"
NOT_MENTIONED = "not_mentioned"


@dataclass
class Hypothesis:
    """假设集元素：(诊断 z, 信道名 u)。"""
    z: str
    u: str


@dataclass
class SemanticObs:
    """感知器对患者一句话在某个槽上的软观测。

    dist 的键是报告值（options / 模糊 token / unknown / not_mentioned），值为概率，和为 1。
    manner 是"怎么说"的信号：intensity（程度词强度 0-1）、flooding（0/1）、self_attribution（0/1）、hedging（0-1）。
    """
    slot: str
    tag: str | None
    form: str
    dist: dict[Any, float]
    manner: dict[str, float] = field(default_factory=dict)
    raw: str = ""

    def to_dict(self) -> dict:
        return {"slot": self.slot, "tag": self.tag, "form": self.form, "dist": {str(k): v for k, v in self.dist.items()}, "manner": self.manner, "raw": self.raw}


@dataclass
class LedgerEntry:
    """账本一条：只追加，不修改。core 内容是原文 + 解析结果 + 当时的上下文。"""
    turn: int
    kind: str                              # patient | record | family | test
    utterance: str
    question: str
    obs: list[SemanticObs] = field(default_factory=list)
    true_values: dict[str, Any] | None = None   # 核实事件：{"slot|tag": value}
    n_asked_snapshot: dict[str, int] = field(default_factory=dict)
    likelihood_version: int = 0

    def to_dict(self) -> dict:
        return {"turn": self.turn, "kind": self.kind, "utterance": self.utterance, "question": self.question,
                "obs": [o.to_dict() for o in self.obs], "true_values": self.true_values, "likelihood_version": self.likelihood_version}


@dataclass
class Candidate:
    """一个候选动作。slot/tag/form 是价值计算用的抽象；text 是说给患者听的原句。"""
    kind: ActionKind
    text: str
    slot: str | None = None
    tag: str | None = None
    form: str | None = None
    target: str | None = None
    is_calibration: bool = False
    value: float = 0.0
    value_detail: dict = field(default_factory=dict)
    source: str = "menu"                   # menu | llm | redflag

    def key(self) -> tuple:
        return (self.kind.value, self.slot, self.tag, self.form, self.target)

    def to_dict(self) -> dict:
        d = to_dict(self); d["kind"] = self.kind.value; return d


@dataclass
class AgentConfig:
    """agent 的开关；默认全开 = 完整方法。"""
    z_only: bool = False
    lookahead: bool = True
    anchoring: bool = True
    stopping: bool = True
    monitor: bool = True
    manner_signals: bool = True
    u_library: str = "default"
    likelihood: str = "table"              # table | llm
    objective: str = "rvoi"                # rvoi（期望损失下降，默认）| caig（信息增益，供 P1-P3 检验）
    style: str = "concise"                 # 对话风格旋钮：concise | detailed | warm；只改措辞，不改问什么
    question_cost: float | None = None     # None = 用簇配置的 loss.question_cost
    epsilon_stop: float = 0.01             # 所有候选价值低于此（nat）视为无信息可得
    tau_stop: float = 0.6                  # 最大后验低于此且无信息可得 -> 不可辨识
    tau_red: float = 0.15                  # 红旗后验高于此不许下诊断
    tau_diagnose: float = 0.85             # 最大后验高于此且红旗已排除 -> 下诊断
    surprisal_threshold: float = 3.0       # nat
    temper_alpha: float = 0.5
    max_candidates: int = 40
    lookahead_top: int = 10
    n_llm_questions: int = 6
    n_llm_clarify: int = 2
    seed: int = 0

    @staticmethod
    def from_dict(d: dict) -> "AgentConfig":
        known = {f for f in AgentConfig.__dataclass_fields__}
        return AgentConfig(**{k: v for k, v in d.items() if k in known})
