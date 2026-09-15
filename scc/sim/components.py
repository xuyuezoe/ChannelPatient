"""Factories for SimComponents (stub / llm)."""
from __future__ import annotations
from scc.types import SimComponents
from scc.env.cases import ClusterConfig
from scc.sim.classifier import KeywordClassifier, LLMClassifier
from scc.sim.phraser import TemplatePhraser, LLMPhraser
from scc.sim.checker import RegexChecker, NLIChecker
from scc.sim.persona import TemplateRole, PWPRole


def stub_components(cfg: ClusterConfig, seed: int = 0) -> SimComponents:
    return SimComponents(KeywordClassifier(cfg), TemplatePhraser(cfg, seed), RegexChecker(cfg), TemplateRole(), name="stub")


def llm_components(cfg: ClusterConfig, conv_client, meta_client, checker: str = "regex", role: str = "template") -> SimComponents:
    chk = NLIChecker(cfg, meta_client) if checker.startswith("nli") else RegexChecker(cfg)
    rl = PWPRole(meta_client) if role == "pwp" else TemplateRole()
    return SimComponents(LLMClassifier(cfg, meta_client), LLMPhraser(cfg, conv_client), chk, rl, name="llm")
