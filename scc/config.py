"""Project settings: relay URL/key from pilot/.env, proxy handling, defaults."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "pilot" / ".env"
PROXY_VARS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines into os.environ (setdefault). Same format as pilot/f4_pilot.py."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def proxy_free_env() -> dict:
    """Copy of os.environ without proxy variables (the relay must be reached directly)."""
    return {k: v for k, v in os.environ.items() if k not in PROXY_VARS}


@dataclass
class Settings:
    base_url: str = ""
    api_key: str = ""
    cache_dir: Path = ROOT / ".cache" / "llm"
    default_models: dict = field(default_factory=lambda: {
        "conversational": "qwen3.7-plus", "meta": "qwen3.7-max", "checker": "regex"})
    lang: str = "en"
    clusters_dir: Path = ROOT / "scc" / "env" / "clusters"
    results_dir: Path = ROOT / "results"
    external_dir: Path = ROOT / "external"


def _make() -> Settings:
    load_env()
    return Settings(base_url=os.environ.get("OPENAI_BASE_URL", ""), api_key=os.environ.get("OPENAI_API_KEY", ""))


settings = _make()
