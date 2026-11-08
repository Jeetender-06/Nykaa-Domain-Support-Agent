from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    runtime: Path = field(default_factory=lambda: ROOT / os.getenv("NYKAA_RUNTIME_DIR", ".runtime"))
    model_path: Path = field(
        default_factory=lambda: ROOT / os.getenv("NYKAA_MODEL_PATH", "models/all-MiniLM-L6-v2")
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("NYKAA_MAX_TOKENS", "120000")))
    max_cost_usd: float = field(
        default_factory=lambda: float(os.getenv("NYKAA_MAX_COST_USD", "0.20"))
    )
    admin_token: str = field(default_factory=lambda: os.getenv("NYKAA_ADMIN_TOKEN", ""))
    top_k: int = 3
    cache_capacity: int = 256
    max_sessions: int = 128
    history_turns: int = 6

    def __post_init__(self) -> None:
        if os.getenv("MOCK_LLM", "true").lower() not in {"true", "1"}:
            raise ValueError("This graded implementation supports MOCK_LLM=true only")
        self.runtime.mkdir(parents=True, exist_ok=True)

    @property
    def kb_path(self) -> Path:
        return self.root / "knowledge_base"

    @property
    def calibration_path(self) -> Path:
        return self.runtime / "calibration.json"
