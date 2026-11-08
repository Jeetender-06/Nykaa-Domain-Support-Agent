"""Environment is configured before importing any telemetry-producing dependencies."""

import os
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env", override=False)
os.environ.setdefault("CREWAI_STORAGE_DIR", str(_root / ".runtime" / "crewai"))
os.environ.setdefault("XDG_CACHE_HOME", str(_root / ".cache"))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

for _name, _value in {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "ANONYMIZED_TELEMETRY": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "CREWAI_TRACING_ENABLED": "false",
}.items():
    os.environ[_name] = _value

__version__ = "1.0.0"

# CrewAI 1.15.37 initializes its cloud token manager even with tracing disabled,
# and its credential path has no environment override. Scope that storage to the
# project before importing CrewAI: offline mode must not read a user's saved
# cloud credentials or write into their platform-wide application directory.
try:
    from crewai_core.token_manager import TokenManager  # noqa: E402
except ModuleNotFoundError:
    TokenManager = None  # diagnose.py can run before dependencies are installed


def _local_crew_credentials() -> Path:
    path = Path(os.environ["CREWAI_STORAGE_DIR"]) / "credentials"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


if TokenManager is not None:
    TokenManager._get_secure_storage_path = staticmethod(_local_crew_credentials)
