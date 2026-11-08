from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from guardrails import guard_input, mask_pii


class AuditLog:
    """One atomic JSONL append per event; no raw HTTP bodies or exception values."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **event) -> None:
        event["timestamp"] = datetime.now(UTC).isoformat()
        if "query" in event:
            event["query"] = guard_input(str(event["query"])).text[:2000]
        # Defense in depth for all textual fields, including trace/path values.
        safe = json.loads(json.dumps(event, default=str))
        for key, value in safe.items():
            if isinstance(value, str):
                safe[key] = mask_pii(value)
        line = json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
