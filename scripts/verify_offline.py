"""Run tests with external DNS and network access blocked before imports.

Run after dependency/model provisioning: python scripts/verify_offline.py.
Loopback connections needed by Windows asyncio and FastAPI TestClient are
allowed; internet/network egress is blocked and recorded. This tests for
external network access, not the absence of local interprocess sockets.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
attempts: list[str] = []
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_sendto = socket.socket.sendto


def deny_dns(*args, **kwargs):
    attempts.append("DNS resolution")
    raise AssertionError("Network access forbidden during offline verification")


def is_loopback(address) -> bool:
    """Allow only numeric 127.0.0.0/8 and ::1 addresses for local IPC."""
    if not isinstance(address, tuple) or not address:
        return False
    try:
        return ipaddress.ip_address(address[0]).is_loopback
    except (ValueError, TypeError, IndexError):
        return False


def deny_connect(sock, address):
    if sock.family in {socket.AF_INET, socket.AF_INET6} and not is_loopback(address):
        attempts.append("external TCP/IP connection")
        raise AssertionError("External network access forbidden during offline verification")
    return original_connect(sock, address)


def deny_connect_ex(sock, address):
    if sock.family in {socket.AF_INET, socket.AF_INET6} and not is_loopback(address):
        attempts.append("external TCP/IP connection_ex")
        raise AssertionError("External network access forbidden during offline verification")
    return original_connect_ex(sock, address)


def deny_sendto(sock, *args, **kwargs):
    if sock.family in {socket.AF_INET, socket.AF_INET6}:
        # sendto(data, addr), sendto(data, flags, addr), or address=...
        address = kwargs.get("address") or (args[-1] if args else None)
        if not is_loopback(address):
            attempts.append("external UDP/IP sendto")
            raise AssertionError("External network access forbidden during offline verification")
    return original_sendto(sock, *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Also regenerate all transcripts with egress blocked",
    )
    args, pytest_args = parser.parse_known_args()
    socket.getaddrinfo = deny_dns
    socket.create_connection = deny_dns
    socket.socket.connect = deny_connect
    socket.socket.connect_ex = deny_connect_ex
    socket.socket.sendto = deny_sendto
    import sys
    sys.path.insert(0, str(ROOT))
    import bootstrap
    import pytest

    class TestCounts:
        def __init__(self):
            self.counts = {"passed": 0, "failed": 0, "skipped": 0}

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.counts[report.outcome] += 1

    test_counts = TestCounts()
    started = time.perf_counter()
    result = pytest.main([str(ROOT / "tests"), "-q", *pytest_args], plugins=[test_counts])
    if args.evidence and result == 0:
        from demo import generate_evidence

        generate_evidence()
    # Allow any queued telemetry work to surface an attempted call. One short
    # wait is intentional verification, not an application dependency.
    time.sleep(0.3)
    report = {
        "pytest_exit_code": result,
        "tests": test_counts.counts,
        "evidence_regenerated_with_egress_blocked": bool(args.evidence and result == 0),
        "outbound_attempts": len(attempts),
        "attempt_types": attempts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mode": "MOCK_LLM",
        "result": "passed" if result == 0 and not attempts else "failed",
    }
    destination = ROOT / "reports" / "offline_verification.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(result or int(bool(attempts)))
