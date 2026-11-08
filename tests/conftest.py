from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest

from config import Settings
from service import SupportService


@pytest.fixture
def settings(tmp_path):
    return Settings(runtime=tmp_path / "runtime", admin_token="test-admin-token")


@pytest.fixture
def service(settings):
    return SupportService(settings)
