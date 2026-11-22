import json
from datetime import datetime
from unittest.mock import Mock

import pytest

import hermes_time
from tools import get_time


@pytest.mark.parametrize(("utc", "local"), [
    ("2026-11-22T09:54:49Z", "2026-11-22T04:54:49-05:00"),
    ("2026-07-01T12:00:00Z", "2026-07-01T08:00:00-04:00"),
    ("2026-03-08T06:59:59Z", "2026-03-08T01:59:59-05:00"),
    ("2026-03-08T07:00:00Z", "2026-03-08T03:00:00-04:00"),
    ("2026-11-01T05:30:00Z", "2026-11-01T01:30:00-04:00"),
    ("2026-11-01T06:30:00Z", "2026-11-01T01:30:00-05:00"),
])
def test_explicit_zone_and_single_instant(monkeypatch, utc, local):
    monkeypatch.setattr(hermes_time, "get_timezone_name", lambda: "Asia/Tokyo")
    clock = Mock()
    clock.time.return_value = datetime.fromisoformat(utc).timestamp() + 0.25
    monkeypatch.setattr(get_time, "time", clock)
    result = json.loads(get_time.get_time({"timezone": "America/New_York"}))
    clock.time.assert_called_once_with()
    assert result == {"utc": utc, "local": local, "timezone": "America/New_York"}
    assert datetime.fromisoformat(local) == datetime.fromisoformat(utc)


@pytest.mark.parametrize(("setting", "expected"), [("", "UTC"), ("Asia/Tokyo", "Asia/Tokyo")])
def test_default_zone(tmp_path, monkeypatch, setting, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    if setting:
        (tmp_path / "config.yaml").write_text(f"timezone: {setting}\n", encoding="utf-8")
    result = json.loads(get_time.registry.dispatch("get_time", {}))
    assert result["timezone"] == expected
    if not setting:
        assert result["local"] == result["utc"].replace("Z", "+00:00")


@pytest.mark.parametrize("args", [{}, {"timezone": "Missing/Place"}, {"timezone": "EST"}, {"timezone": "localtime"}, {"timezone": None}])
def test_invalid_zone(monkeypatch, args):
    monkeypatch.setattr(hermes_time, "get_timezone_name", lambda: "Invalid/Default")
    assert json.loads(get_time.get_time(args))["code"] == "invalid_timezone"
