"""Tests for list_windows normalization (issue #56704)."""

from tools.computer_use.cua_backend import _parse_list_windows_entries


def test_parse_list_windows_skips_null_pid_or_window_id():
    raw = [
        {"app_name": "panel", "pid": None, "window_id": 99, "is_on_screen": True},
        {"app_name": "dock", "pid": 42, "window_id": None, "is_on_screen": True},
        {
            "app_name": "Terminal",
            "pid": 100,
            "window_id": 7,
            "is_on_screen": True,
            "title": "bash",
            "z_index": 3,
        },
    ]
    parsed = _parse_list_windows_entries(raw)
    assert len(parsed) == 1
    assert parsed[0]["app_name"] == "Terminal"
    assert parsed[0]["pid"] == 100
    assert parsed[0]["window_id"] == 7
    assert parsed[0]["title"] == "bash"
    assert parsed[0]["z_index"] == 3
    assert parsed[0]["off_screen"] is False


def test_parse_list_windows_skips_non_numeric_ids():
    raw = [
        {"app_name": "bad", "pid": "nope", "window_id": 1, "is_on_screen": True},
        {"app_name": "ok", "pid": "200", "window_id": "2", "is_on_screen": False},
    ]
    parsed = _parse_list_windows_entries(raw)
    assert len(parsed) == 1
    assert parsed[0]["pid"] == 200
    assert parsed[0]["window_id"] == 2
    assert parsed[0]["off_screen"] is True
