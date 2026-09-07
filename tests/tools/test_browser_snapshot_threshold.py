"""Behavior tests for config-driven browser snapshot thresholds."""

import json
import os
from unittest.mock import Mock

import pytest

from hermes_cli.config import DEFAULT_CONFIG
from tools import browser_camofox, browser_tool
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session


@pytest.fixture(autouse=True)
def isolated_snapshot_threshold(tmp_path, monkeypatch):
    """Use a real, isolated config file and reset module-level caches."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # hermes_cli.config.read_raw_config caches on (mtime_ns, size); on
    # filesystems with coarse/multigrain timestamps two rapid same-size
    # rewrites can share one cache key and serve stale content, which is
    # orthogonal to the browser-tool cache lifecycle under test here.
    from hermes_cli import config as _config_mod

    _config_mod._RAW_CONFIG_CACHE.clear()

    original_cached = browser_tool._cached_snapshot_threshold
    original_resolved = browser_tool._snapshot_threshold_resolved
    browser_tool._cached_snapshot_threshold = None
    browser_tool._snapshot_threshold_resolved = False
    yield tmp_path
    browser_tool._cached_snapshot_threshold = original_cached
    browser_tool._snapshot_threshold_resolved = original_resolved
    _config_mod._RAW_CONFIG_CACHE.clear()


_MTIME_FLOOR: dict = {}


def _write_threshold(hermes_home, value):
    path = hermes_home / "config.yaml"
    path.write_text(
        f"browser:\n  snapshot_threshold: {value}\n",
        encoding="utf-8",
    )
    # read_raw_config caches on (mtime_ns, size); on filesystems with
    # coarse/multigrain timestamps two rapid same-size rewrites can share one
    # cache key and serve stale content. Force a strictly increasing,
    # fine-grained mtime so the cache contract is what this test actually
    # exercises (a plain "+1ms" bump is not enough: each rewrite first
    # resets mtime to the same coarse tick).
    st = path.stat()
    key = str(path)
    new_mtime = max(st.st_mtime_ns, _MTIME_FLOOR.get(key, 0)) + 1_000_000_000
    os.utime(path, ns=(st.st_atime_ns, new_mtime))
    _MTIME_FLOOR[key] = new_mtime


def _long_snapshot(chars: int) -> str:
    line = "button [ref=e1] example content\n"
    return line * ((chars // len(line)) + 2)


def test_default_matches_browser_config(isolated_snapshot_threshold):
    assert browser_tool.get_browser_snapshot_threshold() == (
        DEFAULT_CONFIG["browser"]["snapshot_threshold"]
    )


def test_reads_profile_config_override(isolated_snapshot_threshold):
    _write_threshold(isolated_snapshot_threshold, 30000)

    assert browser_tool.get_browser_snapshot_threshold() == 30000


def test_clamps_small_values_to_safe_floor(isolated_snapshot_threshold):
    _write_threshold(isolated_snapshot_threshold, 10)

    assert browser_tool.get_browser_snapshot_threshold() == (
        browser_tool.MIN_SNAPSHOT_THRESHOLD
    )


def test_invalid_values_fall_back_to_default(isolated_snapshot_threshold):
    _write_threshold(isolated_snapshot_threshold, "not-a-number")

    assert browser_tool.get_browser_snapshot_threshold() == (
        browser_tool.DEFAULT_SNAPSHOT_THRESHOLD
    )


def test_cleanup_reloads_updated_profile_config(isolated_snapshot_threshold):
    _write_threshold(isolated_snapshot_threshold, 12000)
    assert browser_tool.get_browser_snapshot_threshold() == 12000

    _write_threshold(isolated_snapshot_threshold, 15001)
    assert browser_tool.get_browser_snapshot_threshold() == 12000

    bt_lifecycle.cleanup_all_browsers()
    assert browser_tool.get_browser_snapshot_threshold() == 15001


def test_browser_snapshot_applies_profile_threshold(
    isolated_snapshot_threshold,
    monkeypatch,
):
    _write_threshold(isolated_snapshot_threshold, 1000)
    snapshot = _long_snapshot(1500)

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt_cloud, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(
        bt_session,
        "_run_browser_command",
        lambda *args, **kwargs: {
            "success": True,
            "data": {"snapshot": snapshot, "refs": {"e1": {}}},
        },
    )

    result = json.loads(browser_tool.browser_snapshot(task_id="threshold-test"))

    assert result["success"] is True
    assert len(result["snapshot"]) < len(snapshot)
    assert "more lines truncated" in result["snapshot"]


def test_browser_navigation_applies_profile_threshold(
    isolated_snapshot_threshold,
    monkeypatch,
):
    _write_threshold(isolated_snapshot_threshold, 1000)
    snapshot = _long_snapshot(1500)
    task_id = "threshold-navigate-test"

    monkeypatch.setattr(bt_cloud, "_is_local_backend", lambda: True)
    monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(
        bt_session,
        "_get_session_info",
        lambda session_key: {
            "session_name": "threshold-test",
            "_first_nav": False,
            "features": {"local": True, "proxies": True},
        },
    )
    monkeypatch.setattr(
        bt_session,
        "_run_browser_command",
        Mock(
            side_effect=[
                {
                    "success": True,
                    "data": {
                        "title": "Example",
                        "url": "https://example.com/",
                    },
                },
                {
                    "success": True,
                    "data": {"snapshot": snapshot, "refs": {"e1": {}}},
                },
            ]
        ),
    )

    result = json.loads(
        browser_tool.browser_navigate(
            "https://example.com",
            task_id=task_id,
        )
    )
    browser_tool._last_active_session_key.pop(task_id, None)

    assert result["success"] is True
    assert len(result["snapshot"]) < len(snapshot)
    assert "more lines truncated" in result["snapshot"]


def test_camofox_navigation_applies_same_profile_threshold(
    isolated_snapshot_threshold,
    monkeypatch,
):
    _write_threshold(isolated_snapshot_threshold, 1000)
    snapshot = _long_snapshot(1500)
    session = {"tab_id": "tab-1", "user_id": "user-1"}

    monkeypatch.setattr(
        browser_camofox,
        "_rewrite_loopback_url_for_camofox",
        lambda url: (url, None),
    )
    monkeypatch.setattr(browser_camofox, "_get_session", lambda task_id: session)
    monkeypatch.setattr(
        browser_camofox,
        "_post",
        lambda *args, **kwargs: {"url": "https://example.com", "title": "Example"},
    )
    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda *args, **kwargs: {"snapshot": snapshot, "refsCount": 1},
    )
    monkeypatch.setattr(browser_camofox, "get_vnc_url", lambda: None)

    result = json.loads(
        browser_camofox.camofox_navigate(
            "https://example.com",
            task_id="threshold-test",
        )
    )

    assert result["success"] is True
    assert len(result["snapshot"]) < len(snapshot)
    assert "more lines truncated" in result["snapshot"]


def test_camofox_snapshot_applies_same_profile_threshold(
    isolated_snapshot_threshold,
    monkeypatch,
):
    _write_threshold(isolated_snapshot_threshold, 1000)
    snapshot = _long_snapshot(1500)
    session = {"tab_id": "tab-1", "user_id": "user-1"}

    monkeypatch.setattr(browser_camofox, "_get_session", lambda task_id: session)
    monkeypatch.setattr(
        browser_camofox,
        "_camofox_private_page_block",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda *args, **kwargs: {"snapshot": snapshot, "refsCount": 1},
    )

    result = json.loads(
        browser_camofox.camofox_snapshot(task_id="threshold-snapshot-test")
    )

    assert result["success"] is True
    assert len(result["snapshot"]) < len(snapshot)
    assert "more lines truncated" in result["snapshot"]
