"""Active work uses host-owned registries, never transcripts or OS process scans."""
import json
import sys
import threading

import pytest

from tui_gateway.usage_active_work import active_work, category, load_level, MAX_RECORDS


@pytest.mark.parametrize("count, expected", [(0, "calm"), (1, "busy"), (3, "busy"), (4, "heavy")])
def test_load_thresholds_retain_unknown_categories(count, expected):
    result = load_level({"a": category(count, "test"), "b": category(None, "test")})
    assert result["level"] == expected
    assert result["known_count"] == count
    assert load_level({"a": category(None, "test")})["level"] == "unknown"


def test_registered_subagents_are_counted_without_reading_goals(tmp_path, monkeypatch):
    from tools import delegate_tool_registry as registry
    owner = {"profile_home": str(tmp_path), "running": True, "history_lock": threading.Lock()}
    foreign = {"profile_home": str(tmp_path / "foreign")}
    monkeypatch.setattr(registry, "_active_subagents", {})
    registry._register_subagent({"subagent_id": "child", "owner_session_record": owner, "goal": "fixture-secret"})
    registry._register_subagent({"subagent_id": "foreign", "owner_session_record": foreign, "goal": "fixture-secret"})
    result = active_work(tmp_path, {"owner": owner}, threading.Lock())
    assert result["categories"]["subagents"]["count"] == 1
    assert "fixture-secret" not in json.dumps(result)
    registry._unregister_subagent("child")
    assert active_work(tmp_path, {"owner": owner}, threading.Lock())["categories"]["subagents"]["count"] == 0


def test_cron_execution_owners_are_profile_scoped_and_legacy_is_unknown(tmp_path, monkeypatch):
    from cron import scheduler
    monkeypatch.setattr(scheduler, "_running_job_ids", set())
    monkeypatch.setattr(scheduler, "_running_fire_owners", {
        "job": {object(): ("fixture-secret", tmp_path), object(): (None, tmp_path / "foreign")}})
    snapshot = lambda: active_work(tmp_path, {}, threading.Lock())["categories"]["cron_executions"]
    assert snapshot()["count"] == 1
    monkeypatch.setattr(scheduler, "_running_job_ids", {"unscoped-job"})
    assert snapshot()["status"] == "unknown"
    assert snapshot()["count"] is None


def test_registry_absence_saturation_and_contention_are_not_zero(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "cron.scheduler", raising=False)
    monkeypatch.delitem(sys.modules, "tools.delegate_tool_registry", raising=False)
    lock = threading.Lock()
    with lock:
        result = active_work(tmp_path, {}, lock)
    assert all(c["count"] is None for c in result["categories"].values())
    assert result["load"]["level"] == "unknown"
    crowded = {str(i): {} for i in range(MAX_RECORDS + 1)}
    assert active_work(tmp_path, crowded, lock)["categories"]["agent_turns"]["count"] is None
    assert active_work(tmp_path, {"bad": {"history_lock": lock}}, threading.Lock())["categories"]["agent_turns"]["count"] is None
