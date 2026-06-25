# Copyright 2024 Nous Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for subagent terminal-status visibility (issue #52318).

The TUI /agents overlay polls ``delegation.status`` (backed by
``list_active_subagents``) on an interval.  Before this fix, a child was
removed from the registry the instant it finished, so the overlay never saw a
terminal status and stayed stuck on "running".  These tests pin the registry
contract: after ``_finalize_subagent`` a child briefly appears with its
terminal status, and is then pruned by age and count.
"""

from __future__ import annotations

import pytest

from tools import delegate_tool


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test from process-global subagent registry state."""
    with delegate_tool._active_subagents_lock:
        delegate_tool._active_subagents.clear()
        delegate_tool._recently_finished_subagents.clear()
    yield
    with delegate_tool._active_subagents_lock:
        delegate_tool._active_subagents.clear()
        delegate_tool._recently_finished_subagents.clear()


def _register(sid: str, status: str = "running") -> None:
    delegate_tool._register_subagent(
        {
            "subagent_id": sid,
            "parent_id": None,
            "depth": 0,
            "goal": f"goal-{sid}",
            "model": "test-model",
            "started_at": 0.0,
            "status": status,
            "tool_count": 0,
            "agent": object(),  # stand-in live handle; must never be exposed
        }
    )


def test_running_subagent_is_listed_as_running():
    _register("a")
    active = delegate_tool.list_active_subagents()
    assert len(active) == 1
    assert active[0]["subagent_id"] == "a"
    assert active[0]["status"] == "running"
    # The live agent handle must never leak through the snapshot.
    assert "agent" not in active[0]


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_finalized_subagent_surfaces_terminal_status(status):
    _register("a")
    delegate_tool._finalize_subagent("a", status)

    listed = delegate_tool.list_active_subagents()
    assert len(listed) == 1
    rec = listed[0]
    assert rec["subagent_id"] == "a"
    # The core regression: the overlay's next poll must see the terminal
    # status, not the stale "running" it was registered with.
    assert rec["status"] == status
    assert "finished_at" in rec
    assert "agent" not in rec

    # The child must no longer be in the live set.
    with delegate_tool._active_subagents_lock:
        assert "a" not in delegate_tool._active_subagents
        assert "a" in delegate_tool._recently_finished_subagents


def test_empty_status_defaults_to_completed():
    _register("a")
    delegate_tool._finalize_subagent("a", "")
    rec = delegate_tool.list_active_subagents()[0]
    assert rec["status"] == "completed"


def test_finished_record_is_pruned_after_ttl(monkeypatch):
    _register("a")
    delegate_tool._finalize_subagent("a", "completed")
    assert len(delegate_tool.list_active_subagents()) == 1

    # Advance the clock past the retention window; the next snapshot prunes it.
    real_time = delegate_tool.time.time
    monkeypatch.setattr(
        delegate_tool.time,
        "time",
        lambda: real_time() + delegate_tool._FINISHED_SUBAGENT_TTL_SECONDS + 1.0,
    )
    assert delegate_tool.list_active_subagents() == []


def test_finished_records_are_bounded_by_count(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_FINISHED_SUBAGENT_MAX", 3)
    for i in range(6):
        sid = f"s{i}"
        _register(sid)
        delegate_tool._finalize_subagent(sid, "completed")

    listed = delegate_tool.list_active_subagents()
    assert len(listed) == 3
    # Oldest evicted first; only the most recent survive.
    surviving = {r["subagent_id"] for r in listed}
    assert surviving == {"s3", "s4", "s5"}


def test_active_and_finished_coexist_in_snapshot():
    _register("live")
    _register("done")
    delegate_tool._finalize_subagent("done", "completed")

    listed = {r["subagent_id"]: r["status"] for r in delegate_tool.list_active_subagents()}
    assert listed == {"live": "running", "done": "completed"}


def test_finalize_unknown_subagent_is_noop():
    # No registration: finalize must not raise and must not invent a record.
    delegate_tool._finalize_subagent("ghost", "completed")
    assert delegate_tool.list_active_subagents() == []
