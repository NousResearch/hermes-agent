"""``post_memory_write``: one observer event per persisted change to the built-in memory
files, from every write path — the memory tool, approved staged replays (``/memory
approve`` and plugins that call ``apply_memory_pending`` directly), and direct store use.
An approved replay names the pending record it applied, so an observer can attribute a
change that happened long after the tool call that staged it (#144979)."""

import os
import shutil
import tempfile

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_mem_hook_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def events(monkeypatch):
    calls = []

    def invoke_hook(name, **kwargs):
        if name == "post_memory_write":
            calls.append(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "post_memory_write")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    return calls


@pytest.fixture
def store(hermes_home):
    from tools.memory_tool import MemoryStore
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.load_from_disk()
    return s


def test_registered_as_a_valid_hook():
    from hermes_cli.plugins import VALID_HOOKS
    assert "post_memory_write" in VALID_HOOKS


def test_each_store_write_reports_its_action_and_entries(store, events):
    store.add("memory", "uses uv")
    store.patch("memory", "uv", "uv 0.13")
    store.add("memory", "tabs over spaces")
    store.remove("memory", "tabs")
    store.apply_batch("user", [{"action": "add", "content": "name: Ada"}])

    assert [(e["action"], e["target"], e["content"]) for e in events] == [
        ("add", "memory", "uses uv"),
        ("patch", "memory", "uv 0.13"),
        ("add", "memory", "tabs over spaces"),
        ("remove", "memory", None),
        ("batch", "user", None),
    ]
    assert events[1]["entries_before"] == ["uses uv"]
    assert events[1]["entries_after"] == ["uses uv 0.13"]
    assert events[3]["entries_before"] == ["uses uv 0.13", "tabs over spaces"]
    assert events[3]["entries_after"] == ["uses uv 0.13"]
    assert all(e["pending_id"] is None for e in events)


def test_no_event_when_nothing_was_written(store, events):
    store.add("memory", "uses uv")
    events.clear()
    store.add("memory", "uses uv")  # duplicate: unchanged file
    store.remove("memory", "no such entry")  # refused
    store.add("memory", "x" * 600)  # over the limit
    store.resolve_batch_entries("memory", [{"action": "remove", "old_text": "uv"}])  # dry run
    assert events == []


def test_memory_tool_write_has_no_pending_id(store, events):
    from tools.memory_tool import memory_tool
    memory_tool(operations=[{"action": "add", "content": "prefers short replies"}], target="user", store=store)
    assert len(events) == 1
    assert events[0]["action"] == "batch" and events[0]["pending_id"] is None
    assert events[0]["entries_after"] == ["prefers short replies"]


def test_approved_staged_write_names_its_pending_record(store, events):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    record = wa.stage_write("memory", {"action": "add", "target": "user", "content": "lives in Lisbon"},
                            summary="add", origin="foreground")
    assert events == []  # staging writes nothing to MEMORY.md / USER.md

    handle_pending_subcommand(wa.MEMORY, ["approve", record["id"]], memory_store=store)

    assert len(events) == 1
    assert events[0]["pending_id"] == record["id"]
    assert events[0]["entries_after"] == ["lives in Lisbon"]


def test_plugin_replay_through_apply_memory_pending_names_its_pending_record(hermes_home, events):
    """The memory-review dashboard replays a staged payload with a fresh on-disk store."""
    from tools import write_approval as wa
    from tools.memory_tool import apply_memory_pending, load_on_disk_store
    record = wa.stage_write("memory", {"action": "batch", "target": "memory",
                                       "operations": [{"action": "add", "content": "deploys on Fridays"}]},
                            summary="batch", origin="background_review")

    result = apply_memory_pending(wa.get_pending(wa.MEMORY, record["id"])["payload"], load_on_disk_store())

    assert result["success"]
    assert [(e["action"], e["pending_id"]) for e in events] == [("batch", record["id"])]


def test_replay_id_does_not_leak_into_later_writes(store, events):
    from tools import write_approval as wa
    from tools.memory_tool import apply_memory_pending
    record = wa.stage_write("memory", {"action": "add", "target": "memory", "content": "a"},
                            summary="a", origin="foreground")
    apply_memory_pending(wa.get_pending(wa.MEMORY, record["id"])["payload"], store)
    store.add("memory", "b")
    assert [e["pending_id"] for e in events] == [record["id"], None]


def test_a_failing_observer_never_breaks_the_write(store, monkeypatch):
    def boom(name, **kwargs):
        raise RuntimeError("observer crashed")

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", boom)
    result = store.add("memory", "still saved")
    assert result["success"]
    assert store.memory_entries == ["still saved"]
