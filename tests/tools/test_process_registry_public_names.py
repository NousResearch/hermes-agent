"""``ProcessRegistry.terminate_host_pid`` — the identity-verified tree kill, public.

A plugin that spawns its own long-lived children needs the same kill the
registry uses: it refuses to signal a PID whose start time no longer matches
(PID reuse) and takes the whole tree down on every platform. The private
spelling ``_terminate_host_pid`` stays as an alias bound to the same
classmethod; every internal caller reads the public name, so overriding the
public name is what takes effect.
"""

import inspect
from unittest.mock import patch

from tools.process_registry import ProcessRegistry


def test_private_spelling_is_an_alias_of_the_public_classmethod():
    assert ProcessRegistry.terminate_host_pid == ProcessRegistry._terminate_host_pid
    assert (inspect.getattr_static(ProcessRegistry, "terminate_host_pid")
            is inspect.getattr_static(ProcessRegistry, "_terminate_host_pid"))


def test_terminate_host_pid_refuses_a_recycled_pid(monkeypatch):
    import tools.process_registry_termination as prt

    calls = []
    monkeypatch.setattr(ProcessRegistry, "_host_pid_is_ours", classmethod(lambda cls, pid, start: False))
    monkeypatch.setattr(prt.subprocess, "run", lambda *a, **k: calls.append(("run", a)))
    monkeypatch.setattr(prt.os, "kill", lambda *a: calls.append(("kill", a)))
    ProcessRegistry.terminate_host_pid(4242, expected_start=999)
    assert calls == []


def test_overriding_the_public_name_reaches_the_lightpanda_tree_kill():
    from tools import browser_lightpanda

    with patch.object(ProcessRegistry, "terminate_host_pid") as kill:
        browser_lightpanda._tree_kill(4242, 999)
    kill.assert_called_once_with(4242, expected_start=999)


def test_overriding_the_public_name_reaches_the_browser_daemon_reaper(monkeypatch):
    from tools import browser_tool_lifecycle

    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 777)
    with patch.object(ProcessRegistry, "terminate_host_pid") as kill:
        assert browser_tool_lifecycle._terminate_verified_daemon(4242, "s", lambda *a: None) is True
    kill.assert_called_once_with(4242, 777)

