"""``ProcessRegistry.terminate_host_pid`` — the identity-verified tree kill, public.

A plugin that spawns its own long-lived children needs the same kill the
registry uses: it refuses to signal a PID whose start time no longer matches
(PID reuse) and takes the whole tree down on every platform. The public name
is the SAME classmethod as the private spelling, so internal callers and
patches of ``_terminate_host_pid`` are unchanged.
"""

from tools.process_registry import ProcessRegistry


def test_public_name_is_the_private_classmethod():
    assert ProcessRegistry.terminate_host_pid == ProcessRegistry._terminate_host_pid
    assert (ProcessRegistry.__dict__["terminate_host_pid"]
            is ProcessRegistry.__dict__["_terminate_host_pid"])


def test_terminate_host_pid_refuses_a_recycled_pid(monkeypatch):
    import tools.process_registry as pr

    calls = []
    monkeypatch.setattr(ProcessRegistry, "_host_pid_is_ours", classmethod(lambda cls, pid, start: False))
    monkeypatch.setattr(pr.subprocess, "run", lambda *a, **k: calls.append(("run", a)))
    monkeypatch.setattr(pr.os, "kill", lambda *a: calls.append(("kill", a)))
    ProcessRegistry.terminate_host_pid(4242, expected_start=999)
    assert calls == []
