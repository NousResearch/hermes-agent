"""Process-start interruption follows the owning tool thread across deadline workers."""

import threading
from types import SimpleNamespace

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment
from tools.interrupt import is_interrupted, set_interrupt


@pytest.mark.parametrize("backend", ["base", "managed"])
@pytest.mark.parametrize("interrupted", [True, False])
def test_interrupt_before_backend_start(monkeypatch, tmp_path, backend, interrupted):
    starts = []

    class Env(BaseEnvironment):
        def _before_execute(self):
            pass

        def _run_bash(self, *_args, **_kwargs):
            starts.append(threading.get_ident())
            return object()

        def _wait_for_process(self, *_args, **_kwargs):
            return {"output": "natural exit", "returncode": 130}

        def cleanup(self):
            pass

    if backend == "base":
        env = Env(cwd=str(tmp_path), timeout=2)
    else:
        env = ManagedModalEnvironment.__new__(ManagedModalEnvironment)
        env.cwd, env.timeout, env._sandbox_id = str(tmp_path), 2, "test"
        monkeypatch.setattr(env, "_prepare_command", lambda command: (command, None))
        monkeypatch.setattr(env, "cleanup", lambda: None)

        def request(*_args, **_kwargs):
            starts.append(threading.get_ident())
            return SimpleNamespace(status_code=200, json=lambda: {
                "status": "completed", "output": "natural exit", "returncode": 130})

        monkeypatch.setattr(env, "_request", request)
    try:
        set_interrupt(interrupted)
        result = env.execute("true")
        assert result["returncode"] == 130
        assert bool(result.get("_process_start_cancelled")) is interrupted
        assert len(starts) == (0 if interrupted else 1)
        if backend == "base" and not interrupted:
            assert starts[0] != threading.get_ident()  # actual deadline-worker seam
        set_interrupt(False)
        fresh = env.execute("true")
        assert not fresh.get("_process_start_cancelled")
        assert len(starts) == (1 if interrupted else 2)
    finally:
        set_interrupt(False)


def test_interrupt_publication_orders_only_its_owned_start():
    import tools.interrupt as interrupts

    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    result = []

    def start():
        entered.set()
        assert release.wait(3)
        return "process"

    owner = threading.Thread(target=lambda: result.append(interrupts.start_if_not_interrupted(start)), daemon=True)
    owner.start()
    assert entered.wait(3)
    publisher = threading.Thread(target=lambda: (set_interrupt(True, owner.ident), published.set()), daemon=True)
    publisher.start()
    try:
        assert not published.wait(0.03)
        # A different session is not serialized behind this process creation.
        assert interrupts.start_if_not_interrupted(lambda: "other") == (True, "other")
        assert not is_interrupted()
    finally:
        release.set()
        owner.join(3)
        publisher.join(3)
        set_interrupt(False, owner.ident)
    assert published.is_set()
    assert result == [(True, "process")]
