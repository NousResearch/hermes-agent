"""Detached work must retain process admission, including scoped side-agent cleanup."""

import contextvars
from pathlib import Path
import queue
import threading


def test_side_workers_hold_admission_through_cleanup_and_preserve_profile_scope(tmp_path, monkeypatch):
    from hermes_cli import backend_retirement
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret, set_multiplex_active
    from tui_gateway import server

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("terminal:\n  backend: local\n")
        (home / ".env").write_text(f"RETIREMENT_SCOPE={home.name}\n")
    marker = contextvars.ContextVar("retirement_test_marker", default="missing")
    marker.set("inherited")
    emitted = queue.Queue()
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.put(a))
    set_multiplex_active(True)
    try:
        for home in [homes[0], homes[1], homes[0]]:
            entered, release, cleaning, release_cleanup = (threading.Event() for _ in range(4))
            observed = []

            def body():
                observed.append((get_hermes_home(), get_secret("RETIREMENT_SCOPE"), marker.get()))
                entered.set()
                assert release.wait(10)
                return "finished"

            def cleanup():
                cleaning.set()
                assert release_cleanup.wait(10)

            session = {"profile_home": str(home), "cwd": str(tmp_path)}
            try:
                result = server._spawn_side_agent("r", session, "side", "parent", "background.complete", body, cleanup=cleanup)
                assert result["result"]["task_id"] == "side"
                assert entered.wait(10)
                assert fence.prepare() == {"ok": False, "idle": False}
                assert observed == [(home, home.name, "inherited")]
                release.set()
                assert cleaning.wait(10)
                assert fence.prepare() == {"ok": False, "idle": False}
            finally:
                release.set()
                release_cleanup.set()
                # The cleanup event happens before the admission finalizer: join the real worker.
                for thread in threading.enumerate():
                    if thread.name == "side-agent-side":
                        thread.join(10)
            emitted.get(timeout=10)
            token = fence.prepare()["token"]
            assert "error" in server._spawn_side_agent("r", session, "late", "parent", "background.complete", body)
            assert fence.cancel(token) == {"ok": True}
    finally:
        set_multiplex_active(False)
