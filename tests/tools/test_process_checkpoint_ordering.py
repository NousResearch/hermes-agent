"""Audit: older concurrent checkpoint must not remove a live process on recovery."""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.process_registry import ProcessRegistry, ProcessSession


def test_checkpoint_concurrent_updates_recover_all_live_processes(
    tmp_path, monkeypatch
):
    import utils
    import tools.process_registry as pr

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr, "_checkpoint_path", lambda: checkpoint)
    registry = ProcessRegistry()
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            creationflags=windows_hide_flags(),
        )
        for _ in range(2)
    ]
    old_write_entered, release_old, newer_done = (threading.Event() for _ in range(3))
    real_write = utils.atomic_json_write

    def ordered_write(path, entries, **kwargs):
        if [item["session_id"] for item in entries] == ["first"]:
            old_write_entered.set()
            assert release_old.wait(15)
        return real_write(path, entries, **kwargs)

    monkeypatch.setattr(utils, "atomic_json_write", ordered_write)

    def add_session(index, name):
        with registry._lock:
            registry._running[name] = ProcessSession(
                id=name,
                command="audit sleeping child",
                pid=processes[index].pid,
                task_id="audit",
                host_start_time=registry._safe_host_start_time(processes[index].pid),
            )
        registry._write_checkpoint()
        if index:
            newer_done.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            old = pool.submit(add_session, 0, "first")
            assert old_write_entered.wait(10)
            newer = pool.submit(add_session, 1, "second")
            # Allow the competing write to settle if writes are not serialized, then
            # release the old I/O. A serialized implementation instead queues it.
            newer_done.wait(3)
            release_old.set()
            old.result(10)
            newer.result(10)
        restored = ProcessRegistry()
        assert restored.recover_from_checkpoint() == 2
        assert set(restored._running) == {"first", "second"}
    finally:
        release_old.set()
        for process in processes:
            process.terminate()
            process.wait(10)
