"""Interrupting ``a; b`` must never let ``b`` start (pi #11252, macOS).

macOS signals a process group's members one by one, so killing the foreground ``sleep`` can wake
the waiting shell before the shell itself is signalled; the shell then runs the next command of
the line during cancellation. Real processes, real interrupt path, under CPU load. Thirty background
members widen the window (macOS signals each one separately): on a macos-latest runner main let
`b` run in 22, 62 and 70 of 200 interrupts with this shape, the fixed head in 0 of 600.
"""
import contextlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.waiting import wait_until
from tools import interrupt
from tools.environments.local import LocalEnvironment


def _burn(stop):
    while not stop.is_set():
        sum(i * i for i in range(20000))


def _interrupt_once(work: str, n: int) -> bool:
    marker = os.path.join(work, f"leak-{n}")
    env = LocalEnvironment(cwd=work, timeout=60)
    tid: list[int] = []
    done = threading.Event()

    def run():
        tid.append(threading.get_ident())
        try:
            env.execute("sleep 30 & " * 30 + f"sleep 30; echo leaked > {marker}", timeout=60)
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    wait_until(lambda: tid, timeout=10)
    # The interrupt must land while `sleep 30` is the shell's foreground child, i.e. mid-command.
    time.sleep(0.2)  # health: allow HX013 -- the delay before the stop IS the scenario under test
    interrupt.set_interrupt(True, tid[0])
    try:
        assert done.wait(30), "the interrupted command never returned"
    finally:
        interrupt.set_interrupt(False, tid[0])
    # A leaked `echo` runs right after the shell wakes; absence is only provable after a window.
    time.sleep(0.3)  # health: allow HX013 -- proving a file is NOT written needs a bounded wait
    return os.path.exists(marker)


@pytest.mark.platforms("macos")
def test_interrupt_never_runs_the_next_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    stop = multiprocessing.Event()
    burners = [multiprocessing.Process(target=_burn, args=(stop,), daemon=True)
               for _ in range(os.cpu_count() or 4)]
    for proc in burners:
        proc.start()
    try:
        with ThreadPoolExecutor(6) as pool:
            leaked = sum(pool.map(lambda n: _interrupt_once(str(tmp_path), n), range(80)))
    finally:
        stop.set()
        for proc in burners:
            with contextlib.suppress(Exception):
                proc.join(5)
    assert leaked == 0, f"{leaked}/80 interrupts let the next command of `a; b` run"
