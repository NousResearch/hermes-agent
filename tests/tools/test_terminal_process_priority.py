"""terminal.process_priority: low runs host terminal commands below Hermes' own CPU priority."""

import os
import time

import pytest

from tests.waiting import wait_until
from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry

_NICE = "cat /proc/self/stat | awk '{print $19}'"  # the shell's own niceness, no ps needed


def _write_priority(home, value):
    cfg = home / "config.yaml"
    cfg.write_text(f"terminal:\n  process_priority: {value}\n", encoding="utf-8")
    bump = time.time() + (5 if value == "low" else 0)
    os.utime(cfg, (bump, bump))  # distinct mtime so the cached config read sees the rewrite


def _foreground_nice(tmp_path, monkeypatch):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=20)
    try:
        return int(env.execute(_NICE)["output"].strip().splitlines()[-1])
    finally:
        env.cleanup()


def _background_nice(tmp_path, monkeypatch):
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self: None)
    session = ProcessRegistry().spawn_local(_NICE, cwd=str(tmp_path))
    out = wait_until(lambda: session.output_buffer.strip(), timeout=20,
                     what="background niceness output")
    return int(out.splitlines()[-1])


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("spawn", [_foreground_nice, _background_nice], ids=["foreground", "background"])
def test_low_priority_lowers_terminal_niceness_and_normal_keeps_it(tmp_path, monkeypatch, spawn):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    base = os.nice(0)

    _write_priority(tmp_path, "normal")
    assert spawn(tmp_path, monkeypatch) == base

    _write_priority(tmp_path, "low")
    assert spawn(tmp_path, monkeypatch) == min(19, base + 10)
