"""Oversized terminal commands are staged as scripts, not passed via ``bash -c``.

Git for Windows' bash silently truncates a ``-c`` argument around 8 KiB
(measured on this contract: 8,175 bytes survives, 8,213 is cut mid-token),
corrupting the program text. ``LocalEnvironment._bash_argv`` stages commands
over ``_BASH_ARGV_MAX_SAFE_BYTES`` as a temp script and runs
``bash [-l] <script>`` instead; the script is unlinked as soon as the wait
finishes and the terminal-temp sweep is the hard-kill backstop.
"""

import os
import time
from unittest.mock import patch

import pytest

from tools.environments import local as local_mod
from tools.environments.local import (
    _BASH_ARGV_MAX_SAFE_BYTES,
    _TERMINAL_SCRIPT_PREFIX,
    LocalEnvironment,
)


def _mk_env(tmp_path, monkeypatch):
    """A LocalEnvironment whose temp resolution is pinned to tmp_path."""
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(tmp_path / "term-cache"))
    (tmp_path / "term-cache").mkdir()
    env = LocalEnvironment(cwd=str(tmp_path))
    return env


def _long_cmd(margin):
    """Deterministic command just over the 8 KiB wall whose failure mode is a
    loud, fast corruption: the load-bearing echo sits at the very END, closing
    an inert quoted string, so a truncated ``-c`` argument cuts the quote-closer
    and the echo away (bash: "unexpected EOF while looking for matching quote",
    rc=2 — verified on Git for Windows bash 5.2.37: 8,175 bytes works, 8,300
    truncates). Padding is a no-op ``: "xxx…"`` colon command, inert when whole.
    """
    total = _BASH_ARGV_MAX_SAFE_BYTES + margin
    tail = '"; echo HELLO-$((4000+2))'
    return ': "' + "x" * (total - len(tail) - 3) + tail


def test_short_command_keeps_direct_bash_c(tmp_path, monkeypatch):
    env = _mk_env(tmp_path, monkeypatch)
    args, staged = env._bash_argv("/bin/bash", "echo hi", login=False)
    assert args == ["/bin/bash", "-c", "echo hi"]
    assert staged is None


def test_login_short_command_keeps_bash_lc(tmp_path, monkeypatch):
    env = _mk_env(tmp_path, monkeypatch)
    args, staged = env._bash_argv("/bin/bash", "echo hi", login=True)
    assert args == ["/bin/bash", "-l", "-c", "echo hi"]
    assert staged is None


def test_oversized_command_is_staged_as_script(tmp_path, monkeypatch):
    env = _mk_env(tmp_path, monkeypatch)
    cmd = _long_cmd(300)
    args, staged = env._bash_argv("/bin/bash", cmd, login=False)
    try:
        assert staged is not None and staged.exists()
        assert args == ["/bin/bash", local_mod._bash_safe_path(str(staged))]
        assert staged.read_text(encoding="utf-8") == cmd + "\n"
        assert staged.name.startswith(_TERMINAL_SCRIPT_PREFIX)
    finally:
        staged.unlink(missing_ok=True)


def test_oversized_login_command_stages_with_dash_l(tmp_path, monkeypatch):
    env = _mk_env(tmp_path, monkeypatch)
    cmd = _long_cmd(300)
    args, staged = env._bash_argv("/bin/bash", cmd, login=True)
    try:
        assert args == ["/bin/bash", "-l", local_mod._bash_safe_path(str(staged))]
        assert staged.exists()
    finally:
        staged.unlink(missing_ok=True)


def test_staging_lands_in_managed_cache_and_prune_reclaims(tmp_path, monkeypatch):
    """Scripts stage into the managed terminal cache and the idle sweep reclaims
    stale ones after the grace window while preserving fresh ones (a concurrent
    session's not-yet-spawned script is never swept mid-flight)."""
    env = _mk_env(tmp_path, monkeypatch)
    # Point the managed cache at tmp_path via get_hermes_home, like the
    # tests/tools real-home sandbox does.
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "home")
    cmd = _long_cmd(300)
    _, staged = env._bash_argv("/bin/bash", cmd, login=False)
    assert staged is not None
    cache = tmp_path / "home" / "cache" / "terminal"
    assert staged.parent == cache
    stale = cache / f"{_TERMINAL_SCRIPT_PREFIX}stale.sh"
    stale.write_text("echo stale\n", encoding="utf-8")
    old = time.time() - (local_mod._TERMINAL_SCRIPT_GRACE_S + 3600)
    os.utime(stale, (old, old))
    local_mod.cleanup_terminal_temp_cache()
    assert not stale.exists()
    assert staged.exists()  # fresh: preserved
    staged.unlink(missing_ok=True)


def test_run_bash_unlinks_script_after_wait(tmp_path, monkeypatch):
    """End to end through _run_bash + _wait_for_process: the script exists while
    the command runs and is gone once the wait returns."""
    env = _mk_env(tmp_path, monkeypatch)
    cmd = _long_cmd(300)
    result = env.execute(cmd)
    assert result["returncode"] == 0
    assert "HELLO-4002" in result["output"]
    # nothing left behind in the staging dir
    leftovers = [p for p in (tmp_path / "term-cache").iterdir()
                 if p.name.startswith(_TERMINAL_SCRIPT_PREFIX)]
    assert leftovers == []


def test_wait_unlink_skipped_while_process_still_running(tmp_path, monkeypatch):
    """The yield-to-background return leaves the process RUNNING; the staged
    script must survive (bash may not have opened it yet). The adopted session
    is monitored by the process registry, which never unlinks the script — the
    terminal-temp sweep reclaims it once its grace window passes."""
    from types import SimpleNamespace
    from tools.environments.base import BaseEnvironment

    env = _mk_env(tmp_path, monkeypatch)
    staged = tmp_path / "term-cache" / f"{_TERMINAL_SCRIPT_PREFIX}keep.sh"
    staged.write_text("echo hi\n", encoding="utf-8")
    proc = SimpleNamespace(poll=lambda: None, _hermes_staged_script=str(staged))
    with patch.object(BaseEnvironment, "_wait_for_process", lambda self, p, timeout=120, **kw: {"yielded": True}):
        env._wait_for_process(proc)
    assert staged.exists()  # still running -> not unlinked
    proc.poll = lambda: 0
    with patch.object(BaseEnvironment, "_wait_for_process", lambda self, p, timeout=120, **kw: {}):
        env._wait_for_process(proc)
    assert not staged.exists()  # exited -> unlinked


@pytest.mark.skipif(os.name != "nt", reason="machine-real: Git for Windows -c truncation")
def test_windows_bash_c_8kib_wall_is_real_and_staging_avoids_it(tmp_path, monkeypatch):
    """The hazard this contract exists for, proven on the real Git for Windows
    bash: an ~8 KiB ``-c`` command truncates (no HELLO, nonzero rc), while the
    same command staged as a script runs intact."""
    import subprocess
    bash = local_mod._find_bash()
    # 7,000 + 1,300 = 8,300 bytes — 87 past the measured 8,213 truncation
    # point (8,175 survives). The staging-semantics tests use the 7,000
    # contract floor; only this machine-real proof needs to cross the wall.
    long_cmd = _long_cmd(1300)
    # (1) direct -c: the quote-closing tail is cut -> loud parse error
    direct = subprocess.run([bash, "-c", long_cmd], capture_output=True, text=True, timeout=30)
    assert direct.returncode != 0 and "HELLO" not in direct.stdout, (
        "expected Git for Windows to truncate the ~8KiB -c argument "
        f"(rc={direct.returncode!r} stdout={direct.stdout[:80]!r})")
    # (2) staged script: intact
    env = _mk_env(tmp_path, monkeypatch)
    result = env.execute(long_cmd)
    assert result["returncode"] == 0
    assert "HELLO-4002" in result["output"]
