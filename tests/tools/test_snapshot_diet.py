"""The session snapshot must not carry interactive git tab-completion code.

Git for Windows' ``/etc/profile.d/git-prompt.sh`` + ``git-completion.bash`` define ~85
``_git*`` / ``__git*`` / ``___git*`` functions in EVERY login shell. Capturing them into
``hermes-snap-*.sh`` made the snapshot 75 KB / 2,333 lines — 88% of it completion code a
non-interactive agent command never calls — and every command paid to ``source`` it.

Two changes are covered here:

* ``_capture_functions_script`` filters functions by name with a bash builtin (``[[ =~ ]]``)
  instead of an ``awk`` + ``grep`` pipeline, and drops the git completion namespace while
  keeping every other function (a user's own included).
* ``_export_dump_excluding_session_vars`` no longer wraps its ``unset`` in a ``( … )``
  subshell, so the per-command re-dump does not fork (a subshell costs ~26 ms per command
  on Windows Git Bash) — while still stripping the bridged per-session vars.

The before/after measurements (snapshot bytes, cost of sourcing it, wrapper latency) build
BOTH variants from the same code path, so they stay meaningful on hosts that ship no git
completion at all (those tests skip instead of asserting a difference that cannot exist).
"""

from __future__ import annotations

import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.environments import base_session_env
from tools.environments.base_session_env import (
    _SNAP_TMP_SUFFIX,
    _SNAPSHOT_FUNCTION_EXCLUDED_REGEXES,
    _capture_functions_script,
    _export_dump_excluding_session_vars,
    _snapshot_bootstrap_script,
    _wrap_command_script,
)
from tools.environments.local import LocalEnvironment, _find_bash

GIT_FN_NAMES = (
    "__git_ps1",
    "__git_complete",
    "___git_complete",
    "__git_aliased_command",
    "_git_checkout",
    "_git_complete_command",
)
KEPT_FN_NAMES = (
    "deploy_helper",        # ordinary user function
    "__hermes_local_util",  # double underscore, but not the git namespace
    "git_push_all",         # mentions git, no leading underscore
)
DROPPED_OTHER_NAMES = ("_private_helper",)  # single underscore: dropped, as before

CWD_MARKER = "__HERMES_CWD_diettest__"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bash() -> str:
    """The Git Bash / POSIX bash the terminal backends use (NOT ``bash`` on PATH).

    On Windows a bare ``bash`` resolves to the WSL launcher, which re-parses argv and
    silently drops ``$name`` tokens — every assertion below would then test nothing.
    """
    found = _find_bash()
    if not found:
        pytest.skip("no bash available for the session-snapshot scripts")
    return found


def _bash_path(path: str | Path) -> str:
    """Native path -> the form the shell sees (``C:\\x`` becomes ``/c/x`` on Windows)."""
    text = str(path)
    if sys.platform == "win32":
        from tools.environments.local import _windows_to_msys_path
        text = _windows_to_msys_path(text)
    return text


def _run_bash(script: str, *, login: bool = False) -> subprocess.CompletedProcess:
    args = [_bash(), *(["-l"] if login else []), "-c", script]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _function_names(snapshot_text: str) -> list[str]:
    """``declare -f`` headers (``name ()``) in a snapshot."""
    return [m.group(1) for line in snapshot_text.splitlines() if (m := re.match(r"^([^\s()]+) \(\)", line))]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _captured_names(preamble: str) -> list[str]:
    """Names ``_capture_functions_script`` keeps for the functions *preamble* defines."""
    script = f"{preamble}\n{_capture_functions_script()}\nprintf '%s\\n' \"$__hermes_fns\""
    proc = _run_bash(script)
    assert proc.returncode == 0, proc.stderr
    return sorted(proc.stdout.split())


def _login_shell_git_functions() -> int:
    """How many git completion functions this host's login shell defines (0 = none)."""
    script = (
        "__hermes_n=0; "
        "while IFS= read -r __hermes_d; do "
        "__hermes_f=${__hermes_d##* }; "
        "if [[ $__hermes_f =~ ^_+git ]]; then __hermes_n=$((__hermes_n + 1)); fi; "
        'done <<< "$(declare -F)"; printf "%s" "$__hermes_n"')
    proc = _run_bash(script, login=True)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip() or 0)
    except ValueError:
        return 0


@pytest.fixture
def snapshot_env(tmp_path):
    """A real LocalEnvironment with a live login-shell snapshot (cleaned up after)."""
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    if not Path(env._snapshot_path).exists():
        pytest.skip("login-shell snapshot could not be created on this host")
    try:
        yield env
    finally:
        env.cleanup()


def _bootstrap_snapshot(tmp_path: Path, name: str) -> Path:
    """Run the real bootstrap over *tmp_path/hermes-snap-<name>.sh* and return the file."""
    snap = tmp_path / f"hermes-snap-{name}.sh"
    quoted = shlex.quote(_bash_path(snap))
    script = _snapshot_bootstrap_script(
        quoted_cwd=shlex.quote(_bash_path(tmp_path)),
        quoted_snap=quoted,
        snap_tmp_template=f"{quoted}{_SNAP_TMP_SUFFIX}",
        excluded_names=(),
        cwd_marker=CWD_MARKER)
    proc = _run_bash(script, login=True)
    assert proc.returncode == 0, proc.stderr
    assert snap.exists(), f"bootstrap did not publish {snap}: {proc.stderr}"
    return snap


# ---------------------------------------------------------------------------
# 1. The filter itself: git completion out, everything else in.
# ---------------------------------------------------------------------------

class TestFunctionFilter:
    def test_exclusion_rules_are_the_git_namespace_plus_single_underscore(self):
        """The rule set is the documented contract the shell snippet interpolates."""
        assert _SNAPSHOT_FUNCTION_EXCLUDED_REGEXES == (r"^_[^_]", r"^_+git")

    def test_git_completion_and_single_underscore_helpers_are_dropped(self):
        preamble = "".join(f"{name}() {{ :; }}\n" for name in (*GIT_FN_NAMES, *DROPPED_OTHER_NAMES))
        preamble += "".join(f"{name}() {{ :; }}\n" for name in KEPT_FN_NAMES)

        kept = _captured_names(preamble)

        for dropped in (*GIT_FN_NAMES, *DROPPED_OTHER_NAMES):
            assert dropped not in kept, f"{dropped} must not be captured into the snapshot"
        for name in KEPT_FN_NAMES:
            assert name in kept, f"{name} must survive: only the git namespace is interactive-only"

    def test_empty_declare_output_cannot_fall_back_to_dumping_every_function(self):
        """A blank name must be skipped: bare ``declare -f`` dumps ALL functions.

        ``[ -n "$__hermes_fn" ] || continue`` is what keeps a stray blank line from turning
        ``declare -f $__hermes_fns`` into a full dump (the same bloat in a different form).
        """
        script = "\n".join([
            "real_fn() { :; }",
            '__hermes_fns=""',
            "for __hermes_decl in 'declare -f real_fn' ''; do",
            "  __hermes_fn=${__hermes_decl##* }",
            '  [ -n "$__hermes_fn" ] || continue',
            '  __hermes_fns="$__hermes_fns $__hermes_fn"',
            "done",
            'printf "[%s]" "$__hermes_fns"',
        ])
        proc = _run_bash(script)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "[ real_fn]"

    def test_capture_uses_no_external_process(self):
        """No ``awk``/``grep`` spawn per snapshot: each fork+exec is ~30-80 ms on Git Bash."""
        script = _capture_functions_script()
        assert "awk" not in script and "grep" not in script
        assert "declare -F" in script


# ---------------------------------------------------------------------------
# 2. The real snapshot: no git completion code, user functions intact.
# ---------------------------------------------------------------------------

def test_local_environment_snapshot_has_no_git_completion_functions(snapshot_env):
    text = _read(Path(snapshot_env._snapshot_path))
    names = _function_names(text)
    leaked = [n for n in names if re.match(r"^_+git", n)]
    assert not leaked, f"{len(leaked)} interactive git completion function(s) captured: {leaked[:8]}"
    # No completion bodies either — ``__gitcomp``/``__git_complete`` only exist inside them.
    # (A bare ``__git_ps1`` may still appear inside PS1's env value; that is a prompt, not code.)
    assert "__gitcomp" not in text and "__git_complete" not in text


def test_snapshot_still_captures_a_user_function(tmp_path, monkeypatch):
    """Filtering must not touch the user's own shell functions.

    ``terminal.shell_init_files`` is what a real install uses to add functions; the same
    guard keeps the probe function out of the user's real rc files.
    """
    init_file = tmp_path / "user-fns.sh"
    init_file.write_text("hermes_diet_probe_fn() { printf 'kept\\n'; }\n")
    monkeypatch.setattr(
        "tools.environments.local._read_terminal_shell_init_config",
        lambda: ([str(init_file)], False))

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    try:
        snap = Path(env._snapshot_path)
        if not snap.exists():
            pytest.skip("login-shell snapshot could not be created on this host")
        assert "hermes_diet_probe_fn" in _function_names(_read(snap))
        # And it is usable from a command, i.e. the snapshot really re-defines it.
        assert "kept" in env.execute("hermes_diet_probe_fn")["output"]
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# 3. Measured diet: snapshot size and sourcing cost, before vs after.
# ---------------------------------------------------------------------------

def _legacy_capture() -> str:
    """The pre-change capture: every non-single-underscore function, via awk + grep."""
    return "__hermes_fns=$(declare -F | awk '{print $3}' | grep -vE '^_[^_]') || true"


def test_snapshot_size_and_sourcing_cost_drop_when_git_completion_is_excluded(tmp_path, monkeypatch):
    """Before/after on the same login shell: the 75 KB / 2,333-line snapshot diet.

    Both files come from the real ``_snapshot_bootstrap_script``; only the capture step
    differs (legacy ``awk``/``grep`` filter vs the new builtin one), so the comparison
    isolates the diet from everything else on the host.
    """
    if _login_shell_git_functions() < 10:
        pytest.skip("this host's login shell defines no git completion functions to diet out")

    with monkeypatch.context() as legacy_patch:
        legacy_patch.setattr(base_session_env, "_capture_functions_script", _legacy_capture)
        legacy = _bootstrap_snapshot(tmp_path, "legacy")
    current = _bootstrap_snapshot(tmp_path, "current")

    legacy_bytes, current_bytes = legacy.stat().st_size, current.stat().st_size
    legacy_names = _function_names(_read(legacy))
    current_names = _function_names(_read(current))

    assert any(re.match(r"^_+git", n) for n in legacy_names), "the legacy capture should carry git completion"
    assert not [n for n in current_names if re.match(r"^_+git", n)]
    assert current_bytes < legacy_bytes * 0.5, (
        f"snapshot is {current_bytes} bytes vs {legacy_bytes} before the diet")

    # What a command pays: sourcing the snapshot, measured inside one long-lived shell so
    # bash startup noise cancels out.
    loop = 'for i in $(seq 1 40); do source "$1" >/dev/null 2>&1 || true; done'
    timings: dict[str, list[float]] = {"legacy": [], "current": []}
    for _ in range(2):
        for label, path in (("legacy", legacy), ("current", current)):
            args = [_bash(), "-c", loop, "_", _bash_path(path)]
            start = time.perf_counter()
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            timings[label].append(time.perf_counter() - start)
            assert proc.returncode == 0, proc.stderr
    legacy_s = statistics.median(timings["legacy"])
    current_s = statistics.median(timings["current"])
    assert current_s < legacy_s, (
        f"sourcing 40 snapshots: {current_s * 1000:.0f} ms now vs {legacy_s * 1000:.0f} ms before")


# ---------------------------------------------------------------------------
# 4. Per-command wrapper: no subshell fork, same env contract.
# ---------------------------------------------------------------------------

def _wrapper_for(snapshot: Path, command: str) -> str:
    quoted = shlex.quote(_bash_path(snapshot))
    return _wrap_command_script(
        command,
        quoted_cwd=shlex.quote(_bash_path(snapshot.parent)),
        quoted_snap=quoted,
        snap_tmp_template=f"{quoted}{_SNAP_TMP_SUFFIX}",
        passthrough_names=(),
        snapshot_ready=True,
        cwd_marker=CWD_MARKER)


def test_redump_does_not_fork_a_subshell():
    """The dump's ``unset`` runs in the disposing shell: one less fork per command."""
    script = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
    assert "( unset" not in script
    assert ") || true" not in script
    # Same contract: vars unset first, ``export -p`` dumped, redirect on the group.
    assert script.startswith("{ unset ")
    assert "export -p; }" in script
    assert "|| true" in script  # a readonly var cannot abort the dump


def test_redump_keeps_user_exports_and_still_drops_session_vars(tmp_path):
    """Removing the subshell must not change WHAT lands in the snapshot."""
    snapshot = tmp_path / "hermes-snap-diet.sh"
    snapshot.write_text("", encoding="utf-8")

    proc = _run_bash(_wrapper_for(
        snapshot,
        "export HERMES_DIET_PROBE=kept; export HERMES_SESSION_ID=leaky; export HERMES_RPC_TOKEN=leaky"))
    assert proc.returncode == 0, proc.stderr

    text = _read(snapshot)
    assert 'declare -x HERMES_DIET_PROBE="kept"' in text
    assert "HERMES_SESSION_ID" not in text
    assert "HERMES_RPC_TOKEN" not in text

    # The published file is still a snapshot a command can source.
    sourced = _run_bash(
        f'source {shlex.quote(_bash_path(snapshot))} >/dev/null 2>&1 || true; printf "%s" "$HERMES_DIET_PROBE"')
    assert sourced.returncode == 0, sourced.stderr
    assert sourced.stdout == "kept"


def test_per_command_wrapper_latency_is_not_worse(tmp_path):
    """Interleaved A/B of the per-command wrapper: the subshell removal must not regress."""
    snapshot = tmp_path / "hermes-snap-speed.sh"
    snapshot.write_text("", encoding="utf-8")
    current = _wrapper_for(snapshot, "true")
    legacy = current.replace(
        "{ unset ${!HERMES_SESSION_*}", "{ ( unset ${!HERMES_SESSION_*}", 1,
    ).replace("2>/dev/null || true; export -p; }", "2>/dev/null; export -p; ) || true; }", 1)
    assert "( unset" in legacy and ") || true" in legacy

    times: dict[str, list[float]] = {"legacy": [], "current": []}
    for _ in range(2):  # warm-up (page cache, antivirus on the shell binary)
        for script in (legacy, current):
            _run_bash(script)
    for i in range(7):
        order = (("legacy", legacy), ("current", current)) if i % 2 == 0 else (("current", current), ("legacy", legacy))
        for label, script in order:
            start = time.perf_counter()
            proc = _run_bash(script)
            times[label].append(time.perf_counter() - start)
            assert proc.returncode == 0, proc.stderr

    legacy_ms = statistics.median(times["legacy"]) * 1000
    current_ms = statistics.median(times["current"]) * 1000
    assert current_ms <= legacy_ms * 1.1, (
        f"per-command wrapper regressed: {current_ms:.0f} ms now vs {legacy_ms:.0f} ms before")
