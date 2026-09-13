"""Adjacent-duplicate PATH collapse in the shared bash session snapshot.

Regression coverage for issue #108508. The snapshot's ``export -p`` dump re-emits the inherited
PATH on *every* command, so a polluted parent PATH compounds: each snapshot generation appends
the same entry again, and MSYS bash eventually mis-translates a PATH carrying 70+ duplicates (the
observed case was a single Windows entry with a trailing backslash). ``_PATH_DEDUPE_AWK`` filters
the dump to collapse **adjacent** duplicates only — a PATH may legitimately repeat a directory
non-adjacently for shadowing, so a global unique would change semantics.

Two layers, mirroring ``test_snapshot_session_id_leak.py``:

* the awk program **executed for real** against synthetic ``export -p`` output, so a broken
  escaping edit fails loudly instead of silently emitting a mangled PATH line into a snapshot
  that every later command sources;
* the per-command wrapper, asserted to actually wire the filter into the dump pipeline.

The shell comes from ``tools.environments.local._find_bash()`` rather than ``shutil.which``: on
some Windows hosts ``which bash`` resolves to a WSL relay shim that cannot exec (that is why
``_find_bash`` probes candidates with ``_bash_starts`` instead of trusting ``which``).
"""

from __future__ import annotations

import subprocess

import pytest

from tools.environments.base_session_env import _PATH_DEDUPE_AWK, _wrap_command_script


def _resolve_shell() -> str | None:
    """The same bash the LocalEnvironment uses, or None when this host has no usable one."""
    try:
        from tools.environments.local import _find_bash

        bash = _find_bash()
    except Exception:  # noqa: BLE001 — any failure means "no usable shell here"
        return None
    # `which bash` is not enough on Windows (WSL relay shims exec-fail); confirm awk too, since
    # the filter needs it on PATH inside that shell.
    try:
        probe = subprocess.run([bash, "-c", "command -v awk"], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    return bash if probe.returncode == 0 else None


_BASH = _resolve_shell()
requires_shell = pytest.mark.skipif(_BASH is None, reason="no usable bash+awk on this host")


def _run_awk(stdin: str) -> str:
    """Feed *stdin* through the real filter program and return its stdout."""
    proc = subprocess.run(
        [_BASH, "-c", _PATH_DEDUPE_AWK],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"awk program failed: {proc.stderr!r}"
    return proc.stdout


# ---------------------------------------------------------------------------
# The awk program itself.
# ---------------------------------------------------------------------------

@requires_shell
def test_adjacent_duplicates_are_collapsed():
    out = _run_awk('declare -x PATH="/a:/b:/b:/b:/c"\n')
    assert out == 'declare -x PATH="/a:/b:/c"\n'


@requires_shell
def test_non_adjacent_repeats_survive():
    """A directory may repeat non-adjacently for shadowing; that must not be touched."""
    out = _run_awk('declare -x PATH="/a:/b:/a"\n')
    assert out == 'declare -x PATH="/a:/b:/a"\n'


@requires_shell
def test_real_world_shape_is_collapsed():
    """One entry repeated many times in a row — the shape the pollution bug produced."""
    polluted = "/usr/bin:/dupe:/dupe:/dupe:/dupe:/bin"
    out = _run_awk(f'declare -x PATH="{polluted}"\n')
    assert out == 'declare -x PATH="/usr/bin:/dupe:/bin"\n'


@requires_shell
def test_non_path_lines_are_passed_through_untouched():
    stdin = 'declare -x EDITOR="vim"\ndeclare -x PATH="/a:/a:/b"\ndeclare -x LANG="en_US.UTF-8"\n'
    out = _run_awk(stdin)
    assert out == 'declare -x EDITOR="vim"\ndeclare -x PATH="/a:/b"\ndeclare -x LANG="en_US.UTF-8"\n'


@requires_shell
def test_empty_path_value_is_not_mangled():
    """A PATH of "" must round-trip, not vanish or become a stray line."""
    out = _run_awk('declare -x PATH=""\n')
    assert out == 'declare -x PATH=""\n'


# ---------------------------------------------------------------------------
# The wrapper actually wires it in.
# ---------------------------------------------------------------------------

def _wrap(*, snapshot_ready: bool = True) -> str:
    return _wrap_command_script(
        "echo hi",
        quoted_cwd="/tmp/project",
        quoted_snap="/tmp/snap",
        snap_tmp_template="/tmp/snap.tmp.XXXXXXXXXX",
        passthrough_names=(),
        snapshot_ready=snapshot_ready,
        cwd_marker="__HERMES_CWD__",
    )


def test_wrapper_pipes_the_dump_through_the_filter():
    script = _wrap()
    assert _PATH_DEDUPE_AWK in script
    # The filter must sit between the dump and the temp file, i.e. it edits what gets written.
    assert f"| {_PATH_DEDUPE_AWK} > " in script
    assert script.index(_PATH_DEDUPE_AWK) > script.index("export -p")


def test_wrapper_omits_the_filter_when_there_is_no_snapshot():
    assert _PATH_DEDUPE_AWK not in _wrap(snapshot_ready=False)
