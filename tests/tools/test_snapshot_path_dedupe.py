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


@requires_shell
def test_leading_empty_component_is_preserved():
    """An empty PATH component means the current directory; it must not be dropped.

    The old loop compared each field against ``last`` initialized to ``""``, so a leading ``:``
    compared equal and vanished — changing command resolution rather than deduplicating.
    """
    out = _run_awk('declare -x PATH=":/bin:/usr/bin"\n')
    assert out == 'declare -x PATH=":/bin:/usr/bin"\n'


@requires_shell
def test_adjacent_duplicates_after_a_leading_empty_are_still_collapsed():
    out = _run_awk('declare -x PATH=":/bin:/bin:/usr"\n')
    assert out == 'declare -x PATH=":/bin:/usr"\n'


@requires_shell
def test_interior_and_trailing_empty_components_survive():
    out = _run_awk('declare -x PATH="/a::/b:"\n')
    assert out == 'declare -x PATH="/a::/b:"\n'


@requires_shell
def test_ansi_c_quoted_line_is_passed_through_untouched():
    """Bash emits ANSI-C quoting for values with special characters; the guard must skip them.

    The rewrite strips a leading ``declare -x PATH="`` and a trailing ``"``, neither of which is
    present in ``$'...'`` form — rewriting it would corrupt the value.
    """
    stdin = "declare -x PATH=$'/a:/a:/b\\n'\ndeclare -x EDITOR=\"vim\"\n"
    assert _run_awk(stdin) == stdin


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


def test_wrapper_stages_the_dump_then_filters_it():
    script = _wrap()
    raw, tmp = '"$__hermes_snap_raw"', '"$__hermes_snap_tmp"'
    # The dump lands in its own staging file, and the filter READS that file into the snapshot temp.
    assert f"{_PATH_DEDUPE_AWK} < {raw} > {tmp}" in script
    # Guards the exact shape that silently did nothing: a redirect on the dump's brace group sitting
    # upstream of the pipe, which sends the dump away from awk and points both sides at one file.
    assert f"> {raw} | " not in script
    assert "| awk" not in script
    # Both staging files are cleaned up on EVERY outcome, not just the failure path: the raw dump
    # carries the whole environment, so leaving it behind on success accumulated one per command.
    assert f"; rm -f {tmp} {raw}" in script
    assert f"|| rm -f {tmp} {raw}" not in script


def test_wrapper_omits_the_filter_when_there_is_no_snapshot():
    assert _PATH_DEDUPE_AWK not in _wrap(snapshot_ready=False)


@requires_shell
def test_round_trip_actually_dedupes_the_persisted_path(tmp_path):
    """Run the generated wrapper for real and inspect the snapshot it publishes.

    This is the test the string assertions cannot substitute for: a pipeline whose redirect sits
    upstream of the filter passes every "is the filter wired in?" assertion while the dump never
    reaches awk, leaving duplicates in the snapshot and racing the same file from both sides.
    """
    snap = tmp_path / "snapshot.env"
    script = _wrap_command_script(
        "true",
        quoted_cwd="/tmp",
        quoted_snap=f'"{snap.as_posix()}"',
        snap_tmp_template=f'"{tmp_path.as_posix()}/snap.tmp.XXXXXXXXXX"',
        passthrough_names=(),
        snapshot_ready=True,
        cwd_marker="__HERMES_CWD__",
    )
    runner = tmp_path / "run.sh"
    runner.write_text(
        'export PATH="/usr/bin:/dupe:/dupe:/bin:$PATH"\n'
        "export HERMES_DEDUPE_ROUNDTRIP=kept\n"
        f"{script}\n",
        encoding="utf-8",
        newline="\n",
    )
    proc = subprocess.run([_BASH, str(runner)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr

    assert snap.exists(), "wrapper published no snapshot at all"
    content = snap.read_text(encoding="utf-8", errors="replace")
    assert content.strip(), "snapshot was published empty"

    pathline = next((l for l in content.splitlines() if l.startswith("declare -x PATH=")), "")
    assert pathline, "snapshot dropped PATH entirely"
    assert "/dupe:/dupe" not in pathline, f"adjacent duplicates survived: {pathline[:160]}"

    # Not just a PATH-only file either: the rest of the environment must round-trip.
    assert "HERMES_DEDUPE_ROUNDTRIP" in content


@requires_shell
def test_successful_round_trip_leaves_no_staging_files(tmp_path):
    """The raw dump holds the whole environment; it must not survive a successful publish.

    The success chain used to move only the filtered temp into place and clean the raw staging file
    on the failure path alone, so every command left one complete ``export -p`` dump (secrets
    included) beside the snapshot.
    """
    snap = tmp_path / "snapshot.env"
    script = _wrap_command_script(
        "true",
        quoted_cwd="/tmp",
        quoted_snap=f'"{snap.as_posix()}"',
        snap_tmp_template=f'"{tmp_path.as_posix()}/snap.tmp.XXXXXXXXXX"',
        passthrough_names=(),
        snapshot_ready=True,
        cwd_marker="__HERMES_CWD__",
    )
    runner = tmp_path / "run.sh"
    runner.write_text(
        'export PATH="/dupe:/dupe:$PATH"\n'
        f"{script}\n",
        encoding="utf-8",
        newline="\n",
    )
    proc = subprocess.run([_BASH, str(runner)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr

    assert snap.exists(), "wrapper published no snapshot"
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("snap.tmp."))
    assert leftovers == [], f"staging files left behind after a successful run: {leftovers}"
