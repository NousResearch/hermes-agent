"""Tests for the shared login-shell PATH restore (#56634).

A Debian-family ``/etc/profile`` hardcode-resets PATH before sourcing
``/etc/profile.d/*.sh``, so ``init_session``'s login-shell ``export -p`` would
snapshot a PATH without the image/host venv. ``BaseEnvironment`` repairs it: it
probes the ambient *non-login* PATH (which never sourced ``/etc/profile``) and,
before ``export -p``, prepends the entries the login shell dropped. These tests
drive the merge shell and probe directly (no Docker), plus the full
``init_session`` snapshot chain against a simulated Debian login shell.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess

import pytest

from tools.environments.base import BaseEnvironment

BASH = shutil.which("bash")
requires_bash = pytest.mark.skipif(BASH is None, reason="bash not available")

# Debian's non-root login PATH, set before profile.d is sourced.
DEBIAN_RESET_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games"
VENV_PREFIX = "/opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin"


def _run_restore(ambient: str, login_path: str, prelude: str = "") -> str:
    """Render the restore snippet, run it atop *login_path*, echo the result."""
    snippet = BaseEnvironment._render_login_path_restore(ambient)
    script = f"{prelude}PATH={shlex.quote(login_path)}\n{snippet}printf '%s' \"${{PATH-}}\"\n"
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout


# ----- merge shell behavior -----


def test_render_is_noop_without_ambient() -> None:
    assert BaseEnvironment._render_login_path_restore(None) == ""
    assert BaseEnvironment._render_login_path_restore("") == ""


@requires_bash
def test_restores_venv_dropped_by_debian_reset() -> None:
    out = _run_restore(f"{VENV_PREFIX}:/usr/bin:/bin", DEBIAN_RESET_PATH)
    assert out == f"{VENV_PREFIX}:{DEBIAN_RESET_PATH}"
    entries = out.split(":")
    assert entries[0] == "/opt/hermes/bin"
    assert entries.index("/opt/hermes/.venv/bin") < entries.index("/usr/bin")


@requires_bash
def test_noop_when_nothing_missing() -> None:
    login = "/usr/local/bin:/usr/bin:/bin"
    assert _run_restore("/usr/bin:/bin", login) == login


@requires_bash
def test_empty_login_path_introduces_no_current_dir_entry() -> None:
    out = _run_restore("/opt/hermes/.venv/bin:/usr/bin", "")
    assert out == "/opt/hermes/.venv/bin:/usr/bin"
    assert not out.startswith(":") and not out.endswith(":") and "::" not in out


@requires_bash
def test_unset_login_path_is_handled() -> None:
    # Genuinely unset (not just empty) -- ${PATH-} must not trip nounset.
    snippet = BaseEnvironment._render_login_path_restore("/opt/hermes/.venv/bin:/usr/bin")
    script = f"unset PATH\n{snippet}/usr/bin/printf '%s' \"${{PATH-}}\"\n"
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "/opt/hermes/.venv/bin:/usr/bin"


@requires_bash
def test_dedupes_repeated_ambient_entries() -> None:
    assert _run_restore("/x:/x:/y", "/z") == "/x:/y:/z"


@requires_bash
def test_dedup_treats_glob_chars_literally() -> None:
    # The dedup `case` pattern quotes the entry, so glob chars (* ? [ ]) match
    # literally: "/opt/x*z" must NOT be treated as already-present just because
    # it globs onto "/opt/xyz". (A real duplicate still dedupes.)
    assert _run_restore("/opt/xyz:/opt/x*z", "") == "/opt/xyz:/opt/x*z"
    assert _run_restore("/opt/a?c:/opt/[ab]", "/usr/bin") == "/opt/a?c:/opt/[ab]:/usr/bin"
    assert _run_restore("/o*t:/o*t", "/z") == "/o*t:/z"


@requires_bash
def test_skips_empty_ambient_entries() -> None:
    assert _run_restore(":/opt/v::/usr/bin:", "/bin") == "/opt/v:/usr/bin:/bin"


@requires_bash
def test_preserves_path_entry_with_spaces() -> None:
    assert _run_restore("/opt/my venv/bin:/usr/bin", "/bin") == "/opt/my venv/bin:/usr/bin:/bin"


@requires_bash
def test_survives_set_eu_and_leaves_flags_enabled() -> None:
    # A login profile may leave set -e/-u active: the merge must not abort the
    # bootstrap and, since it no longer toggles options, must leave both on.
    snippet = BaseEnvironment._render_login_path_restore(f"{VENV_PREFIX}:/usr/bin")
    script = (
        "set -e\nset -u\n"
        f"PATH={shlex.quote(DEBIAN_RESET_PATH)}\n"
        f"{snippet}"
        'printf "%s|" "${PATH-}"; case $- in *e*) printf E;; esac; case $- in *u*) printf U;; esac\n'
    )
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    path, flags = r.stdout.split("|")
    assert path.split(":")[0] == "/opt/hermes/bin"
    assert flags == "EU"


# ----- ambient-PATH probe -----


class _ProbeEnv(BaseEnvironment):
    """Concrete env with canned _run_bash/_wait_for_process for probe tests."""

    def __init__(self, output: str = "", returncode: int = 0, raise_exc=None):
        self._canned = {"output": output, "returncode": returncode}
        self._raise = raise_exc
        self.ran = False
        super().__init__(cwd="/tmp", timeout=10)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self.ran = True
        self._last_login = login
        self._last_cmd = cmd_string
        if self._raise is not None:
            raise self._raise
        return object()

    def _wait_for_process(self, proc, timeout=120):
        return self._canned

    def cleanup(self):
        pass


def _marker(env: _ProbeEnv) -> str:
    return f"__HERMES_PATH_{env._session_id}__"


def test_probe_disabled_does_not_even_spawn() -> None:
    env = _ProbeEnv()
    env._restore_ambient_login_path = False
    assert env._capture_ambient_login_path() is None
    assert env.ran is False


def test_probe_parses_marker_framed_value_ignoring_noise() -> None:
    env = _ProbeEnv()
    m = _marker(env)
    path = "/opt/hermes/.venv/bin:/usr/bin"
    env._canned = {"output": f"BASH_ENV noise\n{m}{path}{m}", "returncode": 0}
    assert env._capture_ambient_login_path() == path
    assert env._last_login is False
    assert "builtin printf" in env._last_cmd and "${PATH-}" in env._last_cmd


def test_probe_returns_value_verbatim_without_stripping() -> None:
    env = _ProbeEnv()
    m = _marker(env)
    path = "  /leading:/trailing/dir  "  # surrounding spaces are legal, must survive
    env._canned = {"output": f"{m}{path}{m}", "returncode": 0}
    assert env._capture_ambient_login_path() == path


def test_probe_uses_output_key_not_stdout() -> None:
    env = _ProbeEnv()
    m = _marker(env)
    env._canned = {"output": f"{m}/usr/bin{m}", "returncode": 0, "stdout": "WRONG"}
    assert env._capture_ambient_login_path() == "/usr/bin"


def test_probe_nonzero_returncode_returns_none() -> None:
    env = _ProbeEnv()
    m = _marker(env)
    env._canned = {"output": f"{m}/usr/bin{m}", "returncode": 1}
    assert env._capture_ambient_login_path() is None


def test_probe_missing_marker_returns_none() -> None:
    assert _ProbeEnv(output="no markers", returncode=0)._capture_ambient_login_path() is None


def test_probe_empty_path_returns_none() -> None:
    env = _ProbeEnv()
    m = _marker(env)
    env._canned = {"output": f"{m}{m}", "returncode": 0}
    assert env._capture_ambient_login_path() is None


def test_probe_exception_returns_none() -> None:
    assert _ProbeEnv(raise_exc=RuntimeError("boom"))._capture_ambient_login_path() is None


@requires_bash
def test_builtin_printf_defeats_shadowed_printf() -> None:
    # The probe uses `builtin printf` so a printf shadowed by a function (which a
    # non-login shell can inherit via an exported function/BASH_ENV) cannot make
    # it capture the format string instead of $PATH.
    marker, path = "__M__", "/opt/hermes/.venv/bin:/usr/bin"
    shadow = 'printf() { builtin printf "%s" "$1"; }\n'
    fmt = f'"{marker}%s{marker}" {shlex.quote(path)}'
    naive = subprocess.run([BASH, "-c", shadow + f"printf {fmt}"], capture_output=True, text=True)
    safe = subprocess.run([BASH, "-c", shadow + f"builtin printf {fmt}"], capture_output=True, text=True)
    assert naive.stdout != f"{marker}{path}{marker}"  # shadowed printf corrupts it
    assert safe.stdout == f"{marker}{path}{marker}"  # builtin is immune


# ----- full init_session snapshot chain -----


class _FakeDebianEnv(BaseEnvironment):
    """A login shell whose /etc/profile resets PATH; non-login keeps the venv."""

    def __init__(self, ambient: str, reset: str):
        self._ambient = ambient
        self._reset = reset
        super().__init__(cwd="/tmp", timeout=10)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        env = dict(os.environ, PATH=self._ambient)
        script = f"export PATH={shlex.quote(self._reset)}\n{cmd_string}" if login else cmd_string
        return subprocess.run(
            [BASH, "-c", script], capture_output=True, text=True, env=env, timeout=timeout
        )

    def _wait_for_process(self, proc, timeout=120):
        return {"output": proc.stdout, "returncode": proc.returncode}

    def cleanup(self):
        pass


@requires_bash
def test_init_session_snapshot_restores_venv_end_to_end() -> None:
    env = _FakeDebianEnv(ambient=f"{VENV_PREFIX}:/usr/bin:/bin", reset=DEBIAN_RESET_PATH)
    try:
        env.init_session()
        assert env._snapshot_ready
        r = subprocess.run(
            [BASH, "-c", f"source {shlex.quote(env._snapshot_path)} >/dev/null 2>&1; printf '%s' \"$PATH\""],
            capture_output=True, text=True, timeout=30,
        )
        entries = r.stdout.split(":")
        assert entries[0] == "/opt/hermes/bin"
        assert entries.index("/opt/hermes/.venv/bin") < entries.index("/usr/bin")
    finally:
        for p in (env._snapshot_path, env._cwd_file):
            try:
                os.unlink(p)
            except OSError:
                pass


def test_local_env_gates_restore_on_windows() -> None:
    from tools.environments.local import _IS_WINDOWS, LocalEnvironment

    assert LocalEnvironment._restore_ambient_login_path == (not _IS_WINDOWS)
