"""Behavioural regression: the login-shell env snapshot must keep Hermes-managed
PATH entries even when a login profile resets PATH.

Debian's ``/etc/profile`` unconditionally resets PATH for non-root shells,
dropping the venv dir the process environment put there — so the snapshot
(``export -p``) serialises the reset PATH and bare ``python3``/``pip`` resolve
to the system interpreter (#56634). These tests read the snapshot back from a
real ``LocalEnvironment`` instead of asserting on injected markers.

See: https://github.com/NousResearch/hermes-agent/issues/56634
"""

import pathlib
import re

import pytest

from tools.environments.local import LocalEnvironment

_PATH_LINE = re.compile(r'^(?:declare -x|export) PATH="(.*)"$', re.MULTILINE)


def _managed_setup(tmp_path, monkeypatch, *, profile_body: str):
    """A resetting (or no-op) profile plus one fake Hermes-managed bin dir."""
    profile = tmp_path / "profile.sh"
    profile.write_text(profile_body, encoding="utf-8")
    managed = tmp_path / "managed-bin"
    managed.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "tools.environments.local._resolve_shell_init_files", lambda: [str(profile)]
    )
    monkeypatch.setattr(
        "tools.environments.local._resolve_hermes_bin_dir", lambda: str(managed)
    )
    monkeypatch.setattr(
        "tools.environments.local._managed_runtime_path_entries", list
    )
    return managed


def _snapshot_path_value(env: LocalEnvironment) -> list[str]:
    content = pathlib.Path(env._snapshot_path).read_text(encoding="utf-8-sig")
    match = _PATH_LINE.search(content)
    assert match, f"no PATH line in snapshot:\n{content}"
    return match.group(1).split(":")


class TestLoginShellPathPreservation:
    @pytest.mark.platforms("posix")
    def test_snapshot_keeps_managed_dir_when_profile_resets_path(
        self, tmp_path, monkeypatch
    ):
        """Debian-style profile reset (PATH=... before the snapshot's export -p)
        must not drop the managed dir from the captured snapshot."""
        managed = _managed_setup(
            tmp_path, monkeypatch, profile_body='PATH="/usr/local/bin:/usr/bin:/bin"\n'
        )

        env = LocalEnvironment(
            cwd=str(tmp_path), env={"PATH": f"{managed}:/usr/local/bin:/usr/bin:/bin"}
        )
        try:
            assert str(managed) in _snapshot_path_value(env)
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_snapshot_does_not_duplicate_managed_dir(self, tmp_path, monkeypatch):
        """A profile that leaves PATH alone: the restore's case guard must not
        prepend a second copy of an entry the shell already has."""
        managed = _managed_setup(
            tmp_path, monkeypatch, profile_body=": # profile that keeps PATH\n"
        )

        env = LocalEnvironment(
            cwd=str(tmp_path), env={"PATH": f"{managed}:/usr/local/bin:/usr/bin:/bin"}
        )
        try:
            entries = _snapshot_path_value(env)
            assert entries.count(str(managed)) == 1
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_non_login_path_keeps_managed_dir_without_profile(
        self, tmp_path, monkeypatch
    ):
        """Non-login commands never source the profile, so the resetting profile
        can't drop the managed dir — the baseline the login path is compared to."""
        managed = _managed_setup(
            tmp_path, monkeypatch, profile_body='PATH="/usr/local/bin:/usr/bin:/bin"\n'
        )

        env = LocalEnvironment(
            cwd=str(tmp_path), env={"PATH": f"{managed}:/usr/local/bin:/usr/bin:/bin"}
        )
        try:
            proc = env._run_bash('printf "%s" "$PATH"', login=False, timeout=30)
            out, _ = proc.communicate(timeout=30)
            assert str(managed) in out.split(":")
        finally:
            env.cleanup()


def _exe(directory, name: str):
    tool = directory / name
    tool.write_text(f"#!/bin/sh\necho {directory.name}\n", encoding="utf-8")
    tool.chmod(0o755)


@pytest.mark.platforms("posix")
def test_dropped_spawn_path_dirs_return_after_the_login_path(tmp_path, monkeypatch):
    """Any spawn-env dir a resetting profile dropped (not only Hermes' own: the gateway's
    pm runtime, a ``docker run -e PATH=`` addition) comes back in parent order AFTER the
    login PATH, so a command the login PATH resolves keeps resolving to the same binary."""
    user_bin, extra_a, extra_b, rel_bin = (tmp_path / n for n in ("user-bin", "extra-a", "extra-b", "rel-bin"))
    for d in (user_bin, extra_a, extra_b, rel_bin):
        d.mkdir()
    _exe(user_bin, "shared-tool")
    _exe(extra_a, "shared-tool")
    _exe(extra_b, "only-in-parent")
    _exe(rel_bin, "only-relative")
    _managed_setup(tmp_path, monkeypatch,
                   profile_body=f'PATH="/usr/local/bin:/usr/bin:/bin"\nPATH="{user_bin}:$PATH"\n')
    # A relative entry resolves against the terminal's cwd, not this process's.
    monkeypatch.chdir(user_bin)
    env = LocalEnvironment(cwd=str(tmp_path), env={"PATH": f"{extra_a}:{extra_b}:rel-bin:/usr/bin:/bin"})
    try:
        entries = _snapshot_path_value(env)
        assert entries.index(str(user_bin)) < entries.index("/usr/bin") < entries.index(str(extra_a)) \
            < entries.index(str(extra_b)) < entries.index("rel-bin")
        assert entries.count("/usr/bin") == 1
        out = env.execute("only-in-parent; shared-tool; only-relative")["output"].split()
        assert out == ["extra-b", "user-bin", "rel-bin"]
    finally:
        env.cleanup()


@pytest.mark.platforms("posix")
def test_background_command_keeps_spawn_path_dirs_a_login_profile_resets(tmp_path, monkeypatch):
    """Background commands run ``bash -lic`` with no snapshot, so a login profile that resets
    PATH Debian-style dropped every parent-only dir: the terminal tool's background spawn
    re-appends them through the same helper as the foreground, profile dirs first."""
    from types import SimpleNamespace

    from tests.waiting import wait_until
    from tools.process_registry import ProcessRegistry
    from tools.terminal_tool_background import _spawn

    profile_bin, parent_bin = tmp_path / "profile-bin", tmp_path / "parent-bin"
    for d in (profile_bin, parent_bin):
        d.mkdir()
    _exe(parent_bin, "only-in-parent")
    # Stand-in login shell whose profile resets PATH. Runner login shells pick their own
    # HOME and startup files (a CI runner's never read a test's ~/.profile), so the reset
    # is pinned here; the spawn env, argv and restore helper are the production ones.
    login_shell = tmp_path / "login-shell"
    login_shell.write_text(f'#!/usr/bin/env bash\nPATH="{profile_bin}:/usr/local/bin:/usr/bin:/bin"\n'
                           'exec /bin/bash --noprofile --norc -c "$2"\n', encoding="utf-8")
    login_shell.chmod(0o755)
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: str(login_shell))
    registry = ProcessRegistry()
    session = _spawn(registry, env=SimpleNamespace(env={"PATH": f"{parent_bin}:/usr/bin:/bin"}),
                     env_type="local", command='only-in-parent; printf "PATH=%s\\n" "$PATH"',
                     cwd=str(tmp_path), effective_task_id="t", task_id="t", session_key="",
                     effective_pty=False)
    assert wait_until(lambda: session.exited, timeout=30), session.output_buffer
    assert "parent-bin" in session.output_buffer.split(), session.output_buffer
    path_line = next(ln for ln in session.output_buffer.splitlines() if ln.startswith("PATH="))
    entries = path_line.removeprefix("PATH=").split(":")
    assert entries[0] == str(profile_bin)
    assert entries.index("/bin") < entries.index(str(parent_bin))
