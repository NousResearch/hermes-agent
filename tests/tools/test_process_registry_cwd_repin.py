"""Background commands must retain their requested cwd after shell startup (#76144)."""

import os
import shlex
import shutil

import pytest

from tools.process_registry import ProcessRegistry


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr("tools.process_registry._SYSTEMD_SCOPE_AVAILABLE", False)
    reg = ProcessRegistry()
    yield reg
    for sid in list(reg._running):
        reg.kill_process(sid)


def _wait_for_exit(session):
    assert session._completion_event.wait(20), "background command did not finish"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("shell_name", ["dash", "sh"])
@pytest.mark.parametrize("use_pty", [False, True])
@pytest.mark.parametrize("remove_cwd", [False, True])
def test_posix_rc_cannot_override_requested_cwd(
    registry, tmp_path, monkeypatch, shell_name, use_pty, remove_cwd
):
    """Real shells bypass rc cd functions and fail closed if rc removes the cwd."""
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is not installed")
    home = tmp_path / "home"
    home.mkdir()
    workdir = tmp_path / "work dir 'quoted'"
    workdir.mkdir()
    observed = tmp_path / "cwd.out"
    profile = home / ".profile"
    profile.write_text(
        "cd /\n"
        + (f"rmdir -- {shlex.quote(str(workdir))}\n" if remove_cwd else "")
        + "cd() { return 1; }\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ENV", str(profile))
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: shell)

    # Observe the cwd via a file: a short-lived PTY need not retain its stdout.
    command = f"pwd -P > {shlex.quote(str(observed))}"
    session = registry.spawn_local(command, cwd=str(workdir), use_pty=use_pty)
    _wait_for_exit(session)
    assert (session._pty is not None) is use_pty
    if remove_cwd:
        assert session.exit_code == 126, session.output_buffer
        assert not observed.exists(), "the command ran despite an unavailable cwd"
    else:
        assert session.exit_code == 0, session.output_buffer
        assert observed.read_text().strip() == os.path.realpath(workdir)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("use_pty", [False, True])
def test_git_bash_rc_cannot_override_native_windows_cwd(
    registry, tmp_path, monkeypatch, use_pty
):
    """A real Git Bash spawn converts a native cwd containing spaces before cd."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text("cd /\ncd() { return 1; }\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    workdir = tmp_path / "work dir 'quoted'"
    workdir.mkdir()

    observed = tmp_path / "cwd.out"
    command = f"pwd -W > {shlex.quote(observed.as_posix())}"
    session = registry.spawn_local(command, cwd=str(workdir), use_pty=use_pty)
    _wait_for_exit(session)
    assert (session._pty is not None) is use_pty
    assert session.exit_code == 0, session.output_buffer
    assert os.path.samefile(observed.read_text().strip(), workdir)
