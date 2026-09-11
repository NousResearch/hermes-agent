from pathlib import Path

from tools.environments.local_no_new_privs import (
    no_new_privs_enabled,
    prepare_systemd_run_escape,
)


def _status(tmp_path: Path, value: int) -> Path:
    status = tmp_path / "status"
    status.write_text(f"Name:\tpython\nNoNewPrivs:\t{value}\n", encoding="utf-8")
    return status


def test_no_new_privs_reads_proc_status_value(tmp_path):
    assert no_new_privs_enabled(_status(tmp_path, 1)) is True
    assert no_new_privs_enabled(_status(tmp_path, 0)) is False
    assert no_new_privs_enabled(tmp_path / "missing") is False


def test_systemd_escape_transfers_sanitized_environment_over_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/alice")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("API_TOKEN", "must-not-reach-systemd-client")
    run_env = {
        "EMPTY": "",
        "PATH": "/opt/hermes/bin:/usr/bin",
        "SUDO_PASSWORD": "terminal-secret",
        "VALUE": "contains=equals\nand-newline",
    }

    escape = prepare_systemd_run_escape(
        ["/bin/bash", "-c", "sudo -S id"],
        run_env,
        "/home/alice/project",
        "terminal-secret\ncommand-input\n",
        has_sudo=True,
        platform="linux",
        status_path=_status(tmp_path, 1),
        which=lambda command: "/usr/bin/systemd-run" if command == "systemd-run" else None,
        getuid=lambda: 1000,
        path_exists=lambda path: path == "/run/user/1000/bus",
    )

    assert escape is not None
    assert escape.args[:7] == [
        "/usr/bin/systemd-run",
        "--user",
        "--pipe",
        "--quiet",
        "--collect",
        "--wait",
        "--working-directory=/home/alice/project",
    ]
    assert escape.args[-3:] == ["/bin/bash", "-c", "sudo -S id"]
    assert "terminal-secret" not in "\0".join(escape.args)
    assert escape.env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
    assert escape.env["HOME"] == "/home/alice"
    assert escape.env["PATH"] == "/usr/bin"
    assert escape.env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert "API_TOKEN" not in escape.env
    assert escape.stdin_data == (
        "EMPTY=\0"
        "PATH=/opt/hermes/bin:/usr/bin\0"
        "SUDO_PASSWORD=terminal-secret\0"
        "VALUE=contains=equals\nand-newline\0"
        "\0"
        "terminal-secret\ncommand-input\n"
    )


def test_systemd_escape_leaves_unaffected_commands_on_direct_path(tmp_path):
    common = {
        "args": ["/bin/bash", "-c", "id"],
        "run_env": {"PATH": "/usr/bin"},
        "cwd": "/tmp",
        "stdin_data": None,
        "platform": "linux",
        "status_path": _status(tmp_path, 1),
        "which": lambda _command: "/usr/bin/systemd-run",
        "getuid": lambda: 1000,
        "path_exists": lambda _path: True,
    }

    assert prepare_systemd_run_escape(**common, has_sudo=False) is None
    assert prepare_systemd_run_escape(**{**common, "platform": "darwin"}, has_sudo=True) is None
    assert prepare_systemd_run_escape(
        **{**common, "status_path": _status(tmp_path, 0)}, has_sudo=True
    ) is None
    assert prepare_systemd_run_escape(
        **{**common, "path_exists": lambda _path: False}, has_sudo=True
    ) is None
