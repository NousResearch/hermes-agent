"""Desktop Linux: sudo under Electron's inherited NoNewPrivs (#108595).

Packaged Electron with a setuid chrome-sandbox latches PR_SET_NO_NEW_PRIVS on
the main process; the spawned backend inherits it and kernel-refuses sudo's
setuid bit (NOPASSWD and SUDO_PASSWORD both fail with "no new privileges").
Escape hatch: systemd-run --user --pipe so the command runs in a fresh user
unit outside that tree.
"""

from __future__ import annotations

import functools
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

import tools.terminal_tool_sudo as terminal_tool


_STUB_SYSTEMD_RUN_TEMPLATE = r'''#!@PYTHON@
"""Stub systemd-run emulating --user --pipe environment semantics.

A real user unit does NOT inherit the caller's environment: the unit child
gets the user manager's environment plus EnvironmentFile/--setenv
assignments, minus UnsetEnvironment names. This stub builds exactly that
(a minimal manager env — PATH and HOME — plus the parsed assignments),
records what the child received, and runs the command with it.
"""
import json
import os
import shlex
import subprocess
import sys

RECORD_PATH = @RECORD_PATH@


def _parse_env_file(path):
    env = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, value = line.partition("=")
            if not sep:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                inner = value[1:-1]
                out = []
                i = 0
                while i < len(inner):
                    ch = inner[i]
                    if ch == "\\" and i + 1 < len(inner):
                        nxt = inner[i + 1]
                        out.append({"n": "\n", "r": "\r"}.get(nxt, nxt))
                        i += 2
                    else:
                        out.append(ch)
                        i += 1
                value = "".join(out)
            env[name] = value
    return env


def main():
    argv = sys.argv[1:]
    try:
        dash = argv.index("--")
    except ValueError:
        print("stub systemd-run: missing '--' separator", file=sys.stderr)
        return 125
    opts, cmd = argv[:dash], argv[dash + 1:]
    env_file = None
    unset_names = []
    setenv = {}
    working_directory = None
    for opt in opts:
        if opt.startswith("--property=EnvironmentFile="):
            env_file = shlex.split(opt.split("=", 2)[2])[0]
        elif opt.startswith("--property=UnsetEnvironment="):
            unset_names = shlex.split(opt.split("=", 2)[2])[0].split()
        elif opt.startswith("--setenv="):
            name, _, value = opt.split("=", 1)[1].partition("=")
            setenv[name] = value
        elif opt.startswith("--working-directory="):
            working_directory = shlex.split(opt.split("=", 1)[1])[0]
        # --user --pipe --wait --quiet --collect --expand-environment=no
        # --unit=... carry no env semantics for this stub.
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
    }
    if env_file:
        child_env.update(_parse_env_file(env_file))
    for name in unset_names:
        child_env.pop(name, None)
    child_env.update(setenv)
    with open(RECORD_PATH, "w", encoding="utf-8") as fh:
        json.dump(child_env, fh, sort_keys=True)
    if working_directory:
        os.chdir(working_directory)
    if not cmd:
        return 126
    return subprocess.run(cmd, env=child_env, stdin=subprocess.DEVNULL).returncode


sys.exit(main())
'''


def _write_systemd_run_stub(directory, record_path) -> str:
    """Materialize the stub systemd-run and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / "systemd-run"
    stub.write_text(
        _STUB_SYSTEMD_RUN_TEMPLATE
        .replace("@PYTHON@", sys.executable)
        .replace("@RECORD_PATH@", repr(str(record_path))),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return str(stub)


@functools.cache
def _user_systemd_bus_usable() -> bool:
    """A reachable user systemd manager — the capability the NNP escape rides on.

    Probes with the same ``systemctl --user show-environment`` call the product
    code's manager-env lookup (``_nnp_manager_keys_to_unset``) makes, so hosts
    without a user bus (the CI Linux runner: ``failed to connect to bus:
    no medium found``) skip the wrap harness instead of failing it. The product
    code itself degrades the same way — no wrap escape exists without the bus —
    so skipping the harness there mirrors the product's own capability gate.
    """
    ctl = "/usr/bin/systemctl"
    if not os.path.isfile(ctl) or not os.path.isfile("/usr/bin/systemd-run"):
        return False
    try:
        probe = subprocess.run(
            [ctl, "--user", "show-environment"],
            capture_output=True,
            timeout=3,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


@pytest.mark.platforms("linux")
def test_wraps_sudo_in_systemd_run_pipe_when_no_new_privs(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")

    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true", cwd="/tmp")

    assert wrapped.startswith("/usr/bin/systemd-run")
    assert " --user " in f" {wrapped} " or "--user" in wrapped
    assert "--pipe" in wrapped
    assert "--wait" in wrapped
    assert "--unit=hermes-nnp-sudo-" in wrapped
    assert "--working-directory=/tmp" in wrapped
    assert "sudo -n true" in wrapped
    assert shutil.which("systemd-run") is not None or wrapped.startswith("/usr/bin/systemd-run")


@pytest.mark.platforms("linux")
def test_wrap_units_are_unique(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")
    a = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true")
    b = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true")
    assert terminal_tool._nnp_sudo_unit_from_command(a) != terminal_tool._nnp_sudo_unit_from_command(b)


def test_does_not_wrap_when_no_new_privs_is_clear(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: False)

    assert terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true") == "sudo -n true"


def test_does_not_wrap_commands_without_sudo(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)

    assert terminal_tool._wrap_local_command_for_no_new_privs("id -un") == "id -un"


def test_does_not_use_untrusted_systemd_run_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: None)
    fake = tmp_path / "systemd-run"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))

    assert terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true") == "sudo -n true"


@pytest.mark.skipif(
    not shutil.which("setpriv") or not os.path.isfile("/usr/bin/systemd-run")
    or not _user_systemd_bus_usable(),
    reason="setpriv + /usr/bin/systemd-run + a reachable user systemd bus required for the kernel-latch harness",
)
def test_wrapped_sudo_does_not_hit_kernel_no_new_privs_latch(monkeypatch):
    """setpriv reproduces Electron's latch; wrap must actually reach sudo."""
    raw = subprocess.run(
        ["setpriv", "--no-new-privs", "sudo", "-n", "true"],
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert raw.returncode != 0
    assert "no new privileges" in (raw.stderr or "").lower()

    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs("sudo -n true", cwd="/tmp")
    escaped = subprocess.run(
        ["setpriv", "--no-new-privs", "bash", "-lc", wrapped],
        capture_output=True,
        text=True,
        timeout=12,
    )
    combined = f"{escaped.stdout}\n{escaped.stderr}".lower()
    assert "no new privileges" not in combined
    assert "failed to connect" not in combined
    sudo_ran = escaped.returncode == 0 or "password is required" in combined
    assert sudo_ran, combined
    unit = terminal_tool._nnp_sudo_unit_from_command(wrapped)
    assert unit
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "stop", unit],
        capture_output=True,
        timeout=5,
    )


def test_trusted_helper_stat_rejects_group_or_world_writable():
    class _St:
        st_mode = 0o100755 | 0o022
        st_uid = 0

    assert terminal_tool._is_trusted_helper_stat(_St()) is False


def test_trusted_helper_stat_accepts_root_owned_0755():
    class _St:
        st_mode = 0o100755
        st_uid = 0

    assert terminal_tool._is_trusted_helper_stat(_St()) is True


@pytest.mark.platforms("linux")
def test_wrap_passes_environment_file(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")
    env_file = tmp_path / "nnp.env"
    env_file.write_text("HERMES_NNP_PROBE=from-profile\n")
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs(
        "sudo -n true", cwd="/tmp", env_file=str(env_file)
    )
    assert f"EnvironmentFile={env_file}" in wrapped


@pytest.mark.platforms("linux")
def test_wrap_unsets_manager_only_names(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")
    wrapped = terminal_tool._wrap_local_command_for_no_new_privs(
        "sudo -n true", unset_names=["HERMES_NNP_SENTINEL"]
    )
    assert "--expand-environment=no" in wrapped
    assert "HERMES_NNP_SENTINEL" in wrapped
    assert "UnsetEnvironment=" in wrapped


@pytest.mark.platforms("linux")
def test_manager_keys_to_unset_keeps_run_env_names(monkeypatch):
    class _Completed:
        returncode = 0
        stdout = "HOME=/home/u\nHERMES_NNP_SENTINEL=from-manager\nPATH=/usr/bin\n"

    monkeypatch.setattr(terminal_tool.subprocess, "run", lambda *a, **k: _Completed())
    names = terminal_tool._nnp_manager_keys_to_unset({"HOME": "/tmp", "PATH": "/bin"})
    assert "HERMES_NNP_SENTINEL" in names
    assert "HOME" not in names
    assert "PATH" not in names


def test_write_nnp_env_file_is_owner_only(tmp_path):
    path = terminal_tool._write_nnp_env_file({"HERMES_NNP_PROBE": "xyz", "EMPTY": ""}, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert 'HERMES_NNP_PROBE="xyz"' in text
    if sys.platform != "win32":
        assert (os.stat(path).st_mode & 0o077) == 0


def test_write_nnp_env_file_keeps_empty_values(tmp_path):
    """Empty keys must be written so a user-manager value cannot fill the omission."""
    path = terminal_tool._write_nnp_env_file({"MANAGER_ONLY": ""}, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert 'MANAGER_ONLY=""' in text


def test_write_nnp_env_file_quotes_whitespace_and_escapes(tmp_path):
    path = terminal_tool._write_nnp_env_file(
        {
            "SPACED": " leading and trailing ",
            "QUOTED": 'say "hi"',
            "SLASHED": r"C:\temp\nnp",
        },
        str(tmp_path),
    )
    text = open(path, encoding="utf-8").read()
    assert 'SPACED=" leading and trailing "' in text
    assert r'QUOTED="say \"hi\""' in text
    assert r'SLASHED="C:\\temp\\nnp"' in text


def test_release_nnp_sudo_env_file_is_exactly_once(tmp_path):
    path = tmp_path / "hermes-nnp-env-once.env"
    path.write_text("A=1\n", encoding="utf-8")

    class _Proc:
        pass

    proc = _Proc()
    proc._nnp_sudo_env_file = str(path)
    terminal_tool._release_nnp_sudo_env_file(proc)
    assert not path.exists()
    assert getattr(proc, "_nnp_sudo_env_file", None) is None
    terminal_tool._release_nnp_sudo_env_file(proc)  # idempotent


def test_wait_unlinks_nnp_env_file_after_natural_exit(monkeypatch, tmp_path):
    from tools.environments.local import LocalEnvironment

    path = tmp_path / "hermes-nnp-env-wait.env"
    path.write_text("A=1\n", encoding="utf-8")

    class _Proc:
        def poll(self):
            return 0

    proc = _Proc()
    proc._nnp_sudo_env_file = str(path)
    monkeypatch.setattr(LocalEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        "tools.environments.base.BaseEnvironment._wait_for_process",
        lambda self, p, *a, **k: {"returncode": 0},
    )
    env = LocalEnvironment(cwd=str(tmp_path))
    result = env._wait_for_process(proc, timeout=1)
    assert result["returncode"] == 0
    assert not path.exists()
    assert getattr(proc, "_nnp_sudo_env_file", None) is None


def test_unit_for_kill_comes_from_proc_not_environment():
    class _Proc:
        pass

    proc_a = _Proc()
    proc_b = _Proc()
    proc_a._nnp_sudo_unit = "hermes-nnp-sudo-1-aaaaaaaa.service"
    proc_b._nnp_sudo_unit = "hermes-nnp-sudo-2-bbbbbbbb.service"
    assert terminal_tool._nnp_sudo_unit_from_proc(proc_a) == "hermes-nnp-sudo-1-aaaaaaaa.service"
    assert terminal_tool._nnp_sudo_unit_from_proc(proc_b) == "hermes-nnp-sudo-2-bbbbbbbb.service"


@pytest.mark.platforms("posix")
def test_execute_env_propagates_through_stub_systemd_run(monkeypatch, tmp_path):
    """Real ``LocalEnvironment.execute()``: a late profile env reaches the unit child.

    The CI Linux runner has no user systemd bus, so the real-bus test above
    skips there (and on every macOS host) — proving nothing where it matters.
    This one runs everywhere the wrap can fire: ``execute()`` is driven
    end-to-end with a stub ``systemd-run`` standing in for the trusted binary.
    The stub emulates ``--user --pipe`` env semantics: the unit child does NOT
    inherit the caller's environment, it gets a minimal manager env plus the
    EnvironmentFile the wrap writes (``_run_bash``), and it records exactly
    what the child received.

    The sentinel is set on the environment AFTER construction, so it is
    provably absent from the init-session snapshot the wrapper sources — the
    per-process EnvironmentFile is the only channel. ``nnp-unfiled-marker``
    rides the caller's env but cannot be written to the EnvironmentFile (its
    key fails the writer's name regex), so it reaching the child would mean
    the stub leaked the parent env; its absence proves the recorded env was
    really built from the EnvironmentFile. Removing the env-file handoff — or
    making it lossy — fails this test; passing it does not.
    """
    from tools.environments.local import LocalEnvironment

    record_path = tmp_path / "stub-child-env.json"
    stub = _write_systemd_run_stub(tmp_path / "bin", record_path)
    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: stub)

    sentinel_value = "hermes-nnp-sentinel-via-stub"
    local = LocalEnvironment(cwd=str(tmp_path))
    local.env["HERMES_NNP_TEST_SENTINEL"] = sentinel_value
    local.env["nnp-unfiled-marker"] = "must-not-arrive"
    result = local.execute(
        'printf "%s" "$HERMES_NNP_TEST_SENTINEL" && sudo -n true',
        timeout=60,
    )
    output = result.get("output") or ""
    combined = output.lower()
    assert "no new privileges" not in combined, output
    assert "failed to connect" not in combined, output
    assert sentinel_value in output, output
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    assert recorded.get("HERMES_NNP_TEST_SENTINEL") == sentinel_value, recorded
    assert "nnp-unfiled-marker" not in recorded, recorded


@pytest.mark.platforms("linux")
@pytest.mark.skipif(
    not os.path.isfile("/usr/bin/systemd-run") or not _user_systemd_bus_usable(),
    reason="/usr/bin/systemd-run + a reachable user systemd bus required for a real wrapped execute()",
)
def test_execute_env_propagates_to_wrapped_sudo_child(monkeypatch, tmp_path):
    """Real ``LocalEnvironment.execute()``: a late profile env reaches the wrapped unit.

    Drives the unmocked execute() path — ``_prepare_command`` → ``_wrap_command``
    → ``_run_bash`` (EnvironmentFile + systemd-run --user --pipe wrap) — with a
    sentinel var set on the environment AFTER construction, so it is provably
    absent from the init-session snapshot the wrapper sources. The command bears
    a real ``sudo`` so the wrap fires, but the sentinel is printed BEFORE sudo:
    sudo's own ``env_reset`` cannot scrub it, so the value seen inside the
    transient unit can only have arrived through the per-process EnvironmentFile
    the wrap writes (``--pipe`` gives the unit the user manager's env, NOT the
    parent's). Removing the env-file handoff — or making it lossy — fails this.
    """
    from tools.environments.local import LocalEnvironment

    monkeypatch.setattr(terminal_tool, "_process_has_no_new_privs", lambda: True)
    monkeypatch.setattr(terminal_tool, "_trusted_systemd_run_binary", lambda: "/usr/bin/systemd-run")

    sentinel_value = "hermes-nnp-sentinel-from-execute"
    local = LocalEnvironment(cwd=str(tmp_path))
    local.env["HERMES_NNP_TEST_SENTINEL"] = sentinel_value
    result = local.execute(
        'printf "%s" "$HERMES_NNP_TEST_SENTINEL" && sudo -n true',
        timeout=30,
    )
    output = result.get("output") or ""
    combined = output.lower()
    # The wrapper must have reached the transient unit — neither the kernel
    # latch nor a bus failure is an acceptable outcome on a bus-capable host.
    assert "no new privileges" not in combined, output
    assert "failed to connect" not in combined, output
    assert sentinel_value in output, output
