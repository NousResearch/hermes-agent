r"""Windows-path viability and PM runtime selection for bot relay (#93590).

Runtime and path failures covered here:

1. ``waiter_command`` used to embed the reply path into generated ``python -c``
   source, where the Windows execution layer's backslash folding turned
   ``C:\\Users`` into a unicode escape and SyntaxErrored the script. The waiter
   is now a runner entrypoint (``bot_mode_dm.py --wait-reply``) that takes the
   path as argv, rewritten to forward slashes the way the delivery runner's
   argv is — the tracked local backend runs commands through Git Bash there.

2. ``local_delivery_command`` hardcoded ``"hermes"``, relying on PATH —
   which service contexts (systemd units, desktop launchers, non-login
   SSH shells) do not provide, so delivery died with ENOENT. It now
   prefers this install's published launcher, then the interpreter's venv
   bin/Scripts sibling, PATH, and the bare name. The #93091
   turn-lock recognition in bot_mode_dm matches the CLI element by
   basename so resolved absolute paths (and ``hermes.exe``) still take
   the per-profile lock.

3. A bare store or legacy interpreter can lack the committed dependencies or
   load them with the wrong ABI. Delivery and reply-waiter commands select
   PM's committed Python; unmanaged installs retain the caller's interpreter.
"""

import json
import shlex
from pathlib import Path

import pytest

import tools.bot_mode_dm as bot_mode_dm
import tools.bot_relay as bot_relay
import pytest


ENV = {"id": "d" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}


@pytest.mark.parametrize("committed", [False, True])
def test_runner_commands_select_committed_python_and_published_cli(tmp_path, monkeypatch, committed):
    """Resolve real PM facts: neither a legacy sibling nor PATH owns a committed install."""
    from pm.environments import install_state_dir, runtime_facts_path, venv_python

    root = tmp_path / "source"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(bot_mode_dm, "__file__", str(root / "tools" / "bot_mode_dm.py"))
    monkeypatch.setattr(bot_relay, "__file__", str(root / "tools" / "bot_relay.py"))
    legacy_python = tmp_path / "legacy" / ("python.exe" if bot_relay.sys.platform == "win32" else "python")
    legacy_python.parent.mkdir()
    legacy_python.touch()
    monkeypatch.setattr(bot_relay.sys, "executable", str(legacy_python))
    cli_name = "hermes.exe" if bot_relay.sys.platform == "win32" else "hermes"
    legacy_cli = legacy_python.parent / cli_name
    legacy_cli.touch()
    monkeypatch.setattr(bot_relay.shutil, "which", lambda name: str(legacy_cli))

    expected_python = legacy_python
    expected_cli = legacy_cli
    if committed:
        environment = install_state_dir(root) / "environments" / "current" / "venv"
        environment.mkdir(parents=True)
        (environment / "pyvenv.cfg").touch()
        expected_python = venv_python(environment)
        expected_python.parent.mkdir(parents=True)
        expected_python.touch()
        runtime_facts_path(root).write_text(
            json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8"
        )
        expected_cli = root / ".hermes" / "bin" / cli_name
        expected_cli.parent.mkdir(parents=True)
        expected_cli.touch()

    def native_arg(path):
        value = str(path)
        return value.replace("\\", "/") if bot_relay.sys.platform == "win32" else value

    child_argv = bot_relay.local_delivery_command("ops", "query.json")
    assert child_argv[0] == str(expected_cli)
    assert child_argv[1:3] == ["-p", "ops"]
    for stdin_file in (False, True):
        parts = shlex.split(bot_mode_dm._delivery_command(child_argv, "query.json", stdin_file=stdin_file))
        assert parts[0] == native_arg(expected_python)
        assert parts[2:5] == ["--run-delivery", "stdin" if stdin_file else "query-file", "query.json"]
        assert parts[5] == native_arg(expected_cli)
    waiter = shlex.split(bot_relay.waiter_command(tmp_path / "home", ENV))
    assert waiter[0] == native_arg(expected_python)
    assert waiter[2] == "--wait-reply"


def test_runner_commands_reject_a_missing_committed_generation(tmp_path, monkeypatch):
    """An invalid committed selection cannot silently run an older dependency set."""
    from pm.environments import install_state_dir, runtime_facts_path

    root = tmp_path / "source"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(bot_mode_dm, "__file__", str(root / "tools" / "bot_mode_dm.py"))
    facts = runtime_facts_path(root)
    facts.parent.mkdir(parents=True)
    facts.write_text(json.dumps({"packages": {"venv": {
        "environment": str(install_state_dir(root) / "environments" / "missing" / "venv")
    }}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="dependency environment is missing"):
        bot_mode_dm._delivery_command(["hermes", "-p", "ops"], "query.json", stdin_file=False)
    with pytest.raises(RuntimeError, match="dependency environment is missing"):
        bot_relay.waiter_command(tmp_path / "home", ENV)


@pytest.mark.platforms("windows")
def test_waiter_argv_uses_forward_slashes_on_windows():
    """On native Windows the reply path rides as a forward-slash argv element, like the delivery
    runner's paths: Git Bash runs those, and parses a backslash path as a command name."""
    parts = shlex.split(bot_relay.waiter_command("C:\\Users\\joshu\\.hermes", ENV))

    assert "-c" not in parts and "--wait-reply" in parts
    assert parts[parts.index("--wait-reply") + 1] == f"C:/Users/joshu/.hermes/bot_relay/replies/{ENV['id']}.json"
    assert not any("\\" in part for part in parts)


@pytest.mark.platforms("linux")
def test_local_delivery_resolves_sibling_hermes(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    sibling = bin_dir / "hermes"
    sibling.touch()
    sibling.chmod(0o755)
    monkeypatch.setattr("sys.executable", str(bin_dir / "python"))

    argv = bot_relay.local_delivery_command("ops", "query.json")
    assert argv[0] == str(sibling)
    assert argv[1:3] == ["-p", "ops"]
    assert argv[argv.index("--query-file") + 1] == "query.json"


def test_local_delivery_uses_shutil_which_when_no_sibling(tmp_path, monkeypatch):
    """Without a venv sibling, a PATH hit (shutil.which) wins next —
    interactive shells keep resolving exactly what they resolve today."""
    empty = tmp_path / "nowhere"
    empty.mkdir(parents=True)
    monkeypatch.setattr("sys.executable", str(empty / "python"))
    # Keep this checkout's own published install launcher out of the resolution
    # when probing the fallback ladder (#124868).
    monkeypatch.setattr(bot_relay, "__file__", str(empty / "bot_relay.py"))
    which_hit = str(tmp_path / "usr-local-bin" / "hermes")
    monkeypatch.setattr(
        bot_relay.shutil, "which", lambda name: which_hit if name == "hermes" else None
    )

    argv = bot_relay.local_delivery_command("ops", "query.json")
    assert argv[0] == which_hit


def test_local_delivery_falls_back_to_bare_name(tmp_path, monkeypatch):
    empty = tmp_path / "nowhere"
    empty.mkdir(parents=True)
    monkeypatch.setattr("sys.executable", str(empty / "python"))
    monkeypatch.setattr(bot_relay.shutil, "which", lambda name: None)
    monkeypatch.setattr(bot_relay, "__file__", str(empty / "bot_relay.py"))

    argv = bot_relay.local_delivery_command("ops", "query.json")
    assert argv[0] == "hermes"
    assert argv[1:3] == ["-p", "ops"]


def test_delivery_lock_recognizes_resolved_cli_paths(tmp_path, monkeypatch):
    """The #93091 per-profile turn lock must keep matching delivery argvs
    now that argv[0] may be a resolved absolute path (or hermes.exe)."""
    acquired = []

    class _Ctx:
        def __enter__(self):
            acquired.append("locked")
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bot_relay, "acquire_turn_lock", lambda root, profile: _Ctx())
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with bot_mode_dm._delivery_lock(
        [str(tmp_path / "venv" / "bin" / "hermes"), "-p", "ops", "chat"],
        stdin_file=False,
    ):
        pass
    with bot_mode_dm._delivery_lock(["hermes", "-p", "ops", "chat"], stdin_file=False):
        pass
    with bot_mode_dm._delivery_lock(
        ["C:\\venv\\Scripts\\hermes.exe", "-p", "ops", "chat"], stdin_file=False
    ):
        pass
    assert acquired == ["locked", "locked", "locked"]

    # Unrelated argvs still bypass the lock entirely.
    with bot_mode_dm._delivery_lock(["python", "-m", "whatever"], stdin_file=False):
        pass
    assert acquired == ["locked", "locked", "locked"]
