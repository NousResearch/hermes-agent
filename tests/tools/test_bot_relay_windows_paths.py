r"""Windows-path viability and venv CLI resolution for bot relay (#93590).

Two failures on a Windows desktop install talking to a remote gateway:

1. ``waiter_command`` used to embed the reply path into generated ``python -c``
   source, where the Windows execution layer's backslash folding turned
   ``C:\\Users`` into a unicode escape and SyntaxErrored the script. The waiter
   is now a runner entrypoint (``bot_mode_dm.py --wait-reply``) that takes the
   path as argv, rewritten to forward slashes the way the delivery runner's
   argv is — the tracked local backend runs commands through Git Bash there.

2. ``local_delivery_command`` hardcoded ``"hermes"``, relying on PATH —
   which service contexts (systemd units, desktop launchers, non-login
   SSH shells) do not provide, so delivery died with ENOENT. It now
   resolves the CLI next to this gateway's own interpreter (the venv
   bin/Scripts sibling), falling back to the bare name. The #93091
   turn-lock recognition in bot_mode_dm matches the CLI element by
   basename so resolved absolute paths (and ``hermes.exe``) still take
   the per-profile lock.
"""

import shlex
from pathlib import Path

import pytest

import tools.bot_mode_dm as bot_mode_dm
import tools.bot_relay as bot_relay


ENV = {"id": "d" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}


@pytest.mark.windows_only
def test_waiter_argv_uses_forward_slashes_on_windows():
    """On native Windows the reply path rides as a forward-slash argv element, like the delivery
    runner's paths: Git Bash runs those, and parses a backslash path as a command name."""
    parts = shlex.split(bot_relay.waiter_command("C:\\Users\\joshu\\.hermes", ENV))

    assert "-c" not in parts and "--wait-reply" in parts
    assert parts[parts.index("--wait-reply") + 1] == f"C:/Users/joshu/.hermes/bot_relay/replies/{ENV['id']}.json"
    assert not any("\\" in part for part in parts)


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

    monkeypatch.setattr(bot_relay, "acquire_turn_lock",
                    lambda root, profile, **kwargs: _Ctx())
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
