r"""Windows-path viability and venv CLI resolution for bot relay (#93590).

Two failures on a Windows desktop install talking to a remote gateway:

1. ``waiter_command`` used to embed the reply path into generated ``python -c``
   source, where the Windows execution layer's backslash folding turned
   ``C:\\Users`` into a unicode escape and SyntaxErrored the script. The waiter
   is now a runner entrypoint (``bot_mode_dm.py --wait-reply``) that takes the
   path as argv, rewritten to forward slashes the way the delivery runner's
   argv is — the tracked local backend runs commands through Git Bash there.

2. the delivery runner argv hardcoded ``"hermes"``, relying on PATH —
   which service contexts (systemd units, desktop launchers, non-login
   SSH shells) do not provide, so delivery died with ENOENT. ``_hermes_cli``
   resolves the CLI next to this gateway's own interpreter (the venv
   bin/Scripts sibling), falling back to the bare name.
"""

import shlex
from pathlib import Path

import pytest

from hermes_agent.tools import bot_relay


ENV = {"id": "d" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}


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
    # The checkout's own published launcher (.hermes/bin, written by scripts/run_tests.sh's venv
    # step) outranks the sibling by design (#124868): keep it out of this probe.
    monkeypatch.setattr("hermes_agent.hermes_cli._startup_fast.project_root_str", lambda: str(tmp_path / "checkout"))

    assert bot_relay._hermes_cli() == str(sibling)


def test_local_delivery_uses_shutil_which_when_no_sibling(tmp_path, monkeypatch):
    """Without a venv sibling, a PATH hit (shutil.which) wins next —
    interactive shells keep resolving exactly what they resolve today."""
    empty = tmp_path / "nowhere"
    empty.mkdir(parents=True)
    monkeypatch.setattr("sys.executable", str(empty / "python"))
    # Keep this checkout's own published install launcher out of the resolution
    # when probing the fallback ladder (#124868).
    monkeypatch.setattr("hermes_agent.hermes_cli._startup_fast.project_root_str", lambda: str(empty))
    which_hit = str(tmp_path / "usr-local-bin" / "hermes")
    monkeypatch.setattr(
        bot_relay.shutil, "which", lambda name: which_hit if name == "hermes" else None
    )

    assert bot_relay._hermes_cli() == which_hit


def test_local_delivery_falls_back_to_bare_name(tmp_path, monkeypatch):
    empty = tmp_path / "nowhere"
    empty.mkdir(parents=True)
    monkeypatch.setattr("sys.executable", str(empty / "python"))
    monkeypatch.setattr(bot_relay.shutil, "which", lambda name: None)
    monkeypatch.setattr("hermes_agent.hermes_cli._startup_fast.project_root_str", lambda: str(empty))

    assert bot_relay._hermes_cli() == "hermes"
