"""The ``bot_mode_dm.py`` runner entry boots Hermes' dependency environment itself.

``message_agent`` spawns ``<sys.executable> tools/bot_mode_dm.py --run-delivery …`` and the relay
spawns ``… --wait-reply …`` (``bot_relay.waiter_command``). Under a PM-managed install
``sys.executable`` is the bare store interpreter: dependencies are activated in-process at boot,
and the terminal backend strips the Hermes-owned ``PYTHONPATH``, so a runner that does not boot
like every other entry point dies on its first third-party import. Live, every local Bot Chat DM
came back ``Live admission outcome unknown: No module named 'ruamel'. Do not resend.``

Each test launches the real script under ``-I -S`` (no site-packages, no ``PYTHON*`` env), with a
PM-committed dependency record in a temp ``HERMES_HOME`` as the only way to reach the packages.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from pm.environments import install_state_dir, runtime_facts_path, site_packages
from tools import bot_relay

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools" / "bot_mode_dm.py"
# A bare interpreter: isolated (-I ignores PYTHONPATH/PYTHONHOME, user site) and no site
# module (-S), so no third-party package is importable unless the entry activates one.
BARE = [sys.executable, "-I", "-S"]


@pytest.fixture
def committed_home(tmp_path, monkeypatch):
    """A temp ``HERMES_HOME`` whose install state commits a dependency generation for REPO.

    The generation's site-packages carries a ``.pth`` naming this interpreter's own package
    directories, so activating it (and only activating it) makes the real dependencies importable.
    """
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    venv = install_state_dir(REPO) / "environments" / "gen" / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "version = {}.{}.{}\n".format(*sys.version_info[:3]), encoding="utf-8")
    packages = site_packages(venv)
    packages.mkdir(parents=True)
    dirs = [p for p in sys.path if p and Path(p).is_dir() and Path(p).resolve() != REPO]
    (packages / "hermes-test-deps.pth").write_text("\n".join(dirs) + "\n", encoding="utf-8")
    runtime_facts_path(REPO).write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["HERMES_HOME"] = str(home)
    return home, env


def _run(argv, env):
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, cwd=str(REPO.parent), timeout=120)


def test_delivery_runner_admits_through_dependencies_it_activates(tmp_path, committed_home):
    """Live admission (``_admit_live_dm``) imports Hermes' third-party graph. Launched bare, the
    runner must activate the committed environment, find no live owner, and hand the DM to the
    transport, which consumes the file — never report an ``ambiguous`` import failure."""
    _home, env = committed_home
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    dm_file = tmp_path / "dm.txt"
    dm_file.write_text("hello teammate", encoding="utf-8")
    # The transport echoes the DM file it receives via --query-file.
    transport = [sys.executable, "-I", "-S", "-c", "import sys; print(open(sys.argv[2], encoding='utf-8').read())"]

    result = _run([*BARE, str(RUNNER), "--run-delivery", "query-file", str(dm_file),
                   "--profile-home", str(profile_home), *transport], env)

    assert "No module named" not in result.stdout + result.stderr, result
    assert result.returncode == 0, result
    assert result.stdout.strip() == "hello teammate"
    assert not dm_file.exists()


def test_reply_waiter_runs_through_the_bootstrapped_entry(tmp_path, committed_home):
    """The relay's reply waiter shares the entry: its argv, exactly as ``waiter_command`` builds it,
    must pass the bootstrap untouched and print the reply the sender wakes on."""
    _home, env = committed_home
    root = tmp_path / "root"
    envelope = {"id": "e" * 32, "target_handle": "researcher", "target_connection": "ssh-vps"}
    reply_path = bot_relay.relay_root(root) / bot_relay.REPLIES_DIR / f"{envelope['id']}.json"
    reply_path.parent.mkdir(parents=True)
    reply_path.write_text(json.dumps({"reply": "pong"}), encoding="utf-8")
    argv = shlex.split(bot_relay.waiter_command(root, envelope))

    result = _run([*BARE, *argv[1:]], env)

    assert result.returncode == 0, result
    assert result.stdout.splitlines() == ["Reply from @researcher on ssh-vps:", "pong"]
