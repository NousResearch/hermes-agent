"""The session snapshot on disk never holds a secret, and the shell still keeps it (#62336).

``hermes-snap-*.sh`` carries a terminal session's exports from one command to the next. A
service credential an agent exported, or one the profile's ``.env`` defines, used to land in it
in plaintext and come back after every delete. The values now travel in memory instead; these
run the real local backend and real bash, and read the file itself.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

from tools.environments.base import shell_state_scope
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.platforms("posix")


def _snapshot_text(env: LocalEnvironment) -> str:
    return "".join(open(path).read() for path in glob.glob(env._snapshot_path[:-3] + "*.sh"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("PORTAL_CLIENT_SECRET=SYNTH-dotenv\nODD_CUSTOM_NAME=SYNTH-dotenv-custom\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PORTAL_CLIENT_SECRET", "SYNTH-dotenv")
    monkeypatch.setenv("ODD_CUSTOM_NAME", "SYNTH-dotenv-custom")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    yield environment
    environment.cleanup()


def test_secret_exports_never_reach_the_snapshot_file(env):
    env.execute(
        "export CAROL_CLIENT_ID=SYNTH-id CAROL_CLIENT_SECRET='SYNTH-a b' github_token=SYNTH-lower "
        "GitHub_Token=SYNTH-mixed SESSION_COOKIE=SYNTH-cookie slack_webhook_url=https://hooks/SYNTH-hook "
        "DATABASE_URL=postgres://u:SYNTH-pw@db/app MULTI_TOKEN=$'SYNTH-l1\\nSYNTH-l2' KEEP_ME=ordinary; "
        "readonly RO_TOKEN=SYNTH-ro; export RO_TOKEN")
    with shell_state_scope(("child",)):
        env.execute("export CHILD_API_KEY=SYNTH-child")
    # A name the session already holds stays a secret after it leaves the profile's .env.
    (Path(os.environ["HERMES_HOME"]) / ".env").write_text("")
    env.execute("true")

    snapshot = _snapshot_text(env)
    assert "SYNTH" not in snapshot
    assert 'KEEP_ME="ordinary"' in snapshot  # ordinary shell state is still persisted
    # readonly survives the trip through memory
    out = env.execute('echo "[$RO_TOKEN][$ODD_CUSTOM_NAME]"; (RO_TOKEN=x) 2>/dev/null || echo readonly')["output"]
    assert out.split() == ["[SYNTH-ro][SYNTH-dotenv-custom]", "readonly"]


def test_secret_exports_still_carry_to_the_next_command(env):
    env.execute("export CAROL_CLIENT_SECRET='SYNTH-a b' MULTI_TOKEN=$'SYNTH-l1\\nSYNTH-l2' KEEP_ME=ordinary")
    with shell_state_scope(("child",)):
        assert env.execute('echo "[$CAROL_CLIENT_SECRET]"')["output"].strip() == "[SYNTH-a b]"
        env.execute("export CHILD_API_KEY=SYNTH-child")

    out = env.execute('echo "[$CAROL_CLIENT_SECRET][$MULTI_TOKEN][$ODD_CUSTOM_NAME][${CHILD_API_KEY-unset}]"')
    assert out["output"].strip() == "[SYNTH-a b][SYNTH-l1\nSYNTH-l2][SYNTH-dotenv-custom][unset]"

    env.execute("unset CAROL_CLIENT_SECRET")
    assert env.execute('echo "[${CAROL_CLIENT_SECRET-unset}]"')["output"].strip() == "[unset]"
