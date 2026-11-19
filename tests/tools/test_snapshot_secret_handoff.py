"""The session snapshot on disk never holds a secret, and the shell still keeps it (#62336).

``hermes-snap-*.sh`` carries a terminal session's exports from one command to the next. A
service credential an agent exported, or one the profile's ``.env`` defines, used to land in it
in plaintext and come back after every delete. The values now travel in memory instead; these
run the real local backend and real bash, and read the file itself.
"""

from __future__ import annotations

import glob
import os
import threading
from pathlib import Path

import pytest

from tools.environments.base import shell_state_scope
from tools.environments.local import LocalEnvironment
from tests.waiting import wait_until

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
        "readonly RO_TOKEN=SYNTH-ro; export RO_TOKEN; declare -ax API_TOKEN=(SYNTH-one 'SYNTH two')")
    with shell_state_scope(("child",)):
        env.execute("export CHILD_API_KEY=SYNTH-child")
    # A name the session ever held stays a secret after it is unset, leaves .env and comes back.
    env.execute("unset ODD_CUSTOM_NAME")
    (Path(os.environ["HERMES_HOME"]) / ".env").write_text("")
    env.execute("export ODD_CUSTOM_NAME=SYNTH-reintroduced")
    # A command that leaves IFS empty, then failed publishes, must neither leak nor lose them.
    env.execute("export IFS=''")
    for _ in range(10):
        env.execute("mv() { return 1; }")

    snapshot = _snapshot_text(env)
    assert "SYNTH" not in snapshot
    assert 'KEEP_ME="ordinary"' in snapshot  # ordinary shell state is still persisted
    # readonly and array secrets keep their shape through memory
    out = env.execute('unset -f mv; IFS=$\' \\t\\n\'; echo "[$RO_TOKEN][$ODD_CUSTOM_NAME][${#API_TOKEN[@]}|${API_TOKEN[1]}]"; '
                      '(RO_TOKEN=x) 2>/dev/null || echo readonly')["output"]
    assert out.splitlines() == ["[SYNTH-ro][SYNTH-reintroduced][2|SYNTH two]", "readonly"]


def test_secret_exports_still_carry_to_the_next_command(env):
    env.execute("export CAROL_CLIENT_SECRET='SYNTH-a b' MULTI_TOKEN=$'SYNTH-l1\\nSYNTH-l2' KEEP_ME=ordinary")
    with shell_state_scope(("child",)):
        assert env.execute('echo "[$CAROL_CLIENT_SECRET]"')["output"].strip() == "[SYNTH-a b]"
        env.execute("export CHILD_API_KEY=SYNTH-child")

    out = env.execute('echo "[$CAROL_CLIENT_SECRET][$MULTI_TOKEN][$ODD_CUSTOM_NAME][${CHILD_API_KEY-unset}]"')
    assert out["output"].strip() == "[SYNTH-a b][SYNTH-l1\nSYNTH-l2][SYNTH-dotenv-custom][unset]"

    env.execute("unset CAROL_CLIENT_SECRET")
    assert env.execute('echo "[${CAROL_CLIENT_SECRET-unset}]"')["output"].strip() == "[unset]"


def test_overlapping_commands_never_mix_one_dump_with_another(env, tmp_path):
    """The next command's ordinary exports and secrets come from the same dump, even when an
    earlier command publishes its file, stalls, and reports its secrets after a later one."""
    published = tmp_path / "published"
    slow = (f'export PAIR=A SERVICE_TOKEN=SYNTH-A; mv() {{ command mv "$@"; : > {published}; '
            'sleep 1; }')
    first = threading.Thread(target=env.execute, args=(slow,))
    first.start()
    wait_until(published.exists, timeout=10, what="first command published its snapshot")
    env.execute("export PAIR=B SERVICE_TOKEN=SYNTH-B")
    first.join()
    assert env.execute('echo "$PAIR $SERVICE_TOKEN"')["output"].strip() in {"A SYNTH-A", "B SYNTH-B"}
