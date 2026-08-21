"""Windows Git Bash command transport preserves the user's command bytes."""

import glob
import platform

import pytest

from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows",
    reason="exercises the Windows argv to Git Bash boundary",
)


@pytest.fixture
def env(tmp_path):
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    yield environment
    environment.cleanup()


def test_backslash_content_survives_byte_exact(env, tmp_path):
    target = tmp_path / "content.txt"
    cmd = r"printf '%s' 'a\nb\\d' > " + "'" + target.as_posix() + "'"

    result = env.execute(cmd, timeout=30)

    assert result["returncode"] == 0
    assert target.read_bytes() == rb"a\nb\\d"


def test_heredoc_multiline_backslashes_survive(env, tmp_path):
    target = tmp_path / "heredoc.txt"
    cmd = (
        f"cat > '{target.as_posix()}' <<'EOF'\n"
        "line1 C:\\Users\\x\n"
        "line2\n"
        "EOF"
    )

    result = env.execute(cmd, timeout=30)

    assert result["returncode"] == 0
    assert target.read_bytes() == b"line1 C:\\Users\\x\nline2\n"


def test_exit_codes_and_environment_snapshot_still_propagate(env):
    assert env.execute("false", timeout=30)["returncode"] == 1
    env.execute("export HERMES_TRANSPORT_PROBE=portable", timeout=30)
    result = env.execute("echo relay=$HERMES_TRANSPORT_PROBE", timeout=30)
    assert (result.get("output") or "").strip() == "relay=portable"


def test_transport_file_is_removed_before_command_runs(env):
    snap_dir = str(env._snapshot_path.rsplit("/", 1)[0])
    result = env.execute(
        f"grep -r HERMES_SELF_MATCH_CANARY '{snap_dir}' ; true", timeout=30
    )

    assert "HERMES_SELF_MATCH_CANARY" not in (result.get("output") or "")
    assert glob.glob(f"{env._snapshot_path}.cmd.*") == []
