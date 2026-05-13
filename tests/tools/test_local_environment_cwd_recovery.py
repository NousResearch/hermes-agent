"""Regression tests for local terminal cwd recovery."""

from pathlib import Path

from tools.environments.local import LocalEnvironment


def test_local_environment_recovers_from_corrupt_popen_cwd(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    try:
        env.cwd = "%s"
        result = env.execute("pwd", cwd=str(tmp_path), timeout=10)

        assert result["returncode"] == 0
        assert result["output"].strip() == str(tmp_path)
        assert Path(env.cwd).is_dir()
    finally:
        env.cleanup()


def test_local_environment_ignores_literal_percent_s_cwd_marker(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    try:
        env.cwd = str(tmp_path)
        marker = env._cwd_marker
        result = {"output": f"{marker}%s{marker}\n"}

        env._update_cwd(result)

        assert env.cwd == str(tmp_path)
        assert result["output"] == ""
    finally:
        env.cleanup()
