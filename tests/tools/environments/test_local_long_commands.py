"""Git Bash must execute complete wrappers beyond its Windows -c limit (#156436)."""

import json
import os
from pathlib import Path

import pytest

from tests.tools._child_env_fixtures import child_env as _isolated_child_env
from tools.environments.local import LocalEnvironment


pytestmark = pytest.mark.platforms("windows")


@pytest.fixture(name="local_shell")
def _local_shell(_isolated_child_env, monkeypatch):  # health: allow F811 -- pytest injects the imported fixture by its name
    script_dir = _isolated_child_env / "shell files ' 雪"
    script_dir.mkdir()
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(script_dir))
    env = LocalEnvironment(cwd=str(_isolated_child_env), timeout=30)
    try:
        yield env
    finally:
        env.cleanup()


@pytest.mark.parametrize("padding", ["short", "x" * 9000, "\U0001f600" * 5000],
                         ids=["short", "ascii", "utf16-surrogate-pairs"])
def test_complete_command_preserves_stdin_status_and_session(local_shell, padding):
    env = local_shell
    script_dir = Path(env.get_temp_dir())
    original_files = set(script_dir.iterdir())
    next_cwd = Path(env.cwd) / "next"
    next_cwd.mkdir()
    payload = "literal $value 'quotes' 雪"
    result = env.execute(
        f"# {padding}\n"
        "cd next\n"
        "export LONG_COMMAND_STATE=preserved\n"
        "IFS= read -r payload\n"
        "printf '%s\\n' \"$payload\"\n"
        "(exit 23)",
        stdin_data=payload + "\n",
    )
    assert result["returncode"] == 23, result
    assert result["output"].strip() == payload
    assert Path(env.cwd).resolve() == next_cwd.resolve()
    followup = env.execute("printf '%s' \"$LONG_COMMAND_STATE\"")
    assert followup["returncode"] == 0, followup
    assert followup["output"] == "preserved"
    assert set(script_dir.iterdir()) == original_files


def test_many_config_declared_keys_keep_terminal_and_file_reads_working(local_shell):
    env = local_shell
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(exist_ok=True)
    names = [f"CUSTOM_PROVIDER_{index:02d}_API_KEY" for index in range(20)]
    (home / "config.yaml").write_text(json.dumps({
        "providers": {f"provider{index}": {"key_env": name}
                      for index, name in enumerate(names)},
    }), encoding="utf-8")
    source = Path(env.cwd) / "payload.txt"
    source.write_text("file contents\n", encoding="utf-8", newline="\n")
    result = env.execute("printf 'terminal works\\n'; cat payload.txt")
    assert result["returncode"] == 0, result
    assert result["output"] == "terminal works\nfile contents\n"
