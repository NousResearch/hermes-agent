"""A background handle must refer to the script, not a launcher or log watcher."""
import json
from unittest.mock import patch

import pytest

from tests.tools.test_terminal_foreground_timeout_cap import _make_env_config
from tools.terminal_tool import terminal_tool


@pytest.mark.parametrize(('background', 'timeout'), [(False, 30), (False, 9999), (True, 30)])
def test_detached_script_refused_before_any_part_runs(tmp_path, background, timeout):
    # A compound launcher is especially easy to mistake for a successful foreground job.
    # Use a harmless fake tmux: on old code the foreground case really writes the marker.
    marker = tmp_path / 'launched'
    command = f'tmux() {{ touch {marker}; }}; true && tmux new-session -d -s build "zsh $PWD/build.sh" && true'
    with (patch('tools.terminal_tool._get_env_config', return_value=_make_env_config(cwd=str(tmp_path))),
          patch('tools.terminal_tool._start_cleanup_thread')):
        reply = json.loads(terminal_tool(command=command, background=background, timeout=timeout))
    assert 'background=true' in (reply.get('error') or '')
    assert not marker.exists()


@pytest.mark.parametrize(('command', 'blocked'), [
    ('tmux new -ds build ./build.sh', True),
    ('tmux -L jobs new-session -s build -d bash ./tests', True),
    ('tmux new-session -d -s build "zsh /work/build.sh"', True),
    ("echo 'tmux new-session -d -s build bash build.sh'", False),
    ('cat <<\'END\'\ntmux new-session -d -s build bash build.sh\nEND', False),
    ('tmux new-session -d -s build', False),
    ('tmux new-session -d -s review "hermes --continue"', False),
    ('tmux new-session -d -s review bash', False),
    ('tmux new-session -s build bash build.sh', False),
    ('tmux has-session -t build', False),
    ('tmux send-keys -t build "bash build.sh" Enter', False),
])
def test_only_detached_script_payloads(command, blocked):
    from tools.terminal_tool_guards import detached_tmux_script_guidance
    assert bool(detached_tmux_script_guidance(command)) is blocked
