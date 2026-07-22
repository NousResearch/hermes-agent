"""Tests for terminal process timeout, interactive prompt detection, GUI app detection, CPU idle detection, and non-interactive env injection."""

import os
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.environments.base import (
    _strip_ansi,
    _detect_interactive_prompt,
    _detect_gui_child_process,
    _inspect_process_tree_activity,
    _NONINTERACTIVE_ENV_DEFAULTS,
    _BoundedOutputCollector,
    BaseEnvironment,
)
from tools.environments.local import _make_run_env


def test_strip_ansi():
    raw = "\x1b[32mSelect option [1-3]: \x1b[0m"
    assert _strip_ansi(raw) == "Select option [1-3]: "


def test_detect_interactive_prompt():
    # Y/N prompts
    assert _detect_interactive_prompt("Do you want to continue? [Y/n]") is not None
    assert _detect_interactive_prompt("Overwrite file? (y/N) ") is not None

    # Password prompts
    assert _detect_interactive_prompt("[sudo] password for admin:") is not None
    assert _detect_interactive_prompt("Enter passphrase for key '/root/.ssh/id_rsa':") is not None

    # Menu choices
    assert _detect_interactive_prompt("Select an option:") is not None
    assert _detect_interactive_prompt("Enter choice [1-5]:") is not None

    # Press key
    assert _detect_interactive_prompt("Press enter to continue...") is not None

    # Normal output should NOT match
    assert _detect_interactive_prompt("Successfully built 15 targets.") is None
    assert _detect_interactive_prompt("Processing item 42 of 100...") is None


def test_noninteractive_env_defaults():
    # Check that defaults dictionary contains essential non-interactive keys
    assert _NONINTERACTIVE_ENV_DEFAULTS["CI"] == "1"
    assert _NONINTERACTIVE_ENV_DEFAULTS["DEBIAN_FRONTEND"] == "noninteractive"
    assert _NONINTERACTIVE_ENV_DEFAULTS["GIT_TERMINAL_PROMPT"] == "0"
    assert _NONINTERACTIVE_ENV_DEFAULTS["PIP_NO_INPUT"] == "1"
    assert _NONINTERACTIVE_ENV_DEFAULTS["PYTHONUNBUFFERED"] == "1"

    # Verify _make_run_env includes them
    run_env = _make_run_env({})
    assert run_env.get("CI") == "1"
    assert run_env.get("GIT_TERMINAL_PROMPT") == "0"
    assert run_env.get("PIP_NO_INPUT") == "1"


def test_bounded_output_collector_tail():
    collector = _BoundedOutputCollector(max_chars=1000)
    collector.append("Line 1\n")
    collector.append("Line 2\n")
    collector.append("Line 3\n")

    tail = collector.get_tail(chars=15)
    assert "Line 3" in tail
    assert collector.last_update_time > 0


class DummyEnvironment(BaseEnvironment):
    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        pass

    def cleanup(self):
        pass


def test_environment_env_injection():
    env = DummyEnvironment(cwd=".", timeout=60, env={"CUSTOM_VAR": "value"})
    assert env.env["CUSTOM_VAR"] == "value"
    assert env.env["CI"] == "1"
    assert env.env["GIT_TERMINAL_PROMPT"] == "0"


def test_wait_for_process_prompt_detection():
    env = DummyEnvironment(cwd=".", timeout=60)
    
    # Mock ProcessHandle
    proc = MagicMock()
    proc.poll.return_value = None  # Process is still running
    proc.pid = 12345
    proc.stdout = None

    # Mock output collector returning prompt
    with patch("tools.environments.base._BoundedOutputCollector") as MockCollector:
        mock_collector_inst = MagicMock()
        MockCollector.return_value = mock_collector_inst
        mock_collector_inst.get_tail.return_value = "Do you want to proceed? [y/N]"
        mock_collector_inst.last_update_time = time.monotonic() - 6.0  # 6s idle
        mock_collector_inst.render.side_effect = lambda suffix="": "Do you want to proceed? [y/N]" + suffix

        res = env._wait_for_process(proc, timeout=60, inactivity_timeout=60)
        assert res["returncode"] == 124
        assert "interactive prompt" in res["output"]


def test_wait_for_process_inactivity_timeout():
    env = DummyEnvironment(cwd=".", timeout=60)
    
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 12345
    proc.stdout = None

    with patch("tools.environments.base._BoundedOutputCollector") as MockCollector, \
         patch("tools.environments.base._inspect_process_tree_activity") as mock_inspect:
        mock_collector_inst = MagicMock()
        MockCollector.return_value = mock_collector_inst
        mock_collector_inst.get_tail.return_value = "Building project..."
        mock_collector_inst.last_update_time = time.monotonic() - 15.0  # 15s idle
        mock_collector_inst.render.side_effect = lambda suffix="": "Building project..." + suffix
        # Active CPU usage (50.0%) so inactivity timeout triggers instead of 10s idle check
        mock_inspect.return_value = {"alive": True, "total_cpu": 50.0, "gui_app": None, "child_count": 1}

        res = env._wait_for_process(proc, timeout=60, inactivity_timeout=10)
        assert res["returncode"] == 124
        assert "output inactivity" in res["output"]


def test_wait_for_process_gui_detection():
    env = DummyEnvironment(cwd=".", timeout=60)
    
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 12345
    proc.stdout = None

    with patch("tools.environments.base._BoundedOutputCollector") as MockCollector, \
         patch("tools.environments.base._detect_gui_child_process") as mock_detect_gui:
        mock_collector_inst = MagicMock()
        MockCollector.return_value = mock_collector_inst
        mock_collector_inst.get_tail.return_value = "Extracted zip..."
        mock_collector_inst.last_update_time = time.monotonic() - 4.0
        mock_collector_inst.render.side_effect = lambda suffix="": "Extracted zip..." + suffix
        mock_detect_gui.return_value = "SoundVolumeView.exe"

        res = env._wait_for_process(proc, timeout=60, inactivity_timeout=60)
        assert res["returncode"] == 0
        assert "Detected desktop GUI application 'SoundVolumeView.exe'" in res["output"]


def test_wait_for_process_cpu_idle_detection():
    env = DummyEnvironment(cwd=".", timeout=60)

    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 12345
    proc.stdout = None

    with patch("tools.environments.base._BoundedOutputCollector") as MockCollector, \
         patch("tools.environments.base._inspect_process_tree_activity") as mock_inspect:
        mock_collector_inst = MagicMock()
        MockCollector.return_value = mock_collector_inst
        mock_collector_inst.get_tail.return_value = "Waiting on input..."
        mock_collector_inst.last_update_time = time.monotonic() - 11.0  # 11s idle
        mock_collector_inst.render.side_effect = lambda suffix="": "Waiting on input..." + suffix
        # 0.0% CPU usage (sleeping/idle process)
        mock_inspect.return_value = {"alive": True, "total_cpu": 0.0, "gui_app": None, "child_count": 0}

        res = env._wait_for_process(proc, timeout=60, inactivity_timeout=60)
        assert res["returncode"] == 124
        assert "completely idle (0% CPU" in res["output"]
