"""Model-visible pipeline_exit_codes when an earlier pipe stage fails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.environments.local import LocalEnvironment
from tools.terminal_hints import PIPELINE_EARLY_STAGE_FAILED_HINT
from tools.terminal_tool_result import _pipeline_exit_codes_for_model, finalize_foreground_result


@pytest.fixture
def profile_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


def _finalize(env: LocalEnvironment, command: str, cwd: Path) -> dict:
    raw = env.execute(command, cwd=str(cwd))
    return json.loads(
        finalize_foreground_result(
            command=command,
            result=raw,
            env=env,
            env_type="local",
            effective_task_id="pipe-exit-codes",
            task_id="pipe-exit-codes",
            session_id="pipe-exit-codes",
            session_key="pipe-exit-codes",
            workdir=None,
            command_cwd=str(cwd),
            approval_note=None,
        )
    )


@pytest.fixture
def terminal_env(profile_env: Path) -> LocalEnvironment:
    env = LocalEnvironment(cwd=str(profile_env), timeout=60)
    try:
        yield env
    finally:
        env.cleanup()


def test_exit7_pipe_tail_surfaces_codes_and_hint(terminal_env, profile_env):
    command = 'python3 -c "import sys; sys.exit(7)" 2>&1 | tail -5'
    result = _finalize(terminal_env, command, profile_env)
    assert result["exit_code"] == 0
    assert result["pipeline_exit_codes"] == [7, 0]
    assert result["hint"] == PIPELINE_EARLY_STAGE_FAILED_HINT


def test_false_pipe_true_masks_exit_code(terminal_env, profile_env):
    result = _finalize(terminal_env, "false | true", profile_env)
    assert result["exit_code"] == 0
    assert result["pipeline_exit_codes"] == [1, 0]
    assert "pipeline" in (result.get("hint") or "").lower()


def test_true_pipe_false_only_last_fails(terminal_env, profile_env):
    result = _finalize(terminal_env, "true | false", profile_env)
    assert result["exit_code"] == 1
    assert "pipeline_exit_codes" not in result
    assert result.get("hint") != PIPELINE_EARLY_STAGE_FAILED_HINT


def test_all_stages_zero_pipe_omits_field(terminal_env, profile_env):
    result = _finalize(terminal_env, "echo ok | cat", profile_env)
    assert result["exit_code"] == 0
    assert "pipeline_exit_codes" not in result
    assert result.get("hint") != PIPELINE_EARLY_STAGE_FAILED_HINT


def test_plain_command_no_pipeline_field(terminal_env, profile_env):
    command = 'python3 -c "import sys; sys.exit(3)"'
    result = _finalize(terminal_env, command, profile_env)
    assert result["exit_code"] == 3
    assert "pipeline_exit_codes" not in result


def test_heredoc_skips_capture(terminal_env, profile_env):
    command = "cat <<'EOF'\nline one\nEOF"
    result = _finalize(terminal_env, command, profile_env)
    assert result["exit_code"] == 0
    assert "pipeline_exit_codes" not in result


def test_trailing_backslash_skips_capture(terminal_env, profile_env):
    command = "echo hi \\"
    result = _finalize(terminal_env, command, profile_env)
    assert result["exit_code"] == 0
    assert "pipeline_exit_codes" not in result


def test_import_error_pipe_tail_hint_without_failure_shapes(terminal_env, profile_env):
    command = 'python3 -c "import nope_module_xyz" 2>&1 | tail -3'
    result = _finalize(terminal_env, command, profile_env)
    assert result["exit_code"] == 0
    assert result["pipeline_exit_codes"] == [1, 0]
    assert result["hint"] == PIPELINE_EARLY_STAGE_FAILED_HINT


def test_yes_pipe_head_sigpipe_not_surfaced(terminal_env, profile_env):
    result = _finalize(terminal_env, "yes | head -1", profile_env)
    assert result["exit_code"] == 0
    assert "pipeline_exit_codes" not in result
    assert result.get("hint") != PIPELINE_EARLY_STAGE_FAILED_HINT


def test_seq_pipe_head_sigpipe_not_surfaced(terminal_env, profile_env):
    result = _finalize(terminal_env, "seq 1 200000 | head -1", profile_env)
    assert result["exit_code"] == 0
    assert "pipeline_exit_codes" not in result
    assert result.get("hint") != PIPELINE_EARLY_STAGE_FAILED_HINT


def test_real_failure_before_sigpipe_keeps_full_array(terminal_env, profile_env):
    result = _finalize(terminal_env, "false | yes | head -1", profile_env)
    assert result["exit_code"] == 0
    assert result["pipeline_exit_codes"] == [1, 141, 0]
    assert result["hint"] == PIPELINE_EARLY_STAGE_FAILED_HINT


def test_both_stages_failed_keeps_nonzero_hint_path(terminal_env, profile_env):
    result = _finalize(terminal_env, "false | false", profile_env)
    assert result["exit_code"] == 1
    assert result["pipeline_exit_codes"] == [1, 1]
    assert result.get("hint") != PIPELINE_EARLY_STAGE_FAILED_HINT


@pytest.mark.parametrize(
    ("pipeline_status", "expected"),
    [
        (None, None),
        ([], None),
        ([0], None),
        ([0, 0], None),
        ([141, 0], None),
        ([141, 141, 0], None),
        ([0, 1], None),
        ([1, 0], [1, 0]),
        ([141, 1, 0], [141, 1, 0]),
        ([2, 141, 0], [2, 141, 0]),
    ],
)
def test_pipeline_exit_codes_for_model(pipeline_status, expected):
    assert _pipeline_exit_codes_for_model(pipeline_status) == expected