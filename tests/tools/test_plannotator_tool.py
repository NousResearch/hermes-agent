"""Tests for tools/plannotator_tool.py."""

import json
from subprocess import CompletedProcess
from unittest.mock import patch

from tools.plannotator_tool import plannotator_session_tool


def test_annotate_requires_absolute_artifact_path():
    result = json.loads(
        plannotator_session_tool(
            {
                "action": "annotate",
                "artifact_path": "relative.md",
            }
        )
    )

    assert "must be an absolute path" in result["error"]


def test_review_uses_default_bridge_template_without_target():
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="URL=https://plannotator-demo.example/\nPID=321\nLOG=/tmp/plannotator.log\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(plannotator_session_tool({"action": "review"}))

    assert result["success"] is True
    assert result["url"] == "https://plannotator-demo.example/"
    command = run_mock.call_args.args[0][2]
    assert "start_session.py review" in command
    assert result["suggested_message"].startswith("Temporary review URL:")


def test_review_with_target_and_strategy_passes_values_into_template():
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="URL=https://review.example/\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(
            plannotator_session_tool(
                {
                    "action": "review",
                    "review_target": "https://github.com/example/repo/pull/7",
                    "exposure_strategy": "tailscale-funnel",
                    "command_template": "launch-review {review_target_arg} --strategy {exposure_strategy}",
                }
            )
        )

    assert result["success"] is True
    command = run_mock.call_args.args[0][2]
    assert "launch-review" in command
    assert "https://github.com/example/repo/pull/7" in command
    assert "--strategy tailscale-funnel" in command
    assert run_mock.call_args.kwargs["env"]["PLANNOTATOR_EXPOSURE_STRATEGY"] == "tailscale-funnel"


def test_last_action_reports_launcher_failure():
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=2,
        stdout="",
        stderr="unsupported",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed):
        result = json.loads(plannotator_session_tool({"action": "last"}))

    assert "launcher failed" in result["error"]
    assert result["stderr"] == "unsupported"
