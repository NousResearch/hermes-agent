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


def test_prepare_returns_reserved_url_without_waiting():
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="HOST=plannotator-demo.example\nURL=https://plannotator-demo.example/\n",
        stderr="",
    )

    with patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(plannotator_session_tool({"action": "prepare"}))

    assert result["success"] is True
    assert result["host"] == "plannotator-demo.example"
    assert result["url"] == "https://plannotator-demo.example/"
    assert result["waited_for_completion"] is False
    assert "start_session.py prepare" in run_mock.call_args.args[0][2]


def test_review_uses_default_bridge_template_without_target():
    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout="HOST=plannotator-demo.example\nURL=https://plannotator-demo.example/\nPID=321\nLOG=/tmp/plannotator.log\n",
        stderr="",
    )

    with (
        patch("tools.exposure_helpers.subprocess.run", return_value=completed) as run_mock,
        patch("tools.plannotator_tool._wait_for_plannotator_completion", return_value={"completed": True, "status": "completed"}) as wait_mock,
    ):
        result = json.loads(plannotator_session_tool({"action": "review"}))

    assert result["success"] is True
    assert result["host"] == "plannotator-demo.example"
    assert result["url"] == "https://plannotator-demo.example/"
    command = run_mock.call_args.args[0][2]
    assert "start_session.py review" in command
    assert result["suggested_message"].startswith("Temporary review URL:")
    assert result["waited_for_completion"] is True
    wait_mock.assert_called_once_with(
        pid="321",
        log_path="/tmp/plannotator.log",
        timeout_seconds=3600,
        poll_interval_seconds=2.0,
    )


def test_review_with_target_strategy_and_fixed_host_passes_env_values():
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
                    "fixed_host": "review-fixed.example.com",
                    "command_template": "launch-review {review_target_arg} --strategy {exposure_strategy}",
                    "wait_for_completion": False,
                }
            )
        )

    assert result["success"] is True
    command = run_mock.call_args.args[0][2]
    assert "launch-review" in command
    assert "https://github.com/example/repo/pull/7" in command
    assert "--strategy tailscale-funnel" in command
    assert run_mock.call_args.kwargs["env"]["PLANNOTATOR_EXPOSURE_STRATEGY"] == "tailscale-funnel"
    assert run_mock.call_args.kwargs["env"]["PLANNOTATOR_HOST"] == "review-fixed.example.com"
    assert result["waited_for_completion"] is False


def test_inline_review_prepares_sends_and_waits():
    prepare_result = {
        "success": True,
        "action": "prepare",
        "host": "review-fixed.example.com",
        "url": "https://review-fixed.example.com/",
        "suggested_message": "Temporary review URL:\nhttps://review-fixed.example.com/",
        "waited_for_completion": False,
    }
    review_result = {
        "success": True,
        "action": "review",
        "host": "review-fixed.example.com",
        "url": "https://review-fixed.example.com/",
        "completed": True,
        "waited_for_completion": True,
        "final_log": "Code review completed — no changes requested.\n",
    }

    with (
        patch("tools.plannotator_tool._launch_plannotator", side_effect=[prepare_result, review_result]) as launch_mock,
        patch("tools.plannotator_tool._send_inline_url_message", return_value={"success": True, "message_id": 123}) as send_mock,
    ):
        result = json.loads(plannotator_session_tool({"action": "inline_review"}))

    assert result["success"] is True
    assert result["inline_message_sent"] is True
    assert result["prepared_host"] == "review-fixed.example.com"
    assert result["send_message_result"]["success"] is True
    assert launch_mock.call_count == 2
    assert launch_mock.call_args_list[1].kwargs == {}
    second_args = launch_mock.call_args_list[1].args[0]
    assert second_args["action"] == "review"
    assert second_args["fixed_host"] == "review-fixed.example.com"
    assert second_args["wait_for_completion"] is True
    send_mock.assert_called_once()


def test_inline_review_returns_error_if_send_fails():
    prepare_result = {
        "success": True,
        "action": "prepare",
        "host": "review-fixed.example.com",
        "url": "https://review-fixed.example.com/",
        "suggested_message": "Temporary review URL:\nhttps://review-fixed.example.com/",
        "waited_for_completion": False,
    }

    with (
        patch("tools.plannotator_tool._launch_plannotator", return_value=prepare_result) as launch_mock,
        patch("tools.plannotator_tool._send_inline_url_message", return_value={"error": "No active messaging session context found."}),
    ):
        result = json.loads(plannotator_session_tool({"action": "inline_review"}))

    assert "Failed to send prepared Plannotator URL message" in result["error"]
    assert result["prepared_host"] == "review-fixed.example.com"
    assert launch_mock.call_count == 1


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


def test_wait_returns_final_log_when_process_exits(tmp_path):
    log_path = tmp_path / "review.log"
    log_path.write_text("Code review completed — looks good.\n")

    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout=f"URL=https://review.example/\nPID=777\nLOG={log_path}\n",
        stderr="",
    )

    with (
        patch("tools.exposure_helpers.subprocess.run", return_value=completed),
        patch("tools.plannotator_tool._pid_is_running", side_effect=[True, False]),
        patch("tools.plannotator_tool.time.sleep", return_value=None),
    ):
        result = json.loads(
            plannotator_session_tool(
                {
                    "action": "review",
                    "completion_timeout_seconds": 60,
                    "poll_interval_seconds": 0.25,
                }
            )
        )

    assert result["completed"] is True
    assert result["timed_out"] is False
    assert result["status"] == "completed"
    assert "looks good" in result["final_log"]


def test_wait_times_out_and_preserves_last_log(tmp_path):
    log_path = tmp_path / "review.log"
    log_path.write_text("Still waiting for annotations...\n")

    completed = CompletedProcess(
        args=["bash", "-lc", "echo"],
        returncode=0,
        stdout=f"URL=https://review.example/\nPID=888\nLOG={log_path}\n",
        stderr="",
    )

    monotonic_values = iter([0.0, 0.0, 61.0])

    with (
        patch("tools.exposure_helpers.subprocess.run", return_value=completed),
        patch("tools.plannotator_tool._pid_is_running", return_value=True),
        patch("tools.plannotator_tool.time.sleep", return_value=None),
        patch("tools.plannotator_tool.time.monotonic", side_effect=lambda: next(monotonic_values)),
    ):
        result = json.loads(
            plannotator_session_tool(
                {
                    "action": "review",
                    "completion_timeout_seconds": 60,
                    "poll_interval_seconds": 0.25,
                }
            )
        )

    assert result["completed"] is False
    assert result["timed_out"] is True
    assert result["status"] == "timeout"
    assert result["session_still_running"] is True
    assert "Still waiting" in result["final_log"]
