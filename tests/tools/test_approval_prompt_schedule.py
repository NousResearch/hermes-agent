"""The plain-input approval fallback (no prompt_toolkit callback) on a schedule card."""
from unittest.mock import patch

from tools.approval import prompt_dangerous_approval


def test_plain_input_schedule_prompt_shows_the_job_not_the_synthetic_command(capsys):
    """It prints the job (in the schedule header) and never the synthetic ``<cronjob_manage>`` placeholder
    the gate passes as the command; a command prompt still prints its command."""
    with patch("builtins.input", return_value="deny"):
        prompt_dangerous_approval("<cronjob_manage> (schedule a new job)",
                                  'Schedule "water" (every 2h): drink water', kind="schedule")
    rendered = capsys.readouterr().out
    assert "drink water" in rendered and "<cronjob_manage>" not in rendered

    with patch("builtins.input", return_value="deny"):
        prompt_dangerous_approval("rm -rf /tmp/x", "recursive delete")
    assert "rm -rf /tmp/x" in capsys.readouterr().out
