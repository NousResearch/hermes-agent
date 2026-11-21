"""An approval timeout logs the pattern key, so its cause can be traced later."""

import logging
from unittest.mock import patch

from tools import approval as mod


def test_cli_timeout_logs_pattern_not_command(caplog):
    mod._session_approved.clear()
    mod._permanent_approved.clear()

    cfg = {"approvals": {"mode": "manual"}}
    with patch.dict("os.environ", {"HERMES_INTERACTIVE": "1"}), \
            patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
            caplog.at_level(logging.WARNING, logger="tools.approval"):
        result = mod.check_all_command_guards(
            "rm -rf /var/data", "local",
            approval_callback=lambda *a, **kw: "timeout",
        )

    assert result.get("outcome") == "timeout"
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and r.getMessage().startswith("Approval timeout")]
    assert len(warnings) == 1
    assert result["pattern_key"] in warnings[0]
    assert "/var/data" not in warnings[0]
