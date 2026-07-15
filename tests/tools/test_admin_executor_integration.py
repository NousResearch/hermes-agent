"""OPT-IN real-UAC integration tests for elevated execution.

These tests drive the REAL ShellExecuteExW(runas) UAC prompt on an
interactive Windows session.  They are intentionally NOT part of the default
suite: they require a human at the keyboard to click the UAC dialog, and they
must not block ordinary CI.

Run with:

    HERMES_RUN_ELEVATED_INTEGRATION=1 python -m pytest tests/tools/test_admin_executor_integration.py -v -s

``test_real_uac_approved`` pops a UAC prompt; approve it (click Yes).
``test_real_uac_cancelled`` pops a UAC prompt; dismiss it (click No).

Everything EXCEPT the interactive UAC dialog is already covered by
``test_admin_executor.py`` (fake API adapter) and the real-helper Unicode
tests in ``TestElevatedHelperReal`` (which run without UAC on Windows).  The
two tests here prove the last untested link: the actual ShellExecuteExW ->
UAC -> elevated helper -> CreateProcessW chain with real privilege
elevation.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

_RUN_INTEGRATION = os.getenv("HERMES_RUN_ELEVATED_INTEGRATION") == "1"


def _elevated_echo(marker: str, timeout: int = 30) -> dict:
    """Run a trivial elevated command that echoes a marker via the real UAC path."""
    from tools.admin_executor import execute_elevated

    return execute_elevated(f"echo {marker}", timeout=timeout)


@unittest.skipUnless(
    sys.platform == "win32" and _RUN_INTEGRATION,
    "opt-in: set HERMES_RUN_ELEVATED_INTEGRATION=1 on an interactive Windows session",
)
class TestRealUACIntegration(unittest.TestCase):
    def test_real_uac_approved(self):
        """Approve the UAC prompt: the elevated helper runs and echoes back."""
        marker = "uac-ok-中文😀"
        result = _elevated_echo(marker)
        self.assertEqual(result["exit_code"], 0, result.get("error"))
        self.assertIsNone(result["error"])
        self.assertIn(marker, result["output"])

    def test_real_uac_cancelled(self):
        """Dismiss the UAC prompt: a distinct 'cancelled' result is returned,
        never conflated with launch success or a generic failure."""
        result = _elevated_echo("should-not-run")
        self.assertEqual(result["exit_code"], -1)
        self.assertEqual(result.get("error_kind"), "cancelled")
        self.assertIn("cancelled", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
