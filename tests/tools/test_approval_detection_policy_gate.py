"""Terminal-gate coverage for security-policy config keys (#104697 review).

Pins two review findings against ``tools/approval_detection.py``:

* the policy patterns must match in time bounded by the input length (the first
  cut spelled global flags as an ambiguous repetition and took seconds on a long
  flag run, holding the GIL before argparse could reject the command), and
* every spelling argparse actually accepts for ``approvals suggest --apply``
  must trip the gate — argparse abbreviation made ``--a`` reach ``--apply``
  while the pattern only matched the full spelling.
"""

import time

import pytest

from tools.approval_detection import detect_dangerous_command

# A flag run long enough to expose exponential backtracking (the reviewed shape took
# ~4 s at 32 flags) but trivial for a possessive-quantifier pattern. The bound is
# deliberately loose: linear matching is sub-millisecond, so it only catches a real
# backtracking regression, never a slow runner.
_LONG_FLAG_RUN = "hermes " + "--x " * 32 + "config show"
_BOUND_SECONDS = 1.0


def test_long_global_flag_run_is_checked_in_bounded_time():
    started = time.perf_counter()
    dangerous, _, _ = detect_dangerous_command(_LONG_FLAG_RUN)
    elapsed = time.perf_counter() - started

    assert dangerous is False
    assert elapsed < _BOUND_SECONDS, (
        f"policy patterns took {elapsed:.3f}s on a {_LONG_FLAG_RUN.count('--x')}-flag run; "
        "the global-flag fragment must use the possessive _GLOBAL_FLAGS form"
    )


@pytest.mark.parametrize("flag", ["--a", "--ap", "--app", "--appl", "--apply"])
def test_approvals_suggest_apply_abbreviations_detected(flag):
    """argparse accepts every prefix of ``--apply`` and writes command_allowlist."""
    dangerous, _, description = detect_dangerous_command(f"hermes approvals suggest {flag} 1")

    assert dangerous is True, f"{flag} escaped the terminal gate"
    assert "command_allowlist" in description


@pytest.mark.parametrize(
    "command",
    [
        "hermes config set approvals.mode off",
        "hermes -p ade config set --force security.redact_secrets false",
        "python -m hermes_cli.main config unset command_allowlist",
        "hermes approvals off",
        "hermes approvals suggest --a 1,3",
        "hermes -p ade config set --fo approvals.mode off",
        "hermes -p ade config set --for approvals.mode off",
        "hermes -p ade config set --forc approvals.mode off",
    ],
)
def test_policy_spellings_still_detected_under_global_flags(command):
    """Reusing _GLOBAL_FLAGS must not drop the profile-flag and module-form spellings."""
    dangerous, _, _ = detect_dangerous_command(command)

    assert dangerous is True, command
