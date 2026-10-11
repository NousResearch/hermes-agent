"""Structured outcome flags on the terminal result (issue #93700): a bare
``exit_code`` was the only machine-readable success signal, so a signal death
and a cut payload reached the model as prose — or, when spill redaction drops
the handle, as nothing at all. These pin the two flags an agent can branch on
instead of reading the English note.
"""

import json

from tools.terminal_tool_result import finalize_foreground_result


def _finalize(result: dict) -> dict:
    """Run the real tool-layer finalizer (env=None: no cwd hand-off involved)."""
    return json.loads(finalize_foreground_result(
        command="echo hi", result=result, env=None, env_type="local",
        effective_task_id="t", task_id="t", session_id="s", session_key="k",
        workdir=None, command_cwd=None, approval_note=None))


def test_signal_death_is_structured_with_its_confidence():
    """``signal`` carries the confidence the exit code actually has: definite for
    a subprocess ``-signum``, hedged for the shell's 128+signum band, and absent
    wherever the prose note is absent (an application's own exit code stays
    unlabeled)."""
    definite = _finalize({"output": "", "returncode": -15})
    assert definite["signal"] == {"number": 15, "name": "SIGTERM", "definite": True}

    shell_band = _finalize({"output": "", "returncode": 143})
    assert shell_band["signal"] == {"number": 15, "name": "SIGTERM", "definite": False}

    # 168 = 128+40: uncurated signum, the prose stays silent -> no structured claim.
    ambiguous = _finalize({"output": "", "returncode": 128 + 40})
    assert "signal" not in ambiguous
    assert "exit_code_meaning" not in ambiguous

    clean = _finalize({"output": "ok", "returncode": 0})
    assert "signal" not in clean


def test_output_truncated_reports_a_cut_payload_without_a_spill_handle():
    """The flag says "what you got was cut" even when the spill handle is gone,
    and also covers a cut the collector never saw."""
    spilled = _finalize({"output": "x", "returncode": 0, "output_total_chars": 10 ** 6,
                         "full_output_path": "/nonexistent/spill.log"})
    assert spilled["output_truncated"] is True
    # Spill redaction failed, so the handle and its note are dropped — the flag is
    # then the only signal the payload is incomplete.
    assert "full_output_path" not in spilled

    # Output the tool-layer truncation pass cuts on its own (no collector overflow).
    cut = _finalize({"output": "a" * 200_000, "returncode": 0})
    assert cut["output_truncated"] is True

    clean = _finalize({"output": "ok", "returncode": 0})
    assert "output_truncated" not in clean
