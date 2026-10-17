"""A foreground command whose ONLY backgrounding is a trailing `&` runs as terminal(background=true).

Refusing it cost a full model turn to resend the same command with background=true (488 refusals in
30 days of one user's sessions). Ambiguous shapes (inline `a & b`, nohup/disown/setsid, long-lived
servers without `&`) keep the refusal: the caller may expect later commands to run after a check.
"""
import json
import shutil
import subprocess
import time
from unittest.mock import patch

import pytest


def _env_config(cwd):
    return {
        "env_type": "local", "timeout": 180, "cwd": str(cwd), "host_cwd": None, "modal_mode": "auto",
        "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": "",
    }


def test_trailing_amp_runs_as_tracked_background_through_the_same_guards(tmp_path, monkeypatch):
    """Real local backend + registry: one tracked session, the payload runs once, the approval
    guard sees the command that actually runs, and the result says what Hermes did."""
    from tools.process_registry import process_registry
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    marker = tmp_path / "ran"
    seen = []

    def guard(command, env_type, **kw):
        seen.append(command)
        return {"approved": True}

    with patch("tools.terminal_tool._get_env_config", return_value=_env_config(tmp_path)), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", side_effect=guard):
        result = json.loads(terminal_tool(command=f"echo x >> {marker} && sleep 0.2 &"))

    assert result.get("error") is None, result
    assert result["session_id"].startswith("proc_")
    assert "trailing & was converted to background=true" in result["auto_backgrounded"]
    assert "hint" not in result  # the silent-bg hint would tell it to re-launch a second copy
    assert seen == [f"echo x >> {marker} && sleep 0.2"]
    session = process_registry.get(result["session_id"])
    assert session.command == f"echo x >> {marker} && sleep 0.2"
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.read_text(encoding="utf-8-sig").count("x") == 1


@pytest.mark.parametrize("command, expected", [
    ("python3 -m http.server 8765 & curl -s localhost:8765", "'&' backgrounding"),
    ("sleep 1 & sleep 2 &", "'&' backgrounding"),
    ("nohup ./worker.sh > /dev/null 2>&1 &", "nohup/disown/setsid"),
    ("python3 -m http.server 8765", "long-lived server"),
    # Approval keys on the `&` here (gateway outside its service manager): no conversion.
    ("hermes gateway run &", "'&' backgrounding"),
])
def test_ambiguous_backgrounding_is_still_refused_unchanged(tmp_path, command, expected):
    from tools.terminal_tool import terminal_tool

    with patch("tools.terminal_tool._get_env_config", return_value=_env_config(tmp_path)), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._acquire_env", side_effect=AssertionError("must not run")):
        result = json.loads(terminal_tool(command=command))
    assert expected in result["error"]


def test_converted_command_never_gets_relaunch_hint_even_with_another_hint(tmp_path, monkeypatch):
    """A converted `&` command whose spawn also earns another hint (CI-poller shape) still carries
    the conversion note and never the silent-background "re-launch" advice (a second copy)."""
    from tools.terminal_tool import terminal_tool
    from tools.terminal_tool_background import _HOMEBREW_CI_POLLER_HINT, _SILENT_BACKGROUND_HINT

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    with patch("tools.terminal_tool._get_env_config", return_value=_env_config(tmp_path)), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        result = json.loads(terminal_tool(command="echo statusCheckRollup >/dev/null &"))

    assert result.get("error") is None, result
    assert "trailing & was converted to background=true" in result["auto_backgrounded"]
    assert _HOMEBREW_CI_POLLER_HINT in result["hint"]
    assert _SILENT_BACKGROUND_HINT not in result["hint"]


@pytest.mark.parametrize("command", [
    # A second background job hidden in an executable substitution would escape tracking.
    'pid="$(sleep 60 >/dev/null 2>&1 & echo $!)" && printf "%s" "$pid" &',
    'pid="`sleep 60 >/dev/null 2>&1 & echo $!`" && printf "%s" "$pid" &',
    # Same danger, different approval key: a pre-approval of the stripped key must not apply.
    "hermes gateway run && cp src .env &",
    # The escaped space belongs to the argument.
    "printf '<%s>' trailing\\ &",
])
def test_conversion_never_changes_what_runs_or_how_it_is_approved(command):
    """Either the old refusal stands, or the converted command is the same shell program with
    the same approval identity, minus only the background operator."""
    from tools.approval_detection import detect_dangerous_command
    from tools.terminal_tool_guards import trailing_amp_background_command

    stripped = trailing_amp_background_command(command)
    if stripped is None:
        return
    assert "$(" not in stripped and "`" not in stripped
    assert detect_dangerous_command(stripped) == detect_dangerous_command(command)
    if shutil.which("bash"):
        def out(script):
            return subprocess.run(["bash", "-c", script], check=False, capture_output=True, timeout=10).stdout
        assert out(stripped) == out(command + " wait")
