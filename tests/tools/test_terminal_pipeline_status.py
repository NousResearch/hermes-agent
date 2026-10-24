"""The bash wrapper reports each pipeline segment's own exit status."""

import subprocess

from tools.environments.base_session_env import _wrap_command_script

CWD_MARKER = "__HERMES_CWD_test__"
PIPE_MARKER = "__HERMES_PIPE_test__"


def _script(command: str) -> str:
    return _wrap_command_script(
        command, quoted_cwd="/tmp", quoted_snap="/nonexistent-snapshot",
        snap_tmp_template="/tmp/hermes-snap.XXXXXX", passthrough_names=(),
        snapshot_ready=False, cwd_marker=CWD_MARKER, pipe_marker=PIPE_MARKER,
    )


def _run(command: str) -> str:
    return subprocess.run(["bash", "-c", _script(command)], capture_output=True, text=True, check=False).stdout


def test_pipeline_status_captures_the_leading_command():
    output = _run("false | true")

    assert f"{PIPE_MARKER}1 0{PIPE_MARKER}" in output


def test_plain_command_reports_one_status():
    output = _run("false")

    assert f"{PIPE_MARKER}1{PIPE_MARKER}" in output


def test_wrapper_captures_pipestatus_before_anything_else():
    script = _script("echo hi")

    assert script.index('__hermes_ps=("${PIPESTATUS[@]}")') < script.index("umask 077")
    assert script.index("__hermes_ec=${__hermes_ps[${#__hermes_ps[@]}-1]}") < script.index(PIPE_MARKER)
