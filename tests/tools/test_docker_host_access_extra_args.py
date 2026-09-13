"""Conservative detection of bind-mount arguments in ``docker_extra_args``.

``_docker_has_host_access`` feeds ``has_host_access`` into
``approval._should_skip_container_guards``, which skips dangerous-command and
``execute_code`` approval for a container judged isolated. A bind mount supplied
through ``docker_extra_args`` reaches host files exactly like one in
``docker_volumes``, so it must count the same way.

The detection is deliberately conservative: it flags *potential* binds and does
not model which flags consume a following value. A mount-looking token that is
really another option's value therefore enables guards too. These tests pin both
directions — the detections and the accepted false positive.
"""

import pytest

import tools.terminal_tool as terminal_tool
from tools.environments.docker import extra_args_may_bind_host_path


def _docker_config(**overrides):
    config = {
        "env_type": "docker",
        "cwd": "/workspace",
        "docker_volumes": [],
        "docker_extra_args": [],
        "docker_mount_cwd_to_workspace": False,
    }
    config.update(overrides)
    return config


# --- every spelling docker's flag parser accepts for a host bind -----------------------------

@pytest.mark.parametrize("extra_args", [
    pytest.param(["-v", "/tmp:/mnt"], id="short-separate"),
    pytest.param(["-v/tmp:/mnt"], id="short-attached"),
    pytest.param(["-v=/tmp:/mnt"], id="short-equals"),
    pytest.param(["-iv/tmp:/mnt"], id="cluster-attached"),
    pytest.param(["-iv", "/tmp:/mnt"], id="cluster-separate"),
    pytest.param(["-iv=/tmp:/mnt"], id="cluster-equals"),
    pytest.param(["--volume", "/etc:/mnt/etc:ro"], id="long-separate"),
    pytest.param(["--volume=/Users/me/.ssh:/mnt/ssh"], id="long-equals"),
    pytest.param(["-v", "~/.aws:/mnt/aws:ro"], id="tilde-source"),
    pytest.param(["-v", "./rel:/mnt"], id="dot-relative-volume"),
])
def test_volume_forms_are_detected(extra_args):
    assert extra_args_may_bind_host_path(extra_args) is True
    assert terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args)) is True


@pytest.mark.parametrize("extra_args", [
    pytest.param(["--mount", "type=bind,src=/Users/me,dst=/mnt"], id="src"),
    pytest.param(["--mount", "type=bind,source=/var/log,destination=/mnt"], id="source"),
    pytest.param(["--mount=type=bind,src=/tmp,dst=/mnt"], id="equals-joined"),
    pytest.param(["--mount", 'type=bind,"source=/tmp/a,b",target=/mnt'], id="csv-quoted-comma"),
    pytest.param(["--mount", "type=bind,src=.,dst=/mnt"], id="dot-source"),
    pytest.param(["--mount", "type=bind,src=..,dst=/mnt"], id="dotdot-source"),
    pytest.param(["--mount", "type=bind,src=cache,source=/tmp,dst=/mnt"], id="repeated-source-alias"),
    pytest.param(["--mount", "src=/tmp,type=bind,dst=/mnt"], id="type-after-source"),
    pytest.param(["--mount", "TYPE=BIND,SRC=/tmp,dst=/mnt"], id="case-insensitive"),
])
def test_bind_mounts_are_detected(extra_args):
    assert extra_args_may_bind_host_path(extra_args) is True
    assert terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args)) is True


# --- docker trims the whole value before CSV parsing; so must we -----------------------------

@pytest.mark.parametrize("extra_args", [
    pytest.param(["--mount", ' "type=bind",source=/tmp,target=/mnt'], id="leading-space-quoted-type"),
    pytest.param(["--mount", '\t"type=bind",source=/tmp,target=/mnt'], id="leading-tab-quoted-type"),
    pytest.param(["--mount", ' type=bind,src=/tmp,dst=/mnt'], id="leading-space-bare-type"),
    pytest.param(["--mount", 'type=bind,src=/tmp,dst=/mnt '], id="trailing-space"),
    pytest.param(["--mount=" + ' "type=bind",source=/tmp,target=/mnt'], id="equals-joined-leading-space"),
    pytest.param(["--mount=" + '\t"type=bind",src=/tmp,dst=/mnt'], id="equals-joined-leading-tab"),
    pytest.param(["--mount", '"type"="bind",src=/tmp,dst=/mnt'], id="quoted-key-and-value"),
])
def test_whitespace_and_quoting_do_not_hide_a_bind(extra_args):
    """Docker trims the value before CSV parsing; preserving outer whitespace kept a quoted
    field from starting at position 0, so ``"type=bind"`` parsed as the key ``"type``."""
    assert extra_args_may_bind_host_path(extra_args) is True
    assert terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args)) is True


# --- the deliberate false positive ----------------------------------------------------------

def test_mount_looking_value_of_another_flag_enables_guards():
    """Accepted imprecision: docker consumes this as a --label value, we still flag it.

    Detecting a potential bind is the goal; precisely determining isolation is not.
    Changing this to False would require modelling which flags consume a value.
    """
    extra_args = ["--label", "--volume=/tmp:/mnt"]
    assert extra_args_may_bind_host_path(extra_args) is True


# --- must stay isolated ---------------------------------------------------------------------

@pytest.mark.parametrize("extra_args", [
    pytest.param(["--network", "none"], id="network"),
    pytest.param(["--network=none", "--tmpfs", "/tmp", "--shm-size=1g"], id="unrelated-flags"),
    pytest.param(["--read-only", "--cap-drop", "ALL"], id="hardening-flags"),
    pytest.param(["-v", "hermes-cache:/workspace/.cache"], id="named-volume"),
    pytest.param(["-v", "my.cache:/data"], id="named-volume-with-dot"),
    pytest.param(["-v", "/workspace/node_modules"], id="anonymous-volume"),
    pytest.param(["--mount", "type=volume,src=hermes-data,dst=/data"], id="type-volume"),
    pytest.param(["--mount", "type=tmpfs,dst=/scratch"], id="type-tmpfs"),
    pytest.param(["--mount", "source=cache,target=/mnt"], id="omitted-type-defaults-volume"),
    pytest.param([], id="empty"),
    pytest.param([None, 42, {"v": "/host"}, "--network"], id="malformed-entries"),
])
def test_isolated_configurations_are_not_flagged(extra_args):
    assert extra_args_may_bind_host_path(extra_args) is False
    assert terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args)) is False


def test_short_cluster_without_v_is_ignored():
    """``-it`` carries no volume flag even though it is a shorthand cluster."""
    assert extra_args_may_bind_host_path(["-it", "/tmp:/mnt"]) is False


# --- windows drive sources ------------------------------------------------------------------

def test_windows_drive_source_is_host_access():
    assert extra_args_may_bind_host_path(["-v", r"C:\host:C:\container"]) is True


def test_windows_drive_anonymous_volume_is_not_host_access():
    assert extra_args_may_bind_host_path(["-v", "C:/container"]) is False


# --- existing behaviour must not regress ----------------------------------------------------

def test_docker_volumes_still_detected():
    assert terminal_tool._docker_has_host_access(
        _docker_config(docker_volumes=["/Users/me/code:/workspace"])) is True


def test_mount_cwd_to_workspace_still_detected():
    assert terminal_tool._docker_has_host_access(
        _docker_config(host_cwd="/Users/me/code", docker_mount_cwd_to_workspace=True)) is True


def test_clean_docker_config_has_no_host_access():
    assert terminal_tool._docker_has_host_access(_docker_config()) is False


def test_non_docker_backend_is_never_host_access():
    """Backends other than docker return before the extra_args branch."""
    assert terminal_tool._docker_has_host_access(
        _docker_config(env_type="singularity",
                       docker_extra_args=["-v", "/Users/me:/mnt/home"])) is False


# --- the guard behaviour this exists to protect ----------------------------------------------

@pytest.mark.parametrize("extra_args,expect_skip", [
    pytest.param(["--mount", "type=bind,src=/Users/me,dst=/mnt"], False, id="bind-enforces-approval"),
    pytest.param(["-iv/tmp:/mnt"], False, id="cluster-enforces-approval"),
    pytest.param(["--mount", ' "type=bind",source=/tmp,target=/mnt'], False, id="quoted-bind-enforces-approval"),
    pytest.param(["--network", "none"], True, id="isolated-keeps-fast-path"),
])
def test_guard_skip_follows_detection(extra_args, expect_skip):
    from tools.approval import _should_skip_container_guards
    has_access = terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args))
    assert _should_skip_container_guards("docker", has_host_access=has_access) is expect_skip


# --- propagation through both approval entry points ------------------------------------------

@pytest.mark.parametrize("extra_args,expect_guarded", [
    pytest.param(["--mount", "type=bind,src=/Users/me,dst=/mnt"], True, id="bind-guarded"),
    pytest.param(["-iv/tmp:/mnt"], True, id="cluster-guarded"),
    pytest.param(["--mount", ' "type=bind",source=/tmp,target=/mnt'], True, id="quoted-bind-guarded"),
    pytest.param(["--network", "none"], False, id="isolated-fast-path"),
])
def test_dangerous_command_guard_receives_host_access(extra_args, expect_guarded):
    """terminal_tool passes the predicate into check_dangerous_command (terminal_tool.py:908)."""
    from tools.approval import check_dangerous_command
    has_access = terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args))
    assert has_access is expect_guarded
    # With host access the container fast-path must not apply, so a hardline command is
    # blocked rather than waved through as "isolated enough".
    verdict = check_dangerous_command("rm -rf /", "docker", has_host_access=has_access)
    assert verdict.get("approved") is not True if expect_guarded else True


@pytest.mark.parametrize("extra_args,expect_guarded", [
    pytest.param(["--mount", "type=bind,src=/Users/me,dst=/mnt"], True, id="bind-guarded"),
    pytest.param(["--network", "none"], False, id="isolated-fast-path"),
])
def test_execute_code_guard_receives_host_access(extra_args, expect_guarded):
    """code_execution_tool passes the same predicate into check_execute_code_guard (line 710)."""
    from tools.approval import check_execute_code_guard
    has_access = terminal_tool._docker_has_host_access(_docker_config(docker_extra_args=extra_args))
    assert has_access is expect_guarded
    verdict = check_execute_code_guard("import os; os.system('rm -rf /')", "docker",
                                       has_host_access=has_access)
    assert isinstance(verdict, dict) and "approved" in verdict
