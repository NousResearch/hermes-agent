"""A literal ``kill`` of the current Hermes PID needs approval.

Name-based rules already cover ``pkill hermes`` and ``kill $(pgrep ...)``; no static pattern
can know the runtime PID, so the detector parses real command-position ``kill`` invocations
and every PID operand. Signal 0 and the list/help modes send nothing and stay approval-free,
and backends in another PID namespace skip the comparison.
"""

import os

import pytest

import tools.approval as approval_module
from tools.approval import detect_dangerous_command


def _assert_self_kill(command):
    dangerous, key, desc = detect_dangerous_command(command)

    assert dangerous is True
    assert key == desc
    assert "self-termination" in desc


def _assert_not_flagged(command):
    assert detect_dangerous_command(command) == (False, None, None)


class TestRuntimeSelfPidKillGuard:
    def test_kill_own_pid_requires_approval(self):
        own_pid = os.getpid()

        dangerous, key, desc = detect_dangerous_command(f"kill {own_pid}")

        assert dangerous is True
        assert key == desc == f"kill own Hermes process (self-termination, PID {own_pid})"

    @pytest.mark.parametrize(
        "template",
        [
            "kill -9 {pid}",
            "kill -TERM {pid}",
            "kill -s TERM {pid}",
            "/bin/kill --signal TERM {pid}",
            "env kill --signal TERM {pid}",
            "/usr/bin/env kill --signal TERM {pid}",
            "env FLAG=1 kill --signal TERM {pid}",
            "env -vu UNUSED /bin/kill {pid}",
            "/usr/bin/env -vC /tmp /bin/kill {pid}",
            "sudo --chdir /tmp /bin/kill {pid}",
            "sudo --chroot / /bin/kill {pid}",
            "sudo --role staff_r /bin/kill {pid}",
            "sudo --command-timeout 5 /bin/kill {pid}",
            "sudo --type staff_t /bin/kill {pid}",
            "sudo --other-user root /bin/kill {pid}",
            "sudo -iu root /bin/kill {pid}",
            "env --unset UNUSED -- kill --signal TERM {pid}",
            "env --argv0 custom -- kill --signal TERM {pid}",
            "env -a custom -- kill --signal TERM {pid}",
            "exec -- kill --signal TERM {pid}",
            "kill -- {pid}",
            "kill -n 9 {pid}",
            "kill -9 -- {pid}",
            "kill {pid} -0",
            "kill {pid} -s 0",
            "kill 999999999 {pid}",
            "kill 000{pid}",
            "kill +000{pid}",
            "/usr/bin/kill -TERM {pid}",
            "command -- kill {pid}",
            "command -p kill {pid}",
            "command -p -- kill {pid}",
            "builtin -- kill {pid}",
            "exec -- /bin/kill {pid}",
            "exec -c /bin/kill {pid}",
            "exec -c -- /bin/kill {pid}",
            "exec -l /bin/kill {pid}",
            "exec -a replacement /bin/kill {pid}",
            "exec -ca replacement /bin/kill {pid}",
            "setsid -f /bin/kill {pid}",
            "/usr/bin/setsid -f /bin/kill {pid}",
            "setsid -f -- /bin/kill {pid}",
            "setsid --wait /bin/kill {pid}",
            "time -p kill {pid}",
            "/usr/bin/time -p /bin/kill {pid}",
            "/usr/bin/time -f %e /bin/kill {pid}",
            "/usr/bin/time -p kill --signal TERM {pid}",
            "env time kill --signal TERM {pid}",
            "command time kill --signal TERM {pid}",
            "exec time kill --signal TERM {pid}",
            "sudo time kill --signal TERM {pid}",
            "nohup time kill --signal TERM {pid}",
            "setsid time kill --signal TERM {pid}",
            "nice -n 5 time kill --signal TERM {pid}",
            "/usr/bin/nohup /bin/kill {pid}",
            "env -u UNUSED /bin/kill {pid}",
            "env -u UNUSED -- /bin/kill {pid}",
            "env --unset UNUSED -- /bin/kill {pid}",
            "env -C /tmp /bin/kill {pid}",
            "env --chdir /tmp -- /bin/kill {pid}",
            "(kill {pid})",
            "echo ready; kill {pid} >/dev/null",
            "kill 2>/dev/null {pid}",
            "kill -9 > /dev/null {pid}",
            "bash -c 'kill {pid}'",
            "k'il'l {pid}",
        ],
    )
    def test_kill_own_pid_with_signal_forms_requires_approval(self, template):
        _assert_self_kill(template.format(pid=os.getpid()))

    @pytest.mark.parametrize(
        "template",
        [
            "kill --signal TERM {pid}",
            "kill --queue 7 {pid}",
            "kill --timeout 100 TERM {pid}",
            "builtin time kill --signal TERM {pid}",
            "time time kill --signal TERM {pid}",
        ],
    )
    def test_bash_kill_unsupported_long_options_are_not_flagged(self, template):
        _assert_not_flagged(template.format(pid=os.getpid()))

    @pytest.mark.parametrize(
        "template",
        [
            "env -S 'kill {pid}'",
            "env --split-string='kill -9 {pid}'",
            "/usr/bin/env -vS'kill {pid}'",
            "env -vu UNUSED -S 'kill {pid}'",
            "env -S 'kill --signal TERM {pid}'",
        ],
    )
    def test_env_split_string_kill_of_own_pid_requires_approval(self, template):
        _assert_self_kill(template.format(pid=os.getpid()))

    @pytest.mark.parametrize(
        "command",
        [
            "env -S 'printf ok'",
            "env --split-string='kill 999999999'",
            "/usr/bin/env -vS'printf ok'",
        ],
    )
    def test_env_split_string_without_own_pid_keeps_its_classification(self, command):
        _assert_not_flagged(command)

    @pytest.mark.parametrize(
        "template",
        [
            "kill -0 {pid}",
            "kill -00 {pid}",
            "kill -s 0 {pid}",
            "kill -s 00 {pid}",
            "kill -s +0 {pid}",
            "kill --signal 0 {pid}",
            "kill -s0 {pid}",
            "kill --signal=0 {pid}",
            "kill --signal=00 {pid}",
            "kill -n 0 {pid}",
            "kill -n 00 {pid}",
            "kill -n0 {pid}",
            "kill -0 -- {pid}",
            "kill -SIG0 {pid}",
            "/usr/bin/kill -s0 {pid}",
            "/usr/bin/kill --signal=0 {pid}",
        ],
    )
    def test_kill_own_pid_with_signal_zero_is_not_flagged(self, template):
        _assert_not_flagged(template.format(pid=os.getpid()))

    @pytest.mark.parametrize(
        "template",
        [
            'echo "kill {pid}"',
            "printf '%s' 'kill -9 {pid}'",
            "echo 'prefix kill {pid} suffix'",
            "echo ok # kill {pid}",
        ],
    )
    def test_quoted_kill_text_is_not_treated_as_a_command(self, template):
        _assert_not_flagged(template.format(pid=os.getpid()))

    @pytest.mark.parametrize(
        "template",
        [
            "kill -l {pid}",
            "kill -lTERM {pid}",
            "kill --list {pid}",
            "kill --list=TERM {pid}",
            "kill -L {pid}",
            "kill -L9 {pid}",
            "kill -h {pid}",
            "kill -V {pid}",
        ],
    )
    def test_kill_signal_listing_modes_are_not_flagged(self, template):
        _assert_not_flagged(template.format(pid=os.getpid()))

    def test_kill_unrelated_pid_is_not_flagged_by_self_pid_guard(self):
        _assert_not_flagged(f"kill {os.getpid() + 1_000_000}")

    def test_remote_pid_namespace_does_not_apply_local_self_pid_guard(self):
        dangerous, key, desc = detect_dangerous_command(
            f"kill {os.getpid()}", protect_local_pid=False
        )

        assert (dangerous, key, desc) == (False, None, None)

    def test_kill_own_pid_after_many_operands_requires_approval(self):
        operands = ["999999999"] * 300 + [str(os.getpid())]

        _assert_self_kill("kill " + " ".join(operands))

    @pytest.mark.parametrize("kind", ["operands", "assignments"])
    def test_oversized_segment_before_separator_is_still_inspected(self, kind):
        own_pid = os.getpid()
        if kind == "operands":
            command = "kill " + " ".join(["999999999"] * 4096 + [str(own_pid)])
        else:
            command = " ".join(["A=1"] * 4096 + ["kill", str(own_pid)])

        _assert_self_kill(command + "; :")

    @pytest.mark.parametrize(
        "prefix",
        [
            " ".join(["command"] * 13),
            " ".join(f"SELF_PID_GUARD_{index}=1" for index in range(13)),
        ],
    )
    def test_kill_own_pid_after_many_prefix_words_requires_approval(self, prefix):
        _assert_self_kill(f"{prefix} kill {os.getpid()}")


class TestSelfPidGuardBackendScope:
    """Only the local backend shares this process's PID namespace."""

    @pytest.fixture
    def observed(self, monkeypatch):
        calls = []

        def fake_detect(command, *, protect_local_pid=True):
            calls.append((command, protect_local_pid))
            return (False, None, None)

        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(approval_module, "detect_dangerous_command", fake_detect)
        return calls

    @pytest.mark.parametrize("env_type, protected", [("local", True), ("ssh", False)])
    def test_combined_guard_scopes_local_pid_protection(self, observed, env_type, protected):
        result = approval_module.check_all_command_guards("true", env_type)

        assert result["approved"] is True
        assert observed == [("true", protected)]

    @pytest.mark.parametrize("env_type, protected", [("local", True), ("ssh", False)])
    def test_pattern_guard_scopes_local_pid_protection(self, observed, env_type, protected):
        result = approval_module.check_dangerous_command("true", env_type)

        assert result["approved"] is True
        assert observed == [("true", protected)]
