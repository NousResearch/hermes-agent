"""Tests for tools.admin_executor — Windows elevated command execution.

round-7 additions:
- framed named-pipe control protocol (nonce + client-PID binding, TOCTOU);
- system cmd.exe absolute-path resolution (search-hijack resistance);
- checked Job Object binding (job_bound) and truthful timeout termination;
- STARTUPINFOEXW handle allowlist and CRT fd ownership (helper side);
- merged stdout/stderr contract (matches normal terminal stderr=STDOUT).
"""

import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

import ctypes

from tools.elevated_protocol import (
    KIND_CANCELLED,
    KIND_CONNECTED,
    KIND_DONE,
    KIND_READY,
    ProtocolError,
    make_cancel,
    make_request,
    pack_message,
    unpack_frame,
)


class TestAdminExecutorPlatformChecks(unittest.TestCase):
    """Platform detection and capability checks."""

    @patch("tools.admin_executor.sys")
    def test_is_windows_true(self, mock_sys):
        mock_sys.platform = "win32"
        from tools.admin_executor import is_windows
        self.assertTrue(is_windows())

    @patch("tools.admin_executor.sys")
    def test_is_windows_false(self, mock_sys):
        mock_sys.platform = "linux"
        from tools.admin_executor import is_windows
        self.assertFalse(is_windows())

    @patch("tools.admin_executor.is_windows", return_value=False)
    def test_is_running_as_admin_non_windows(self, _):
        from tools.admin_executor import is_running_as_admin
        self.assertFalse(is_running_as_admin())

    @patch("tools.admin_executor.is_windows", return_value=False)
    def test_can_elevate_non_windows(self, _):
        from tools.admin_executor import can_elevate
        self.assertFalse(can_elevate())


class TestExecuteElevatedValidation(unittest.TestCase):
    """Input validation for execute_elevated."""

    @patch("tools.admin_executor.is_windows", return_value=False)
    def test_rejects_non_windows(self, _):
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("only supported on Windows", result["error"])

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=True)
    def test_rejects_already_admin(self, *_):
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("Already running as administrator", result["error"])


class TestCanElevate(unittest.TestCase):
    """can_elevate() logic."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=True)
    def test_already_admin_returns_false(self, *_):
        from tools.admin_executor import can_elevate
        self.assertFalse(can_elevate())

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_can_elevate_on_windows(self, mock_ctypes, *_):
        mock_ctypes.WinDLL.return_value = MagicMock(ShellExecuteExW=MagicMock())
        from tools.admin_executor import can_elevate
        self.assertTrue(can_elevate())


class _FakeElevationApi:
    """Scriptable fake for admin_executor._WindowsElevationApi.

    Simulates ShellExecuteExW, handle accounting and in-memory named pipes so
    every lifecycle path can be exercised without a UAC prompt: launch
    classification, client-PID verification (per-pipe, plus the hProcess
    identity and trusted-descendant walk), the framed control protocol,
    cancel/ack, timeout and handle close accounting.
    """

    ERROR_FILE_NOT_FOUND = 2
    ERROR_PATH_NOT_FOUND = 3
    ERROR_ACCESS_DENIED = 5
    ERROR_CANCELLED = 1223

    WAIT_OBJECT_0 = 0x00000000
    WAIT_ABANDONED = 0x00000080
    WAIT_TIMEOUT = 0x00000102
    WAIT_FAILED = 0xFFFFFFFF
    INFINITE = 0xFFFFFFFF
    STILL_ACTIVE = 259

    def __init__(self):
        self.launch_result = (0x1234, 0)  # (hProcess, last_error)
        self.terminate_result = True
        self.exit_code_result = (True, 0)
        self.launch_calls = []
        self.closed = []
        self.terminate_calls = []
        # Named-pipe simulation (server side).
        self.pipes = {}
        self._next_handle = 0x2000
        # GetNamedPipeClientProcessId results, per pipe kind.
        self.control_client_pid = 0x1111  # control pipe client
        self.output_client_pid = 0x1111   # output pipe client
        self.helper_pid = 0x1111  # pid the helper reports in connected
        # hProcess identity (GetProcessId) + descendant-walk result.
        self.hprocess_pid = 0x1111  # == helper by default (base python)
        self.descendant_result = True
        self.open_process_result = None  # None -> OpenProcess fails (dead)
        # Launch nonce: pinned to what _run_elevated's token_hex patch
        # produces ("f"*32 for a 16-byte nonce) so the fake's injected
        # CONNECTED/READY/DONE/CANCELLED messages pass the parent's nonce
        # validation.
        self.nonce = "f" * 32
        # Bounded connect simulation: per-pipe success flags.
        self.control_connect_ok = True
        self.output_connect_ok = True
        # Security-descriptor sentinel (non-None = DACL built).
        self.security_attributes = object()
        # Auto-protocol scripting.
        self.auto_connected = True
        self.ready_msg = {"kind": KIND_READY, "v": 1, "job_bound": True, "child_pid": 777}
        self.done_msg = {"kind": KIND_DONE, "v": 1, "rc": 0, "job_bound": True, "child_pid": 777}
        self.cancelled_msg = None

    # -- ShellExecuteExW / handle accounting ---------------------------------

    def shellexecute_ex(self, verb, file, parameters, directory):
        self.launch_calls.append((verb, file, parameters, directory))
        return self.launch_result

    def close_handle(self, handle):
        self.closed.append(handle)

    def terminate_process(self, handle, exit_code=1):
        self.terminate_calls.append((handle, exit_code))
        return self.terminate_result

    def get_exit_code(self, handle):
        return self.exit_code_result

    def wait_for_single_object(self, handle, timeout_ms):
        # h_process liveness poll: the fake helper stays alive unless told
        # otherwise (helper_pid is never "signalled" as exited).
        return self.WAIT_TIMEOUT

    def open_process(self, access, pid):
        return self.open_process_result

    # -- named-pipe simulation ------------------------------------------------

    def build_pipe_security_attributes(self):
        return self.security_attributes

    def create_named_pipe(self, name, *, message_mode=False, security_attributes=None):
        self._next_handle += 1
        h = self._next_handle
        self.pipes[h] = {
            "to_server": b"",
            "to_client": b"",
            "message_mode": message_mode,
        }
        if message_mode and self.auto_connected:
            self.pipes[h]["to_server"] = pack_message(
                {"kind": KIND_CONNECTED, "v": 1, "nonce": self.nonce,
                 "pid": self.helper_pid}
            )
        return h

    def connect_named_pipe_bounded(self, handle, timeout_s):
        pipe = self.pipes.get(handle, {})
        return self.control_connect_ok if pipe.get("message_mode") else self.output_connect_ok

    def get_named_pipe_client_process_id(self, handle):
        pipe = self.pipes.get(handle, {})
        if pipe.get("message_mode"):
            return self.control_client_pid
        return self.output_client_pid

    def get_process_id(self, handle):
        return self.hprocess_pid

    def is_trusted_descendant(self, ancestor_pid, child_pid, max_depth=16):
        return self.descendant_result

    def peek_pipe(self, handle):
        return len(self.pipes[handle]["to_server"])

    def read_pipe_bytes(self, handle, size):
        pipe = self.pipes[handle]
        buf = pipe["to_server"]
        if pipe["message_mode"]:
            # Message-mode pipes deliver one message per read; preserve the
            # boundary so two injected messages are consumed one at a time.
            if len(buf) < 4:
                return b""
            (length,) = struct.unpack("<I", buf[:4])
            total = 4 + length
            if total > (1024 * 1024) + 4:
                # Invalid (oversized) length: hand the whole buffer back so
                # unpack_frame rejects it instead of waiting forever.
                chunk = buf
                pipe["to_server"] = b""
                return chunk
            if len(buf) < total:
                return b""
            chunk = buf[:total]
            pipe["to_server"] = buf[total:]
            return chunk
        chunk = buf[:size]
        pipe["to_server"] = buf[size:]
        return chunk

    def write_pipe_bytes(self, handle, data):
        self.pipes[handle]["to_client"] += data
        # Scripted responses: request -> ready (+done), cancel -> cancelled.
        # The fake auto-fills the launch nonce so message validation passes
        # without every test having to hard-code it.
        try:
            msg = unpack_frame(data)[0]
        except Exception:
            msg = {}
        kind = msg.get("kind")

        def _with_nonce(base):
            d = dict(base or {})
            d.setdefault("nonce", self.nonce)
            return d

        if kind == "request" and self.ready_msg:
            self.pipes[handle]["to_server"] += pack_message(_with_nonce(self.ready_msg))
        if kind == "request" and self.done_msg:
            self.pipes[handle]["to_server"] += pack_message(_with_nonce(self.done_msg))
        if kind == "cancel" and self.cancelled_msg:
            self.pipes[handle]["to_server"] += pack_message(_with_nonce(self.cancelled_msg))

    def inject(self, handle, data):
        """Test helper: inject bytes as if written by the helper client."""
        self.pipes[handle]["to_server"] += data

    def client_writes(self, handle):
        """Test helper: bytes the server wrote toward the helper client."""
        data = self.pipes[handle]["to_client"]
        self.pipes[handle]["to_client"] = b""
        return data

    # -- test plumbing ---------------------------------------------------------

    def control_handle(self):
        for h, pipe in self.pipes.items():
            if pipe["message_mode"]:
                return h
        return None


def _run_elevated(command="echo hello", timeout=5, fake=None):
    """Run execute_elevated with the fake API and a prepared tmp dir."""
    from tools.admin_executor import execute_elevated

    fake = fake or _FakeElevationApi()
    # The channel generates nonce/token via secrets.token_hex; pin them so
    # the fake's injected messages carry the SAME nonce the parent validates.
    with patch(
        "tools.admin_executor.secrets.token_hex",
        side_effect=lambda n: "f" * (2 * n),
    ):
        return execute_elevated(command, timeout=timeout, _api=fake), fake


def _run_elevated_direct(command="echo hello", timeout=5, fake=None, **kwargs):
    """Like _run_elevated but passes extra kwargs through to execute_elevated
    (used by tests that need cwd=, etc. on the direct call)."""
    from tools.admin_executor import execute_elevated

    fake = fake or _FakeElevationApi()
    with patch(
        "tools.admin_executor.secrets.token_hex",
        side_effect=lambda n: "f" * (2 * n),
    ):
        return execute_elevated(command, timeout=timeout, _api=fake, **kwargs), fake


# ---------------------------------------------------------------------------
# Framed protocol (pure, cross-platform)
# ---------------------------------------------------------------------------


class TestElevatedProtocol(unittest.TestCase):
    """tools.elevated_protocol: framing + validation fail closed."""

    def test_pack_unpack_roundtrip(self):
        msg = make_request("nonce123", "echo hi", "C:/work", {"A": "b"})
        data = pack_message(msg)
        parsed, consumed = unpack_frame(data)
        self.assertEqual(parsed, msg)
        self.assertEqual(consumed, len(data))

    def test_truncated_frame_rejected(self):
        data = pack_message({"kind": KIND_READY, "v": 1})
        for cut in (1, 2, 3, len(data) - 1):
            with self.assertRaises(ProtocolError):
                unpack_frame(data[:cut])

    def test_oversized_frame_rejected(self):
        # length field claims more than the cap.
        data = struct.pack("<I", 999_999_999) + b"x" * 8
        with self.assertRaises(ProtocolError):
            unpack_frame(data)

    def test_invalid_json_rejected(self):
        data = struct.pack("<I", 4) + b"notjson"
        with self.assertRaises(ProtocolError):
            unpack_frame(data)

    def test_nonce_mismatch_rejected(self):
        msg = make_request("expected-nonce", "echo hi", None, {})
        with self.assertRaises(ProtocolError):
            from tools.elevated_protocol import validate_parent_message

            validate_parent_message(msg, "different-nonce")

    def test_replay_rejected_against_fresh_nonce(self):
        # A replayed request captured from a previous launch carries the old
        # nonce; validating it against the new nonce must fail.
        old = make_request("old-nonce", "echo compromised", None, {})
        with self.assertRaises(ProtocolError):
            from tools.elevated_protocol import validate_parent_message

            validate_parent_message(old, "new-nonce")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ProtocolError):
            from tools.elevated_protocol import validate_parent_message

            validate_parent_message({"kind": "evil", "v": 1, "nonce": "n"}, "n")

    def test_wrong_version_rejected(self):
        with self.assertRaises(ProtocolError):
            from tools.elevated_protocol import validate_parent_message

            validate_parent_message(
                {"kind": "request", "v": 99, "nonce": "n", "command": "x"}, "n"
            )

    def test_output_paths_rejected_from_request(self):
        """The helper must NEVER be told to write files: any output path field
        in the request is refused (fail closed)."""
        from tools.elevated_protocol import validate_parent_message

        for key in ("stdout_path", "stderr_path", "rc_path", "done_path"):
            msg = make_request("n", "echo hi", None, {})
            msg[key] = "C:/anywhere/evil.txt"
            with self.assertRaises(ProtocolError):
                validate_parent_message(msg, "n")

    def test_oversized_fields_rejected(self):
        from tools.elevated_protocol import (
            MAX_COMMAND_BYTES,
            MAX_ENV_TOTAL_BYTES,
            validate_parent_message,
        )

        huge_cmd = "x" * (MAX_COMMAND_BYTES + 1)
        with self.assertRaises(ProtocolError):
            validate_parent_message(make_request("n", huge_cmd, None, {}), "n")

        huge_env = {"K": "v" * (MAX_ENV_TOTAL_BYTES + 10)}
        with self.assertRaises(ProtocolError):
            validate_parent_message(make_request("n", "echo", None, huge_env), "n")

    def test_non_string_env_rejected(self):
        from tools.elevated_protocol import validate_parent_message

        msg = make_request("n", "echo", None, {"K": 123})
        with self.assertRaises(ProtocolError):
            validate_parent_message(msg, "n")


class TestHelperMessageValidation(unittest.TestCase):
    """round-8: every helper->parent message carries the launch nonce and is
    validated for required fields, types and legal state transitions."""

    @patch("tools.admin_executor.secrets.token_hex", return_value="n" * 32)
    def test_connected_requires_nonce(self, _):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message({"kind": KIND_CONNECTED, "v": 1, "pid": 1}, "n" * 32)

    def test_helper_message_wrong_nonce_rejected(self):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_CONNECTED, "v": 1, "nonce": "wrong", "pid": 1},
                "expected-nonce",
            )

    def test_ready_requires_job_bound_and_child_pid(self):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message({"kind": KIND_READY, "v": 1, "nonce": "n"}, "n")
        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_READY, "v": 1, "nonce": "n", "job_bound": True}, "n"
            )

    def test_done_requires_rc(self):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message({"kind": KIND_DONE, "v": 1, "nonce": "n"}, "n")

    def test_cancelled_requires_terminated(self):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_CANCELLED, "v": 1, "nonce": "n"}, "n"
            )

    def test_wrong_field_types_rejected(self):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_READY, "v": 1, "nonce": "n",
                 "job_bound": "yes", "child_pid": 1}, "n"
            )
        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_DONE, "v": 1, "nonce": "n", "rc": "0"}, "n"
            )
        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_CONNECTED, "v": 1, "nonce": "n", "pid": "1"}, "n"
            )

    def test_unexpected_extra_field_rejected(self):
        from tools.elevated_protocol import validate_helper_message

        with self.assertRaises(ProtocolError):
            validate_helper_message(
                {"kind": KIND_DONE, "v": 1, "nonce": "n", "rc": 0,
                 "sneaky": "x"}, "n"
            )

    def test_connected_passes_with_nonce(self):
        from tools.elevated_protocol import validate_helper_message

        msg = validate_helper_message(
            {"kind": KIND_CONNECTED, "v": 1, "nonce": "n", "pid": 123}, "n"
        )
        self.assertEqual(msg["pid"], 123)

    def test_state_machine_legal_sequence(self):
        from tools.elevated_protocol import _HelperStateMachine

        sm = _HelperStateMachine()
        sm.transition(KIND_CONNECTED)
        sm.transition(KIND_READY)
        sm.transition(KIND_DONE)

    def test_state_machine_first_must_be_connected(self):
        from tools.elevated_protocol import _HelperStateMachine, ProtocolError

        sm = _HelperStateMachine()
        with self.assertRaises(ProtocolError):
            sm.transition(KIND_READY)

    def test_state_machine_illegal_transition_rejected(self):
        from tools.elevated_protocol import _HelperStateMachine, ProtocolError

        sm = _HelperStateMachine()
        sm.transition(KIND_CONNECTED)
        with self.assertRaises(ProtocolError):
            sm.transition(KIND_CANCELLED)  # cancelled only after ready

    def test_state_machine_no_message_after_terminal(self):
        from tools.elevated_protocol import _HelperStateMachine, ProtocolError

        sm = _HelperStateMachine()
        sm.transition(KIND_CONNECTED)
        sm.transition(KIND_DONE)
        with self.assertRaises(ProtocolError):
            sm.transition(KIND_DONE)  # repeated done after terminal

    def test_state_machine_connected_to_done_shortcut(self):
        """CreateProcessW-failed shortcut: connected -> done is legal."""
        from tools.elevated_protocol import _HelperStateMachine

        sm = _HelperStateMachine()
        sm.transition(KIND_CONNECTED)
        sm.transition(KIND_DONE)


class TestIdentityBinding(unittest.TestCase):
    """round-8: verify_client binds hProcess + BOTH pipe clients to the
    helper-reported pid.  A rogue client, non-descendant pid, or a
    control/output pipe mismatch is refused (fail closed)."""

    def _make_channel(self, fake=None):
        from tools.admin_executor import _ElevatedPipeChannel

        fake = fake or _FakeElevationApi()
        ch = _ElevatedPipeChannel(fake)
        ch.create()
        self.addCleanup(ch.close)
        return ch, fake

    def test_hprocess_pid_equals_helper_accepted(self):
        ch, fake = self._make_channel()
        # default: hprocess_pid == helper_pid == control == output
        self.assertTrue(ch.verify_client(0x1234, fake.helper_pid))

    def test_output_pipe_client_mismatch_refused(self):
        ch, fake = self._make_channel()
        fake.output_client_pid = 0x9999  # rogue client squats output pipe
        self.assertFalse(ch.verify_client(0x1234, fake.helper_pid))

    def test_control_pipe_client_mismatch_refused(self):
        ch, fake = self._make_channel()
        fake.control_client_pid = 0x9999
        self.assertFalse(ch.verify_client(0x1234, fake.helper_pid))

    def test_missing_helper_pid_refused(self):
        ch, fake = self._make_channel()
        self.assertFalse(ch.verify_client(0x1234, None))

    def test_non_descendant_launcher_pid_refused(self):
        """A launcher shim whose child is NOT the helper pid must be refused
        (self-reported PID is not trusted on its own)."""
        ch, fake = self._make_channel()
        fake.hprocess_pid = 0xAAAA  # ShellExecuteExW handle = launcher shim
        fake.descendant_result = False  # helper is NOT its descendant
        self.assertFalse(ch.verify_client(0x1234, fake.helper_pid))

    def test_trusted_descendant_launcher_accepted(self):
        """A launcher shim whose REAL helper is a verified descendant is
        accepted (venv launcher path)."""
        ch, fake = self._make_channel()
        fake.hprocess_pid = 0xAAAA
        fake.descendant_result = True
        self.assertTrue(ch.verify_client(0x1234, fake.helper_pid))


class TestBoundedConnect(unittest.TestCase):
    """round-8: wait_connect must fail closed within a bound when either
    pipe never connects or the helper exits early — and must not leave a
    blocked thread behind."""

    def _channel(self, fake):
        from tools.admin_executor import _ElevatedPipeChannel

        ch = _ElevatedPipeChannel(fake)
        ch.create()
        self.addCleanup(ch.close)
        return ch

    @patch("tools.admin_executor.is_windows", return_value=True)
    def test_both_pipes_connect_ok(self, _):
        fake = _FakeElevationApi()
        ch = self._channel(fake)
        ch.wait_connect(timeout_s=5)  # fake connects both immediately
        self.assertIsNotNone(ch._h_control)
        self.assertIsNotNone(ch._h_output)

    @patch("tools.admin_executor.is_windows", return_value=True)
    def test_output_pipe_never_connects_times_out(self, _):
        fake = _FakeElevationApi()
        fake.output_connect_ok = False
        ch = self._channel(fake)
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            ch.wait_connect(timeout_s=1)
        self.assertLess(time.monotonic() - start, 5)
        # The timed-out channel can be closed without leaking handles.
        ch.close()
        self.assertIsNone(ch._h_control)
        self.assertIsNone(ch._h_output)

    @patch("tools.admin_executor.is_windows", return_value=True)
    def test_control_pipe_never_connects_times_out(self, _):
        fake = _FakeElevationApi()
        fake.control_connect_ok = False
        ch = self._channel(fake)
        with self.assertRaises(TimeoutError):
            ch.wait_connect(timeout_s=1)
        ch.close()

    @patch("tools.admin_executor.is_windows", return_value=True)
    def test_helper_exit_before_connect_fails_fast(self, _):
        fake = _FakeElevationApi()
        # Simulate the helper exiting: wait_for_single_object returns
        # WAIT_OBJECT_0 (signalled) immediately, and the output pipe never
        # connects (helper died before reaching it).
        fake.wait_for_single_object = lambda h, ms: fake.WAIT_OBJECT_0
        fake.output_connect_ok = False
        ch = self._channel(fake)
        start = time.monotonic()
        with self.assertRaises(TimeoutError) as ctx:
            ch.wait_connect(timeout_s=30, h_process=0x1234)
        self.assertLess(time.monotonic() - start, 10)
        self.assertIn("helper process exited", str(ctx.exception))
        # No blocked worker threads survive: the connect workers were joined
        # and the channel can be closed cleanly.
        ch.close()
        self.assertIsNone(ch._h_control)
        self.assertIsNone(ch._h_output)

    @patch("tools.admin_executor.is_windows", return_value=True)
    def test_no_blocked_threads_left_after_timeout(self, _):
        """After a connect timeout the worker threads must be joinable — the
        daemon flags alone are not enough; the handles are closed so the
        blocking ConnectNamedPipe aborts."""
        fake = _FakeElevationApi()
        fake.output_connect_ok = False
        ch = self._channel(fake)
        import threading as _th

        before = _th.active_count()
        with self.assertRaises(TimeoutError):
            ch.wait_connect(timeout_s=1)
        # Give any straggler a moment to unwind, then verify no NEW threads
        # from this call are still alive.
        time.sleep(0.2)
        after = _th.active_count()
        self.assertLessEqual(after, before + 1)


class TestEnvBlockValidation(unittest.TestCase):
    """round-8: the helper's environment block is sorted case-insensitively,
    rejects NUL/'=' in keys and NUL in values, and is NEVER None (an empty
    env still yields an explicit double-NUL Unicode block so the elevated
    child does not inherit the helper's ADMIN environment)."""

    def _block(self, env):
        from tools.elevated_helper import _build_env_block

        return _build_env_block(env)

    def _block_text(self, block):
        """Full UTF-16LE text of a Unicode buffer (block.value truncates at
        the first embedded NUL, so read the raw array via pointer)."""
        ctypes.cast(block, ctypes.c_void_p)
        buf = (ctypes.c_wchar * (ctypes.sizeof(block) // 2)).from_address(
            ctypes.addressof(block)
        )
        text = ctypes.string_at(buf, ctypes.sizeof(block)).decode(
            "utf-16-le", errors="replace"
        )
        return text.rstrip("\x00")

    def test_empty_env_yields_explicit_block_not_none(self):
        block = self._block({})
        self.assertIsNotNone(block)
        # The buffer is a Unicode buffer; sizeof gives the byte size which
        # includes the trailing NUL terminator pair.
        raw_size = ctypes.sizeof(block)
        self.assertGreaterEqual(raw_size, 4)
        # Its UTF-16LE bytes end with a double-NUL terminator.
        raw = block.value.encode("utf-16-le")
        self.assertTrue(raw.endswith(b"\x00\x00") or raw == b"")

    def test_env_sorted_case_insensitive(self):
        block = self._block({"Zebra": "1", "apple": "2", "Banana": "3"})
        # Recover the ordered entries from the full Unicode block.
        entries = [e for e in self._block_text(block).split("\x00") if e]
        keys = [e.split("=", 1)[0] for e in entries]
        self.assertEqual(keys, ["apple", "Banana", "Zebra"])  # case-insensitive

    def test_unicode_keys_and_values_roundtrip(self):
        block = self._block({"中文": "测试", "EMOJI😀": "值"})
        entries = [e for e in self._block_text(block).split("\x00") if e]
        self.assertEqual(len(entries), 2)
        self.assertIn("中文=测试", entries)
        self.assertIn("EMOJI😀=值", entries)

    def test_key_with_equals_rejected(self):
        from tools.elevated_protocol import ProtocolError

        with self.assertRaises(ProtocolError):
            self._block({"BAD=KEY": "v"})

    def test_key_with_nul_rejected(self):
        from tools.elevated_protocol import ProtocolError

        with self.assertRaises(ProtocolError):
            self._block({"BAD\x00KEY": "v"})

    def test_value_with_nul_rejected(self):
        from tools.elevated_protocol import ProtocolError

        with self.assertRaises(ProtocolError):
            self._block({"K": "v\x00x"})

    def test_non_string_rejected(self):
        from tools.elevated_protocol import ProtocolError

        with self.assertRaises(ProtocolError):
            self._block({"K": 123})


class TestSuspendedJobBinding(unittest.TestCase):
    """round-8: CREATE_SUSPENDED -> Job -> ResumeThread order.  The child is
    created suspended, assigned to the job with KILL_ON_JOB_CLOSE, and ONLY
    then resumed.  A binding failure terminates the still-suspended child and
    reports an explicit error — never runs an unbound admin command."""

    def test_job_binding_all_steps_success(self):
        import ctypes
        from tools.elevated_helper import _JobBinding

        job = _JobBinding()
        fake_k32 = {
            "CreateJobObjectW": lambda *a: 0x77,
            "SetInformationJobObject": lambda *a: True,
            "AssignProcessToJobObject": lambda *a: True,
        }
        ok = job.setup(type("K", (), fake_k32)(), 0x1234)
        self.assertTrue(ok)
        self.assertTrue(job.job_bound)
        self.assertIsNone(job.failure_reason)

    def test_job_creation_failure_reports_reason(self):
        from tools.elevated_helper import _JobBinding

        job = _JobBinding()
        fake_k32 = {
            "CreateJobObjectW": lambda *a: None,
            "SetInformationJobObject": lambda *a: True,
            "AssignProcessToJobObject": lambda *a: True,
        }
        ok = job.setup(type("K", (), fake_k32)(), 0x1234)
        self.assertFalse(ok)
        self.assertFalse(job.job_bound)
        self.assertIsNotNone(job.failure_reason)
        self.assertIn("CreateJobObjectW", job.failure_reason)

    def test_assign_failure_reports_reason(self):
        from tools.elevated_helper import _JobBinding

        job = _JobBinding()
        fake_k32 = {
            "CreateJobObjectW": lambda *a: 0x77,
            "SetInformationJobObject": lambda *a: True,
            "AssignProcessToJobObject": lambda *a: False,
        }
        ok = job.setup(type("K", (), fake_k32)(), 0x1234)
        self.assertFalse(ok)
        self.assertFalse(job.job_bound)
        self.assertIn("AssignProcessToJobObject", job.failure_reason)

    def test_limit_failure_reports_reason(self):
        from tools.elevated_helper import _JobBinding

        job = _JobBinding()
        fake_k32 = {
            "CreateJobObjectW": lambda *a: 0x77,
            "SetInformationJobObject": lambda *a: False,
            "AssignProcessToJobObject": lambda *a: True,
        }
        ok = job.setup(type("K", (), fake_k32)(), 0x1234)
        self.assertFalse(ok)
        self.assertFalse(job.job_bound)
        self.assertIn("SetInformationJobObject", job.failure_reason)

    def test_python_executable_prefers_base_executable(self):
        import sys as _sys
        from tools.admin_executor import _python_executable_for_elevation

        with patch.object(
            _sys, "_base_executable", "C:/Python/real/python.exe", create=True
        ):
            with patch("os.path.isfile", return_value=True):
                self.assertEqual(
                    _python_executable_for_elevation(), "C:/Python/real/python.exe"
                )


# ---------------------------------------------------------------------------
# ShellExecuteExW contract + pipe channel (fake API)
# ---------------------------------------------------------------------------


class TestElevationApiContract(unittest.TestCase):
    """ShellExecuteExW contract: last-error classification, client-PID
    binding, framed request, cancel/timeout semantics.  UAC cancel is a
    distinct outcome, never conflated with launch success or a system error."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_launch_cancelled_1223(self, *_):
        fake = _FakeElevationApi()
        fake.launch_result = (None, 1223)  # ShellExecuteExW FALSE + ERROR_CANCELLED
        result, fake = _run_elevated(fake=fake)
        self.assertEqual(result["exit_code"], -1)
        self.assertEqual(result["error_kind"], "cancelled")
        self.assertIn("cancelled", result["error"].lower())
        self.assertIn("1223", result["error"])
        # No request was ever sent (fail closed before the pipe protocol).
        ctl = fake.control_handle()
        self.assertEqual(fake.client_writes(ctl) if ctl else b"", b"")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_launch_access_denied_5(self, *_):
        fake = _FakeElevationApi()
        fake.launch_result = (None, 5)
        result, _ = _run_elevated(fake=fake)
        self.assertEqual(result["exit_code"], -1)
        self.assertEqual(result["error_kind"], "access_denied")
        self.assertIn("access denied", result["error"].lower())

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_launch_not_found_2_and_3(self, *_):
        for code in (2, 3):
            fake = _FakeElevationApi()
            fake.launch_result = (None, code)
            result, _ = _run_elevated(fake=fake)
            self.assertEqual(result["error_kind"], "not_found")
            self.assertIn("not found", result["error"].lower())

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_launch_other_system_error(self, *_):
        fake = _FakeElevationApi()
        fake.launch_result = (None, 31)
        result, _ = _run_elevated(fake=fake)
        self.assertEqual(result["exit_code"], -1)
        self.assertEqual(result["error_kind"], "other")
        self.assertIn("31", result["error"])

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_success_roundtrip(self, *_):
        """Happy path: connected -> request -> ready -> done -> rc=0."""
        fake = _FakeElevationApi()
        result, fake = _run_elevated()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["error"], None)
        # The helper launch carries ONLY pipe names + nonce — never the
        # command, cwd or env (no request TOCTOU surface on the command line).
        verb, file, parameters, directory = fake.launch_calls[0]
        self.assertEqual(verb, "runas")
        self.assertTrue(file.endswith("python.exe") or file.lower().endswith("python.exe"))
        self.assertIn("elevated_helper.py", parameters)
        self.assertIn(r"\\.\pipe\hermes-elevated-ctl-", parameters)
        self.assertIn(r"\\.\pipe\hermes-elevated-out-", parameters)
        self.assertNotIn("echo hello", parameters)
        # The request DID travel on the control pipe with the sanitized env.
        ctl = fake.control_handle()
        sent = fake.client_writes(ctl)
        self.assertTrue(sent)
        request = unpack_frame(sent)[0]
        self.assertEqual(request["kind"], "request")
        self.assertEqual(request["command"], "echo hello")
        # The nonce used in the request must be the one passed on the command
        # line (binding to THIS launch).
        self.assertIn(request["nonce"], parameters)
        # Sanitized env: provider secrets are absent, HERMES_HOME present.
        self.assertNotIn("ANTHROPIC_API_KEY", request["env"])
        self.assertIn("HERMES_HOME", request["env"])
        # The elevated process handle was closed after the run.
        self.assertIn(0x1234, fake.closed)

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_pid_mismatch_refused(self, *_):
        """A client whose PID does not match the helper-reported pid is
        refused BEFORE the request is sent (fail closed)."""
        fake = _FakeElevationApi()
        fake.control_client_pid = 0x9999  # control pipe client (OS check)
        result, fake = _run_elevated(fake=fake)
        self.assertEqual(result["error_kind"], "pid_mismatch")
        ctl = fake.control_handle()
        self.assertEqual(fake.client_writes(ctl), b"")  # nothing sent
        self.assertIn(0x1234, fake.closed)

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_protocol_error_fail_closed(self, *_):
        fake = _FakeElevationApi()
        fake.auto_connected = False
        # Inject a garbage frame as the first helper message.
        ctl = None
        original = fake.create_named_pipe

        def create(name, *, message_mode=False, security_attributes=None):
            nonlocal ctl
            h = original(name, message_mode=message_mode,
                         security_attributes=security_attributes)
            if message_mode:
                ctl = h
                fake.inject(h, b"\xff\xff\xff\xffjunkjunk")
            return h

        fake.create_named_pipe = create
        result, fake = _run_elevated(fake=fake)
        self.assertEqual(result["error_kind"], "protocol")
        self.assertIn("protocol error", result["error"].lower())

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_timeout_cancel_terminated(self, *_):
        """Timeout with a cancelled ack (terminated=true) reports terminated."""
        fake = _FakeElevationApi()
        fake.done_msg = None  # command never finishes
        fake.cancelled_msg = {"kind": KIND_CANCELLED, "v": 1, "terminated": True,
                              "job_bound": True, "rc": 1}
        result, fake = _run_elevated(timeout=1, fake=fake)
        self.assertEqual(result["error_kind"], "timeout")
        self.assertIn("terminated", result["error"].lower())
        # The helper was told to cancel over the control pipe.
        ctl = fake.control_handle()
        sent = fake.client_writes(ctl)
        self.assertIn("cancel", sent.decode("utf-8", "replace"))
        self.assertIn(0x1234, fake.closed)

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_timeout_cancel_unterminated_may_still_run(self, *_):
        """Timeout whose cancel ack says terminated=false must report 'may
        still be running' (no false termination claim) and must NOT leave a
        raw output file behind."""
        fake = _FakeElevationApi()
        fake.done_msg = None
        fake.cancelled_msg = {"kind": KIND_CANCELLED, "v": 1, "terminated": False,
                              "job_bound": False, "rc": -1}
        created_files = []
        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            created_files.append(path)
            return fd, path

        with patch("tools.admin_executor.tempfile.mkstemp", side_effect=tracking_mkstemp):
            result, fake = _run_elevated(timeout=1, fake=fake)
        self.assertEqual(result["error_kind"], "timeout")
        self.assertIn("may still be running", result["error"].lower())
        self.assertNotIn("_keep_tmp_dir", result)
        # The parent-owned raw output file must not survive.
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output must be deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_timeout_no_ack_with_job_bound_child_dead(self, *_):
        """Timeout, no cancel ack: helper terminated, job was bound, child PID
        verified gone -> terminated may be claimed (with evidence)."""
        fake = _FakeElevationApi()
        fake.done_msg = None
        fake.cancelled_msg = None  # helper unresponsive
        fake.terminate_result = True
        fake.exit_code_result = (True, 1)  # helper no longer STILL_ACTIVE
        fake.open_process_result = None  # child PID cannot be opened -> dead
        with patch("tools.admin_executor._CANCEL_ACK_TIMEOUT_S", 1):
            result, fake = _run_elevated(timeout=1, fake=fake)
        self.assertEqual(result["error_kind"], "timeout")
        self.assertIn("terminated", result["error"].lower())
        self.assertTrue(fake.terminate_calls)
        self.assertIn(0x1234, fake.closed)

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_timeout_no_ack_job_not_bound_may_still_run(self, *_):
        """Timeout, no ack, helper dead BUT job was never bound -> must NOT
        claim termination; report 'may still be running'."""
        fake = _FakeElevationApi()
        fake.done_msg = None
        fake.cancelled_msg = None
        fake.ready_msg = {"kind": KIND_READY, "v": 1, "job_bound": False, "child_pid": 777}
        fake.terminate_result = True
        fake.exit_code_result = (True, 1)
        created_files = []
        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            created_files.append(path)
            return fd, path

        with patch("tools.admin_executor._CANCEL_ACK_TIMEOUT_S", 1):
            with patch("tools.admin_executor.tempfile.mkstemp", side_effect=tracking_mkstemp):
                result, fake = _run_elevated(timeout=1, fake=fake)
        self.assertEqual(result["error_kind"], "timeout")
        self.assertIn("may still be running", result["error"].lower())
        self.assertNotIn("_keep_tmp_dir", result)
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output must be deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_handle_closed_on_every_path(self, *_):
        # success
        fake = _FakeElevationApi()
        _run_elevated(fake=fake)
        self.assertIn(0x1234, fake.closed)
        # cancel path: no hProcess exists (pipes still get closed).
        fake2 = _FakeElevationApi()
        fake2.launch_result = (None, 1223)
        _run_elevated(fake=fake2)
        self.assertNotIn(0x1234, fake2.closed)
        # timeout path
        fake3 = _FakeElevationApi()
        fake3.done_msg = None
        fake3.cancelled_msg = {"kind": KIND_CANCELLED, "v": 1, "terminated": True}
        _run_elevated(timeout=1, fake=fake3)
        self.assertIn(0x1234, fake3.closed)
        # pid mismatch path
        fake4 = _FakeElevationApi()
        fake4.control_client_pid = 999
        _run_elevated(fake=fake4)
        self.assertIn(0x1234, fake4.closed)


class TestOutputFileLifecycle(unittest.TestCase):
    """Raw output file lifecycle: the parent-owned ``hermes_elevated_out_*``
    file is deleted on EVERY path — small-output success (read-then-delete),
    overflow (moved to the raw staging dir), launch/protocol/pid-mismatch
    error, cancel, and timeout.  Nothing relies on ``channel.close`` making
    the file vanish; the delete is part of the result collection itself.
    """

    def _track_files(self):
        created_files = []
        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            created_files.append(path)
            return fd, path

        return (
            patch("tools.admin_executor.tempfile.mkstemp", side_effect=tracking_mkstemp),
            created_files,
        )

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_output_file_deleted_on_success_small_output(self, *_):
        fake = _FakeElevationApi()
        tracker, created_files = self._track_files()
        with tracker:
            result, _ = _run_elevated(fake=fake)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(created_files, "output file must have been created")
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output not deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_output_file_staged_on_overflow(self, *_):
        """Overflowing output is MOVED to a throwaway raw staging dir (never
        left in the system temp as hermes_elevated_out_*)."""
        fake = _FakeElevationApi()
        tracker, created_files = self._track_files()
        raw_staged = []
        original_stage = None
        from tools import admin_executor as _ae

        def tracking_stage(output_file):
            p = original_stage(output_file)
            raw_staged.append(p)
            return p

        original_stage = _ae._stage_raw_output
        with tracker:
            with patch(
                "tools.admin_executor._stage_raw_output", side_effect=tracking_stage
            ):
                result, _ = _run_elevated(command="echo hello", fake=fake)
        # Success path with the default small output: the overflow branch is
        # NOT taken, so nothing was staged and the raw file was deleted.
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(raw_staged, [])
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output not deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_output_file_deleted_on_launch_failure(self, *_):
        fake = _FakeElevationApi()
        fake.launch_result = (None, 5)
        tracker, created_files = self._track_files()
        with tracker:
            result, _ = _run_elevated(fake=fake)
        self.assertEqual(result["exit_code"], -1)
        self.assertEqual(result["error_kind"], "access_denied")
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output not deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_output_file_deleted_on_cancelled_uac(self, *_):
        fake = _FakeElevationApi()
        fake.launch_result = (None, 1223)
        tracker, created_files = self._track_files()
        with tracker:
            result, _ = _run_elevated(fake=fake)
        self.assertIn("cancelled", result["error"].lower())
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output not deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_output_file_deleted_on_protocol_error(self, *_):
        fake = _FakeElevationApi()
        fake.auto_connected = False
        ctl = None
        original = fake.create_named_pipe

        def create(name, *, message_mode=False, security_attributes=None):
            nonlocal ctl
            h = original(name, message_mode=message_mode,
                         security_attributes=security_attributes)
            if message_mode:
                ctl = h
                fake.inject(h, b"\xff\xff\xff\xffjunkjunk")
            return h

        fake.create_named_pipe = create
        tracker, created_files = self._track_files()
        with tracker:
            result, _ = _run_elevated(fake=fake)
        self.assertEqual(result["error_kind"], "protocol")
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output not deleted: {p}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_output_file_deleted_on_pid_mismatch(self, *_):
        fake = _FakeElevationApi()
        fake.control_client_pid = 0x9999
        tracker, created_files = self._track_files()
        with tracker:
            result, _ = _run_elevated(fake=fake)
        self.assertEqual(result["error_kind"], "pid_mismatch")
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output not deleted: {p}")


class TestCwdValidation(unittest.TestCase):
    """Defense-in-depth cwd validation before the helper is launched."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_rejects_cwd_with_shell_metacharacters(self, *_):
        fake = _FakeElevationApi()
        from tools.admin_executor import execute_elevated
        for bad in ("C:/tmp/a&b", "C:/tmp/a|b", 'C:/tmp/a"b', "C:/tmp/a<b"):
            result = execute_elevated("echo hello", cwd=bad, _api=fake)
            self.assertEqual(result["exit_code"], -1)
            self.assertIn("Blocked", result["error"])
            self.assertEqual(fake.launch_calls, [])

    def test_validate_cwd_for_script_allowlist(self):
        from tools.admin_executor import _validate_cwd_for_script
        self.assertIsNone(_validate_cwd_for_script("C:/Users/test/My Folder"))
        self.assertIsNone(_validate_cwd_for_script(""))
        self.assertIsNone(_validate_cwd_for_script("C:\\Users\\test"))
        self.assertIsNotNone(_validate_cwd_for_script("C:/tmp/a&b"))
        self.assertIsNotNone(_validate_cwd_for_script("C:/tmp/a|b"))
        self.assertIsNotNone(_validate_cwd_for_script('C:/tmp/a"b'))
        self.assertIsNotNone(_validate_cwd_for_script("C:/tmp/a;rm -rf /"))
        self.assertIsNotNone(_validate_cwd_for_script("C:/tmp/$(whoami)"))

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_cwd_none_safe_getcwd_allowed(self, *_):
        fake = _FakeElevationApi()
        with patch("tools.admin_executor.os.getcwd", return_value="C:/Users/test/AppData"):
            result, fake = _run_elevated_direct("echo hello", timeout=1, fake=fake)
        self.assertEqual(len(fake.launch_calls), 1)
        self.assertIsNone(result.get("error"))  # not blocked, success path

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_cwd_none_dangerous_getcwd_blocked(self, *_):
        fake = _FakeElevationApi()
        with patch("tools.admin_executor.os.getcwd", return_value="C:/tmp/a&b"):
            with patch("tools.admin_executor.tempfile.mkstemp") as mock_mkstemp:
                from tools.admin_executor import execute_elevated
                result = execute_elevated("echo hello", _api=fake)
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("Blocked", result["error"])
        mock_mkstemp.assert_not_called()
        self.assertEqual(fake.launch_calls, [])


class TestElevatedEnvironmentSanitization(unittest.TestCase):
    """The elevated child receives the SAME scrubbed environment as a normal
    local terminal subprocess — never the raw Hermes service-process env."""

    @patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-ant-sentinel-123",
            "GH_TOKEN": "ghp_sentinel-token",
            "OPENAI_API_KEY": "sk-sentinel-openai",
            "SENTINEL_NORMAL_VAR": "keep_me",
        },
        clear=False,
    )
    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_provider_and_gateway_secrets_scrubbed_plain_vars_kept(self, *_):
        fake = _FakeElevationApi()
        _run_elevated(fake=fake)
        ctl = fake.control_handle()
        request = unpack_frame(fake.client_writes(ctl))[0]
        env = request["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("GH_TOKEN", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertEqual(env.get("SENTINEL_NORMAL_VAR"), "keep_me")
        self.assertIn("HERMES_HOME", env)
        self.assertEqual(env.get("PYTHONUTF8"), "1")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_commandline_never_contains_command_or_env(self, *_):
        """ShellExecuteExW parameters must not carry the command, cwd, env or
        any output path — only helper + pipe names + nonce."""
        fake = _FakeElevationApi()
        _run_elevated(command="echo TOP SECRET COMMAND & echo MORE", fake=fake)
        parameters = fake.launch_calls[0][2]
        self.assertNotIn("TOP SECRET", parameters)
        self.assertNotIn("output.txt", parameters)
        self.assertNotIn("rc.txt", parameters)
        self.assertNotIn("@echo", parameters)


class TestTimeoutSemantics(unittest.TestCase):
    """Timeout must report a truthful, explainable status — never silent success."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    def test_timeout_reports_may_still_be_running(self, *_):
        fake = _FakeElevationApi()
        fake.done_msg = None
        fake.cancelled_msg = {"kind": KIND_CANCELLED, "v": 1, "terminated": False}
        created_files = []
        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            created_files.append(path)
            return fd, path

        with patch("tools.admin_executor.tempfile.mkstemp", side_effect=tracking_mkstemp):
            result, fake = _run_elevated_direct("sleep 100", timeout=1, fake=fake)

        self.assertEqual(result["exit_code"], -1)
        self.assertIn("timed out", result["error"].lower())
        self.assertIn("may still be running", result["error"].lower())
        for p in created_files:
            self.assertFalse(os.path.exists(p), f"Raw output must be deleted: {p}")


# ---------------------------------------------------------------------------
# Real helper execution (no UAC): exercises the actual CreateProcessW Unicode
# contract, the named-pipe protocol, the absolute system cmd path (search
# hijack) and Job Object termination on Windows.  These run whenever the
# suite runs on Windows; they do NOT require an interactive UAC prompt.  UAC
# approve/cancel itself is covered by the opt-in integration file.
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if sys.platform != "win32":
        return True
    import ctypes

    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    ctypes.windll.kernel32.CloseHandle(h)
    return True


@unittest.skipUnless(sys.platform == "win32", "Windows-only helper execution")
class TestElevatedHelperReal(unittest.TestCase):
    """Real Windows execution through tools/elevated_helper.py over the real
    named-pipe protocol (the exact chain the elevated process runs)."""

    def _run_helper(self, command, cwd=None, env=None, cancel_after_ready=False):
        """Drive the real helper as a REAL subprocess through real named
        pipes — the exact chain the elevated process runs after UAC approval
        (ShellExecuteExW launches python.exe tools/elevated_helper.py)."""
        import subprocess as _sp

        from tools.admin_executor import _ElevatedPipeChannel, _WindowsElevationApi

        api = _WindowsElevationApi()
        ch = _ElevatedPipeChannel(api)
        ch.create()
        helper_path = os.path.abspath("tools/elevated_helper.py")
        proc = _sp.Popen(
            [sys.executable, helper_path, ch.control_name, ch.output_name,
             ch.nonce],
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
        )
        try:
            ch.wait_connect()
            first = ch.read_message(timeout_ms=30_000)
            self.assertEqual(first["kind"], KIND_CONNECTED)
            # The helper-reported pid (the process that opened the pipe) must
            # match GetNamedPipeClientProcessId on the pipe itself.
            client_pid = api.get_named_pipe_client_process_id(ch._h_control)
            self.assertEqual(client_pid, first.get("pid"),
                             "pipe client PID must match the helper-reported pid")

            base_env = {
                "PYTHONUTF8": "1",
                "PATH": os.environ.get("PATH", ""),
            }
            if env:
                base_env.update(env)

            ch.start_output_drain()
            ch.send(make_request(ch.nonce, command, cwd, base_env))

            progress = ch.read_message(timeout_ms=30_000)
            self.assertEqual(progress["kind"], KIND_READY,
                             msg=f"helper progress={progress}")
            child_pid = progress.get("child_pid")

            if cancel_after_ready:
                ch.send(make_cancel(ch.nonce))
                ack = ch.read_message(timeout_ms=30_000)
                ch.join_output()
                output = open(ch.output_path, encoding="utf-8",
                              errors="replace").read()
                return {
                    "ack_kind": ack.get("kind"),
                    "terminated": ack.get("terminated"),
                    "job_bound": ack.get("job_bound"),
                    "output": output,
                    "child_pid": child_pid,
                }

            done = ch.read_message(timeout_ms=90_000)
            ch.join_output()
            output = open(ch.output_path, encoding="utf-8",
                          errors="replace").read()
            return {
                "done_kind": done.get("kind"),
                "rc": done.get("rc"),
                "job_bound": done.get("job_bound"),
                "child_pid": child_pid,
                "output": output,
            }
        finally:
            ch.close()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    def test_system_cmd_executes_and_merges_stdout_stderr(self):
        result = self._run_helper(
            "python -c \"import sys; print('OUT1'); "
            "sys.stderr.write('ERR1')\"",
            env={"PYTHONUTF8": "1"},
        )
        self.assertEqual(result["done_kind"], "done")
        self.assertEqual(result["rc"], 0)
        # stderr is merged into the same output stream (normal terminal
        # contract: stderr=STDOUT).
        self.assertIn("OUT1", result["output"])
        self.assertIn("ERR1", result["output"])

    def test_search_hijack_resisted_fake_cmd_in_cwd(self):
        """A fake cmd.exe in the working directory must NEVER be executed:
        the helper resolves the shell via GetSystemDirectoryW and passes the
        absolute path as lpApplicationName."""
        base = tempfile.mkdtemp(prefix="hermes_hijack_")
        try:
            fake_cmd = os.path.join(base, "cmd.exe")
            with open(fake_cmd, "wb") as f:
                f.write(b"NOT A REAL PE FILE" * 8)
            result = self._run_helper("echo hijack-guard-ok", cwd=base,
                                      env={"PYTHONUTF8": "1"})
            self.assertEqual(result["rc"], 0)
            self.assertIn("hijack-guard-ok", result["output"])
            # The fake file was never executed/modified.
            with open(fake_cmd, "rb") as f:
                self.assertEqual(f.read(), b"NOT A REAL PE FILE" * 8)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_unicode_program_output_and_unicode_cwd(self):
        base = tempfile.mkdtemp(prefix="hermes_uni_")
        try:
            unicode_dir = os.path.join(base, "中文 😀 ディレクトリ (x)")
            os.makedirs(unicode_dir)
            result = self._run_helper(
                'python -c "import os; print(os.getcwd()); '
                "print('中文😀emoji 日本テスト')\"",
                cwd=unicode_dir,
                env={"PYTHONUTF8": "1"},
            )
            self.assertEqual(result["rc"], 0)
            self.assertIn("中文 😀 ディレクトリ (x)", result["output"])
            self.assertIn("中文😀emoji 日本テスト", result["output"])
            self.assertNotIn("\ufffd", result["output"])
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_shell_special_characters(self):
        result = self._run_helper(
            'echo one & echo "two & three (four)" & echo five!',
            env={"PYTHONUTF8": "1"},
        )
        self.assertEqual(result["rc"], 0)
        self.assertIn("one", result["output"])
        self.assertIn("two & three (four)", result["output"])
        self.assertIn("five!", result["output"])

    def test_job_bound_reported(self):
        result = self._run_helper("echo hi", env={"PYTHONUTF8": "1"})
        self.assertTrue(result["job_bound"],
                        "job create/limit/assign must all succeed on Windows")

    def test_env_isolation_child_sees_only_passed_env(self):
        result = self._run_helper(
            "python -c \"import os; print('SENTINEL_DROP_ME' in os.environ)\"",
            env={"PYTHONUTF8": "1", "SENTINEL_KEEP_ME": "visible"},
        )
        self.assertEqual(result["rc"], 0)
        self.assertEqual(result["output"].strip(), "False")

    def test_cancel_terminates_process_tree(self):
        """Cancel after ready must terminate the whole tree (parent, child and
        grandchild) via TerminateJobObject and report terminated=true."""
        result = self._run_helper(
            "python -c \"import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(60)']); "
            "print('GRANDCHILD='+str(p.pid), flush=True); time.sleep(60)\"",
            env={"PYTHONUTF8": "1"},
            cancel_after_ready=True,
        )
        self.assertEqual(result["ack_kind"], "cancelled")
        self.assertTrue(result["terminated"],
                        "job-bound tree must be reported terminated")
        self.assertTrue(result["job_bound"])
        # The direct child (cmd.exe) must be gone.
        child = result.get("child_pid")
        if child:
            deadline = time.monotonic() + 10
            while _pid_alive(child) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(_pid_alive(child),
                             "cmd child must be terminated")
        # The grandchild pid announced in output must be gone too.
        import re

        m = re.search(r"GRANDCHILD=(\d+)", result["output"])
        if m:
            grandchild = int(m.group(1))
            deadline = time.monotonic() + 10
            while _pid_alive(grandchild) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(_pid_alive(grandchild),
                             "grandchild must be terminated by the job")


class TestBoundedOutputRead(unittest.TestCase):
    """Elevated output must never be read unbounded into memory."""

    def test_small_file_reads_fully(self, tmp_path=None):
        from tools.admin_executor import _read_output_bounded
        import tempfile as _tf
        import os as _os

        d = _tf.mkdtemp(prefix="hermes_test_bounded_")
        try:
            p = _os.path.join(d, "out.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("hello small output")
            out, total = _read_output_bounded(p, 1000)
            self.assertEqual(out, "hello small output")
            self.assertEqual(total, len("hello small output"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_large_file_returns_bounded_window_and_true_total(self):
        from tools.admin_executor import _read_output_bounded
        import tempfile as _tf
        import os as _os

        d = _tf.mkdtemp(prefix="hermes_test_bounded_")
        try:
            p = _os.path.join(d, "out.txt")
            content = ("A" * 200_000) + "TAIL999"
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            out, total = _read_output_bounded(p, 1000)
            self.assertEqual(total, len(content))
            self.assertLessEqual(len(out), 1000)
            self.assertIn("TAIL999", out)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_file_returns_empty(self):
        from tools.admin_executor import _read_output_bounded
        out, total = _read_output_bounded("C:/nonexistent/definitely-missing.log", 1000)
        self.assertEqual(out, "")
        self.assertEqual(total, 0)


class TestCharCountSemantics(unittest.TestCase):
    """output_total_chars must count Unicode characters, not bytes."""

    def test_multibyte_large_file_counts_chars_not_bytes(self):
        from tools.admin_executor import _read_output_bounded
        import tempfile as _tf
        import os as _os

        d = _tf.mkdtemp(prefix="hermes_test_charcount_")
        try:
            p = _os.path.join(d, "out.txt")
            content = ("中文" * 100_000) + "TAIL_ASCII_TAIL"
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            out, total = _read_output_bounded(p, 1000)
            self.assertEqual(total, len(content))
            self.assertGreater(_os.path.getsize(p), total)
            self.assertLessEqual(len(out), 1000)
            self.assertIn("TAIL_ASCII_TAIL", out)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_bytes_over_cap_but_chars_under_cap_not_truncated(self):
        from tools.admin_executor import _read_output_bounded
        import tempfile as _tf
        import os as _os

        d = _tf.mkdtemp(prefix="hermes_test_charundercap_")
        try:
            p = _os.path.join(d, "out.txt")
            content = ("中文" * 15_000) + "ENDMARK"
            with open(p, "wb") as f:
                f.write(content.encode("utf-8"))
            self.assertGreater(_os.path.getsize(p), 50_000)
            out, total = _read_output_bounded(p, 50_000)
            self.assertEqual(total, len(content))
            self.assertEqual(out, content)
            self.assertNotIn("\ufffd", out)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_multibyte_crosses_read_chunks(self):
        from tools.admin_executor import _read_output_bounded
        import tempfile as _tf
        import os as _os

        d = _tf.mkdtemp(prefix="hermes_test_crosschunk_")
        try:
            p = _os.path.join(d, "out.txt")
            unit = "中文😀A测试🚀"
            content = unit * 5_000
            with open(p, "wb") as f:
                f.write(content.encode("utf-8"))
            out, total = _read_output_bounded(p, len(content) - 100)
            self.assertEqual(total, len(content))
            self.assertNotIn("\ufffd", out)
            self.assertEqual(out[-600:], content[-600:])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestRawStagingCleanup(unittest.TestCase):
    """Stale ``hermes_elevated_raw_*`` staging dirs are reaped on the next
    execution (hard-crash leftovers are never claimed to be cleaned in real
    time)."""

    def _make_raw_dir(self, root, name, old=False):
        import time as _time

        p = os.path.join(root, name)
        os.mkdir(p)
        if old:
            t = _time.time() - 100_000
            os.utime(p, (t, t))
        return p

    def test_stale_raw_dirs_reaped_fresh_and_foreign_kept(self):
        import tempfile as _tf

        root = _tf.mkdtemp(prefix="hermes_test_rawttl_")
        try:
            with patch(
                "tools.admin_executor.tempfile.gettempdir", return_value=root,
            ):
                stale = self._make_raw_dir(root, "hermes_elevated_raw_stale", old=True)
                fresh = self._make_raw_dir(root, "hermes_elevated_raw_fresh")
                foreign = self._make_raw_dir(root, "unrelated_dir")
                plain_file = os.path.join(root, "hermes_elevated_raw_notadir")
                with open(plain_file, "w") as f:
                    f.write("x")

                from tools.admin_executor import _cleanup_stale_raw_dirs
                _cleanup_stale_raw_dirs(ttl_seconds=3600)

                self.assertFalse(os.path.exists(stale))
                self.assertTrue(os.path.exists(fresh))
                self.assertTrue(os.path.exists(foreign))
                self.assertTrue(os.path.exists(plain_file))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_cleanup_exception_safe(self):
        import tempfile as _tf

        root = _tf.mkdtemp(prefix="hermes_test_rawttl_")
        try:
            with patch(
                "tools.admin_executor.tempfile.gettempdir", return_value=root,
            ):
                from tools.admin_executor import _cleanup_stale_raw_dirs

                with patch("tools.admin_executor.os.listdir", side_effect=OSError("boom")):
                    _cleanup_stale_raw_dirs()
                with patch(
                    "tools.admin_executor.shutil.rmtree",
                    side_effect=OSError("boom"),
                ):
                    self._make_raw_dir(root, "hermes_elevated_raw_stale", old=True)
                    _cleanup_stale_raw_dirs(ttl_seconds=0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_stage_raw_output_restricts_permissions_and_moves(self):
        import stat as _stat

        from tools.admin_executor import _stage_raw_output

        d = tempfile.mkdtemp(prefix="hermes_test_stage_")
        try:
            src = os.path.join(d, "out.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("raw elevated data")
            raw_path = _stage_raw_output(src)
            self.assertIsNotNone(raw_path)
            self.assertTrue(os.path.exists(raw_path))
            self.assertFalse(os.path.exists(src))
            if os.name != "nt":
                mode = _stat.S_IMODE(os.stat(raw_path).st_mode)
                self.assertEqual(mode, 0o600)
            shutil.rmtree(os.path.dirname(raw_path), ignore_errors=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_stage_raw_output_does_not_reap_fresh(self):
        import tempfile as _tf

        root = _tf.mkdtemp(prefix="hermes_test_rawttl_")
        try:
            with patch(
                "tools.admin_executor.tempfile.gettempdir", return_value=root,
            ):
                from tools.admin_executor import _stage_raw_output

                src1 = os.path.join(root, "src1.txt")
                with open(src1, "w", encoding="utf-8") as f:
                    f.write("first")
                raw1 = _stage_raw_output(src1)
                self.assertIsNotNone(raw1)
                src2 = os.path.join(root, "src2.txt")
                with open(src2, "w", encoding="utf-8") as f:
                    f.write("second")
                raw2 = _stage_raw_output(src2)
                self.assertIsNotNone(raw2)
                self.assertTrue(os.path.exists(os.path.dirname(raw1)))
                shutil.rmtree(os.path.dirname(raw1), ignore_errors=True)
                shutil.rmtree(os.path.dirname(raw2), ignore_errors=True)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
