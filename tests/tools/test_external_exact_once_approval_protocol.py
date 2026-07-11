"""RED contract for NLS-184's FD-based external exact-once approval protocol.

This module intentionally names the proposed production FD ABI.  It contains
no in-process approval adapter seam: the pipe harness is a test-only external
peer, and Hermes receives only its Ed25519 public verification key.
"""

from __future__ import annotations

import base64
import copy
import errno
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools import approval as approval_module


PROTOCOL = "hermes.external-approval"
VERSION = 1
MODE_VALUE = "exact-once"
OPERATION_KIND = "terminal.command"
TOOL_IDENTITY = "terminal"
SESSION_ID = "nls-184-session"
SECRET = "NLS184_CREDENTIAL_MATERIAL_DO_NOT_TRANSPORT"
# Dangerous host command with dynamic bytes: v1 fingerprints every byte exactly.
# Exact-once only authorizes commands that normal analysis would warn on.
COMMAND = f"rm -rf /tmp/nls-184-7f9e-$(uuidgen)-$(uname -s)-{SECRET}"
SAFE_HOST_COMMAND = "echo nls-184-safe-host-command"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "external_approval_v1"
PROCESS_HELPER = FIXTURES / "protocol_process.py"
PROCESS_MARKER = "nls-184-external-approval-process"


def _write_external_mode_config(
    profile_home: str, *, mode: str = MODE_VALUE, verification_key: bytes | None = None
) -> Path:
    """Configure external approval, including its non-secret pinned public key."""
    config_path = Path(profile_home) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    key_line = ""
    if verification_key is not None:
        key_line = (
            "    verification_key: "
            + base64.b64encode(verification_key).decode("ascii")
            + "\n"
        )
    config_path.write_text(
        "approvals:\n"
        "  mode: manual\n"
        "  external:\n"
        f"    mode: {mode}\n"
        + key_line,
        encoding="utf-8",
    )
    return config_path


def _canonical(value: dict) -> bytes:
    """The v1 signature payload: UTF-8 JSON, sorted keys, no extra whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fingerprint(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _unsigned_grant(request: dict, *, choice: str = "approve_once", **overrides) -> dict:
    now = int(time.time())
    grant = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "kind": "grant",
        "algorithm": "Ed25519",
        "approval_id": request["approval_id"],
        "operation": dict(request["operation"]),
        "session": dict(request["session"]),
        "issued_at": now - 1,
        "expires_at": now + 3_600,
        "choice": choice,
    }
    grant.update(overrides)
    return grant


def _sign(private_key: Ed25519PrivateKey, grant: dict) -> dict:
    signed = copy.deepcopy(grant)
    signed["signature"] = base64.b64encode(private_key.sign(_canonical(signed))).decode("ascii")
    return signed


@dataclass
class _PipeHarness:
    """Test-only peer; production sees only two FDs and config pins its public key."""

    grant_read_fd: int
    grant_write_fd: int
    records_read_fd: int
    records_write_fd: int
    record_buffer: bytearray = field(default_factory=bytearray)

    @classmethod
    def create(cls) -> "_PipeHarness":
        grant_read_fd, grant_write_fd = os.pipe()
        records_read_fd, records_write_fd = os.pipe()
        return cls(grant_read_fd, grant_write_fd, records_read_fd, records_write_fd)

    def send_grant(self, grant: dict) -> None:
        os.write(self.grant_write_fd, _canonical(grant) + b"\n")

    def send_raw_grant(self, raw: bytes) -> None:
        os.write(self.grant_write_fd, raw + b"\n")

    def take_record(self) -> dict:
        while b"\n" not in self.record_buffer:
            chunk = os.read(self.records_read_fd, 65536)
            if not chunk:
                raise EOFError("record output closed before a complete newline-delimited record")
            self.record_buffer.extend(chunk)
        raw, _, remainder = self.record_buffer.partition(b"\n")
        self.record_buffer = bytearray(remainder)
        return json.loads(raw.decode("utf-8"))

    def close_record_reader(self) -> None:
        os.close(self.records_read_fd)
        self.records_read_fd = -1

    def close(self) -> None:
        for fd in (self.grant_read_fd, self.grant_write_fd, self.records_read_fd, self.records_write_fd):
            if fd >= 0:
                os.close(fd)


@pytest.fixture
def external_protocol(monkeypatch, tmp_path):
    """Configure the FD-only ABI and pin the test peer's public key in config."""
    harness = _PipeHarness.create()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "headless-test")
    _write_external_mode_config(
        profile_home, mode=MODE_VALUE, verification_key=public_key
    )

    monkeypatch.setenv("HERMES_HOME", profile_home)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(
        approval_module,
        "_run_tirith_command_check",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
    )
    monkeypatch.setattr(
        approval_module, "_command_matches_permanent_allowlist", lambda _command: False
    )
    monkeypatch.setattr(approval_module, "is_approved", lambda *_args, **_kwargs: False)
    token = approval_module.set_current_session_key(SESSION_ID)
    approval_module.clear_session(SESSION_ID)
    approval_module.configure_external_approval_fd_protocol(
        grant_input_fd=harness.grant_read_fd,
        record_output_fd=harness.records_write_fd,
    )
    try:
        yield harness, private_key, profile_home
    finally:
        approval_module.clear_external_approval_fd_protocol()
        harness.close()
        approval_module.clear_session(SESSION_ID)
        approval_module.reset_current_session_key(token)


def _run_guarded(command: str, executions: list[str]) -> dict:
    """Execution fixture proving guard decisions admit zero or one side effects."""
    result = approval_module.check_all_command_guards(command, "local")
    if result["approved"]:
        executions.append(command)
    return result


def _assert_no_protocol_records(harness: _PipeHarness) -> None:
    """Fail if the record FD already has bytes (including a partial frame)."""
    import select

    ready, _, _ = select.select([harness.records_read_fd], [], [], 0)
    assert not ready, "expected no external protocol records on the record FD"
    assert harness.record_buffer == bytearray()


def _request_then_signed_grant(harness, private_key, executions: list[str]) -> tuple[dict, dict]:
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    harness.send_grant(_sign(private_key, _unsigned_grant(request)))
    return first, request


def test_v1_fixtures_are_canonical_and_document_the_schema_contract():
    """Fixtures are transport records, not unsigned authorization shortcuts."""
    schema = json.loads((FIXTURES / "schema.json").read_text())
    assert schema["$id"].endswith("/v1/schema.json")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    for name, expected_kind in (("request.json", "request"), ("grant.json", "grant"), ("receipt.json", "receipt")):
        raw = (FIXTURES / name).read_bytes()
        record = json.loads(raw)
        # The fixture stores one canonical record plus the FD's required newline frame.
        assert raw == _canonical(record) + b"\n"
        assert record["protocol"] == PROTOCOL
        assert record["version"] == VERSION
        assert record["kind"] == expected_kind


def test_request_builder_binds_terminal_identity_and_every_raw_command_byte():
    request = approval_module.build_external_approval_request(
        command=COMMAND,
        operation_kind=OPERATION_KIND,
        tool_identity=TOOL_IDENTITY,
        session_id=SESSION_ID,
        profile="/tmp/hermes/profiles/headless-test",
    )

    assert request["protocol"] == PROTOCOL
    assert request["version"] == VERSION
    assert request["kind"] == "request"
    assert request["operation"] == {
        "kind": OPERATION_KIND,
        "tool": TOOL_IDENTITY,
        "fingerprint": _fingerprint(COMMAND),
    }
    assert SECRET not in json.dumps(request, sort_keys=True)


@pytest.mark.parametrize(
    "changed_command",
    (
        COMMAND.replace("NLS184_CREDENTIAL", "NLS184_CHANGED"),
        COMMAND.replace("/tmp/nls-184-7f9e", "/tmp/nls-184-other"),
        COMMAND.replace("$(uuidgen)", "$(uuidgen)-again"),
        COMMAND.replace("$(uname -s)", "$(uname -m)"),
        COMMAND + " ",
    ),
    ids=("credential", "temp-path", "uuid", "shell-substitution", "whitespace"),
)
def test_every_dynamic_or_shell_byte_requires_a_new_approval(changed_command):
    original = approval_module.build_external_approval_request(
        command=COMMAND, operation_kind=OPERATION_KIND, tool_identity=TOOL_IDENTITY,
        session_id=SESSION_ID, profile="profile",
    )
    changed = approval_module.build_external_approval_request(
        command=changed_command, operation_kind=OPERATION_KIND, tool_identity=TOOL_IDENTITY,
        session_id=SESSION_ID, profile="profile",
    )
    assert original["operation"]["fingerprint"] == _fingerprint(COMMAND)
    assert changed["operation"]["fingerprint"] == _fingerprint(changed_command)
    assert changed["operation"]["fingerprint"] != original["operation"]["fingerprint"]
    assert changed["approval_id"] != original["approval_id"]


def test_named_profile_binding_is_stable_and_not_machine_specific(monkeypatch):
    import hermes_constants

    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.setattr(
        hermes_constants,
        "get_hermes_home",
        lambda: Path("/srv/hermes/profiles/linear-agent-dev"),
    )
    assert approval_module._external_approval_profile_binding() == "linear-agent-dev"

    monkeypatch.setenv("HERMES_PROFILE", "linear-agent-explicit")
    assert approval_module._external_approval_profile_binding() == "linear-agent-explicit"


def test_protocol_records_use_dedicated_fds_not_stdio_or_a_generic_adapter(external_protocol, capsys):
    harness, _private_key, _profile_home = external_protocol
    executions: list[str] = []

    result = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    captured = capsys.readouterr()

    assert result["approved"] is False
    assert executions == []
    assert request["kind"] == "request"
    assert SECRET not in json.dumps(request, sort_keys=True)
    assert _canonical(request).decode("utf-8") not in captured.out
    assert _canonical(request).decode("utf-8") not in captured.err
    assert not hasattr(approval_module, "set_external_approval_adapter")


def _read_all_records(fd: int) -> list[dict]:
    chunks = []
    while chunk := os.read(fd, 65536):
        chunks.append(chunk)
    return [json.loads(line) for line in b"".join(chunks).splitlines()]


def _run_protocol_process(*, role: str, profile_home: str, verification_key: bytes, grant: dict | None = None):
    """Run one isolated Hermes process with protocol records restricted to pipes."""
    _write_external_mode_config(
        profile_home, mode=MODE_VALUE, verification_key=verification_key
    )
    grant_read_fd, grant_write_fd = os.pipe()
    record_read_fd, record_write_fd = os.pipe()
    if grant is not None:
        os.write(grant_write_fd, _canonical(grant) + b"\n")
    process = subprocess.Popen(
        [
            sys.executable, str(PROCESS_HELPER), role,
            "--grant-fd", str(grant_read_fd),
            "--record-fd", str(record_write_fd),
            "--session-id", SESSION_ID,
            "--hermes-home", profile_home,
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(grant_read_fd, record_write_fd),
        close_fds=True,
        env={
            **os.environ,
            # Prefer this worktree over any editable install of hermes-agent.
            "PYTHONPATH": str(Path(__file__).parents[2])
            + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        },
    )
    os.close(grant_read_fd)
    os.close(grant_write_fd)
    os.close(record_write_fd)
    try:
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"protocol child {role!r} did not exit within 10 seconds: {stderr}")
        records = _read_all_records(record_read_fd)
    finally:
        os.close(record_read_fd)
    assert process.returncode == 0, stderr
    return json.loads(stdout), records


def test_valid_ed25519_grant_is_consumed_once_across_resumed_hermes_processes(tmp_path):
    """Durable consumption must survive process exit, not just global reconfiguration."""
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    verification_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "durable-replay-test")

    first_marker, first_records = _run_protocol_process(
        role="request", profile_home=profile_home, verification_key=verification_key
    )
    request = first_records[0]
    grant = _sign(private_key, _unsigned_grant(request))
    second_marker, second_records = _run_protocol_process(
        role="consume", profile_home=profile_home, verification_key=verification_key, grant=grant
    )
    third_marker, third_records = _run_protocol_process(
        role="replay", profile_home=profile_home, verification_key=verification_key, grant=grant
    )

    assert first_marker == {
        "approved": False,
        "executions": 0,
        "marker": PROCESS_MARKER,
        "role": "request",
    }
    assert first_records == [request]
    assert request["kind"] == "request"
    assert second_marker == {
        "approved": True,
        "executions": 1,
        "marker": PROCESS_MARKER,
        "role": "consume",
    }
    assert len(second_records) == 1
    assert second_records[0]["kind"] == "receipt"
    assert second_records[0]["approval_id"] == request["approval_id"]
    assert third_marker == {
        "approved": False,
        "executions": 0,
        "marker": PROCESS_MARKER,
        "role": "replay",
    }
    assert len(third_records) == 1
    assert third_records[0]["kind"] == "request"
    assert third_records[0]["approval_id"] == request["approval_id"]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda grant: grant.update(signature=base64.b64encode(b"\x00" * 64).decode("ascii")),
        lambda grant: grant["operation"].update(fingerprint="0" * 64),
        lambda grant: grant["session"].update(id="other-session"),
        lambda grant: grant["session"].update(profile="other-profile"),
    ),
    ids=("invalid-signature", "tampered-after-signing", "wrong-session-after-signing", "wrong-profile-after-signing"),
)
def test_invalid_signature_and_post_signing_tampering_fail_closed(external_protocol, mutate):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    _first, request = _request_then_signed_grant(harness, private_key, executions)
    # Replace the queued valid record with the exact adversarial record.
    os.read(harness.grant_read_fd, 65536)
    grant = _sign(private_key, _unsigned_grant(request))
    mutate(grant)
    harness.send_grant(grant)

    result = _run_guarded(COMMAND, executions)

    assert result["approved"] is False
    assert executions == []


def test_trusted_signature_with_an_unsupported_choice_still_fails_closed(external_protocol):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    harness.send_grant(_sign(private_key, _unsigned_grant(request, choice="approve_session")))

    result = _run_guarded(COMMAND, executions)

    assert first["approved"] is False
    assert result["approved"] is False
    assert executions == []


def test_duplicate_choice_member_is_not_accepted_as_an_unsigned_json_fixture(external_protocol):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    valid = _sign(private_key, _unsigned_grant(request))
    raw_with_duplicate_choice = _canonical(valid).replace(
        b'"choice":"approve_once",',
        b'"choice":"approve_once","choice":"approve_once",',
        1,
    )
    harness.send_raw_grant(raw_with_duplicate_choice)

    result = _run_guarded(COMMAND, executions)

    assert first["approved"] is False
    assert result["approved"] is False
    assert executions == []


def test_grant_signed_by_a_different_key_fails_closed(external_protocol):
    harness, _trusted_private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    wrong_private_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
    harness.send_grant(_sign(wrong_private_key, _unsigned_grant(request)))

    result = _run_guarded(COMMAND, executions)

    assert first["approved"] is False
    assert result["approved"] is False
    assert executions == []


@pytest.mark.parametrize(
    "remove_path",
    (
        ("protocol",), ("version",), ("approval_id",), ("operation", "fingerprint"),
        ("session", "id"), ("session", "profile"), ("issued_at",), ("expires_at",),
        ("choice",), ("signature",),
    ),
    ids=("protocol", "version", "approval-id", "fingerprint", "session-id", "profile", "issued-at", "expires-at", "choice", "signature"),
)
def test_grant_missing_any_security_binding_field_fails_closed(external_protocol, remove_path):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    grant = _sign(private_key, _unsigned_grant(request))
    target = grant
    for key in remove_path[:-1]:
        target = target[key]
    target.pop(remove_path[-1])
    harness.send_grant(grant)

    result = _run_guarded(COMMAND, executions)

    assert first["approved"] is False
    assert result["approved"] is False
    assert executions == []


def test_unsigned_fixture_never_authorizes_and_receipt_failure_fails_closed(external_protocol):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    unsigned = _unsigned_grant(request)
    harness.send_grant(unsigned)
    unsigned_result = _run_guarded(COMMAND, executions)

    # Make receipt delivery fail only after a correctly signed grant is available.
    harness.close_record_reader()
    harness.send_grant(_sign(private_key, _unsigned_grant(request)))
    receipt_failed_result = _run_guarded(COMMAND, executions)
    # A failure to report consumption must not permit an execution or a replay.
    replay_result = _run_guarded(COMMAND, executions)

    assert first["approved"] is False
    assert unsigned_result["approved"] is False
    assert receipt_failed_result["approved"] is False
    assert replay_result["approved"] is False
    assert executions == []


def test_normal_tool_subprocess_cannot_inherit_or_read_protocol_fds(external_protocol):
    harness, _private_key, _profile_home = external_protocol
    tool_kwargs = approval_module.external_approval_tool_subprocess_kwargs()
    completed = subprocess.run(
        [
            sys.executable, str(PROCESS_HELPER), "fd-probe", "--probe-fds",
            str(harness.grant_read_fd), str(harness.records_write_fd),
        ],
        text=True,
        capture_output=True,
        check=True,
        **tool_kwargs,
    )

    assert tool_kwargs["close_fds"] is True
    assert tool_kwargs.get("pass_fds", ()) == ()
    assert os.get_inheritable(harness.grant_read_fd) is False
    assert os.get_inheritable(harness.records_write_fd) is False
    assert json.loads(completed.stdout) == {"grant": False, "records": False}


def test_external_grant_never_mutates_yolo_session_or_permanent_allowlists(external_protocol):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    before_permanent = set(approval_module._permanent_approved)
    before_session = set(approval_module._session_approved.get(SESSION_ID, set()))
    _first, _request = _request_then_signed_grant(harness, private_key, executions)

    result = _run_guarded(COMMAND, executions)
    _receipt = harness.take_record()

    assert result["approved"] is True
    assert approval_module.is_session_yolo_enabled(SESSION_ID) is False
    assert approval_module._session_approved.get(SESSION_ID, set()) == before_session
    assert approval_module._permanent_approved == before_permanent


def test_config_mode_off_keeps_external_protocol_inactive_even_with_fds(monkeypatch, tmp_path):
    """approvals.external.mode defaults/off must not activate exact-once even when FDs are wired."""
    harness = _PipeHarness.create()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "mode-off")
    _write_external_mode_config(
        profile_home, mode="off", verification_key=public_key
    )
    monkeypatch.setenv("HERMES_HOME", profile_home)
    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(
        approval_module,
        "_run_tirith_command_check",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
    )
    monkeypatch.setattr(
        approval_module, "_command_matches_permanent_allowlist", lambda _command: False
    )
    monkeypatch.setattr(approval_module, "is_approved", lambda *_args, **_kwargs: False)
    token = approval_module.set_current_session_key(SESSION_ID)
    approval_module.configure_external_approval_fd_protocol(
        grant_input_fd=harness.grant_read_fd,
        record_output_fd=harness.records_write_fd,
    )
    try:
        assert approval_module._is_external_exact_once_active() is False
        executions: list[str] = []
        result = _run_guarded(COMMAND, executions)
        assert result.get("external_approval") is None
        assert executions == [] or result["approved"] in (True, False)
    finally:
        approval_module.clear_external_approval_fd_protocol()
        harness.close()
        approval_module.reset_current_session_key(token)


def test_config_exact_once_activates_without_behavioral_env_var(external_protocol):
    harness, _private_key, profile_home = external_protocol
    assert (Path(profile_home) / "config.yaml").read_text().count("exact-once") == 1
    assert os.getenv("HERMES_EXTERNAL_APPROVAL_MODE") in (None, "")
    assert approval_module._is_external_exact_once_active() is True
    executions: list[str] = []
    result = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    assert result["approved"] is False
    assert request["kind"] == "request"
    assert executions == []


def test_cli_parser_accepts_only_hidden_external_approval_fd_flags():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args([
        "chat",
        "-q",
        "ping",
        "--external-approval-grant-fd",
        "3",
        "--external-approval-record-fd", "4",
    ])
    assert args.external_approval_grant_fd == 3
    assert args.external_approval_record_fd == 4
    assert not hasattr(args, "external_approval_verification_key")
    with pytest.raises(SystemExit):
        parser.parse_args([
            "chat", "-q", "ping", "--external-approval-verification-key",
            base64.b64encode(bytes(range(32))).decode("ascii"),
        ])


def test_cli_bootstrap_wires_inherited_fds_and_keeps_records_off_stdio(tmp_path, monkeypatch):
    """Node adapters spawn hermes chat -q with pass_fds; bootstrap must configure the ABI."""
    from argparse import Namespace

    from hermes_cli.main import bootstrap_external_approval_cli

    harness = _PipeHarness.create()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "cli-bootstrap")
    _write_external_mode_config(
        profile_home, mode=MODE_VALUE, verification_key=public_key
    )
    monkeypatch.setenv("HERMES_HOME", profile_home)
    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

    args = Namespace(
        external_approval_grant_fd=harness.grant_read_fd,
        external_approval_record_fd=harness.records_write_fd,
    )
    token = approval_module.set_current_session_key(SESSION_ID)
    try:
        bootstrap_external_approval_cli(args)
        assert approval_module._is_external_exact_once_active() is True
        executions: list[str] = []
        result = _run_guarded(COMMAND, executions)
        request = harness.take_record()
        assert result["approved"] is False
        assert request["kind"] == "request"
        assert SECRET not in json.dumps(request, sort_keys=True)
        assert executions == []
    finally:
        approval_module.clear_external_approval_fd_protocol()
        harness.close()
        approval_module.reset_current_session_key(token)


def test_cli_bootstrap_grant_without_record_fails_closed(monkeypatch):
    """Grant FD alone is rejected — adapters cannot half-enable consume without records."""
    from argparse import Namespace

    from hermes_cli.main import bootstrap_external_approval_cli

    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    with pytest.raises(SystemExit) as exc:
        bootstrap_external_approval_cli(Namespace(
            external_approval_grant_fd=3,
            external_approval_record_fd=None,
        ))
    assert exc.value.code == 2
    assert approval_module._external_fd_protocol is None


def test_cli_bootstrap_record_only_configures_and_emits_request(tmp_path, monkeypatch):
    """NLS-191: first-turn smoke may pass only --external-approval-record-fd."""
    from argparse import Namespace

    from hermes_cli.main import bootstrap_external_approval_cli

    harness = _PipeHarness.create()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "record-only")
    _write_external_mode_config(
        profile_home, mode=MODE_VALUE, verification_key=public_key
    )
    monkeypatch.setenv("HERMES_HOME", profile_home)
    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(
        approval_module,
        "_run_tirith_command_check",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
    )
    monkeypatch.setattr(
        approval_module, "_command_matches_permanent_allowlist", lambda _command: False
    )
    monkeypatch.setattr(approval_module, "is_approved", lambda *_args, **_kwargs: False)

    args = Namespace(
        external_approval_grant_fd=None,
        external_approval_record_fd=harness.records_write_fd,
    )
    token = approval_module.set_current_session_key(SESSION_ID)
    try:
        bootstrap_external_approval_cli(args)
        protocol = approval_module._external_fd_protocol
        assert protocol is not None
        assert protocol.grant_input_fd is None
        assert protocol.record_output_fd == harness.records_write_fd
        assert approval_module._is_external_exact_once_active() is True
        assert approval_module._try_read_grant_line() is None

        executions: list[str] = []
        result = _run_guarded(COMMAND, executions)
        request = harness.take_record()
        assert result["approved"] is False
        assert result.get("external_approval") == "awaiting_grant"
        assert request["kind"] == "request"
        assert SECRET not in json.dumps(request, sort_keys=True)
        assert executions == []
        # Record-only must never fall back to stdin or other generic channels.
        assert approval_module._try_read_grant_line() is None
    finally:
        approval_module.clear_external_approval_fd_protocol()
        harness.close()
        approval_module.reset_current_session_key(token)


def test_configure_record_only_try_read_grant_returns_none(tmp_path, monkeypatch):
    """Record-only protocol is active for emit, but grant read is a hard no-op."""
    harness = _PipeHarness.create()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "record-only-configure")
    _write_external_mode_config(
        profile_home, mode=MODE_VALUE, verification_key=public_key
    )
    monkeypatch.setenv("HERMES_HOME", profile_home)
    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    token = approval_module.set_current_session_key(SESSION_ID)
    try:
        approval_module.configure_external_approval_fd_protocol(
            grant_input_fd=None,
            record_output_fd=harness.records_write_fd,
        )
        assert approval_module._is_external_exact_once_active() is True
        assert approval_module._try_read_grant_line() is None
    finally:
        approval_module.clear_external_approval_fd_protocol()
        harness.close()
        approval_module.reset_current_session_key(token)


def test_local_environment_subprocess_cannot_read_protocol_fds(external_protocol, tmp_path):
    """Real terminal launch path must close protocol FDs — not only the unused kwargs helper."""
    from tools.environments.local import LocalEnvironment

    harness, _private_key, _profile_home = external_protocol
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    probe = (
        f"{sys.executable} {PROCESS_HELPER} fd-probe --probe-fds "
        f"{harness.grant_read_fd} {harness.records_write_fd}"
    )
    proc = env._run_bash(probe)
    stdout, _stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0, stdout
    assert json.loads(stdout) == {"grant": False, "records": False}


def test_concurrent_processes_cannot_both_consume_the_same_grant(tmp_path):
    """O_EXCL consume markers must admit exactly one winner across Hermes processes."""
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    verification_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "race-test")
    _write_external_mode_config(profile_home, mode=MODE_VALUE)

    first_marker, first_records = _run_protocol_process(
        role="request", profile_home=profile_home, verification_key=verification_key
    )
    request = first_records[0]
    grant = _sign(private_key, _unsigned_grant(request))

    grant_payload = _canonical(grant) + b"\n"
    procs = []
    record_readers = []
    for _ in range(2):
        grant_read_fd, grant_write_fd = os.pipe()
        record_read_fd, record_write_fd = os.pipe()
        os.write(grant_write_fd, grant_payload)
        process = subprocess.Popen(
            [
                sys.executable, str(PROCESS_HELPER), "consume",
                "--grant-fd", str(grant_read_fd),
                "--record-fd", str(record_write_fd),
                "--session-id", SESSION_ID,
                "--hermes-home", profile_home,
            ],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(grant_read_fd, record_write_fd),
            close_fds=True,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).parents[2])
                + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            },
        )
        os.close(grant_read_fd)
        os.close(grant_write_fd)
        os.close(record_write_fd)
        procs.append(process)
        record_readers.append(record_read_fd)

    markers = []
    receipts = []
    for process, record_read_fd in zip(procs, record_readers):
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"race child hung: {stderr}")
        assert process.returncode == 0, stderr
        markers.append(json.loads(stdout))
        receipts.extend(_read_all_records(record_read_fd))
        os.close(record_read_fd)

    approved = [m for m in markers if m["approved"] is True]
    denied = [m for m in markers if m["approved"] is False]
    assert first_marker["role"] == "request"
    assert len(approved) == 1
    assert len(denied) == 1
    assert sum(m["executions"] for m in markers) == 1
    assert sum(1 for r in receipts if r.get("kind") == "receipt") == 1


# ---------------------------------------------------------------------------
# Review-fix RED contracts (Codex attempt-3 blockers)
# ---------------------------------------------------------------------------


def test_claim_fsyncs_parent_consumed_directory_before_authorizing(monkeypatch, tmp_path):
    """After the O_EXCL marker fsync, the parent consumed dir must be fsynced too."""
    profile_home = tmp_path / "profiles" / "parent-fsync"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    approval_id = "appr_v1_parent_fsync"
    marker = approval_module._consumed_marker_path(approval_id)
    consumed_dir = marker.parent

    fsynced_paths: list[str] = []
    real_fsync = os.fsync
    real_open = os.open

    def tracking_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        tracking_open._fd_paths[fd] = str(path)  # type: ignore[attr-defined]
        return fd

    tracking_open._fd_paths = {}  # type: ignore[attr-defined]

    def tracking_fsync(fd):
        path = tracking_open._fd_paths.get(fd)  # type: ignore[attr-defined]
        if path is None:
            try:
                path = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                path = f"<fd:{fd}>"
        fsynced_paths.append(path)
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "fsync", tracking_fsync)

    assert approval_module._claim_consumed_approval_id(approval_id) is True
    assert marker.is_file()
    assert any(Path(p).resolve() == marker.resolve() for p in fsynced_paths)
    assert any(Path(p).resolve() == consumed_dir.resolve() for p in fsynced_paths)
    marker_idx = next(
        i for i, p in enumerate(fsynced_paths) if Path(p).resolve() == marker.resolve()
    )
    dir_idx = next(
        i for i, p in enumerate(fsynced_paths)
        if Path(p).resolve() == consumed_dir.resolve()
    )
    assert dir_idx > marker_idx


def test_parent_consumed_directory_fsync_failure_is_permanent_fail_closed(
    monkeypatch, tmp_path
):
    """Directory open/fsync/close failure after a successful claim must stick and deny."""
    profile_home = tmp_path / "profiles" / "parent-fsync-fail"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    approval_id = "appr_v1_parent_fsync_fail"
    marker = approval_module._consumed_marker_path(approval_id)
    consumed_dir = str(marker.parent)

    real_open = os.open
    real_fsync = os.fsync
    real_close = os.close
    dir_fds: set[int] = set()

    def selective_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if str(path) == consumed_dir and not (flags & os.O_CREAT):
            dir_fds.add(fd)
        return fd

    def failing_dir_fsync(fd):
        if fd in dir_fds:
            raise OSError(errno.EIO, "injected parent directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", selective_open)
    monkeypatch.setattr(os, "fsync", failing_dir_fsync)

    assert approval_module._claim_consumed_approval_id(approval_id) is False
    assert marker.is_file(), "failed parent fsync must leave the claim marker in place"
    monkeypatch.setattr(os, "open", real_open)
    monkeypatch.setattr(os, "fsync", real_fsync)
    monkeypatch.setattr(os, "close", real_close)
    assert approval_module._claim_consumed_approval_id(approval_id) is False


@pytest.mark.parametrize(
    "bad_args",
    (
        {"grant_fd": 0, "record_fd": 4},
        {"grant_fd": 3, "record_fd": 1},
        {"grant_fd": 3, "record_fd": 3},
        {"grant_fd": -1, "record_fd": 4},
    ),
    ids=("grant-stdin", "record-stdout", "duplicate-fds", "negative-fd"),
)
def test_cli_bootstrap_rejects_stdio_duplicate_and_negative_fds(monkeypatch, bad_args):
    from argparse import Namespace

    from hermes_cli.main import bootstrap_external_approval_cli

    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    with pytest.raises(SystemExit) as exc:
        bootstrap_external_approval_cli(Namespace(
            external_approval_grant_fd=bad_args["grant_fd"],
            external_approval_record_fd=bad_args["record_fd"],
        ))
    assert exc.value.code == 2


def test_cli_bootstrap_rejects_closed_and_wrong_direction_fds(monkeypatch):
    from argparse import Namespace

    from hermes_cli.main import bootstrap_external_approval_cli

    monkeypatch.delenv("HERMES_EXTERNAL_APPROVAL_MODE", raising=False)
    r, w = os.pipe()
    closed_fd = r
    os.close(r)
    os.close(w)
    with pytest.raises(SystemExit) as exc:
        bootstrap_external_approval_cli(Namespace(
            external_approval_grant_fd=closed_fd,
            external_approval_record_fd=max(closed_fd + 1, 10),
        ))
    assert exc.value.code == 2

    grant_r, grant_w = os.pipe()
    rec_r, rec_w = os.pipe()
    try:
        with pytest.raises(SystemExit) as exc:
            bootstrap_external_approval_cli(Namespace(
                external_approval_grant_fd=grant_w,
                external_approval_record_fd=rec_r,
            ))
        assert exc.value.code == 2
    finally:
        for fd in (grant_r, grant_w, rec_r, rec_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_configure_does_not_suppress_set_inheritable_failure(monkeypatch):
    grant_r, grant_w = os.pipe()
    rec_r, rec_w = os.pipe()
    try:
        def boom(fd, inheritable):
            raise OSError(errno.EBADF, "injected set_inheritable failure")

        monkeypatch.setattr(os, "set_inheritable", boom)
        with pytest.raises(OSError):
            approval_module.configure_external_approval_fd_protocol(
                grant_input_fd=grant_r,
                record_output_fd=rec_w,
            )
        assert approval_module._external_fd_protocol is None
    finally:
        approval_module.clear_external_approval_fd_protocol()
        for fd in (grant_r, grant_w, rec_r, rec_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_grant_select_and_read_errors_deny_cleanly(external_protocol, monkeypatch):
    harness, _private_key, _profile_home = external_protocol
    executions: list[str] = []

    def boom_select(*_args, **_kwargs):
        raise OSError(errno.EINTR, "injected select failure")

    monkeypatch.setattr(approval_module.select, "select", boom_select)
    result = _run_guarded(COMMAND, executions)
    assert result["approved"] is False
    assert executions == []
    request = harness.take_record()
    assert request["kind"] == "request"


def test_strict_grant_rejects_extra_keys_and_wrong_native_types(external_protocol):
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    assert first["approved"] is False

    cases = []

    extra_top = _unsigned_grant(request)
    extra_top["extra_field"] = "nope"
    cases.append(_sign(private_key, extra_top))

    extra_op = _unsigned_grant(request)
    extra_op["operation"] = dict(extra_op["operation"], extra="x")
    cases.append(_sign(private_key, extra_op))

    extra_sess = _unsigned_grant(request)
    extra_sess["session"] = dict(extra_sess["session"], extra="x")
    cases.append(_sign(private_key, extra_sess))

    bool_version = _unsigned_grant(request)
    bool_version["version"] = True
    cases.append(_sign(private_key, bool_version))

    float_ts = _unsigned_grant(request)
    float_ts["issued_at"] = float(float_ts["issued_at"]) + 0.5
    cases.append(_sign(private_key, float_ts))

    for grant in cases:
        harness.send_grant(grant)
        result = _run_guarded(COMMAND, executions)
        assert result["approved"] is False, grant
        assert executions == []


def test_validly_signed_noncanonical_wire_bytes_are_rejected(external_protocol):
    """Signature verification must require the raw FD line to already be canonical JSON."""
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []
    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    signed = _sign(private_key, _unsigned_grant(request))
    canonical = _canonical(signed)

    # Single-line spaced JSON still parses to the same object; must not verify
    # via re-canonicalization of a parsed dict.
    noncanonical = json.dumps(signed, sort_keys=True, separators=(", ", ": ")).encode("utf-8")
    assert b"\n" not in noncanonical
    assert noncanonical != canonical
    harness.send_raw_grant(noncanonical)

    result = _run_guarded(COMMAND, executions)

    assert first["approved"] is False
    assert result["approved"] is False
    assert executions == []


def test_config_pins_the_trust_root_and_cli_cannot_select_a_different_key(tmp_path, monkeypatch):
    """Only the configured public key, never a child CLI value, may verify grants."""
    harness = _PipeHarness.create()
    trusted_private = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    untrusted_private = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
    trusted_public = trusted_private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "pinned-root")
    _write_external_mode_config(
        profile_home, mode=MODE_VALUE, verification_key=trusted_public
    )
    monkeypatch.setenv("HERMES_HOME", profile_home)
    from argparse import Namespace

    from hermes_cli.main import bootstrap_external_approval_cli

    token = approval_module.set_current_session_key(SESSION_ID)
    try:
        # This obsolete/forged Namespace member models inherited plumbing from
        # an older child. Bootstrap must ignore it and retain the config key.
        bootstrap_external_approval_cli(Namespace(
            external_approval_grant_fd=harness.grant_read_fd,
            external_approval_record_fd=harness.records_write_fd,
            external_approval_verification_key=base64.b64encode(
                untrusted_private.public_key().public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw
                )
            ).decode("ascii"),
        ))
        first = _run_guarded(COMMAND, [])
        request = harness.take_record()
        assert first["approved"] is False
        harness.send_grant(_sign(untrusted_private, _unsigned_grant(request)))
        assert _run_guarded(COMMAND, [])["approved"] is False
        assert harness.take_record()["kind"] == "request"
    finally:
        approval_module.clear_external_approval_fd_protocol()
        harness.close()
        approval_module.reset_current_session_key(token)


@pytest.mark.parametrize("stdio_fd", (0, 1, 2), ids=("stdin", "stdout", "stderr"))
def test_configure_rejects_fds_that_alias_stdio(stdio_fd):
    """Numeric checks are insufficient: os.dup(0/1/2) is still stdio."""
    alias_fd = os.dup(stdio_fd)
    grant_r, grant_w = os.pipe()
    record_r, record_w = os.pipe()
    try:
        kwargs = {
            "grant_input_fd": alias_fd if stdio_fd == 0 else grant_r,
            "record_output_fd": alias_fd if stdio_fd != 0 else record_w,
        }
        with pytest.raises(OSError) as exc:
            approval_module.configure_external_approval_fd_protocol(**kwargs)
        assert exc.value.errno == errno.EINVAL
    finally:
        approval_module.clear_external_approval_fd_protocol()
        for fd in (alias_fd, grant_r, grant_w, record_r, record_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_protocol_directory_creation_fsyncs_each_new_private_directory_parent(monkeypatch, tmp_path):
    """A first-use crash cannot lose either private protocol directory entry."""
    profile_home = tmp_path / "profiles" / "durable-protocol-dirs"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    external_dir = profile_home / ".external-approval"
    consumed_dir = external_dir / "consumed"
    opened: dict[int, Path] = {}
    fsynced: list[Path] = []
    real_open = os.open
    real_fsync = os.fsync

    def tracking_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        opened[fd] = Path(path)
        return fd

    def tracking_fsync(fd):
        if fd in opened:
            fsynced.append(opened[fd].resolve())
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "fsync", tracking_fsync)
    assert approval_module._claim_consumed_approval_id("appr_v1_fresh_dirs") is True
    assert profile_home.resolve() in fsynced
    assert external_dir.resolve() in fsynced
    assert consumed_dir.resolve() in fsynced


def test_protocol_directory_parent_fsync_failure_denies_before_marker(monkeypatch, tmp_path):
    """No one-shot grant is consumed if first-use directory durability fails."""
    profile_home = tmp_path / "profiles" / "durable-protocol-dir-fail"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    approval_id = "appr_v1_dir_fsync_failure"
    marker = approval_module._consumed_marker_path(approval_id)
    real_open = os.open
    real_fsync = os.fsync
    parent_fds: set[int] = set()

    def tracking_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if Path(path) == profile_home:
            parent_fds.add(fd)
        return fd

    def fail_parent_fsync(fd):
        if fd in parent_fds:
            raise OSError(errno.EIO, "injected first-use parent fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    assert approval_module._claim_consumed_approval_id(approval_id) is False
    assert not marker.exists()


@pytest.mark.parametrize("failure", ("mkdir", "chmod", "open", "fsync", "close"))
def test_private_protocol_directory_setup_fails_closed_on_every_io_step(
    monkeypatch, tmp_path, failure
):
    """Each first-use create/publish I/O failure denies the grant before a marker."""
    profile_home = tmp_path / "profiles" / f"private-dir-{failure}"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    external_dir = profile_home / ".external-approval"
    real_mkdir = os.mkdir
    real_chmod = os.chmod
    real_open = os.open
    real_fsync = os.fsync
    real_close = os.close

    if failure == "mkdir":
        monkeypatch.setattr(
            os, "mkdir",
            lambda path, mode=0o777, *, dir_fd=None: (
                (_ for _ in ()).throw(OSError(errno.EIO, "injected mkdir failure"))
                if Path(path) == external_dir else real_mkdir(path, mode, dir_fd=dir_fd)
            ),
        )
    elif failure == "chmod":
        monkeypatch.setattr(
            os, "chmod",
            lambda path, mode, *, dir_fd=None, follow_symlinks=True: (
                (_ for _ in ()).throw(OSError(errno.EPERM, "injected chmod failure"))
                if Path(path) == external_dir
                else real_chmod(path, mode, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            ),
        )
    elif failure == "open":
        monkeypatch.setattr(
            os, "open",
            lambda path, flags, mode=0o777, *, dir_fd=None: (
                (_ for _ in ()).throw(OSError(errno.EIO, "injected open failure"))
                if Path(path) == profile_home
                else real_open(path, flags, mode, dir_fd=dir_fd)
            ),
        )
    elif failure == "fsync":
        monkeypatch.setattr(
            os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError(errno.EIO, "injected fsync failure"))
        )
    else:
        monkeypatch.setattr(
            os, "close", lambda _fd: (_ for _ in ()).throw(OSError(errno.EIO, "injected close failure"))
        )

    assert approval_module._claim_consumed_approval_id(f"appr_v1_{failure}") is False


def test_oversized_unterminated_grant_permanently_fails_closed(external_protocol, monkeypatch):
    """A malicious never-newline frame cannot grow memory or recover later."""
    harness, _private_key, _profile_home = external_protocol
    monkeypatch.setattr(approval_module.select, "select", lambda *_args: ([harness.grant_read_fd], [], []))
    monkeypatch.setattr(
        approval_module.os,
        "read",
        lambda *_args: b"x" * (64 * 1024 + 1),
    )
    result = _run_guarded(COMMAND, [])
    assert result["approved"] is False
    assert len(approval_module._grant_read_buffer) <= 64 * 1024
    assert approval_module._external_fd_protocol_failed is True


def test_safe_host_command_emits_no_external_protocol_records(external_protocol):
    """Safe host commands must not enter the exact-once request/grant/receipt path."""
    harness, _private_key, _profile_home = external_protocol
    executions: list[str] = []

    result = _run_guarded(SAFE_HOST_COMMAND, executions)

    assert result["approved"] is True
    assert result.get("external_approval") is None
    assert executions == [SAFE_HOST_COMMAND]
    _assert_no_protocol_records(harness)


def test_dangerous_command_exact_once_emits_request_then_consumes_once(external_protocol):
    """Dangerous commands still require a signed grant and emit request then receipt."""
    harness, private_key, _profile_home = external_protocol
    executions: list[str] = []

    first = _run_guarded(COMMAND, executions)
    assert first["approved"] is False
    assert first.get("external_approval") == "awaiting_grant"
    request = harness.take_record()
    assert request["kind"] == "request"
    assert request["operation"]["fingerprint"] == _fingerprint(COMMAND)
    assert executions == []

    harness.send_grant(_sign(private_key, _unsigned_grant(request)))
    second = _run_guarded(COMMAND, executions)
    receipt = harness.take_record()

    assert second["approved"] is True
    assert second.get("external_approval") == "consumed"
    assert receipt["kind"] == "receipt"
    assert receipt["approval_id"] == request["approval_id"]
    assert executions == [COMMAND]

    third = _run_guarded(COMMAND, executions)
    replay_request = harness.take_record()
    assert third["approved"] is False
    assert replay_request["kind"] == "request"
    assert executions == [COMMAND]


def test_permanently_allowlisted_command_emits_no_external_protocol_records(
    external_protocol, monkeypatch
):
    """Permanently allowlisted command text skips exact-once records entirely."""
    harness, _private_key, _profile_home = external_protocol
    allowlisted = "bash -c 'echo nls-184-permanently-allowlisted'"
    monkeypatch.setattr(
        approval_module,
        "_command_matches_permanent_allowlist",
        lambda command: command == allowlisted,
    )
    executions: list[str] = []

    result = _run_guarded(allowlisted, executions)

    assert result["approved"] is True
    assert result.get("external_approval") is None
    assert executions == [allowlisted]
    _assert_no_protocol_records(harness)


def test_yolo_and_approvals_mode_off_do_not_bypass_external_when_warnings_exist(
    external_protocol, monkeypatch
):
    """Headless exact-once stays enforced even under yolo / approvals.mode=off."""
    harness, private_key, _profile_home = external_protocol
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "off")
    approval_module.enable_session_yolo(SESSION_ID)
    executions: list[str] = []

    first = _run_guarded(COMMAND, executions)
    request = harness.take_record()
    assert first["approved"] is False
    assert request["kind"] == "request"
    assert executions == []

    harness.send_grant(_sign(private_key, _unsigned_grant(request)))
    second = _run_guarded(COMMAND, executions)
    receipt = harness.take_record()
    assert second["approved"] is True
    assert receipt["kind"] == "receipt"
    assert executions == [COMMAND]


def test_approval_module_imports_when_fcntl_unavailable(tmp_path):
    """Native Windows (no fcntl) must still import approval when external mode is off."""
    script = tmp_path / "import_without_fcntl.py"
    script.write_text(
        "\n".join(
            [
                "import builtins",
                "import sys",
                "real_import = builtins.__import__",
                "def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):",
                "    if name == 'fcntl':",
                "        raise ImportError('simulated Windows without fcntl')",
                "    return real_import(name, globals, locals, fromlist, level)",
                "builtins.__import__ = blocked_import",
                "sys.modules.pop('fcntl', None)",
                "for key in list(sys.modules):",
                "    if key == 'tools.approval' or key.startswith('tools.approval.'):",
                "        del sys.modules[key]",
                "import tools.approval as approval",
                "assert approval.fcntl is None",
                "assert approval._is_external_exact_once_active() is False",
                "print('IMPORT_OK')",
            ]
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "IMPORT_OK" in completed.stdout


def test_configure_external_protocol_fails_closed_when_fcntl_unavailable(
    external_protocol, monkeypatch
):
    """Active exact-once wiring must fail closed when fcntl cannot validate FDs."""
    harness, _private_key, _profile_home = external_protocol
    approval_module.clear_external_approval_fd_protocol()
    monkeypatch.setattr(approval_module, "fcntl", None)

    with pytest.raises(OSError):
        approval_module.configure_external_approval_fd_protocol(
            grant_input_fd=harness.grant_read_fd,
            record_output_fd=harness.records_write_fd,
        )
    assert approval_module._is_external_exact_once_active() is False
