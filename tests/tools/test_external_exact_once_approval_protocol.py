"""RED contract for NLS-184's FD-based external exact-once approval protocol.

This module intentionally names the proposed production FD ABI.  It contains
no in-process approval adapter seam: the pipe harness is a test-only external
peer, and Hermes receives only its Ed25519 public verification key.
"""

from __future__ import annotations

import base64
import copy
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
MODE_ENV = "HERMES_EXTERNAL_APPROVAL_MODE"
MODE_VALUE = "exact-once"
OPERATION_KIND = "terminal.command"
TOOL_IDENTITY = "terminal"
SESSION_ID = "nls-184-session"
SECRET = "NLS184_CREDENTIAL_MATERIAL_DO_NOT_TRANSPORT"
# The dynamic bytes here are deliberate: v1 fingerprints every byte exactly.
COMMAND = (
    f"printf '%s\\n' '2026-07-11T00:00:00Z /tmp/nls-184-7f9e "
    f"$(uuidgen) $(uname -s) {SECRET}'"
)
FIXTURES = Path(__file__).parents[1] / "fixtures" / "external_approval_v1"
PROCESS_HELPER = FIXTURES / "protocol_process.py"
PROCESS_MARKER = "nls-184-external-approval-process"


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
    """Test-only external peer; production sees only two integer FDs and a public key."""

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
    """Configure the proposed FD-only ABI with an external test peer's public key."""
    harness = _PipeHarness.create()
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    profile_home = str(tmp_path / "profiles" / "headless-test")

    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv(MODE_ENV, MODE_VALUE)
    monkeypatch.setenv("HERMES_HOME", profile_home)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    token = approval_module.set_current_session_key(SESSION_ID)
    approval_module.clear_session(SESSION_ID)
    approval_module.configure_external_approval_fd_protocol(
        grant_input_fd=harness.grant_read_fd,
        record_output_fd=harness.records_write_fd,
        verification_key=public_key,
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
        COMMAND.replace("2026-07-11T00:00:00Z", "2026-07-11T00:00:01Z"),
        COMMAND.replace("/tmp/nls-184-7f9e", "/tmp/nls-184-other"),
        COMMAND.replace("$(uuidgen)", "$(uuidgen)-again"),
        COMMAND.replace("$(uname -s)", "$(uname -m)"),
        COMMAND + " ",
    ),
    ids=("timestamp", "temp-path", "uuid", "shell-substitution", "whitespace"),
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
    grant_read_fd, grant_write_fd = os.pipe()
    record_read_fd, record_write_fd = os.pipe()
    if grant is not None:
        os.write(grant_write_fd, _canonical(grant) + b"\n")
    process = subprocess.Popen(
        [
            sys.executable, str(PROCESS_HELPER), role,
            "--grant-fd", str(grant_read_fd),
            "--record-fd", str(record_write_fd),
            "--verification-key", base64.b64encode(verification_key).decode("ascii"),
            "--session-id", SESSION_ID,
            "--hermes-home", profile_home,
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(grant_read_fd, record_write_fd),
        close_fds=True,
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
    _first, _request = _request_then_signed_grant(harness, private_key, executions)

    result = _run_guarded(COMMAND, executions)
    _receipt = harness.take_record()

    assert result["approved"] is True
    assert approval_module.is_session_yolo_enabled(SESSION_ID) is False
    assert approval_module._session_approved.get(SESSION_ID, set()) == set()
    assert approval_module._permanent_approved == set()
