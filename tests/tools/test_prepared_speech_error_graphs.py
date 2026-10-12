"""Prepared speech exports native failure facts without native diagnostics."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

import pytest
import httpx

from tools import tts_tool
from tools.tts_command_error_facts import (
    CommandFailureEvent,
    CommandFailureFacts,
    CommandOperation,
    CommandStream,
    set_command_failure_facts,
)
from tools.tts_prepared import resolve_prepared_speech, synthesize_prepared_speech
from tools.tts_prepared_types import (
    PreparedSpeechRequest,
    SpeechCause,
    SpeechCauseRelation,
    SpeechCauseType,
    SpeechFailureCode,
    SpeechSynthesisError,
)


class _WireFailure(TypedDict):
    code: str
    causes: list[dict[str, Any]]


@pytest.fixture
def native_speech(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(
        json.dumps(
            {
                "tts": {
                    "provider": "openai",
                    "openai": {
                        "api_key": "fixture-only",
                        "base_url": "https://speech.invalid/v1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    resolution = resolve_prepared_speech()
    output = tmp_path / "result.mp3"

    def fail_with(error: BaseException):
        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(tts_tool, "_generate_openai_tts", fail)
        with pytest.raises(SpeechSynthesisError) as caught:
            synthesize_prepared_speech(
                PreparedSpeechRequest("Approved speech.", str(output), resolution)
            )
        return caught.value

    return fail_with, output


def _wire(
    identity: str, index: int, relation="root", parent=None, **facts
) -> dict[str, Any]:
    return {
        "identity": identity,
        "relation": relation,
        "status_code": None,
        "retry_after": None,
        "exit_code": None,
        "context_suppressed": False,
        "errno": None,
        "stream": None,
        "operation": None,
        "parent_index": parent,
        "node_index": index,
        **facts,
    }


@pytest.mark.parametrize(
    "case",
    [
        "errno",
        "native_errno_slot",
        "shared_cycle",
        "read_close",
        "read_close_shared",
        "shadow_storage",
        "secondary_timeout",
        "wrapped_secondary",
        "forged_helper",
        "false_cause",
    ],
)
def test_prepared_failures_preserve_native_facts_and_graph_edges(
    native_speech, caplog, case
):
    fail_with, output = native_speech
    touched = []

    class NativeSlotError(OSError):
        @property
        def errno(self):
            touched.append("active property")
            return 999

    class FalseCause(TimeoutError):
        def __bool__(self):
            touched.append("truthiness")
            return False

    class ShadowStorageError(OSError):
        @property
        def __dict__(self):
            touched.append("dictionary property")
            return {}

    error = (
        NativeSlotError(
            errno.EIO, "private-native-sentinel", "/private-native-sentinel"
        )
        if case == "native_errno_slot"
        else OSError(errno.EIO, "private-native-sentinel", "/private-native-sentinel")
    )
    expected = [_wire("os_error", 0, errno=errno.EIO)]
    code = "provider_error"
    if case == "shadow_storage":
        error = ShadowStorageError(errno.EIO, "private-native-sentinel")
    if case == "false_cause":
        root = RuntimeError("private-native-sentinel")
        root.__cause__ = FalseCause("private-native-sentinel")
        expected = [
            _wire("unknown", 0, context_suppressed=True),
            _wire("timeout", 1, "cause", 0),
        ]
        code, error = "timeout", root
    if case == "shared_cycle":
        root = RuntimeError("private-native-sentinel")
        root.__cause__ = error
        root.__context__ = error
        error.__context__ = root
        expected = [
            _wire("unknown", 0, context_suppressed=True),
            _wire("os_error", 1, "cause", 0, errno=errno.EIO),
            _wire("unknown", 0, "context", 1, context_suppressed=True),
            _wire("os_error", 1, "context", 0, errno=errno.EIO),
        ]
        error = root
    if case in {
        "read_close",
        "read_close_shared",
        "shadow_storage",
        "secondary_timeout",
        "wrapped_secondary",
        "forged_helper",
    }:
        cleanup = OSError(errno.EBADF, "private-native-sentinel")
        operation = CommandOperation.CLOSE
        stream = CommandStream.STDOUT
        cleanup_wire = _wire(
            "os_error",
            1,
            "cleanup",
            0,
            errno=errno.EBADF,
            stream="stdout",
            operation="close",
        )
        expected = [
            _wire("os_error", 0, errno=errno.EIO, stream="stdout", operation="read")
        ]
        if case == "read_close_shared":
            cleanup = error
            cleanup_wire = _wire(
                "os_error",
                0,
                "cleanup",
                0,
                errno=errno.EIO,
                stream="stdout",
                operation="close",
            )
        if case == "secondary_timeout":
            cleanup = TimeoutError("private-native-sentinel")
            cleanup.__cause__ = error
            operation, stream = CommandOperation.PROCESS_CLEANUP, None
            cleanup_wire = _wire(
                "timeout",
                1,
                "cleanup",
                0,
                context_suppressed=True,
                operation="process_cleanup",
            )
        if case == "wrapped_secondary":
            cleanup = RuntimeError("private-native-sentinel")
            request = httpx.Request(
                "GET", "https://speech.invalid/private-native-sentinel"
            )
            response = httpx.Response(
                401, request=request, headers={"retry-after": "3"}
            )
            cleanup.__cause__ = httpx.HTTPStatusError(
                "private-native-sentinel", request=request, response=response
            )
            operation, stream = CommandOperation.PROCESS_CLEANUP, None
            cleanup_wire = _wire(
                "unknown",
                1,
                "cleanup",
                0,
                context_suppressed=True,
                operation="process_cleanup",
            )
        event = CommandFailureEvent(cleanup, operation, stream)
        set_command_failure_facts(
            error,
            CommandFailureFacts(CommandOperation.READ, CommandStream.STDOUT, (event,)),
        )
        expected.append(cleanup_wire)
        if case == "secondary_timeout":
            expected.append(
                _wire(
                    "os_error",
                    0,
                    "cause",
                    1,
                    errno=errno.EIO,
                    stream="stdout",
                    operation="read",
                )
            )
        if case == "wrapped_secondary":
            expected.append(
                _wire("httpx_status", 2, "cause", 1, status_code=401, retry_after=3.0)
            )
        if case == "forged_helper":
            object.__setattr__(event, "stream", "private-native-sentinel")
            expected = [_wire("os_error", 0, errno=errno.EIO)]
    error.add_note("private-native-sentinel")
    result = fail_with(error)
    payload = result.to_dict()
    assert payload == {"code": code, "causes": expected}
    assert (touched, output.exists(), result.__cause__, result.__context__) == (
        [],
        False,
        None,
        None,
    )
    records = [
        record for record in caplog.records if record.name == "tools.tts_prepared"
    ]
    assert "private-native-sentinel" not in json.dumps(payload)
    for record in records:
        assert record.exc_info is not None
        assert (type(record.exc_info[1]), record.exc_info[2]) == (
            SpeechSynthesisError,
            None,
        )
        assert record.exc_info[1].to_dict() == payload


def _decode(payload: _WireFailure) -> SpeechSynthesisError:
    causes = []
    for value in payload["causes"]:
        cause = dict(value)
        cause["identity"] = SpeechCauseType(cause["identity"])
        cause["relation"] = SpeechCauseRelation(cause["relation"])
        cause["stream"] = (
            CommandStream(cause["stream"]) if cause["stream"] is not None else None
        )
        cause["operation"] = (
            CommandOperation(cause["operation"])
            if cause["operation"] is not None
            else None
        )
        causes.append(SpeechCause(**cause))
    return SpeechSynthesisError(SpeechFailureCode(payload["code"]), tuple(causes))


_ROUNDTRIP = """
import json,sys
from tools.tts_prepared_types import SpeechCause,SpeechCauseType,SpeechCauseRelation,SpeechFailureCode,SpeechSynthesisError
from tools.tts_command_error_facts import CommandStream,CommandOperation
payload=json.load(sys.stdin)
causes=[]
for item in payload['causes']:
    item['identity']=SpeechCauseType(item['identity'])
    item['relation']=SpeechCauseRelation(item['relation'])
    item['stream']=CommandStream(item['stream']) if item['stream'] is not None else None
    item['operation']=CommandOperation(item['operation']) if item['operation'] is not None else None
    causes.append(SpeechCause(**item))
error=SpeechSynthesisError(SpeechFailureCode(payload['code']),tuple(causes))
assert set(vars(error))=={'code','causes'}
assert error.__cause__ is None and error.__context__ is None
print(json.dumps(error.to_dict()))
"""


@pytest.mark.parametrize(
    "case",
    [
        "roundtrip",
        "legacy",
        "undeclared_parent",
        "forward_node",
        "alias_position_parent",
        "inconsistent_alias",
        "boolean_errno",
        "boolean_node",
        "mixed_indices",
        "missing_cleanup_parent",
        "invalid_stream",
        "invalid_operation",
        "extra_root",
    ],
)
def test_speech_graph_process_roundtrip_and_reference_validation(case):
    payload: _WireFailure = {
        "code": "provider_error",
        "causes": [
            _wire("os_error", 0, errno=errno.EIO, stream="stdout", operation="read"),
            _wire(
                "os_error",
                1,
                "cleanup",
                0,
                errno=errno.EBADF,
                stream="stdout",
                operation="close",
            ),
            _wire("os_error", 0, "context", 1, errno=errno.EIO),
            _wire(
                "os_error",
                0,
                "cleanup",
                0,
                errno=errno.EIO,
                stream="stderr",
                operation="close",
            ),
        ],
    }
    if case == "legacy":
        error = SpeechSynthesisError(
            SpeechFailureCode.PROVIDER_ERROR,
            (
                SpeechCause(SpeechCauseType.OS_ERROR),
                SpeechCause(SpeechCauseType.UNKNOWN, SpeechCauseRelation.CAUSE),
            ),
        )
        assert error.to_dict() == {
            "code": "provider_error",
            "causes": [
                _wire("os_error", 0),
                _wire("unknown", 1, "cause", 0),
            ],
        }
        return
    if case == "roundtrip":
        process = subprocess.run(
            [sys.executable, "-c", _ROUNDTRIP],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        assert (json.loads(process.stdout), process.stderr) == (payload, "")
        assert _decode(payload).to_dict() == payload
        return
    edits = {
        "undeclared_parent": lambda: payload["causes"][1].update(parent_index=6),
        "forward_node": lambda: payload["causes"][1].update(node_index=6),
        "alias_position_parent": lambda: payload["causes"].append(
            _wire("unknown", 4, "cause", 2)
        ),
        "inconsistent_alias": lambda: payload["causes"][2].update(errno=errno.EBADF),
        "boolean_errno": lambda: payload["causes"][0].update(errno=True),
        "boolean_node": lambda: payload["causes"][0].update(node_index=True),
        "mixed_indices": lambda: payload["causes"][1].update(node_index=None),
        "missing_cleanup_parent": lambda: payload["causes"][1].update(
            parent_index=None
        ),
        "invalid_stream": lambda: payload["causes"][0].update(
            stream="private-native-sentinel"
        ),
        "invalid_operation": lambda: payload["causes"][0].update(
            operation="private-native-sentinel"
        ),
        "extra_root": lambda: payload["causes"].append(_wire("unknown", 4)),
    }
    edits[case]()
    with pytest.raises(ValueError):
        _decode(payload)
