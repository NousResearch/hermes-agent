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


def _aiohttp_error(kind: str, status: object, headers: object) -> BaseException:
    aiohttp = pytest.importorskip("aiohttp")
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    request = aiohttp.RequestInfo(
        URL("https://private-aiohttp-sentinel.invalid/secret"),
        "GET",
        CIMultiDictProxy(CIMultiDict({"Authorization": "private-aiohttp-sentinel"})),
    )
    if kind == "unknown":
        error = RuntimeError("private-aiohttp-sentinel")
    else:
        error = getattr(aiohttp, kind)(
            request, (), status=403, message="private-aiohttp-sentinel"
        )
    object.__setattr__(error, "status", status)
    object.__setattr__(error, "headers", headers)
    return error


def _assert_aiohttp_wire(result, expected: _WireFailure, caplog) -> None:
    payload = result.to_dict()
    assert payload == expected
    process = subprocess.run(
        [sys.executable, "-c", _ROUNDTRIP],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert (json.loads(process.stdout), process.stderr) == (expected, "")
    assert (result.__cause__, result.__context__, vars(result)) == (
        None,
        None,
        {"code": SpeechFailureCode(expected["code"]), "causes": result.causes},
    )
    records = [
        record for record in caplog.records if record.name == "tools.tts_prepared"
    ]
    assert records
    for record in records:
        assert record.exc_info is not None
        assert (type(record.exc_info[1]), record.exc_info[2]) == (
            SpeechSynthesisError,
            None,
        )
        assert record.exc_info[1].to_dict() == expected
    assert "private-aiohttp-sentinel" not in json.dumps(payload) + caplog.text


@pytest.mark.parametrize(
    "kind,status,headers,expected_status,expected_retry,code,placement",
    [
        (
            "ClientResponseError",
            401,
            {"Retry-After": "3"},
            401,
            3.0,
            "authentication",
            "root",
        ),
        (
            "WSServerHandshakeError",
            403,
            {"retry-after": "0"},
            403,
            0.0,
            "authentication",
            "root",
        ),
        (
            "ClientResponseError",
            429,
            {"rEtRy-AfTeR": "2.5"},
            429,
            2.5,
            "rate_limit",
            "root",
        ),
        ("ContentTypeError", 500, None, 500, None, "provider_error", "root"),
        ("ClientHttpProxyError", 407, None, 407, None, "provider_error", "root"),
        ("TooManyRedirects", 302, None, 302, None, "provider_error", "root"),
        ("ClientResponseError", 0, None, None, None, "provider_error", "root"),
        ("ClientResponseError", True, None, None, None, "provider_error", "root"),
        ("ClientResponseError", "403", None, None, None, "provider_error", "root"),
        ("ClientResponseError", 403.0, None, None, None, "provider_error", "root"),
        ("ClientResponseError", 99, None, None, None, "provider_error", "root"),
        ("ClientResponseError", 600, None, None, None, "provider_error", "root"),
        (
            "ClientResponseError",
            503,
            {"Retry-After": "Fri, 01 Jan 2021 00:00:00 GMT"},
            503,
            None,
            "provider_error",
            "root",
        ),
        (
            "ClientResponseError",
            503,
            {"Retry-After": "nan"},
            503,
            None,
            "provider_error",
            "root",
        ),
        (
            "ClientResponseError",
            503,
            {"Retry-After": "inf"},
            503,
            None,
            "provider_error",
            "root",
        ),
        (
            "ClientResponseError",
            503,
            {"Retry-After": "-1"},
            503,
            None,
            "provider_error",
            "root",
        ),
        (
            "ClientResponseError",
            503,
            {"Retry-After": True},
            503,
            None,
            "provider_error",
            "root",
        ),
        ("ClientResponseError", 503, [], 503, None, "provider_error", "root"),
        pytest.param(
            "ClientResponseError",
            403,
            {"Retry-After": 10**1000},
            403,
            None,
            "authentication",
            "root",
            id="overflow_retry",
        ),
        ("unknown", 403, {"Retry-After": "3"}, None, None, "provider_error", "root"),
        ("ClientResponseError", 403, None, 403, None, "authentication", "cause"),
        ("ClientResponseError", 403, None, 403, None, "provider_error", "suppressed"),
        ("ClientResponseError", 403, None, 403, None, "provider_error", "cleanup"),
    ],
)
def test_aiohttp_response_facts_use_native_types_and_preserve_primary_policy(
    native_speech,
    caplog,
    kind,
    status,
    headers,
    expected_status,
    expected_retry,
    code,
    placement,
):
    fail_with, output = native_speech
    error = _aiohttp_error(kind, status, headers)
    identity = "unknown" if kind == "unknown" else "aiohttp_status"
    expected = [
        _wire(identity, 0, status_code=expected_status, retry_after=expected_retry)
    ]
    if placement != "root":
        wrapper = RuntimeError("private-aiohttp-sentinel")
        if placement == "cause":
            wrapper.__cause__ = error
            expected = [
                _wire("unknown", 0, context_suppressed=True),
                _wire(
                    identity,
                    1,
                    "cause",
                    0,
                    status_code=expected_status,
                    retry_after=expected_retry,
                ),
            ]
        if placement == "suppressed":
            wrapper.__context__ = error
            wrapper.__suppress_context__ = True
            expected = [
                _wire("unknown", 0, context_suppressed=True),
                _wire(
                    identity,
                    1,
                    "context",
                    0,
                    status_code=expected_status,
                    retry_after=expected_retry,
                ),
            ]
        if placement == "cleanup":
            set_command_failure_facts(
                wrapper,
                CommandFailureFacts(
                    CommandOperation.READ,
                    CommandStream.STDOUT,
                    (CommandFailureEvent(error, CommandOperation.PROCESS_CLEANUP),),
                ),
            )
            expected = [
                _wire("unknown", 0, operation="read", stream="stdout"),
                _wire(
                    identity,
                    1,
                    "cleanup",
                    0,
                    status_code=expected_status,
                    retry_after=expected_retry,
                    operation="process_cleanup",
                ),
            ]
        error = wrapper
    result = fail_with(error)
    _assert_aiohttp_wire(result, {"code": code, "causes": expected}, caplog)
    assert output.exists() is False


@pytest.mark.parametrize(
    "status,code,attempts",
    [
        (403, "authentication", 2),
        (401, "authentication", 1),
        (429, "rate_limit", 1),
        (503, "provider_error", 1),
    ],
)
def test_edge_native_status_retry_reaches_safe_prepared_graph_offline(
    tmp_path, monkeypatch, caplog, status, code, attempts
):
    aiohttp = pytest.importorskip("aiohttp")
    edge_tts = pytest.importorskip("edge_tts")
    from edge_tts.drm import DRM
    from multidict import CIMultiDict, CIMultiDictProxy

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(json.dumps({"tts": {"provider": "edge"}}))
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: edge_tts)
    monkeypatch.setattr(DRM, "clock_skew_seconds", 0.0)
    monkeypatch.setattr(DRM, "get_unix_timestamp", lambda: 1609459200.0)
    observed = []
    headers = CIMultiDictProxy(
        CIMultiDict(
            {
                "Date": "Fri, 01 Jan 2021 00:00:00 GMT",
                "Retry-After": "3",
                "private-aiohttp-sentinel": "private-aiohttp-sentinel",
            }
        )
    )

    def reject_handshake(*args, **kwargs):
        observed.append(status)
        raise _aiohttp_error("WSServerHandshakeError", status, headers)

    monkeypatch.setattr(aiohttp.ClientSession, "ws_connect", reject_handshake)
    resolution = resolve_prepared_speech()
    output = tmp_path / "speech.mp3"
    with pytest.raises(SpeechSynthesisError) as caught:
        synthesize_prepared_speech(
            PreparedSpeechRequest("Approved speech.", str(output), resolution)
        )
    expected = [_wire("aiohttp_status", 0, status_code=status, retry_after=3.0)]
    if attempts == 2:
        expected.append(
            _wire(
                "aiohttp_status", 1, "context", 0, status_code=status, retry_after=3.0
            )
        )
    _assert_aiohttp_wire(caught.value, {"code": code, "causes": expected}, caplog)
    assert (observed, output.exists()) == ([status] * attempts, False)
