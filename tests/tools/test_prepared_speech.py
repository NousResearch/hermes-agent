"""Prepared native handoff and safe configured-speech invariants."""

from __future__ import annotations

import io
import json
import shlex
import sys
import subprocess
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

import httpx
import openai
import pytest

from agent import tts_registry
from agent.tts_provider import TTSProvider
from tools import tts_tool
from tools.tts_command_error_facts import CommandOperation
from tools.tts_prepared import resolve_prepared_speech, synthesize_prepared_speech
from tools.tts_prepared_types import (
    PreparedSpeechDeclaration,
    PreparedSpeechRequest,
    SpeechCause,
    SpeechCauseType,
    SpeechCauseRelation,
    SpeechControl,
    SpeechFact,
    SpeechFactSource,
    SpeechFailureCode,
    SpeechFormat,
    SpeechSettings,
    SpeechSynthesisError,
    SpeechSynthesisResult,
)


class _SpoofedStatusError(RuntimeError):
    status_code = 429
    retry_after = 3


class _TransportMode(TypedDict):
    status: int
    format: str | None
    retry: str


TEXT = "Exact  narration.\nThe cost is £5 & 25%.\n𐐀 pronunciation.  \n"
MP3 = b"ID3fixture-native-audio"


def _audio(format: str) -> bytes:
    if format == "mp3":
        return MP3
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00")
    return data.getvalue()


def _declaration() -> PreparedSpeechDeclaration:
    return PreparedSpeechDeclaration(
        "fixture-v1",
        ("fixture_implementation",),
        True,
        frozenset(SpeechFormat),
        frozenset(SpeechControl),
        4000,
        0.25,
        4.0,
    )


@pytest.fixture
def speech(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    tts_registry._registry.reset_for_tests()
    observed: list[dict[str, Any]] = []
    mode: _TransportMode = {"status": 200, "format": None, "retry": "3"}

    def write_config(config):
        (home / "config.yaml").write_text(json.dumps({"tts": config}), encoding="utf-8")

    def factory(**kwargs):
        def handle(request):
            body = json.loads(request.content)
            observed.append(body)
            if mode["status"] != 200:
                return httpx.Response(
                    mode["status"],
                    json={"error": {"message": "private-native-sentinel"}},
                    headers={"retry-after": mode["retry"]},
                )
            return httpx.Response(
                200,
                content=_audio(mode["format"] or body["response_format"]),
                headers={"content-type": "application/octet-stream"},
            )

        return openai.OpenAI(
            **kwargs,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: factory)
    yield home, observed, mode, write_config
    tts_registry._registry.reset_for_tests()


class _Plugin(TTSProvider):
    def __init__(self, observed):
        self.observed = observed

    @property
    def name(self):
        return "fixture-plugin"

    prepared_speech = _declaration()

    def synthesize(
        self,
        text,
        output_path,
        *,
        model=None,
        voice=None,
        speed=None,
        format="mp3",
        **extra,
    ):
        self.observed.append(
            {
                "input": text,
                "model": model,
                "voice": voice,
                "speed": speed,
                "response_format": format,
                **extra,
            }
        )
        Path(output_path).write_bytes(_audio(format))
        return output_path


def _openai_config():
    return {
        "provider": "openai",
        "openai": {"api_key": "fixture-only", "base_url": "https://speech.invalid/v1"},
    }


def _command_config(tmp_path):
    received = tmp_path / "command-input.bin"
    details = tmp_path / "command-details.json"
    code = (
        "import json,sys;from pathlib import Path;"
        f"Path({str(received)!r}).write_bytes(Path(sys.argv[1]).read_bytes());"
        f"Path({str(details)!r}).write_text(json.dumps({{'input':Path(sys.argv[1]).read_bytes().decode(),'model':sys.argv[3],'voice':sys.argv[4],'speed':float(sys.argv[5]),'response_format':sys.argv[6],'instructions':sys.argv[7]}}));"
        "Path(sys.argv[2]).write_bytes(b'ID3fixture-native-audio')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} {{input_path}} {{output_path}} {{model}} {{voice}} {{speed}} {{format}} {{instructions}}"
    declaration = {
        "revision": "fixture-v1",
        "identity_coverage": ["fixture_implementation"],
        "prepared_native_text": True,
        "formats": ["mp3"],
        "controls": [c.value for c in SpeechControl],
        "input_limit": 4000,
        "speed_min": 0.25,
        "speed_max": 4.0,
    }
    return (
        {
            "provider": "fixture-command",
            "providers": {
                "fixture-command": {
                    "type": "command",
                    "command": command,
                    "prepared": declaration,
                }
            },
        },
        received,
        details,
    )


@pytest.mark.parametrize(
    "route,format",
    [
        ("openai", "mp3"),
        ("openai", "wav"),
        ("command", "mp3"),
        ("plugin", "mp3"),
        ("plugin", "wav"),
        ("edge", "mp3"),
        ("profiles", "mp3"),
        ("credential_rotation", "mp3"),
        ("cap_boundary", "mp3"),
        ("unknown_controls", "mp3"),
        ("legacy_error_label", "mp3"),
    ],
)
def test_prepared_handoff_preserves_text_controls_and_observed_output(
    speech, tmp_path, monkeypatch, route, format
):
    home, observed, _, write_config = speech
    if route == "legacy_error_label":
        monkeypatch.setattr(
            tts_tool, "_dispatch_to_plugin_provider", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            tts_tool, "_select_builtin_engine", lambda provider: ("neutts", None)
        )

        def fail(*args, **kwargs):
            raise RuntimeError("fixture failure")

        monkeypatch.setattr(tts_tool, "_synthesize_builtin", fail)
        result = tts_tool._text_to_speech_single(
            TEXT,
            str(tmp_path / "legacy.mp3"),
            provider="unknown-provider",
            tts_config={},
            command_provider_config=None,
            want_opus=False,
            instructions=None,
        )
        assert json.loads(result) == {
            "error": "TTS generation failed (neutts): fixture failure",
            "success": False,
        }
        return
    if route == "profiles":
        from agent import secret_scope
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        homes = [home, tmp_path / "second-profile", home]
        homes[1].mkdir()
        expected_calls = []
        actual_results = []
        expected_results = []
        multiplex = secret_scope.set_multiplex_context(True)
        try:
            for selected in homes:
                token = set_hermes_home_override(str(selected))
                secrets = secret_scope.set_secret_scope(
                    {"OPENAI_API_KEY": "fixture-only"}, profile_home=str(selected)
                )
                try:
                    model, voice = selected.name + "-model", selected.name + "-voice"
                    config = _openai_config()
                    config["openai"].update(model=model, voice=voice, speed=1.2)
                    (selected / "config.yaml").write_text(
                        json.dumps({"tts": config}), encoding="utf-8"
                    )
                    resolution = resolve_prepared_speech(SpeechSettings())
                    output = selected / "cache" / "audio" / "prepared.mp3"
                    actual_results.append(
                        synthesize_prepared_speech(
                            PreparedSpeechRequest(TEXT, str(output), resolution)
                        )
                    )
                    expected_results.append(
                        SpeechSynthesisResult(
                            str(output),
                            SpeechFormat.MP3,
                            SpeechFact("openai", SpeechFactSource.OBSERVED),
                            SpeechFact(model, SpeechFactSource.CONFIGURED),
                            SpeechFact(voice, SpeechFactSource.CONFIGURED),
                            SpeechFact(1.2, SpeechFactSource.CONFIGURED),
                        )
                    )
                    expected_calls.append(
                        {
                            "input": TEXT,
                            "model": model,
                            "voice": voice,
                            "speed": 1.2,
                            "response_format": "mp3",
                        }
                    )
                finally:
                    secret_scope.reset_secret_scope(secrets)
                    reset_hermes_home_override(token)
        finally:
            secret_scope.reset_multiplex_context(multiplex)
        assert (actual_results, observed) == (expected_results, expected_calls)
        assert [
            selected.joinpath("cache/audio/prepared.mp3").read_bytes()
            for selected in homes
        ] == [MP3] * 3
        return
    if route == "unknown_controls":
        tts_registry._registry.register(_Plugin(observed))
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda **kwargs: None
        )
        write_config({"provider": "fixture-plugin"})
        resolution = resolve_prepared_speech(SpeechSettings())
        output = tmp_path / "unknown.mp3"
        result = synthesize_prepared_speech(
            PreparedSpeechRequest(TEXT, str(output), resolution)
        )
        unknown = SpeechFact(None, SpeechFactSource.UNKNOWN)
        assert (result, observed) == (
            SpeechSynthesisResult(
                str(output),
                SpeechFormat.MP3,
                SpeechFact("fixture-plugin", SpeechFactSource.OBSERVED),
                unknown,
                unknown,
                unknown,
            ),
            [
                {
                    "input": TEXT,
                    "model": None,
                    "voice": None,
                    "speed": None,
                    "response_format": "mp3",
                }
            ],
        )
        return
    settings = SpeechSettings(
        SpeechFormat(format), "fixture-model", "fixture-voice", 1.2, "calm"
    )
    command_files = None

    def command():
        nonlocal command_files
        config, received, details = _command_config(tmp_path)
        command_files = received, details
        return config

    def plugin():
        tts_registry._registry.register(_Plugin(observed))
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda **kwargs: None
        )
        return {"provider": "fixture-plugin"}

    def edge():
        async def generate(text, output_path, config):
            observed.append(
                {
                    "input": text,
                    "voice": config["edge"]["voice"],
                    "speed": config["edge"]["speed"],
                    "response_format": "mp3",
                }
            )
            Path(output_path).write_bytes(MP3)
            return output_path

        monkeypatch.setattr(tts_tool, "_generate_edge_tts", generate)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object)
        return {"provider": "edge"}

    config = {
        "credential_rotation": _openai_config,
        "cap_boundary": _openai_config,
        "openai": _openai_config,
        "command": command,
        "plugin": plugin,
        "edge": edge,
    }[route]()
    if route == "edge":
        settings = SpeechSettings(SpeechFormat.MP3, voice="fixture-voice", speed=1.2)
    if route in {"openai", "credential_rotation", "cap_boundary"}:
        settings = replace(settings, model="gpt-4o-mini-tts")
        config["openai"]["base_url"] = "https://api.openai.com/v1"
    write_config(config)
    resolution = resolve_prepared_speech(settings)
    if route == "credential_rotation":
        config["openai"]["api_key"] = "rotated-fixture-only"
        write_config(config)
    text = "𐐀" * 4096 if route == "cap_boundary" else TEXT
    output = tmp_path / ("result." + format)
    result = synthesize_prepared_speech(
        PreparedSpeechRequest(text, str(output), resolution)
    )
    if command_files is not None:
        received, details = command_files
        observed.extend([json.loads(details.read_text())])
        assert received.read_bytes() == TEXT.encode("utf-8")
    expected_call = {
        "input": text,
        "voice": "fixture-voice",
        "speed": 1.2,
        "response_format": format,
    }
    if route != "edge":
        expected_call.update(model=settings.model, instructions="calm")
    assert observed == [expected_call]
    source = SpeechFactSource.CONFIGURED
    expected_engine = {
        "credential_rotation": "openai",
        "cap_boundary": "openai",
        "openai": "openai",
        "command": "fixture-command",
        "plugin": "fixture-plugin",
        "edge": "edge",
    }[route]
    assert result == SpeechSynthesisResult(
        str(output),
        SpeechFormat(format),
        SpeechFact(expected_engine, SpeechFactSource.OBSERVED),
        SpeechFact(None, SpeechFactSource.INAPPLICABLE)
        if route == "edge"
        else SpeechFact(settings.model, source),
        SpeechFact("fixture-voice", source),
        SpeechFact(1.2, source),
    )
    assert output.read_bytes() == _audio(format)


def _native_failure_fixture(
    case: str, monkeypatch
) -> tuple[SpeechFailureCode, tuple[SpeechCause, ...]]:
    native_request = httpx.Request(
        "POST",
        "https://speech.invalid/private-native-sentinel",
        headers={"authorization": "private-native-sentinel"},
        content=b"private-native-sentinel",
    )
    errors = {
        "command_exit": subprocess.CalledProcessError(
            7,
            "private-native-sentinel",
            output="private-native-sentinel",
            stderr="private-native-sentinel",
        ),
        "command_timeout": subprocess.TimeoutExpired(
            "private-native-sentinel",
            1,
            output="private-native-sentinel",
            stderr="private-native-sentinel",
        ),
        "nested_timeout": RuntimeError("private-native-sentinel"),
        "httpx_timeout": httpx.ReadTimeout(
            "private-native-sentinel", request=native_request
        ),
        "spoofed_status": _SpoofedStatusError("429 retry in3s private-native-sentinel"),
    }
    original = errors[case]
    if case == "nested_timeout":
        original.__cause__ = openai.APITimeoutError(request=native_request)
        original.__cause__.__cause__ = errors["httpx_timeout"]

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fail)
    expected_causes = {
        "command_exit": (
            SpeechCause(
                SpeechCauseType.COMMAND_EXIT,
                exit_code=7,
                operation=CommandOperation.NATIVE_EXIT,
            ),
        ),
        "command_timeout": (
            SpeechCause(
                SpeechCauseType.COMMAND_TIMEOUT,
                operation=CommandOperation.NATIVE_TIMEOUT,
            ),
        ),
        "httpx_timeout": (SpeechCause(SpeechCauseType.HTTPX_TIMEOUT),),
        "nested_timeout": (
            SpeechCause(SpeechCauseType.UNKNOWN, context_suppressed=True),
            SpeechCause(
                SpeechCauseType.OPENAI_TIMEOUT,
                SpeechCauseRelation.CAUSE,
                context_suppressed=True,
            ),
            SpeechCause(SpeechCauseType.HTTPX_TIMEOUT, SpeechCauseRelation.CAUSE),
        ),
        "spoofed_status": (SpeechCause(SpeechCauseType.UNKNOWN),),
    }
    code = (
        SpeechFailureCode.TIMEOUT
        if case in {"command_timeout", "nested_timeout", "httpx_timeout"}
        else SpeechFailureCode.PROVIDER_ERROR
    )
    return code, expected_causes[case]


def _dispatch_failure_fixture(
    case, speech, tmp_path, monkeypatch, settings, resolution
):
    _, observed, _, write_config = speech

    def command_exit():
        nonlocal resolution
        causes = ()
        config, _, _ = _command_config(tmp_path)
        config["providers"]["fixture-command"]["command"] = (
            f"{shlex.quote(sys.executable)} -c {shlex.quote('import sys;sys.exit(7)')} {{input_path}} {{output_path}} {{model}} {{voice}} {{speed}} {{format}} {{instructions}}"
        )
        write_config(config)
        resolution = resolve_prepared_speech(settings)
        code = SpeechFailureCode.PROVIDER_ERROR
        causes = (
            SpeechCause(SpeechCauseType.UNKNOWN, context_suppressed=True, node_index=0),
            SpeechCause(
                SpeechCauseType.COMMAND_EXIT,
                SpeechCauseRelation.CAUSE,
                exit_code=7,
                operation=CommandOperation.NATIVE_EXIT,
                node_index=1,
                parent_index=0,
            ),
            SpeechCause(
                SpeechCauseType.COMMAND_EXIT,
                SpeechCauseRelation.CONTEXT,
                exit_code=7,
                operation=CommandOperation.NATIVE_EXIT,
                node_index=1,
                parent_index=0,
            ),
        )
        return resolution, code, causes

    def plugin_discovery():
        nonlocal resolution
        causes = ()
        tts_registry._registry.register(_Plugin(observed))
        write_config({"provider": "fixture-plugin"})
        resolution = resolve_prepared_speech(settings)

        def discover(**kwargs):
            raise TimeoutError("private-native-sentinel")

        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", discover)
        code, causes = (
            SpeechFailureCode.TIMEOUT,
            (SpeechCause(SpeechCauseType.TIMEOUT),),
        )
        return resolution, code, causes

    def managed_coercion():
        nonlocal resolution
        causes = ()
        from tools import tts_tool_openai

        write_config({"provider": "nous"})
        resolution = resolve_prepared_speech(SpeechSettings(model="tts-1-hd"))
        monkeypatch.setattr(
            tts_tool_openai,
            "_resolve_openai_audio_client_config",
            lambda **kwargs: ("fixture-only", "https://speech.invalid/v1", True),
        )
        code = SpeechFailureCode.UNSUPPORTED_CONTROL
        return resolution, code, causes

    def output():
        nonlocal resolution
        causes = ()
        external = tmp_path / "external.mp3"
        external.write_bytes(MP3)

        def generate(text, output_path, *args, **kwargs):
            if case == "symlink":
                Path(output_path).symlink_to(external)
            elif case == "outside_path":
                Path(output_path).write_bytes(MP3)
                return str(external)
            else:
                Path(output_path).write_bytes(b"not-an-audio-container")
            return output_path

        if case == "outside_path":
            provider = _Plugin(observed)
            monkeypatch.setattr(provider, "synthesize", generate)
            tts_registry._registry.register(provider)
            monkeypatch.setattr(
                "hermes_cli.plugins._ensure_plugins_discovered",
                lambda **kwargs: None,
            )
            write_config({"provider": "fixture-plugin"})
            resolution = resolve_prepared_speech(settings)
        else:
            monkeypatch.setattr(tts_tool, "_generate_openai_tts", generate)
        code = (
            SpeechFailureCode.OUTPUT_FORMAT
            if case == "unsupported_container"
            else SpeechFailureCode.OUTPUT_PATH
        )
        return resolution, code, causes

    handlers = {
        "native_command_exit": command_exit,
        "plugin_discovery": plugin_discovery,
        "managed_coercion": managed_coercion,
    }
    return handlers.get(case, output)()


class _RejectionFixture:
    def __init__(self, speech, tmp_path, monkeypatch):
        self.speech = speech
        _, self.observed, self.mode, self.write_config = speech
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self.settings = SpeechSettings(
            model="fixture-model", voice="fixture-voice", speed=1.2
        )
        self.write_config(_openai_config())
        self.resolution = resolve_prepared_speech(self.settings)
        self.output = tmp_path / "not-created" / "result.mp3"
        self.text = TEXT
        self.code = SpeechFailureCode.PROVIDER_ERROR
        self.causes: tuple[SpeechCause, ...] = ()
        self.resolve_only = False

    def run(self):
        if self.resolve_only:
            return resolve_prepared_speech(self.settings)
        return synthesize_prepared_speech(
            PreparedSpeechRequest(self.text, str(self.output), self.resolution)
        )

    def fail_with(self, error):
        def fail(*args, **kwargs):
            raise error

        self.monkeypatch.setattr(tts_tool, "_generate_openai_tts", fail)

    def simple_failure(self, case):
        native = {
            "timeout": TimeoutError("private-native-sentinel"),
            "unknown": RuntimeError("429 retry in3s private-native-sentinel"),
        }
        self.fail_with(native[case])
        self.code = (
            SpeechFailureCode.TIMEOUT
            if case == "timeout"
            else SpeechFailureCode.PROVIDER_ERROR
        )
        self.causes = (
            SpeechCause(
                SpeechCauseType.TIMEOUT
                if case == "timeout"
                else SpeechCauseType.UNKNOWN
            ),
        )

    def status(self, case):
        identity, status = {
            "rate": (SpeechCauseType.OPENAI_RATE_LIMIT, 429),
            "auth": (SpeechCauseType.OPENAI_AUTHENTICATION, 401),
            "server": (SpeechCauseType.OPENAI_SERVER, 503),
        }[case]
        self.mode["status"] = status
        self.code = {
            "rate": SpeechFailureCode.RATE_LIMIT,
            "auth": SpeechFailureCode.AUTHENTICATION,
            "server": SpeechFailureCode.PROVIDER_ERROR,
        }[case]
        self.causes = (
            SpeechCause(
                identity,
                status_code=status,
                retry_after=3.0,
                context_suppressed=True,
                node_index=0,
            ),
            SpeechCause(
                SpeechCauseType.HTTPX_STATUS,
                SpeechCauseRelation.CONTEXT,
                status_code=status,
                retry_after=3.0,
                node_index=1,
                parent_index=0,
            ),
        )

    def retry(self, case):
        self.mode.update(
            status=429,
            retry={"retry_negative": "-1", "retry_nonfinite": "NaN", "retry_zero": "0"}[
                case
            ],
        )
        self.code = SpeechFailureCode.RATE_LIMIT
        self.causes = (
            SpeechCause(
                SpeechCauseType.OPENAI_RATE_LIMIT,
                status_code=429,
                retry_after=0.0 if case == "retry_zero" else None,
                context_suppressed=True,
                node_index=0,
            ),
            SpeechCause(
                SpeechCauseType.HTTPX_STATUS,
                SpeechCauseRelation.CONTEXT,
                status_code=429,
                retry_after=0.0 if case == "retry_zero" else None,
                node_index=1,
                parent_index=0,
            ),
        )

    def command_declaration(self, case):
        config, _, _ = _command_config(self.tmp_path)
        section = config["providers"]["fixture-command"]

        def missing(field):
            section["command"] = section["command"].replace("{" + field + "}", "unused")

        def unknown_range():
            section["prepared"].pop("speed_min")
            section["prepared"].pop("speed_max")

        {
            "missing_input": lambda: missing("input_path"),
            "missing_output": lambda: missing("output_path"),
            "unknown_range": unknown_range,
            "missing_declaration": lambda: section.pop("prepared"),
        }[case]()
        self.write_config(config)
        self.resolve_only = True
        self.code = (
            SpeechFailureCode.UNKNOWN_CAPABILITY
            if case == "missing_declaration"
            else SpeechFailureCode.UNSUPPORTED_CONTROL
        )

    def settings_case(self, case):
        config, self.settings = {
            "gemini": ({"provider": "gemini"}, SpeechSettings()),
            "ignored_speed": ({"provider": "elevenlabs"}, SpeechSettings(speed=1.2)),
            "clamped_speed": (_openai_config(), SpeechSettings(speed=4.1)),
            "configured_clamp": (
                {"provider": "openai", "openai": {"speed": 4.1}},
                SpeechSettings(),
            ),
        }[case]
        self.write_config(config)
        self.resolve_only = True
        self.code = (
            SpeechFailureCode.UNKNOWN_CAPABILITY
            if case == "gemini"
            else SpeechFailureCode.UNSUPPORTED_CONTROL
        )

    def forge(self, case):
        replacements = {
            "forged": {
                "input_limit": SpeechFact(999999, SpeechFactSource.NATIVE_POLICY)
            },
            "forged_provenance": {
                "model": replace(
                    self.resolution.model, source=SpeechFactSource.OBSERVED
                )
            },
            "forged_controls": {"controls": frozenset()},
        }
        self.resolution = replace(self.resolution, **replacements[case])
        self.code = SpeechFailureCode.RESOLUTION_CHANGED

    def drift(self, case):
        config = _openai_config()
        config["openai"]["base_url"] = "https://other.invalid/v1"
        self.write_config(config)
        self.code = SpeechFailureCode.RESOLUTION_CHANGED

    def format(self, case):
        self.mode["format"] = "wav"
        self.code = SpeechFailureCode.OUTPUT_FORMAT

    def limit(self, case):
        self.text = "x" * 4097
        self.code = SpeechFailureCode.INPUT_LIMIT

    def dispatch(self, case):
        self.resolution, self.code, self.causes = _dispatch_failure_fixture(
            case,
            self.speech,
            self.tmp_path,
            self.monkeypatch,
            self.settings,
            self.resolution,
        )

    def native_failure(self, case):
        self.code, self.causes = _native_failure_fixture(case, self.monkeypatch)

    def connection(self, case):
        def factory(**kwargs):
            def handle(request):
                raise httpx.ConnectError("private-native-sentinel", request=request)

            return openai.OpenAI(
                **kwargs,
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle)),
            )

        self.monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: factory)
        self.causes = (
            SpeechCause(
                SpeechCauseType.OPENAI_CONNECTION, context_suppressed=True, node_index=0
            ),
            SpeechCause(
                SpeechCauseType.HTTPX_NETWORK,
                SpeechCauseRelation.CAUSE,
                node_index=1,
                parent_index=0,
            ),
            SpeechCause(
                SpeechCauseType.HTTPX_NETWORK,
                SpeechCauseRelation.CONTEXT,
                node_index=1,
                parent_index=0,
            ),
        )

    def active_declaration(self, case):
        observed = self.observed

        class ActivePlugin(_Plugin):
            @property
            def prepared_speech(self):
                observed.append({"declaration": "executed"})
                return _declaration()

        tts_registry._registry.register(ActivePlugin(observed))
        self.write_config({"provider": "fixture-plugin"})
        self.resolve_only = True
        self.code = SpeechFailureCode.UNKNOWN_CAPABILITY

    def instruction_model(self, case):
        model = {
            "instructions_tts1": "tts-1",
            "instructions_hd": "tts-1-hd",
            "instructions_custom": "gpt-4o-mini-tts",
        }[case]
        config = _openai_config()
        config["openai"]["base_url"] = (
            "https://speech.invalid/v1"
            if case == "instructions_custom"
            else "https://api.openai.com/v1"
        )
        self.write_config(config)
        self.settings = SpeechSettings(
            model=model, voice="fixture-voice", instructions="calm"
        )
        self.code = SpeechFailureCode.UNSUPPORTED_CONTROL
        self.resolution = resolve_prepared_speech(self.settings)

    def instructions_late_route(self, case):
        from tools import tts_tool_openai

        config = _openai_config()
        config["openai"]["model"] = "gpt-4o-mini-tts"
        config["openai"].pop("base_url")
        self.write_config(config)
        self.settings = SpeechSettings(instructions="calm")
        self.resolution = resolve_prepared_speech(self.settings)
        self.monkeypatch.setattr(
            tts_tool_openai,
            "_resolve_openai_audio_client_config",
            lambda **kwargs: ("fixture-only", "https://speech.invalid/v1", True),
        )
        self.code = SpeechFailureCode.UNSUPPORTED_CONTROL

    def edge_precision(self, case):
        import asyncio

        control = []

        class Edge:
            class Communicate:
                def __init__(self, text, **kwargs):
                    control.append({"input": text, **kwargs})

                async def save(self, output_path):
                    Path(output_path).write_bytes(MP3)

        self.monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: Edge)
        native_speed = 1.20000000001
        asyncio.run(
            tts_tool._generate_edge_tts(
                TEXT,
                str(self.tmp_path / "edge-control.mp3"),
                {"edge": {"voice": "fixture-voice", "speed": native_speed}},
            )
        )
        assert control == [{"input": TEXT, "voice": "fixture-voice", "rate": "+20%"}]
        self.write_config({"provider": "edge"})
        self.settings = SpeechSettings(voice="fixture-voice", speed=native_speed)
        self.resolve_only = True
        self.code = SpeechFailureCode.UNSUPPORTED_CONTROL

    def edge_fallback(self, case):
        self.write_config({"provider": "edge"})
        self.settings = SpeechSettings(voice="fixture-voice", speed=1.2)
        self.resolution = resolve_prepared_speech(self.settings)
        self.monkeypatch.setattr(tts_tool, "_importable", lambda importer: False)
        self.monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: True)

        def generate(engine, text, output_path, *args):
            self.observed.append({"engine": engine, "input": text})
            Path(output_path).write_bytes(MP3)

        self.monkeypatch.setattr(tts_tool, "_synthesize_builtin", generate)
        self.code = SpeechFailureCode.RESOLUTION_CHANGED

    def incoming_error(self, case):
        cause = SpeechCause(SpeechCauseType.UNKNOWN)
        object.__setattr__(cause, "relation", "private-native-sentinel")
        object.__setattr__(cause, "status_code", True)
        object.__setattr__(
            cause,
            "request",
            httpx.Request("GET", "https://speech.invalid/private-native-sentinel"),
        )
        error = SpeechSynthesisError(SpeechFailureCode.PROVIDER_ERROR)
        error.causes = (cause,)
        self.fail_with(error)
        self.causes = (SpeechCause(SpeechCauseType.UNKNOWN),)


_REJECTION_SETUPS = {
    **dict.fromkeys(("timeout", "unknown"), _RejectionFixture.simple_failure),
    **dict.fromkeys(("rate", "auth", "server"), _RejectionFixture.status),
    **dict.fromkeys(
        ("retry_negative", "retry_nonfinite", "retry_zero"), _RejectionFixture.retry
    ),
    **dict.fromkeys(
        ("missing_input", "missing_output", "unknown_range", "missing_declaration"),
        _RejectionFixture.command_declaration,
    ),
    **dict.fromkeys(
        ("gemini", "ignored_speed", "clamped_speed", "configured_clamp"),
        _RejectionFixture.settings_case,
    ),
    **dict.fromkeys(
        ("forged", "forged_provenance", "forged_controls"), _RejectionFixture.forge
    ),
    **dict.fromkeys(
        (
            "native_command_exit",
            "plugin_discovery",
            "managed_coercion",
            "outside_path",
            "symlink",
            "unsupported_container",
        ),
        _RejectionFixture.dispatch,
    ),
    **dict.fromkeys(
        (
            "command_exit",
            "command_timeout",
            "nested_timeout",
            "httpx_timeout",
            "spoofed_status",
        ),
        _RejectionFixture.native_failure,
    ),
    **dict.fromkeys(
        ("instructions_tts1", "instructions_hd", "instructions_custom"),
        _RejectionFixture.instruction_model,
    ),
    "drift": _RejectionFixture.drift,
    "format": _RejectionFixture.format,
    "limit": _RejectionFixture.limit,
    "connection": _RejectionFixture.connection,
    "active_declaration": _RejectionFixture.active_declaration,
    "instructions_late_route": _RejectionFixture.instructions_late_route,
    "edge_precision": _RejectionFixture.edge_precision,
    "edge_fallback": _RejectionFixture.edge_fallback,
    "incoming_error": _RejectionFixture.incoming_error,
}


@pytest.mark.parametrize("case", tuple(_REJECTION_SETUPS))
def test_prepared_rejects_unfaithful_requests_and_exports_safe_native_causes(
    speech, tmp_path, monkeypatch, caplog, case
):
    fixture = _RejectionFixture(speech, tmp_path, monkeypatch)
    with pytest.raises(SpeechSynthesisError) as caught:
        _REJECTION_SETUPS[case](fixture, case)
        fixture.run()
    error = caught.value
    expected = SpeechSynthesisError(fixture.code, fixture.causes)
    assert error.to_dict() == expected.to_dict()
    assert (
        error.args,
        error.__cause__,
        error.__context__,
        error.__suppress_context__,
    ) == ((fixture.code.value,), None, None, False)
    assert "private-native-sentinel" not in json.dumps(error.to_dict())
    records = [
        record for record in caplog.records if record.name == "tools.tts_prepared"
    ]
    assert [
        (record.exc_info[0], record.exc_info[2])
        for record in records
        if record.exc_info is not None
    ] == [(SpeechSynthesisError, None)]
    assert not fixture.output.exists()
    passive = {
        "forged",
        "forged_provenance",
        "forged_controls",
        "drift",
        "limit",
        "gemini",
        "ignored_speed",
        "clamped_speed",
        "configured_clamp",
        "missing_input",
        "missing_output",
        "unknown_range",
        "missing_declaration",
        "active_declaration",
        "instructions_tts1",
        "instructions_hd",
        "instructions_custom",
        "edge_precision",
    }
    if case in passive:
        assert fixture.observed == []
        assert not fixture.output.parent.exists()
    if case == "edge_fallback":
        assert fixture.observed == []
