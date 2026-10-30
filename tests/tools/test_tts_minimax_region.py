"""MiniMax TTS region, endpoint, and credential selection tests."""

from unittest.mock import MagicMock, patch

import pytest

from tools.tts_tool import (
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_CN_BASE_URL,
    DEFAULT_MINIMAX_CN_VOICE_DESIGN_URL,
    DEFAULT_MINIMAX_FILE_UPLOAD_URL,
    DEFAULT_MINIMAX_VOICE_CLONE_MODEL,
    DEFAULT_MINIMAX_VOICE_CLONE_URL,
    _generate_minimax_tts,
    _clone_minimax_voice,
    _design_minimax_voice,
    _resolve_minimax_tts_runtime,
    _upload_minimax_voice_audio,
    check_tts_requirements,
    minimax_voice_clone_tool,
    minimax_voice_design_tool,
    minimax_voice_upload_tool,
)


GLOBAL_CREDENTIAL_SENTINEL = "FAKE_GLOBAL_CREDENTIAL"
CN_CREDENTIAL_SENTINEL = "FAKE_CN_CREDENTIAL"


@pytest.fixture(autouse=True)
def _fake_minimax_credentials(monkeypatch):
    values = {}
    monkeypatch.setattr(
        "tools.tts_tool.get_env_value",
        lambda name, default=None: values.get(name, default),
    )
    return values


@pytest.mark.parametrize(
    ("config", "credentials", "expected"),
    [
        pytest.param(
            {},
            {"MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL},
            (
                "global",
                DEFAULT_MINIMAX_BASE_URL,
                "MINIMAX_API_KEY",
                GLOBAL_CREDENTIAL_SENTINEL,
            ),
            id="global-only",
        ),
        pytest.param(
            {},
            {"MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL},
            (
                "cn",
                DEFAULT_MINIMAX_CN_BASE_URL,
                "MINIMAX_CN_API_KEY",
                CN_CREDENTIAL_SENTINEL,
            ),
            id="china-only",
        ),
        pytest.param(
            {},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            (
                "global",
                DEFAULT_MINIMAX_BASE_URL,
                "MINIMAX_API_KEY",
                GLOBAL_CREDENTIAL_SENTINEL,
            ),
            id="both-default-to-global",
        ),
        pytest.param(
            {"minimax": {"region": "global"}},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            (
                "global",
                DEFAULT_MINIMAX_BASE_URL,
                "MINIMAX_API_KEY",
                GLOBAL_CREDENTIAL_SENTINEL,
            ),
            id="explicit-global",
        ),
        pytest.param(
            {"minimax": {"region": "cn"}},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            (
                "cn",
                DEFAULT_MINIMAX_CN_BASE_URL,
                "MINIMAX_CN_API_KEY",
                CN_CREDENTIAL_SENTINEL,
            ),
            id="explicit-china",
        ),
    ],
)
def test_runtime_selection_matrix(
    _fake_minimax_credentials,
    config,
    credentials,
    expected,
):
    _fake_minimax_credentials.update(credentials)

    runtime = _resolve_minimax_tts_runtime(config)

    assert (
        runtime.region,
        runtime.endpoint,
        runtime.credential_source,
        runtime.api_key,
    ) == expected


@pytest.mark.parametrize(
    ("region", "credentials", "missing_source"),
    [
        pytest.param(
            "global",
            {"MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL},
            "MINIMAX_API_KEY",
            id="global-does-not-borrow-china-key",
        ),
        pytest.param(
            "cn",
            {"MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL},
            "MINIMAX_CN_API_KEY",
            id="china-does-not-borrow-global-key",
        ),
    ],
)
def test_explicit_region_requires_matching_credential(
    _fake_minimax_credentials,
    region,
    credentials,
    missing_source,
):
    _fake_minimax_credentials.update(credentials)

    with pytest.raises(ValueError, match=missing_source):
        _resolve_minimax_tts_runtime({"minimax": {"region": region}})


@pytest.mark.parametrize(
    ("config", "credentials", "expected"),
    [
        pytest.param(
            {"provider": "minimax"},
            {"MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL},
            True,
            id="china-only-available",
        ),
        pytest.param(
            {"provider": "minimax", "minimax": {"region": "cn"}},
            {"MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL},
            False,
            id="selected-region-missing",
        ),
        pytest.param(
            {"provider": "minimax", "minimax": {"region": "invalid"}},
            {
                "MINIMAX_API_KEY": GLOBAL_CREDENTIAL_SENTINEL,
                "MINIMAX_CN_API_KEY": CN_CREDENTIAL_SENTINEL,
            },
            False,
            id="invalid-region",
        ),
    ],
)
def test_availability_uses_atomic_runtime(
    monkeypatch,
    _fake_minimax_credentials,
    config,
    credentials,
    expected,
):
    _fake_minimax_credentials.update(credentials)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)

    assert check_tts_requirements() is expected


def test_runtime_repr_excludes_raw_credential(_fake_minimax_credentials):
    _fake_minimax_credentials["MINIMAX_API_KEY"] = GLOBAL_CREDENTIAL_SENTINEL

    runtime = _resolve_minimax_tts_runtime({})

    assert GLOBAL_CREDENTIAL_SENTINEL not in repr(runtime)


def test_voice_clone_uploads_audio_then_clones(
    tmp_path,
    _fake_minimax_credentials,
):
    _fake_minimax_credentials["MINIMAX_API_KEY"] = GLOBAL_CREDENTIAL_SENTINEL
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFF....WAVE")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url == DEFAULT_MINIMAX_FILE_UPLOAD_URL:
            response.json.return_value = {
                "file": {"file_id": "file-123"},
                "base_resp": {"status_code": 0},
            }
        else:
            response.json.return_value = {
                "voice_id": "voice-created",
                "base_resp": {"status_code": 0},
            }
        return response

    with patch("requests.post", side_effect=fake_post):
        file_id = _upload_minimax_voice_audio(
            str(sample),
            purpose="voice_clone",
            tts_config={},
        )
        result = _clone_minimax_voice(
            file_id=file_id,
            voice_id="voice-created",
            model=None,
            tts_config={},
        )

    assert file_id == "file-123"
    assert result == {
        "voice_id": "voice-created",
        "model": DEFAULT_MINIMAX_VOICE_CLONE_MODEL,
    }
    assert calls[0][0] == DEFAULT_MINIMAX_FILE_UPLOAD_URL
    assert calls[0][1]["data"] == {"purpose": "voice_clone"}
    assert calls[0][1]["headers"]["Authorization"] == f"Bearer {GLOBAL_CREDENTIAL_SENTINEL}"
    assert calls[1][0] == DEFAULT_MINIMAX_VOICE_CLONE_URL
    assert calls[1][1]["json"] == {
        "file_id": "file-123",
        "voice_id": "voice-created",
        "model": DEFAULT_MINIMAX_VOICE_CLONE_MODEL,
    }


def test_voice_design_uses_china_region_and_generated_voice_id(
    _fake_minimax_credentials,
):
    _fake_minimax_credentials["MINIMAX_CN_API_KEY"] = CN_CREDENTIAL_SENTINEL
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "data": {"voice_id": "designed-voice"},
            "base_resp": {"status_code": 0},
        }
        return response

    with patch("requests.post", side_effect=fake_post):
        result = _design_minimax_voice(
            prompt="warm narrator",
            preview_text="Welcome to the evening news.",
            voice_id=None,
            tts_config={},
        )

    assert result == {"voice_id": "designed-voice"}
    assert captured["url"] == DEFAULT_MINIMAX_CN_VOICE_DESIGN_URL
    assert captured["kwargs"]["json"] == {
        "prompt": "warm narrator",
        "preview_text": "Welcome to the evening news.",
    }
    assert captured["kwargs"]["headers"]["Authorization"] == f"Bearer {CN_CREDENTIAL_SENTINEL}"


def test_voice_clone_rejects_unknown_model(_fake_minimax_credentials):
    _fake_minimax_credentials["MINIMAX_API_KEY"] = GLOBAL_CREDENTIAL_SENTINEL

    with pytest.raises(ValueError, match="speech-2.8-hd"):
        _clone_minimax_voice(
            file_id="file-123",
            voice_id="voice-created",
            model="unknown-model",
            tts_config={},
        )


def test_voice_tool_wrappers_return_json(tmp_path, monkeypatch, _fake_minimax_credentials):
    _fake_minimax_credentials["MINIMAX_API_KEY"] = GLOBAL_CREDENTIAL_SENTINEL
    sample = tmp_path / "sample.mp3"
    sample.write_bytes(b"audio")

    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr(
        "tools.tts_tool._upload_minimax_voice_audio",
        lambda *_args, **_kwargs: "file-123",
    )
    monkeypatch.setattr(
        "tools.tts_tool._clone_minimax_voice",
        lambda **_kwargs: {
            "voice_id": "voice-created",
            "model": DEFAULT_MINIMAX_VOICE_CLONE_MODEL,
        },
    )
    monkeypatch.setattr(
        "tools.tts_tool._design_minimax_voice",
        lambda **_kwargs: {"voice_id": "designed-voice"},
    )

    clone_result = minimax_voice_clone_tool(str(sample), "voice-created")
    design_result = minimax_voice_design_tool(
        "warm narrator",
        "Welcome to the evening news.",
        "designed-voice",
    )

    assert '"success": true' in clone_result
    assert '"file_id": "file-123"' in clone_result
    assert '"voice_id": "designed-voice"' in design_result


def test_voice_upload_tool_supports_prompt_audio(
    tmp_path,
    monkeypatch,
    _fake_minimax_credentials,
):
    _fake_minimax_credentials["MINIMAX_CN_API_KEY"] = CN_CREDENTIAL_SENTINEL
    sample = tmp_path / "prompt.m4a"
    sample.write_bytes(b"audio")
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr(
        "tools.tts_tool._upload_minimax_voice_audio",
        lambda audio_path, purpose, tts_config: f"{purpose}-file",
    )

    result = minimax_voice_upload_tool(str(sample), "prompt_audio")

    assert '"success": true' in result
    assert '"file_id": "prompt_audio-file"' in result
