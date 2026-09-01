from unittest.mock import MagicMock, patch


def test_deepgram_is_a_registered_builtin():
    from agent.transcription_registry import _BUILTIN_NAMES
    from tools.transcription_tools import BUILTIN_STT_PROVIDERS

    assert "deepgram" in BUILTIN_STT_PROVIDERS
    assert "deepgram" in _BUILTIN_NAMES


def test_transcribe_audio_dispatches_explicit_deepgram(tmp_path, monkeypatch):
    from tools import transcription_tools

    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(
        transcription_tools,
        "_load_stt_config",
        lambda: {
            "provider": "deepgram",
            "deepgram": {"model": "nova-3", "language": "ru"},
        },
    )
    monkeypatch.setattr(
        transcription_tools,
        "get_env_value",
        lambda name, default=None: "dg-test" if name == "DEEPGRAM_API_KEY" else default,
    )

    with patch(
        "tools.transcription_tools._transcribe_deepgram",
        return_value={"success": True, "transcript": "привет", "provider": "deepgram"},
        create=True,
    ) as transcribe_deepgram:
        result = transcription_tools.transcribe_audio(str(audio_path), source="gateway")

    assert result == {"success": True, "transcript": "привет", "provider": "deepgram"}
    transcribe_deepgram.assert_called_once_with(
        str(audio_path),
        "nova-3",
        language="ru",
        prompt=None,
    )


def test_transcribe_deepgram_uses_config_and_paragraph_transcript(tmp_path, monkeypatch):
    from tools import transcription_tools

    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(
        transcription_tools,
        "get_env_value",
        lambda name, default=None: "dg-test" if name == "DEEPGRAM_API_KEY" else default,
    )
    monkeypatch.setattr(
        transcription_tools,
        "_load_stt_config",
        lambda: {
            "provider": "deepgram",
            "deepgram": {
                "model": "nova-3",
                "language": "ru",
                "smart_format": True,
                "paragraphs": True,
                "utterances": True,
                "numerals": True,
            },
        },
    )

    response = MagicMock(status_code=200)
    response.json.return_value = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "raw transcript",
                            "paragraphs": {"transcript": "formatted transcript"},
                        }
                    ]
                }
            ]
        }
    }

    with patch("requests.post", return_value=response) as post:
        result = transcription_tools._transcribe_deepgram(
            str(audio_path),
            "nova-3",
            language="ru",
            prompt=None,
        )

    assert result == {
        "success": True,
        "transcript": "formatted transcript",
        "provider": "deepgram",
    }
    kwargs = post.call_args.kwargs
    assert kwargs["params"] == {
        "model": "nova-3",
        "smart_format": "true",
        "paragraphs": "true",
        "utterances": "true",
        "numerals": "true",
        "language": "ru",
    }
    assert kwargs["headers"]["Authorization"] == "Token dg-test"
    assert kwargs["headers"]["Content-Type"] == "audio/ogg"
    assert kwargs["timeout"] == 120
