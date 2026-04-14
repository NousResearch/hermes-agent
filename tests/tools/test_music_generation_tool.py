import json
from unittest.mock import MagicMock, patch

import httpx
from agent.background_task import BackgroundTaskHandle, SessionOrigin


def _mock_response(payload):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def test_prompt_only_flow_generates_lyrics_before_creating_song(monkeypatch):
    from tools.music_generation_tool import SENSEAUDIO_MUSIC_MODEL, music_generate_tool

    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.side_effect = [
        _mock_response(
            {
                "data": [
                    {
                        "text": "[intro-medium] ; [verse] Soft lights glow tonight ; [chorus] Drift into the lo-fi haze",
                        "title": "Midnight Notes",
                    }
                ]
            }
        ),
        _mock_response({"task_id": "song-task-123"}),
    ]

    pending_data = {
        "status": "SUCCESS",
        "response": {
            "data": [
                {
                    "audio_url": "https://example.com/track.mp3",
                    "cover_url": "https://example.com/cover.jpg",
                    "duration": 98,
                    "title": "Midnight Notes",
                    "lyrics": "[intro-medium] ; [verse] Soft lights glow tonight ; [chorus] Drift into the lo-fi haze",
                }
            ]
        },
    }

    monkeypatch.setenv("SENSEAUDIO_API_KEY", "test-key")

    background_create = MagicMock(return_value=None)

    with patch("tools.music_generation_tool.current_session_origin", return_value=SessionOrigin("", "", "")), \
         patch("tools.music_generation_tool.httpx.Client", return_value=client), \
         patch("tools.music_generation_tool.background_tasks.create", background_create), \
         patch("tools.music_generation_tool._download_track_to_local", return_value="/tmp/midnight-notes.mp3"), \
         patch("tools.music_generation_tool._poll_sync", return_value=pending_data):
        result = json.loads(
            music_generate_tool(
                prompt="Relaxing lo-fi music for studying and cafe ambience.",
                style="lo-fi",
            )
        )

    assert result["provider"] == "senseaudio"
    assert result["tracks"][0]["url"] == "https://example.com/track.mp3"
    assert result["tracks"][0]["localPath"] == "/tmp/midnight-notes.mp3"
    background_create.assert_not_called()

    assert client.post.call_count == 2

    lyrics_call = client.post.call_args_list[0]
    assert lyrics_call.args[0].endswith("/music/lyrics/create")
    assert lyrics_call.kwargs["json"] == {
        "prompt": "Relaxing lo-fi music for studying and cafe ambience.",
        "provider": SENSEAUDIO_MUSIC_MODEL,
    }

    song_call = client.post.call_args_list[1]
    assert song_call.args[0].endswith("/music/song/create")
    assert song_call.kwargs["json"] == {
        "model": SENSEAUDIO_MUSIC_MODEL,
        "lyrics": "[intro-medium] ; [verse] Soft lights glow tonight ; [chorus] Drift into the lo-fi haze",
        "custom_mode": False,
        "instrumental": False,
        "style": "lo-fi",
        "title": "Midnight Notes",
    }


def test_create_song_surfaces_senseaudio_error_details():
    from tools.music_generation_tool import _create_song

    response = MagicMock(spec=httpx.Response)
    response.status_code = 400
    response.json.return_value = {
        "code": "invalid",
        "message": "Invalid lyrics format",
        "ref_code": 400709,
    }
    response.text = '{"code":"invalid","message":"Invalid lyrics format","ref_code":400709}'
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400 Bad Request",
        request=MagicMock(),
        response=response,
    )

    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = None
    client.post.return_value = response

    with patch("tools.music_generation_tool.httpx.Client", return_value=client):
        try:
            _create_song("lo-fi prompt")
        except ValueError as exc:
            assert "Invalid lyrics format" in str(exc)
            assert "code=invalid" in str(exc)
            assert "ref_code=400709" in str(exc)
        else:
            raise AssertionError("Expected _create_song() to raise ValueError")


def test_music_generate_returns_started_for_cli_session(monkeypatch):
    from tools.music_generation_tool import music_generate_tool

    monkeypatch.setenv("SENSEAUDIO_API_KEY", "test-key")

    handle = BackgroundTaskHandle(
        task_id="bg-task-1",
        session_key="cli-session-123",
        origin=SessionOrigin(
            session_key="cli-session-123",
            platform="cli",
            chat_id="cli-session-123",
        ),
        label="music generation",
    )

    def _fake_create(*, coro, session_key, origin, label):
        coro.close()
        return handle

    with patch(
        "tools.music_generation_tool.current_session_origin",
        return_value=SessionOrigin("cli-session-123", "cli", "cli-session-123"),
    ), \
         patch(
             "tools.music_generation_tool._create_lyrics_from_prompt",
             return_value={"lyrics": "[verse] hello world", "title": "Test Song"},
         ), \
         patch("tools.music_generation_tool._create_song", return_value="sense-task-1"), \
         patch("tools.music_generation_tool.background_tasks.create", side_effect=_fake_create), \
         patch("tools.music_generation_tool._poll_sync") as mock_poll_sync:
        result = json.loads(music_generate_tool(prompt="warm lofi beat"))

    assert result["status"] == "started"
    assert result["task_id"] == "sense-task-1"
    assert "background" in result["message"].lower()
    mock_poll_sync.assert_not_called()


def test_music_generation_poll_timeout_extended_to_30_minutes():
    from tools.music_generation_tool import POLL_MAX_WAIT

    assert POLL_MAX_WAIT == 1800


def test_format_result_downloads_tracks_to_local_paths():
    from tools.music_generation_tool import _format_result

    pending_data = {
        "status": "SUCCESS",
        "response": {
            "data": [
                {
                    "audio_url": "https://example.com/generated.mp3",
                    "cover_url": "https://example.com/cover.jpg",
                    "duration": 123,
                    "title": "Night Drive",
                    "lyrics": "[verse] neon lights blur",
                }
            ]
        },
    }

    with patch(
        "tools.music_generation_tool._download_track_to_local",
        return_value="/tmp/night-drive.mp3",
    ) as mock_download:
        result = _format_result(pending_data, ignored_overrides=[])

    mock_download.assert_called_once_with(
        "https://example.com/generated.mp3",
        file_name="Night Drive.mp3",
    )
    track = result["tracks"][0]
    assert track["url"] == "https://example.com/generated.mp3"
    assert track["localPath"] == "/tmp/night-drive.mp3"
    assert track["mediaTag"] == "MEDIA:/tmp/night-drive.mp3"
    assert track["metadata"]["source_url"] == "https://example.com/generated.mp3"


def test_wake_message_includes_local_media_delivery_path():
    from tools.music_generation_tool import _wake_message

    message = _wake_message(
        {
            "tracks": [
                {
                    "url": "https://example.com/generated.mp3",
                    "localPath": "/tmp/night-drive.mp3",
                    "fileName": "Night Drive.mp3",
                    "mediaTag": "MEDIA:/tmp/night-drive.mp3",
                    "metadata": {
                        "title": "Night Drive",
                        "duration": 123,
                    },
                }
            ],
            "lyrics": ["[verse] neon lights blur"],
        }
    )

    assert "Saved: /tmp/night-drive.mp3" in message
    assert "MEDIA:/tmp/night-drive.mp3" in message


def test_get_music_provider_defaults_to_senseaudio():
    from tools.music_generation_tool import DEFAULT_PROVIDER, _get_provider

    assert DEFAULT_PROVIDER == "senseaudio"
    assert _get_provider({}) == "senseaudio"


def test_music_generate_dispatches_generate_to_configured_provider():
    from tools.music_generation_tool import music_generate_tool

    with patch("tools.music_generation_tool._load_music_config", return_value={"provider": "senseaudio"}), \
         patch("tools.music_generation_tool._senseaudio_generate", return_value='{"status":"started"}') as mock_generate:
        result = music_generate_tool(prompt="warm lofi beat")

    assert json.loads(result) == {"status": "started"}
    mock_generate.assert_called_once()


def test_music_generate_dispatches_status_to_configured_provider():
    from tools.music_generation_tool import music_generate_tool

    with patch("tools.music_generation_tool._load_music_config", return_value={"provider": "senseaudio"}), \
         patch("tools.music_generation_tool._senseaudio_status", return_value='{"status":"running"}') as mock_status:
        result = music_generate_tool(action="status")

    assert json.loads(result) == {"status": "running"}
    mock_status.assert_called_once()
