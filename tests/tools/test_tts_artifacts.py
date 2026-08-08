"""TTS artifact path annotations — mirror image_generate's agent_visible contract.

Under docker/ssh/modal the audio bytes land in a relocated Hermes cache. The
tool must keep host paths for MEDIA/gateway delivery while exposing
agent_visible_file_path for terminal/file follow-up.
"""

import json
from types import SimpleNamespace


def test_postprocess_adds_agent_visible_path_for_active_ssh_env(monkeypatch, tmp_path):
    from tools import tts_tool

    hermes_home = tmp_path / ".hermes"
    audio_dir = hermes_home / "cache" / "audio"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "tts_test.ogg"
    audio_path.write_bytes(b"OggS")

    sync_calls = []

    class FakeSyncManager:
        def sync(self, *, force=False):
            sync_calls.append(force)

    env = SimpleNamespace(
        _remote_home="/home/remotesshuser",
        _sync_manager=FakeSyncManager(),
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: env,
    )

    raw = json.dumps({
        "success": True,
        "file_path": str(audio_path),
        "file_paths": [str(audio_path)],
        "media_tag": f"MEDIA:{audio_path}",
    })
    result = json.loads(tts_tool._postprocess_tts_result(raw, task_id="task-1"))

    assert result["file_path"] == str(audio_path)
    assert result["media_tag"] == f"MEDIA:{audio_path}"
    assert result["host_file_path"] == str(audio_path)
    assert result["agent_visible_file_path"] == (
        "/home/remotesshuser/.hermes/cache/audio/tts_test.ogg"
    )
    assert result["host_file_paths"] == [str(audio_path)]
    assert result["agent_visible_file_paths"] == [
        "/home/remotesshuser/.hermes/cache/audio/tts_test.ogg"
    ]
    assert sync_calls == [True]


def test_postprocess_adds_docker_root_hermes_without_env(monkeypatch, tmp_path):
    from tools import tts_tool

    hermes_home = tmp_path / ".hermes"
    audio_dir = hermes_home / "cache" / "audio"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "tts_docker.mp3"
    audio_path.write_bytes(b"ID3")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({"success": True, "file_path": str(audio_path)})
    result = json.loads(tts_tool._postprocess_tts_result(raw))

    assert result["file_path"] == str(audio_path)
    assert result["agent_visible_file_path"] == (
        "/root/.hermes/cache/audio/tts_docker.mp3"
    )


def test_postprocess_noop_on_local_backend(monkeypatch, tmp_path):
    from tools import tts_tool

    hermes_home = tmp_path / ".hermes"
    audio_dir = hermes_home / "cache" / "audio"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "tts_local.mp3"
    audio_path.write_bytes(b"ID3")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({"success": True, "file_path": str(audio_path)})
    result = json.loads(tts_tool._postprocess_tts_result(raw))

    assert result == {"success": True, "file_path": str(audio_path)}
    assert "agent_visible_file_path" not in result


def test_postprocess_annotates_multi_chunk_paths(monkeypatch, tmp_path):
    from tools import tts_tool

    hermes_home = tmp_path / ".hermes"
    audio_dir = hermes_home / "cache" / "audio"
    audio_dir.mkdir(parents=True)
    first = audio_dir / "tts_a.ogg"
    second = audio_dir / "tts_b.ogg"
    first.write_bytes(b"OggS1")
    second.write_bytes(b"OggS2")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({
        "success": True,
        "file_path": str(first),
        "file_paths": [str(first), str(second)],
        "media_tag": f"MEDIA:{first}\nMEDIA:{second}",
    })
    result = json.loads(tts_tool._postprocess_tts_result(raw))

    assert result["media_tag"].startswith("MEDIA:")
    assert "agent_visible_file_path" in result
    assert result["agent_visible_file_paths"] == [
        "/root/.hermes/cache/audio/tts_a.ogg",
        "/root/.hermes/cache/audio/tts_b.ogg",
    ]
