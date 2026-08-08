"""Video artifact path annotations — mirror image_generate's agent_visible contract.

Under docker/ssh/modal local video bytes land in a relocated Hermes cache.
``video`` stays the host/gateway path; ``agent_visible_video`` is for
terminal/file follow-up inside the sandbox.
"""

import json
from types import SimpleNamespace


def test_postprocess_adds_agent_visible_video_for_active_ssh_env(monkeypatch, tmp_path):
    from tools import video_generation_tool

    hermes_home = tmp_path / ".hermes"
    video_dir = hermes_home / "cache" / "videos"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "deepinfra_clip.mp4"
    video_path.write_bytes(b"ftyp")

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

    raw = json.dumps({"success": True, "video": str(video_path)})
    result = json.loads(
        video_generation_tool._postprocess_video_generate_result(raw, task_id="task-1")
    )

    assert result["video"] == str(video_path)
    assert result["host_video"] == str(video_path)
    assert result["agent_visible_video"] == (
        "/home/remotesshuser/.hermes/cache/videos/deepinfra_clip.mp4"
    )
    assert sync_calls == [True]


def test_postprocess_adds_docker_root_hermes_without_env(monkeypatch, tmp_path):
    from tools import video_generation_tool

    hermes_home = tmp_path / ".hermes"
    video_dir = hermes_home / "cache" / "videos"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "clip.mp4"
    video_path.write_bytes(b"ftyp")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({"success": True, "video": str(video_path)})
    result = json.loads(video_generation_tool._postprocess_video_generate_result(raw))

    assert result["video"] == str(video_path)
    assert result["agent_visible_video"] == "/root/.hermes/cache/videos/clip.mp4"


def test_postprocess_noop_on_local_backend(monkeypatch, tmp_path):
    from tools import video_generation_tool

    hermes_home = tmp_path / ".hermes"
    video_dir = hermes_home / "cache" / "videos"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "local.mp4"
    video_path.write_bytes(b"ftyp")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({"success": True, "video": str(video_path)})
    result = json.loads(video_generation_tool._postprocess_video_generate_result(raw))

    assert result == {"success": True, "video": str(video_path)}
    assert "agent_visible_video" not in result


def test_postprocess_noop_on_http_url(monkeypatch):
    from tools import video_generation_tool

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({
        "success": True,
        "video": "https://cdn.example/clip.mp4",
    })
    result = json.loads(video_generation_tool._postprocess_video_generate_result(raw))

    assert result == {
        "success": True,
        "video": "https://cdn.example/clip.mp4",
    }
    assert "agent_visible_video" not in result


def test_handle_video_generate_postprocesses_local_result(monkeypatch, tmp_path):
    from tools import video_generation_tool

    hermes_home = tmp_path / ".hermes"
    video_dir = hermes_home / "cache" / "videos"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "handled.mp4"
    video_path.write_bytes(b"ftyp")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    class FakeProvider:
        name = "fake"

        def default_model(self):
            return "fake-model"

        def generate(self, prompt, **kwargs):
            return {"success": True, "video": str(video_path), "prompt": prompt}

    monkeypatch.setattr(
        video_generation_tool, "_resolve_active_provider", lambda: FakeProvider()
    )
    monkeypatch.setattr(
        video_generation_tool, "_read_configured_video_provider", lambda: "fake"
    )
    monkeypatch.setattr(
        video_generation_tool, "_read_configured_video_model", lambda: "fake-model"
    )
    monkeypatch.setattr(
        "tools.image_generation_tool._confine_source_images",
        lambda *a, **k: (None, None, None),
    )
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    out = json.loads(
        video_generation_tool._handle_video_generate(
            {"prompt": "a cat walks"},
            task_id="t1",
        )
    )
    assert out["video"] == str(video_path)
    assert out["agent_visible_video"] == "/root/.hermes/cache/videos/handled.mp4"
