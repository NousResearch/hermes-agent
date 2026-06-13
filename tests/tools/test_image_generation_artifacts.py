import json
from types import SimpleNamespace


def test_postprocess_adds_agent_visible_image_for_active_ssh_env(monkeypatch, tmp_path):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "xai_grok-imagine-image_test.jpg"
    image_path.write_bytes(b"jpg")

    sync_calls = []

    class FakeSyncManager:
        def sync(self, *, force=False):
            sync_calls.append(force)

    env = SimpleNamespace(
        _remote_home="/home/remotesshuser",
        _sync_manager=FakeSyncManager(),
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: env)

    raw = json.dumps({"success": True, "image": str(image_path)})
    result = json.loads(
        image_generation_tool._postprocess_image_generate_result(raw, task_id="task-1")
    )

    assert result["image"] == str(image_path)
    assert result["host_image"] == str(image_path)
    assert result["agent_visible_image"] == (
        "/home/remotesshuser/.hermes/cache/images/xai_grok-imagine-image_test.jpg"
    )
    assert sync_calls == [True]


def test_postprocess_maps_docker_cache_path_without_active_env(monkeypatch, tmp_path):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "generated.png"
    image_path.write_bytes(b"png")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: None)

    raw = json.dumps({"success": True, "image": str(image_path)})
    result = json.loads(image_generation_tool._postprocess_image_generate_result(raw))

    assert result["image"] == str(image_path)
    assert result["agent_visible_image"] == "/root/.hermes/cache/images/generated.png"


def test_postprocess_maps_ssh_cache_path_without_active_env(monkeypatch, tmp_path):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "first-call.png"
    image_path.write_bytes(b"png")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: None)

    raw = json.dumps({"success": True, "image": str(image_path)})
    result = json.loads(image_generation_tool._postprocess_image_generate_result(raw))

    assert result["image"] == str(image_path)
    assert result["agent_visible_image"] == "~/.hermes/cache/images/first-call.png"


def test_postprocess_leaves_remote_image_urls_unchanged(monkeypatch):
    from tools import image_generation_tool

    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: None)

    raw = json.dumps({"success": True, "image": "https://example.com/image.png"})

    assert image_generation_tool._postprocess_image_generate_result(raw) == raw


def test_handle_image_generate_postprocesses_plugin_result(monkeypatch, tmp_path):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "plugin.png"
    image_path.write_bytes(b"png")

    env = SimpleNamespace(_remote_home="/home/remote", _sync_manager=None)

    seen_task_ids = []

    def fake_active_env(task_id):
        seen_task_ids.append(task_id)
        return env

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", fake_active_env)
    monkeypatch.setattr(
        image_generation_tool,
        "_dispatch_to_plugin_provider",
        lambda prompt, aspect_ratio: json.dumps({"success": True, "image": str(image_path)}),
    )

    result = json.loads(
        image_generation_tool._handle_image_generate(
            {"prompt": "draw", "aspect_ratio": "square"},
            task_id="plugin-task",
        )
    )

    assert seen_task_ids == ["plugin-task"]
    assert result["agent_visible_image"] == "/home/remote/.hermes/cache/images/plugin.png"
