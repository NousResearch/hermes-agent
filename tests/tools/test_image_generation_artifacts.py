import json
from types import SimpleNamespace

import pytest


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


def test_postprocess_preserves_raw_text_and_non_success_json_strings():
    from tools import image_generation_tool

    unchanged_results = (
        "provider returned unparseable raw text",
        json.dumps({"success": False, "error": "generation failed"}),
        json.dumps("provider returned serialized text"),
    )

    for raw in unchanged_results:
        result = image_generation_tool._postprocess_image_generate_result(raw)

        assert isinstance(result, str)
        assert result == raw


def test_postprocess_requires_normalized_string_input():
    from tools import image_generation_tool

    with pytest.raises(TypeError, match="requires a normalized string"):
        image_generation_tool._postprocess_image_generate_result(
            {"success": True, "image": "/tmp/image.png"}
        )


def test_postprocess_transformed_result_is_valid_json_and_preserves_fields(
    monkeypatch, tmp_path
):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_path = hermes_home / "cache" / "images" / "generated-雪.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    env = SimpleNamespace(_remote_home="/home/remote", _sync_manager=None)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: env)

    original = {
        "success": True,
        "image": str(image_path),
        "provider": "azure-openai",
        "metadata": {"prompt": "draw 雪"},
    }
    raw = image_generation_tool._postprocess_image_generate_result(
        json.dumps(original, ensure_ascii=False)
    )
    result = json.loads(raw)

    assert isinstance(raw, str)
    assert result == {
        **original,
        "host_image": str(image_path),
        "agent_visible_image": "/home/remote/.hermes/cache/images/generated-雪.png",
    }


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
        lambda prompt, aspect_ratio, **kw: json.dumps({"success": True, "image": str(image_path)}),
    )

    result = json.loads(
        image_generation_tool._handle_image_generate(
            {"prompt": "draw", "aspect_ratio": "square"},
            task_id="plugin-task",
        )
    )

    assert seen_task_ids == ["plugin-task"]
    assert result["agent_visible_image"] == "/home/remote/.hermes/cache/images/plugin.png"


def test_normalize_image_tool_result_preserves_raw_unicode_string():
    from tools import image_generation_tool

    raw = '  {"message": "雪"}  '

    assert image_generation_tool._normalize_image_tool_result(raw) == raw


def test_normalize_image_tool_result_serializes_mapping_as_unicode_json():
    from tools import image_generation_tool

    result = image_generation_tool._normalize_image_tool_result(
        {"success": True, "image": "https://example.com/雪.png"}
    )

    assert "雪" in result
    assert json.loads(result) == {
        "success": True,
        "image": "https://example.com/雪.png",
    }


def test_normalize_image_tool_result_rejects_unsupported_and_unserializable_values():
    from tools import image_generation_tool

    for value in (["unexpected"], {"success": True, "bad": {1, 2}}):
        raw = image_generation_tool._normalize_image_tool_result(value)
        result = json.loads(raw)

        assert isinstance(raw, str)
        assert result["error_type"] == "tool_result_contract"
        assert result["tool"] == "image_generate"
        assert result["result_type"] == type(value).__name__


def test_handle_image_generate_normalizes_all_provider_branches(monkeypatch):
    from tools import image_generation_tool

    mapping_result = {
        "success": True,
        "image": "https://example.com/雪.png",
        "provider": "test",
    }

    for branch in ("plugin", "managed_krea", "fal"):
        with monkeypatch.context() as scoped:
            scoped.setattr(
                image_generation_tool,
                "_dispatch_to_plugin_provider",
                lambda *args, **kwargs: mapping_result if branch == "plugin" else None,
            )
            scoped.setattr(
                image_generation_tool,
                "_maybe_route_managed_krea",
                lambda *args, **kwargs: mapping_result if branch == "managed_krea" else None,
            )
            scoped.setattr(
                image_generation_tool,
                "image_generate_tool",
                lambda *args, **kwargs: mapping_result,
            )

            raw = image_generation_tool._handle_image_generate({"prompt": "draw"})

            assert isinstance(raw, str)
            assert "雪" in raw
            assert json.loads(raw) == mapping_result
