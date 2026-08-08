"""browser_vision screenshot cache placement + agent_visible annotation."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.browser_camofox import camofox_navigate, camofox_vision


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def test_camofox_vision_writes_under_mounted_cache_screenshots(monkeypatch, tmp_path):
    """Camofox must use get_hermes_dir(cache/screenshots), not the legacy
    top-level browser_screenshots dir that Docker does not bind-mount."""
    from hermes_constants import get_hermes_dir
    from tools.credential_files import map_cache_path_to_container

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    with (
        patch("tools.browser_camofox.requests.post") as mock_post,
        patch("tools.browser_camofox._get_raw") as mock_get_raw,
        patch("tools.browser_camofox._get") as mock_get,
        patch("agent.auxiliary_client.call_llm") as mock_llm,
        patch("tools.browser_camofox.load_config", return_value={}),
        patch("tools.browser_camofox._camofox_private_page_block", return_value=None),
    ):
        mock_post.return_value = _mock_response(
            json_data={"tabId": "tab-cache", "url": "https://example.com"}
        )
        camofox_navigate("https://example.com", task_id="t-cache")

        raw = MagicMock()
        raw.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        mock_get_raw.return_value = raw
        mock_get.return_value = {"snapshot": ""}
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_llm.return_value = MagicMock(choices=[mock_choice])

        result = json.loads(camofox_vision("what?", annotate=False, task_id="t-cache"))

    path = Path(result["screenshot_path"])
    expected_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
    assert path.parent.resolve() == expected_dir.resolve()
    assert path.is_file()
    assert "browser_screenshots" not in path.parts or "cache" in path.parts
    assert path.parent.name == "screenshots"
    assert map_cache_path_to_container(str(path)) == (
        f"/root/.hermes/cache/screenshots/{path.name}"
    )


def test_annotate_browser_vision_adds_agent_visible_under_docker(monkeypatch, tmp_path):
    from tools import browser_tool

    hermes_home = tmp_path / ".hermes"
    shot_dir = hermes_home / "cache" / "screenshots"
    shot_dir.mkdir(parents=True)
    shot = shot_dir / "browser_screenshot_abc.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({
        "success": True,
        "analysis": "ok",
        "screenshot_path": str(shot),
    })
    out = json.loads(browser_tool._annotate_browser_vision_result(raw))
    assert out["screenshot_path"] == str(shot)
    assert out["agent_visible_screenshot"] == (
        f"/root/.hermes/cache/screenshots/{shot.name}"
    )


def test_annotate_browser_vision_native_meta_under_ssh(monkeypatch, tmp_path):
    from tools import browser_tool

    hermes_home = tmp_path / ".hermes"
    shot_dir = hermes_home / "cache" / "screenshots"
    shot_dir.mkdir(parents=True)
    shot = shot_dir / "native.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")

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

    payload = {
        "type": "multimodal",
        "meta": {"screenshot_path": str(shot)},
        "text_summary": f"Screenshot path: {shot}",
    }
    out = browser_tool._annotate_browser_vision_result(payload, task_id="t1")
    assert out["meta"]["agent_visible_screenshot"] == (
        f"/home/remotesshuser/.hermes/cache/screenshots/{shot.name}"
    )
    assert out["meta"]["screenshot_path"] == str(shot)
    assert sync_calls == [True]


def test_annotate_browser_vision_noop_on_local(monkeypatch, tmp_path):
    from tools import browser_tool

    hermes_home = tmp_path / ".hermes"
    shot_dir = hermes_home / "cache" / "screenshots"
    shot_dir.mkdir(parents=True)
    shot = shot_dir / "local.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "tools.image_generation_tool._active_terminal_env",
        lambda task_id: None,
    )

    raw = json.dumps({"success": True, "screenshot_path": str(shot)})
    out = json.loads(browser_tool._annotate_browser_vision_result(raw))
    assert out == {"success": True, "screenshot_path": str(shot)}
    assert "agent_visible_screenshot" not in out
