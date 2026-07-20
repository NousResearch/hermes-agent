import importlib
import json
from pathlib import Path
from unittest.mock import patch

from toolsets import resolve_toolset


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _import_module():
    module = importlib.import_module("tools.prompt_faithful_image_tool")
    return importlib.reload(module)


def test_requires_key(monkeypatch):
    monkeypatch.delenv("NXFL_API_KEY", raising=False)
    module = _import_module()

    payload = json.loads(module.prompt_faithful_image_generate_tool("画一只猫"))

    assert payload["success"] is False
    assert "NXFL_API_KEY" in payload["error"]


def test_sends_exact_prompt_and_returns_media_tag(monkeypatch):
    monkeypatch.setenv("NXFL_API_KEY", "sk-test")
    monkeypatch.setenv("NXFL_IMAGE_API_URL", "https://api.example.test/v1/chat/completions")
    monkeypatch.setenv("NXFL_IMAGE_MODEL", "gemini-imagen")
    module = _import_module()

    prompt = "按这个提示词原样生图，不要改写：夜晚下雨的重庆街头，35mm 纪实摄影"
    captured_requests = []

    def fake_urlopen(req, timeout=0):
        if getattr(req, "full_url", "").startswith("https://api.example.test/"):
            captured_requests.append(json.loads(req.data.decode("utf-8")))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "![生成结果](https://cdn.example.test/out/test-image.png)"
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            return _FakeResponse(body, status=200)
        if getattr(req, "full_url", "") == "https://cdn.example.test/out/test-image.png":
            return _FakeResponse(b"\x89PNG\r\n\x1a\n" + (b"0" * 4096), status=200)
        raise AssertionError(f"Unexpected URL: {getattr(req, 'full_url', req)!r}")

    with patch("tools.prompt_faithful_image_tool.urllib.request.urlopen", side_effect=fake_urlopen):
        payload = json.loads(module.prompt_faithful_image_generate_tool(prompt))

    assert captured_requests
    assert captured_requests[0]["model"] == "gemini-imagen"
    assert captured_requests[0]["messages"][0]["content"][0]["text"] == prompt
    assert payload["success"] is True
    assert payload["prompt"] == prompt
    assert payload["image_url"] == "https://cdn.example.test/out/test-image.png"
    assert payload["media_tag"].startswith("MEDIA:")
    local_path = Path(payload["local_path"])
    assert local_path.is_absolute()
    assert local_path.exists()
    assert payload["media_tag"] == f"MEDIA:{local_path}"


def test_extracts_markdown_image_url():
    module = _import_module()

    image_url = module.extract_image_url(
        {
            "choices": [
                {
                    "message": {
                        "content": "![生成图片](https://static.example.test/generated/foo.webp)"
                    }
                }
            ]
        }
    )

    assert image_url == "https://static.example.test/generated/foo.webp"


def test_registry_handler_returns_prompt_and_media(monkeypatch):
    monkeypatch.setenv("NXFL_API_KEY", "sk-test")
    module = _import_module()

    def fake_urlopen(req, timeout=0):
        if getattr(req, "data", None):
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "![图](https://cdn.example.test/out/registry-check.png)"
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            return _FakeResponse(body, status=200)
        return _FakeResponse(b"\x89PNG\r\n\x1a\n" + (b"1" * 4096), status=200)

    with patch("tools.prompt_faithful_image_tool.urllib.request.urlopen", side_effect=fake_urlopen):
        raw = module.registry.dispatch(
            "prompt_faithful_image_generate",
            {"prompt": "只按这句话生图：一把放在木桌上的旧左轮手枪"},
        )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["prompt"] == "只按这句话生图：一把放在木桌上的旧左轮手枪"
    assert payload["media_tag"].startswith("MEDIA:")


def test_rejects_non_image_downloads(monkeypatch):
    monkeypatch.setenv("NXFL_API_KEY", "sk-test")
    monkeypatch.setenv("NXFL_IMAGE_API_URL", "https://api.example.test/v1/chat/completions")
    module = _import_module()

    def fake_urlopen(req, timeout=0):
        if getattr(req, "full_url", "").startswith("https://api.example.test/"):
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "https://cdn.example.test/out/not-an-image"
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            return _FakeResponse(body, status=200)
        if getattr(req, "full_url", "") == "https://cdn.example.test/out/not-an-image":
            return _FakeResponse(
                (b"<html>" + b"0" * 4096 + b"</html>"),
                status=200,
                headers={"Content-Type": "text/html"},
            )
        raise AssertionError(f"Unexpected URL: {getattr(req, 'full_url', req)!r}")

    with patch("tools.prompt_faithful_image_tool.urllib.request.urlopen", side_effect=fake_urlopen):
        payload = json.loads(module.prompt_faithful_image_generate_tool("只按原样生图"))

    assert payload["success"] is False
    assert "did not return an image" in payload["error"]


def test_model_tools_exposes_tool_in_image_gen_toolset(monkeypatch):
    monkeypatch.setenv("NXFL_API_KEY", "sk-test")
    import model_tools

    defs = model_tools.get_tool_definitions(enabled_toolsets=["image_gen"], quiet_mode=True)
    names = {tool_def["function"]["name"] for tool_def in defs}

    assert "prompt_faithful_image_generate" in resolve_toolset("image_gen")
    assert "prompt_faithful_image_generate" in names


def test_model_tools_hides_tool_without_key(monkeypatch):
    monkeypatch.delenv("NXFL_API_KEY", raising=False)
    import model_tools

    defs = model_tools.get_tool_definitions(enabled_toolsets=["image_gen"], quiet_mode=True)
    names = {tool_def["function"]["name"] for tool_def in defs}

    assert "prompt_faithful_image_generate" not in names
