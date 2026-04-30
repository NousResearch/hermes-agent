import json
from importlib import import_module, reload


class _FakeDownloadResponse:
    def __init__(self, body=b"fake-video-bytes"):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _reload_video_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FAL_KEY", "fal-test-key")
    module = import_module("tools.video_generation_tool")
    return reload(module)


def test_text_to_video_submits_default_fal_endpoint_and_saves_mp4(monkeypatch, tmp_path):
    video_generation_tool = _reload_video_tool(monkeypatch, tmp_path)
    captured = {}

    class FakeHandle:
        def get(self):
            return {
                "video": {
                    "url": "https://v3.fal.media/files/lion/render_output.mp4",
                    "content_type": "video/mp4",
                    "file_name": "render_output.mp4",
                }
            }

    def fake_submit(model, arguments):
        captured["model"] = model
        captured["arguments"] = arguments
        return FakeHandle()

    def fake_urlopen(request, timeout=None):
        captured["download_url"] = request.full_url
        captured["download_timeout"] = timeout
        return _FakeDownloadResponse()

    monkeypatch.setattr(video_generation_tool, "_submit_fal_request", fake_submit)
    monkeypatch.setattr(video_generation_tool.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(video_generation_tool.video_generate_tool(
        prompt="a small robot walking through rainy neon streets",
        mode="text_to_video",
        duration="5",
        aspect_ratio="portrait",
    ))

    assert result["success"] is True
    assert result["provider"] == "fal"
    assert result["mode"] == "text_to_video"
    assert result["model"] == "fal-ai/kling-video/v2.1/master/text-to-video"
    assert result["video_url"] == result["video"]
    assert result["media_path"].startswith(str(tmp_path / "generated-videos"))
    assert open(result["media_path"], "rb").read() == b"fake-video-bytes"
    assert captured["download_url"] == result["video"]
    assert captured["download_timeout"] == 300
    assert captured["arguments"] == {
        "prompt": "a small robot walking through rainy neon streets",
        "duration": "5",
        "aspect_ratio": "9:16",
        "negative_prompt": "blur, distort, and low quality",
        "cfg_scale": 0.5,
    }


def test_image_to_video_requires_image_url_and_submits_image_endpoint(monkeypatch, tmp_path):
    video_generation_tool = _reload_video_tool(monkeypatch, tmp_path)
    captured = {}

    class FakeHandle:
        def get(self):
            return {"video": {"url": "https://v3.fal.media/files/rabbit/i2v.mp4"}}

    monkeypatch.setattr(video_generation_tool, "_submit_fal_request", lambda model, arguments: captured.update({"model": model, "arguments": arguments}) or FakeHandle())
    monkeypatch.setattr(video_generation_tool.urllib.request, "urlopen", lambda request, timeout=None: _FakeDownloadResponse(b"image-to-video"))

    result = json.loads(video_generation_tool.video_generate_tool(
        prompt="gentle camera push-in, leaves moving in the wind",
        mode="image_to_video",
        image_url="https://example.com/source.png",
        duration=10,
        aspect_ratio="square",
        cfg_scale=0.7,
    ))

    assert result["success"] is True
    assert result["mode"] == "image_to_video"
    assert captured["model"] == "fal-ai/kling-video/v2.1/master/image-to-video"
    assert captured["arguments"] == {
        "prompt": "gentle camera push-in, leaves moving in the wind",
        "image_url": "https://example.com/source.png",
        "duration": "10",
        "aspect_ratio": "1:1",
        "negative_prompt": "blur, distort, and low quality",
        "cfg_scale": 0.7,
    }


def test_image_to_video_without_image_url_returns_validation_error(monkeypatch, tmp_path):
    video_generation_tool = _reload_video_tool(monkeypatch, tmp_path)
    result = json.loads(video_generation_tool.video_generate_tool(prompt="slow cinematic motion", mode="image_to_video"))
    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "image_url is required" in result["error"]


def test_check_requirements_accepts_direct_fal_key(monkeypatch, tmp_path):
    video_generation_tool = _reload_video_tool(monkeypatch, tmp_path)
    assert video_generation_tool.check_video_generation_requirements() is True
