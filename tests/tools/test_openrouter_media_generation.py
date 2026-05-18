from __future__ import annotations

import base64
import importlib.util
from pathlib import Path


def _load_plugin(path: str, name: str):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, root / path / "__init__.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload, status_code=200, text="OK"):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ImageClient:
    last_payload = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self.__class__.last_payload = json
        b64 = base64.b64encode(b"fake-png").decode("ascii")
        return _FakeResponse({
            "choices": [{
                "message": {
                    "content": "done",
                    "images": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }],
                }
            }],
            "usage": {"cost": 0.01},
        })


class _VideoClient:
    post_payload = None
    poll_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self.__class__.post_payload = json
        return _FakeResponse({
            "id": "job-1",
            "polling_url": "https://openrouter.ai/api/v1/videos/job-1",
            "status": "pending",
        })

    def get(self, url, headers=None, timeout=None):
        self.__class__.poll_count += 1
        return _FakeResponse({
            "id": "job-1",
            "polling_url": "https://openrouter.ai/api/v1/videos/job-1",
            "status": "completed",
            "unsigned_urls": ["https://openrouter.ai/api/v1/videos/job-1/content?index=0"],
            "usage": {"cost": 0.25},
        })


def test_openrouter_image_provider_saves_data_url(monkeypatch, tmp_path):
    module = _load_plugin("plugins/image_gen/openrouter", "openrouter_image_plugin_test")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(module.httpx, "Client", _ImageClient)

    provider = module.OpenRouterImageGenProvider()
    result = provider.generate("draw a cat", aspect_ratio="portrait", model="custom/image-model")

    assert result["success"] is True
    assert result["provider"] == "openrouter"
    assert result["model"] == "custom/image-model"
    assert result["aspect_ratio"] == "portrait"
    assert result["image"].endswith(".png")
    assert Path(result["image"]).read_bytes() == b"fake-png"
    assert _ImageClient.last_payload is not None
    assert _ImageClient.last_payload["modalities"] == ["image", "text"]
    assert _ImageClient.last_payload["image_config"]["aspect_ratio"] == "9:16"


def test_openrouter_image_provider_reports_missing_key(monkeypatch):
    module = _load_plugin("plugins/image_gen/openrouter", "openrouter_image_plugin_no_key_test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    provider = module.OpenRouterImageGenProvider()
    result = provider.generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "auth_required"


def test_openrouter_video_provider_submits_and_polls(monkeypatch):
    module = _load_plugin("plugins/video_gen/openrouter", "openrouter_video_plugin_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_VIDEO_POLL_INTERVAL", "0")
    monkeypatch.setattr(module.httpx, "Client", _VideoClient)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    provider = module.OpenRouterVideoGenProvider()
    result = provider.generate(
        "a beach dog",
        model="custom/video-model",
        image_url="https://example.com/first.png",
        duration=4,
        aspect_ratio="9:16",
        resolution="1080p",
        audio=False,
        seed=123,
    )

    assert result["success"] is True
    assert result["provider"] == "openrouter"
    assert result["model"] == "custom/video-model"
    assert result["modality"] == "image"
    assert result["video"].endswith("/content?index=0")
    assert _VideoClient.post_payload is not None
    assert _VideoClient.post_payload["frame_images"][0]["frame_type"] == "first_frame"
    assert _VideoClient.post_payload["generate_audio"] is False
    assert _VideoClient.post_payload["seed"] == 123
    assert _VideoClient.poll_count >= 1


def test_openrouter_video_provider_reference_images(monkeypatch):
    module = _load_plugin("plugins/video_gen/openrouter", "openrouter_video_refs_plugin_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_VIDEO_POLL_INTERVAL", "0")
    monkeypatch.setattr(module.httpx, "Client", _VideoClient)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    provider = module.OpenRouterVideoGenProvider()
    result = provider.generate(
        "match this style",
        reference_image_urls=["https://example.com/ref.png"],
    )

    assert result["success"] is True
    assert result["modality"] == "text"
    assert _VideoClient.post_payload is not None
    assert "frame_images" not in _VideoClient.post_payload
    assert _VideoClient.post_payload["input_references"][0]["image_url"]["url"] == "https://example.com/ref.png"
