"""Tests for dedicated FAL media utility tools."""

import json


class _FakeFalHandler:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def get(self):
        if self.endpoint in {
            "fal-ai/birefnet/v2/video",
            "fal-ai/seedvr/upscale/video",
        }:
            return {
                "video": {
                    "url": "https://example.test/out.mp4",
                    "width": 1920,
                    "height": 1080,
                    "fps": 24,
                    "duration": 1,
                    "num_frames": 24,
                    "content_type": "video/mp4",
                },
                "seed": 123,
            }
        if self.endpoint in {
            "fal-ai/birefnet/v2",
            "fal-ai/ideogram/remove-background",
            "fal-ai/seedvr/upscale/image",
        }:
            return {
                "image": {
                    "url": "https://example.test/out.png",
                    "width": 1024,
                    "height": 1024,
                    "content_type": "image/png",
                },
                "seed": 123,
            }
        return {
            "images": [
                {
                    "url": "https://example.test/edit.png",
                    "width": 1024,
                    "height": 1024,
                }
            ]
        }


def test_fal_media_utility_tools_dispatch_to_expected_endpoints(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import model_tools  # noqa: F401 - triggers tool discovery
    import tools.fal_image_edit_tool as image_edit_tool
    import tools.fal_remove_background_tool as remove_background_tool
    import tools.fal_remove_video_background_tool as remove_video_background_tool
    import tools.fal_upscale_image_tool as upscale_image_tool
    import tools.fal_upscale_video_tool as upscale_video_tool
    from tools.registry import registry

    calls = []

    def fake_submit(endpoint, arguments):
        calls.append({"endpoint": endpoint, "arguments": arguments})
        return _FakeFalHandler(endpoint)

    for module in (
        image_edit_tool,
        remove_background_tool,
        remove_video_background_tool,
        upscale_image_tool,
        upscale_video_tool,
    ):
        monkeypatch.setattr(module, "_submit_fal_request", fake_submit)

    results = {
        "image_edit": registry.dispatch(
            "image_edit",
            {
                "prompt": "make the product blue",
                "image_url": "https://example.test/in.png",
            },
        ),
        "remove_background": registry.dispatch(
            "remove_background",
            {"image_url": "https://example.test/in.png"},
        ),
        "remove_video_background": registry.dispatch(
            "remove_video_background",
            {"video_url": "https://example.test/in.mp4"},
        ),
        "upscale_image": registry.dispatch(
            "upscale_image",
            {"image_url": "https://example.test/in.png"},
        ),
        "upscale_video": registry.dispatch(
            "upscale_video",
            {"video_url": "https://example.test/in.mp4"},
        ),
    }

    assert {name: json.loads(raw)["success"] for name, raw in results.items()} == {
        "image_edit": True,
        "remove_background": True,
        "remove_video_background": True,
        "upscale_image": True,
        "upscale_video": True,
    }
    assert [call["endpoint"] for call in calls] == [
        "fal-ai/gemini-3.1-flash-image-preview/edit",
        "fal-ai/ideogram/remove-background",
        "fal-ai/birefnet/v2/video",
        "fal-ai/seedvr/upscale/image",
        "fal-ai/seedvr/upscale/video",
    ]
    assert calls[0]["arguments"]["image_urls"] == ["https://example.test/in.png"]
    assert calls[1]["arguments"]["image_url"] == "https://example.test/in.png"
    assert set(calls[1]["arguments"]) == {"image_url", "sync_mode"}
    assert calls[2]["arguments"]["video_url"] == "https://example.test/in.mp4"
    assert calls[3]["arguments"]["image_url"] == "https://example.test/in.png"
    assert calls[4]["arguments"]["video_url"] == "https://example.test/in.mp4"


def test_remove_background_can_use_birefnet_model(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.fal_remove_background_tool as remove_background_tool

    calls = []

    def fake_submit(endpoint, arguments):
        calls.append({"endpoint": endpoint, "arguments": arguments})
        return _FakeFalHandler(endpoint)

    monkeypatch.setattr(remove_background_tool, "_submit_fal_request", fake_submit)

    result = json.loads(
        remove_background_tool.remove_background_tool(
            image_url="https://example.test/in.png",
            model="General Use (Light)",
            output_mask=True,
        )
    )

    assert result["success"] is True
    assert result["model"] == "General Use (Light)"
    assert calls == [
        {
            "endpoint": "fal-ai/birefnet/v2",
            "arguments": {
                "image_url": "https://example.test/in.png",
                "model": "General Use (Light)",
                "operating_resolution": "1024x1024",
                "output_format": "png",
                "output_mask": True,
                "mask_only": False,
                "refine_foreground": True,
                "sync_mode": False,
            },
        }
    ]


def test_upscale_video_rejects_image_inputs(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.fal_upscale_video_tool as upscale_video_tool

    result = json.loads(
        upscale_video_tool.upscale_video_tool(
            video_url="https://example.test/not-a-video.png",
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "must reference a video file" in result["error"]


def test_upscale_video_uploads_local_video_before_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.image_generation_tool as image_tool
    import tools.fal_upscale_video_tool as upscale_video_tool

    local_video = tmp_path / "input.mp4"
    local_video.write_bytes(b"fake mp4")
    submitted = {}

    class FakeFalClient:
        @staticmethod
        def upload_file(path):
            assert path == str(local_video)
            return "https://fal.media/uploaded/input.mp4"

    class Handler:
        def get(self):
            return {"video": {"url": "https://example.test/out.mp4"}}

    def fake_submit(_endpoint, arguments):
        submitted.update(arguments)
        return Handler()

    monkeypatch.setattr(image_tool, "fal_client", FakeFalClient)
    monkeypatch.setattr(upscale_video_tool, "_submit_fal_request", fake_submit)

    result = json.loads(upscale_video_tool.upscale_video_tool(video_url=str(local_video)))

    assert result["success"] is True
    assert submitted["video_url"] == "https://fal.media/uploaded/input.mp4"
    assert submitted["output_format"] == "X264 (.mp4)"
    assert submitted["output_quality"] == "high"
    assert submitted["output_write_mode"] == "balanced"


def test_upscale_video_local_video_requires_direct_fal_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)

    import tools.fal_upscale_video_tool as upscale_video_tool

    local_video = tmp_path / "input.mp4"
    local_video.write_bytes(b"fake mp4")

    result = json.loads(upscale_video_tool.upscale_video_tool(video_url=str(local_video)))

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "Local video file inputs require a direct FAL_KEY" in result["error"]


def test_remove_video_background_uploads_local_video_before_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.image_generation_tool as image_tool
    import tools.fal_remove_video_background_tool as remove_video_background_tool

    local_video = tmp_path / "input.webm"
    local_video.write_bytes(b"fake webm")
    submitted = {}

    class FakeFalClient:
        @staticmethod
        def upload_file(path):
            assert path == str(local_video)
            return "https://fal.media/uploaded/input.webm"

    class Handler:
        def get(self):
            return {"video": {"url": "https://example.test/out.webm"}}

    def fake_submit(_endpoint, arguments):
        submitted.update(arguments)
        return Handler()

    monkeypatch.setattr(image_tool, "fal_client", FakeFalClient)
    monkeypatch.setattr(remove_video_background_tool, "_submit_fal_request", fake_submit)

    result = json.loads(
        remove_video_background_tool.remove_video_background_tool(video_url=str(local_video))
    )

    assert result["success"] is True
    assert submitted["video_url"] == "https://fal.media/uploaded/input.webm"


def test_upscale_video_rejects_documentation_page_urls(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.fal_upscale_video_tool as upscale_video_tool

    result = json.loads(
        upscale_video_tool.upscale_video_tool(
            video_url="https://docs.fal.ai/model/fal-ai/seedvr/upscale/video",
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "direct, publicly downloadable video file URL" in result["error"]


def test_remove_video_background_rejects_documentation_page_urls(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.fal_remove_video_background_tool as remove_video_background_tool

    result = json.loads(
        remove_video_background_tool.remove_video_background_tool(
            video_url="https://docs.fal.ai/model/fal-ai/birefnet/v2/video",
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "direct, publicly downloadable video file URL" in result["error"]


def test_upscale_video_omits_missing_metadata(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")

    import tools.fal_upscale_video_tool as upscale_video_tool

    class Handler:
        def get(self):
            return {"video": {"url": "https://example.test/out.mp4"}}

    monkeypatch.setattr(upscale_video_tool, "_submit_fal_request", lambda *_args, **_kw: Handler())

    result = json.loads(
        upscale_video_tool.upscale_video_tool(
            video_url="https://example.test/in.mp4",
        )
    )

    assert result["success"] is True
    assert result["video"] == "https://example.test/out.mp4"
    assert result["output_url"] == "https://example.test/out.mp4"
    assert result["media_type"] == "video"
    assert "width" not in result
    assert "height" not in result
    assert "fps" not in result
