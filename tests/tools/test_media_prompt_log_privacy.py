"""Media operations receive their prompt without duplicating it into diagnostics."""

import json
import logging
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("editing", [False, True])
def test_image_request_preserves_prompt_without_logging_it(monkeypatch, caplog, editing):
    from tools import image_generation_tool as image

    monkeypatch.setattr(image, "fal_key_is_configured", lambda: True)
    model_id = "fal-ai/flux-2/klein/9b"
    meta = image.FAL_MODELS[model_id]
    prompt = "private product concept\nprivate continuation"
    sources = ["https://example.com/image.png"] if editing else []
    monkeypatch.setattr(image, "_resolve_fal_model", lambda: (model_id, meta))
    monkeypatch.setattr(image, "_dispatch_to_plugin_provider", lambda *a, **kw: None)
    monkeypatch.setattr(image, "_maybe_route_managed_krea", lambda *a, **kw: None)
    monkeypatch.setattr(image, "_confine_source_images", lambda url, refs, task: (url, refs, None))
    monkeypatch.setattr(image, "_debug", Mock())
    submit = Mock(return_value=object())
    monkeypatch.setattr(image, "_submit_fal_request", submit)
    monkeypatch.setattr(image, "_wait_fal_result", lambda handle: {"images": [{"url": "https://example.com/generated.png"}]})
    monkeypatch.setattr(image, "_postprocess_image_generate_result", lambda raw, **kw: raw)
    with caplog.at_level(logging.INFO, logger=image.__name__):
        result = image.registry.get_entry("image_generate").handler(
            {"prompt": prompt, "image_url": sources[0] if editing else None, "upscale": False})

    assert json.loads(result)["success"] is True
    assert submit.call_args.kwargs["arguments"]["prompt"] == prompt
    assert "private" not in caplog.text
    assert "chars=" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_analysis_keeps_prompt_at_stage_and_out_of_diagnostics(monkeypatch, caplog, tmp_path, kind):
    from tools import vision_tools as vision

    monkeypatch.setattr(vision, "_debug", Mock())
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    prompt = "private source explanation\nprivate continuation"
    media = tmp_path / "media.bin"
    media.write_bytes(b"fake media bytes")
    monkeypatch.setattr(vision, "_should_use_native_vision_fast_path", lambda: False)
    monkeypatch.setattr(vision, "_load_auxiliary_client", lambda: None)
    monkeypatch.setattr(vision, "_configured_aux_model", lambda *a: "test-model")
    monkeypatch.setattr(vision, "_aux_call_kwargs", lambda messages, *a, **kw: {"messages": messages})
    captured = []
    async def call(kwargs, *args):
        captured.append(kwargs["messages"])
        return "useful analysis"
    monkeypatch.setattr(vision, "_call_vision_llm", call)
    if kind == "image":
        monkeypatch.setattr(vision, "_prepare_image", AsyncMock(return_value=SimpleNamespace(
            path=media, size_bytes=10, mime="image/png", crop_offset=None)))
        monkeypatch.setattr(vision, "_run_encode_on_cpu_executor", AsyncMock(return_value="data:image/png;base64,eA=="))
        monkeypatch.setattr(vision, "async_call_llm", AsyncMock(return_value=object()), raising=False)
    else:
        monkeypatch.setattr(vision, "_materialize_video", AsyncMock(return_value=media))
        monkeypatch.setattr(vision, "_detect_video_mime_type", lambda path: "video/mp4")
        monkeypatch.setattr(vision, "_video_to_base64_data_url", lambda *a, **kw: "data:video/mp4;base64,eA==")
    with caplog.at_level(logging.INFO, logger=vision.__name__):
        entry = vision.registry.get_entry("vision_analyze" if kind == "image" else "video_analyze")
        result = await entry.handler({f"{kind}_url": "https://example.com/media", "question": prompt})

    sent_prompt = captured[0][0]["content"][0]["text"]
    assert prompt in sent_prompt
    assert json.loads(result)["analysis"] == "useful analysis"
    assert "private" not in caplog.text
    assert "chars=" in caplog.text
