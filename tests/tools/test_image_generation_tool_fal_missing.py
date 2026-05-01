from __future__ import annotations

import importlib
import sys

import pytest


def test_image_generation_tool_imports_without_fal_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "fal_client", None)

    import tools.image_generation_tool as image_generation_tool

    reloaded = importlib.reload(image_generation_tool)

    assert reloaded.fal_client is None


@pytest.mark.parametrize(
    ("call_name", "args"),
    [
        ("_submit_fal_request", ("fal-ai/flux-2-pro", {"prompt": "x"})),
        ("_upscale_image", ("https://example.com/image.png", "draw a cat")),
    ],
)
def test_fal_code_paths_raise_importerror_with_guidance(monkeypatch, call_name, args):
    monkeypatch.setitem(sys.modules, "fal_client", None)

    import tools.image_generation_tool as image_generation_tool

    reloaded = importlib.reload(image_generation_tool)

    with pytest.raises(ImportError) as exc_info:
        getattr(reloaded, call_name)(*args)

    message = str(exc_info.value)
    assert "fal_client is not installed" in message
    assert "pip install fal-client" in message
