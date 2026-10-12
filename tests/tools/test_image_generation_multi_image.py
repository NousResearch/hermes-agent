"""FAL multi-image generation: ``image_gen.fal.num_images`` asks for N images and every one comes back.

FAL bills each image a request returns; the tool used to keep ``images[0]`` and drop the rest.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def image_tool(monkeypatch):
    import importlib

    import tools.image_generation_tool as mod
    mod = importlib.reload(mod)
    monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux-2-pro")
    monkeypatch.setattr(mod, "fal_key_is_configured", lambda: True)
    monkeypatch.setattr(mod, "_resolve_managed_fal_gateway", lambda: None)
    return mod


class _FakeHandle:
    def __init__(self, result):
        self._result = result

    def get(self):
        return self._result


def _generate(image_tool, monkeypatch, *, config, urls):
    sent = {}

    def _submit(endpoint, arguments=None):
        sent.update(arguments or {})
        return _FakeHandle({"images": [{"url": u, "width": 1024, "height": 768} for u in urls]})

    monkeypatch.setattr(image_tool, "_submit_fal_request", _submit)
    with patch("hermes_cli.config.load_config", return_value=config):
        out = json.loads(image_tool.image_generate_tool("a cat"))
    return out, sent


def test_configured_num_images_is_requested_and_every_image_returned(image_tool, monkeypatch):
    urls = ["https://fal/a.png", "https://fal/b.png", "https://fal/c.png"]
    out, sent = _generate(image_tool, monkeypatch, config={"image_gen": {"fal": {"num_images": 3}}}, urls=urls)

    assert sent["num_images"] == 3
    assert out["success"] is True
    assert out["image"] == "https://fal/a.png"
    assert out["additional_images"] == ["https://fal/b.png", "https://fal/c.png"]


def test_unset_num_images_keeps_the_single_image_default(image_tool, monkeypatch):
    out, sent = _generate(image_tool, monkeypatch, config={}, urls=["https://fal/a.png"])

    assert sent["num_images"] == 1
    assert out["image"] == "https://fal/a.png"
    assert "additional_images" not in out


def test_fal_plugin_passes_every_image_through(image_tool, monkeypatch):
    """``image_gen.provider: fal`` routes through the plugin, which reshapes the tool's JSON."""
    from plugins.image_gen.fal import FalImageGenProvider

    urls = ["https://fal/a.png", "https://fal/b.png"]
    monkeypatch.setattr(image_tool, "_submit_fal_request", lambda endpoint, arguments=None: _FakeHandle(
        {"images": [{"url": u} for u in urls]}))
    with patch("hermes_cli.config.load_config", return_value={"image_gen": {"fal": {"num_images": 2}}}):
        result = FalImageGenProvider().generate(prompt="a cat")

    assert result["image"] == "https://fal/a.png"
    assert result["additional_images"] == ["https://fal/b.png"]
