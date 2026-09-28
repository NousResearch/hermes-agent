"""Tests for the image-to-image / editing surface of ``image_generate``.

Mirrors the video-gen image-to-video tests: the unified ``image_generate``
tool routes to a provider's edit endpoint when ``image_url`` /
``reference_image_urls`` is supplied, otherwise to text-to-image. Coverage:

- In-tree FAL edit payload construction (``_build_fal_edit_payload``)
- In-tree FAL routing (text vs edit endpoint) via ``image_generate_tool``
- Plugin dispatch forwards image_url / reference_image_urls to ``generate()``
- ``capabilities()`` honesty drives the dynamic tool-schema description
- Models without an edit endpoint reject image inputs with a clear error
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
import yaml

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_cfg(home, cfg: dict):
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


# ---------------------------------------------------------------------------
# In-tree FAL edit payload + routing
# ---------------------------------------------------------------------------


class TestFalEditPayload:
    def test_edit_payload_includes_image_urls(self):
        from tools.image_generation_tool import _build_fal_edit_payload

        payload = _build_fal_edit_payload(
            "fal-ai/nano-banana-pro", "make it night", ["https://x/y.png"],
            "landscape",
        )
        assert payload["prompt"] == "make it night"
        assert payload["image_urls"] == ["https://x/y.png"]
        # nano-banana edit advertises aspect_ratio in edit_supports
        assert payload.get("aspect_ratio") == "16:9"


    def test_singular_edit_image_param_kling_image_v3(self):
        """Kling Image v3's i2i endpoint takes a SINGULAR `image_url` string;
        the catalog opts in via edit_image_param and only the first source
        image is sent — no `image_urls` list may leak into the payload."""
        from tools.image_generation_tool import _build_fal_edit_payload

        payload = _build_fal_edit_payload(
            "fal-ai/kling-image/v3/text-to-image", "make it winter",
            ["https://x/a.png", "https://x/b.png"], "landscape",
        )
        assert payload["prompt"] == "make it winter"
        assert payload["image_url"] == "https://x/a.png"
        assert "image_urls" not in payload
        assert payload.get("aspect_ratio") == "16:9"
        assert payload.get("resolution") == "2K"


class TestMandatoryKeysSurviveWhitelist:
    """A model whose whitelist forgets the mandatory keys must not produce a
    request with the prompt / source images silently stripped."""

    _SIZES = {"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"}

    def test_edit_keeps_prompt_and_image_urls(self, monkeypatch):
        from tools import image_generation_tool as t

        fake = {
            "size_style": "image_size_preset",
            "sizes": self._SIZES,
            "edit_supports": {"seed"},  # intentionally omits prompt + image_urls
        }
        monkeypatch.setitem(t.FAL_MODELS, "test/edit-model", fake)
        payload = t._build_fal_edit_payload(
            "test/edit-model", "make it blue", ["https://x/y.png"], "square",
        )
        assert payload["prompt"] == "make it blue"
        assert payload["image_urls"] == ["https://x/y.png"]

    def test_text_keeps_prompt(self, monkeypatch):
        from tools import image_generation_tool as t

        fake = {
            "size_style": "image_size_preset",
            "sizes": self._SIZES,
            "supports": {"seed"},  # intentionally omits prompt
        }
        monkeypatch.setitem(t.FAL_MODELS, "test/text-model", fake)
        payload = t._build_fal_payload("test/text-model", "a cat", aspect_ratio="square")
        assert payload["prompt"] == "a cat"


class TestFalRouting:
    def _patch_submit(self, monkeypatch, image_tool, capture: dict):
        class _Handler:
            def get(self_inner):
                return {"images": [{"url": "https://out/img.png", "width": 1, "height": 1}]}

        def fake_submit(endpoint, arguments):
            capture["endpoint"] = endpoint
            capture["arguments"] = arguments
            return _Handler()

        monkeypatch.setattr(image_tool, "_submit_fal_request", fake_submit)
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)

    def test_text_to_image_uses_base_endpoint(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)

        # Routing test — disable the (default-on) upscale pass so the captured
        # endpoint is the generation submit, not the upscaler.
        raw = image_tool.image_generate_tool(
            prompt="a cat", aspect_ratio="square", upscale=False,
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "text"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro"
        assert "image_urls" not in capture["arguments"]


    def test_edit_skips_upscaler(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        # flux-2-pro has upscale=True for text-to-image, but edits must skip it.
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/flux-2-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        upscale_called = {"hit": False}
        monkeypatch.setattr(
            image_tool, "_upscale_image",
            lambda *a, **k: upscale_called.__setitem__("hit", True) or None,
        )

        raw = image_tool.image_generate_tool(
            prompt="tweak", image_url="https://in/src.png",
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "image"
        assert capture["endpoint"] == image_tool.FAL_MODELS["fal-ai/flux-2-pro"]["edit_endpoint"]
        assert upscale_called["hit"] is False


# ---------------------------------------------------------------------------
# Plugin dispatch forwarding
# ---------------------------------------------------------------------------


class _EditCapableProvider(ImageGenProvider):
    def __init__(self):
        self.received: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "editcap"

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 4}

    def generate(self, prompt, aspect_ratio="landscape", *, image_url=None,
                 reference_image_urls=None, **kwargs):
        self.received = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "image_url": image_url,
            "reference_image_urls": reference_image_urls,
        }
        return {
            "success": True, "image": "/tmp/out.png", "model": "editcap-1",
            "prompt": prompt, "aspect_ratio": aspect_ratio,
            "modality": "image" if image_url else "text", "provider": "editcap",
        }


class _LegacyProvider(ImageGenProvider):
    """Provider whose generate() predates image_url (no **kwargs absorb)."""

    @property
    def name(self) -> str:
        return "legacy"

    def generate(self, prompt, aspect_ratio="landscape"):  # narrow signature
        return {"success": True, "image": "/tmp/legacy.png", "provider": "legacy"}


class TestPluginDispatchImageToImage:
    def test_dispatch_forwards_image_url(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as reg

        provider = _EditCapableProvider()
        reg.register_provider(provider)
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "editcap")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
        monkeypatch.setattr(reg, "get_provider", lambda n: provider if n == "editcap" else None)

        raw = image_tool._dispatch_to_plugin_provider(
            "make night", "square",
            image_url="https://in/src.png",
            reference_image_urls=["https://in/ref.png"],
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "image"
        assert provider.received["image_url"] == "https://in/src.png"
        assert provider.received["reference_image_urls"] == ["https://in/ref.png"]


    def test_legacy_provider_edit_request_surfaces_clear_error(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as reg

        provider = _LegacyProvider()
        reg.register_provider(provider)
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "legacy")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
        monkeypatch.setattr(reg, "get_provider", lambda n: provider if n == "legacy" else None)

        raw = image_tool._dispatch_to_plugin_provider(
            "edit it", "square", image_url="https://in/src.png",
        )
        out = json.loads(raw)
        assert out["success"] is False
        assert out["error_type"] == "modality_unsupported"


# ---------------------------------------------------------------------------
# Dynamic schema reflects active capabilities
# ---------------------------------------------------------------------------


class TestDynamicSchema:
    def _no_discovery(self, monkeypatch):
        import hermes_cli.plugins as plugins_module
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)


    def test_builder_wired_into_registry(self):
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        entry = registry._tools["image_generate"]
        assert entry.dynamic_schema_overrides is not None
        out = entry.dynamic_schema_overrides()
        assert "description" in out
