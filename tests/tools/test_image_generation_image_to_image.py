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

import base64
import json
from typing import Any, Dict

import pytest
import hermes_yaml as yaml

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


class TestLocalSourceConsent:
    @pytest.mark.parametrize("changed", ["plugin", "model", "gateway", "unrelated"])
    def test_approval_binds_selected_destination(self, cfg_home, monkeypatch, tmp_path, changed):
        import tools.image_generation_tool as image_tool
        from tools import approval_prompt

        source = tmp_path / "private.png"
        source.write_bytes(base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
        initial = {"provider": "openai" if changed == "plugin" else "fal",
                   "model": "fal-ai/nano-banana-pro"}
        _write_cfg(cfg_home, {"image_gen": initial})
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)

        class Gateway:
            gateway_origin = "https://managed-a.example/fal-queue"

        gateway = Gateway()
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway",
                            lambda: gateway if changed == "gateway" else None)
        sent = []
        monkeypatch.setattr(image_tool, "_dispatch_to_plugin_provider",
                            lambda *a, **k: sent.append("plugin") or image_tool._provider_error("sent", "provider_exception"))
        monkeypatch.setattr(image_tool, "_submit_fal_request",
                            lambda *a, **k: sent.append("fal") or object())
        monkeypatch.setattr(image_tool, "_wait_fal_result",
                            lambda handle: {"images": [{"url": "https://out.example/image.png"}]})

        class FakeFal:
            def upload(self, data, mime):
                sent.append("upload")
                return "https://fal.storage/private.png"

        monkeypatch.setattr(image_tool, "fal_client", FakeFal())

        def approve(message, description, **kwargs):
            if changed == "plugin":
                _write_cfg(cfg_home, {"image_gen": {**initial, "provider": "xai"}})
            elif changed == "model":
                _write_cfg(cfg_home, {"image_gen": {**initial, "model": "fal-ai/flux-2-pro"}})
            elif changed == "gateway":
                gateway.gateway_origin = "https://managed-b.example/fal-queue"
            else:
                _write_cfg(cfg_home, {"image_gen": initial, "terminal": {"theme": "dark"}})
            return "accept"

        monkeypatch.setattr(approval_prompt, "request_elicitation_consent", approve)

        result = json.loads(image_tool._handle_image_generate({
            "prompt": "make it night", "image_url": str(source), "upscale": False,
        }))

        if changed == "unrelated":
            assert result["success"] is True
            assert sent == ["upload", "fal"]
        else:
            assert result["error_type"] == "source_export_destination_changed"
            assert sent == []

    def test_approval_binds_canonical_path_and_validated_bytes(self, cfg_home, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool
        from tools import approval_prompt
        from PIL import Image

        original = tmp_path / "original.png"
        replacement = tmp_path / "replacement.png"
        link = tmp_path / "selected.png"
        Image.new("RGB", (1, 1), "red").save(original)
        Image.new("RGB", (1, 1), "blue").save(replacement)
        original_bytes = original.read_bytes()
        link.symlink_to(original)
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)

        prompts = []

        def approve(message, description, **kwargs):
            prompts.append(message)
            link.unlink()
            link.symlink_to(replacement)
            return "accept"

        monkeypatch.setattr(approval_prompt, "request_elicitation_consent", approve)

        class FakeFal:
            def __init__(self):
                self.uploads = []

            def upload(self, data, mime):
                self.uploads.append((data, mime))
                return "https://fal.storage/selected.png"

        fake = FakeFal()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        class Handler:
            def get(self):
                return {"images": [{"url": "https://out/edited.png", "width": 1, "height": 1}]}

        monkeypatch.setattr(image_tool, "_submit_fal_request", lambda endpoint, arguments, **kwargs: Handler())

        result = json.loads(image_tool._handle_image_generate({
            "prompt": "make it night", "image_url": str(link), "upscale": False,
        }))

        assert result["success"] is True
        assert str(original) in prompts[0]
        assert fake.uploads == [(original_bytes, "image/png")]

    @pytest.mark.parametrize("entry,answer,error_type", [
        ("handler", "decline", "source_export_denied"),
        ("handler", "cancel", "source_export_denied"),
        ("direct", "decline", "source_export_denied"),
        ("direct", "cancel", "source_export_denied"),
        ("plugin_fallback", "accept", "source_export_destination_changed"),
    ])
    def test_local_export_requires_approved_destination(
        self, cfg_home, monkeypatch, tmp_path, entry, answer, error_type
    ):
        import tools.image_generation_tool as image_tool
        from tools import approval_prompt

        source = tmp_path / "private.png"
        source.write_bytes(base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
        provider = "broken" if entry == "plugin_fallback" else "fal"
        _write_cfg(cfg_home, {"image_gen": {"provider": provider, "model": "fal-ai/nano-banana-pro"}})
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)
        monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *a, **k: answer)
        if answer != "accept":
            monkeypatch.setattr(image_tool, "resolve_canonical_source_sync",
                                lambda *a, **k: pytest.fail("read before consent"))
        monkeypatch.setattr(image_tool, "_submit_fal_request", lambda *a, **k: pytest.fail("submitted after denial"))
        if entry == "plugin_fallback":
            monkeypatch.setattr(image_tool, "_get_plugin_provider", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing")))

        if entry != "direct":
            raw = image_tool._handle_image_generate({
                "prompt": "make it night", "image_url": str(source), "upscale": False,
            })
        else:
            raw = image_tool.image_generate_tool(
                prompt="make it night", image_url=str(source), upscale=False)
        result = json.loads(raw)

        assert result["success"] is False
        assert result["error_type"] == error_type

    @pytest.mark.parametrize("managed,source_kind", [
        (False, "local"), (True, "local"), (False, "remote"),
        (False, "data"), (False, "mislabelled_data"),
    ])
    def test_sources_reach_selected_fal_endpoint_with_local_consent(
        self, cfg_home, monkeypatch, tmp_path, managed, source_kind
    ):
        import tools.image_generation_tool as image_tool
        from tools import approval_prompt

        source = tmp_path / "attached.png"
        pixels = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
        source.write_bytes(pixels)
        data_url = f"data:image/png;base64,{base64.b64encode(pixels).decode('ascii')}"
        source_ref = (str(source) if source_kind == "local" else
                      "https://example.com/attached.png" if source_kind == "remote" else
                      data_url.replace("image/png", "image/jpeg") if source_kind == "mislabelled_data" else
                      data_url)
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: not managed)

        class Gateway:
            gateway_origin = "https://nous.example/fal-queue"

        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway",
                            lambda: Gateway() if managed else None)
        approval = []
        monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                            lambda message, description, **kwargs: approval.append((message, description)) or "accept")

        class FakeFal:
            def __init__(self):
                self.uploads = []

            def upload(self, data, mime):
                self.uploads.append((data, mime))
                return "https://fal.storage/attached.png"

        fake = FakeFal()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        class Handler:
            def get(self):
                return {"images": [{"url": "https://out/edited.png", "width": 1, "height": 1}]}

        submitted = []
        monkeypatch.setattr(image_tool, "_submit_fal_request",
                            lambda endpoint, arguments, **kwargs: submitted.append((endpoint, arguments)) or Handler())

        result = json.loads(image_tool._handle_image_generate({
            "prompt": "make it night", "image_url": source_ref, "upscale": False,
        }))

        assert result["success"] is True
        expected_source = ((data_url if managed else "https://fal.storage/attached.png")
                           if source_kind == "local" else data_url if source_kind == "mislabelled_data"
                           else source_ref)
        assert fake.uploads == ([(pixels, "image/png")] if source_kind == "local" and not managed else [])
        expected_payload = image_tool._build_fal_edit_payload(
            "fal-ai/nano-banana-pro", "make it night", [expected_source])
        assert submitted == [("fal-ai/nano-banana-pro/edit", expected_payload)]
        if source_kind == "local":
            assert len(approval) == 1
            assert str(source) in approval[0][0]
            assert ("Nous managed FAL gateway" if managed else "FAL.ai storage") in approval[0][0]
            assert "fal-ai/nano-banana-pro/edit" in approval[0][0]
        else:
            assert approval == []


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
