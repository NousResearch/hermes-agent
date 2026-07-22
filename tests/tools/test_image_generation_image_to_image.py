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
import os
from typing import Any, Dict, List, Optional

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

    def test_edit_payload_strips_keys_outside_edit_supports(self):
        from tools.image_generation_tool import _build_fal_edit_payload

        # gpt-image-2 edit does NOT advertise image_size (auto-inferred), so
        # it must be stripped even though the text-to-image path sets it.
        payload = _build_fal_edit_payload(
            "fal-ai/gpt-image-2", "swap bg", ["https://x/y.png"], "square",
        )
        assert "image_size" not in payload
        assert payload["image_urls"] == ["https://x/y.png"]
        assert payload["quality"] == "medium"

    def test_text_only_model_has_no_edit_endpoint(self):
        from tools.image_generation_tool import FAL_MODELS

        # z-image/turbo is a pure text-to-image model — no edit endpoint.
        assert "edit_endpoint" not in FAL_MODELS["fal-ai/z-image/turbo"]
        # while nano-banana-pro is edit-capable
        assert FAL_MODELS["fal-ai/nano-banana-pro"].get("edit_endpoint")


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

        def fake_submit(endpoint, arguments, **kwargs):
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

        raw = image_tool.image_generate_tool(prompt="a cat", aspect_ratio="square")
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "text"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro"
        assert "image_urls" not in capture["arguments"]

    def test_image_to_image_routes_to_edit_endpoint(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)

        raw = image_tool.image_generate_tool(
            prompt="make it night",
            aspect_ratio="square",
            image_url="https://in/src.png",
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "image"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro/edit"
        assert capture["arguments"]["image_urls"] == ["https://in/src.png"]

    def test_reference_images_clamped_to_model_cap(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        # nano-banana-pro caps at 2 reference images.
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)

        raw = image_tool.image_generate_tool(
            prompt="blend",
            image_url="https://in/a.png",
            reference_image_urls=["https://in/b.png", "https://in/c.png", "https://in/d.png"],
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert capture["arguments"]["image_urls"] == ["https://in/a.png", "https://in/b.png"]

    def test_text_only_model_rejects_image_url(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/z-image/turbo"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)

        raw = image_tool.image_generate_tool(
            prompt="edit this", image_url="https://in/src.png",
        )
        out = json.loads(raw)
        assert out["success"] is False
        assert "image-to-image" in out["error"]
        # Must NOT have submitted anything.
        assert capture == {}

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

    def test_dispatch_text_only_when_no_image(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as reg

        provider = _EditCapableProvider()
        reg.register_provider(provider)
        monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "editcap")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)
        monkeypatch.setattr(reg, "get_provider", lambda n: provider if n == "editcap" else None)

        raw = image_tool._dispatch_to_plugin_provider("a dog", "landscape")
        out = json.loads(raw)
        assert out["success"] is True
        assert provider.received["image_url"] is None
        assert "reference_image_urls" not in provider.received or provider.received["reference_image_urls"] is None

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


class _PluginBothProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "both"

    def is_available(self) -> bool:
        return True

    def default_model(self) -> Optional[str]:
        return "both-v1"

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 5}

    def generate(self, prompt, aspect_ratio="landscape", *, image_url=None,
                 reference_image_urls=None, **kwargs):
        return {"success": True}


class TestDynamicSchema:
    def _no_discovery(self, monkeypatch):
        import hermes_cli.plugins as plugins_module
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **k: None)

    def test_fal_edit_model_advertises_both(self, cfg_home, monkeypatch):
        from tools.image_generation_tool import _build_dynamic_image_schema

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        desc = _build_dynamic_image_schema()["description"]
        assert "text-to-image" in desc and "image-to-image" in desc
        assert "routes automatically" in desc

    def test_fal_text_only_model_warns(self, cfg_home, monkeypatch):
        from tools.image_generation_tool import _build_dynamic_image_schema

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/z-image/turbo"}})
        desc = _build_dynamic_image_schema()["description"]
        assert "text-to-image only" in desc
        assert "NOT capable of image-to-image" in desc

    def test_plugin_both_provider_advertises_refs(self, cfg_home, monkeypatch):
        from tools.image_generation_tool import _build_dynamic_image_schema
        from agent import image_gen_registry as reg

        _write_cfg(cfg_home, {"image_gen": {"provider": "both"}})
        reg.register_provider(_PluginBothProvider())
        self._no_discovery(monkeypatch)

        desc = _build_dynamic_image_schema()["description"]
        assert "image-to-image / editing" in desc
        assert "up to 5 reference image(s)" in desc

    def test_builder_wired_into_registry(self):
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        entry = registry._tools["image_generate"]
        assert entry.dynamic_schema_overrides is not None
        out = entry.dynamic_schema_overrides()
        assert "description" in out


# ---------------------------------------------------------------------------
# Source-image resolution (local file / data URI uploads to FAL)
# ---------------------------------------------------------------------------


class _FakeFalClient:
    """Stand-in for the ``fal_client`` module recording direct-mode uploads.

    Managed mode no longer touches the client — the resolver builds the data
    URI itself from the sniffed bytes — so only ``upload_file`` is stubbed.
    """

    def __init__(self, *, upload_error: Optional[Exception] = None):
        self.uploaded: List[str] = []
        self._upload_error = upload_error

    def upload_file(self, path: str) -> str:
        if self._upload_error is not None:
            raise self._upload_error
        self.uploaded.append(path)
        return f"https://fal.storage/{os.path.basename(path)}"


_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-pixels"


def _data_uri(raw: bytes, mime: str = "image/png") -> str:
    import base64

    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


@pytest.fixture
def src_image(tmp_path):
    path = tmp_path / "source.png"
    path.write_bytes(_PNG_BYTES)
    return str(path)


@pytest.fixture(autouse=True)
def _restore_fal_module_globals():
    """Snapshot/restore the process-wide ``fal_client`` global + managed cache.

    ``_load_fal_client()`` short-circuits on any truthy ``fal_client``, so a
    real lazy import in one test would otherwise leak into a later test that
    forgot to patch it. Restoring makes ordering irrelevant.
    """
    import tools.image_generation_tool as image_tool

    saved = (
        image_tool.fal_client,
        image_tool._managed_fal_client,
        image_tool._managed_fal_client_config,
    )
    yield
    (
        image_tool.fal_client,
        image_tool._managed_fal_client,
        image_tool._managed_fal_client_config,
    ) = saved


def _local_png(path, marker: bytes = b"") -> str:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + marker)
    return str(path)


class TestFalSourceImageResolution:
    @pytest.mark.parametrize(
        "ref",
        [
            "https://example.com/a.png",
            "http://example.com/a.png",
            _data_uri(_PNG_BYTES),
            "  https://example.com/spaced.png  ",
        ],
    )
    def test_remote_and_data_refs_pass_through(self, monkeypatch, ref):
        import tools.image_generation_tool as image_tool

        # No fal_client access expected for passthrough refs (mode is irrelevant).
        monkeypatch.setattr(image_tool, "fal_client", None)
        assert image_tool._resolve_fal_source_image(ref, managed=True) == ref.strip()

    @pytest.mark.parametrize("managed", [False, True])
    def test_local_file_routes_by_mode(self, monkeypatch, src_image, managed):
        import tools.image_generation_tool as image_tool

        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        resolved = image_tool._resolve_fal_source_image(src_image, managed=managed)

        # Managed mode inlines a real data URI (built from sniffed bytes, not
        # the filename); direct mode uploads and never inlines.
        expected_resolved = (
            _data_uri(_PNG_BYTES)
            if managed
            else f"https://fal.storage/{os.path.basename(src_image)}"
        )
        assert {"resolved": resolved, "uploaded": fake.uploaded} == {
            "resolved": expected_resolved,
            "uploaded": [] if managed else [src_image],
        }

    @pytest.mark.parametrize("managed", [False, True])
    def test_denylisted_local_path_is_rejected(self, monkeypatch, tmp_path, managed):
        import tools.image_generation_tool as image_tool

        # A secret-bearing file on the read denylist must be refused even when
        # its bytes are a valid image — mirrors the file Read tool's guard.
        env_file = tmp_path / ".env"
        env_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        with pytest.raises(ValueError, match="Access denied"):
            image_tool._resolve_fal_source_image(str(env_file), managed=managed)
        assert fake.uploaded == []

    def test_non_image_local_file_is_rejected(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        # A readable-but-not-an-image path (e.g. a model-supplied secret) must
        # never be uploaded or inlined.
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\n")
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        for managed in (False, True):
            with pytest.raises(ValueError, match="not a recognised image file"):
                image_tool._resolve_fal_source_image(str(secret), managed=managed)
        assert fake.uploaded == []

    def test_extensionless_image_sniffs_correct_mime(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        # No extension, JPEG magic bytes — the data URI must reflect the bytes,
        # not the (absent) extension.
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 16
        blob = tmp_path / "attachment_blob"
        blob.write_bytes(jpeg)
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        resolved = image_tool._resolve_fal_source_image(str(blob), managed=True)
        assert resolved == _data_uri(jpeg, mime="image/jpeg")

    @pytest.mark.parametrize(
        "magic, expected_mime",
        [
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"\xff\xd8\xff\xe0", "image/jpeg"),
            (b"GIF89a", "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
            (b"BM\x00\x00", "image/bmp"),
            (b"\x00\x00\x00\x18ftypheic", "image/heic"),
        ],
    )
    def test_managed_inline_sniffs_each_image_type(
        self, monkeypatch, tmp_path, magic, expected_mime
    ):
        import tools.image_generation_tool as image_tool

        blob = tmp_path / "blob"
        blob.write_bytes(magic)
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        resolved = image_tool._resolve_fal_source_image(str(blob), managed=True)
        assert resolved == _data_uri(magic, mime=expected_mime)

    def test_non_image_data_uri_is_rejected(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        with pytest.raises(ValueError, match="not an image"):
            image_tool._resolve_fal_source_image(
                "data:text/plain;base64,aGVsbG8=", managed=True
            )

    @pytest.mark.parametrize("managed", [False, True])
    def test_data_uri_declaring_image_but_carrying_non_image_is_rejected(
        self, monkeypatch, managed
    ):
        import tools.image_generation_tool as image_tool

        # A secret base64'd under an image/png label must not slip through on
        # the strength of the declared MIME — the payload is sniffed too.
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        disguised = _data_uri(b"-----BEGIN OPENSSH PRIVATE KEY-----\n")
        assert disguised.startswith("data:image/png;base64,")
        with pytest.raises(ValueError, match="not a recognised image file"):
            image_tool._resolve_fal_source_image(disguised, managed=managed)

    def test_non_base64_data_uri_is_rejected(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        # Non-base64 image data URIs can't be byte-sniffed, so they aren't
        # trusted on their label alone.
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        with pytest.raises(ValueError, match="must be base64-encoded"):
            image_tool._resolve_fal_source_image(
                "data:image/png,not-base64-data", managed=True
            )

    def test_empty_base64_data_uri_is_rejected(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        with pytest.raises(ValueError, match="not a recognised image file"):
            image_tool._resolve_fal_source_image("data:image/png;base64,", managed=True)

    @pytest.mark.parametrize("lead", ["", "\n  \n"])
    def test_whitespace_in_base64_data_uri_is_accepted(self, monkeypatch, lead):
        import base64

        import tools.image_generation_tool as image_tool

        # Some encoders wrap base64 at 76 columns and may emit leading
        # whitespace; RFC 2397 permits it and it must not break the sniff.
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 80
        b64 = base64.b64encode(raw).decode("ascii")
        wrapped = lead + "\n".join(b64[i : i + 12] for i in range(0, len(b64), 12))
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        # A data: URI passes through unchanged once validated.
        ref = f"data:image/png;base64,{wrapped}"
        assert image_tool._resolve_fal_source_image(ref, managed=False) == ref

    def test_remote_file_uri_host_is_rejected(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        with pytest.raises(ValueError, match="Unsupported remote file:// host"):
            image_tool._resolve_fal_source_image(
                "file://server/share/photo.png", managed=False
            )

    def test_localhost_file_uri_is_resolved(self, monkeypatch, src_image):
        import tools.image_generation_tool as image_tool

        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        resolved = image_tool._resolve_fal_source_image(
            f"file://localhost{src_image}", managed=False
        )
        assert resolved == f"https://fal.storage/{os.path.basename(src_image)}"
        assert fake.uploaded == [src_image]

    def test_oversized_managed_inline_is_rejected(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "_MAX_INLINE_BASE64_BYTES", 8)
        big = tmp_path / "big.png"
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # > 8 bytes
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        with pytest.raises(ValueError, match="too large to send through the managed"):
            image_tool._resolve_fal_source_image(str(big), managed=True)

    def test_cap_is_measured_on_encoded_length(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        # 16 raw bytes encode (with the data: header) to ~46 chars. A cap that
        # sits between the raw and encoded sizes must reject — proving the check
        # is on the payload actually sent, not the raw file.
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # 16 bytes
        small = tmp_path / "small.png"
        small.write_bytes(raw)
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        encoded_len = len(_data_uri(raw))
        assert len(raw) < 30 < encoded_len  # guard the test's own premise
        monkeypatch.setattr(image_tool, "_MAX_INLINE_BASE64_BYTES", 30)

        with pytest.raises(ValueError, match="bytes encoded, max"):
            image_tool._resolve_fal_source_image(str(small), managed=True)

    def test_oversized_data_uri_is_capped_in_managed_mode(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "_MAX_INLINE_BASE64_BYTES", 32)
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        big_data_uri = _data_uri(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        # Managed mode caps an inlined data URI just like a local file...
        with pytest.raises(ValueError, match="too large to send through the managed"):
            image_tool._resolve_fal_source_image(big_data_uri, managed=True)
        # ...but direct mode passes it through (FAL fetches/accepts it).
        assert image_tool._resolve_fal_source_image(big_data_uri, managed=False) == big_data_uri

    def test_direct_mode_has_no_inline_size_cap(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        # The size cap is a managed-gateway concern; direct uploads stream from
        # disk, so a large source must still upload.
        monkeypatch.setattr(image_tool, "_MAX_INLINE_BASE64_BYTES", 8)
        big = tmp_path / "big.png"
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        resolved = image_tool._resolve_fal_source_image(str(big), managed=False)
        assert resolved == f"https://fal.storage/{big.name}"
        assert fake.uploaded == [str(big)]

    def test_file_uri_is_resolved(self, monkeypatch, src_image):
        import tools.image_generation_tool as image_tool

        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        resolved = image_tool._resolve_fal_source_image(
            f"file://{src_image}", managed=False
        )
        assert resolved == f"https://fal.storage/{os.path.basename(src_image)}"
        assert fake.uploaded == [src_image]

    def test_upload_error_propagates_in_direct_mode(self, monkeypatch, src_image):
        import tools.image_generation_tool as image_tool

        fake = _FakeFalClient(upload_error=RuntimeError("network down"))
        monkeypatch.setattr(image_tool, "fal_client", fake)

        # Direct mode must surface upload failures, not silently inline an
        # oversized data URI that then fails at submit.
        with pytest.raises(RuntimeError, match="network down"):
            image_tool._resolve_fal_source_image(src_image, managed=False)

    def test_tilde_path_is_expanded(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        home = tmp_path / "home"
        home.mkdir()
        _local_png(home / "pic.png")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))

        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        resolved = image_tool._resolve_fal_source_image("~/pic.png", managed=False)
        assert resolved == "https://fal.storage/pic.png"
        assert fake.uploaded == [str(home / "pic.png")]

    def test_empty_reference_raises(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        with pytest.raises(ValueError, match="Empty source image reference"):
            image_tool._resolve_fal_source_image("   ", managed=False)

    def test_missing_local_file_raises(self, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        with pytest.raises(ValueError, match="Source image not found"):
            image_tool._resolve_fal_source_image(str(tmp_path / "nope.png"), managed=False)

    def test_resolve_many_preserves_order(self, monkeypatch, src_image):
        import tools.image_generation_tool as image_tool

        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        resolved = image_tool._resolve_fal_source_images(
            ["https://example.com/a.png", src_image], managed=False
        )
        assert resolved == [
            "https://example.com/a.png",
            f"https://fal.storage/{os.path.basename(src_image)}",
        ]

    def test_resolve_many_empty_list(self, monkeypatch):
        import tools.image_generation_tool as image_tool

        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        assert image_tool._resolve_fal_source_images([], managed=False) == []


class TestFalRoutingWithLocalSources:
    """Drive ``image_generate_tool`` so resolution is exercised in context."""

    def _patch_submit(self, monkeypatch, image_tool, capture: dict):
        class _Handler:
            def get(self_inner):
                return {"images": [{"url": "https://out/img.png", "width": 1, "height": 1}]}

        def fake_submit(endpoint, arguments, **kwargs):
            capture["endpoint"] = endpoint
            capture["arguments"] = arguments
            return _Handler()

        monkeypatch.setattr(image_tool, "_submit_fal_request", fake_submit)
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)

    def test_local_image_url_is_uploaded_before_submit(self, cfg_home, monkeypatch, src_image):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        raw = image_tool.image_generate_tool(
            prompt="make it night",
            aspect_ratio="square",
            image_url=src_image,
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "image"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro/edit"
        hosted = f"https://fal.storage/{os.path.basename(src_image)}"
        assert capture["arguments"]["prompt"] == "make it night"
        assert capture["arguments"]["image_urls"] == [hosted]
        assert fake.uploaded == [src_image]

    @pytest.mark.parametrize("managed", [False, True])
    def test_mixed_local_and_remote_sources_preserve_order(
        self, cfg_home, monkeypatch, tmp_path, managed
    ):
        import tools.image_generation_tool as image_tool

        # nano-banana-pro caps at 2 references: a remote primary + one local ref.
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        if managed:
            monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: object())
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        local_bytes = b"\x89PNG\r\n\x1a\nref-pixels"
        local_ref = tmp_path / "ref.png"
        local_ref.write_bytes(local_bytes)
        raw = image_tool.image_generate_tool(
            prompt="blend",
            image_url="https://in/primary.png",
            reference_image_urls=[str(local_ref)],
        )
        out = json.loads(raw)
        assert out["success"] is True
        # The remote primary passes through; the local ref is uploaded (direct)
        # or inlined as a data URI (managed), and order is preserved.
        resolved_local = (
            _data_uri(local_bytes)
            if managed
            else f"https://fal.storage/{local_ref.name}"
        )
        assert capture["arguments"]["image_urls"] == [
            "https://in/primary.png",
            resolved_local,
        ]
        assert fake.uploaded == ([] if managed else [str(local_ref)])

    def test_reference_images_only_still_routes_to_edit(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        # No primary image_url — references alone must still trigger editing.
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        raw = image_tool.image_generate_tool(
            prompt="in this style",
            reference_image_urls=["https://in/ref.png"],
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert out["modality"] == "image"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro/edit"
        assert capture["arguments"]["image_urls"] == ["https://in/ref.png"]

    def test_clamped_reference_is_never_resolved(self, cfg_home, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        # nano-banana-pro caps at 2; the third source is clamped away and must
        # never be resolved. The dropped ref is a *present* file, so its absence
        # from fal.uploaded can only mean resolution never reached it.
        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        primary = _local_png(tmp_path / "a.png", b"A")
        kept_ref = _local_png(tmp_path / "b.png", b"B")
        dropped_ref = _local_png(tmp_path / "c.png", b"C")  # present, but clamped

        raw = image_tool.image_generate_tool(
            prompt="blend",
            image_url=primary,
            reference_image_urls=[kept_ref, dropped_ref],
        )
        out = json.loads(raw)
        assert out["success"] is True
        assert fake.uploaded == [primary, kept_ref]
        assert capture["arguments"]["image_urls"] == [
            f"https://fal.storage/{os.path.basename(primary)}",
            f"https://fal.storage/{os.path.basename(kept_ref)}",
        ]

    def test_data_uri_image_url_passes_through(self, cfg_home, monkeypatch):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)

        data_uri = _data_uri(b"\x89PNG\r\n\x1a\ninline")
        raw = image_tool.image_generate_tool(prompt="edit", image_url=data_uri)
        out = json.loads(raw)
        assert out["success"] is True
        assert capture["arguments"]["image_urls"] == [data_uri]
        # A data: URI is already submittable; nothing is uploaded.
        assert fake.uploaded == []

    def test_missing_local_image_url_fails_cleanly(self, cfg_home, monkeypatch, tmp_path):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())

        raw = image_tool.image_generate_tool(
            prompt="edit", image_url=str(tmp_path / "missing.png"),
        )
        out = json.loads(raw)
        assert out["success"] is False
        assert "Source image not found" in out["error"]
        # Resolution failed before any request was submitted.
        assert capture == {}

    def test_upload_error_surfaces_as_clean_failure(self, cfg_home, monkeypatch, src_image):
        import tools.image_generation_tool as image_tool

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})
        capture: dict = {}
        self._patch_submit(monkeypatch, image_tool, capture)
        monkeypatch.setattr(
            image_tool, "fal_client",
            _FakeFalClient(upload_error=RuntimeError("fal upload 500")),
        )

        raw = image_tool.image_generate_tool(prompt="edit", image_url=src_image)
        out = json.loads(raw)
        assert out["success"] is False
        assert "fal upload 500" in out["error"]
        # Upload failed during resolution, before submission.
        assert capture == {}


# ---------------------------------------------------------------------------
# End-to-end: a locally-attached image flows through the real registered
# handler into the edit request. Only the network boundary
# (``_submit_fal_request``) is stubbed.
# ---------------------------------------------------------------------------


class TestLocalAttachmentIntegration:
    def test_attached_local_image_reaches_edit_endpoint_as_data_uri(
        self, cfg_home, monkeypatch, tmp_path
    ):
        import base64

        import tools.image_generation_tool as image_tool

        # A user-attached image as it lands on disk after the gateway
        # downloads it from the chat platform (Matrix/Telegram/etc.).
        pixels = b"\x89PNG\r\n\x1a\n" + b"hermes-attachment-bytes" * 8
        attachment = tmp_path / "from_matrix.png"
        attachment.write_bytes(pixels)

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})

        # Managed-gateway (keyless) mode — the common Matrix deployment — so
        # resolution inlines the attachment as a data URI, no upload, no network.
        monkeypatch.setattr(image_tool, "fal_client", _FakeFalClient())
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: False)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: object())

        capture: dict = {}

        class _Handler:
            def get(self_inner):
                return {"images": [{"url": "https://out/edited.png", "width": 2, "height": 2}]}

        def fake_submit(endpoint, arguments, **kwargs):
            capture["endpoint"] = endpoint
            capture["arguments"] = arguments
            return _Handler()

        monkeypatch.setattr(image_tool, "_submit_fal_request", fake_submit)

        # Drive the exact entry point the agent calls.
        raw = image_tool._handle_image_generate({
            "prompt": "keep this character, make it night",
            "aspect_ratio": "square",
            "image_url": str(attachment),
        })
        out = json.loads(raw)

        assert out["success"] is True
        assert out["modality"] == "image"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro/edit"
        assert capture["arguments"]["image_urls"] == [_data_uri(pixels)]

        # The inlined payload round-trips back to the original pixels.
        _, b64 = capture["arguments"]["image_urls"][0].split(",", 1)
        assert base64.b64decode(b64) == pixels

    def test_attached_local_image_uploaded_in_direct_mode_via_handler(
        self, cfg_home, monkeypatch, tmp_path
    ):
        import tools.image_generation_tool as image_tool

        attachment = tmp_path / "from_telegram.png"
        attachment.write_bytes(b"\x89PNG\r\n\x1a\nbytes")

        _write_cfg(cfg_home, {"image_gen": {"model": "fal-ai/nano-banana-pro"}})

        fake = _FakeFalClient()
        monkeypatch.setattr(image_tool, "fal_client", fake)
        monkeypatch.setattr(image_tool, "fal_key_is_configured", lambda: True)
        monkeypatch.setattr(image_tool, "_resolve_managed_fal_gateway", lambda: None)

        capture: dict = {}

        class _Handler:
            def get(self_inner):
                return {"images": [{"url": "https://out/edited.png", "width": 2, "height": 2}]}

        def fake_submit(endpoint, arguments, **kwargs):
            capture["endpoint"] = endpoint
            capture["arguments"] = arguments
            return _Handler()

        monkeypatch.setattr(image_tool, "_submit_fal_request", fake_submit)

        raw = image_tool._handle_image_generate({
            "prompt": "make it night",
            "aspect_ratio": "square",
            "image_url": str(attachment),
        })
        out = json.loads(raw)

        assert out["success"] is True
        assert out["modality"] == "image"
        assert capture["endpoint"] == "fal-ai/nano-banana-pro/edit"
        assert capture["arguments"]["image_urls"] == [
            f"https://fal.storage/{attachment.name}"
        ]
        assert fake.uploaded == [str(attachment)]
