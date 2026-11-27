"""Coverage tests for the error / edge paths of ``video_generation_tool``.

Companion to the four existing video-gen test files. This suite deliberately
hits only the branches the dispatch/schema/surface tests leave untouched —
config-load failures, availability resolution edge cases, coercion
normalization, handler rejections, provider-contract errors, the dynamic
schema's fault-tolerance fallbacks, and caveat formatting.

Every test asserts a *behavior contract* (what the function returns or does),
never a snapshot of an unrelated constant.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from agent import video_gen_registry
from agent.video_gen_provider import VideoGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


# ---------------------------------------------------------------------------
# Fixture helpers — stub the config readers / discovery so handler and schema
# tests never touch the real config.yaml or discover user plugins.
# ---------------------------------------------------------------------------


def _stub_handler_env(monkeypatch, *, configured: Optional[str] = "fake",
                      provider: Optional[Any] = None):
    """Stub config reads + plugin discovery + source confinement.

    ``_confine_source_images`` is re-imported fresh inside the handler, so it
    is patched on its owning module, not the consumer.
    """
    import hermes_cli.plugins as plugins_module
    import tools.image_generation_tool as igt
    from tools import video_generation_tool as vgt

    monkeypatch.setattr(vgt, "_read_configured_video_provider", lambda: configured)
    monkeypatch.setattr(vgt, "_read_configured_video_model", lambda: None)
    monkeypatch.setattr(
        igt, "_confine_source_images",
        lambda image_url, refs, task_id: (image_url, refs, None),
    )
    monkeypatch.setattr(vgt, "_resolve_active_provider", lambda: provider)
    monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# Config readers
# ---------------------------------------------------------------------------


class TestReadVideoGenSection:
    def test_config_load_exception_returns_empty(self, monkeypatch):
        import hermes_cli.config as cfg_mod
        from tools import video_generation_tool as vgt

        def _boom():
            raise ValueError("config corrupt")

        monkeypatch.setattr(cfg_mod, "load_config", _boom)
        assert vgt._read_video_gen_section() == {}

    def test_non_dict_video_gen_section_returns_empty(self, monkeypatch):
        import hermes_cli.config as cfg_mod
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(cfg_mod, "load_config", lambda: {"video_gen": "not-a-dict"})
        assert vgt._read_video_gen_section() == {}

    def test_dict_section_returned_verbatim(self, monkeypatch):
        import hermes_cli.config as cfg_mod
        from tools import video_generation_tool as vgt

        section = {"provider": "xai", "model": "m1"}
        monkeypatch.setattr(cfg_mod, "load_config", lambda: {"video_gen": section})
        assert vgt._read_video_gen_section() is section


class TestReadConfiguredProvider:
    @pytest.mark.parametrize("section", [
        {"provider": ""},
        {"provider": "   "},
        {"provider": None},
        {"provider": 123},
        {},
    ])
    def test_unconfigured_returns_none(self, monkeypatch, section):
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(vgt, "_read_video_gen_section", lambda: section)
        assert vgt._read_configured_video_provider() is None

    def test_stripped_value_returned(self, monkeypatch):
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(vgt, "_read_video_gen_section", lambda: {"provider": "  fal  "})
        assert vgt._read_configured_video_provider() == "fal"


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


class _Avail:
    def __init__(self, available: bool, name: str = "p"):
        self._available = available
        self.name = name

    def is_available(self) -> bool:
        return self._available


class _AvailRaises:
    name = "raises"

    def is_available(self) -> bool:
        raise RuntimeError("doomed")


class TestCheckRequirements:
    def test_available_provider_returns_true(self, monkeypatch):
        from agent import video_gen_registry as reg
        import hermes_cli.plugins as plugins_module
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(reg, "list_providers", lambda: [_Avail(True, "ok")])
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *_a, **_k: None)
        assert vgt.check_video_generation_requirements() is True

    def test_raising_provider_is_skipped(self, monkeypatch):
        """A provider whose is_available() raises must not stop the scan."""
        from agent import video_gen_registry as reg
        import hermes_cli.plugins as plugins_module
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(
            reg, "list_providers",
            lambda: [_AvailRaises(), _Avail(True, "ok")],
        )
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *_a, **_k: None)
        assert vgt.check_video_generation_requirements() is True

    def test_no_providers_returns_false(self, monkeypatch):
        from agent import video_gen_registry as reg
        import hermes_cli.plugins as plugins_module
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(reg, "list_providers", lambda: [])
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *_a, **_k: None)
        assert vgt.check_video_generation_requirements() is False

    def test_discovery_failure_returns_false(self, monkeypatch):
        """Outer except: plugin discovery (or import) failing is non-fatal."""
        from agent import video_gen_registry as reg
        import hermes_cli.plugins as plugins_module
        from tools import video_generation_tool as vgt

        def _boom(*_a, **_k):
            raise RuntimeError("discovery blew up")

        monkeypatch.setattr(reg, "list_providers", lambda: [_Avail(True, "ok")])
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", _boom)
        assert vgt.check_video_generation_requirements() is False


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class TestResolveActiveProvider:
    def test_active_provider_raise_returns_none(self, monkeypatch):
        from agent import video_gen_registry as reg
        import hermes_cli.plugins as plugins_module
        from tools import video_generation_tool as vgt

        def _boom(*_a, **_k):
            raise RuntimeError("get_active_provider blew up")

        monkeypatch.setattr(reg, "get_active_provider", _boom)
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *_a, **_k: None)
        assert vgt._resolve_active_provider() is None


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


class TestCoerceInt:
    @pytest.mark.parametrize("value", [None, "", "abc", "12.5", [1], {"a": 1}])
    def test_invalid_returns_none(self, value):
        from tools import video_generation_tool as vgt

        assert vgt._coerce_int(value) is None

    @pytest.mark.parametrize("value,expected", [("12", 12), (12, 12), (12.9, 12), ("007", 7)])
    def test_valid_returns_int(self, value, expected):
        from tools import video_generation_tool as vgt

        assert vgt._coerce_int(value) == expected


class TestCoerceBool:
    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", "TRUE", "On"])
    def test_truthy_variants(self, value):
        from tools import video_generation_tool as vgt

        assert vgt._coerce_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", "Off"])
    def test_falsy_variants(self, value):
        from tools import video_generation_tool as vgt

        assert vgt._coerce_bool(value) is False

    @pytest.mark.parametrize("value", ["garbage", "maybe", 1, [], {}])
    def test_unknown_returns_none(self, value):
        from tools import video_generation_tool as vgt

        assert vgt._coerce_bool(value) is None

    def test_none_and_bool_passthrough(self):
        from tools import video_generation_tool as vgt

        assert vgt._coerce_bool(None) is None
        assert vgt._coerce_bool(True) is True
        assert vgt._coerce_bool(False) is False


class TestNormalizeReferenceImages:
    def test_none_returns_none(self):
        from tools import video_generation_tool as vgt

        assert vgt._normalize_reference_images(None) is None

    def test_string_wrapped_into_list(self):
        from tools import video_generation_tool as vgt

        assert vgt._normalize_reference_images("  a.png  ") == ["a.png"]

    def test_filters_and_strips(self):
        from tools import video_generation_tool as vgt

        assert vgt._normalize_reference_images(
            [" a.png ", "", "  b.png  ", 5, None]
        ) == ["a.png", "b.png"]

    def test_empty_result_returns_none(self):
        from tools import video_generation_tool as vgt

        assert vgt._normalize_reference_images(["", "   "]) is None
        assert vgt._normalize_reference_images([5]) is None

    def test_non_sequence_returns_none(self):
        from tools import video_generation_tool as vgt

        assert vgt._normalize_reference_images(5) is None
        assert vgt._normalize_reference_images({"a": "b"}) is None

    def test_tuple_accepted(self):
        from tools import video_generation_tool as vgt

        assert vgt._normalize_reference_images(("x.png", "y.png")) == ["x.png", "y.png"]


# ---------------------------------------------------------------------------
# Handler contracts
# ---------------------------------------------------------------------------


class TestHandlerErrors:
    def test_empty_prompt_returns_required_error(self, monkeypatch):
        from tools import video_generation_tool as vgt

        _stub_handler_env(monkeypatch, configured="fake")
        raw = vgt._handle_video_generate({"prompt": "   "})
        assert json.loads(raw)["error"] == "prompt is required for video generation"

    @pytest.mark.parametrize("extra", [
        {"operation": "extend"},
        {"video_url": "https://example.com/v.mp4"},
    ])
    def test_edit_extend_rejected(self, monkeypatch, extra):
        from tools import video_generation_tool as vgt

        _stub_handler_env(monkeypatch, configured="fake")
        args = {"prompt": "a dog", **extra}
        raw = vgt._handle_video_generate(args)
        payload = json.loads(raw)
        assert "video edit/extend" in payload["error"]

    def test_confine_error_returned_verbatim(self, monkeypatch):
        import tools.image_generation_tool as igt
        from tools import video_generation_tool as vgt

        _stub_handler_env(monkeypatch, configured="fake")
        error_json = json.dumps({
            "success": False, "image": None,
            "error": "Could not read source image: denied",
            "error_type": "ImageResolutionError",
        })
        # Confinement runs before the prompt check; the error is returned
        # verbatim for an otherwise-valid call (valid prompt, no forbidden args).
        monkeypatch.setattr(
            igt, "_confine_source_images",
            lambda image_url, refs, task_id: (image_url, refs, error_json),
        )
        assert vgt._handle_video_generate({"prompt": "a dog"}) == error_json


class _NarrowSignatureProvider(VideoGenProvider):
    """generate() did not widen to accept kwargs → TypeError on dispatch."""

    @property
    def name(self) -> str:
        return "narrow"

    def default_model(self) -> Optional[str]:
        return "m-narrow"

    def generate(self, prompt):
        return {"success": True}


class _NonDictProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "nondict"

    def default_model(self) -> Optional[str]:
        return "m-nondict"

    def generate(self, prompt, **kwargs):
        return "not a dict"


class _GenericRaisingProvider(VideoGenProvider):
    """generate() raises a non-TypeError → provider_exception contract."""

    @property
    def name(self) -> str:
        return "boom"

    def default_model(self) -> Optional[str]:
        return "m-boom"

    def generate(self, prompt, **kwargs):
        raise RuntimeError("exploded")


class TestHandlerProviderErrors:
    def test_narrow_signature_is_provider_contract_error(self, monkeypatch):
        from tools import video_generation_tool as vgt

        _stub_handler_env(monkeypatch, configured="narrow", provider=_NarrowSignatureProvider())
        raw = vgt._handle_video_generate({"prompt": "a dog"})
        payload = json.loads(raw)
        assert payload["success"] is False
        assert payload["error_type"] == "provider_contract"
        assert payload["provider"] == "narrow"
        assert "signature is out of date" in payload["error"]

    def test_non_dict_result_is_provider_contract_error(self, monkeypatch):
        from tools import video_generation_tool as vgt

        _stub_handler_env(monkeypatch, configured="nondict", provider=_NonDictProvider())
        raw = vgt._handle_video_generate({"prompt": "a dog"})
        payload = json.loads(raw)
        assert payload["success"] is False
        assert payload["error_type"] == "provider_contract"
        assert payload["provider"] == "nondict"
        assert payload["error"] == "Provider returned a non-dict result"

    def test_generic_provider_exception_is_provider_exception(self, monkeypatch):
        from tools import video_generation_tool as vgt

        _stub_handler_env(monkeypatch, configured="boom", provider=_GenericRaisingProvider())
        raw = vgt._handle_video_generate({"prompt": "a dog"})
        payload = json.loads(raw)
        assert payload["success"] is False
        assert payload["error_type"] == "provider_exception"
        assert payload["provider"] == "boom"
        assert "exploded" in payload["error"]


# ---------------------------------------------------------------------------
# Dynamic schema — fault tolerance + caveat formatting
# ---------------------------------------------------------------------------


class _FragileProvider(VideoGenProvider):
    def __init__(self, *, name: str = "fragile", raise_caps: bool = False,
                 raise_models: bool = False, models: Optional[List[Dict[str, Any]]] = None,
                 caps: Optional[Dict[str, Any]] = None):
        self._name = name
        self._raise_caps = raise_caps
        self._raise_models = raise_models
        self._models = models if models is not None else [{"id": "m1", "modalities": ["text"]}]
        self._caps = caps if caps is not None else {"modalities": ["text"]}

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> Dict[str, Any]:
        if self._raise_caps:
            raise RuntimeError("caps boom")
        return self._caps

    def list_models(self) -> List[Dict[str, Any]]:
        if self._raise_models:
            raise RuntimeError("models boom")
        return self._models

    def default_model(self) -> Optional[str]:
        entry = self._models[0] if self._models else {}
        return entry.get("id", "m1")

    def generate(self, prompt, **kwargs):
        return {"success": True}


class TestDynamicSchemaFallbacks:
    def _schema_with(self, monkeypatch, provider):
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(vgt, "_read_configured_video_model", lambda: None)
        monkeypatch.setattr(vgt, "_resolve_active_provider", lambda: provider)
        return vgt._build_dynamic_video_schema()

    def test_capabilities_raise_falls_back_to_empty(self, monkeypatch):
        schema = self._schema_with(monkeypatch, _FragileProvider(raise_caps=True))
        # caps == {} → no capability-gated axes advertised; static axes remain.
        props = schema["parameters"]["properties"]
        assert "text-to-video only" in schema["description"]
        assert "image_url" not in props
        assert {"prompt", "aspect_ratio", "duration", "model", "resolution"} <= set(props)

    def test_list_models_raise_falls_back_to_empty(self, monkeypatch):
        schema = self._schema_with(monkeypatch, _FragileProvider(raise_models=True))
        # models == [] → model_meta {} → whole surface driven by caps.
        assert "text-to-video only" in schema["description"]
        assert "prompt" in schema["parameters"]["properties"]

    def test_xai_branch_surfaces_chaining_and_storage_notices(self, monkeypatch):
        # The model entry carries a singular 'modality' key (FAL-style single-
        # modality catalog entries) → exercises the dynamic build's
        # model_modalities.add() path; the provider is named 'xai' so the
        # chaining + storage notice branch also runs.
        # Stub the storage notice so the test is hermetic (does not read the
        # host's live config.yaml, where storage may be disabled).
        import tools.xai_http as xai_http

        monkeypatch.setattr(
            xai_http, "xai_storage_notice_text", lambda *_a, **_k: "storage ok"
        )
        schema = self._schema_with(
            monkeypatch,
            _FragileProvider(
                name="xai",
                caps={"modalities": ["text", "image"]},
                models=[{"id": "m-xai", "modality": "image"}],
            ),
        )
        desc = schema["description"]
        assert "image-to-video only" in desc
        assert "chaining" in desc
        assert "storage: storage ok" in desc

    def test_xai_notice_fetch_failure_is_tolerated(self, monkeypatch):
        """If xai_storage_notice_text raises, the xai branch drops the notice
        line instead of failing the whole schema build."""
        import tools.xai_http as xai_http

        def _boom(*_a, **_k):
            raise RuntimeError("no storage module")

        monkeypatch.setattr(xai_http, "xai_storage_notice_text", _boom)
        schema = self._schema_with(
            monkeypatch,
            _FragileProvider(
                name="xai",
                caps={"modalities": ["text"]},
                models=[{"id": "m-xai", "modality": "image"}],
            ),
        )
        desc = schema["description"]
        assert "chaining" in desc
        assert "storage:" not in desc

    def test_no_provider_still_builds(self, monkeypatch):
        from tools import video_generation_tool as vgt

        monkeypatch.setattr(vgt, "_read_configured_video_model", lambda: None)
        monkeypatch.setattr(vgt, "_resolve_active_provider", lambda: None)
        schema = vgt._build_dynamic_video_schema()
        assert "No video backend is available" in schema["description"]
        assert sorted(schema["parameters"]["properties"]) == ["prompt"]


class TestFormatModelCaveats:
    def test_fal_singular_modality_key_adds_image_only(self):
        from tools import video_generation_tool as vgt

        caveats = vgt._format_model_caveats({"modality": "image"}, {})
        assert any("image-to-video only" in c for c in caveats)

    def test_text_only_model(self):
        from tools import video_generation_tool as vgt

        caveats = vgt._format_model_caveats({"modalities": ["text"]}, {})
        assert any("text-to-video only" in c for c in caveats)

    def test_dual_modality_no_caveat(self):
        from tools import video_generation_tool as vgt

        assert vgt._format_model_caveats({"modalities": ["text", "image"]}, {}) == []
