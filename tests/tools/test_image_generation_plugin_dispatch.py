from __future__ import annotations

import json
import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class _FakeCodexProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {
            "success": True,
            "image": "/tmp/codex-test.png",
            "model": "gpt-5.2-codex",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "codex",
        }


class TestPluginDispatch:
    def test_dispatch_routes_to_codex_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        image_gen_registry.register_provider(_FakeCodexProvider())

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: _FakeCodexProvider() if name == "codex" else None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["image"] == "/tmp/codex-test.png"
        assert payload["aspect_ratio"] == "square"

    def test_dispatch_reports_missing_registered_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: missing-codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "missing-codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "landscape")
        payload = json.loads(dispatched)

        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert "image_gen.provider='missing-codex'" in payload["error"]

    def test_dispatch_force_refreshes_plugins_when_provider_initially_missing(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from hermes_cli import plugins as plugins_module
        from agent import image_gen_registry as registry_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")

        calls = []
        provider_state = {"provider": None}

        def fake_ensure_plugins_discovered(force=False):
            calls.append(force)
            if force:
                provider_state["provider"] = _FakeCodexProvider()

        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", fake_ensure_plugins_discovered)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: provider_state["provider"])

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw hammy", "portrait")
        payload = json.loads(dispatched)

        assert calls == [False, True]
        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["aspect_ratio"] == "portrait"

    def test_dispatch_forwards_reference_images_to_provider(self, monkeypatch, tmp_path):
        """When the schema-level ``reference_images`` arg is non-empty, it must
        flow through the dispatcher and land in the provider's ``generate()``
        kwargs. Providers that do not implement reference-image support absorb
        the kwarg via ``**kwargs`` — this test asserts the wiring, not the
        downstream behavior.
        """
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        captured = {}

        class _CaptureProvider(ImageGenProvider):
            @property
            def name(self) -> str:
                return "capture"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                captured.update({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
                return {
                    "success": True,
                    "image": "/tmp/capture.png",
                    "model": "test-model",
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "provider": "capture",
                }

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: capture\n")
        image_gen_registry.register_provider(_CaptureProvider())
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "capture")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **kw: None)
        monkeypatch.setattr(
            registry_module, "get_provider",
            lambda name: _CaptureProvider() if name == "capture" else None,
        )

        refs = ["/tmp/source-a.png", "https://example.com/source-b.png"]
        dispatched = image_generation_tool._dispatch_to_plugin_provider(
            "blend these",
            "square",
            reference_images=refs,
        )
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert captured["reference_images"] == refs

    def test_dispatch_omits_reference_images_kwarg_when_none(self, monkeypatch, tmp_path):
        """An empty ``reference_images`` list should not appear in the
        provider call kwargs at all — keeping the call shape identical to
        the pre-PR behavior for plain text-to-image."""
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        captured = {}

        class _CaptureProvider(ImageGenProvider):
            @property
            def name(self) -> str:
                return "capture"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                captured.update({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
                return {
                    "success": True,
                    "image": "/tmp/capture.png",
                    "model": "test-model",
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "provider": "capture",
                }

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: capture\n")
        image_gen_registry.register_provider(_CaptureProvider())
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "capture")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda *a, **kw: None)
        monkeypatch.setattr(
            registry_module, "get_provider",
            lambda name: _CaptureProvider() if name == "capture" else None,
        )

        image_generation_tool._dispatch_to_plugin_provider("just text", "landscape")
        assert "reference_images" not in captured

        captured.clear()
        image_generation_tool._dispatch_to_plugin_provider(
            "still just text",
            "landscape",
            reference_images=[],
        )
        assert "reference_images" not in captured


class TestAutoAttachReferenceImages:
    """Cover the ``_handle_image_generate`` auto-inject path: when the
    active session has registered attachment paths (gateway wrote them
    after consuming a native image turn), the handler appends them to
    the model-supplied ``reference_images`` so "edit this" Just Works
    without the model needing to know on-disk paths.
    """

    def setup_method(self, method):
        from agent import image_routing
        image_routing._RECENT_ATTACHED_IMAGE_PATHS.clear()
        # Reset session-key contextvar between tests.
        from tools import approval
        approval._approval_session_key.set("")

    def teardown_method(self, method):
        from agent import image_routing
        image_routing._RECENT_ATTACHED_IMAGE_PATHS.clear()
        from tools import approval
        approval._approval_session_key.set("")

    def _wire_capture_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        captured = {}

        class _CaptureProvider(ImageGenProvider):
            @property
            def name(self) -> str:
                return "capture"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                captured.update({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
                return {
                    "success": True, "image": "/tmp/capture.png", "model": "test-model",
                    "prompt": prompt, "aspect_ratio": aspect_ratio, "provider": "capture",
                }

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: capture\n")
        image_gen_registry.register_provider(_CaptureProvider())
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: "capture"
        )
        monkeypatch.setattr(
            plugins_module, "_ensure_plugins_discovered", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            registry_module, "get_provider",
            lambda name: _CaptureProvider() if name == "capture" else None,
        )
        return captured

    def test_no_session_no_register_no_inject(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        captured = self._wire_capture_provider(monkeypatch, tmp_path)

        image_generation_tool._handle_image_generate({"prompt": "draw a cat"})
        assert "reference_images" not in captured

    def test_session_with_register_auto_attaches(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent.image_routing import register_recent_attached_image_paths
        from tools.approval import set_current_session_key

        captured = self._wire_capture_provider(monkeypatch, tmp_path)

        set_current_session_key("sess-X")
        register_recent_attached_image_paths("sess-X", ["/tmp/auto.png"])

        image_generation_tool._handle_image_generate({"prompt": "edit this"})
        assert captured["reference_images"] == ["/tmp/auto.png"]

    def test_explicit_refs_take_precedence_auto_appended(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent.image_routing import register_recent_attached_image_paths
        from tools.approval import set_current_session_key

        captured = self._wire_capture_provider(monkeypatch, tmp_path)

        set_current_session_key("sess-X")
        register_recent_attached_image_paths("sess-X", ["/tmp/auto.png", "/tmp/auto2.png"])

        image_generation_tool._handle_image_generate({
            "prompt": "blend",
            "reference_images": ["/tmp/explicit.png"],
        })
        # Explicit first, auto entries appended in order, no duplicates.
        assert captured["reference_images"] == [
            "/tmp/explicit.png", "/tmp/auto.png", "/tmp/auto2.png",
        ]

    def test_dedup_when_explicit_overlaps_auto(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent.image_routing import register_recent_attached_image_paths
        from tools.approval import set_current_session_key

        captured = self._wire_capture_provider(monkeypatch, tmp_path)

        set_current_session_key("sess-X")
        register_recent_attached_image_paths("sess-X", ["/tmp/a.png", "/tmp/b.png"])

        image_generation_tool._handle_image_generate({
            "prompt": "edit",
            "reference_images": ["/tmp/a.png"],
        })
        # /tmp/a.png appears once even though both explicit and auto have it.
        assert captured["reference_images"] == ["/tmp/a.png", "/tmp/b.png"]

    def test_other_session_register_does_not_leak(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent.image_routing import register_recent_attached_image_paths
        from tools.approval import set_current_session_key

        captured = self._wire_capture_provider(monkeypatch, tmp_path)

        register_recent_attached_image_paths("other-session", ["/tmp/other.png"])
        set_current_session_key("sess-X")  # different session

        image_generation_tool._handle_image_generate({"prompt": "draw"})
        assert "reference_images" not in captured

    def test_register_persists_across_multiple_calls(self, monkeypatch, tmp_path):
        """Reads are non-destructive: a single attachment can be referenced
        across multiple turns ("now red", "now bigger") in the same session.
        """
        from tools import image_generation_tool
        from agent.image_routing import register_recent_attached_image_paths
        from tools.approval import set_current_session_key

        captured = self._wire_capture_provider(monkeypatch, tmp_path)

        set_current_session_key("sess-X")
        register_recent_attached_image_paths("sess-X", ["/tmp/auto.png"])

        for prompt in ("make it red", "now bigger", "rotate 90"):
            image_generation_tool._handle_image_generate({"prompt": prompt})
            assert captured["reference_images"] == ["/tmp/auto.png"]
            captured.clear()
