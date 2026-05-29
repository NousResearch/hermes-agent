"""End-to-end regression for #24015 — capture routing via auxiliary.vision.

When ``computer_use(action='capture', mode='som'|'vision')`` returns a
screenshot, ``_capture_response`` previously always returned a
``_multimodal`` envelope. For non-vision main models, or when the user
explicitly configured ``auxiliary.vision`` in ``config.yaml``, that
envelope tripped HTTP 404 / 400 at the provider boundary even though a
perfectly good vision backend was sitting in config waiting to be used.

This file exercises the integrated ``_capture_response`` flow with
deterministic stubs for:

* ``should_route_capture_to_aux_vision`` (the policy decision)
* ``_run_async`` (sync->async bridge)
* ``vision_analyze_tool`` (the aux LLM call)
* ``hermes_constants.get_hermes_dir`` (cache path)

…so the full code path is covered without a live cua-driver, a real
auxiliary client, or network access.
"""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# 1×1 PNG (transparent) — minimal bytes that decode cleanly.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
    "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# 1×1 JPEG — used to verify mime detection works for either stream type.
_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Override get_hermes_dir so cache writes land under tmp_path."""
    cache_dir = tmp_path / "cache_vision"
    cache_dir.mkdir()

    def _fake_get(*_args, **_kw):
        return cache_dir

    with patch("hermes_constants.get_hermes_dir", _fake_get):
        yield cache_dir


def _make_capture(
    *,
    png_b64: str = _PNG_B64,
    mode: str = "som",
    elements=None,
    app: str = "Safari",
    window_title: str = "GitHub – Issue #24015",
    width: int = 1280,
    height: int = 800,
):
    from tools.computer_use.backend import CaptureResult, UIElement

    elements = list(elements or [
        UIElement(index=0, role="AXButton", label="Sign in",
                  bounds=(10, 20, 80, 30)),
        UIElement(index=1, role="AXTextField", label="username",
                  bounds=(10, 60, 200, 24)),
    ])
    raw = base64.b64decode(png_b64, validate=False)
    return CaptureResult(
        mode=mode,
        width=width,
        height=height,
        png_b64=png_b64,
        elements=elements,
        app=app,
        window_title=window_title,
        png_bytes_len=len(raw),
    )


def _stub_aux_analysis(text: str):
    """Return a fake vision_analyze_tool coroutine result (JSON envelope)."""
    return json.dumps({"success": True, "analysis": text})


# ---------------------------------------------------------------------------
# _capture_response: routing OFF (current/native behaviour)
# ---------------------------------------------------------------------------

class TestCaptureResponseDefaultPath:
    """When routing helper says 'native', the existing multimodal envelope wins."""

    def test_som_capture_returns_multimodal_envelope_when_native(self):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(png_b64=_PNG_B64, mode="som")
        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=False):
            resp = cu_tool._capture_response(cap)

        assert isinstance(resp, dict)
        assert resp.get("_multimodal") is True
        # Image part must use image/png MIME for a PNG payload.
        image_part = next(
            p for p in resp["content"] if p.get("type") == "image_url"
        )
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert "vision_analysis" not in resp

    def test_jpeg_capture_returns_image_jpeg_mime_when_native(self):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(png_b64=_JPEG_B64, mode="som")
        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=False):
            resp = cu_tool._capture_response(cap)

        url = next(p for p in resp["content"] if p.get("type") == "image_url")
        assert url["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_ax_only_capture_returns_text_regardless_of_routing(self):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="ax", png_b64="")
        # ax mode never has a PNG so neither path matters; assert pure text.
        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True) as routing:
            resp = cu_tool._capture_response(cap)

        # ax never even consults the routing helper — short-circuited above
        # the image branch.
        routing.assert_not_called()
        assert isinstance(resp, str)
        body = json.loads(resp)
        assert body["mode"] == "ax"


# ---------------------------------------------------------------------------
# _capture_response: routing ON (the #24015 fix)
# ---------------------------------------------------------------------------

class TestCaptureResponseRoutedToAuxVision:
    """When routing helper says 'aux', the PNG is pre-analysed and a text
    response is returned with no image_url parts at all."""

    def test_som_capture_returns_text_with_vision_analysis(
        self, tmp_cache_dir,
    ):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="som")

        captured_calls = {}

        def _fake_run_async(coro):
            captured_calls["called"] = True
            return _stub_aux_analysis(
                "A Safari window showing a GitHub issue page with a 'Sign "
                "in' button and a 'username' text field."
            )

        # vision_analyze_tool is async; force a sync MagicMock so we can
        # assert positional args without dealing with awaitables.
        fake_vat = MagicMock(return_value="<coro>")

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async", side_effect=_fake_run_async), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            resp = cu_tool._capture_response(cap)

        # Must be a JSON string, NOT a multimodal envelope. This is exactly
        # the contract that prevents #24015's HTTP 404 from firing on the
        # next agent turn.
        assert isinstance(resp, str)
        body = json.loads(resp)
        assert body["mode"] == "som"
        assert body["app"] == "Safari"
        assert "Sign in" in body["vision_analysis"]
        assert body["vision_analysis_routed_via"] == "auxiliary.vision"
        # The original AX-only metadata (window title, element index, app)
        # is preserved alongside the new vision analysis so the agent loses
        # no context vs the multimodal path.
        assert body["window_title"] == "GitHub – Issue #24015"
        assert len(body["elements"]) == 2

        assert captured_calls.get("called") is True
        # vision_analyze_tool was invoked with a path under the patched cache
        # and a non-empty prompt.
        args, _kwargs = fake_vat.call_args
        path_arg, prompt_arg = args[0], args[1]
        assert str(tmp_cache_dir) in path_arg
        assert "macOS application screenshot" in prompt_arg
        # AX summary is included so the aux model can ground its description
        # against the same set-of-mark index the agent will see.
        assert "Sign in" in prompt_arg

    def test_temp_screenshot_file_is_cleaned_up_after_routing(
        self, tmp_cache_dir,
    ):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="som")
        # We capture the path the aux call sees so we can assert it's gone
        # after _capture_response returns.
        observed_path = {}

        def _fake_run_async(_coro):
            return _stub_aux_analysis("description goes here")

        def _fake_vat(image_path, _prompt):
            observed_path["path"] = image_path
            # File must exist while aux is being arranged.
            assert os.path.exists(image_path)
            return "<coro>"

        fake_vat = MagicMock(side_effect=_fake_vat)

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async", side_effect=_fake_run_async), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            cu_tool._capture_response(cap)

        # File must be unlinked after _capture_response returns.
        assert observed_path["path"]
        assert not os.path.exists(observed_path["path"])

    def test_temp_file_cleaned_up_even_when_aux_call_raises(
        self, tmp_cache_dir,
    ):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="som")
        observed_path = {}

        def _fake_vat(image_path, _prompt):
            observed_path["path"] = image_path
            return "<coro>"

        def _fake_run_async(_coro):
            raise RuntimeError("aux LLM down")

        fake_vat = MagicMock(side_effect=_fake_vat)

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async", side_effect=_fake_run_async), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            resp = cu_tool._capture_response(cap)

        # Aux failure → fall back to multimodal envelope (so the user still
        # gets *something* useful even if vision is broken).
        assert isinstance(resp, dict)
        assert resp.get("_multimodal") is True
        # Temp file must still be cleaned up.
        assert observed_path["path"]
        assert not os.path.exists(observed_path["path"])

    def test_empty_aux_analysis_falls_back_to_multimodal(self, tmp_cache_dir):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="som")

        def _fake_run_async(_coro):
            return _stub_aux_analysis("")

        fake_vat = MagicMock(return_value="<coro>")

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async", side_effect=_fake_run_async), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            resp = cu_tool._capture_response(cap)

        # Empty analysis is treated as failure — we'd rather show pixels
        # than embed an empty 'vision_analysis' string into the result.
        assert isinstance(resp, dict)
        assert resp.get("_multimodal") is True

    def test_invalid_aux_response_falls_back_to_multimodal(self, tmp_cache_dir):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="som")

        def _fake_run_async(_coro):
            return 1234  # not a string at all

        fake_vat = MagicMock(return_value="<coro>")

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async", side_effect=_fake_run_async), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            resp = cu_tool._capture_response(cap)

        assert isinstance(resp, dict)
        assert resp.get("_multimodal") is True


# ---------------------------------------------------------------------------
# _should_route_through_aux_vision: end-to-end with real config plumbing
# ---------------------------------------------------------------------------

class TestRoutingDecisionWiring:
    """Verify _should_route_through_aux_vision wires the right config + helper."""

    def test_explicit_aux_vision_in_config_routes_to_aux(self):
        from tools.computer_use import tool as cu_tool

        cfg = {
            "model": {"default": "tencent/hy3-preview", "provider": "openrouter"},
            "auxiliary": {
                "vision": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash",
                }
            },
        }
        with patch("agent.auxiliary_client._read_main_provider",
                   return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model",
                   return_value="tencent/hy3-preview"), \
             patch("hermes_cli.config.load_config", return_value=cfg):
            assert cu_tool._should_route_through_aux_vision() is True

    def test_no_explicit_aux_and_vision_capable_main_keeps_multimodal(self):
        from tools.computer_use import tool as cu_tool

        cfg = {
            "model": {"default": "claude-opus-4-5", "provider": "anthropic"},
        }
        with patch("agent.auxiliary_client._read_main_provider",
                   return_value="anthropic"), \
             patch("agent.auxiliary_client._read_main_model",
                   return_value="claude-opus-4-5"), \
             patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("tools.computer_use.vision_routing._lookup_supports_vision",
                   return_value=True), \
             patch("tools.computer_use.vision_routing."
                   "_provider_accepts_multimodal_tool_result",
                   return_value=True):
            assert cu_tool._should_route_through_aux_vision() is False

    def test_config_load_failure_disables_routing_safely(self):
        from tools.computer_use import tool as cu_tool

        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("config.yaml unreadable")):
            # No exception should bubble up — fail open by returning False
            # so the legacy multimodal envelope continues to work.
            assert cu_tool._should_route_through_aux_vision() is False

    def test_helper_decision_exception_is_swallowed(self):
        from tools.computer_use import tool as cu_tool
        from tools.computer_use import vision_routing as vr_mod

        with patch("agent.auxiliary_client._read_main_provider",
                   return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model",
                   return_value="x"), \
             patch("hermes_cli.config.load_config", return_value={}), \
             patch.object(vr_mod, "should_route_capture_to_aux_vision",
                          side_effect=ValueError("policy bug")):
            assert cu_tool._should_route_through_aux_vision() is False


# ---------------------------------------------------------------------------
# Bug reproduction marker — proves the fix is needed.
# ---------------------------------------------------------------------------

class TestBugReproductionAnchor:
    """Without the fix, this test would assert the wrong thing.

    On upstream/main HEAD prior to this branch, _capture_response returns a
    multimodal envelope unconditionally — so when a non-vision main model
    is configured, the captured PNG is delivered to the main provider as
    image_url content and the request is rejected with HTTP 404. We don't
    have a live provider here, but we can pin the contract: with routing
    enabled the response MUST be a JSON string with no image_url parts.
    """

    def test_non_vision_main_model_never_returns_image_url_when_routed(
        self, tmp_cache_dir,
    ):
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(mode="som")

        def _fake_run_async(_coro):
            return _stub_aux_analysis(
                "Screenshot showing a GitHub.com window with a sign-in "
                "form."
            )

        fake_vat = MagicMock(return_value="<coro>")

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async", side_effect=_fake_run_async), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            resp = cu_tool._capture_response(cap)

        # Must be a string (text-only result).
        assert isinstance(resp, str)
        # Must NOT contain a base64 image URL anywhere — that's what tripped
        # 'No endpoints found that support image input' on the reporter's
        # main provider in #24015.
        assert "data:image" not in resp
        assert "image_url" not in resp
