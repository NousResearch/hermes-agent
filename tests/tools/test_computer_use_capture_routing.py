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

# 8×8 PNG (transparent) — minimal provider-acceptable bytes that decode cleanly.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
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
        assert "desktop application screenshot" in prompt_arg
        # AX summary is included so the aux model can ground its description
        # against the same set-of-mark index the agent will see.
        assert "Sign in" in prompt_arg


    def test_invalid_aux_response_degrades_to_text_payload(self, tmp_cache_dir):
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

        assert isinstance(resp, str)
        body = json.loads(resp)
        assert body.get("vision_unavailable") is True


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


# ---------------------------------------------------------------------------
# capture(question=...) — issue #87818
# ---------------------------------------------------------------------------

class TestCaptureQuestion:
    """A caller-supplied ``question`` directs the aux-vision analysis.

    Before #87818 there was nothing to direct it with: ``question`` was not a
    declared parameter, so a hand-crafted one rode into ``args`` on the back of
    a permissive schema and was then dropped on the floor, and the aux model
    always got the same hardcoded "describe what is visible" prompt no matter
    what the caller wanted to know.

    The narrow-but-load-bearing part of this contract is that a question
    replaces the *instruction* only. The AX/SOM index below it is what lets the
    aux model's answer name element numbers that the main model can click by
    index, so replacing the whole prompt with the question would trade this bug
    for a worse one.
    """

    @staticmethod
    def _prompt_for(cap, **kwargs) -> str:
        """Run the routed path and return the prompt handed to the aux model."""
        from tools.computer_use import tool as cu_tool

        fake_vat = MagicMock(return_value="<coro>")

        with patch.object(cu_tool, "_should_route_through_aux_vision",
                          return_value=True), \
             patch("model_tools._run_async",
                   side_effect=lambda _coro: _stub_aux_analysis("ok")), \
             patch("tools.vision_tools.vision_analyze_tool",
                   new_callable=lambda: fake_vat):
            cu_tool._capture_response(cap, **kwargs)

        return fake_vat.call_args[0][1]

    def test_question_reaches_the_aux_vision_prompt(self, tmp_cache_dir):
        prompt = self._prompt_for(
            _make_capture(mode="som"),
            question="Is the Sign in button enabled?",
        )

        assert "Is the Sign in button enabled?" in prompt
        # The general-description instruction is what the question replaces.
        assert "Describe what is visible" not in prompt

    def test_ax_som_index_survives_a_directed_question(self, tmp_cache_dir):
        """The regression guard for the obvious wrong fix.

        Swapping the whole prompt for the caller's question would read as
        working — the question reaches the model, the answer comes back — while
        silently costing ``mode='som'`` the element index its click-by-index
        contract is built on. Assert the index is still there.
        """
        prompt = self._prompt_for(
            _make_capture(mode="som"),
            question="What error is in the dialog?",
        )

        assert "AX/SOM index for cross-reference:" in prompt
        # …and the actual index content, not just the header.
        assert "Sign in" in prompt
        assert "username" in prompt

    def test_directed_question_keeps_the_do_not_invent_clause(
        self, tmp_cache_dir,
    ):
        """A yes/no question invites confabulation more readily than an open
        describe does, so the anti-invention clause is needed more on this
        branch, not less."""
        prompt = self._prompt_for(
            _make_capture(mode="som"),
            question="Is there an unsaved-changes indicator?",
        )

        assert "Do not invent details that are not actually visible." in prompt

    def test_no_question_falls_back_to_the_general_description(
        self, tmp_cache_dir,
    ):
        prompt = self._prompt_for(_make_capture(mode="som"))

        assert "Describe what is visible in this desktop application" in prompt
        assert "Answer this question" not in prompt
        # Unchanged from before #87818 on this branch.
        assert "AX/SOM index for cross-reference:" in prompt
        assert "Do not invent details that are not actually visible." in prompt

    def test_blank_question_falls_back_rather_than_asking_nothing(
        self, tmp_cache_dir,
    ):
        """``question=""`` must not produce an instruction to answer an empty
        question — it is indistinguishable from not passing one at all."""
        prompt = self._prompt_for(_make_capture(mode="som"), question="")

        assert "Describe what is visible in this desktop application" in prompt
        assert "Answer this question" not in prompt


class TestCaptureQuestionPlumbing:
    """``question`` must be reachable from a real tool call, not just from a
    direct ``_capture_response`` call in a test."""

    def test_declared_in_the_schema(self):
        """Threading the value through without declaring it leaves the feature
        unreachable for any well-behaved client, and stripped outright by a
        strict-schema one. Both halves are the fix."""
        from tools.computer_use.schema import COMPUTER_USE_SCHEMA

        prop = COMPUTER_USE_SCHEMA["parameters"]["properties"]["question"]
        assert prop["type"] == "string"
        assert "capture" in prop["description"]

    def test_dispatch_threads_question_from_tool_args(self):
        from tools.computer_use import tool as cu_tool

        backend = MagicMock()
        backend.capture.return_value = _make_capture(mode="som")

        with patch.object(cu_tool, "_capture_response",
                          return_value="{}") as fake_resp:
            cu_tool._dispatch(
                backend,
                "capture",
                {"mode": "som", "question": "Is the Save button enabled?"},
            )

        _args, kwargs = fake_resp.call_args
        assert kwargs["question"] == "Is the Save button enabled?"

    def test_dispatch_passes_none_when_no_question_given(self):
        from tools.computer_use import tool as cu_tool

        backend = MagicMock()
        backend.capture.return_value = _make_capture(mode="som")

        with patch.object(cu_tool, "_capture_response",
                          return_value="{}") as fake_resp:
            cu_tool._dispatch(backend, "capture", {"mode": "som"})

        _args, kwargs = fake_resp.call_args
        assert kwargs["question"] is None


class TestCoerceQuestion:
    """``_coerce_question`` is the same shape as ``_coerce_max_elements``: a
    malformed tool-call argument degrades to the default behaviour instead of
    reaching the prompt."""

    @pytest.mark.parametrize("value", [None, 42, 3.5, True, [], {}, object()])
    def test_non_strings_are_dropped(self, value):
        from tools.computer_use import tool as cu_tool

        assert cu_tool._coerce_question(value) is None

    @pytest.mark.parametrize("value", ["", "   ", "\n\t "])
    def test_blank_strings_are_dropped(self, value):
        from tools.computer_use import tool as cu_tool

        assert cu_tool._coerce_question(value) is None

    def test_surrounding_whitespace_is_stripped(self):
        from tools.computer_use import tool as cu_tool

        assert cu_tool._coerce_question("  what is on screen?  ") == (
            "what is on screen?"
        )

    def test_overlong_question_is_capped(self):
        """An unbounded question is interpolated ahead of the AX/SOM index and
        can crowd out the index it is meant to accompany."""
        from tools.computer_use import tool as cu_tool

        coerced = cu_tool._coerce_question("q" * 5000)

        assert coerced is not None
        assert len(coerced) == cu_tool._MAX_QUESTION_CHARS
