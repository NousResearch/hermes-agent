"""A model without vision is an ANSWER, not a crash (#89114).

#89114 was filed as "Vision tool crashes with BadRequestError when model lacks
image support", with an ``openai.BadRequestError`` traceback pasted in.  The
traceback is real but it is not a crash: it is ``vision_analyze_tool``'s own
``logger.error(..., exc_info=True)`` line, emitted from inside the handler that
had *already* classified the failure and was about to return
``success: false`` with a readable sentence.  Logging a handled, classified,
user-actionable outcome at ERROR with a full stack made it indistinguishable
from an unhandled exception — to the reporter, and to anyone reading logs.

Three properties are pinned here:

1. ``vision_analyze_tool`` does not raise on the reporter's exact provider
   message, and logs it at WARNING without a traceback.  An *unclassified*
   failure still gets ERROR + ``exc_info`` — that is the case where a stack
   actually earns its place.
2. ``browser_vision`` reaches the same auxiliary vision model by the same
   route, so the same rejection must produce the same kind of answer.  Before
   this change it alone handed the agent a raw ``Error code: 400 - {...}``
   blob while ``vision_analyze`` produced a sentence.
3. ``computer_use`` -- the tool the issue names -- stops presenting a failed
   analysis as ``vision_analysis``, and takes the graceful-degradation path
   it already had instead.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vision_tools import is_vision_capability_error, vision_analyze_tool


# The exact string the provider returned in #89114 (mimo-v2.5 via an
# OpenCode-compatible Go provider).  Everything below keys off it.
REPORTED_ERROR = (
    "Error code: 400 - {'error': {'message': 'This model does not support "
    "image inputs', 'type': 'invalid_request_error', 'param': 'messages', "
    "'code': 'invalid_request_error'}}"
)


class _BadRequestError(Exception):
    """Stand-in for ``openai.BadRequestError`` — the tests only use its text."""


class TestCapabilityClassification:
    """``is_vision_capability_error`` — the shared predicate."""

    @pytest.mark.parametrize("message", [
        REPORTED_ERROR,
        "This model does not support image inputs",
        "model is not multimodal",
        "unrecognized request argument supplied: image_url",
        "image input is not supported for this model",
        "does not support vision",
    ])
    def test_capability_rejections_are_recognised(self, message):
        assert is_vision_capability_error(_BadRequestError(message)) is True

    @pytest.mark.parametrize("message", [
        # A size rejection is a different remedy (resize and retry), and
        # _is_image_size_error already owns it.
        "image is too large: maximum payload size is 20 MB",
        # Billing is a different remedy again (top up the account).
        "Error code: 402 - insufficient credits",
        # An outage says nothing about the model's capabilities; calling it a
        # capability error would tell the user to change their config for a
        # problem that fixes itself.
        "Error code: 500 - internal server error",
        "Connection timed out",
    ])
    def test_unrelated_failures_are_not_capability_errors(self, message):
        assert is_vision_capability_error(_BadRequestError(message)) is False


class TestVisionAnalyzeDoesNotCrash:
    """The report itself: ``vision_analyze`` on a text-only model."""

    @staticmethod
    async def _run_with_error(error, image_path):
        with (
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                return_value="data:image/png;base64,abc",
            ),
            patch(
                "tools.vision_tools.async_call_llm",
                new_callable=AsyncMock,
                side_effect=error,
            ),
        ):
            return json.loads(
                await vision_analyze_tool(str(image_path), "describe this", "mimo-v2.5")
            )

    @pytest.fixture
    def image(self, tmp_path):
        img = tmp_path / "screenshot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return img

    @pytest.mark.asyncio
    async def test_reported_error_returns_an_answer_instead_of_raising(self, image):
        """The `needs-repro` question: it does not propagate, it answers."""
        result = await self._run_with_error(_BadRequestError(REPORTED_ERROR), image)

        assert result["success"] is False
        # The analysis must name the cause in words the agent can relay.
        assert "does not support vision" in result["analysis"]
        assert "mimo-v2.5" in result["analysis"]

    @pytest.mark.asyncio
    async def test_classified_failure_logs_warning_without_a_traceback(
        self, image, caplog
    ):
        """The actual defect: a handled outcome logged like an unhandled crash."""
        with caplog.at_level(logging.WARNING, logger="tools.vision_tools"):
            await self._run_with_error(_BadRequestError(REPORTED_ERROR), image)

        records = [
            r for r in caplog.records
            if "Error analyzing image" in r.getMessage()
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None, (
            "a classified, answered failure must not print a stack trace — "
            "the traceback in #89114 is this very log line"
        )

    @pytest.mark.asyncio
    async def test_unclassified_failure_keeps_error_and_traceback(
        self, image, caplog
    ):
        """The negative half: surprises still get the loudest treatment."""
        with caplog.at_level(logging.WARNING, logger="tools.vision_tools"):
            await self._run_with_error(
                _BadRequestError("kernel panic in the tensor mines"), image
            )

        records = [
            r for r in caplog.records
            if "Error analyzing image" in r.getMessage()
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info is not None

    @pytest.mark.asyncio
    async def test_content_policy_still_routes_to_the_capability_message(self, image):
        """Behaviour preservation across the helper extraction.

        ``content_policy`` was one arm of the inline hint tuple this change
        replaced with ``is_vision_capability_error``.  It is *not* a capability
        problem, so it stays out of the shared predicate — but the branch must
        still fire for it, or the refactor silently changed the truth table.
        """
        assert is_vision_capability_error(
            _BadRequestError("content_policy violation")
        ) is False

        result = await self._run_with_error(
            _BadRequestError("content_policy violation"), image
        )
        assert "does not support vision" in result["analysis"]

    @pytest.mark.asyncio
    async def test_credit_errors_keep_their_own_remedy(self, image):
        """Ordering guard: billing is checked before capability."""
        result = await self._run_with_error(
            _BadRequestError("Error code: 402 - insufficient credits"), image
        )
        assert "top up" in result["analysis"].lower()


class TestBrowserVisionSurfacesTheSameAnswer:
    """``browser_vision`` takes the same route to the same aux model."""

    @staticmethod
    def _invoke(error, screenshot):
        import tools.browser_tool as bt

        def fake_run(task_id, command, args=None, **kwargs):
            return {"success": True, "data": {"path": str(screenshot)}}

        with (
            patch.object(bt, "_is_camofox_mode", lambda: False),
            patch.object(bt, "_is_local_backend", lambda: True),
            patch.object(bt, "_get_browser_engine", lambda: "chrome"),
            patch.object(bt, "_cleanup_old_screenshots", lambda *a, **k: None),
            patch.object(bt, "_run_browser_command", fake_run),
            patch.object(bt, "_get_vision_model", lambda: "mimo-v2.5"),
            patch("tools.vision_tools._should_use_native_vision_fast_path",
                  lambda: False),
            patch.object(bt, "_lazy_call_llm", side_effect=error),
        ):
            return json.loads(bt.browser_vision("what is on screen?"))

    @pytest.fixture
    def screenshot(self, tmp_path):
        shot = tmp_path / "browser_screenshot.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return shot

    def test_capability_rejection_becomes_a_sentence_not_a_400_blob(
        self, screenshot
    ):
        result = self._invoke(_BadRequestError(REPORTED_ERROR), screenshot)

        assert result["success"] is False
        assert "does not support image inputs" in result["error"]
        assert "auxiliary.vision" in result["error"], (
            "the message must name the config key that fixes it — otherwise "
            "the agent can relay the failure but not the remedy"
        )

    def test_unclassified_failure_keeps_the_raw_error(self, screenshot):
        """No over-reach: an unrecognised failure is reported verbatim."""
        result = self._invoke(_BadRequestError("socket hang up"), screenshot)

        assert result["success"] is False
        assert "socket hang up" in result["error"]
        assert "auxiliary.vision" not in result["error"]

    def test_screenshot_is_preserved_on_a_classified_failure(self, screenshot):
        """The pre-existing contract still holds through the new branch."""
        result = self._invoke(_BadRequestError(REPORTED_ERROR), screenshot)

        assert result.get("screenshot_path") == str(screenshot)


class TestComputerUseDegradesInsteadOfNarratingTheFailure:
    """``computer_use`` is the tool #89114 was actually filed against.

    ``_route_capture_through_aux_vision`` read ``analysis`` out of
    ``vision_analyze_tool``'s envelope without checking ``success``.  Because
    that field is populated on the failure path too, a capability rejection
    was merged into the response as ``vision_analysis`` — the main model was
    handed a 400 blob under a key that promises a description of the screen.

    The caller already had the graceful degradation the reporter asks for
    (screenshot omitted, ``vision_unavailable``, element index still
    drivable); it simply never fired, because the helper never reported
    failure.
    """

    # 8x8 transparent PNG — same bytes the existing routing tests use.
    _PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
        "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
    )

    @pytest.fixture
    def cache_dir(self, tmp_path):
        cache = tmp_path / "cache_vision"
        cache.mkdir()
        with patch("hermes_constants.get_hermes_dir", lambda *a, **k: cache):
            yield cache

    def _capture(self):
        import base64

        from tools.computer_use.backend import CaptureResult, UIElement

        return CaptureResult(
            mode="som",
            width=1280,
            height=800,
            png_b64=self._PNG_B64,
            elements=[
                UIElement(index=0, role="AXButton", label="Sign in",
                          bounds=(10, 20, 80, 30)),
            ],
            app="Safari",
            window_title="Hermes",
            png_bytes_len=len(base64.b64decode(self._PNG_B64, validate=False)),
        )

    def _route(self, envelope, cache_dir):
        from tools.computer_use import tool as cu_tool

        with (
            patch.object(cu_tool, "_should_route_through_aux_vision",
                         return_value=True),
            patch("model_tools._run_async", side_effect=lambda _c: envelope),
            patch("tools.vision_tools.vision_analyze_tool",
                  new_callable=lambda: MagicMock(return_value="<coro>")),
        ):
            return json.loads(cu_tool._capture_response(self._capture()))

    def test_capability_failure_degrades_to_the_ax_payload(self, cache_dir):
        envelope = json.dumps({
            "success": False,
            "error": f"Error analyzing image: {REPORTED_ERROR}",
            "analysis": (
                "mimo-v2.5 does not support vision or our request was not "
                f"accepted by the server. Error: {REPORTED_ERROR}"
            ),
        })

        body = self._route(envelope, cache_dir)

        assert body.get("vision_unavailable") is True
        assert "vision_analysis" not in body, (
            "a failed analysis must not be presented as a description of the "
            "screen — the main model cannot tell the difference"
        )
        assert "Error code: 400" not in json.dumps(body)
        # The degradation is useful, not just safe: the element index the
        # agent drives with survives.
        assert len(body["elements"]) == 1
        assert body["elements"][0]["label"] == "Sign in"

    def test_successful_analysis_is_still_routed_through(self, cache_dir):
        """The guard is narrow: a real description reaches the model intact."""
        envelope = json.dumps({
            "success": True,
            "analysis": "A Safari window showing a 'Sign in' button.",
        })

        body = self._route(envelope, cache_dir)

        assert body["vision_analysis_routed_via"] == "auxiliary.vision"
        assert "Sign in" in body["vision_analysis"]
        assert "vision_unavailable" not in body
