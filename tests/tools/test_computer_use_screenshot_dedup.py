"""Screenshot dedup — unchanged frames are delivered without their image.

Port of openclaw/openclaw#129924: a capture whose pixels are byte-identical
to the previous capture of the same target in the same session returns text
metadata plus an explicit "screen unchanged" note instead of the multimodal
image block. Changed pixels, changed targets, other sessions, and streak
exhaustion all deliver full pixels again.

Past a few identical captures in a row that note inverts: an idle desktop and
a capture pipeline that stopped repainting are byte-for-byte the same, and
only the second one is dangerous to keep acting on (#132125).
"""

import base64
import json
import struct
import zlib

import pytest

import tools.computer_use.tool as cu_tool
from tools.computer_use.backend import CaptureResult


def _png_b64(seed: int) -> str:
    """A tiny real PNG (so dimension sniffing works) with seed-varied bytes."""
    width = height = 64
    raw = b"".join(
        b"\x00" + bytes(((x + y + seed) % 256) for x in range(width * 3))
        for y in range(height)
    )
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _cap(seed: int = 1, app: str = "kate", title: str = "notes") -> CaptureResult:
    return CaptureResult(
        mode="som", width=64, height=64, png_b64=_png_b64(seed),
        elements=[], app=app, window_title=title,
    )


@pytest.fixture(autouse=True)
def _fresh_dedup(monkeypatch):
    cu_tool._reset_screenshot_dedup()
    # Keep the aux-vision router out of the way — these tests exercise the
    # multimodal branch directly.
    monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: False)
    monkeypatch.setattr(cu_tool, "_persist_capture_image", lambda cap: None)
    yield
    cu_tool._reset_screenshot_dedup()


def _is_multimodal(resp) -> bool:
    return isinstance(resp, dict) and resp.get("_multimodal") is True


class TestScreenshotDedup:
    def test_first_capture_delivers_image(self):
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert _is_multimodal(resp)

    def test_identical_second_capture_omits_image(self):
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert not _is_multimodal(resp)
        data = json.loads(resp)
        assert data["screen_unchanged"] is True
        assert "screen unchanged" in data["summary"]
        # Text metadata is still complete: the model can keep acting by element index.
        assert data["app"] == "kate" and "elements" in data

    def test_changed_pixels_deliver_image_again(self):
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=2), session_id="s1")
        assert _is_multimodal(resp)

    def test_different_target_delivers_image(self):
        cu_tool._capture_response(_cap(seed=1, app="kate"), session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=1, app="kcalc"), session_id="s1")
        assert _is_multimodal(resp)

    def test_sessions_do_not_share_dedup_state(self):
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s2")
        assert _is_multimodal(resp)

    def test_streak_cap_redelivers_pixels(self):
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        omitted = 0
        for _ in range(cu_tool._SCREENSHOT_DEDUP_MAX_STREAK):
            resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
            assert not _is_multimodal(resp)
            omitted += 1
        # Streak exhausted — identical pixels must be re-delivered in full.
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert _is_multimodal(resp)
        assert omitted == cu_tool._SCREENSHOT_DEDUP_MAX_STREAK
        # And the re-delivery resets the streak: next identical capture
        # dedups again.
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert not _is_multimodal(resp)

    def test_no_session_id_never_dedups(self):
        cu_tool._capture_response(_cap(seed=1))
        resp = cu_tool._capture_response(_cap(seed=1))
        assert _is_multimodal(resp)

    def test_change_resets_streak_state(self):
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        cu_tool._capture_response(_cap(seed=1), session_id="s1")  # omitted
        cu_tool._capture_response(_cap(seed=3), session_id="s1")  # new pixels
        resp = cu_tool._capture_response(_cap(seed=3), session_id="s1")
        assert not _is_multimodal(resp)  # dedups against the NEW frame

    def test_ax_mode_untouched(self):
        cap = CaptureResult(mode="ax", width=64, height=64, png_b64=None,
                            elements=[], app="kate", window_title="notes")
        resp = cu_tool._capture_response(cap, session_id="s1")
        assert not _is_multimodal(resp)
        assert "screen_unchanged" not in json.loads(resp)

    def test_explicit_capture_action_after_input_is_subject_to_dedup(self):
        """The dispatch path (explicit `capture` and `capture_after`) carries the session key; an unchanged
        frame is reported as unchanged rather than re-sent, and the streak cap still forces re-delivery."""
        backend = cu_tool._NoopBackend()
        frame = _cap(seed=1)
        backend.capture = lambda **kw: frame
        first = cu_tool._dispatch(backend, "capture", {"mode": "som"}, session_id="s1")
        assert _is_multimodal(first)
        after_click = cu_tool._dispatch(backend, "click", {"element": 1, "capture_after": True}, session_id="s1")
        assert not _is_multimodal(after_click) and json.loads(after_click)["screen_unchanged"] is True
        assert json.loads(after_click)["ok"] is True  # action payload merged into the text capture
        explicit = cu_tool._dispatch(backend, "capture", {"mode": "som"}, session_id="s1")
        assert not _is_multimodal(explicit)
        assert _is_multimodal(cu_tool._dispatch(backend, "capture", {"mode": "som"}, session_id="s1"))

    def test_release_session_forgets_dedup_state(self):
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        cu_tool.release_computer_use_session("s1")
        assert _is_multimodal(cu_tool._capture_response(_cap(seed=1), session_id="s1"))

    def test_compaction_boundary_redelivers_pixels(self):
        """The summary may have dropped the frame an "unchanged" note points at, so the first
        capture after compaction must carry the image even when the screen is byte-identical."""
        from agent.conversation_compression import _reset_read_dedup_caches

        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert not _is_multimodal(cu_tool._capture_response(_cap(seed=1), session_id="s1"))
        _reset_read_dedup_caches("task", session_id="s1")
        assert _is_multimodal(cu_tool._capture_response(_cap(seed=1), session_id="s1"))


class TestFrozenCaptureDetection:
    """Identical bytes are also what a capture pipeline that stopped repainting looks like."""

    def _capture_n(self, n, session_id="s1", seed=1):
        """Deliver n identical captures; return the last response."""
        resp = cu_tool._capture_response(_cap(seed=seed), session_id=session_id)
        for _ in range(n - 1):
            resp = cu_tool._capture_response(_cap(seed=seed), session_id=session_id)
        return resp

    @staticmethod
    def _payload(resp) -> str:
        return resp if isinstance(resp, str) else json.dumps(resp)

    @staticmethod
    def _text(resp) -> str:
        return resp if isinstance(resp, str) else resp["text_summary"]

    def test_frozen_run_is_flagged_not_certified(self):
        """The verdict fires on the threshold capture and the note no longer certifies the pixels."""
        below = self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS - 1)
        assert "capture_frozen_suspected" not in self._payload(below)
        assert "still shows the current state" in self._text(below)
        frozen = self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS)
        assert "capture_frozen_suspected" in self._payload(frozen)
        assert "still shows the current state" not in self._text(frozen)
        assert "stale" in self._text(frozen)

    def test_frozen_note_names_the_run_length(self):
        resp = self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS + 2)
        assert f"no change in {cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS + 2} consecutive captures" in self._text(resp)

    def test_identical_run_survives_the_omission_streak_cap(self):
        """Re-delivering pixels after the omission streak must not reset the evidence: the frames really
        are identical, so the run keeps climbing across the re-delivery instead of restarting at 1."""
        withheld = self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS)
        assert not _is_multimodal(withheld)
        assert "capture_frozen_suspected" in self._payload(withheld)
        # The next identical capture exhausts the omission streak, so the pixels go back out — and the run
        # counter must carry on from where it was rather than restart, or the warning would vanish here.
        delivered = self._capture_n(1)
        assert _is_multimodal(delivered)
        assert (f"no change in {cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS + 1} consecutive captures"
                in delivered["text_summary"])

    def test_frozen_verdict_rides_on_the_delivered_frame(self):
        """Whichever way the dedup policy resolved this capture — image withheld or re-sent after the
        streak cap — the frozen verdict travels with it, so the note can never certify a stale frame."""
        withheld = self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS)
        assert "capture_frozen_suspected" in self._payload(withheld)
        assert not _is_multimodal(withheld)  # the policy still saves the pixels while the streak allows it
        # The next identical capture exhausts the streak cap, so the pixels go back out — the prose warning
        # has to ride along on the delivered frame too, not only on the withheld ones.
        resp = self._capture_n(1)
        assert _is_multimodal(resp)
        assert "no change in" in resp["text_summary"]
        assert "stale, not the screen" in resp["text_summary"]

    def test_dedup_policy_still_governs_the_image(self):
        """The escalation is wording only: withholding stays the streak cap's decision, not the frozen
        verdict's, so a genuinely idle long-running session keeps its token saving."""
        seen = [self._capture_n(i) for i in range(1, cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS + 3)]
        assert [not _is_multimodal(r) for r in seen] == [False, True, True, False, True], seen

    def test_changed_pixels_clear_the_frozen_verdict(self):
        self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS)
        resp = cu_tool._capture_response(_cap(seed=9), session_id="s1")
        assert _is_multimodal(resp)
        assert "capture_frozen_suspected" not in resp["text_summary"]
        # And the counter restarts against the new frame.
        assert not _is_multimodal(cu_tool._capture_response(_cap(seed=9), session_id="s1"))

    def test_short_idle_run_keeps_the_unchanged_wording(self):
        """Two identical captures is the ordinary capture→act→capture loop; escalating there would
        cry wolf on every static screen."""
        cu_tool._capture_response(_cap(seed=1), session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        data = json.loads(resp)
        assert data["screen_unchanged"] is True
        assert "capture_frozen_suspected" not in data
        assert "still shows the current state" in data["summary"]

    def test_frozen_run_is_per_session(self):
        self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS, session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s2")
        assert _is_multimodal(resp)
        assert "capture_frozen_suspected" not in resp["text_summary"]

    def test_dedup_check_reports_the_run(self):
        """The counter is the contract between the two states, so pin its shape rather than its value."""
        assert cu_tool._screenshot_dedup_check("s9", "d1", ("kate", "notes")) == (False, 1)
        assert cu_tool._screenshot_dedup_check("s9", "d1", ("kate", "notes")) == (True, 2)
        assert cu_tool._screenshot_dedup_check("s9", "d1", ("kate", "notes")) == (True, 3)
        # Streak exhausted → pixels come back, but the identical run keeps climbing.
        assert cu_tool._screenshot_dedup_check("s9", "d1", ("kate", "notes")) == (False, 4)
        # A different target is a different frame.
        assert cu_tool._screenshot_dedup_check("s9", "d1", ("kcalc", "notes")) == (False, 1)
        # New pixels restart it.
        assert cu_tool._screenshot_dedup_check("s9", "d2", ("kate", "notes")) == (False, 1)

    def test_session_release_forgets_the_frozen_run(self):
        self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS)
        cu_tool.release_computer_use_session("s1")
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert _is_multimodal(resp)
        assert "capture_frozen_suspected" not in resp["text_summary"]

    def test_compaction_boundary_clears_the_frozen_run(self):
        """Compaction rewrites the history the frozen note points at, so the next capture must start
        from the delivered frame again."""
        from agent.conversation_compression import _reset_read_dedup_caches

        self._capture_n(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS)
        _reset_read_dedup_caches("task", session_id="s1")
        resp = cu_tool._capture_response(_cap(seed=1), session_id="s1")
        assert _is_multimodal(resp)
        assert "capture_frozen_suspected" not in resp["text_summary"]

    def test_ax_mode_never_reports_frozen(self):
        """An ax capture carries no pixels, so there is nothing to compare and nothing to distrust."""
        cap = CaptureResult(mode="ax", width=64, height=64, png_b64=None,
                            elements=[], app="kate", window_title="notes")
        resp = cu_tool._capture_response(cap, session_id="s1")
        for _ in range(cu_tool._SCREENSHOT_FROZEN_IDENTICAL_RUNS + 1):
            resp = cu_tool._capture_response(cap, session_id="s1")
        assert "capture_frozen_suspected" not in json.loads(resp)
