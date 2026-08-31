"""Screenshot dedup — unchanged frames are delivered without their image.

Port of openclaw/openclaw#129924: a capture whose pixels are byte-identical
to the previous capture of the same target in the same session returns text
metadata plus an explicit "screen unchanged" note instead of the multimodal
image block. Changed pixels, changed targets, other sessions, and streak
exhaustion all deliver full pixels again.
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
