"""Desktop image submit path must never block on vision calls (#83291).

The old ``_enrich_with_attached_images`` pre-analyzed every attached image
with the auxiliary vision model *before* the turn was dispatched — serial
blocking calls of 60-90s per large photo, silent failure swallowing, and an
interrupt window that killed the turn with zero API calls. The replacement
``_build_image_ref_message`` only references the image paths so the agent
analyzes them in-loop with ``vision_analyze`` (the same shape as the
``@folder:`` path, which was always fast).
"""

import time
from unittest.mock import patch

import pytest

from tui_gateway.server import _build_image_ref_message


@pytest.fixture()
def img(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff fake jpeg")
    return p


def test_never_calls_vision_on_submit_path(img):
    """The whole point of the fix: zero vision traffic before dispatch."""

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("submit path called the vision tool")

    with patch("tools.vision_tools.vision_analyze_tool", _boom):
        out = _build_image_ref_message("what is this?", [str(img)])

    assert "what is this?" in out
    assert str(img) in out
    assert "vision_analyze" in out


def test_returns_instantly_for_many_images(tmp_path):
    """No per-image network round-trips: 20 images in well under a second."""
    paths = []
    for i in range(20):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n fake")
        paths.append(str(p))

    start = time.monotonic()
    out = _build_image_ref_message("caption", paths)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    for p in paths:
        assert p in out


def test_missing_paths_skipped(tmp_path, img):
    gone = tmp_path / "nope.png"
    out = _build_image_ref_message("hi", [str(gone), str(img)])
    assert str(gone) not in out
    assert str(img) in out


def test_text_preserved_and_trails_refs(img):
    out = _build_image_ref_message("my caption", [str(img)])
    assert out.index("vision_analyze") < out.index("my caption")


def test_no_text_defaults_to_question_when_no_valid_images(tmp_path):
    out = _build_image_ref_message("", [str(tmp_path / "missing.png")])
    assert out == "What do you see in this image?"


def test_no_text_with_image_yields_refs_only(img):
    out = _build_image_ref_message("", [str(img)])
    assert str(img) in out
    assert out.strip().startswith("[The user attached an image")
