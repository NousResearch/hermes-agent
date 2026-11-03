"""A computer-use capture is compressed before it is baked into the tool result (#135497).

A capture embedded in a tool result is re-sent on EVERY later turn, so one raw 1568x887 PNG
(~1.9 MB base64, ~476K billed tokens) can wedge a whole session. The embed rides the same
1568-long-edge / JPEG-ladder budget the vision path already enforces, and never hands the
wire a payload larger than the raw capture.
"""
from __future__ import annotations

import base64
import io
import random

import pytest

from tools.computer_use.backend import CaptureResult
from tools.computer_use.tool import _embed_capture_image_url

PIL = pytest.importorskip("PIL.Image")


def _png_bytes(size=(1800, 1400), noisy=True) -> bytes:
    img = PIL.new("RGB", size, (200, 40, 40))
    if noisy:
        rnd = random.Random(13)
        img.putdata([
            (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
            for _ in range(size[0] * size[1])
        ])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _capture(raw: bytes) -> CaptureResult:
    return CaptureResult(
        mode="vision", width=1800, height=1400,
        png_b64=base64.b64encode(raw).decode(), image_mime_type="image/png",
        png_bytes_len=len(raw),
    )


def _long_edge(data_url: str) -> int:
    with PIL.open(io.BytesIO(base64.b64decode(data_url.partition(",")[2]))) as img:
        return max(img.size)


def test_oversized_capture_is_embedded_smaller(tmp_path):
    raw = _png_bytes()
    src = tmp_path / "capture.png"
    src.write_bytes(raw)

    url = _embed_capture_image_url(_capture(raw), str(src))

    assert url.startswith("data:")
    assert len(url) < (len(raw) * 4) // 3
    assert _long_edge(url) <= 1568


def test_embed_falls_back_to_the_raw_capture_when_no_source_file(tmp_path):
    """No screenshot on disk still budgets: the encoder writes its own temp source."""
    raw = _png_bytes(size=(1800, 1400))
    url = _embed_capture_image_url(_capture(raw), None)

    assert url.startswith("data:")
    assert _long_edge(url) <= 1568
