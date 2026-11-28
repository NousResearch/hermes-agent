"""Dimension caps should preserve resolution before the byte-budget ladder."""

import base64
from io import BytesIO
import random

import pytest

pytest.importorskip("PIL")
from PIL import Image

from tools.vision_tools import _resize_image_for_vision


@pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (4000, 2000)])
def test_dimension_only_resize_fits_long_edge_and_reports_scale(tmp_path, size):
    source = tmp_path / "capture.png"
    Image.new("RGB", size, (20, 40, 60)).save(source)
    original = source.read_bytes()
    scale = {}
    result = _resize_image_for_vision(
        source, mime_type="image/png", max_dimension=1568,
        max_base64_bytes=256 * 1024, scale_out=scale,
    )
    with Image.open(BytesIO(base64.b64decode(result.partition(",")[2]))) as output:
        assert max(output.size) == 1568
        assert abs(output.width / output.height - size[0] / size[1]) < 0.01
        assert scale == dict(orig_width=size[0], orig_height=size[1],
                             new_width=output.width, new_height=output.height)
    assert len(result) <= 256 * 1024
    assert source.read_bytes() == original


def test_byte_budget_can_shrink_further_after_dimension_fit(tmp_path):
    source = tmp_path / "dense.png"
    Image.frombytes("RGB", (1600, 900), random.Random(0).randbytes(1600 * 900 * 3)).save(source)
    result = _resize_image_for_vision(
        source, mime_type="image/png", max_dimension=1568,
        max_base64_bytes=256 * 1024,
    )
    with Image.open(BytesIO(base64.b64decode(result.partition(",")[2]))) as output:
        assert max(output.size) < 1568
    assert len(result) <= 256 * 1024
