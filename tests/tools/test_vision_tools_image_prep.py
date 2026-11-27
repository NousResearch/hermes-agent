"""Tests for the full-decode validation of raster images fed to vision_analyze.

An MPF (Multi-Picture Format) header can claim follow-on frames the file does
not actually carry — phone photos round-tripped through Picasa-era tools do
this — and Pillow raises while seeking the claimed frame even though frame 1
is fully decodable (and the embed path only ever re-encodes frame 1). The
validator must still reject a broken FIRST frame: a timed-out download looks
like a valid container with a truncated pixel stream and must fail loudly.
"""

from __future__ import annotations

import struct

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from tools.vision_tools_image_prep import _validate_raster_image_decodable

pytestmark = pytest.mark.skipif(Image is None, reason="Pillow not installed")


def _write_two_frame_mpo(path):
    first = Image.new("RGB", (40, 30), "red")
    second = Image.new("RGB", (20, 15), "blue")
    first.save(path, format="MPO", save_all=True, append_images=[second])
    return path


def _patch_second_frame_offset(path, new_offset):
    """Rewrite the MPF MPEntry data offset of frame 2, leaving frame 1 intact."""
    data = bytearray(path.read_bytes())
    marker = data.find(b"MPF\x00")
    assert marker != -1, "test MPO has no MPF APP2 segment"
    tiff = marker + 4
    bo = "<" if bytes(data[tiff : tiff + 2]) == b"II" else ">"
    ifd = tiff + struct.unpack_from(bo + "I", data, tiff + 4)[0]
    entry_count = struct.unpack_from(bo + "H", data, ifd)[0]
    for i in range(entry_count):
        entry = ifd + 2 + i * 12
        if struct.unpack_from(bo + "H", data, entry)[0] == 0xB002:  # MPEntry tag
            entries = tiff + struct.unpack_from(bo + "I", data, entry + 8)[0]
            # MPEntry record layout: attributes(4) size(4) data_offset(4) ...
            struct.pack_into(bo + "I", data, entries + 16 + 8, new_offset)
            path.write_bytes(bytes(data))
            return
    raise AssertionError("test MPO has no MPEntry tag")


def test_valid_two_frame_mpo_validates_clean(tmp_path):
    path = _write_two_frame_mpo(tmp_path / "two_frame.mpo")
    assert _validate_raster_image_decodable(path) is None


def test_claimed_second_frame_past_eof_is_not_rejected(tmp_path):
    # Pillow raises ValueError("No data found for frame") while seeking the
    # claimed frame 2; frame 1 is intact, so the image must stay usable.
    path = _write_two_frame_mpo(tmp_path / "dangling.mpo")
    _patch_second_frame_offset(path, 0x40000000)  # past EOF
    assert _validate_raster_image_decodable(path) is None


def test_claimed_second_frame_into_frame_one_is_not_rejected(tmp_path):
    # Pillow raises SyntaxError("not a JPEG file") when the claimed frame 2
    # offset lands mid-stream; same reasoning as above.
    path = _write_two_frame_mpo(tmp_path / "mid.mpo")
    _patch_second_frame_offset(path, 40)  # inside frame 1, not an SOI
    assert _validate_raster_image_decodable(path) is None


def test_truncated_first_frame_is_still_rejected(tmp_path):
    path = tmp_path / "truncated.png"
    img = Image.new("RGB", (60, 40), "green")
    img.save(path, format="PNG")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # cut through the pixel stream
    error = _validate_raster_image_decodable(path)
    assert error is not None and error.startswith("Image could not be fully decoded:")


def test_frame_budget_still_rejects_long_animations(tmp_path):
    path = _write_two_frame_mpo(tmp_path / "two_frame.mpo")
    error = _validate_raster_image_decodable(path, max_frames=1)
    assert error is not None and error.startswith(
        "Image validation rejected animation:"
    )


def test_pixel_budget_still_rejects_large_animations(tmp_path):
    path = _write_two_frame_mpo(tmp_path / "two_frame.mpo")
    error = _validate_raster_image_decodable(path, max_pixels=40 * 30 - 1)
    assert error is not None and error.startswith(
        "Image validation rejected animation:"
    )
