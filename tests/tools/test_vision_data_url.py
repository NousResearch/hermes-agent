"""Regression tests for vision tools accepting base64 data: URLs.

Agents routinely hand `vision_analyze` an inline base64 image — e.g. one
just produced by `image_generate` and QA'd in the same turn. Before
`_data_url_to_temp_file`, those calls were rejected as an "invalid image
source" because the resolver only accepted http(s) URLs and local paths.
"""

from __future__ import annotations

from tools.vision_tools import _data_url_to_temp_file

# A real 1x1 PNG as a data URL.
_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_data_url_decoded_to_temp_file():
    path = _data_url_to_temp_file(_PNG_DATA_URL)
    try:
        assert path is not None
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.suffix == ".png"
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def test_jpeg_data_url_uses_jpg_extension():
    # 'image/jpeg' should normalize to a .jpg extension.
    tiny_jpeg = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD"
    path = _data_url_to_temp_file(tiny_jpeg)
    try:
        assert path is not None and path.suffix == ".jpg"
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def test_non_data_url_returns_none():
    assert _data_url_to_temp_file("https://example.com/a.jpg") is None
    assert _data_url_to_temp_file("/tmp/local-image.jpg") is None


def test_malformed_data_url_returns_none():
    assert _data_url_to_temp_file("data:image/png;base64,") is None
