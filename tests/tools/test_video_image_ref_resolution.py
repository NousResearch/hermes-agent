"""Regression tests for video_generate's image-source normalization.

Covers `_resolve_video_image_ref`: agents routinely pass a local file path
as `image_url` (e.g. an image generated earlier in the session), which video
providers reject because they only accept HTTPS or base64 data URLs. The
helper inlines local paths as data URLs and passes real URLs through.
"""

from __future__ import annotations

from tools.video_generation_tool import _resolve_video_image_ref


def test_https_url_passes_through():
    val, err = _resolve_video_image_ref("https://example.com/frame.jpg")
    assert err is None
    assert val == "https://example.com/frame.jpg"


def test_http_url_passes_through():
    val, err = _resolve_video_image_ref("http://example.com/frame.jpg")
    assert err is None
    assert val == "http://example.com/frame.jpg"


def test_data_url_passes_through():
    data = "data:image/png;base64,iVBORw0KGgo="
    val, err = _resolve_video_image_ref(data)
    assert err is None
    assert val == data


def test_local_path_is_inlined_as_data_url(tmp_path):
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes")
    val, err = _resolve_video_image_ref(str(img))
    assert err is None
    assert val is not None and val.startswith("data:image/png;base64,")


def test_missing_path_returns_error():
    val, err = _resolve_video_image_ref("/no/such/image.jpg")
    assert val is None
    assert err is not None and "not found" in err


def test_empty_value_is_noop():
    val, err = _resolve_video_image_ref("")
    assert val is None and err is None
