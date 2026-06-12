"""Tests for the Avocado image-upload bridge tool (AVOCADO FORK)."""

import json
from unittest.mock import patch

import tools.avocado_upload_tool as aut
from hermes_constants import get_hermes_dir


_GRANT = {
    "fileId": "mcp-source/user_x/abc.jpg",
    "uploadUrl": "https://example.supabase.co/storage/upload/sign/x?token=t",
    "maxSizeMb": 12,
}

_JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 64
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _FakePrepareEntry:
    name = "mcp_avocado_prepare_image_upload"

    @staticmethod
    def handler(args, **kwargs):
        # Real shape observed live: JSON payload as a text block, wrapped
        # in the MCP handler's {"result": ...} envelope.
        return json.dumps({"result": json.dumps(_GRANT)})


def _cached_image(data=_JPEG_BYTES, name="img_test.jpg"):
    cache_dir = get_hermes_dir("cache/images", "image_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    path.write_bytes(data)
    return path


class TestExtractUploadGrant:
    def test_result_as_json_text(self):
        raw = json.dumps({"result": json.dumps(_GRANT)})
        grant, err = aut._extract_upload_grant(raw)
        assert err is None
        assert grant["fileId"] == _GRANT["fileId"]
        assert grant["uploadUrl"] == _GRANT["uploadUrl"]

    def test_structured_content(self):
        raw = json.dumps({"result": "ok", "structuredContent": _GRANT})
        grant, err = aut._extract_upload_grant(raw)
        assert err is None
        assert grant["fileId"] == _GRANT["fileId"]

    def test_error_envelope(self):
        grant, err = aut._extract_upload_grant(json.dumps({"error": "boom"}))
        assert grant is None
        assert "boom" in err

    def test_regex_fallback(self):
        raw = 'preamble {"uploadUrl": "https://u", "fileId": "f1", "maxSizeMb": 12} trailer'
        grant, err = aut._extract_upload_grant(raw)
        assert err is None
        assert grant == {"uploadUrl": "https://u", "fileId": "f1", "maxSizeMb": 12.0}

    def test_no_grant(self):
        grant, err = aut._extract_upload_grant(json.dumps({"result": "nope"}))
        assert grant is None
        assert err


class TestDetectImageMime:
    def test_known_formats(self):
        assert aut._detect_image_mime(_PNG_BYTES) == "image/png"
        assert aut._detect_image_mime(_JPEG_BYTES) == "image/jpeg"
        webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 8
        assert aut._detect_image_mime(webp) == "image/webp"

    def test_unsupported(self):
        assert aut._detect_image_mime(b"GIF89a" + b"\x00" * 16) is None
        assert aut._detect_image_mime(b"not an image") is None


class TestHandler:
    def test_no_mcp_server(self):
        with patch.object(aut, "_find_prepare_entry", return_value=None):
            result = json.loads(
                aut._handle_avocado_upload_image({"file_path": "/tmp/x.jpg"})
            )
        assert result["success"] is False
        assert "Avocado MCP" in result["error"]

    def test_missing_file(self):
        with patch.object(aut, "_find_prepare_entry", return_value=_FakePrepareEntry):
            result = json.loads(
                aut._handle_avocado_upload_image({"file_path": "/tmp/missing.jpg"})
            )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_path_outside_cache_rejected(self, tmp_path):
        outside = tmp_path / "evil.jpg"
        outside.write_bytes(_JPEG_BYTES)
        with patch.object(aut, "_find_prepare_entry", return_value=_FakePrepareEntry):
            result = json.loads(
                aut._handle_avocado_upload_image({"file_path": str(outside)})
            )
        assert result["success"] is False
        assert "media cache" in result["error"]

    def test_non_image_bytes_rejected(self):
        path = _cached_image(b"definitely not an image, e.g. an env file")
        with patch.object(aut, "_find_prepare_entry", return_value=_FakePrepareEntry):
            result = json.loads(
                aut._handle_avocado_upload_image({"file_path": str(path)})
            )
        assert result["success"] is False
        assert "Unsupported image format" in result["error"]

    def test_happy_path_uploads_and_returns_file_id(self):
        path = _cached_image()

        class _Resp:
            status_code = 200
            text = '{"Key": "ok"}'

        with patch.object(aut, "_find_prepare_entry", return_value=_FakePrepareEntry), \
                patch("httpx.put", return_value=_Resp()) as mock_put:
            result = json.loads(
                aut._handle_avocado_upload_image(
                    {"file_path": str(path), "purpose": "video"}
                )
            )

        assert result["success"] is True
        assert result["file_id"] == _GRANT["fileId"]
        assert result["mime_type"] == "image/jpeg"
        assert result["purpose"] == "video"
        assert "mcp_avocado_edit_image" in result["next_step"]
        assert "mcp_avocado_generate_video" in result["next_step"]

        (url,), kwargs = mock_put.call_args
        assert url == _GRANT["uploadUrl"]
        assert kwargs["headers"]["Content-Type"] == "image/jpeg"
        assert kwargs["content"] == _JPEG_BYTES

    def test_upload_http_error(self):
        path = _cached_image()

        class _Resp:
            status_code = 403
            text = "signature expired"

        with patch.object(aut, "_find_prepare_entry", return_value=_FakePrepareEntry), \
                patch("httpx.put", return_value=_Resp()):
            result = json.loads(
                aut._handle_avocado_upload_image({"file_path": str(path)})
            )
        assert result["success"] is False
        assert "403" in result["error"]

    def test_oversize_vs_grant_limit(self):
        path = _cached_image(_JPEG_BYTES + b"\x00" * (13 * 1024 * 1024))
        with patch.object(aut, "_find_prepare_entry", return_value=_FakePrepareEntry):
            result = json.loads(
                aut._handle_avocado_upload_image({"file_path": str(path)})
            )
        assert result["success"] is False
        assert "12" in result["error"]
