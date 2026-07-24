"""Tests for the QQ空间 (QZone) 说说 publishing tool.

Tests real logic: g_tk computation, cookie parsing, publish-form building,
QZone response parsing, availability gating, the OneBot HTTP client, and
handler validation — all with mocked HTTP so no network is touched.
"""

import json
from unittest.mock import patch

import pytest

from tools import qzone_tool as qz
from tools.qzone_tool import (
    _build_publish_form,
    _build_richval,
    _build_upload_form,
    _check_qzone_available,
    _compute_gtk,
    _download_image,
    _extract_cookie_value,
    _extract_pic_info,
    _generate_image,
    _handle_qzone_publish,
    _load_image_reference,
    _onebot_call,
    _parse_publish_response,
    _parse_upload_response,
    _read_image_file,
)


# ---------------------------------------------------------------------------
# Fake HTTP response
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# g_tk computation
# ---------------------------------------------------------------------------

class TestComputeGtk:
    def test_empty_skey_is_seed(self):
        # With no characters the hash stays at the DJB seed 5381.
        assert _compute_gtk("") == 5381

    def test_known_value(self):
        # Locked-in regression value for the standard QZone hash.
        assert _compute_gtk("test") == 2090756197

    def test_result_fits_31_bits(self):
        for skey in ("p_skey_abc", "ZZZZZZZZZZ", "@#$%^&*()"):
            assert 0 <= _compute_gtk(skey) <= 0x7FFFFFFF

    def test_deterministic(self):
        assert _compute_gtk("abc123") == _compute_gtk("abc123")


# ---------------------------------------------------------------------------
# Cookie parsing
# ---------------------------------------------------------------------------

class TestExtractCookieValue:
    COOKIE = "uin=o0010001; skey=@abcDEF; p_skey=PpKkEeYy; pt4_token=tok123"

    def test_extracts_middle_key(self):
        assert _extract_cookie_value(self.COOKIE, "skey") == "@abcDEF"

    def test_extracts_p_skey(self):
        assert _extract_cookie_value(self.COOKIE, "p_skey") == "PpKkEeYy"

    def test_extracts_first_key(self):
        assert _extract_cookie_value(self.COOKIE, "uin") == "o0010001"

    def test_missing_key_returns_none(self):
        assert _extract_cookie_value(self.COOKIE, "nope") is None

    def test_empty_string_returns_none(self):
        assert _extract_cookie_value("", "p_skey") is None

    def test_value_containing_equals_sign(self):
        # partition() keeps everything after the first '=' in the value.
        assert _extract_cookie_value("token=a=b=c; x=1", "token") == "a=b=c"


# ---------------------------------------------------------------------------
# Publish form building
# ---------------------------------------------------------------------------

class TestBuildPublishForm:
    def test_text_goes_into_con(self):
        form = _build_publish_form("今天天气真好", "10001")
        assert form["con"] == "今天天气真好"

    def test_hostuin_is_stringified(self):
        form = _build_publish_form("hi", 10001)
        assert form["hostuin"] == "10001"

    def test_qzreferrer_includes_uin(self):
        form = _build_publish_form("hi", "10001")
        assert form["qzreferrer"] == "https://user.qzone.qq.com/10001"

    def test_format_is_json(self):
        assert _build_publish_form("hi", "10001")["format"] == "json"

    def test_no_images_keeps_richtype_empty(self):
        form = _build_publish_form("hi", "10001")
        assert form["richtype"] == ""
        assert form["richval"] == ""

    def test_with_images_sets_richtype_and_richval(self):
        pics = [{"albumid": "a1", "lloc": "l1", "sloc": "s1", "type": 0,
                 "width": 800, "height": 600}]
        form = _build_publish_form("hi", "10001", pics)
        assert form["richtype"] == "1"
        assert form["richval"] != ""
        assert "a1" in form["richval"]


# ---------------------------------------------------------------------------
# QZone response parsing
# ---------------------------------------------------------------------------

class TestParsePublishResponse:
    def test_success_with_tid(self):
        result = _parse_publish_response(b'{"ret":0,"tid":"feedtid123"}')
        assert result["ok"] is True
        assert result["tid"] == "feedtid123"

    def test_success_t1_tid_fallback(self):
        result = _parse_publish_response(b'{"ret":0,"t1_tid":"alt456"}')
        assert result["ok"] is True
        assert result["tid"] == "alt456"

    def test_success_str_input(self):
        result = _parse_publish_response('{"ret":0,"tid":"x"}')
        assert result["ok"] is True

    def test_jsonp_callback_wrapper_is_stripped(self):
        raw = b'_Callback({"ret":0,"tid":"wrapped789"});'
        result = _parse_publish_response(raw)
        assert result["ok"] is True
        assert result["tid"] == "wrapped789"

    def test_error_ret_nonzero(self):
        result = _parse_publish_response(b'{"ret":-3000,"msg":"need verify"}')
        assert result["ok"] is False
        assert result["code"] == -3000
        assert "need verify" in result["error"]

    def test_error_nonzero_subcode(self):
        result = _parse_publish_response(b'{"ret":0,"subcode":-4001,"msg":"bad"}')
        assert result["ok"] is False

    def test_unparseable_response(self):
        result = _parse_publish_response(b"<html>403 Forbidden</html>")
        assert result["ok"] is False
        assert "unparseable" in result["error"]

    def test_empty_response(self):
        result = _parse_publish_response(b"")
        assert result["ok"] is False

    def test_success_with_code_field(self):
        # Verified live: emotion_cgi_publish_v6 returns `code` (no `ret`).
        raw = b'{"attach":"","code":0,"subcode":0,"tid":"1cbe3d3c17","feedinfo":"<li>"}'
        result = _parse_publish_response(raw)
        assert result["ok"] is True
        assert result["tid"] == "1cbe3d3c17"

    def test_error_code_nonzero(self):
        result = _parse_publish_response(
            b'{"code":-10000,"message":"\xe4\xbd\xbf\xe7\x94\xa8\xe4\xba\xba\xe6\x95\xb0\xe8\xbf\x87\xe5\xa4\x9a"}'
        )
        assert result["ok"] is False
        assert result["code"] == -10000

    def test_code_success_with_bad_subcode(self):
        # A non-zero subcode is still an error even when code is 0.
        result = _parse_publish_response(b'{"code":0,"subcode":-4001,"msg":"verify"}')
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------

class TestCheckAvailable:
    def test_available_when_url_set(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        assert _check_qzone_available() is True

    def test_unavailable_when_url_unset(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        assert _check_qzone_available() is False

    def test_unavailable_when_url_blank(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "   ")
        assert _check_qzone_available() is False


# ---------------------------------------------------------------------------
# OneBot HTTP client
# ---------------------------------------------------------------------------

class TestOnebotCall:
    def test_raises_when_url_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_HTTP_URL", raising=False)
        with pytest.raises(RuntimeError, match="ONEBOT_HTTP_URL"):
            _onebot_call("get_login_info")

    def test_returns_data_on_success(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        body = json.dumps({"status": "ok", "retcode": 0, "data": {"user_id": 42}}).encode()
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            data = _onebot_call("get_login_info")
        assert data == {"user_id": 42}

    def test_raises_on_failed_status(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        body = json.dumps({"status": "failed", "retcode": 1404, "message": "no login"}).encode()
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            with pytest.raises(RuntimeError, match="no login"):
                _onebot_call("get_login_info")

    def test_raises_on_missing_data(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HTTP_URL", "http://127.0.0.1:3000")
        body = json.dumps({"status": "ok", "retcode": 0}).encode()
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            with pytest.raises(RuntimeError, match="no data"):
                _onebot_call("get_login_info")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

_GOOD_COOKIE = "uin=o0010001; skey=@abc; p_skey=PSKEYVALUE; pt4_token=t"


class TestHandler:
    def test_empty_text_no_images_rejected(self):
        result = json.loads(_handle_qzone_publish({"text": "   "}))
        assert "error" in result
        assert "'text', 'images', or 'generate'" in result["error"]

    def test_no_args_rejected(self):
        result = json.loads(_handle_qzone_publish({}))
        assert "error" in result
        assert "'text', 'images', or 'generate'" in result["error"]

    def test_happy_path(self):
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE), \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":0,"tid":"feed999"}') as post:
            result = json.loads(_handle_qzone_publish({"text": "hello world"}))
        assert result["success"] is True
        assert result["tid"] == "feed999"
        assert result["uin"] == "10001"
        # g_tk passed to the request must be the hash of p_skey from the cookie.
        assert post.call_args.args[1] == _compute_gtk("PSKEYVALUE")

    def test_onebot_unreachable(self):
        with patch.object(qz, "_get_login_uin", side_effect=RuntimeError("connection refused")):
            result = json.loads(_handle_qzone_publish({"text": "hi"}))
        assert "error" in result
        assert "borrow QQ login state" in result["error"]

    def test_missing_p_skey(self):
        cookie_without_pskey = "uin=o0010001; skey=@abc"
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=cookie_without_pskey):
            result = json.loads(_handle_qzone_publish({"text": "hi"}))
        assert "error" in result
        assert "p_skey not found" in result["error"]

    def test_qzone_rejects_post(self):
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE), \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":-4001,"msg":"login expired"}'):
            result = json.loads(_handle_qzone_publish({"text": "hi"}))
        assert "error" in result
        assert "login expired" in result["error"]
        assert result["code"] == -4001

    def test_qzone_request_exception(self):
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE), \
             patch.object(qz, "_qzone_publish_post", side_effect=RuntimeError("timeout")):
            result = json.loads(_handle_qzone_publish({"text": "hi"}))
        assert "error" in result
        assert "publish request failed" in result["error"]


# ---------------------------------------------------------------------------
# Local image files
# ---------------------------------------------------------------------------

class TestReadImageFile:
    def test_reads_valid_image(self, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-data")
        data, name = _read_image_file(str(img))
        assert data == b"\x89PNG\r\n\x1a\nfake-image-data"
        assert name == "pic.png"

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _read_image_file(str(tmp_path / "nope.png"))

    def test_unsupported_extension(self, tmp_path):
        bad = tmp_path / "doc.pdf"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="unsupported image type"):
            _read_image_file(str(bad))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            _read_image_file(str(empty))

    def test_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qz, "_MAX_IMAGE_BYTES", 4)
        big = tmp_path / "big.jpg"
        big.write_bytes(b"way too many bytes")
        with pytest.raises(ValueError, match="too large"):
            _read_image_file(str(big))


# ---------------------------------------------------------------------------
# Image upload form + response parsing
# ---------------------------------------------------------------------------

class TestBuildUploadForm:
    def test_picfile_carries_base64(self):
        form = _build_upload_form("BASE64DATA", "p.png", "10001", "sk", "psk", 123)
        assert form["picfile"] == "BASE64DATA"
        assert form["base64"] == "1"

    def test_uin_fields_populated(self):
        form = _build_upload_form("x", "p.png", "10001", "sk", "psk", 123)
        assert form["uin"] == form["p_uin"] == form["zzpaneluin"] == "10001"

    def test_url_includes_gtk(self):
        form = _build_upload_form("x", "p.png", "10001", "sk", "psk", 999)
        assert form["url"].endswith("g_tk=999")


class TestParseUploadResponse:
    def test_success_extracts_pic(self):
        raw = (b'frameElement.callback({"ret":0,"data":'
               b'{"albumid":"al1","lloc":"ll1","sloc":"sl1","width":800,"height":600}});')
        result = _parse_upload_response(raw)
        assert result["ok"] is True
        assert result["pic"]["albumid"] == "al1"
        assert result["pic"]["width"] == 800

    def test_error_ret_nonzero(self):
        result = _parse_upload_response(b'{"ret":-1000,"msg":"upload denied"}')
        assert result["ok"] is False
        assert "upload denied" in result["error"]

    def test_unparseable(self):
        result = _parse_upload_response(b"<html>500</html>")
        assert result["ok"] is False
        assert "unparseable" in result["error"]


class TestExtractPicInfo:
    def test_direct_fields(self):
        pic = _extract_pic_info({"albumid": "a", "lloc": "l", "sloc": "s",
                                 "width": 100, "height": 200})
        assert pic == {"albumid": "a", "lloc": "l", "sloc": "s", "type": 0,
                       "width": 100, "height": 200, "url": ""}

    def test_photoid_fallback_for_lloc_sloc(self):
        pic = _extract_pic_info({"albumid": "a", "photoid": "pid"})
        assert pic["lloc"] == "pid"
        assert pic["sloc"] == "pid"

    def test_pre_fallback_for_url(self):
        pic = _extract_pic_info({"pre": "https://img/pre.jpg"})
        assert pic["url"] == "https://img/pre.jpg"


class TestBuildRichval:
    def test_single_image_segment(self):
        rv = _build_richval([{"albumid": "a1", "lloc": "l1", "sloc": "s1",
                              "type": 0, "width": 800, "height": 600}])
        assert rv == ",a1,l1,s1,0,600,800,,600,800"

    def test_multiple_images_tab_joined(self):
        pics = [
            {"albumid": "a1", "lloc": "l1", "sloc": "s1", "type": 0,
             "width": 1, "height": 2},
            {"albumid": "a2", "lloc": "l2", "sloc": "s2", "type": 0,
             "width": 3, "height": 4},
        ]
        rv = _build_richval(pics)
        assert rv.count("\t") == 1
        assert rv.split("\t")[1] == ",a2,l2,s2,0,4,3,,4,3"

    def test_empty_list(self):
        assert _build_richval([]) == ""


# ---------------------------------------------------------------------------
# Handler — image attachments
# ---------------------------------------------------------------------------

_GOOD_COOKIE_IMG = "uin=o0010001; skey=SKEYVAL; p_skey=PSKEYVALUE; pt4_token=t"
_FAKE_PIC = {"albumid": "a1", "lloc": "l1", "sloc": "s1", "type": 0,
             "width": 800, "height": 600}


class TestHandlerImages:
    def _make_image(self, tmp_path, name="pic.png"):
        img = tmp_path / name
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return str(img)

    def test_image_only_post_allowed(self, tmp_path):
        path = self._make_image(tmp_path)
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE_IMG), \
             patch.object(qz, "_upload_image", return_value=_FAKE_PIC), \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":0,"tid":"t1"}'):
            result = json.loads(_handle_qzone_publish({"images": [path]}))
        assert result["success"] is True
        assert result["images"] == 1

    def test_text_and_images(self, tmp_path):
        paths = [self._make_image(tmp_path, "a.png"), self._make_image(tmp_path, "b.jpg")]
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE_IMG), \
             patch.object(qz, "_upload_image", return_value=_FAKE_PIC) as up, \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":0,"tid":"t1"}'):
            result = json.loads(_handle_qzone_publish({"text": "看图", "images": paths}))
        assert result["success"] is True
        assert result["images"] == 2
        assert up.call_count == 2

    def test_single_image_path_as_string(self, tmp_path):
        path = self._make_image(tmp_path)
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE_IMG), \
             patch.object(qz, "_upload_image", return_value=_FAKE_PIC), \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":0,"tid":"t1"}'):
            result = json.loads(_handle_qzone_publish({"text": "hi", "images": path}))
        assert result["success"] is True

    def test_bad_image_path_fails_before_network(self, tmp_path):
        # No OneBot patch — if it tried the network the test would error out.
        result = json.loads(_handle_qzone_publish(
            {"text": "hi", "images": [str(tmp_path / "missing.png")]}))
        assert "error" in result
        assert "not found" in result["error"]

    def test_too_many_images_rejected(self):
        result = json.loads(_handle_qzone_publish(
            {"text": "hi", "images": [f"/tmp/{i}.png" for i in range(10)]}))
        assert "error" in result
        assert "at most" in result["error"]

    def test_upload_failure_surfaced(self, tmp_path):
        path = self._make_image(tmp_path)
        with patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE_IMG), \
             patch.object(qz, "_upload_image", side_effect=RuntimeError("album full")):
            result = json.loads(_handle_qzone_publish({"text": "hi", "images": [path]}))
        assert "error" in result
        assert "album full" in result["error"]


# ---------------------------------------------------------------------------
# AI image generation
# ---------------------------------------------------------------------------

class TestDownloadImage:
    def test_downloads_and_names_from_url(self):
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(b"imgdata")):
            data, name = _download_image("https://img.example.com/path/cat.png")
        assert data == b"imgdata"
        assert name == "cat.png"

    def test_non_image_extension_falls_back(self):
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(b"x")):
            _, name = _download_image("https://img.example.com/render?id=9")
        assert name == "generated.png"

    def test_empty_download_rejected(self):
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(b"")):
            with pytest.raises(RuntimeError, match="empty"):
                _download_image("https://img.example.com/a.png")

    def test_oversized_download_rejected(self, monkeypatch):
        monkeypatch.setattr(qz, "_MAX_IMAGE_BYTES", 4)
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(b"too many bytes")):
            with pytest.raises(RuntimeError, match="too large"):
                _download_image("https://img.example.com/a.png")


class TestLoadImageReference:
    def test_http_url_delegates_to_download(self):
        with patch.object(qz, "_download_image", return_value=(b"d", "n.png")) as dl:
            result = _load_image_reference("https://img.example.com/n.png")
        assert result == (b"d", "n.png")
        dl.assert_called_once()

    def test_local_path_delegates_to_read(self, tmp_path):
        img = tmp_path / "local.png"
        img.write_bytes(b"\x89PNGlocal")
        data, name = _load_image_reference(str(img))
        assert data == b"\x89PNGlocal"
        assert name == "local.png"

    def test_unusable_reference_raises(self):
        with pytest.raises(RuntimeError, match="unusable reference"):
            _load_image_reference("not-a-url-or-file")


class TestGenerateImage:
    def test_success_with_dict_result(self):
        with patch("tools.image_generation_tool.check_image_generation_requirements",
                   return_value=True), \
             patch("tools.image_generation_tool._handle_image_generate",
                   return_value={"success": True, "image": "https://img/x.png"}), \
             patch.object(qz, "_load_image_reference", return_value=(b"gen", "x.png")):
            data, name = _generate_image("a red panda", "square")
        assert data == b"gen"
        assert name == "x.png"

    def test_success_with_json_string_result(self):
        payload = json.dumps({"success": True, "image": "/tmp/gen.png"})
        with patch("tools.image_generation_tool.check_image_generation_requirements",
                   return_value=True), \
             patch("tools.image_generation_tool._handle_image_generate",
                   return_value=payload), \
             patch.object(qz, "_load_image_reference", return_value=(b"gen", "gen.png")):
            data, name = _generate_image("prompt", "square")
        assert (data, name) == (b"gen", "gen.png")

    def test_no_backend_configured(self):
        with patch("tools.image_generation_tool.check_image_generation_requirements",
                   return_value=False):
            with pytest.raises(RuntimeError, match="no image-generation backend"):
                _generate_image("prompt", "square")

    def test_backend_error_surfaced(self):
        with patch("tools.image_generation_tool.check_image_generation_requirements",
                   return_value=True), \
             patch("tools.image_generation_tool._handle_image_generate",
                   return_value={"error": "quota exceeded", "image": None}):
            with pytest.raises(RuntimeError, match="quota exceeded"):
                _generate_image("prompt", "square")

    def test_no_image_in_result(self):
        with patch("tools.image_generation_tool.check_image_generation_requirements",
                   return_value=True), \
             patch("tools.image_generation_tool._handle_image_generate",
                   return_value={"success": True, "image": None}):
            with pytest.raises(RuntimeError, match="no image"):
                _generate_image("prompt", "square")


# ---------------------------------------------------------------------------
# Handler — AI-generated images
# ---------------------------------------------------------------------------

class TestHandlerGenerate:
    def test_generate_only_post(self):
        with patch.object(qz, "_generate_image", return_value=(b"gen", "g.png")), \
             patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE_IMG), \
             patch.object(qz, "_upload_image", return_value=_FAKE_PIC), \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":0,"tid":"t1"}'):
            result = json.loads(_handle_qzone_publish({"generate": "a sunset"}))
        assert result["success"] is True
        assert result["images"] == 1
        assert result["generated"] is True

    def test_generate_with_text(self):
        with patch.object(qz, "_generate_image", return_value=(b"gen", "g.png")) as gen, \
             patch.object(qz, "_get_login_uin", return_value="10001"), \
             patch.object(qz, "_get_qzone_cookie_string", return_value=_GOOD_COOKIE_IMG), \
             patch.object(qz, "_upload_image", return_value=_FAKE_PIC), \
             patch.object(qz, "_qzone_publish_post", return_value=b'{"ret":0,"tid":"t1"}'):
            result = json.loads(_handle_qzone_publish(
                {"text": "今天的画", "generate": "a landscape", "aspect_ratio": "landscape"}))
        assert result["success"] is True
        # aspect_ratio must be forwarded to image generation.
        assert gen.call_args.args == ("a landscape", "landscape")

    def test_generation_failure_surfaced(self):
        with patch.object(qz, "_generate_image", side_effect=RuntimeError("model down")):
            result = json.loads(_handle_qzone_publish({"generate": "a cat"}))
        assert "error" in result
        assert "Image generation failed" in result["error"]
        assert "model down" in result["error"]

    def test_generate_plus_images_counts_toward_limit(self):
        # 9 attached images + 1 generated = 10, over the cap — rejected on
        # the count check before any file read or generation.
        args = {"images": [f"/tmp/{i}.png" for i in range(9)], "generate": "extra"}
        result = json.loads(_handle_qzone_publish(args))
        assert "error" in result
        assert "at most" in result["error"]

    def test_non_string_generate_rejected(self):
        result = json.loads(_handle_qzone_publish({"generate": 123}))
        assert "error" in result
        assert "must be a text prompt" in result["error"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tool_is_registered(self):
        from tools.registry import registry
        assert registry.get_toolset_for_tool("qzone_publish") == "qzone"
