"""Regression tests for CSV path containment and rest_api_call tool."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# CSV path containment tests
# ---------------------------------------------------------------------------
class TestCsvPathContainment:
    """_validate_csv_path must fail-closed on escapes."""

    def test_relative_path_within_working_dir(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path(str(csv_file), str(tmp_path))
        assert err is None

    def test_path_traversal_escape_rejected(self, tmp_path):
        csv_file = tmp_path / "sub" / ".." / ".." / "etc" / "passwd"
        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path(str(csv_file), str(tmp_path))
        assert err is not None
        assert "outside" in err

    def test_absolute_path_outside_working_dir_rejected(self, tmp_path):
        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path("/etc/passwd", str(tmp_path))
        assert err is not None
        assert "outside" in err

    def test_symlink_escape_rejected(self, tmp_path):
        real = tmp_path / "real.csv"
        real.write_text("a,b\n1,2\n")
        link = tmp_path / "escape"
        link.symlink_to(real)

        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path(str(link), str(tmp_path))
        assert err is None

    def test_working_dir_itself_is_valid(self, tmp_path):
        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path(str(tmp_path), str(tmp_path))
        assert err is None

    def test_output_path_validation_on_export(self, tmp_path):
        out_file = tmp_path / "out.csv"

        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path(str(out_file), str(tmp_path))
        assert err is None

    def test_output_path_escape_rejected(self, tmp_path):
        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path("/tmp/evil.csv", str(tmp_path))
        assert err is not None


# ---------------------------------------------------------------------------
# CSV read/filter/aggregate basic tests
# ---------------------------------------------------------------------------
class TestCsvProcessBasic:
    def test_read(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        from tools.csv_process import csv_process

        result = json.loads(csv_process("read", str(csv_file), working_dir=str(tmp_path)))
        assert result["success"]
        assert result["row_count"] == 2

    def test_filter(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        from tools.csv_process import csv_process

        result = json.loads(csv_process("filter", str(csv_file), filter_condition="age > 28", working_dir=str(tmp_path)))
        assert result["success"]
        assert result["filtered_count"] == 1

    def test_aggregate(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        from tools.csv_process import csv_process

        result = json.loads(csv_process("aggregate", str(csv_file), columns=["age"], working_dir=str(tmp_path)))
        assert result["success"]
        assert result["results"]["age"]["sum"] == 55

    def test_file_not_found(self, tmp_path):
        from tools.csv_process import csv_process

        result = json.loads(csv_process("read", str(tmp_path / "nonexistent.csv"), working_dir=str(tmp_path)))
        assert not result["success"]
        assert "not found" in result["error"]

    def test_unknown_operation(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        from tools.csv_process import csv_process

        result = json.loads(csv_process("bogus", str(csv_file), working_dir=str(tmp_path)))
        assert not result["success"]

    def test_export(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        out_file = tmp_path / "out.csv"

        from tools.csv_process import csv_process

        result = json.loads(csv_process("export", str(csv_file), output_path=str(out_file), filter_condition="age > 28", working_dir=str(tmp_path)))
        assert result["success"]
        assert result["row_count"] == 1
        assert out_file.exists()

    def test_export_without_output_path(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        from tools.csv_process import csv_process

        result = json.loads(csv_process("export", str(csv_file), working_dir=str(tmp_path)))
        assert not result["success"]
        assert "output_path" in result["error"]


# ---------------------------------------------------------------------------
# rest_api_call tests
# ---------------------------------------------------------------------------
class TestRestApiCall:
    def test_success(self):
        resp = MagicMock()
        resp.status = 200
        resp.headers = {"Content-Type": "application/json"}
        resp.read.return_value = json.dumps({"ok": True}).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("tools.rest_api_call.urllib.request.urlopen", return_value=resp):
            from tools.rest_api_call import rest_api_call

            result = json.loads(rest_api_call("https://example.com/api"))
            assert result["success"]
            assert result["status_code"] == 200

    def test_http_error(self):
        error = urllib.error.HTTPError(
            "https://example.com/api", 404, "Not Found", {}, b""
        )
        error.read = MagicMock(return_value=b"")

        with patch("tools.rest_api_call.urllib.request.urlopen", side_effect=error):
            from tools.rest_api_call import rest_api_call

            result = json.loads(rest_api_call("https://example.com/api"))
            assert not result["success"]
            assert result["status_code"] == 404

    def test_url_error(self):
        error = urllib.error.URLError("DNS failure")

        with patch("tools.rest_api_call.urllib.request.urlopen", side_effect=error):
            from tools.rest_api_call import rest_api_call

            result = json.loads(rest_api_call("https://example.com/api"))
            assert not result["success"]
            assert "URL error" in result["error"]

    def test_post_with_body(self):
        resp = MagicMock()
        resp.status = 201
        resp.headers = {"Content-Type": "application/json"}
        resp.read.return_value = json.dumps({"created": True}).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("tools.rest_api_call.urllib.request.urlopen", return_value=resp):
            from tools.rest_api_call import rest_api_call

            result = json.loads(
                rest_api_call(
                    "https://example.com/api",
                    method="POST",
                    body={"key": "value"},
                )
            )
            assert result["success"]
            assert result["status_code"] == 201

    def test_path_traversal_rejected(self, tmp_path):
        """CSV path traversal must be rejected by csv_process."""
        from tools.csv_process import _validate_csv_path

        err = _validate_csv_path(str(tmp_path / ".." / "secret"), str(tmp_path))
        assert err is not None

    def test_non_json_response(self):
        resp = MagicMock()
        resp.status = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.read.return_value = b"<html>hello</html>"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("tools.rest_api_call.urllib.request.urlopen", return_value=resp):
            from tools.rest_api_call import rest_api_call

            result = json.loads(rest_api_call("https://example.com"))
            assert result["success"]
            assert "raw" in result["body"]

    def test_ssl_verify_disabled(self):
        resp = MagicMock()
        resp.status = 200
        resp.headers = {"Content-Type": "application/json"}
        resp.read.return_value = b"{}"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        with patch("tools.rest_api_call.urllib.request.urlopen", return_value=resp) as mock_urlopen:
            from tools.rest_api_call import rest_api_call

            result = json.loads(rest_api_call("https://self-signed.example.com", verify_ssl=False))
            assert result["success"]
            call_args = mock_urlopen.call_args
            ctx = call_args.kwargs.get("context") or call_args[1].get("context")
            assert ctx is not None
