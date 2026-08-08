"""Tests for browser_upload — the file-upload primitive.

Before this, Hermes's browser tool had no way to attach a file to a file
input (only navigate/click/type/etc.), so "upload a file to a website"
tasks were impossible without bootstrapping a separate Playwright browser.
agent-browser already supports an ``upload <selector> <files...>`` verb;
browser_upload exposes it.
"""

import json

import pytest

import tools.browser_tool as bt


def _call(monkeypatch, *, camofox=False, captured=None, success=True):
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: camofox)
    monkeypatch.setattr(bt, "_last_session_key", lambda x: x)

    def _fake_run(task_id, command, args):
        if captured is not None:
            captured["task_id"] = task_id
            captured["command"] = command
            captured["args"] = args
        return {"success": success} if success else {"success": False, "error": "boom"}

    monkeypatch.setattr(bt, "_run_browser_command", _fake_run)


class TestBrowserUpload:
    def test_camofox_mode_not_supported(self, monkeypatch, tmp_path):
        _call(monkeypatch, camofox=True)
        f = tmp_path / "p.pdf"; f.write_bytes(b"%PDF-1.4")
        out = json.loads(bt.browser_upload("input[type=file]", [str(f)]))
        assert out["success"] is False
        assert "camofox" in out["error"].lower()

    def test_missing_file_errors(self, monkeypatch):
        captured = {}
        _call(monkeypatch, captured=captured)
        out = json.loads(bt.browser_upload("input[type=file]", ["/no/such/file.pdf"]))
        assert out["success"] is False
        assert "not found" in out["error"].lower()
        assert "command" not in captured  # never reached agent-browser

    def test_no_files_errors(self, monkeypatch):
        _call(monkeypatch)
        out = json.loads(bt.browser_upload("input[type=file]", []))
        assert out["success"] is False
        assert "no files" in out["error"].lower()

    def test_single_string_file_normalized_and_uploaded(self, monkeypatch, tmp_path):
        captured = {}
        _call(monkeypatch, captured=captured)
        f = tmp_path / "policy.pdf"; f.write_bytes(b"%PDF-1.4 x")
        out = json.loads(bt.browser_upload("input[type=file]", str(f)))  # bare string, not list
        assert out["success"] is True
        assert captured["command"] == "upload"
        # selector first, then absolute file path(s)
        assert captured["args"][0] == "input[type=file]"
        assert captured["args"][1] == str(f.resolve())
        assert out["uploaded"] == [str(f.resolve())]

    def test_multiple_files_all_passed(self, monkeypatch, tmp_path):
        captured = {}
        _call(monkeypatch, captured=captured)
        a = tmp_path / "a.pdf"; a.write_bytes(b"a")
        b = tmp_path / "b.pdf"; b.write_bytes(b"b")
        out = json.loads(bt.browser_upload("@e5", [str(a), str(b)]))
        assert out["success"] is True
        assert captured["args"] == ["@e5", str(a.resolve()), str(b.resolve())]

    def test_bare_ref_gets_at_prefix_but_css_selector_untouched(self, monkeypatch, tmp_path):
        f = tmp_path / "x.pdf"; f.write_bytes(b"x")
        # bare ref "e7" -> "@e7"
        cap1 = {}
        _call(monkeypatch, captured=cap1)
        bt.browser_upload("e7", [str(f)])
        assert cap1["args"][0] == "@e7"
        # CSS selector left as-is
        cap2 = {}
        _call(monkeypatch, captured=cap2)
        bt.browser_upload("input[type=file]", [str(f)])
        assert cap2["args"][0] == "input[type=file]"
        # already-@ ref left as-is
        cap3 = {}
        _call(monkeypatch, captured=cap3)
        bt.browser_upload("@e9", [str(f)])
        assert cap3["args"][0] == "@e9"

    def test_upload_failure_surfaced(self, monkeypatch, tmp_path):
        _call(monkeypatch, success=False)
        f = tmp_path / "x.pdf"; f.write_bytes(b"x")
        out = json.loads(bt.browser_upload("input[type=file]", [str(f)]))
        assert out["success"] is False


class TestBrowserUploadRegistered:
    def test_schema_present(self):
        assert "browser_upload" in bt._BROWSER_SCHEMA_MAP
        sch = bt._BROWSER_SCHEMA_MAP["browser_upload"]
        assert sch["parameters"]["required"] == ["ref", "files"]

    def test_function_exists(self):
        assert callable(bt.browser_upload)
