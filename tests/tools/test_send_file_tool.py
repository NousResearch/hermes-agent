"""Tests for the send_file tool (issue #466)."""

import json

import pytest

from tools.send_file_tool import send_file_tool


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")


class TestSendFileLocal:
    def test_valid_file_returns_media_tag(self, tmp_path, monkeypatch):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4")
        result = json.loads(send_file_tool(str(f)))
        assert result["success"] is True
        assert result["media_tag"].startswith("MEDIA:")
        assert result["file_path"].endswith("report.pdf")

    def test_caption_prepended_to_media_tag(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n")
        result = json.loads(send_file_tool(str(f), message="Here's the data"))
        assert result["media_tag"].startswith("Here's the data\nMEDIA:")

    def test_missing_file_reports_not_found(self, tmp_path):
        result = json.loads(send_file_tool(str(tmp_path / "nope.pdf")))
        assert result.get("success") is not True
        assert "not found" in result["error"].lower()

    def test_denied_local_path_reports_not_allowed(self, tmp_path, monkeypatch):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key = ssh_dir / "id_rsa"
        key.write_text("PRIVATE")
        monkeypatch.setenv("HOME", str(tmp_path))
        result = json.loads(send_file_tool(str(key)))
        assert result.get("success") is not True
        assert "not allowed" in result["error"]

    def test_empty_path_rejected(self):
        result = json.loads(send_file_tool("  "))
        assert "error" in result


class TestSendFileRemote:
    def test_remote_fetch_success(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        staged = tmp_path / "doc_abc_report.pdf"
        staged.write_bytes(b"%PDF")
        monkeypatch.setattr(
            "gateway.media_fetch.fetch_remote_media",
            lambda path, task_id=None: (str(staged), None),
        )
        result = json.loads(send_file_tool("/home/worker/report.pdf", task_id="t1"))
        assert result["success"] is True
        assert result["media_tag"] == f"MEDIA:{staged}"

    def test_remote_fetch_failure_surfaces_reason(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setattr(
            "gateway.media_fetch.fetch_remote_media",
            lambda path, task_id=None: (None, "the file is 60.0 MB, above the 50.0 MB delivery limit"),
        )
        result = json.loads(send_file_tool("/root/huge.zip"))
        assert result.get("success") is not True
        assert "modal backend" in result["error"]
        assert "50.0 MB" in result["error"]


class TestRegistration:
    def test_send_file_registered_in_file_toolset(self):
        import tools.send_file_tool  # noqa: F401 — self-registers on import
        from tools.registry import registry
        from toolsets import resolve_toolset

        entry = registry.get_entry("send_file")
        assert entry is not None
        assert entry.toolset == "file"
        assert "send_file" in resolve_toolset("file")
