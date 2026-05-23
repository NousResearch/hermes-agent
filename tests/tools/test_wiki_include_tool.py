from __future__ import annotations

from pathlib import Path

import pytest

from tools import wiki_include_tool as wiki


def _cfg(root: Path, path: Path, *, source: str = "atlas-wiki", subject: str = "atlas-overview"):
    return {
        "enabled": True,
        "allowed_roots": [str(root)],
        "allowlist": [
            {
                "id": "atlas-public-overview",
                "family": "wiki",
                "subject": subject,
                "path": str(path),
                "source": source,
                "privacy": "least_sensitive",
            }
        ],
    }


def test_allowed_wiki_read_succeeds_with_labels_and_safe_path(tmp_path):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "atlas-overview.md"
    page.write_text("Atlas public overview\n", encoding="utf-8")

    result = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(root, page),
    )

    assert result["success"] is True
    assert result["source"] == "atlas-wiki"
    assert result["privacy"] == "least_sensitive"
    assert result["freshness"].endswith("Z")
    assert result["degraded"] is False
    assert result["degraded_reason"] is None
    assert result["path"] == "wiki:atlas-overview.md"
    assert result["content"] == "Atlas public overview\n"
    assert str(tmp_path) not in result["path"]


def test_include_subject_mismatch_rejects_before_read(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "atlas-overview.md"

    def fail_read(*args, **kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError("read_text should not be called")

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(wiki.WikiIncludeError, match="include/subject mismatch"):
        wiki.read_wiki_include(
            "atlas-public-overview",
            "wrong-subject",
            config=_cfg(root, page),
        )


def test_protected_private_path_rejects_before_read(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    protected = root / "private" / "atlas-overview.md"

    def fail_read(*args, **kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError("read_text should not be called")

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(wiki.WikiIncludeError, match="protected/private path"):
        wiki.read_wiki_include(
            "atlas-public-overview",
            "atlas-overview",
            config=_cfg(root, protected),
        )


def test_output_redacts_obvious_sensitive_values_and_is_log_safe(tmp_path):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "public.md"
    page.write_text(
        "Contact owner@example.com\napi_key=sk-live-test\nAuthorization: Bearer abcdefghijk\n",
        encoding="utf-8",
    )

    result = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(root, page),
    )

    assert "owner@example.com" not in result["content"]
    assert "sk-live-test" not in result["content"]
    assert "abcdefghijk" not in result["content"]
    assert "[REDACTED_EMAIL]" in result["content"]
    assert "[REDACTED]" in result["content"]
    assert str(tmp_path) not in result["path"]


def test_missing_allowlisted_source_is_degraded_no_source(tmp_path):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "public.md"
    page.write_text("Public atlas note\n", encoding="utf-8")

    result = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(root, page, source=""),
    )

    assert result["success"] is True
    assert result["source"] == "wiki:no_source_label"
    assert result["degraded"] is True
    assert result["degraded_reason"] == "descriptor missing source label"


def test_is_file_failure_returns_safe_degraded_response(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "atlas-overview.md"
    page.write_text("Atlas public overview\n", encoding="utf-8")

    original_is_file = Path.is_file

    def broken_is_file(self):
        if self == page:
            raise PermissionError(f"permission denied: {page}")
        return original_is_file(self)

    monkeypatch.setattr(Path, "is_file", broken_is_file)

    result = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(root, page),
    )

    assert result["success"] is False
    assert result["source"] == "atlas-wiki"
    assert result["privacy"] == "least_sensitive"
    assert result["freshness"] == "unavailable"
    assert result["degraded"] is True
    assert result["degraded_reason"] == "allowlisted wiki source is unreadable"
    assert result["path"] == "wiki:atlas-overview.md"
    assert str(page) not in result["error"]
    assert str(tmp_path) not in result["error"]


def test_stat_failure_returns_safe_degraded_response(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "atlas-overview.md"
    page.write_text("Atlas public overview\n", encoding="utf-8")

    original_stat = Path.stat

    def broken_stat(self, *args, **kwargs):
        if self == page:
            raise OSError(f"stat failed for {page}")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", broken_stat)

    result = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(root, page),
    )

    assert result["success"] is False
    assert result["source"] == "atlas-wiki"
    assert result["privacy"] == "least_sensitive"
    assert result["freshness"] == "unavailable"
    assert result["degraded"] is True
    assert result["degraded_reason"] == "allowlisted wiki source is unreadable"
    assert result["path"] == "wiki:atlas-overview.md"
    assert str(page) not in result["error"]
    assert str(tmp_path) not in result["error"]


def test_read_text_failure_returns_safe_degraded_response(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    root.mkdir()
    page = root / "atlas-overview.md"
    page.write_text("Atlas public overview\n", encoding="utf-8")

    original_read_text = Path.read_text

    def broken_read_text(self, *args, **kwargs):
        if self == page:
            raise OSError(f"read failed for {page}")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken_read_text)

    result = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(root, page),
    )

    assert result["success"] is False
    assert result["source"] == "atlas-wiki"
    assert result["privacy"] == "least_sensitive"
    assert result["freshness"] == "unavailable"
    assert result["degraded"] is True
    assert result["degraded_reason"] == "allowlisted wiki source is unreadable"
    assert result["path"] == "wiki:atlas-overview.md"
    assert str(page) not in result["error"]
    assert str(tmp_path) not in result["error"]


def test_disabled_or_missing_source_returns_degraded_no_read(tmp_path, monkeypatch):
    page = tmp_path / "wiki" / "missing.md"

    def fail_read(*args, **kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError("read_text should not be called")

    monkeypatch.setattr(Path, "read_text", fail_read)

    disabled = wiki.read_wiki_include("anything", "anything", config={"enabled": False})
    assert disabled["success"] is False
    assert disabled["degraded"] is True
    assert disabled["degraded_reason"] == "memory_seam.wiki disabled"

    missing = wiki.read_wiki_include(
        "atlas-public-overview",
        "atlas-overview",
        config=_cfg(tmp_path / "wiki", page),
    )
    assert missing["success"] is False
    assert missing["degraded"] is True
    assert missing["degraded_reason"] == "allowlisted wiki source is missing"
    assert missing["path"] == "wiki:missing.md"
