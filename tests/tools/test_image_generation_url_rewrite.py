"""Tests for the image_generate → HTTP URL post-processor.

Covers :func:`tools.image_generation_tool._maybe_rewrite_image_to_url`
and :func:`tools.image_generation_tool._resolve_image_serve_base_url`.
The rewrite is what lets Open WebUI (and any HTTP client) actually
fetch generated images — without it the assistant's markdown points
at an absolute filesystem path the browser can't reach.
"""

from __future__ import annotations

import json

import pytest

from tools import image_generation_tool


def _make_success(path: str = "/some/cache/openai_codex_medium_abcd1234.png") -> str:
    return json.dumps({
        "success": True,
        "image": path,
        "model": "gpt-image-2-medium",
        "prompt": "a cat",
        "aspect_ratio": "square",
        "provider": "openai-codex",
    })


class TestResolveBaseUrl:
    """``_resolve_image_serve_base_url`` reads env var then config.yaml."""

    def test_env_var_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://env.example.com")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  serve_base_url: https://config.example.com\n"
        )
        assert image_generation_tool._resolve_image_serve_base_url() == "https://env.example.com"

    def test_config_yaml_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("IMAGE_SERVE_BASE_URL", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  serve_base_url: https://config.example.com\n"
        )
        assert image_generation_tool._resolve_image_serve_base_url() == "https://config.example.com"

    def test_returns_none_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("IMAGE_SERVE_BASE_URL", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: openai-codex\n")
        assert image_generation_tool._resolve_image_serve_base_url() is None

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://example.com/")
        assert image_generation_tool._resolve_image_serve_base_url() == "https://example.com"

    def test_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "  https://example.com  ")
        assert image_generation_tool._resolve_image_serve_base_url() == "https://example.com"

    def test_empty_env_falls_through_to_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  serve_base_url: https://config.example.com\n"
        )
        assert image_generation_tool._resolve_image_serve_base_url() == "https://config.example.com"


class TestMaybeRewriteImageToUrl:
    """``_maybe_rewrite_image_to_url`` turns local paths into public URLs."""

    def test_rewrites_local_path_when_base_set(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("/var/hermes/cache/images/openai_codex_abc123.png")
        result = json.loads(image_generation_tool._maybe_rewrite_image_to_url(before))
        assert result["image"] == "https://chat.example.com/images/openai_codex_abc123.png"
        # Everything else preserved
        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"

    def test_no_op_when_base_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("IMAGE_SERVE_BASE_URL", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml present
        before = _make_success("/var/hermes/cache/images/openai_codex_abc123.png")
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_no_op_when_already_http_url(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("https://cdn.fal.ai/generated/xyz.png")
        # Already a URL — FAL-style providers return URLs directly; don't double-prefix.
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_no_op_when_already_https_url(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("https://cdn.fal.ai/generated/xyz.png")
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_no_op_when_data_uri(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("data:image/png;base64,iVBORw0KGgoA...")
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_no_op_on_failure_result(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = json.dumps({
            "success": False,
            "image": None,
            "error": "provider failed",
            "error_type": "api_error",
        })
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_no_op_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        # A non-JSON tool_error payload must not crash the post-processor.
        assert image_generation_tool._maybe_rewrite_image_to_url("not json {{{") == "not json {{{"

    def test_no_op_when_image_field_missing(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = json.dumps({"success": True, "model": "x"})
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_no_op_when_image_is_empty_string(self, monkeypatch):
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("")
        assert image_generation_tool._maybe_rewrite_image_to_url(before) == before

    def test_filename_only_is_preserved(self, monkeypatch):
        """Even a bare filename (edge case) should rewrite cleanly."""
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("just_a_filename.png")
        result = json.loads(image_generation_tool._maybe_rewrite_image_to_url(before))
        assert result["image"] == "https://chat.example.com/images/just_a_filename.png"

    def test_nested_path_uses_basename_only(self, monkeypatch):
        """Only the filename is carried into the URL; the cache dir isn't leaked."""
        monkeypatch.setenv("IMAGE_SERVE_BASE_URL", "https://chat.example.com")
        before = _make_success("/some/deeply/nested/cache/dir/image_xyz.png")
        result = json.loads(image_generation_tool._maybe_rewrite_image_to_url(before))
        assert result["image"] == "https://chat.example.com/images/image_xyz.png"

    def test_config_yaml_drives_rewrite_when_no_env(self, monkeypatch, tmp_path):
        """config.yaml alone is sufficient to activate the rewrite."""
        monkeypatch.delenv("IMAGE_SERVE_BASE_URL", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  serve_base_url: https://via-config.example.com\n"
        )
        before = _make_success("/cache/images/pic.png")
        result = json.loads(image_generation_tool._maybe_rewrite_image_to_url(before))
        assert result["image"] == "https://via-config.example.com/images/pic.png"
