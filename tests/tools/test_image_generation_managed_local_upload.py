"""Tests for the managed FAL local-image-reference upload adapter.

Covers `_upload_local_fal_sources`, `_contains_managed_upload`,
`_is_local_image_source`, and the per-request managed-routing override in
`_submit_fal_request` / `image_generate_tool`. Does not exercise a real
network call or a real Nous/FAL credential — all gateway/upload/FAL-client
behavior is mocked.
"""

from __future__ import annotations

import importlib
import json
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.fixture
def image_tool():
    """Fresh import of tools.image_generation_tool per test."""
    import tools.image_generation_tool as mod
    return importlib.reload(mod)


class _Handler:
    """Minimal stand-in for the object `_submit_fal_request` returns."""


def _generation_patches(image_tool, meta):
    """Common patches so `image_generate_tool` reaches `_submit_fal_request`
    without touching real model resolution, waiting, or debug logging."""
    return (
        patch.object(image_tool, "_resolve_fal_model", return_value=("fal/test", meta)),
        patch.object(image_tool, "_wait_fal_result", return_value={
            "images": [{"url": "https://fal.example/out.png"}],
        }),
        patch.object(image_tool, "_debug", Mock(log_call=lambda *a, **k: None, save=lambda: None)),
    )


def test_is_local_image_source_classification(image_tool):
    assert image_tool._is_local_image_source("/opt/data/x.png") is True
    assert image_tool._is_local_image_source("x.png") is True
    assert image_tool._is_local_image_source("file:///opt/data/x.png") is True
    assert image_tool._is_local_image_source("https://example.com/x.png") is False
    assert image_tool._is_local_image_source("data:image/png;base64,AA==") is False
    assert image_tool._is_local_image_source("nous-upload:abc123") is False
    assert image_tool._is_local_image_source(None) is False
    assert image_tool._is_local_image_source("") is False


def test_contains_managed_upload_detects_nested_tokens(image_tool):
    assert image_tool._contains_managed_upload("nous-upload:abc") is True
    assert image_tool._contains_managed_upload("https://x.com/y.png") is False
    assert image_tool._contains_managed_upload({"image_urls": ["nous-upload:abc"]}) is True
    assert image_tool._contains_managed_upload(["a", "https://x.com"]) is False


def test_upload_local_fal_sources_passthrough_for_remote_and_none(image_tool):
    sources = ["https://example.test/a.png", "data:image/png;base64,AA==", None]
    with patch.object(image_tool, "resolve_managed_tool_gateway") as resolve_gateway:
        result = image_tool._upload_local_fal_sources(sources)
    assert result == sources
    resolve_gateway.assert_not_called()


def test_upload_local_fal_sources_uploads_via_managed_gateway(image_tool, tmp_path):
    png = tmp_path / "reference.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    fake_uploader = AsyncMock(return_value="nous-upload:opaque-token")
    resolved = Mock(data=b"\x89PNG\r\n\x1a\nfake", mime="image/png")

    with patch.object(image_tool, "resolve_managed_tool_gateway", return_value=Mock()), \
         patch.object(image_tool, "managed_vendor_endpoints", return_value={
             "base_url": "https://tool-gateway.example/api/fal",
             "upload_path": "/api/uploads/fal",
         }), \
         patch.object(image_tool, "build_managed_media_uploader", return_value=fake_uploader), \
         patch("tools.image_source.resolve_image_source", new=AsyncMock(return_value=resolved)):
        result = image_tool._upload_local_fal_sources([str(png)])

    assert result == ["nous-upload:opaque-token"]
    fake_uploader.assert_awaited_once_with(b"\x89PNG\r\n\x1a\nfake", "image/png")


def test_upload_local_fal_sources_direct_fal_only_preserves_passthrough(image_tool, tmp_path):
    png = tmp_path / "reference.png"
    png.write_bytes(b"fake")

    with patch.object(image_tool, "resolve_managed_tool_gateway", return_value=None), \
         patch.object(image_tool, "fal_key_is_configured", return_value=True):
        result = image_tool._upload_local_fal_sources([str(png)])

    assert result == [str(png)]


def test_upload_local_fal_sources_neither_backend_raises_explicit_error(image_tool, tmp_path):
    png = tmp_path / "reference.png"
    png.write_bytes(b"fake")

    with patch.object(image_tool, "resolve_managed_tool_gateway", return_value=None), \
         patch.object(image_tool, "fal_key_is_configured", return_value=False):
        with pytest.raises(ValueError, match="neither is available"):
            image_tool._upload_local_fal_sources([str(png)])


def test_submit_fal_request_forces_managed_route_for_nous_upload_token(image_tool, caplog):
    managed_client = Mock()
    managed_client.submit.return_value = _Handler()
    direct_client = Mock()
    direct_client.submit.side_effect = AssertionError("direct route was reached")

    with caplog.at_level(logging.INFO), \
         patch.object(image_tool, "_load_fal_client"), \
         patch.object(image_tool, "resolve_managed_tool_gateway", return_value=Mock()), \
         patch.object(image_tool, "_get_managed_fal_client", return_value=managed_client), \
         patch.object(image_tool, "fal_client", direct_client):
        image_tool._submit_fal_request("fal/test/edit", {"image_urls": ["nous-upload:opaque"]})

    managed_client.submit.assert_called_once()
    direct_client.submit.assert_not_called()
    sent = managed_client.submit.call_args.kwargs["arguments"]
    assert sent["image_urls"] == ["nous-upload:opaque"]
    assert "Forcing managed FAL gateway routing" in caplog.text


def test_submit_fal_request_without_nous_upload_uses_normal_selection(image_tool):
    """Requests without a managed-upload token must not be affected at all —
    the pre-existing selection-based routing (_resolve_managed_fal_gateway)
    decides, exactly as before this adapter existed."""
    direct_client = Mock()
    direct_client.submit.return_value = _Handler()

    with patch.object(image_tool, "_load_fal_client"), \
         patch.object(image_tool, "_resolve_managed_fal_gateway", return_value=None) as resolve_direct, \
         patch.object(image_tool, "resolve_managed_tool_gateway") as resolve_managed, \
         patch.object(image_tool, "fal_client", direct_client):
        image_tool._submit_fal_request("fal/test/edit", {"image_url": "https://x.com/y.png"})

    resolve_direct.assert_called_once()
    resolve_managed.assert_not_called()
    direct_client.submit.assert_called_once()


def test_grounding_route_is_visible_in_success_result(image_tool):
    """The 'never silent' contract: a forced managed reroute must surface in
    the actual JSON result, not just a log line."""
    model_id = "fal-ai/flux-2/klein/9b"
    meta = image_tool.FAL_MODELS[model_id]
    managed_client = Mock()
    managed_client.submit.return_value = _Handler()

    patches = _generation_patches(image_tool, meta)
    with patches[0], patches[1], patches[2], \
         patch.object(image_tool, "_resolve_fal_model", return_value=(model_id, meta)), \
         patch.object(image_tool, "resolve_managed_tool_gateway", return_value=Mock()), \
         patch.object(image_tool, "managed_vendor_endpoints", return_value={
             "base_url": "https://tool-gateway.example/api/fal", "upload_path": "/api/uploads/fal",
         }), \
         patch.object(image_tool, "build_managed_media_uploader",
                       return_value=AsyncMock(return_value="nous-upload:opaque")), \
         patch("tools.image_source.resolve_image_source",
               new=AsyncMock(return_value=Mock(data=b"fake", mime="image/png"))), \
         patch.object(image_tool, "_load_fal_client"), \
         patch.object(image_tool, "_get_managed_fal_client", return_value=managed_client):
        payload = json.loads(image_tool.image_generate_tool(
            "edit it", image_url="/opt/data/reference.png",
        ))

    assert payload["success"] is True
    assert payload.get("grounding_route") == "managed_gateway_forced_by_local_reference"


def test_stale_nous_upload_token_error_has_context_and_setup_guidance(image_tool, tmp_path, monkeypatch):
    """PR #94307 review Note 1: when arguments already contain a nous-upload:
    token (i.e. a local source was already uploaded) but managed access is
    now unavailable, the error must (a) stay explicit about the stale/managed
    -token situation, (b) reuse the normal actionable setup/remediation
    guidance from `_build_no_backend_setup_message()`, and (c) the direct FAL
    route must never receive the managed token."""
    model_id = "fal-ai/flux-2/klein/9b"
    meta = image_tool.FAL_MODELS[model_id]
    direct_client = Mock()
    direct_client.submit.side_effect = AssertionError("direct route was reached")

    monkeypatch.setenv("TERMINAL_ENV", "local")
    patches = _generation_patches(image_tool, meta)
    with patches[0], patches[1], patches[2], \
         patch.object(image_tool, "_resolve_fal_model", return_value=(model_id, meta)), \
         patch.object(image_tool, "resolve_managed_tool_gateway", return_value=None), \
         patch.object(image_tool, "fal_client", direct_client):
        # Simulate arguments that already carry a stale nous-upload: token
        # (e.g. a previously-uploaded reference) by feeding it straight in
        # as the image_url — `_upload_local_fal_sources` is bypassed because
        # a nous-upload: string is not a local path, so it flows straight
        # into the `_contains_managed_upload` gate unchanged.
        payload = json.loads(image_tool.image_generate_tool(
            "edit it", image_url="nous-upload:stale-token",
        ))

    assert payload["success"] is False
    error = payload["error"]
    # Explicit about the stale/managed-token condition.
    assert "nous-upload:" in error
    assert "managed FAL" in error
    assert "unavailable" in error
    assert "stale" in error.lower() or "lost managed access" in error.lower()
    # Reuses the normal actionable setup/remediation guidance verbatim
    # (via `_build_no_backend_setup_message()`), not a duplicated blurb.
    setup_message = image_tool._build_no_backend_setup_message()
    assert setup_message in error
    # The direct FAL route must never be reached / never receive the token.
    direct_client.submit.assert_not_called()
