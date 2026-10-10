"""Tests for PWA manifest prefix-awareness and OAuth-gate public paths.

The dashboard PWA manifest (manifest.webmanifest) contains start_url, scope,
and icon src paths that must stay under the X-Forwarded-Prefix when served
behind a reverse proxy.  The OAuth auth gate must also allow unauthenticated
access to the manifest and icon paths.

These test the CONTRACTS:
  1. Without a prefix, manifest paths remain root-absolute (``/``).
  2. With X-Forwarded-Prefix, start_url/scope/icons are prefixed.
  3. The auth gate public-prefix list includes manifest + icons.
"""

import json


# ---------------------------------------------------------------------------
# OAuth-gate public prefix coverage
# ---------------------------------------------------------------------------

def test_manifest_is_public_in_auth_gate():
    from hermes_cli.dashboard_auth.middleware import _GATE_PUBLIC_PREFIXES
    assert any(
        "/manifest.webmanifest".startswith(p) for p in _GATE_PUBLIC_PREFIXES
    ), "manifest.webmanifest must be in the OAuth-gate public prefix list"


def test_icons_are_public_in_auth_gate():
    from hermes_cli.dashboard_auth.middleware import _GATE_PUBLIC_PREFIXES
    assert any(
        "/icons/icon-192.png".startswith(p) for p in _GATE_PUBLIC_PREFIXES
    ), "/icons/ must be in the OAuth-gate public prefix list"


# ---------------------------------------------------------------------------
# Prefix-aware manifest rewriting (unit test of the rewrite logic)
# ---------------------------------------------------------------------------

_SAMPLE_MANIFEST = {
    "name": "Hermes Agent Dashboard",
    "short_name": "Hermes",
    "id": "/",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "icons": [
        {"src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
    ],
}


def _rewrite_manifest(data: dict, prefix: str) -> dict:
    """Replicate the rewrite logic from web_server.py serve_manifest."""
    import copy
    result = copy.deepcopy(data)
    if not prefix:
        return result
    for key in ("id", "start_url", "scope"):
        val = result.get(key)
        if isinstance(val, str) and val.startswith("/"):
            result[key] = prefix + val
    for icon in result.get("icons") or []:
        src = icon.get("src", "")
        if isinstance(src, str) and src.startswith("/"):
            icon["src"] = prefix + src
    return result


def test_no_prefix_leaves_paths_unchanged():
    result = _rewrite_manifest(_SAMPLE_MANIFEST, "")
    assert result["start_url"] == "/"
    assert result["scope"] == "/"
    assert result["icons"][0]["src"] == "/icons/icon-192.png"


def test_prefix_rewrites_start_url():
    result = _rewrite_manifest(_SAMPLE_MANIFEST, "/hermes")
    assert result["start_url"] == "/hermes/"


def test_prefix_rewrites_scope():
    result = _rewrite_manifest(_SAMPLE_MANIFEST, "/hermes")
    assert result["scope"] == "/hermes/"


def test_prefix_rewrites_icon_src():
    result = _rewrite_manifest(_SAMPLE_MANIFEST, "/hermes")
    for icon in result["icons"]:
        assert icon["src"].startswith("/hermes/icons/"), \
            f"icon src must be prefixed: {icon['src']}"


def test_prefix_rewrites_id():
    result = _rewrite_manifest(_SAMPLE_MANIFEST, "/hermes")
    assert result["id"] == "/hermes/"


def test_deep_prefix():
    result = _rewrite_manifest(_SAMPLE_MANIFEST, "/mission-control/hermes")
    assert result["start_url"] == "/mission-control/hermes/"
    assert result["scope"] == "/mission-control/hermes/"
    assert result["icons"][0]["src"] == "/mission-control/hermes/icons/icon-192.png"
