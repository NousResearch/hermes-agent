"""Regression for #124852: ``profiles.get_asset`` builds
``profile_dir/assets/<asset>.<ext>`` straight from the caller-supplied asset
string. A value containing ``../`` escapes the profile's ``assets/`` directory,
while the sibling ``set_asset`` door rejects anything but ``"avatar"`` (4066).
Both doors of one asset store must accept the same asset names.
"""

from __future__ import annotations

import base64

import pytest

from tui_gateway import methods_profiles
from tui_gateway.server import _err  # noqa: F401  (documents the bind_module seam)


PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _get_asset(rid: int, params: dict) -> dict:
    """Call the registered ``profiles.get_asset`` handler with server.py's
    globals (``_ok``/``_err``) bound the way ``method_ctx.bind_module`` binds
    them in production."""
    import tui_gateway.server as server

    pending = {name: fn for name, fn in methods_profiles._registry._pending}
    from tui_gateway.method_ctx import rebind

    handler = rebind(pending["profiles.get_asset"], vars(server))
    return handler(rid, params)


@pytest.fixture()
def profile_with_escape_target(tmp_path, monkeypatch):
    """A profile whose ``assets/avatar.png`` exists, plus a decoy PNG OUTSIDE
    ``assets/`` that a traversal would reach (``<profile>/secret.png``)."""
    from hermes_cli.profiles import get_profile_dir

    profile_dir = tmp_path / "profiles" / "esc-profile"
    (profile_dir / "assets").mkdir(parents=True)
    (profile_dir / "assets" / "avatar.png").write_bytes(PNG_1PX)
    decoy = profile_dir / "secret.png"
    decoy.write_bytes(PNG_1PX)

    # The rebind seam (method_ctx) runs handlers against server.py's globals,
    # so the patch must land on server's published copy, not the module attr.
    monkeypatch.setattr(
        "tui_gateway.server._resolve_profile",
        lambda rid, params: (params.get("name"), profile_dir, None),
    )
    return profile_dir, decoy


def test_get_asset_rejects_traversal_out_of_assets_dir(profile_with_escape_target):
    """``asset: "../secret"`` must NOT resolve ``<profile>/secret.png`` — the
    handler answers 4066 (unknown asset) instead of ``found: true``."""
    _profile_dir, _decoy = profile_with_escape_target
    resp = _get_asset(1, {"name": "esc-profile", "asset": "../secret"})
    assert resp["error"]["code"] == 4066, f"traversal accepted: {resp}"
    assert "found" not in resp.get("result", {})


def test_get_asset_still_serves_avatar(profile_with_escape_target):
    """The allowlist must not break the only legitimate asset."""
    resp = _get_asset(2, {"name": "esc-profile", "asset": "avatar"})
    result = resp.get("result") or {}
    assert result.get("found") is True
    assert result.get("mime") == "image/png"
    assert result.get("size") == len(PNG_1PX)


def test_get_asset_rejects_other_names_like_set_asset(profile_with_escape_target):
    """Non-traversal unknown names get the same 4066 as ``set_asset``."""
    resp = _get_asset(3, {"name": "esc-profile", "asset": "banner"})
    assert resp["error"]["code"] == 4066
