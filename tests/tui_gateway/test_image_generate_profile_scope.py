"""Profile ownership for the UI ``image.generate`` JSON-RPC path.

The real dispatcher, provider registry, secret scope, and image cache writer run here.  Only the
provider transport is fake: it turns the active profile's test credential into a tiny local PNG.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import pytest

import tui_gateway.server as server
from agent.image_gen_provider import ImageGenProvider, save_b64_image, success_response
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home


_PROVIDER = "r41-profile-image"
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


class _Transport:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.ready = threading.Event()

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        self.ready.set()
        return True

    def close(self) -> None:
        return None


class _ProfileImageProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return _PROVIDER

    def __init__(self, expected_account: str, calls: list[dict]) -> None:
        self.expected_account = expected_account
        self.calls = calls

    def is_available(self) -> bool:
        return get_secret("R41_IMAGE_ACCOUNT") == self.expected_account

    def generate(self, prompt: str, aspect_ratio: str = "square", **_kwargs) -> dict:
        account = get_secret("R41_IMAGE_ACCOUNT")
        image = save_b64_image(_PNG_B64, prefix=f"r41-{account}")
        self.calls.append({
            "account": account,
            "home": Path(get_hermes_home()),
            "image": image,
        })
        return success_response(
            image=str(image),
            model="fake-image",
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            provider=self.name,
        )


def _rpc(transport: _Transport, rid: str, params: dict) -> dict:
    before = len(transport.frames)
    transport.ready.clear()
    assert (
        server.dispatch(
            {"jsonrpc": "2.0", "id": rid, "method": "image.generate", "params": params},
            transport,
        )
        is None
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for frame in transport.frames[before:]:
            if frame.get("id") == rid:
                return frame
        transport.ready.wait(0.05)
        transport.ready.clear()
    raise AssertionError(f"timed out waiting for image.generate response {rid}")


@pytest.fixture
def image_profiles(tmp_path, monkeypatch):
    from agent import image_gen_registry
    from agent import secret_scope
    from tui_gateway import launch_profile_policy

    launch = tmp_path / ".hermes"
    secondary = launch / "profiles" / "b"
    calls: list[dict] = []
    transports = {"A": _Transport(), "B": _Transport()}

    for home, account in ((launch, "account-A"), (secondary, "account-B")):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            f"image_gen:\n  provider: {_PROVIDER}\n", encoding="utf-8"
        )
        (home / ".env").write_text(f"R41_IMAGE_ACCOUNT={account}\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("R41_IMAGE_ACCOUNT", "account-A")
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)

    providers = {
        launch: _ProfileImageProvider("account-A", calls),
        secondary: _ProfileImageProvider("account-B", calls),
    }
    previous = {
        home: image_gen_registry.snapshot_registration(_PROVIDER, scope=str(home))
        for home in providers
    }
    for home, provider in providers.items():
        image_gen_registry.register_provider(provider, scope=str(home))

    session_ids = {"A": "image-profile-a", "B": "image-profile-b"}
    monkeypatch.setitem(
        server._sessions,
        session_ids["A"],
        {
            "session_key": session_ids["A"],
            "profile_home": None,
            "transport": transports["A"],
        },
    )
    monkeypatch.setitem(
        server._sessions,
        session_ids["B"],
        {
            "session_key": session_ids["B"],
            "profile_home": str(secondary),
            "transport": transports["B"],
        },
    )

    yield {
        "launch": launch,
        "secondary": secondary,
        "calls": calls,
        "transports": transports,
        "session_ids": session_ids,
    }

    for home, provider in providers.items():
        image_gen_registry.restore_registration(
            _PROVIDER, provider, previous[home], scope=str(home)
        )


def test_sessionless_image_rpc_uses_transport_profile_account_and_cache_a_b_a(
    image_profiles,
):
    """A sessionless UI RPC inherits its sole live transport owner's profile, including credentials
    and provider cache writes; switching B must not consume or overwrite launch profile A state."""
    p = image_profiles

    responses = [
        _rpc(
            p["transports"]["A"],
            "image-a1",
            {"prompt": "A first", "aspect_ratio": "square"},
        ),
        _rpc(
            p["transports"]["B"], "image-b", {"prompt": "B", "aspect_ratio": "square"}
        ),
        _rpc(
            p["transports"]["A"],
            "image-a2",
            {"prompt": "A again", "aspect_ratio": "square"},
        ),
    ]

    assert all("error" not in response for response in responses), responses
    assert [call["account"] for call in p["calls"]] == [
        "account-A",
        "account-B",
        "account-A",
    ]
    assert [call["home"] for call in p["calls"]] == [
        p["launch"],
        p["secondary"],
        p["launch"],
    ]
    for call in p["calls"]:
        assert call["image"].parent == call["home"] / "cache" / "generated" / "images"
        assert call["image"].read_bytes() == base64.b64decode(_PNG_B64)
    assert all(
        response["result"]["image_data"].startswith("data:image/png;base64,")
        for response in responses
    )


def test_image_rpc_rejects_profile_session_mismatch_and_accepts_same_owner(
    image_profiles,
):
    """An explicit profile cannot override a live session owner. A matching owner and a profile-only
    sessionless request both remain usable, and both resolve B rather than the launch account."""
    p = image_profiles
    sid_b = p["session_ids"]["B"]

    unowned = _rpc(_Transport(), "image-unowned", {"prompt": "must not use launch"})
    assert unowned["error"]["code"] == 4065
    assert unowned["error"]["data"]["code"] == "profile_context_required"
    assert p["calls"] == []

    mismatch = _rpc(
        p["transports"]["B"],
        "image-mismatch",
        {
            "prompt": "must not run",
            "profile": "default",
            "session_id": sid_b,
        },
    )
    assert mismatch["error"]["code"] == 4065
    assert mismatch["error"]["data"]["code"] == "profile_context_mismatch"
    assert p["calls"] == []

    same_owner = _rpc(
        p["transports"]["B"],
        "image-same-owner",
        {
            "prompt": "same owner",
            "profile": "b",
            "session_id": sid_b,
        },
    )
    profile_only = _rpc(
        p["transports"]["B"],
        "image-profile-only",
        {
            "prompt": "explicit sessionless owner",
            "profile": "b",
        },
    )

    assert "error" not in same_owner, same_owner
    assert "error" not in profile_only, profile_only
    assert [call["account"] for call in p["calls"]] == ["account-B", "account-B"]
    assert all(
        call["image"].parent == p["secondary"] / "cache" / "generated" / "images"
        for call in p["calls"]
    )
