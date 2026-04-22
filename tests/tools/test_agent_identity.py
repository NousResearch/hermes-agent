"""Unit tests for tools.agent_identity — see toryx-private#718 / doc PR #820."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import agent_identity as ai


@pytest.fixture(autouse=True)
def tmp_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect HERMES_HOME to a tmp dir for every test + reset the
    module-level nonce LRU so tests don't cross-pollute."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Re-import module-level constants pinned to env:
    monkeypatch.setattr(ai, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(ai, "REGISTRY_PATH", tmp_path / "identity_registry.yaml")
    ai._reset_nonce_lru()
    yield
    ai._reset_nonce_lru()


def test_generate_identity_creates_keyfile_and_registry_entry():
    pub = ai.generate_identity("ray")
    assert isinstance(pub, bytes) and len(pub) == 32
    keyfile = ai.HERMES_HOME / "profiles" / "ray" / "keys" / "ed25519_private.pem"
    assert keyfile.exists()
    # 0o600 perms
    assert (keyfile.stat().st_mode & 0o777) == 0o600
    # Registry entry roundtrips
    got = ai.get_pubkey("ray")
    assert got == pub


def test_generate_identity_refuses_overwrite_without_force():
    ai.generate_identity("ray")
    with pytest.raises(FileExistsError):
        ai.generate_identity("ray")


def test_generate_identity_force_rotates():
    pub1 = ai.generate_identity("ray")
    pub2 = ai.generate_identity("ray", force=True)
    assert pub1 != pub2
    # Registry reflects the new key
    assert ai.get_pubkey("ray") == pub2


def test_sign_and_verify_roundtrip():
    ai.generate_identity("ray")
    env = ai.sign_envelope(
        profile_name="ray",
        recipient="luis",
        body={"kind": "research_brief", "topic": "MoE routing"},
    )
    result = ai.verify_envelope(env)
    assert result.valid, f"expected valid, got reason={result.reason!r}"
    assert result.sender == "ray"


def test_verify_rejects_timestamp_out_of_window():
    ai.generate_identity("ray")
    env = ai.sign_envelope("ray", "luis", {"x": 1})
    # Backdate by 10 minutes
    env["timestamp"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    # Re-sign so the body check doesn't fail earlier — but we're specifically
    # testing the timestamp window, so actually skip re-signing: the signature
    # will fail first if we don't, which also proves the test sort of. Instead
    # we tamper AFTER sign, which means signature verification WILL fail because
    # timestamp is part of the signed bytes — but timestamp check runs BEFORE
    # signature check, so we see the timestamp-out-of-window reason first.
    result = ai.verify_envelope(env)
    assert not result.valid
    assert result.reason == "timestamp out of window"


def test_verify_rejects_replay():
    ai.generate_identity("ray")
    env = ai.sign_envelope("ray", "luis", {"x": 1})
    first = ai.verify_envelope(env)
    assert first.valid
    # Second time: same nonce, should fail
    second = ai.verify_envelope(env)
    assert not second.valid
    assert second.reason == "nonce replay"


def test_verify_rejects_unknown_sender():
    # Sign with ray, but then strip ray from the registry so verification
    # finds an unknown sender.
    ai.generate_identity("ray")
    env = ai.sign_envelope("ray", "luis", {"x": 1})
    # Wipe the registry
    ai.REGISTRY_PATH.unlink()
    ai._reset_nonce_lru()
    result = ai.verify_envelope(env)
    assert not result.valid
    assert result.reason == "unknown sender"


def test_verify_rejects_bad_signature():
    ai.generate_identity("ray")
    env = ai.sign_envelope("ray", "luis", {"x": 1})
    # Corrupt the signature
    env["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    ai._reset_nonce_lru()
    result = ai.verify_envelope(env)
    assert not result.valid
    assert result.reason == "bad signature"


def test_verify_rejects_tampered_body():
    ai.generate_identity("ray")
    env = ai.sign_envelope("ray", "luis", {"x": 1})
    # Tamper with body — body_sha256 check catches this before signature
    env["body"] = {"x": 2}
    ai._reset_nonce_lru()
    result = ai.verify_envelope(env)
    assert not result.valid
    # Could be "body_sha256 mismatch" OR "bad signature" depending on order;
    # either one closes the attack
    assert result.reason in {"body_sha256 mismatch", "bad signature"}


def test_verify_rejects_missing_fields():
    result = ai.verify_envelope({"sender_agent": "ray"})
    assert not result.valid
    assert "missing fields" in result.reason


def test_canonical_json_deterministic():
    a = {"b": 2, "a": 1, "c": {"z": 9, "y": 8}}
    b = {"a": 1, "c": {"y": 8, "z": 9}, "b": 2}
    assert ai.canonical_json(a) == ai.canonical_json(b)
    # ASCII form, sorted
    assert ai.canonical_json(a) == b'{"a":1,"b":2,"c":{"y":8,"z":9}}'


def test_two_agents_can_talk():
    ai.generate_identity("ray")
    ai.generate_identity("christopher")
    env = ai.sign_envelope("ray", "christopher", {"note": "check NVDA"})
    r = ai.verify_envelope(env)
    assert r.valid and r.sender == "ray"


def test_nonce_lru_eviction_works():
    lru = ai._NonceLRU(size=3)
    assert lru.check_and_add("a") is True
    assert lru.check_and_add("b") is True
    assert lru.check_and_add("c") is True
    assert lru.check_and_add("a") is False  # replay
    # Eviction: adding "d" should evict the oldest ("a")
    assert lru.check_and_add("d") is True
    # "a" was evicted so now it should be accepted as fresh
    assert lru.check_and_add("a") is True
