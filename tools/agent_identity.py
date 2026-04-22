"""A2A cryptographic identity — Ed25519 signing for agent-to-agent messages.

Operationalizes **USPTO 63/992,041** (Self-Governing Payload Architecture,
Patent 7). Implements the spec in
``docs/strategic/A2A_CRYPTO_IDENTITY.md`` (filed as toryx-private#820).

This module is **transport-agnostic**. Platform adapters (telegram, email,
discord, consult_colleague) call ``sign_envelope`` before emitting a
message and ``verify_envelope`` on receipt, then drop silently on any
verification failure (per spec §5).

Public surface
--------------
- ``generate_identity(profile_name, force=False) -> bytes``
- ``sign_envelope(profile_name, recipient, body, host=None) -> dict``
- ``verify_envelope(envelope) -> VerificationResult``
- ``get_pubkey(profile_name) -> bytes | None``
- ``canonical_json(obj) -> bytes`` (utility; same encoding used by
  toryx-openratings anchor payloads)

Not implemented yet (v2)
------------------------
- Key rotation with 30-day pubkey overlap
- HSM-backed private-key storage
- Transport bindings (those live in each platform adapter)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
REGISTRY_PATH = HERMES_HOME / "identity_registry.yaml"
REPLAY_WINDOW_SECONDS = 300  # 5 min, per spec §4
NONCE_LRU_SIZE = 10_000  # per spec §5


# ---------------------------------------------------------------------------
# Canonical JSON — sorted keys, UTF-8, no whitespace. Same shape as the
# openratings anchor payload canonical form.
# ---------------------------------------------------------------------------
def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Key paths + persistence
# ---------------------------------------------------------------------------
def _profile_key_path(profile_name: str) -> Path:
    return HERMES_HOME / "profiles" / profile_name / "keys" / "ed25519_private.pem"


def generate_identity(profile_name: str, force: bool = False) -> bytes:
    """Generate an Ed25519 keypair for *profile_name*.

    Writes the private key to
    ``~/.hermes/profiles/<profile>/keys/ed25519_private.pem`` (chmod 600)
    and registers the raw public key in ``~/.hermes/identity_registry.yaml``.

    Returns the raw 32-byte public key.

    Raises ``FileExistsError`` if a key already exists and ``force`` is
    False. Rotation intentionally requires explicit opt-in — old keys
    should stay valid during the overlap window (v2 feature).
    """
    path = _profile_key_path(profile_name)
    if path.exists() and not force:
        raise FileExistsError(
            f"Identity already exists at {path}; pass force=True to rotate."
        )
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        Encoding.PEM,
        PrivateFormat.PKCS8,
        NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pem)
    path.chmod(0o600)
    pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    _registry_put(profile_name, pub_bytes)
    logger.info("agent_identity: generated keypair for %s", profile_name)
    return pub_bytes


def load_private_key(profile_name: str) -> Ed25519PrivateKey:
    path = _profile_key_path(profile_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No identity for {profile_name!r} at {path}. "
            f"Run generate_identity({profile_name!r}) first."
        )
    priv = load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise TypeError(f"{path} is not an Ed25519 private key")
    return priv


# ---------------------------------------------------------------------------
# Registry — file-based yaml for v1. Could graduate to a Qdrant collection
# for cross-host lookup later; the API stays the same.
# ---------------------------------------------------------------------------
def _registry_load() -> dict[str, str]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
        if not isinstance(data, dict):
            logger.warning(
                "agent_identity: registry at %s is not a dict, ignoring",
                REGISTRY_PATH,
            )
            return {}
        return data
    except Exception as exc:  # pragma: no cover
        logger.warning("agent_identity: registry load failed: %s", exc)
        return {}


def _registry_put(profile_name: str, pubkey: bytes) -> None:
    import yaml

    reg = _registry_load()
    reg[profile_name] = base64.b64encode(pubkey).decode("ascii")
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(yaml.safe_dump(reg, sort_keys=True))


def get_pubkey(profile_name: str) -> bytes | None:
    """Return the raw 32-byte pubkey for *profile_name* or None."""
    reg = _registry_load()
    b64 = reg.get(profile_name)
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
def _default_host() -> str:
    try:
        return os.uname().nodename
    except AttributeError:  # pragma: no cover — non-POSIX
        import socket

        return socket.gethostname()


def sign_envelope(
    profile_name: str,
    recipient: str,
    body: Mapping[str, Any],
    host: str | None = None,
) -> dict[str, Any]:
    """Build + sign an envelope around *body* addressed to *recipient*.

    The resulting envelope has the shape specified in A2A_CRYPTO_IDENTITY.md §4:
    ``{sender_agent, sender_host, recipient_agent, nonce, timestamp,
        body_sha256, body, signature}``

    Signature covers everything except the ``signature`` field itself,
    serialized as canonical JSON.
    """
    priv = load_private_key(profile_name)

    body_bytes = canonical_json(dict(body))
    envelope: dict[str, Any] = {
        "sender_agent": profile_name,
        "sender_host": host or _default_host(),
        "recipient_agent": recipient,
        "nonce": secrets.token_hex(16),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "body": dict(body),
    }
    signed_bytes = canonical_json(envelope)
    signature = priv.sign(signed_bytes)
    envelope["signature"] = base64.b64encode(signature).decode("ascii")
    return envelope


# ---------------------------------------------------------------------------
# Verification + replay protection
# ---------------------------------------------------------------------------
@dataclass
class VerificationResult:
    valid: bool
    reason: str = ""
    sender: str = ""


class _NonceLRU:
    """Small LRU set for replay protection. Process-local; persists nothing."""

    def __init__(self, size: int = NONCE_LRU_SIZE) -> None:
        self._size = size
        self._seen: dict[str, float] = {}

    def check_and_add(self, nonce: str) -> bool:
        """Return True if *nonce* is fresh (added); False if already seen."""
        if nonce in self._seen:
            return False
        if len(self._seen) >= self._size:
            oldest = min(self._seen.items(), key=lambda kv: kv[1])[0]
            del self._seen[oldest]
        self._seen[nonce] = time.monotonic()
        return True

    def reset(self) -> None:  # for tests
        self._seen.clear()


_nonce_lru = _NonceLRU()


def _reset_nonce_lru() -> None:
    """Test-only helper. Do not call from production code."""
    _nonce_lru.reset()


def verify_envelope(envelope: Mapping[str, Any]) -> VerificationResult:
    """Verify an incoming envelope.

    Per spec §5: check timestamp is in the replay window, check the nonce
    hasn't been seen, look up the sender's pubkey, verify the signature.
    On any failure, return a non-valid result with a ``reason`` — callers
    must drop the message silently and never reply.
    """
    required = {
        "sender_agent",
        "sender_host",
        "recipient_agent",
        "nonce",
        "timestamp",
        "body_sha256",
        "body",
        "signature",
    }
    missing = required - set(envelope)
    if missing:
        return VerificationResult(False, f"missing fields: {sorted(missing)!r}")

    sender = str(envelope["sender_agent"])

    # Timestamp window check
    ts_raw = str(envelope["timestamp"])
    try:
        ts = datetime.fromisoformat(ts_raw)
    except ValueError:
        return VerificationResult(False, "bad timestamp format", sender)
    if ts.tzinfo is None:
        return VerificationResult(False, "timestamp missing timezone", sender)
    now = datetime.now(timezone.utc)
    if abs((now - ts).total_seconds()) > REPLAY_WINDOW_SECONDS:
        return VerificationResult(False, "timestamp out of window", sender)

    # Body hash sanity — catches envelope tamper that's signature-stripped
    body_bytes = canonical_json(dict(envelope["body"]))
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    if body_sha != envelope["body_sha256"]:
        return VerificationResult(False, "body_sha256 mismatch", sender)

    # Nonce replay check — do this after cheaper checks so we don't pollute
    # the LRU with malformed inputs
    nonce = str(envelope["nonce"])
    if not _nonce_lru.check_and_add(nonce):
        return VerificationResult(False, "nonce replay", sender)

    # Sender pubkey lookup
    pubkey_bytes = get_pubkey(sender)
    if pubkey_bytes is None:
        return VerificationResult(False, "unknown sender", sender)

    # Signature verify — canonical JSON of everything except the signature
    signed = {k: v for k, v in envelope.items() if k != "signature"}
    signed_bytes = canonical_json(signed)
    try:
        pub = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        signature = base64.b64decode(str(envelope["signature"]))
        pub.verify(signature, signed_bytes)
    except Exception:
        return VerificationResult(False, "bad signature", sender)

    return VerificationResult(True, "ok", sender)
