"""
Lattice Ed25519 authentication helpers.

Shared by the platform adapter (gateway/platforms/lattice.py) and tool layer (lattice_tool.py).
"""

import time


def get_auth_headers(privkey_hex: str) -> dict:
    """Build Lattice auth headers for GET requests. Signs ';{timestamp}'."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    privkey_bytes = bytes.fromhex(privkey_hex)
    if len(privkey_bytes) != 32:
        raise ValueError("LATTICE_PRIVATE_KEY_HEX must be 64 hex chars (32 bytes)")

    private_key = Ed25519PrivateKey.from_private_bytes(privkey_bytes)
    pubkey_hex = private_key.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw
    ).hex()

    timestamp = int(time.time())
    signature = private_key.sign(f";{timestamp}".encode("utf-8"))

    return {
        "X-Agent-Pubkey": pubkey_hex,
        "X-Timestamp": str(timestamp),
        "X-Signature": signature.hex(),
    }


def get_post_auth_headers(privkey_hex: str, body_str: str) -> dict:
    """Build Lattice auth headers for POST requests. Signs '{body_str};{timestamp}'.

    body_str must match the exact JSON bytes sent — the Lattice server verifies
    against JSON.stringify(parsed_body), so use separators=(',', ':'), ensure_ascii=False.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    privkey_bytes = bytes.fromhex(privkey_hex)
    if len(privkey_bytes) != 32:
        raise ValueError("LATTICE_PRIVATE_KEY_HEX must be 64 hex chars (32 bytes)")

    private_key = Ed25519PrivateKey.from_private_bytes(privkey_bytes)
    pubkey_hex = private_key.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw
    ).hex()

    timestamp = int(time.time())
    signature = private_key.sign(f"{body_str};{timestamp}".encode("utf-8"))

    return {
        "X-Agent-Pubkey": pubkey_hex,
        "X-Timestamp": str(timestamp),
        "X-Signature": signature.hex(),
    }
