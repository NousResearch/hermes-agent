#!/usr/bin/env python3
"""
Lattice Tool -- Agent-to-Agent Messaging

Allows the AI to send messages to other AI agents via the Lattice
agent-to-agent messaging endpoint (/send).

Requires:
- LATTICE_URL env var
- LATTICE_PRIVATE_KEY_HEX env var (64-char hex Ed25519 private key)
"""

import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


LATTICE_SEND_AGENT_SCHEMA = {
    "name": "lattice_send_agent",
    "description": (
        "Send a message to another AI agent via Lattice agent-to-agent messaging.\n\n"
        "The recipient is identified by their Ed25519 public key (hex). "
        "The recipient must be currently connected to the same Lattice server."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient agent's Ed25519 public key (64-char hex)",
            },
            "body": {
                "type": "string",
                "description": "Message to send",
            },
        },
        "required": ["to", "body"],
    },
}


def _get_post_auth_headers(privkey_hex: str, body_str: str) -> dict:
    """Build Lattice auth headers for POST requests.

    Signs '{body_str};{timestamp}'. The body_str must match the exact JSON bytes
    we send — the Lattice server verifies using JSON.stringify(parsed_body), so
    we use the same serialization (no extra spaces, same key order).
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    privkey_bytes = bytes.fromhex(privkey_hex)
    if len(privkey_bytes) != 32:
        raise ValueError("LATTICE_PRIVATE_KEY_HEX must be 64 hex chars (32 bytes)")

    private_key = Ed25519PrivateKey.from_private_bytes(privkey_bytes)
    pubkey_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    pubkey_hex = pubkey_bytes.hex()

    timestamp = int(time.time())
    payload = f"{body_str};{timestamp}".encode("utf-8")
    signature = private_key.sign(payload)
    sig_hex = signature.hex()

    return {
        "X-Agent-Pubkey": pubkey_hex,
        "X-Timestamp": str(timestamp),
        "X-Signature": sig_hex,
    }


async def lattice_send_agent_tool(args: Dict[str, Any], **kwargs) -> str:
    """Send a message to another agent via Lattice /send."""
    import httpx

    from hermes_cli.config import get_env_value

    to = args.get("to", "").strip()
    body_text = args.get("body", "").strip()

    if not to:
        return json.dumps({"error": "Missing required parameter: to"})
    if not body_text:
        return json.dumps({"error": "Missing required parameter: body"})

    lattice_url = (get_env_value("LATTICE_URL") or os.getenv("LATTICE_URL", "")).rstrip(
        "/"
    )
    privkey_hex = (
        get_env_value("LATTICE_PRIVATE_KEY_HEX")
        or os.getenv("LATTICE_PRIVATE_KEY_HEX", "")
    ).strip()

    if not lattice_url:
        return json.dumps({"error": "LATTICE_URL environment variable is not set"})
    if not privkey_hex:
        return json.dumps(
            {"error": "LATTICE_PRIVATE_KEY_HEX environment variable is not set"}
        )

    # Normalize key: strip non-hex (invisible chars, accidental spaces from paste)
    raw_key_len = len(privkey_hex)
    privkey_hex = "".join(
        c for c in privkey_hex if c in "0123456789abcdefABCDEF"
    ).lower()
    if len(privkey_hex) != 64:
        return json.dumps(
            {
                "error": (
                    f"LATTICE_PRIVATE_KEY_HEX must be exactly 64 hex chars (got {len(privkey_hex)}). "
                    f"Raw length was {raw_key_len}. Check ~/.hermes/.env for truncation."
                )
            }
        )

    # Use exact serialization for both signing and sending — Lattice server verifies
    # signature against JSON.stringify(parsed_body), so we must match that format.
    # ensure_ascii=False so Unicode in body matches typical JS output (not \uXXXX).
    body = {"to": to, "body": body_text}
    body_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    body_bytes = body_str.encode("utf-8")

    try:
        headers = {
            "Content-Type": "application/json",
            **_get_post_auth_headers(privkey_hex, body_str),
        }
    except Exception as e:
        return json.dumps({"error": f"Failed to build auth headers: {e}"})

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{lattice_url}/send", content=body_bytes, headers=headers
            )
            if resp.status_code == 404:
                return json.dumps({"error": "Agent not connected"})
            if resp.status_code == 401:
                pubkey_hex = headers["X-Agent-Pubkey"]
                logger.warning(
                    "Lattice 401: pubkey=%s...%s timestamp=%s body=%r",
                    pubkey_hex[:8],
                    pubkey_hex[-8:],
                    headers["X-Timestamp"],
                    body_str,
                )
            resp.raise_for_status()
            return json.dumps({"success": True})
    except httpx.HTTPStatusError as e:
        err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        if e.response.status_code == 401:
            pubkey_hex = headers.get("X-Agent-Pubkey", "")
            err += (
                f" Sender pubkey: {pubkey_hex[:16]}...{pubkey_hex[-8:]}. "
                "Check: same key in ~/.hermes/.env; restart gateway if changed; clock ±30s."
            )
        return json.dumps({"error": err})
    except Exception as e:
        return json.dumps({"error": str(e)})


def check_lattice_send_requirements() -> bool:
    """Return True if LATTICE_URL is configured."""
    try:
        from hermes_cli.config import get_env_value

        return bool(get_env_value("LATTICE_URL") or os.getenv("LATTICE_URL"))
    except ImportError:
        return bool(os.getenv("LATTICE_URL"))


# Alias for registry check_fn
_check_lattice_send = check_lattice_send_requirements


# --- Registry ---
from tools.registry import registry

registry.register(
    name="lattice_send_agent",
    toolset="lattice",
    schema=LATTICE_SEND_AGENT_SCHEMA,
    handler=lattice_send_agent_tool,
    check_fn=_check_lattice_send,
    is_async=True,
    emoji="🔗",
)
