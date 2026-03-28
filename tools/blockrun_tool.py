"""
BlockRun / ClawRouter tool for hermes-agent.

Tools provided:
  Wallet     — setup, balance, address (Base + Solana)
  Images     — generate (DALL-E 3, GPT Image 1, Flux, Nano Banana) + edit
  Prediction — Polymarket, Kalshi, dFlow, Binance candles, cross-platform markets

Supports both chains:
  Base   — BLOCKRUN_WALLET_KEY=0x...        (default)
  Solana — SOLANA_WALLET_KEY=<base58-key>

LLM calls go through the blockrun provider (hermes_cli/blockrun_provider.py),
which attaches an x402-aware httpx transport to the OpenAI client so that
402 Payment Required responses are handled transparently — no agent involvement.
"""

from __future__ import annotations

import os
from typing import Any

from tools.registry import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Return a blockrun_llm.LLMClient, raising a clear error if unavailable."""
    try:
        from blockrun_llm import LLMClient
    except ImportError:
        raise RuntimeError(
            "blockrun_llm is not installed. Run: pip install blockrun-llm"
        )

    key = os.environ.get("BLOCKRUN_WALLET_KEY")
    if not key:
        raise RuntimeError(
            "BLOCKRUN_WALLET_KEY is not set. "
            "Run the `blockrun_wallet_setup` tool first."
        )
    return LLMClient(private_key=key)


def _check_requirements() -> tuple[bool, str]:
    try:
        import blockrun_llm  # noqa: F401
        return True, ""
    except ImportError:
        return False, "blockrun_llm not installed (pip install blockrun-llm)"


# ---------------------------------------------------------------------------
# Tool: blockrun_wallet_setup
# ---------------------------------------------------------------------------

def _handle_wallet_setup(args: dict[str, Any], **_) -> str:
    """
    Create or load a BlockRun wallet.
    Stores the private key at ~/.blockrun/.session (mode 600).
    Prints the wallet address and a funding QR/link.
    """
    try:
        from blockrun_llm.wallet import (
            get_or_create_wallet,
            generate_wallet_qr_ascii,
            get_payment_links,
            save_wallet,
        )
    except ImportError:
        return "❌ blockrun_llm not installed. Run: pip install blockrun-llm"

    # Honour an explicit key override from the environment
    explicit_key = os.environ.get("BLOCKRUN_WALLET_KEY")

    address, private_key, is_new = get_or_create_wallet()

    # Persist and export so the rest of the session can use it
    if is_new:
        wallet_path = save_wallet(private_key)
        os.environ["BLOCKRUN_WALLET_KEY"] = private_key
        status_line = f"✅ New wallet created and saved to {wallet_path}"
    elif explicit_key:
        status_line = "✅ Wallet loaded from BLOCKRUN_WALLET_KEY"
    else:
        status_line = "✅ Existing wallet loaded from ~/.blockrun/.session"

    # Balance
    try:
        from blockrun_llm import LLMClient
        balance = LLMClient(private_key=private_key).get_balance()
        balance_line = f"💰 Balance: ${balance:.4f} USDC on Base"
    except Exception:
        balance_line = "💰 Balance: (unable to fetch — check network)"

    # Funding links
    links = get_payment_links(address)
    qr = generate_wallet_qr_ascii(address)

    lines = [
        status_line,
        f"📬 Address: {address}",
        balance_line,
        "",
        "To fund this wallet with USDC on Base:",
        f"  • BlockRun: {links['blockrun']}",
        f"  • BaseScan: {links['basescan']}",
        "",
        "QR Code (scan with any Ethereum wallet):",
        qr,
        "",
        "Tip: $5–$20 USDC is enough for hundreds of requests.",
        "Testnet faucet: https://faucet.circle.com (select Base Sepolia)",
    ]
    return "\n".join(lines)


WALLET_SETUP_SCHEMA = {
    "name": "blockrun_wallet_setup",
    "description": (
        "Set up a BlockRun / ClawRouter wallet for autonomous x402 payments. "
        "Creates a new wallet if none exists, or loads an existing one. "
        "Shows the wallet address, current USDC balance, and funding instructions. "
        "Must be called before using BlockRun as the LLM provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_wallet_balance
# ---------------------------------------------------------------------------

def _handle_wallet_balance(args: dict[str, Any], **_) -> str:
    """Check current USDC balance and session spending."""
    try:
        client = _get_client()
    except RuntimeError as e:
        return f"❌ {e}"

    try:
        address = client.get_wallet_address()
        balance = client.get_balance()
        spending = client.get_spending()
        lines = [
            f"📬 Address : {address}",
            f"💰 Balance : ${balance:.4f} USDC (Base mainnet)",
            f"📊 Session : ${spending['total_usd']:.4f} spent across {spending['calls']} call(s)",
        ]
        if balance < 0.01:
            lines.append(
                "\n⚠️  Low balance. Fund at: "
                f"https://blockrun.ai/fund?address={address}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Failed to fetch balance: {exc}"


WALLET_BALANCE_SCHEMA = {
    "name": "blockrun_wallet_balance",
    "description": (
        "Check the current USDC balance of the BlockRun wallet on Base mainnet, "
        "plus total spending for the current session."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_wallet_address
# ---------------------------------------------------------------------------

def _handle_wallet_address(args: dict[str, Any], **_) -> str:
    """Return wallet address and funding links without exposing the private key."""
    try:
        from blockrun_llm.wallet import get_wallet_address, get_payment_links
    except ImportError:
        return "❌ blockrun_llm not installed. Run: pip install blockrun-llm"

    address = get_wallet_address()
    if not address:
        return "❌ No wallet configured. Run `blockrun_wallet_setup` first."

    links = get_payment_links(address)
    return "\n".join([
        f"📬 Address : {address}",
        f"🔗 BlockRun: {links['blockrun']}",
        f"🔗 BaseScan: {links['basescan']}",
        f"🔗 EIP-681 : {links['ethereum']}",
    ])


WALLET_ADDRESS_SCHEMA = {
    "name": "blockrun_wallet_address",
    "description": (
        "Return the BlockRun wallet address and funding links. "
        "Safe to call at any time — never exposes the private key."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_solana_wallet_setup
# ---------------------------------------------------------------------------

def _handle_solana_wallet_setup(args: dict[str, Any], **_) -> str:
    """Create or load a Solana wallet for BlockRun x402 payments."""
    try:
        from blockrun_llm.solana_wallet import (
            setup_agent_solana_wallet,
            get_solana_usdc_balance,
        )
    except ImportError:
        return (
            "❌ blockrun_llm Solana support not available. "
            "Run: pip install blockrun-llm solders base58"
        )

    try:
        client = setup_agent_solana_wallet()
        address = client.get_wallet_address()
        os.environ["SOLANA_WALLET_KEY"] = os.environ.get("SOLANA_WALLET_KEY", "")

        try:
            balance = get_solana_usdc_balance(address)
            balance_line = f"💰 Balance : ${balance:.4f} USDC on Solana"
        except Exception:
            balance_line = "💰 Balance : (unable to fetch — check network)"

        return "\n".join([
            "✅ Solana wallet ready",
            f"📬 Address : {address}",
            balance_line,
            "",
            "To fund with USDC on Solana:",
            f"  • https://blockrun.ai/fund?address={address}&chain=solana",
            "  • Send SPL-USDC to the address above from any Solana wallet",
            "",
            "Tip: $5–$20 USDC covers hundreds of requests.",
            "Set BLOCKRUN_CHAIN=solana in your .env to use this wallet by default.",
        ])
    except Exception as exc:
        return f"❌ Solana wallet setup failed: {exc}"


SOLANA_WALLET_SETUP_SCHEMA = {
    "name": "blockrun_solana_wallet_setup",
    "description": (
        "Set up a Solana wallet for BlockRun x402 payments (SPL-USDC). "
        "Creates a new wallet if none exists, or loads an existing one from "
        "SOLANA_WALLET_KEY. Use this if you prefer to pay on Solana instead of Base."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_solana_wallet_balance
# ---------------------------------------------------------------------------

def _handle_solana_wallet_balance(args: dict[str, Any], **_) -> str:
    """Check Solana USDC balance."""
    try:
        from blockrun_llm.solana_wallet import (
            load_solana_wallet,
            get_solana_usdc_balance,
        )
    except ImportError:
        return "❌ blockrun_llm Solana support not available. Run: pip install blockrun-llm solders base58"

    key = os.environ.get("SOLANA_WALLET_KEY") or os.environ.get("BLOCKRUN_SOLANA_KEY")
    if not key:
        return "❌ SOLANA_WALLET_KEY not set. Run `blockrun_solana_wallet_setup` first."

    try:
        from blockrun_llm import SolanaLLMClient
        client = SolanaLLMClient(private_key=key)
        address = client.get_wallet_address()
        balance = get_solana_usdc_balance(address)
        spending = client.get_spending()
        return "\n".join([
            f"📬 Address : {address}",
            f"💰 Balance : ${balance:.4f} USDC (Solana)",
            f"📊 Session : ${spending['total_usd']:.4f} spent across {spending['calls']} call(s)",
        ])
    except Exception as exc:
        return f"❌ Failed to fetch Solana balance: {exc}"


SOLANA_WALLET_BALANCE_SCHEMA = {
    "name": "blockrun_solana_wallet_balance",
    "description": "Check the current SPL-USDC balance of the BlockRun Solana wallet.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_image_generate
# ---------------------------------------------------------------------------

_IMAGE_MODELS = {
    "dall-e-3":          "openai/dall-e-3",
    "gpt-image-1":       "openai/gpt-image-1",
    "flux":              "black-forest/flux-1.1-pro",
    "nano-banana":       "google/nano-banana",
    "nano-banana-pro":   "google/nano-banana-pro",
}

_IMAGE_PRICES = {
    "openai/dall-e-3":              "$0.04",
    "openai/gpt-image-1":           "$0.02",
    "black-forest/flux-1.1-pro":    "$0.04",
    "google/nano-banana":           "$0.05",
    "google/nano-banana-pro":       "$0.10",
}


def _handle_image_generate(args: dict[str, Any], **_) -> str:
    prompt    = args.get("prompt", "").strip()
    model_key = args.get("model", "nano-banana").lower()
    size      = args.get("size", "1024x1024")
    n         = int(args.get("n", 1))

    if not prompt:
        return "❌ 'prompt' is required."

    model_id = _IMAGE_MODELS.get(model_key, model_key)

    try:
        from blockrun_llm.image import ImageClient
    except ImportError:
        return "❌ blockrun_llm not installed. Run: pip install blockrun-llm"

    key = os.environ.get("BLOCKRUN_WALLET_KEY") or os.environ.get("BASE_CHAIN_WALLET_KEY")
    if not key:
        return "❌ BLOCKRUN_WALLET_KEY not set. Run `blockrun_wallet_setup` first."

    try:
        client = ImageClient(private_key=key)
        result = client.generate(prompt=prompt, model=model_id, size=size, n=n)
        lines = [
            f"✅ Generated {len(result.data)} image(s) with {model_id} ({size})",
            f"💰 Cost: {_IMAGE_PRICES.get(model_id, 'see blockrun.ai/api/pricing')} per image",
            "",
        ]
        for i, img in enumerate(result.data, 1):
            lines.append(f"Image {i}: {img.url}")
            if getattr(img, "revised_prompt", None):
                lines.append(f"  Revised prompt: {img.revised_prompt}")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Image generation failed: {exc}"


IMAGE_GENERATE_SCHEMA = {
    "name": "blockrun_image_generate",
    "description": (
        "Generate images using BlockRun's image API (x402 micropayment, ~$0.02–$0.10/image). "
        "Models: dall-e-3, gpt-image-1, flux, nano-banana (default), nano-banana-pro. "
        "Returns image URLs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description / prompt",
            },
            "model": {
                "type": "string",
                "enum": list(_IMAGE_MODELS.keys()),
                "description": "Model to use. Default: nano-banana (Google Gemini Flash, fast + cheap)",
            },
            "size": {
                "type": "string",
                "description": "Image size. Default: 1024x1024. nano-banana-pro supports up to 4096x4096.",
            },
            "n": {
                "type": "integer",
                "description": "Number of images (default: 1)",
            },
        },
        "required": ["prompt"],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_image_edit
# ---------------------------------------------------------------------------

def _handle_image_edit(args: dict[str, Any], **_) -> str:
    prompt     = args.get("prompt", "").strip()
    image_b64  = args.get("image", "").strip()
    mask_b64   = args.get("mask", "").strip() or None
    size       = args.get("size", "1024x1024")

    if not prompt:
        return "❌ 'prompt' is required."
    if not image_b64:
        return "❌ 'image' is required (base64 data URI, e.g. data:image/png;base64,...)."

    try:
        from blockrun_llm.image import ImageClient
    except ImportError:
        return "❌ blockrun_llm not installed. Run: pip install blockrun-llm"

    key = os.environ.get("BLOCKRUN_WALLET_KEY") or os.environ.get("BASE_CHAIN_WALLET_KEY")
    if not key:
        return "❌ BLOCKRUN_WALLET_KEY not set. Run `blockrun_wallet_setup` first."

    try:
        client = ImageClient(private_key=key)
        result = client.edit(
            prompt=prompt,
            image=image_b64,
            mask=mask_b64,
            size=size,
        )
        lines = [f"✅ Edited image with gpt-image-1 ({size})", ""]
        for i, img in enumerate(result.data, 1):
            lines.append(f"Image {i}: {img.url}")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Image edit failed: {exc}"


IMAGE_EDIT_SCHEMA = {
    "name": "blockrun_image_edit",
    "description": (
        "Edit or inpaint an existing image using GPT Image 1 (x402, ~$0.02/image). "
        "Provide the original image as a base64 data URI. "
        "Optionally provide a mask to restrict edits to specific areas."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Edit instruction, e.g. 'Add a sunset background'",
            },
            "image": {
                "type": "string",
                "description": "Original image as base64 data URI (data:image/png;base64,...)",
            },
            "mask": {
                "type": "string",
                "description": "Optional mask as base64 data URI — white areas will be edited",
            },
            "size": {
                "type": "string",
                "description": "Output size. Default: 1024x1024",
            },
        },
        "required": ["prompt", "image"],
    },
}


# ---------------------------------------------------------------------------
# Tool: blockrun_prediction_markets
# ---------------------------------------------------------------------------

_PM_PLATFORMS = ["polymarket", "kalshi", "dflow", "limitless", "opinion", "predict.fun", "binance"]

_PM_ROUTE_HINTS = {
    "markets":        "/{platform}/markets           — list open markets",
    "trades":         "/{platform}/trades            — recent trades",
    "orderbook":      "/{platform}/orderbooks        — live order book",
    "leaderboard":    "/polymarket/leaderboard       — top traders",
    "wallet":         "/polymarket/wallet/{address}  — wallet positions + analytics",
    "smart-money":    "/polymarket/smart-money       — smart money flows",
    "candles":        "/binance/candles/{symbol}     — OHLCV candlesticks",
    "matching":       "/matching-markets             — same event across platforms",
}


def _handle_prediction_markets(args: dict[str, Any], **_) -> str:
    path   = args.get("path", "").strip().lstrip("/")
    method = args.get("method", "GET").upper()
    params = args.get("params") or {}

    if not path:
        # Show available routes
        lines = [
            "Available prediction market endpoints (base: blockrun.ai/api/v1/pm/):",
            "",
            "Platforms: " + ", ".join(_PM_PLATFORMS),
            "",
        ]
        for hint in _PM_ROUTE_HINTS.values():
            lines.append(f"  {hint}")
        lines += [
            "",
            "Pricing: GET = $0.001/call  |  POST (analytics) = $0.005/call",
            "",
            'Example: {"path": "polymarket/markets", "method": "GET"}',
        ]
        return "\n".join(lines)

    key = os.environ.get("BLOCKRUN_WALLET_KEY") or os.environ.get("BASE_CHAIN_WALLET_KEY")
    if not key:
        return "❌ BLOCKRUN_WALLET_KEY not set. Run `blockrun_wallet_setup` first."

    try:
        import httpx
        from hermes_cli.blockrun_provider import BlockRunX402Transport, _load_signer

        signer    = _load_signer(key, "base")
        transport = BlockRunX402Transport(private_key=key, chain="base")

        url = f"https://blockrun.ai/api/v1/pm/{path}"

        with httpx.Client(transport=transport, timeout=30.0) as client:
            if method == "GET":
                response = client.get(url, params=params)
            else:
                response = client.post(url, json=params)

        if response.status_code == 200:
            import json
            data = response.json()
            receipt = response.headers.get("X-Payment-Receipt", "")
            result_str = json.dumps(data, indent=2)
            # Truncate very large responses
            if len(result_str) > 4000:
                result_str = result_str[:4000] + "\n... (truncated)"
            lines = [f"✅ {method} /pm/{path}"]
            if receipt:
                lines.append(f"💳 Tx: {receipt}")
            lines += ["", result_str]
            return "\n".join(lines)
        else:
            return f"❌ {response.status_code}: {response.text[:500]}"

    except Exception as exc:
        return f"❌ Prediction market request failed: {exc}"


PREDICTION_MARKETS_SCHEMA = {
    "name": "blockrun_prediction_markets",
    "description": (
        "Query prediction market data via BlockRun (x402, $0.001–$0.005/call). "
        "Supports Polymarket, Kalshi, dFlow, Limitless, Binance, and more. "
        "Call with no arguments to see all available endpoints. "
        "Examples: polymarket/markets, polymarket/leaderboard, kalshi/trades, "
        "polymarket/wallet/{address}, binance/candles/BTC-USD, matching-markets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "API path after /api/v1/pm/. "
                    "E.g. 'polymarket/markets', 'kalshi/trades', 'binance/candles/BTC-USD'. "
                    "Leave empty to list all available endpoints."
                ),
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST"],
                "description": "HTTP method. Default: GET. Use POST for advanced analytics.",
            },
            "params": {
                "type": "object",
                "description": "Query params (GET) or request body (POST).",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="blockrun_wallet_setup",
    toolset="blockrun",
    schema=WALLET_SETUP_SCHEMA,
    handler=_handle_wallet_setup,
    check_fn=_check_requirements,
    emoji="🔑",
)

registry.register(
    name="blockrun_wallet_balance",
    toolset="blockrun",
    schema=WALLET_BALANCE_SCHEMA,
    handler=_handle_wallet_balance,
    check_fn=_check_requirements,
    emoji="💰",
)

registry.register(
    name="blockrun_wallet_address",
    toolset="blockrun",
    schema=WALLET_ADDRESS_SCHEMA,
    handler=_handle_wallet_address,
    check_fn=_check_requirements,
    emoji="📬",
)

registry.register(
    name="blockrun_solana_wallet_setup",
    toolset="blockrun",
    schema=SOLANA_WALLET_SETUP_SCHEMA,
    handler=_handle_solana_wallet_setup,
    check_fn=_check_requirements,
    emoji="◎",
)

registry.register(
    name="blockrun_solana_wallet_balance",
    toolset="blockrun",
    schema=SOLANA_WALLET_BALANCE_SCHEMA,
    handler=_handle_solana_wallet_balance,
    check_fn=_check_requirements,
    emoji="◎",
)

registry.register(
    name="blockrun_image_generate",
    toolset="blockrun",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=_check_requirements,
    emoji="🎨",
)

registry.register(
    name="blockrun_image_edit",
    toolset="blockrun",
    schema=IMAGE_EDIT_SCHEMA,
    handler=_handle_image_edit,
    check_fn=_check_requirements,
    emoji="✏️",
)

registry.register(
    name="blockrun_prediction_markets",
    toolset="blockrun",
    schema=PREDICTION_MARKETS_SCHEMA,
    handler=_handle_prediction_markets,
    check_fn=_check_requirements,
    emoji="📊",
)
