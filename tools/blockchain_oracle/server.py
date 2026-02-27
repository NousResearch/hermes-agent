"""
Hermes Blockchain Oracle - MCP Server
A Solana blockchain intelligence tool for Hermes Agent.

Exposes 7 tools via Model Context Protocol:
  - solana_wallet_info
  - solana_transaction
  - solana_token_info
  - solana_recent_activity
  - solana_nft_portfolio
  - whale_detector
  - solana_network_stats

Usage:
  hermes-agent --mcp blockchain="hermes-blockchain-oracle"
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server import Server
from mcp.server.stdio import run_server
from mcp.types import (
    Tool,
    TextContent,
)

from .solana_client import SolanaClient

# ── Server Setup ─────────────────────────────────────────────

server = Server("hermes-blockchain-oracle")
solana: SolanaClient | None = None


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Manage the Solana client lifecycle."""
    global solana
    solana = SolanaClient()
    try:
        yield
    finally:
        if solana:
            await solana.close()


# ── Tool Definitions ─────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all available blockchain tools."""
    return [
        Tool(
            name="solana_wallet_info",
            description=(
                "Query a Solana wallet address to get its SOL balance, "
                "token holdings (SPL tokens), account type, and owner program. "
                "Use this to inspect any Solana wallet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "Solana wallet address (base58 public key)",
                    }
                },
                "required": ["address"],
            },
        ),
        Tool(
            name="solana_transaction",
            description=(
                "Look up a Solana transaction by its signature. Returns detailed info "
                "including instructions, balance changes, token transfers, fees, and status. "
                "Great for investigating specific on-chain activity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signature": {
                        "type": "string",
                        "description": "Transaction signature (base58 encoded)",
                    }
                },
                "required": ["signature"],
            },
        ),
        Tool(
            name="solana_token_info",
            description=(
                "Get information about a Solana SPL token by its mint address. "
                "Returns total supply, decimals, top holders, and mint/freeze authorities. "
                "Useful for researching tokens and their distribution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mint_address": {
                        "type": "string",
                        "description": "Token mint address (base58 public key)",
                    }
                },
                "required": ["mint_address"],
            },
        ),
        Tool(
            name="solana_recent_activity",
            description=(
                "Get recent transaction history for a Solana wallet. "
                "Returns the latest transactions with their signatures, timestamps, "
                "and success status. Use to monitor wallet activity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "Solana wallet address (base58 public key)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent transactions to fetch (max 25, default 10)",
                        "default": 10,
                    },
                },
                "required": ["address"],
            },
        ),
        Tool(
            name="solana_nft_portfolio",
            description=(
                "List NFTs owned by a Solana wallet. Identifies NFTs by finding "
                "tokens with amount=1 and decimals=0. Returns mint addresses of NFTs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "Solana wallet address (base58 public key)",
                    }
                },
                "required": ["address"],
            },
        ),
        Tool(
            name="whale_detector",
            description=(
                "Scan recent Solana blocks for large SOL transfers (whale movements). "
                "Detects transfers above a configurable threshold. "
                "Default threshold is 1000 SOL. Use to spot big money moving on-chain."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_sol": {
                        "type": "number",
                        "description": "Minimum SOL amount to flag as whale activity (default: 1000)",
                        "default": 1000,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="solana_network_stats",
            description=(
                "Get current Solana network health and statistics. "
                "Returns current slot, epoch progress, TPS, supply info, "
                "and node version. Use for network health checks."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ── Tool Execution ───────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a blockchain tool and return results."""
    global solana

    if solana is None:
        solana = SolanaClient()

    try:
        if name == "solana_wallet_info":
            address = arguments.get("address", "")
            if not address:
                return [TextContent(type="text", text="Error: 'address' is required")]
            result = await solana.get_wallet_info(address)

        elif name == "solana_transaction":
            signature = arguments.get("signature", "")
            if not signature:
                return [TextContent(type="text", text="Error: 'signature' is required")]
            result = await solana.get_transaction(signature)

        elif name == "solana_token_info":
            mint_address = arguments.get("mint_address", "")
            if not mint_address:
                return [TextContent(type="text", text="Error: 'mint_address' is required")]
            result = await solana.get_token_info(mint_address)

        elif name == "solana_recent_activity":
            address = arguments.get("address", "")
            if not address:
                return [TextContent(type="text", text="Error: 'address' is required")]
            limit = arguments.get("limit", 10)
            result = await solana.get_recent_activity(address, limit)

        elif name == "solana_nft_portfolio":
            address = arguments.get("address", "")
            if not address:
                return [TextContent(type="text", text="Error: 'address' is required")]
            result = await solana.get_nft_portfolio(address)

        elif name == "whale_detector":
            min_sol = arguments.get("min_sol", 1000)
            result = await solana.detect_whales(min_sol)

        elif name == "solana_network_stats":
            result = await solana.get_network_stats()

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Format the result nicely
        formatted = json.dumps(result, indent=2, ensure_ascii=False)
        return [TextContent(type="text", text=formatted)]

    except Exception as e:
        error_msg = f"Blockchain Oracle Error: {type(e).__name__}: {str(e)}"
        return [TextContent(type="text", text=error_msg)]


# ── Entry Point ──────────────────────────────────────────────

def main():
    """Start the MCP server."""
    async def _run():
        global solana
        solana = SolanaClient()
        try:
            await run_server(server)
        finally:
            if solana:
                await solana.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
