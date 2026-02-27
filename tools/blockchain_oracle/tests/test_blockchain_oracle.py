"""
Tests for the Hermes Blockchain Oracle toolset.

Tests cover:
  - SolanaClient instantiation and configuration
  - Address validation (valid/invalid base58)
  - Network stats retrieval (live RPC call)
  - Wallet info for known addresses
  - Token info for known mints
  - Tool manifest (tool.json) integrity
"""

import asyncio
import json
import os
import pytest

from tools.blockchain_oracle.solana_client import SolanaClient


# ── Helpers ───────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# Known test addresses (public, stable)
TOLY_WALLET = "86xCnPeV69n6t3DnyGvkKobf9FdN2H9oiVDdRrbukKM"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


# ── Client Tests ──────────────────────────────────────────────

class TestSolanaClientInit:
    """Test SolanaClient initialization and config."""

    def test_default_rpc_url(self):
        client = SolanaClient()
        assert "solana.com" in client.rpc_url or "SOLANA_RPC_URL" in os.environ

    def test_custom_rpc_url(self, monkeypatch):
        monkeypatch.setenv("SOLANA_RPC_URL", "https://custom-rpc.example.com")
        client = SolanaClient()
        assert client.rpc_url == "https://custom-rpc.example.com"


# ── Address Validation Tests ──────────────────────────────────

class TestAddressValidation:
    """Test base58 address validation."""

    def test_invalid_address_returns_error(self):
        client = SolanaClient()
        result = run_async(client.get_wallet_info("not-a-valid-address"))
        assert "error" in result

    def test_short_address_returns_error(self):
        client = SolanaClient()
        result = run_async(client.get_wallet_info("abc123"))
        assert "error" in result

    def test_empty_address_returns_error(self):
        client = SolanaClient()
        result = run_async(client.get_wallet_info(""))
        assert "error" in result


# ── Live RPC Tests (require network) ─────────────────────────

@pytest.mark.skipif(
    os.environ.get("SKIP_LIVE_TESTS", "0") == "1",
    reason="Live RPC tests skipped (SKIP_LIVE_TESTS=1)"
)
class TestLiveRPC:
    """Tests that make actual Solana RPC calls (require network access)."""

    def test_network_stats(self):
        client = SolanaClient()
        result = run_async(client.get_network_stats())
        assert "current_slot" in result
        assert "epoch" in result
        assert "average_tps" in result
        assert "health" in result
        assert result["current_slot"] > 0

    def test_wallet_info_toly(self):
        client = SolanaClient()
        result = run_async(client.get_wallet_info(TOLY_WALLET))
        assert "address" in result
        assert result["address"] == TOLY_WALLET
        assert "sol_balance" in result
        assert isinstance(result["sol_balance"], (int, float))

    def test_token_info_bonk(self):
        client = SolanaClient()
        result = run_async(client.get_token_info(BONK_MINT))
        assert "mint_address" in result
        assert result["known_symbol"] == "BONK"
        assert "total_supply" in result
        assert "decimals" in result

    def test_token_info_usdc(self):
        client = SolanaClient()
        result = run_async(client.get_token_info(USDC_MINT))
        assert result["known_symbol"] == "USDC"
        assert result["decimals"] == 6

    def test_recent_activity(self):
        client = SolanaClient()
        result = run_async(client.get_recent_activity(TOLY_WALLET, limit=3))
        assert "transactions" in result
        assert result["transaction_count"] <= 3


# ── Tool Manifest Tests ───────────────────────────────────────

class TestToolManifest:
    """Verify tool.json integrity and completeness."""

    def setup_method(self):
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tool.json"
        )
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def test_manifest_has_required_fields(self):
        assert "name" in self.manifest
        assert "description" in self.manifest
        assert "version" in self.manifest
        assert "tools" in self.manifest

    def test_manifest_has_7_tools(self):
        assert len(self.manifest["tools"]) == 7

    def test_all_tools_have_schema(self):
        for tool in self.manifest["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"
            assert "properties" in tool["parameters"]
            assert "required" in tool["parameters"]

    def test_tool_names_match_expected(self):
        expected = {
            "solana_wallet_info",
            "solana_transaction",
            "solana_token_info",
            "solana_recent_activity",
            "solana_nft_portfolio",
            "whale_detector",
            "solana_network_stats",
        }
        actual = {t["name"] for t in self.manifest["tools"]}
        assert actual == expected
