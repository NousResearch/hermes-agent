"""
Hermes Blockchain Oracle - Solana MCP Toolset for Hermes Agent

A comprehensive Solana blockchain intelligence toolset providing 7 tools:
  - solana_wallet_info: Query wallet balances and token holdings
  - solana_transaction: Look up transaction details by signature
  - solana_token_info: Research SPL token metadata and distribution
  - solana_recent_activity: Fetch recent transaction history
  - solana_nft_portfolio: Browse NFTs owned by a wallet
  - whale_detector: Detect large SOL transfers in real-time
  - solana_network_stats: Monitor Solana network health

Author: Deniz Alagoz (@gizdusum)
Repository: https://github.com/gizdusum/hermes-blockchain-oracle
License: MIT
"""

from .solana_client import SolanaClient

__version__ = "0.1.0"
__author__ = "Deniz Alagoz"
__all__ = ["SolanaClient"]
