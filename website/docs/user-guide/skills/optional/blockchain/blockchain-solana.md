---
title: "Solana"
sidebar_label: "Solana"
description: "Query Solana blockchain data with USD pricing — wallet balances, token portfolios with values, transaction details, NFTs, whale detection, and live network s..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Solana

Query Solana blockchain data with USD pricing — wallet balances, token portfolios with values, transaction details, NFTs, whale detection, and live network stats. Uses Solana RPC + CoinGecko. No API key required.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/blockchain/solana` |
| Path | `optional-skills/blockchain/solana` |
| Version | `0.2.0` |
| Author | Deniz Alagoz (gizdusum), enhanced by Hermes Agent |
| License | MIT |
| Tags | `Solana`, `Blockchain`, `Crypto`, `Web3`, `RPC`, `DeFi`, `NFT` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Solana Blockchain Skill

Query Solana on-chain data enriched with USD pricing via CoinGecko.
8 commands: wallet portfolio, token info, transactions, activity, NFTs,
whale detection, network stats, and price lookup.

No API key needed. Uses only Python standard library (urllib, json, argparse).

---

## When to Use

- User asks for a Solana wallet balance, token holdings, or portfolio value
- User wants to inspect a specific transaction by signature
- User wants SPL token metadata, price, supply, or top holders
- User wants recent transaction history for an address
- User wants NFTs owned by a wallet
- User wants to find large SOL transfers (whale detection)
- User wants Solana network health, TPS, epoch, or SOL price
- User asks "what's the price of BONK/JUP/SOL?"

---

## Prerequisites

The helper script uses only Python standard library (urllib, json, argparse).
No external packages required.

Pricing data comes from CoinGecko's free API (no key needed, rate-limited
to ~10-30 requests/minute). For faster lookups, use `--no-prices` flag.

---

## Quick Reference

RPC endpoint (default): https://api.mainnet-beta.solana.com
Override: export SOLANA_RPC_URL=https://your-private-rpc.com

Helper script path: ~/.hermes/skills/blockchain/solana/scripts/solana_client.py

```
python3 solana_client.py wallet   <address> [--limit N] [--all] [--no-prices]
python3 solana_client.py tx       <signature>
python3 solana_client.py token    <mint_address>
python3 solana_client.py activity <address> [--limit N]
python3 solana_client.py nft      <address>
python3 solana_client.py whales   [--min-sol N]
python3 solana_client.py stats
python3 solana_client.py price    <mint_or_symbol>
```

---

## Procedure

### 0. Setup Check

```bash
python3 --version

# Optional: set a private RPC for better rate limits
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 1. Wallet Portfolio

Get SOL balance, SPL token holdings with USD values, NFT count, and
portfolio total. Tokens sorted by value, dust filtered, known tokens
labeled by name (BONK, JUP, USDC, etc.).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

Flags:
- `--limit N` — show top N tokens (default: 20)
- `--all` — show all tokens, no dust filter, no limit
- `--no-prices` — skip CoinGecko price lookups (faster, RPC-only)

Output includes: SOL balance + USD value, token list with prices sorted
by value, dust count, NFT summary, total portfolio value in USD.

### 2. Transaction Details

Inspect a full transaction by its base58 signature. Shows balance changes
in both SOL and USD.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  tx 5j7s8K...your_signature_here
```

Output: slot, timestamp, fee, status, balance changes (SOL + USD),
program invocations.

### 3. Token Info

Get SPL token metadata, current price, market cap, supply, decimals,
mint/freeze authorities, and top 5 holders.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

Output: name, symbol, decimals, supply, price, market cap, top 5
holders with percentages.

### 4. Recent Activity

List recent transactions for an address (default: last 10, max: 25).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  activity 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM --limit 25
```

### 5. NFT Portfolio

List NFTs owned by a wallet (heuristic: SPL tokens with amount=1, decimals=0).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  nft 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

Note: Compressed NFTs (cNFTs) are not detected by this heuristic.

### 6. Whale Detector

Scan the most recent block for large SOL transfers with USD values.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  whales --min-sol 500
```

Note: scans the latest block only — point-in-time snapshot, not historical.

### 7. Network Stats

Live Solana network health: current slot, epoch, TPS, supply, validator
version, SOL price, and market cap.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 8. Price Lookup

Quick price check for any token by mint address or known symbol.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price BONK
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price JUP
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price SOL
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

Known symbols: SOL, USDC, USDT, BONK, JUP, WETH, JTO, mSOL, stSOL,
PYTH, HNT, RNDR, WEN, W, TNSR, DRIFT, bSOL, JLP, WIF, MEW, BOME, PENGU.

---

## Pitfalls

- **CoinGecko rate-limits** — free tier allows ~10-30 requests/minute.
  Price lookups use 1 request per token. Wallets with many tokens may
  not get prices for all of them. Use `--no-prices` for speed.
- **Public RPC rate-limits** — Solana mainnet public RPC limits requests.
  For production use, set SOLANA_RPC_URL to a private endpoint
  (Helius, QuickNode, Triton).
- **NFT detection is heuristic** — amount=1 + decimals=0. Compressed
  NFTs (cNFTs) and Token-2022 NFTs won't appear.
- **Whale detector scans latest block only** — not historical. Results
  vary by the moment you query.
- **Transaction history** — public RPC keeps ~2 days. Older transactions
  may not be available.
- **Token names** — ~25 well-known tokens are labeled by name. Others
  show abbreviated mint addresses. Use the `token` command for full info.
- **Retry on 429** — both RPC and CoinGecko calls retry up to 2 times
  with exponential backoff on rate-limit errors.

---

## Verification

```bash
# Should print current Solana slot, TPS, and SOL price
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```
