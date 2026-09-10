---
title: "Evm — Read-only EVM client: wallets, tokens, gas across 8 chains"
sidebar_label: "Evm"
description: "Read-only EVM client: wallets, tokens, gas across 8 chains"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Evm

Read-only EVM client: wallets, tokens, gas across 8 chains.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/blockchain/evm` |
| Path | `optional-skills/blockchain\evm` |
| Version | `1.0.0` |
| Author | Mibayy (@Mibayy), youssefea (@youssefea), ethernet8023 (@ethernet8023), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `EVM`, `Ethereum`, `BNB`, `BSC`, `Base`, `Arbitrum`, `Polygon`, `Optimism`, `Avalanche`, `zkSync`, `Blockchain`, `Crypto`, `Web3`, `DeFi`, `NFT`, `ENS`, `Whale`, `Security` |
| Related skills | [`solana`](/docs/user-guide/skills/optional/blockchain/blockchain-solana) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# EVM Blockchain Skill

Query EVM-compatible blockchain data across 8 chains with USD pricing.
14 commands: wallet portfolio, token info, transactions, activity, gas tracker,
network stats, price lookup, multi-chain scan, whale detection, ENS resolution,
allowance checker, contract inspector, and transaction decoder.

Supports 8 chains: Ethereum, BNB Chain (BSC), Base, Arbitrum One, Polygon,
Optimism, Avalanche (C-Chain), zkSync Era.

No API key needed. Zero external dependencies — Python standard library only
(urllib, json, argparse, threading).

> **Supersedes the standalone `base` skill.** Base-specific tokens (AERO, DEGEN,
> TOSHI, BRETT, WELL, cbETH, cbBTC, wstETH, rETH) and all Base RPC functionality
> previously living under `optional-skills/blockchain/base/` have been folded
> into this skill. Pass `--chain base` to any command for Base coverage.

---

## When to Use
- User asks for a wallet balance or portfolio on any EVM chain
- User wants to check the same wallet across ALL chains at once
- User wants to inspect a transaction by hash (or decode what it did)
- User wants ERC-20 token metadata, price, supply, or market cap
- User wants recent transaction history for an address
- User wants current gas prices or to compare fees across chains
- User wants to find large whale transfers in recent blocks
- User asks to resolve an ENS name (vitalik.eth) or reverse-lookup an address
- User wants to check if a contract has dangerous token approvals
- User wants to inspect a smart contract (proxy? ERC-20? ERC-721? bytecode size?)
- User wants to compare gas costs across chains before a transaction

---

## Prerequisites
Python 3.8+ standard library only. No pip installs required.
Pricing: CoinGecko free API (rate-limited, ~10-30 req/min).
ENS: ensideas.com public API.
Tx decoding: 4byte.directory public API.

Override RPC endpoint: `export EVM_RPC_URL=https://your-rpc.com`

Helper script path: `~/.hermes/skills/blockchain/evm/scripts/evm_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/blockchain/evm/scripts/evm_client.py

# Network & prices
python $SCRIPT stats                            # Ethereum stats
python $SCRIPT stats --chain arbitrum           # Arbitrum stats
python $SCRIPT compare                          # Gas + prices ALL 8 chains

# Wallet
python $SCRIPT wallet 0xd8dA...96045            # Portfolio (ETH + ERC-20)
python $SCRIPT wallet 0xd8dA...96045 --chain bsc
python $SCRIPT multichain 0xd8dA...96045        # Same wallet on ALL chains

# Tokens & prices
python $SCRIPT price ETH
python $SCRIPT price 0xdAC1...1ec7              # By contract address
python $SCRIPT token 0xdAC1...1ec7              # ERC-20 metadata + market cap

# Transactions
python $SCRIPT tx 0x5c50...f060                 # Transaction details
python $SCRIPT decode 0x5c50...f060             # Decode input data (4byte.directory)
python $SCRIPT activity 0xd8dA...96045          # Recent transactions

# Gas
python $SCRIPT gas                              # Gas prices + cost estimates
python $SCRIPT gas --chain optimism

# Security
python $SCRIPT allowance 0xd8dA...96045         # Dangerous ERC-20 approvals
python $SCRIPT contract 0xdAC1...1ec7           # Contract inspection (proxy? standards?)

# ENS
python $SCRIPT ens vitalik.eth                  # Name -> address + profile
python $SCRIPT ens 0xd8dA...96045               # Address -> ENS name

# Whale detection
python $SCRIPT whale                            # Large transfers (last 20 blocks, >$10k)
python $SCRIPT whale --blocks 50 --min-usd 100000 --chain arbitrum
```

---

## Procedure

### 0. Setup Check
```bash
python --version   # 3.8+ required
python ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats
```

### 1. Wallet Portfolio
Native balance + known ERC-20 tokens, sorted by USD value.
```bash
python $SCRIPT wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
python $SCRIPT wallet 0xd8dA... --chain bsc --no-prices   # faster
```

### 2. Multi-Chain Scan
Scans all 8 chains simultaneously for the same address using threads.
```bash
python $SCRIPT multichain 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```
Output: per-chain native balance + token holdings + grand total USD.

### 3. Compare (Gas + Prices)
All 8 chains queried in parallel. Shows cheapest/most expensive chain.
```bash
python $SCRIPT compare
```

### 4. Transaction Details & Decode
```bash
python $SCRIPT tx 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060
python $SCRIPT decode 0x5c504ed...   # Shows human-readable function signature
```
Decode uses 4byte.directory to translate 0xa9059cbb -> transfer(address,uint256).

### 5. ENS Resolution
```bash
python $SCRIPT ens vitalik.eth          # -> 0xd8dA... + avatar + social links
python $SCRIPT ens 0xd8dA...96045       # -> vitalik.eth
```

### 6. Allowance Checker (Security)
Checks ERC-20 approvals granted to known DEX/bridge contracts.
```bash
python $SCRIPT allowance 0xYourWallet
```
Flags UNLIMITED approvals as HIGH risk.

### 7. Contract Inspector
```bash
python $SCRIPT contract 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48   # USDC (proxy)
python $SCRIPT contract 0xdAC17F958D2ee523a2206206994597C13D831ec7   # USDT (ERC-20)
```
Detects: proxy (EIP-1967/EIP-1167), ERC-20, ERC-721, ERC-165. Shows bytecode size and implementation address for proxies.

### 8. Whale Detection
```bash
python $SCRIPT whale                                    # ETH, last 20 blocks, >$10k
python $SCRIPT whale --blocks 50 --min-usd 50000 --chain bsc
```

### 9. Gas Tracker
```bash
python $SCRIPT gas
python $SCRIPT gas --chain polygon
```
Shows gwei price + USD cost for: transfer, ERC-20 transfer, approve, swap, NFT mint, NFT transfer.

---

## Supported Chains
| Key       | Name           | Native | Chain ID |
|-----------|----------------|--------|----------|
| ethereum  | Ethereum       | ETH    | 1        |
| bsc       | BNB Chain      | BNB    | 56       |
| base      | Base           | ETH    | 8453     |
| arbitrum  | Arbitrum One   | ETH    | 42161    |
| polygon   | Polygon        | POL    | 137      |
| optimism  | Optimism       | ETH    | 10       |
| avalanche | Avalanche C    | AVAX   | 43114    |
| zksync    | zkSync Era     | ETH    | 324      |

---

## Pitfalls
- CoinGecko free tier: ~10-30 req/min. Use `--no-prices` for faster wallet scans.
- Public RPCs may throttle. Set EVM_RPC_URL to a private endpoint for production.
- `wallet` and `allowance` only check known token list (~30 tokens per chain). Use a block explorer for complete token discovery.
- `activity` scans recent blocks only (max 200). For full history, use Etherscan API.
- `multichain` runs 8 parallel threads — can trigger rate limits on public RPCs.
- ENS resolution depends on a single public endpoint (ensideas.com / ens.vitalik.ca) with no fallback. If that endpoint is down, `ens` will fail — re-run later or use a block explorer.
- Tx decoding depends on a single public endpoint (4byte.directory) with no fallback. Selectors not in their database show up as `unknown`.
- **L2 gas estimates are L2-execution only.** On rollups like Base, Arbitrum, Optimism, and zkSync, the actual transaction cost also includes an L1 data-posting fee that depends on calldata size and current L1 gas prices. The `gas` command does not estimate that L1 component. For Base specifically, see the network's L1 fee oracle (contract `0x420000000000000000000000000000000000000F`).
- Address / tx-hash inputs are validated for 0x-prefix + correct length + hex, but EIP-55 checksum casing is **not** enforced (RPC endpoints accept any-case hex).

---

## Verification
```bash
# Should print current block, gas price, ETH price
python ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats

# Should resolve vitalik.eth to 0xd8dA...
python ~/.hermes/skills/blockchain/evm/scripts/evm_client.py ens vitalik.eth
```
