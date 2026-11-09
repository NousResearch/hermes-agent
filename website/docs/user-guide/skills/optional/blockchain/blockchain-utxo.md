---
title: "Utxo — Query Bitcoin, Litecoin, Dogecoin and Zcash on-chain data"
sidebar_label: "Utxo"
description: "Query Bitcoin, Litecoin, Dogecoin and Zcash on-chain data"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Utxo

Query Bitcoin, Litecoin, Dogecoin and Zcash on-chain data.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/blockchain/utxo` |
| Path | `optional-skills/blockchain/utxo` |
| Version | `0.1.0` |
| Author | Atif Nazir (doersf) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Bitcoin`, `Litecoin`, `Dogecoin`, `Zcash`, `UTXO`, `Blockchain`, `Crypto`, `SoChain` |
| Related skills | [`evm`](../../optional/blockchain/blockchain-evm.md), [`solana`](../../optional/blockchain/blockchain-solana.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# UTXO Chains Skill

Read-only Bitcoin (BTC), Litecoin (LTC), Dogecoin (DOGE) and Zcash (ZEC)
lookups through the SoChain REST API v3: balances, address history, unspent
outputs, transactions and their status, blocks, mempool fee rates and prices.
Testnets too. No signing and no broadcasting.

10 commands. Stdlib only (`urllib`, `json`, `argparse`). Zcash coverage is
the transparent side (t-addresses).

---

## When to Use

- User asks for a BTC, LTC, DOGE or ZEC address balance, now or at a past date
- User wants an address's transaction history, filtered by date or direction
- User wants the unspent outputs (UTXOs) of an address
- User wants to inspect a transaction by txid, or know if it's pending, mined,
  replaced or dropped
- User asks for current fee rates or mempool size on a UTXO chain
- User wants a block by height or hash, or a coin's price now or at a past time
- User wants to check whether an address is valid on a network

---

## Prerequisites

Stdlib only — no external packages.

`SOCHAIN_API_KEY` is optional. Without it, lookups work up to 2,500 requests a
day per IP. A free account at https://sochain.com/dashboard (email sign-in) gives
a key with 75,000 requests a month; paid plans raise that. The script sends the
key as the `API-KEY` header.

Helper script: `~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py`

---

## How to Run

Invoke through the `terminal` tool:

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py <command> [args]
```

Output is the API's `data` object as JSON. Amounts are decimal strings in whole
coins (e.g. `"0.00012345"`); times are unix seconds.

---

## Quick Reference

```bash
utxo_client.py balance  <address> [--network NET] [--time T | --height H]
utxo_client.py address  <address> [--network NET]
utxo_client.py txs      <address> [--network NET] [--page N]
utxo_client.py txs      <address> [--from-time T] [--to-time T] [--direction in|out|both] [--page-size N] [--cursor C]
utxo_client.py utxos    <address> [--network NET] [--page N]
utxo_client.py validate <address> [--network NET]
utxo_client.py tx       <txid> [--network NET]
utxo_client.py status   <txid> [--network NET]
utxo_client.py block    <hash_or_height> [--network NET]
utxo_client.py network  [--network NET]
utxo_client.py price    [--network NET] [--at UNIX_SECONDS]
```

Networks: `BTC`, `LTC`, `DOGE`, `ZEC`, and testnets `BTCTEST` (testnet4),
`LTCTEST`, `DOGETEST`, `ZECTEST`. Address commands infer the mainnet network
from the prefix (`bc1`/`1`/`3` BTC, `ltc1`/`L`/`M` LTC, `D`/`A`/`9` DOGE,
`t1`/`t3` ZEC). `tx`, `status`, `block`, `network` and `price` default to `BTC`.

---

## Procedure

### 1. Balance and Address Summary

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  balance bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh

python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  balance DOGE_ADDRESS --time 2025-12-31
```

`balance` returns `confirmed` and `unconfirmed`; with `--time` (unix seconds,
a UTC date meaning the end of that day, or RFC 3339) or `--height`, the
confirmed balance at that point. `address` adds totals sent and received and
transaction counts.

### 2. Transaction History

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  txs LTC_ADDRESS --page 2

python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  txs DOGE_ADDRESS --from-time 2025-01-01 --to-time 2025-12-31 --direction in
```

Unfiltered: 10 per page, pending first, then newest first. With any filter the
result pages by `next_cursor`: pass it back with `--cursor` until it is empty.

### 3. Unspent Outputs

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  utxos bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
```

### 4. Transaction and Status

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  tx TXID --network DOGE

python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py \
  status TXID --network BTC
```

`tx` gives inputs, outputs, fee, block and confirmations. `status` says
`pending`, `mined`, `replaced` (with the replacing txid) or `dropped`.

### 5. Fees, Blocks and Price

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py network --network BTC
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py block 840000
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py price --network ZEC
```

`network` returns the newest block, hashrate, mempool size and the next
projected blocks with their fee rates — use these to suggest a fee.

---

## Pitfalls

- **Pass `--network` for transactions and blocks.** A txid doesn't say which
  chain it's on; a DOGE txid looked up on the default `BTC` is "not found".
- **Ambiguous prefixes.** Testnet `m`/`n`/`2` addresses and old Litecoin `3`
  addresses can't be told apart by prefix; pass `--network`.
- **BTCTEST is testnet4**, not testnet3.
- **Rate limits.** Keyless use is 2,500 requests a day per IP; the script
  retries a 429 twice (honouring `Retry-After`), then says to set
  `SOCHAIN_API_KEY`. A 401 means the key is wrong.
- **Zcash shielded funds** are not visible; only t-addresses and value entering
  or leaving the shielded pools.
- **Read-only.** There is no send or broadcast command.
- **Never shorten addresses or txids** when reporting results.

---

## Verification

```bash
python ~/.hermes/skills/blockchain/utxo/scripts/utxo_client.py network --network BTC
```

Should print the newest Bitcoin block height, mempool size and fee rates.
