---
title: "Stellar — Query Stellar accounts, assets, txs, and contract TTLs"
sidebar_label: "Stellar"
description: "Query Stellar accounts, assets, txs, and contract TTLs"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stellar

Query Stellar accounts, assets, txs, and contract TTLs.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/blockchain/stellar` |
| Path | `optional-skills/blockchain/stellar` |
| Version | `0.1.0` |
| Author | Kaan Kacar (kaankacar) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Stellar`, `Soroban`, `Blockchain`, `Crypto`, `Web3`, `Horizon`, `RPC`, `USDC`, `Payments`, `DeFi` |
| Related skills | [`solana`](../../optional/blockchain/blockchain-solana.md), [`evm`](../../optional/blockchain/blockchain-evm.md), [`mpp-agent`](../../optional/payments/payments-mpp-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Stellar Skill

Query Stellar mainnet or testnet through Horizon and Stellar RPC: account balances
with spendable XLM and reserves, classic assets and their issuers, transactions with
decoded Soroban arguments and failure codes, contract instances with TTL status,
contract events, and `stellar.toml` (SEP-1) metadata. Read-only: no keys, no
signing, no submission. The only write is testnet friendbot funding.

10 commands: `stats`, `account`, `tx`, `activity`, `asset`, `price`, `contract`,
`events`, `toml`, `fund`. Stdlib only (`urllib`, `json`, `argparse`, `tomllib`).
USD values come from the on-chain DEX (XLM/USDC order book), not a price API.

---

## When to Use

- User asks for a Stellar account's XLM, USDC, or other balances, or how much it can spend
- User pastes a `G...` address, `C...` contract id, 64-hex tx hash, or `name*domain` federation address
- User asks why a Stellar transaction failed, or what a Soroban invocation did
- User asks whether a Soroban contract's instance or code is still live (TTL, archival)
- User wants recent events emitted by a contract (token transfers, custom events)
- User asks who issues an asset, who holds it, or which SEP services a domain offers
- User needs a funded testnet account before building

Don't use for: signing or submitting transactions, writing Soroban contracts, or
wallet integration. For building on Stellar, read `references/developer-quickstart.md`
in this skill first; it points at the official `stellar/stellar-dev-skill` for depth.

---

## Prerequisites

Stdlib only. No packages, no API key. Public endpoints:

| Network | Horizon | Stellar RPC |
|---|---|---|
| mainnet (default) | `https://horizon.stellar.org` | `https://mainnet.sorobanrpc.com` |
| testnet | `https://horizon-testnet.stellar.org` | `https://soroban-testnet.stellar.org` |

Optional settings in `${HERMES_HOME:-~/.hermes}/.env` (a project `.env` is a dev fallback):

- `STELLAR_NETWORK`: `mainnet` or `testnet`. `--network` on any command overrides it.
- `STELLAR_HORIZON_URL`, `STELLAR_RPC_URL`: endpoint overrides for the selected
  network. Point `STELLAR_RPC_URL` at your own provider for production volume.

Helper script: `~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py`

---

## How to Run

Invoke through the `terminal` tool:

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py <command> [args] [--network testnet] [--json]
```

Add `--json` to any command for machine-readable output (full decoded storage, raw numbers).

---

## Quick Reference

```bash
stellar_client.py stats                                   # protocol, ledger, fees, RPC retention, XLM/USD
stellar_client.py account  <G... | name*domain> [--no-prices] [--limit N]
stellar_client.py tx       <hash>                         # ops, decoded Soroban args, result codes
stellar_client.py activity <G...> [--limit N]             # recent operations, newest first
stellar_client.py asset    <CODE:ISSUER | USDC | EURC>    # holders, supply, flags, SAC id, issuer org
stellar_client.py price    <XLM | CODE | CODE:ISSUER>     # DEX mid price in XLM and USD
stellar_client.py contract <C...>                         # WASM or SAC, instance + code TTL, storage keys
stellar_client.py events   <C...> [--ledgers N] [--limit N]
stellar_client.py toml     <domain>                       # SEP-1: org, currencies, SEP endpoints
stellar_client.py fund     <G...> --network testnet       # friendbot, testnet only
```

Known mainnet codes resolve to their issuer without `:ISSUER`: USDC, EURC, AQUA,
yXLM, yUSDC, SHX, ARST, VELO. Testnet knows USDC. Anything else needs `CODE:ISSUER`.

---

## Procedure

### 1. Check the network first

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py stats
```

Done when the output shows the protocol version, latest ledger, and a healthy RPC.
Note the RPC retention window: `events` only reaches back that far (about 7 days on
public endpoints). Horizon keeps full transaction history.

### 2. Inspect an account

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  account GAHK7EEG2WWHVKDNT4CEQFZGKF2LGDSW2IVM4S5DP42RBW3K6BTODB4A

python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  account alice*example.com --no-prices
```

Report **spendable** XLM, not the raw balance: every account locks a base reserve
(1 XLM plus 0.5 XLM per trustline, offer, signer, or data entry) and the script
computes it. "Account does not exist" means it was never funded; on Stellar an
account is created by its first XLM deposit. Done when balances, signers and
thresholds, and flags are summarised.

### 3. Explain a transaction

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  tx 2dfb76d474001ecba835f2c725c62f62f3753d9037cfc7f53dd21bf8ad586230
```

Soroban invocations print as `CONTRACT.method(args)` with arguments decoded from
XDR, plus Stellar Asset Contract balance changes. Failed transactions print the
tx-level code (`txBAD_SEQ`, `txINSUFFICIENT_FEE`, ...) and the first failing
operation's code (`payment: UNDERFUNDED`, `invoke_host_function: ENTRY_ARCHIVED`).
Done when you can state what the transaction did or why it failed.

### 4. Look up an asset, issuer, or anchor

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py asset USDC

python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  asset AQUA:GBNZILSTVQZ4R7IKQDGHYGY2QXL5QOFJYQMXPKWRRM5PAV7Y4M67AQUA

python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py toml aqua.network
```

`asset` prints holder counts, supply in accounts versus contracts, issuer flags
(`auth_required`, `auth_revocable`, `auth_clawback_enabled`), the SAC contract id,
and the issuing organisation from its `stellar.toml`. `toml` lists which SEPs a
domain serves (SEP-6/24 deposits, SEP-10 auth, SEP-31 payments, ...). Done when
the asset's issuer, supply, and controls are stated.

### 5. Check a contract's health

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  contract CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75
```

Shows whether the id is a Stellar Asset Contract (and for which asset) or a WASM
contract (hash, code size, function count), plus the TTL of both the instance and
the code: ledgers and days remaining, or **EXPIRED**. An expired entry is the usual
cause of `ENTRY_ARCHIVED` failures; it needs a `restore_footprint` before calls work.
Done when the executable type and both TTL states are reported.

### 6. Read contract events

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  events CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75 --ledgers 60 --limit 10
```

Topics and values are decoded (symbols, addresses, i128 amounts). SAC amounts are
raw integers with 7 decimals (`50000000` = 5 USDC). Busy contracts emit thousands
of events per hour: keep `--ledgers` small (about 5 seconds per ledger) and widen
only when the window is empty. Done when the relevant events are listed with
ledger, time, and tx hash.

### 7. Get a funded testnet account (development only)

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  fund GAHK7EEG2WWHVKDNT4CEQFZGKF2LGDSW2IVM4S5DP42RBW3K6BTODB4A --network testnet
```

Friendbot funds a `G...` address once with testnet XLM. The keypair comes from the
user's wallet or the Stellar CLI (`stellar keys generate`); this skill never
creates or holds secret keys and refuses `S...` seeds as input. Done when the new
balance prints.

---

## Pitfalls

- **Public rate limits.** Horizon allows roughly 3,600 requests per hour per IP and
  the public RPC throttles too. `account` prices each trustline with one order-book
  call; use `--no-prices` for large accounts and set `STELLAR_RPC_URL` for volume.
- **Prices are DEX mid prices.** Thin books give odd numbers, so the spread is
  printed. Testnet prices are not real. XLM/USD falls back to CoinGecko only when
  the XLM/USDC book is empty.
- **Two addresses per asset.** A classic asset is `CODE:G...issuer`; its Stellar
  Asset Contract is a `C...` id. Trustlines and payments use the former, Soroban
  calls use the latter. `asset` prints both.
- **`M...` muxed addresses** are a `G...` account plus an id. Query the `G...`.
- **RPC history is about 7 days** (`stats` prints the exact window). Older contract
  events need an indexer; older transactions still resolve through Horizon.
- **Horizon amounts use 7 decimals**; event values and SAC balance changes are raw
  integers in the same scale.
- **Endpoint overrides follow the selected network.** A mainnet `STELLAR_RPC_URL`
  plus `--network testnet` mixes a mainnet RPC with testnet Horizon. Unset the
  override when switching networks.
- **Horizon's `stellar.toml` link follows the issuer's `home_domain`**, which can
  404 (Circle's USDC points at `circle.com`; the file lives at `centre.io`). `asset`
  degrades gracefully; run `toml <domain>` by hand when you know the right host.

---

## Verification

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py stats

python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py \
  contract CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75
```

The first prints the protocol version, latest ledger, fees, RPC retention, and the
XLM price. The second identifies the mainnet USDC Stellar Asset Contract with a live
instance TTL.
