---
title: "Hyperliquid — Hyperliquid market data, account history, trade review"
sidebar_label: "Hyperliquid"
description: "Hyperliquid market data, account history, trade review"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hyperliquid

Hyperliquid market data, account history, trade review.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/blockchain/hyperliquid` |
| Path | `optional-skills/blockchain/hyperliquid` |
| Version | `0.1.0` |
| Author | Hugo Sequier (Hugo-SEQUIER), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Hyperliquid`, `Blockchain`, `Crypto`, `Trading`, `Perpetuals`, `Spot`, `DeFi` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hyperliquid Skill

Query Hyperliquid market and account data through the public `/info` endpoint.
Read-only — no API key, no signing, no order placement.

12 commands: `dexs`, `markets`, `spots`, `candles`, `funding`, `l2`, `state`,
`spot-balances`, `fills`, `orders`, `review`, `export`. Stdlib only
(`urllib`, `json`, `argparse`).

---

## When to Use

- User asks for Hyperliquid perp or spot market data, candles, funding, or L2 book
- User wants to inspect a wallet's perp positions, spot balances, fills, or orders
- User wants a post-trade review combining recent fills with market context
- User wants to inspect builder-deployed perp dexs or HIP-3 markets
- User wants a normalized JSON export of candles + funding for backtesting prep

---

## Prerequisites

Stdlib only — no external packages, no API key.

The script reads `~/.hermes/.env` for two optional defaults:

- `HYPERLIQUID_API_URL` — defaults to `https://api.hyperliquid.xyz`. Set to
  `https://api.hyperliquid-testnet.xyz` for testnet.
- `HYPERLIQUID_USER_ADDRESS` — default address for `state`, `spot-balances`,
  `fills`, `orders`, and `review`. If unset, pass the address as the first
  positional argument.

A project `.env` in the current working directory is honored as a dev fallback.

Helper script: `~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py`

---

## How to Run

Invoke through the `terminal` tool:

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py <command> [args]
```

Add `--json` to any command for machine-readable output.

---

## Quick Reference

```bash
hyperliquid_client.py dexs
hyperliquid_client.py markets [--dex DEX] [--limit N] [--sort volume|oi|funding_abs|change_abs|name]
hyperliquid_client.py spots [--limit N]
hyperliquid_client.py candles <coin> [--interval 1h] [--hours 24] [--limit N]
hyperliquid_client.py funding <coin> [--hours 72] [--limit N]
hyperliquid_client.py l2 <coin> [--levels N]
hyperliquid_client.py state [address] [--dex DEX]
hyperliquid_client.py spot-balances [address] [--limit N]
hyperliquid_client.py fills [address] [--hours N] [--limit N] [--aggregate-by-time]
hyperliquid_client.py orders [address] [--limit N]
hyperliquid_client.py review [address] [--coin COIN] [--hours N] [--fills N]
hyperliquid_client.py export <coin> [--interval 1h] [--hours N] [--output PATH]
```

For `state`, `spot-balances`, `fills`, `orders`, and `review`, the address is
optional when `HYPERLIQUID_USER_ADDRESS` is set in `~/.hermes/.env`.

---

## Procedure

### 1. Discover DEXs and Markets

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py dexs

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 15 --sort volume

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spots --limit 15
```

- `--dex` only applies to perp endpoints; omit for the first perp dex.
- Spot pairs may show as `PURR/USDC` or aliases like `@107`.
- HIP-3 markets prefix the coin with the dex, e.g. `mydex:BTC`.

### 2. Pull Historical Market Data

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  candles BTC --interval 1h --hours 72 --limit 48

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  funding BTC --hours 168 --limit 30
```

Time-range endpoints paginate. For larger windows, repeat with a later
`startTime` or use `export` (below).

### 3. Inspect Live Order Book

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  l2 BTC --levels 10
```

Use when asked about book depth, near-term liquidity, or potential market
impact of a large order.

### 4. Review an Account

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  state 0xabc...

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spot-balances
```

`state` returns perp positions; `spot-balances` returns spot inventory.
Use these for "how are my positions?", "what am I holding?", "how much is
withdrawable?".

### 5. Review Fills and Orders

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  fills 0xabc... --hours 72 --limit 25

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  orders --limit 25
```

### 6. Generate a Trade Review

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review 0xabc... --hours 72 --fills 50

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review --coin BTC --hours 168
```

Reports realized PnL, fees, win/loss counts, coin breakdowns, market trend
and average funding for each traded perp, plus heuristics (fee drag,
concentration, counter-trend losses).

For deeper post-trade analysis: start with `review` to find problem coins
or windows → pull `fills` and `orders` for that period → pull `candles`
and `funding` for each traded coin → judge decision quality separately
from outcome quality.

### 7. Export a Reusable Dataset

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 1h --hours 168 --output ./btc-1h-7d.json

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 15m --hours 72 --end-time-ms 1760000000000
```

Output JSON contains: schema version, source metadata, exact time window,
normalized candle rows, normalized funding rows, summary stats. Use
`--end-time-ms` for reproducible windows.

---

## Pitfalls

- Public info endpoints are rate-limited. Large historical queries may
  return capped windows; iterate with later `startTime` values.
- `fills --hours ...` uses `userFillsByTime`, which only exposes a
  recent rolling window — not full archive history.
- `historicalOrders` returns recent orders only; not a full export.
- The `review` command is heuristic. It cannot reconstruct intent,
  order placement quality, or true slippage from fills alone.
- The `export` command writes a normalized dataset, not a backtest
  engine. You still need your own slippage/fill model.
- Spot aliases like `@107` are valid identifiers even when the UI shows
  a friendlier name.
- `l2` is a point-in-time snapshot, not a time series.

---

## Verification

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 5
```

Should print the top Hyperliquid perp markets by 24h notional volume.
