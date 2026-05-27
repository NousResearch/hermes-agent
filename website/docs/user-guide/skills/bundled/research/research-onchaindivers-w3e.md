---
title: "Onchaindivers W3E"
sidebar_label: "Onchaindivers W3E"
description: "Use when querying OnchainDivers / W3E hosted warehouse data for Polymarket, Hyperliquid, or Solana research without exposing credentials"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Onchaindivers W3E

Use when querying OnchainDivers / W3E hosted warehouse data for Polymarket, Hyperliquid, or Solana research without exposing credentials.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/onchaindivers-w3e` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `research`, `clickhouse`, `onchaindivers`, `w3e`, `polymarket`, `hyperliquid`, `solana` |
| Related skills | [`polymarket`](/docs/user-guide/skills/bundled/research/research-polymarket), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# OnchainDivers / W3E Research Access

## Overview

Use this skill when the user wants practical research on **Polymarket**, **Hyperliquid**, or **Solana** using the hosted OnchainDivers / W3E data warehouse instead of only public REST endpoints.

This workflow is optimized for:
- quick access validation,
- safe handling of bearer keys and readonly ClickHouse credentials,
- schema discovery,
- writing real SQL against the warehouse,
- and staying honest when docs/examples diverge from the actual tables.

The main lesson: **treat the docs as a map, not as ground truth**. Verify real table names and row availability with live queries before building analysis on top.

## When to Use

- User already has OnchainDivers / W3E access and wants live data checks
- User wants research on Polymarket fills, market metadata, or event metadata
- User wants Hyperliquid fills / funding / wallet state from a warehouse
- User wants Solana research from curated tables instead of raw RPC scraping
- You need repeatable SQL-backed evidence, not hand-wavy market commentary

Do **not** use this skill when:
- public Polymarket REST data is enough,
- the user wants trading execution,
- or credentials are unavailable and no public fallback exists.

## Safe Credential Handling

Never commit or print real secrets into repo files, chat summaries, skills, or Git history.

Preferred pattern:

```bash
export ONCHAINDIVERS_API_KEY='<redacted>'
export W3E_HL_CH_USER='<redacted>'
export W3E_HL_CH_PASS='<redacted>'
```

Use placeholders in docs and examples only.

For bearer API calls, prefer ephemeral shell variables over inline literals:

```bash
curl -sS \
  -H "Authorization: Bearer $ONCHAINDIVERS_API_KEY" \
  -H 'Content-Type: application/json' \
  https://db-access-test.onchaindivers.com/v1/polymarket/query \
  -d '{"query":"SELECT 1"}'
```

For direct ClickHouse HTTP access, use readonly users only.

## Practical Workflow

### 1) Confirm access first

Do not assume stored creds are valid just because local code mentions W3E.

Start with tiny verification queries:

```sql
SELECT count() AS rows FROM raw_event_meta
SELECT count() AS rows FROM raw_market_meta
SELECT count() AS rows FROM polymarket_order_filled_v3
```

For Hyperliquid:

```sql
SELECT count() AS rows FROM agg_fulfilled_order
```

For Solana:

```sql
SELECT count() AS rows FROM solana_blocks
```

If auth is flaky, re-check whether you are using the intended endpoint, header format, and freshest key.

### 2) Discover schema from the live system

Docs may show simplified names such as `markets`, `tokens`, `fills`, or `fundings`, while the live warehouse can expose different physical tables.

Prefer:

```sql
SELECT name
FROM system.tables
WHERE database = currentDatabase()
ORDER BY name
```

If that fails through the bearer endpoint, fall back to:
- docs pages for conceptual orientation,
- known good tables from prior validation,
- or direct ClickHouse access where available.

### 3) Use the warehouse in three layers

For Polymarket, the most useful pattern is:
- `raw_event_meta` → event-level framing
- `raw_market_meta` → market-level metadata
- `polymarket_order_filled_v3` → real fills / executed prices

This is the right path when validating claims about overpriced or underpriced volatility, monthly BTC structures, or entry timing.

### 4) Build from metadata to fills

Do not start with abstract strategy talk. First identify the actual markets, then join to fills.

Good sequence:
1. find candidate BTC monthly markets in metadata,
2. inspect titles / slugs / end dates,
3. only then pull fills for those markets,
4. compare executed prices and realized outcomes.

## Known Reality-Check on Table Names

Based on practical use, these names are the reliable starting points.

### Polymarket

Use these first:
- `raw_event_meta`
- `raw_market_meta`
- `polymarket_order_filled_v3`

### Hyperliquid

Useful starting points:
- `agg_fulfilled_order`
- `raw_node_fills_by_block`
- `view_perpetual_wallet`
- `view_wallet_position`
- `dxn_funding`
- `perp_asset_meta`
- `perp_asset_stats`
- `spot_asset_meta`
- `spot_pair_meta`

### Solana

Useful starting points:
- `solana_blocks`
- `tx_timestamps`
- `jito_tips`
- `pumpfun_token_creation`
- `pumpfun_all_swaps`
- `pumpswap_all_swaps`
- `raydium_all_swaps`
- `meteora_swaps`
- `meteora_dynamic_bonding_swaps`
- `pfamm_migrations`
- `max_caps`

## First Polymarket Query Pattern

When the user wants **monthly BTC markets**, start with metadata discovery, not fills.

Example shape:

```sql
SELECT
  event_id,
  market_id,
  event_title,
  market_question,
  slug,
  end_date,
  active,
  closed
FROM raw_market_meta
WHERE lower(event_title) LIKE '%bitcoin%'
   OR lower(event_title) LIKE '%btc%'
   OR lower(market_question) LIKE '%bitcoin%'
   OR lower(market_question) LIKE '%btc%'
ORDER BY end_date DESC
LIMIT 200
```

Then narrow to monthly-expiry style wording such as:
- `end of month`
- `by may 31`
- `in may`
- `monthly`
- or explicit month/year references in titles and slugs.

After that, pull fills from `polymarket_order_filled_v3` only for the shortlisted markets.

## Query Style Guidelines

- Prefer `SELECT` and `WITH` only
- Keep first queries cheap and inspectable
- Use `LIMIT` during discovery
- Add explicit `ORDER BY` so outputs are reproducible
- Save intermediate candidate IDs before running heavy fill queries
- When comparing strategies, use **executed fill prices**, not just quoted probabilities



## Common Pitfalls

1. **Trusting docs over reality.**
   If docs say `fills` but the live table is `agg_fulfilled_order`, trust the live system.

2. **Assuming a local codebase means current access works.**
   Old helper scripts can survive long after creds rotate.

3. **Leaking keys in shell history or commits.**
   Use env vars and placeholders only.

4. **Jumping directly into strategy conclusions.**
   First prove the market set and entry prices you are talking about actually existed.

5. **Using abstract volatility narratives instead of actual fill data.**
   For Polymarket research, the fill table is where the truth starts.

## Verification Checklist

- [ ] Access verified with a tiny live query
- [ ] No secrets printed or committed
- [ ] Physical table names confirmed from live system or previously verified evidence
- [ ] Metadata query run before fill query
- [ ] Analysis grounded in actual rows, not only docs or intuition
- [ ] If docs and schema disagree, the discrepancy is called out explicitly

## Supporting Reference

See `references/example-queries.md` for sanitized curl + SQL snippets covering Polymarket, Hyperliquid, and Solana.
