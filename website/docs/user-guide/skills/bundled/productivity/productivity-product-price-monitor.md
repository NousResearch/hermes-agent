---
title: "Product Price Monitor — Watch product, flight, or listing prices; alert on target"
sidebar_label: "Product Price Monitor"
description: "Watch product, flight, or listing prices; alert on target"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Product Price Monitor

Watch product, flight, or listing prices; alert on target.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity\product-price-monitor` |
| Version | `0.1.0` |
| Author | Ben Barclay (benbarclay), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Prices`, `Availability`, `Shopping`, `Travel`, `Alerts` |
| Related skills | [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Product Price Monitor

Monitor a concrete purchasable item and alert on a normalized all-in price or availability condition. Handle variants, taxes, fees, currencies, stock, cancellation terms, and duplicate alerts explicitly. Setup runs once in the foreground; the recurring check runs as a `cronjob` tick (the `price-watch` automation blueprint scaffolds this).

## When to Use

- "Alert me when this laptop drops below $1,000."
- "Watch these flights for a fare under $500."
- "Tell me when this hotel has a refundable room."
- "Track ticket/listing availability."
- A cron tick fires for an existing price watch (steps 4-6).

Don't use for: one-off "what does this cost right now" lookups (use `web_search`/`web_extract` directly).

## Procedure — Setup (foreground, once)

### 1. Define the exact item

Record source URL/provider, product/listing ID where available, variant, quantity, location, dates, travelers/guests, membership/login assumptions, condition, seller, and acceptable substitutes. Done when two variants cannot be confused.

### 2. Define the alert condition

Specify currency, all-in vs pre-tax price, maximum price, availability/stock rule, shipping, refundability, cabin/room/ticket class, cooldown, and notification destination. Done when synthetic examples have deterministic alert decisions.

### 3. Establish a live baseline, then schedule

Fetch a bounded live result with `web_extract` or `browser_navigate` and record retrieval time, source price, fees/taxes, availability, and terms. Do not schedule until one foreground fetch works. Write the watch contract (item, condition, baseline observation) to a state file under `~/.hermes/price-watches/<watch-slug>.json`, then create the job:

```
cronjob(action="create",
        schedule="every 6h",
        prompt="Load the product-price-monitor skill and run the tick for the watch contract at ~/.hermes/price-watches/<watch-slug>.json.",
        deliver=<user's destination>)
```

Pick a cadence that respects rate limits and site terms. Done when the baseline matches the exact item contract and the job exists.

## Procedure — Tick (each scheduled run)

### 4. Fetch and normalize

Re-fetch the source. Convert currency only with a timestamped rate and retain the source currency. Separate base price, mandatory fees, shipping/taxes, total, and availability. Exclude volatile page metadata. A failed fetch means unknown state: report or skip, but never overwrite the last good observation with an error page. Done when the observation is comparable to the baseline or explicitly marked failed.

### 5. Compare and suppress duplicates

Alert on threshold entry, qualifying availability, material lower price, or recovery as requested. Store the last good observation and last alert fingerprint in the state file. Replaying the same offer must send no second alert; respect the cooldown. Done when the alert decision is deterministic against stored state.

### 6. Deliver or stay silent

When a condition is met, the alert includes: exact item/variant, observed all-in price and source currency, availability/terms, threshold, retrieval timestamp, source link, and important uncertainty. Never claim inventory is reserved. When nothing qualifies, stay silent — no "still watching" noise unless a periodic all-clear was requested. Done when the state file reflects this run.

## Pitfalls

- Comparing a base fare with an all-in threshold.
- Alerting on the wrong size, seller, cabin, dates, or room terms.
- Overwriting a last-known-good value with an error page.
- Polling aggressively enough to trigger blocking or violate site terms.
- Scheduling before a single foreground fetch has succeeded.

## Verification

- [ ] The watch contract pins the item so two variants cannot be confused.
- [ ] One foreground fetch succeeded before any job was created.
- [ ] Alert decisions replay deterministically from the state file; duplicates suppressed.
- [ ] Failed fetches never replaced last-known-good state.
- [ ] Alerts carry all-in price, source currency, timestamp, and source link.
