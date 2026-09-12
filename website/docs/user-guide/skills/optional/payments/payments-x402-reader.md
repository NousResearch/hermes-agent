---
title: "X402 Reader — Pay x402 to read a public URL as markdown"
sidebar_label: "X402 Reader"
description: "Pay x402 to read a public URL as markdown"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# X402 Reader

Pay x402 to read a public URL as markdown.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/payments/x402-reader` |
| Path | `optional-skills/payments/x402-reader` |
| Version | `0.1.0` |
| Author | twzrd (twzrd-sol), Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `Payments`, `HTTP-402`, `x402`, `Reader`, `Solana` |
| Related skills | [`mpp-agent`](/docs/user-guide/skills/optional/payments/payments-mpp-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# X402 Reader Skill

Turns a public URL into markdown by paying the live x402 reader
(`https://reader.outbid.sh`). Quote is free. Live spend is opt-in and
capped before sign. This is the Solana/Base USDC x402 path; MPP 402s
belong to `mpp-agent`.

Gated `[linux, macos]` while the npm client matures on Windows.

## When to Use

- Need page text and the origin is fat HTML, JS-walled, or already 402s
- A fetch returns `HTTP 402` with x402 `accepts` (Solana or Base USDC)
- User says "scrape", "read this URL", "pay the 402", or "x402 reader"

Don't use for: MPP/`www-authenticate: tempo` (use `mpp-agent`), Stripe
checkout (use `stripe-link-cli`), or adding 402 to your own API.

## Prerequisites

- Node.js 20+ on `PATH` (`node --version` via `terminal`)
- For live pay only: a funded Solana or Base USDC wallet file, plus
  `X402_READER_PAYMENTS_ENABLED=1`
- No wallet is required to quote

## How to Run

Install the official optional skill, then either add the MCP or quote
with `terminal`:

```
hermes skills install official/payments/x402-reader
hermes mcp install x402-reader
```

`hermes mcp install` writes a stdio server (`npx -y x402-reader-mcp@0.1.2`)
that exposes `scrape` ($0.005) and `browse` ($0.05). Unarmed, both tools
return a free 422 miss or a visible 402 quote.

## Quick Reference

| Tool | Route | Price |
|---|---|---:|
| `scrape` | `GET /scrape?url=` | $0.005 USDC |
| `browse` | `GET /browse?url=` | $0.05 USDC |

`scrape` never auto-opens `browse`. A `needs_browser` 422 is terminal.

## Procedure

### 1. Quote (no wallet)

```
curl -sS -D - "https://reader.outbid.sh/scrape?url=https://example.com/"
```

Expect HTTP 402 and an `accepts` array. Parse the body with this skill's
`scripts/quote.py` (no network; stdin is the JSON body):

```
python3 scripts/quote.py
```

Prefer the `solana` / `solana:` row. Amount `5000` is $0.005 USDC.

### 2. Pay under a ceiling

Arm payments only when the user wants the page:

- `X402_READER_PAYMENTS_ENABLED=1`
- `SVM_PRIVATE_KEY_FILE` (Solana JSON keypair) or `EVM_PRIVATE_KEY_FILE`
- Caps default to $0.05 per call and $1.00 per process; refuse above that
  **before** signing

Then call the MCP `scrape` tool, or:

```
npx -y x402-reader-mcp@0.1.2
```

Do not `read_file` the key. Do not paste key material into the transcript.

### 3. Verify

A paid 200 is JSON `{ok, title, markdown, word_count}` plus a
`PAYMENT-RESPONSE` header with `payer` and `transaction`. A leftover 402
is a failed pay, not a quote.

## Pitfalls

- **MPP vs x402.** A `www-authenticate: tempo` header is `mpp-agent`, not
  this skill.
- **No silent browse upgrade.** `needs_browser` is free and terminal.
- **Do not curl a 402 and stop** when the user asked for the page — quote,
  then pay under the cap, or report the price and stop.
- **Wallet keys stay out of context.** Point at a file path; never dump it.

## Verification

```
hermes skills install official/payments/x402-reader
hermes mcp test x402-reader
```

`hermes mcp test` must list `scrape` and `browse`. A quote against
`https://example.com/` must show amount 5000.
