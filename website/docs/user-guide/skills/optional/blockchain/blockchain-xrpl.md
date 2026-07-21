---
title: "Xrpl — Read-only XRP Ledger account and transaction audit"
sidebar_label: "Xrpl"
description: "Read-only XRP Ledger account and transaction audit."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Xrpl

Read-only XRP Ledger account and transaction audit.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/blockchain/xrpl` |
| Path | `optional-skills/blockchain/xrpl` |
| Version | `0.1.0` |
| Author | Ahmet Osrak (Osraka), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `XRP`, `XRPL`, `XRP Ledger`, `Blockchain`, `Crypto`, `Web3`, `RPC`, `DeFi` |
| Related skills | [`solana`](/docs/user-guide/skills/optional/blockchain/blockchain-solana), [`evm`](/docs/user-guide/skills/optional/blockchain/blockchain-evm) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# XRP Ledger Audit Skill

Use this skill for evidence-based, read-only reviews of XRP Ledger accounts,
trust lines, transactions, fees, reserves, and validated-ledger state. It does
not handle private keys, signing, transaction submission, or wallet recovery.

## When to Use

- The user wants an account balance, reserve, or spendable-XRP estimate.
- The user wants trust-line, issued-currency, issuer, or NoRipple exposure reviewed.
- The user wants recent account activity or a transaction hash explained.
- The user wants current fee, reserve, server, or validated-ledger context.
- The user wants a compact read-only risk summary for an XRPL account.

## Prerequisites

- Use the Hermes `terminal` tool for the JSON-RPC requests below.
- Use a public XRPL JSON-RPC endpoint, or ask whether the user has a trusted node.
- No API key, wallet file, seed phrase, private key, or external package is needed.
- Set `XRPL_RPC_URL` to the endpoint to use. The default is
  `https://s1.ripple.com:51234/`.

Public endpoints may rate-limit requests, log metadata, or lack full history.
For a sensitive investigation, ask the user to confirm the endpoint before
querying an address.

## How to Run

1. Ask for the account address, transaction hash, and the network if they were
   not provided. Do not ask for credentials or wallet files.
2. Validate identifiers locally before putting them in a request. A classic
   address should be an `r`-prefixed XRPL address; a transaction hash should be
   a 64-character hexadecimal string. If validation is uncertain, preserve the
   input as untrusted text and do not interpolate it into shell syntax.
3. Set the endpoint in the `terminal` environment and call `server_info` first.
4. Use `ledger_index: "validated"` for account and trust-line reads unless the
   user explicitly asks about open-ledger state.
5. Follow response markers until the requested data is complete.
6. Report the ledger index, request time, endpoint, and uncertainty alongside
   every conclusion. Never present a heuristic as proof of compromise or theft.

Example request run through `terminal`:

```bash
export XRPL_RPC_URL="${XRPL_RPC_URL:-https://s1.ripple.com:51234/}"
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"server_info","params":[{}]}'
```

Use the terminal output directly when it is already readable. If formatting is
needed, use a local standard-library formatter without sending the response to
another service.

## Quick Reference

| Method | Use | Important request fields |
| --- | --- | --- |
| `server_info` | Server health and validated ledger | none |
| `fee` | Current open-ledger fee | none |
| `account_info` | Balance, sequence, flags, owner count | `account`, `ledger_index` |
| `account_lines` | Trust lines and issued-asset exposure | `account`, `ledger_index`, `limit`, `marker` |
| `account_tx` | Account history | `account`, range, `limit`, `marker` |
| `tx` | One transaction | `transaction`, `binary: false` |

The core request shape is:

```json
{"method":"METHOD","params":[{...}]}
```

Keep request bodies explicit. Do not pass a user-provided address or hash as a
shell fragment; encode it as a JSON value after validation.

## Procedure

### 1. Establish Network Context

Run `server_info` and `fee` before interpreting account data:

```bash
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"server_info","params":[{}]}'

curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"fee","params":[{}]}'
```

Record the validated ledger index, `server_state`, `load_factor`,
`reserve_base_xrp`, `reserve_inc_xrp`, `open_ledger_fee`, and
`median_fee`. If the server is not synchronized or the response is an error,
stop the review or clearly mark all later results as uncertain.

### 2. Inspect Account Balance and Reserve

Use `account_info` with a validated ledger:

```bash
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"account_info","params":[{"account":"rEXAMPLE_REPLACE_WITH_CLASSIC_ADDRESS","ledger_index":"validated","strict":true}]}'
```

Interpret `account_data.Balance` as drops and divide by 1,000,000 for XRP.
`OwnerCount` represents ledger objects that consume owner reserve. A rough
estimate is:

```text
reserve_xrp = reserve_base_xrp + OwnerCount * reserve_inc_xrp
spendable_xrp = balance_xrp - reserve_xrp
```

Use the values from the same server context and ledger. A negative estimate
means reserve pressure or stale reserve inputs, not missing funds by itself.
Explain account flags instead of labeling ordinary issuer or gateway flags as
vulnerabilities.

### 3. Review Trust Lines

Call `account_lines` with a conservative page size:

```bash
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"account_lines","params":[{"account":"rEXAMPLE_REPLACE_WITH_CLASSIC_ADDRESS","ledger_index":"validated","limit":100}]}'
```

For each line, record `currency`, issuer `account`, `balance`, `limit`,
`limit_peer`, `no_ripple`, and `no_ripple_peer`. A negative issued-asset
balance can mean that the queried account owes the counterparty. Open rippling,
unknown issuers, large nonzero balances, and dormant lines are review items,
not automatic findings.

If the response contains `marker`, repeat the same request with that marker
until all relevant pages are reviewed. Do not infer that the first page is the
complete account state.

### 4. Summarize Recent Activity

Use `account_tx` with a small initial limit and validated ledger bounds where
the endpoint supports them:

```bash
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"account_tx","params":[{"account":"rEXAMPLE_REPLACE_WITH_CLASSIC_ADDRESS","ledger_index_min":-1,"ledger_index_max":-1,"limit":10,"binary":false,"forward":false}]}'
```

For each returned transaction, capture its hash, ledger index, result code,
`TransactionType`, initiating account, destination, fee, and relevant amount
fields. Highlight changes to trust lines, offers, escrows, checks, tickets,
signer lists, NFTs, AMM objects, and account settings.

When metadata includes a delivered amount, prefer it over a nominal `Amount`
when explaining what a partial payment actually delivered. Continue through
markers when the user asked for a complete time range.

### 5. Inspect One Transaction

Use `tx` with `binary: false`:

```bash
curl --fail-with-body --silent --show-error "$XRPL_RPC_URL" \
  -H 'Content-Type: application/json' \
  --data '{"method":"tx","params":[{"transaction":"REPLACE_WITH_64_HEX_HASH","binary":false}]}'
```

Explain what was attempted, whether it was validated, the result code, affected
accounts and assets, fee, and meaningful metadata changes. A missing or
unvalidated transaction is not evidence that it never existed; identify the
endpoint history limitations.

## Pitfalls

- Do not treat an open-ledger response as final; prefer validated data.
- Do not compare reserve or fee values from different ledger snapshots without
  saying so.
- Do not treat a trust line, negative IOU balance, high `OwnerCount`, or
  `NoRipple` setting as malicious without account context.
- Do not assume `account_lines` or `account_tx` returned every page.
- Do not treat a public RPC error, rate limit, or missing history as a ledger
  fact.
- Do not request, print, store, sign with, or transmit seeds, private keys,
  mnemonics, wallet files, or signing payloads.
- Do not call `submit`, `sign`, `submit_multisigned`, or any transaction-writing
  method. This skill is observation-only.
- Do not use third-party price data to turn a ledger observation into advice.

## Verification

Before reporting results:

1. Confirm the endpoint, request time, server state, and validated ledger index.
2. Confirm that account, trust-line, and transaction responses did not contain
   an error or an unconsumed marker.
3. Recheck important values against a second trusted endpoint when practical.
4. Separate observed fields, arithmetic estimates, interpretation, and unknowns.

Use this response shape:

```text
XRPL read-only review for <account or transaction>

Network: <endpoint, server state, validated ledger, request time>
Observed: <balance, owner count, trust lines, transaction fields>
Estimate: <reserve/spendable arithmetic, if applicable>
Review notes: <contextual risks and uncertainty>
Next safe check: <read-only follow-up, if needed>
```
