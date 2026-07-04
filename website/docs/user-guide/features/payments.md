---
sidebar_position: 16
title: "Payments Review"
description: "Capture invoice and payment-request details from Gmail and other sources for manual payment in your banking app."
---

# Payments Review

The **Payments** page in the web dashboard is a review queue for invoices and payment requests. Hermes does **not** send money or trigger bank transfers. Instead, it captures payment details, highlights missing fields, and gives you structured values to copy into your banking app manually.

## What it does

- **Captures payment requests** into a profile-scoped queue
- **Syncs Gmail** for likely invoice / payment-due messages
- **Stores extracted fields** such as amount, due date, invoice number, payment reference, and bank details
- **Flags uncertainty** when important details are missing or low-confidence
- **Tracks manual review state** so you can move items from new → ready to pay → paid

## Where it lives

Open the [Web Dashboard](./web-dashboard.md) and select **Payments** in the sidebar.

The queue is profile-scoped, like the rest of the management dashboard. If you switch profiles, you are looking at that profile's payment-request store.

## Current sources

### Gmail

The first real ingestion path is **Gmail**.

When Google Workspace is configured, the page can run **Sync Gmail**, which:

1. Searches for likely invoices and payment requests
2. Reads candidate messages through the existing Google Workspace bridge
3. Extracts fields such as:
   - payee
   - amount and currency
   - due date
   - invoice number
   - payment reference
   - sort code / account number
   - IBAN / SWIFT / routing number when present
4. Adds or updates queue entries in the Payments store

Set up Gmail/Google Workspace first:

- [Google Workspace skill](../skills/bundled/productivity/productivity-google-workspace.md)

### Email

The page also detects whether the built-in [Email channel](../messaging/email.md) is configured. That source is currently surfaced as a connection status and review target; the real extraction path is still Gmail-first.

### Uploads

Uploads are supported as a staging concept today. The page points you to the [Files](./web-dashboard.md#files) workflow so invoice PDFs or screenshots can be placed under your Hermes home for later extraction/review flows.

### Slack

Slack appears as a source-status card so the eventual cross-channel review model is visible in one place, but Gmail is the only fully wired sync source right now.

## Page layout

The Payments page is split into three areas:

### Queue

The left column shows captured payment requests with:

- vendor / sender
- title or subject
- amount
- due date
- source
- confidence
- review status

Filters let you narrow by source, status, or free-text search.

### Invoice / Request detail

The middle panel shows:

- sender / source metadata
- received time
- amount and due date
- extracted preview text
- attachment list
- warnings about missing or uncertain fields
- a link back to the original Gmail message when available

### Payment details

The right panel gives you copy-ready fields for your banking app:

- payee
- account holder
- account number
- sort code
- IBAN
- SWIFT
- routing number
- amount
- payment reference
- invoice number
- due date
- billing address
- tax details

## Manual review states

Each payment request can be moved between these states:

- `new`
- `needs_review`
- `ready_to_pay`
- `paid`
- `ignored`

These are review-only states. Marking an item `paid` does not send anything externally.

## Storage

The queue is stored under your profile's Hermes home:

```text
~/.hermes/payments/requests.json
```

If you're using multiple profiles, each profile has its own `payments/requests.json`.

## Safety model

The Payments workflow is intentionally conservative:

- Hermes does **not** initiate transfers
- Missing details remain visible as missing
- Low-confidence extraction is surfaced as warnings instead of hidden
- Gmail sync updates existing entries rather than silently duplicating every read

## Current limitations

As of July 4, 2026:

- **Gmail sync is the only fully wired ingestion path**
- extraction is heuristic, not OCR- or attachment-parser-driven yet
- uploads and other channels are surfaced in the UI, but not fully ingested into the queue
- validation against a user's real invoice examples still depends on the connected mailbox and real message content

## Related

- [Web Dashboard](./web-dashboard.md)
- [Google Workspace](../skills/bundled/productivity/productivity-google-workspace.md)
- [Files](./web-dashboard.md#files)
