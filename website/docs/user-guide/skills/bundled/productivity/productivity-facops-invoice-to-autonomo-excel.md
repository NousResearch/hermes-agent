---
title: "Facops Invoice To Autonomo Excel"
sidebar_label: "Facops Invoice To Autonomo Excel"
description: "Use when invoices, receipts, school payments, or billing emails need extraction and handoff to Jaime's autónomo Excel/accounting workflow via the FacOps prof..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Facops Invoice To Autonomo Excel

Use when invoices, receipts, school payments, or billing emails need extraction and handoff to Jaime's autónomo Excel/accounting workflow via the FacOps profile without mixing personal Gmail state.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/facops-invoice-to-autonomo-excel` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos, linux |
| Tags | `facops`, `invoices`, `receipts`, `autonomo`, `excel`, `gmail` |
| Related skills | [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`gmail-triage-jaime`](/docs/user-guide/skills/bundled/productivity/productivity-gmail-triage-jaime) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# FacOps Invoice to Autónomo Excel

## Overview

For recurring invoice/receipt capture: Anthropic receipts, SaaS bills, school payments, and mail Jaime marks "para Excel de autónomo". Preserve attachments, extract accounting fields, and keep FacOps isolated.

## When to Use

- User says "para Excel de autónomo".
- FacOps watchdog reports a receipt/invoice task.
- Subject contains invoice, receipt, factura, recibo, statement, payment, VAT.
- Attachments include invoice PDFs or receipts.

## Scope Boundaries

Use the FacOps profile/account for invoice tracking. Do not mutate Jaime's personal Gmail labels unless explicitly asked. Do not infer tax category beyond obvious vendor/category; flag uncertainty.

## Workflow

1. Record task identity: FacOps task id, sender, subject, date, amount, currency, attachment names.
2. Extract vendor, invoice/receipt number, issue/paid date, gross total, currency, VAT/tax amount if present, payment method, category suggestion.
3. Preserve evidence: save PDF attachments under FacOps storage; keep original filenames when useful.
4. Handoff to Excel/accounting: stage/append row with attachment path and `needs-review` when uncertain.
5. Acknowledge concisely: "Hecho: marcado para Excel de autónomo" plus task id/status.

## Output Row Shape

`date`, `vendor`, `description`, `invoice_number`, `amount_gross`, `currency`, `vat_amount`, `category`, `source_email`, `attachment_paths`, `review_status`.

## User Simulation Tests

- Anthropic receipt with two PDFs → extract amount/date/vendor and attach both.
- User replies "para Excel" to FacOps task → update FacOps, not personal Gmail.
- Missing VAT → row staged with `needs-review`.
- Duplicate receipt id → no duplicate workbook row.
- Attachment download fails → report blocker and leave task open.

## Common Pitfalls

1. Mixing profiles.
2. Dropping receipt evidence.
3. Guessing VAT.
4. Duplicating rows.

## Verification Checklist

- [ ] Task id/email id recorded.
- [ ] Attachments saved and paths exist.
- [ ] Accounting fields extracted or marked uncertain.
- [ ] Duplicate check performed.
- [ ] FacOps state updated, personal Gmail untouched unless requested.
