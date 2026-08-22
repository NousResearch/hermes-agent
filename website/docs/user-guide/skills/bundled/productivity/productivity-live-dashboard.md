---
title: "Live Dashboard — Build self-updating dashboards from live sources"
sidebar_label: "Live Dashboard"
description: "Build self-updating dashboards from live sources"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Live Dashboard

Build self-updating dashboards from live sources.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/live-dashboard` |
| Version | `0.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Dashboards`, `Monitoring`, `Status`, `Automation`, `Reporting` |
| Related skills | [`product-price-monitor`](/docs/user-guide/skills/bundled/productivity/productivity-product-price-monitor), [`competitor-news-monitor`](/docs/user-guide/skills/bundled/research/research-competitor-news-monitor), [`email-inbox-triage`](/docs/user-guide/skills/bundled/email/email-email-inbox-triage), [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Live Dashboard

Turn one sentence — "make a dashboard for our visa applications, update it daily from the email threads and the case-status site" — into a persistent, self-refreshing status page. The user describes what they want to see; you define the data contract, build a self-contained HTML dashboard, verify one live refresh, then schedule the recurring tick. Inspired by Energy's (getenergy.com) natural-language live dashboards, adapted to Hermes's cron + connector architecture.

Setup runs once in the foreground; the recurring refresh runs as a `cronjob` tick (the `live-dashboard` automation blueprint scaffolds this).

## When to Use

- "Make a dashboard for &lt;project/process> and keep it updated."
- "I want one place to see the status of &lt;deals / applications / bugs / shipments>."
- "Track &lt;thing> across my email and &lt;website> and show me where it stands."
- A cron tick fires for an existing dashboard (steps 5-7).

Don't use for: one-off status questions (answer directly), price/availability thresholds on a single item (use `product-price-monitor`), or company news tracking (use `competitor-news-monitor`).

## Procedure — Setup (foreground, once)

### 1. Define the dashboard contract

From the user's sentence, pin down: the dashboard's purpose in one line, the entities being tracked (rows), the fields per entity (columns/indicators), what "needs attention" means, the sources each field is read from, and the refresh cadence. Ask about anything ambiguous — a dashboard that tracks the wrong grain is worthless. Done when every field on the dashboard names the source it will be read from.

### 2. Verify each source with one live read

For each source, do one bounded foreground read now: email/calendar via the connector skills (`himalaya`, `google-workspace`), websites via `web_extract` or `browser_navigate`, local files via `read_file`. Record what was actually retrievable — auth walls, missing permissions, or empty results surface here, not on the first scheduled run. Drop or replace sources that fail. Done when every field's source returned real data or was explicitly renegotiated with the user.

### 3. Build the dashboard artifact

Write two files under `~/.hermes/dashboards/<slug>/`:

- `dashboard.json` — the contract plus current state: purpose, entities, per-field values, per-field source + retrieval timestamp, a `needs_attention` list, and a change log (append-only, most recent first).
- `index.html` — a single self-contained HTML page (inline CSS, no external requests) rendering the state: a header with purpose and last-updated time, a "Needs attention" section on top, the entity table, and the recent-changes list. Regenerate it from `dashboard.json` on every refresh; never hand-edit HTML state.

Populate both from the step-2 reads and tell the user the file path (and open it where the platform allows). Done when the page renders the live data and every value on it carries a retrieval timestamp in `dashboard.json`.

### 4. Schedule the refresh

Only after step 3 succeeded, create the job:

```
cronjob(action="create",
        schedule=<cadence from the contract, e.g. "0 8 * * *">,
        prompt="Load the live-dashboard skill and run the refresh tick for the dashboard at ~/.hermes/dashboards/<slug>/dashboard.json.",
        deliver=<user's destination>)
```

Pick a cadence that respects source rate limits. Done when the job exists and its prompt names the state-file path.

## Procedure — Tick (each scheduled run)

### 5. Re-read sources and diff

Load `dashboard.json`, re-read each field from its named source, and compute a field-level diff against the stored state. A failed source read means unknown state: keep the last good value, mark the field stale with the failure time, and never overwrite good data with an error. Done when every field is either updated, unchanged, or explicitly marked stale.

### 6. Update state and re-render

Apply the diff to `dashboard.json`: update values and timestamps, append material changes to the change log, and recompute `needs_attention` against the contract's attention rules. Regenerate `index.html` from the updated state. Done when the JSON and HTML agree and the change log entry for this run exists (or the run is recorded as no-change).

### 7. Deliver on material change, else stay silent

If the diff contains material changes or new needs-attention items, deliver a short summary: what changed, what needs attention, and the dashboard path. Otherwise respond with `[SILENT]` — no "still watching" noise unless the user asked for a periodic digest. Done when delivery matches the diff.

## Pitfalls

- Building the page before verifying the sources — auth failures then surface on an unattended run.
- Overwriting last-known-good values with an error page or empty read.
- Rendering state into HTML only — `dashboard.json` is the source of truth; HTML is a projection.
- Alerting on every refresh instead of on material change.
- Tracking the wrong grain (per-thread when the user thinks per-application).

## Verification

- [ ] Every dashboard field names its source, and each source passed one foreground read before scheduling.
- [ ] `dashboard.json` and `index.html` exist and agree; every value carries a retrieval timestamp.
- [ ] Failed reads marked fields stale without destroying last-known-good state.
- [ ] Ticks deliver only on material change; no-change runs were `[SILENT]`.
- [ ] The change log replays the dashboard's history from the state file alone.
