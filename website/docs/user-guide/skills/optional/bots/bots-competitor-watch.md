---
title: "Competitor Watch — Baseline public competitors and report material changes"
sidebar_label: "Competitor Watch"
description: "Baseline public competitors and report material changes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Competitor Watch

Baseline public competitors and report material changes.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/bots/competitor-watch` |
| Path | `optional-skills/bots/competitor-watch` |
| Version | `0.1.0` |
| Author | Mark (unsupportedpastels), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `competitors`, `monitoring`, `baselines`, `citations` |
| Related skills | [`competitor-news-monitor`](../../bundled/research/research-competitor-news-monitor.md), [`grounded-citations`](../../bundled/research/research-grounded-citations.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Competitor Watch Workflow Skill

Create a source-backed public baseline or compare a later capture against it. Offline fixtures teach the comparison format only and never represent current competitor facts.

## When to Use

- The user supplies company names, public URLs, or saved page captures.
- The user wants pricing, product, positioning, launch, or hiring changes.
- Do not use behind logins, to bypass access controls, or to infer missing history.

## Prerequisites

The first useful task works from two pasted or local captures via `read_file`. Current public-page checks require configured `web_search` and `web_extract`, with `browser_navigate` only when needed. `competitor-news-monitor` provides broader monitoring discipline when installed; it does not create network access.

## How to Run

1. Copy `templates/watchlist.csv` into the owning profile workspace without overwriting an existing tracker.
2. Establish a baseline from user data or try `references/sample-input.md` in fixture-only mode.
3. Grade the brief using `references/expected-output-rubric.md`.

## Quick Reference

| State | Report wording |
|---|---|
| First observation | Baseline, not a change |
| Comparable difference | Observed change with before/after evidence |
| Page unavailable | Unknown coverage |
| Ambiguous page redesign | Possible change; manual review needed |

A recurring watch can be recommended after a stable baseline exists. Do not automatically create a scheduled job; explicit activation must approve URLs, cadence, categories, and delivery behavior.

## Procedure

### 1. Define the watch contract

Record named companies/URLs, material-change categories, geography/currency, cadence, and exclusions. **Done when** every target has an approved public source and comparison criterion.

### 2. Capture a dated baseline

Use supplied captures or retrieve public pages with `web_extract`; preserve exact URLs and observation dates. A first observation is a baseline, never retroactively described as a change. **Done when** each target is baseline, unavailable, or excluded.

### 3. Normalize comparable facts

Extract only observable prices, plan labels, claims, release dates, job counts, or other approved fields. Preserve units and source wording. **Done when** comparison fields have source locations and missing values remain missing.

### 4. Compare and assess materiality

Separate meaningful differences from formatting noise. Lead with what changed and why it may affect a decision; label interpretation as analysis. **Done when** every reported change has before, after, date, and URL evidence.

### 5. Update safely

Write a new watchlist or baseline file in the owning profile workspace, never over the supplied capture. Apply `references/expected-output-rubric.md`. **Done when** unknown coverage is visible and no failed fetch is reported as no change.

## Pitfalls

- Fixture-only sample evidence is not current and must not be presented as a live finding.
- Dynamic pages can change markup without changing the offer.
- A missing page means unknown coverage, not no news.
- Do not publish counter-positioning or schedule monitoring without approval.

## Verification

- [ ] Every current claim has a source URL and observation date.
- [ ] Baselines and changes use distinct language.
- [ ] Unknown and incomparable coverage is explicit.
- [ ] Outputs preserve inputs and avoid filename collisions.
