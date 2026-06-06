---
title: "X Radar Builder"
sidebar_label: "X Radar Builder"
description: "Use when creating or tuning X/Twitter radar jobs that monitor scoped topics, dedupe repeats, prefer primary sources, stay silent when there is no signal, and..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# X Radar Builder

Use when creating or tuning X/Twitter radar jobs that monitor scoped topics, dedupe repeats, prefer primary sources, stay silent when there is no signal, and report concise reliability-labeled updates.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/research/x-radar-builder` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | macos, linux |
| Tags | `x`, `twitter`, `radar`, `monitoring`, `cron`, `news` |
| Related skills | [`blogwatcher`](/docs/user-guide/skills/bundled/research/research-blogwatcher), [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# X Radar Builder

## Overview

Build scoped radars for X/Twitter and official sources. A good radar is quiet most of the time. It reports only new, material signal with direct links and reliability labels.

## When to Use

- User asks for Tesla/UAP/AI/model/news radar.
- Existing radar is noisy or misses important launches.
- Need a cron prompt for topic monitoring.
- Need accounts, official sources, dedupe, or quiet hours.

## Workflow

1. Define scope: topics, priority accounts, official source fallbacks, time window, quiet hours.
2. Write retrieval strategy: X search for X content; web/official pages for major model launches and primary confirmation.
3. Apply signal filter: new link or material state change; primary source preferred; rumors labeled; silent if no real signal.
4. Deduplicate: avoid same URL/announcement; use watermarks in scripts; in LLM cron prompts, explicitly forbid stale repeats.
5. Format for Telegram: Spanish for Jaime, max 3 items per topic, reliability label per item.

## Reliability Labels

- Alta: official account, company/government source, primary document.
- Media: credible journalist/analyst/specialist.
- Baja: secondary account or weak provenance.
- Rumor: unconfirmed claim; must be interesting and labeled.

## Cron Prompt Skeleton

```text
Radar horario para <user> sobre <topics>. Usa X search for X, web only for official verification. Window: last hour. If no new material signal, respond exactly [SILENT]. Each item needs direct link and reliability label. Do not repeat stale links.
```

## User Simulation Tests

- No signal → output exactly `[SILENT]`.
- Official Anthropic launch today but X misses it → check official news.
- Same Tesla post seen twice → suppress repeat.
- UAP claim from random account → label Baja/Rumor or omit.
- User adds priority account → update query set and prompt.

## Common Pitfalls

1. Reporting keyword matches.
2. Missing major official launches.
3. Repeating yesterday's news.
4. Noisy "nothing happened" reports.

## Verification Checklist

- [ ] Scope/accounts explicit.
- [ ] Quiet hours considered.
- [ ] Official fallbacks included.
- [ ] Dedupe rule present.
- [ ] `[SILENT]` behavior exact.
- [ ] Reliability labels defined.
