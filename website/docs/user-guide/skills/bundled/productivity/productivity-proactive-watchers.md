---
title: "Proactive Watchers — Anticipate needs: trip moments, owed replies, watched pages"
sidebar_label: "Proactive Watchers"
description: "Anticipate needs: trip moments, owed replies, watched pages"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Proactive Watchers

Anticipate needs: trip moments, owed replies, watched pages.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/productivity/proactive-watchers` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Proactive`, `Cron`, `Reminders`, `Travel`, `Inbox`, `Watchers` |
| Related skills | [`product-price-monitor`](../../bundled/productivity/productivity-product-price-monitor.md), [`maps`](../../bundled/productivity/productivity-maps.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Proactive Watchers Skill

Turn what the user tells you in passing into scheduled help that fires at the right moment, without
being asked: a check-in reminder when the airline window opens, a leave-by time with the metro line,
a draft reply when a message has waited too long. Every job is a `cronjob_manage` create the user
approves on the card; the gate scripts below keep idle ticks free (no model call until something is
due). This skill plans the jobs; it never sends, books or submits anything on its own.

## When to Use

- The user mentions a trip, flight, appointment, event, deadline or document (visa, arrival form).
- The user says they owe someone a reply, or that they keep missing messages.
- Something should be re-checked until it changes (a page, a status, an order, a listing).
- A cron tick of a job this skill created wakes you with a `context` payload.

## Prerequisites

- `cronjob_manage` (the `cronjob` toolset). `script_content` saves a gate script in one call.
- For messages and calendars: a connected app (`manage_connections`); gate scripts read it with the
  read-only `hermes connectors call <app>.<tool> --args '<json>'` CLI. Without one, schedule plain
  time-based reminders instead.
- Templates in `templates/`: `moments_gate.py` (time-anchored plans), `unanswered_gate.py` (owed replies).

## Quick Reference

| Situation | Job shape | Gate |
|---|---|---|
| One fixed moment (dentist tomorrow 3pm) | one-shot `schedule` at the reminder time | none |
| A plan with several moments (trip) | `every 15m`, `repeat` covering the trip, `moments_gate.py` | wakes only at each moment |
| Owed replies (Slack, mail) | `every 15m` on work hours, `unanswered_gate.py` | wakes only on new waiting messages |
| A page or status that may change | `monitor` = the URL | wakes only on a changed page |
| A fixed ping with no reasoning | `no_agent=true` + script | the script's stdout is the message |

Gate contract: the script's last stdout line is `{"wakeAgent": false}` (skip the run, zero tokens) or
`{"wakeAgent": true, "context": {...}}` (you wake with only that context). State persists in
`$HERMES_CRON_STATE_DIR`. A gate that fails sends the user a watchdog alert instead of waking you.

## Procedure

### 1. Spot the moments

From the conversation, list each moment the user would want help at, with its exact time and zone,
and what help looks like then. Trip example (flight NH7 SFO to Tokyo, Fri Nov 27 08:00 PST):

- T-24h: online check-in opens: remind with the check-in link and seat/bag notes.
- T-3h: leave-by time from where they are, with the route (driving time or the metro line and
  transfer), and flight status.
- Before departure: arrival paperwork the destination needs (Japan: Visit Japan Web arrival and
  customs QR; US: ESTA/I-94 checks): what to fill in, and an offer to prepare the answers.
- After landing (destination zone): airport-to-hotel route, transit card tip, hotel check-in time.
- Return leg: the same moments for the flight home.

Use the zone of the place the moment happens in (check-in follows the airline's departure city;
landing help follows the destination). Write `at` with a UTC offset.

### 2. Pick the cheapest job that works

- A single moment: a one-shot job at that time. No gate.
- Two or more moments in one plan: ONE job with `moments_gate.py` as `script_content`, `MOMENTS`
  filled in, schedule `every 15m`, `repeat` = number of 15-minute ticks until the last moment plus
  its window. One approval card covers the whole plan.
- Watching for something to happen (a reply, a change): a gated job; never a plain recurring prompt
  that asks the model every tick whether anything happened.

### 3. Write the prompt for the moment, not the plan

The prompt runs later with no chat context. Say what to do with the `context` payload: "For each due
moment, give the user a 3-line heads-up: what it is, the exact action (link, route, line), and one
thing to double-check. If a route or time depends on live data, look it up first." For owed replies:
"For each waiting message, say who is waiting and since when, and draft a short reply in the user's
voice. Do not send anything."

### 4. Create it and say so

Call `cronjob_manage(action="create", ...)` with `script_content`, `schedule`, `repeat`, `name`,
`prompt` and `deliver` left to the default (the chat you are in). Tell the user in one line what you
set up and that the card is theirs to approve. If they deny it, do not retry and do not offer again
in this conversation.

## Pitfalls

- Times without a zone. A trip crosses zones: every `at` needs an offset.
- One job per moment. Five cards for one trip is approval fatigue; use one moments gate.
- Sending on the user's behalf. Drafts only; a reply, booking or form submission waits for the user.
- Endless watchers. Give every watcher a `repeat` cap or an end moment; trips end.
- Re-asking after a "no". A denied card means the user does not want that job.
- Gate scripts that call the model or print chatty text. The last line must be the gate JSON.

## Verification

- [ ] Each moment has an absolute time with an offset and a concrete action.
- [ ] One job per plan or watch, with a `repeat` cap or end.
- [ ] The gate prints `{"wakeAgent": false}` when nothing is due (run it once by hand if unsure).
- [ ] The prompt is self-contained and never sends, books or submits.
- [ ] The user was told in one line what was scheduled.
