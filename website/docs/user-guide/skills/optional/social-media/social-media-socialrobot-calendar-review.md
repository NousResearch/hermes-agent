---
title: "Socialrobot Calendar Review — Audit the upcoming social queue and fix gaps or overlaps"
sidebar_label: "Socialrobot Calendar Review"
description: "Audit the upcoming social queue and fix gaps or overlaps"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Socialrobot Calendar Review

Audit the upcoming social queue and fix gaps or overlaps.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/socialrobot-calendar-review` |
| Path | `optional-skills/social-media/socialrobot-calendar-review` |
| Version | `0.1.0` |
| Author | Nicolas Torres (ntgussoni), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Social-Media`, `Calendar`, `Queue`, `MCP` |
| Related skills | [`socialrobot-scheduling`](/docs/user-guide/skills/optional/social-media/social-media-socialrobot-scheduling), [`social-media-content-calendar`](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SocialRobot Calendar Review Skill

Audits the user's upcoming social media queue through the SocialRobot MCP
server: what is scheduled, when, on which accounts, and whether the calendar is
healthy (cadence, duplicates, gaps, stale drafts). It proposes concrete fixes
and applies only the changes the user approves. It does not invent content;
filling a gap means drafting a brief for the user or handing off to
`socialrobot-scheduling` / `social-media-content-calendar`.

## When to Use

- "What's in my queue this week?"
- "Why do I have two posts on the same day for the same account?"
- "I have a hole in my calendar next week, what should I do?"
- "Move everything scheduled for Thursday."
- "Clean up my drafts."

Don't use for: creating new content (that is
`social-media-content-calendar` + `socialrobot-scheduling`), or measuring past
performance (that is `socialrobot-analytics` / `socialrobot-campaign-report`).

## Prerequisites

- The `socialrobot` MCP server is installed and connected: `hermes mcp install
  socialrobot`. OAuth signs you in through a browser; an API key
  (`x-api-key` header) works for headless setups. No account yet?
  [Get a free account](https://socialrobot.io).
- Scopes granted include `publishing:read` (audit) and `publishing:write`
  (applying changes).

## How to Run

1. Pull the queue with `list_posts` filtered to `SCHEDULED` (and `DRAFT` when
   asked) for the window in question.
2. Audit against the cadence and mix the user wants.
3. Present findings with concrete proposals.
4. Apply only approved changes, then re-verify.

## Quick Reference

| Tool | Role |
|------|------|
| `list_posts` | Queue source: filter by `status`, `platform`, `dateFrom`, `dateTo`, paginate with `limit`/`cursor`. |
| `reschedule_post` | Move a post to a new future datetime. |
| `update_post` | Change caption, media, or targets on a draft/scheduled post. |
| `delete_post` | Remove a draft or scheduled post. Irreversible; confirm first. |
| `create_post` | Fill an approved gap (or hand the gap to scheduling skills). |

## Procedure

1. **Pull the queue.** `list_posts` with `status: SCHEDULED` and the window
   (`dateFrom`/`dateTo`). Paginate with `cursor` until the window is covered.
   Include `DRAFT` when the user asks for a cleanup.
2. **Audit.**
   - Cadence: count posts per platform per day/week against the user's target.
   - Overlaps: two or more posts to the same account within a short window.
   - Gaps: days or platforms with no coverage in the window.
   - Stale items: drafts past their relevance date, or posts with expired
     claims or dead links.
   - Mix: same content type repeated back to back (for example, three text-only
     posts in a row on one account).
3. **Propose.** Present findings as a short list: each issue, the item (post
   id, platform, datetime), and the proposed fix (reschedule to X, update copy,
   delete, or draft a new post). Get approval before executing anything.
4. **Apply approved changes.** `reschedule_post` and `update_post` for edits,
   `delete_post` only for explicitly approved deletions. For approved gaps,
   hand the brief to `socialrobot-scheduling` rather than improvising content.
5. **Re-verify.** `list_posts` over the same window: no duplicates remain,
   cadence matches the target, and every slot has a purpose.

## Pitfalls

- Deleting without explicit approval; `delete_post` is irreversible.
- Confusing timezones. Confirm the user's timezone; `scheduledFor` datetimes
  are absolute ISO timestamps.
- Over-posting one platform in a single day; platforms throttle or bury
  repeated posts.
- Treating drafts as approved content. Drafts go back to the editorial step,
  not straight to scheduling.
- Forgetting pagination: `limit` caps at 100, so large queues need `cursor`
  continuation.

## Verification

A fresh `list_posts` over the audited window shows the approved changes:
overlaps resolved, gaps filled or explicitly reported, and nothing deleted or
moved that the user did not approve.
