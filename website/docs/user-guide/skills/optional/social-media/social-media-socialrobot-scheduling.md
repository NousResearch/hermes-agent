---
title: "Socialrobot Scheduling — Schedule multi-platform social posts via SocialRobot MCP"
sidebar_label: "Socialrobot Scheduling"
description: "Schedule multi-platform social posts via SocialRobot MCP"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Socialrobot Scheduling

Schedule multi-platform social posts via SocialRobot MCP.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/socialrobot-scheduling` |
| Path | `optional-skills/social-media/socialrobot-scheduling` |
| Version | `0.1.0` |
| Author | Nicolas Torres (ntgussoni), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Social-Media`, `Scheduling`, `Publishing`, `MCP` |
| Related skills | [`social-media-content-calendar`](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar), [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SocialRobot Scheduling Skill

Schedule approved posts to connected social accounts through the SocialRobot
MCP server. Read the server's `create-post` skill with MCP `skills/get` first;
it has the per-platform steps and the SocialRobot team keeps it current. This
skill covers what that one does not: approval gating, timing, verification, and
honest status reporting. It does not write copy; pair it with
`social-media-content-calendar` for a brief-to-published flow.

## When to Use

- "Schedule this post for Tuesday 9am."
- "Publish the approved draft to LinkedIn, X, and Instagram."
- "Move my Monday post to Wednesday, or cancel it."
- "Turn this campaign calendar into scheduled posts."

Don't use for: writing the copy itself (pair with `social-media-content-calendar`
or `humanizer`), or one-off posting without a SocialRobot account (use the
platform skill directly, e.g. `xurl`).

## Prerequisites

- The `socialrobot` MCP server is installed and connected: `hermes mcp install
  socialrobot` (or add `mcp_servers.socialrobot.url: https://socialrobot.io/api/mcp`
  to `~/.hermes/config.yaml`). OAuth signs you in through a browser; an API key
  (`x-api-key` header) works for headless setups. No account yet?
  [Get a free account](https://socialrobot.io).
- The user has linked the social accounts they want to post to inside the
  SocialRobot app first. The MCP server cannot invent platform logins; it only
  schedules to accounts already connected there.
- Scopes granted include `publishing:write` (and `media:write` when the post
  needs media).

## How to Run

1. Confirm the server is connected: `tools/list` returns the SocialRobot tools.
2. Load the server's current workflow: MCP `skills/get` on `create-post` (or
   `resources/read` on `skill://create-post/SKILL.md`). It documents the
   account, media, and per-platform steps, and the SocialRobot team keeps it
   current with the product.
3. Execute the pipeline below, following the server skill for mechanics.

## Quick Reference

`tools/list` and `skills/get` are the source of truth for tool signatures and
workflow details. The orchestration-relevant tools:

| Tool | Role in this skill |
|------|--------------------|
| `list_connected_accounts` | Entry point: discover account IDs; never guess them. |
| `create_post` / `list_posts` | Create, then verify with read-back. |
| `update_post`, `reschedule_post`, `delete_post` | Post-run management. |

## Procedure

1. **Confirm approval state.** Only posts that are approved (never `draft` or
   `needs review`) may be scheduled. If drafts are unapproved, hand them back.
2. **Discover accounts.** `list_connected_accounts`, pick the account ID per
   target platform.
3. **Pick timing.** If the user left timing open or wants the best time, call
   `instagram_best_post_times` for Instagram accounts and use
   `get_account_analytics` trends for other platforms, then set `scheduledFor`
   accordingly. Confirm the user's timezone before choosing datetimes.
4. **Follow the server's `create-post` skill for mechanics**: media upload
   before `create_post` (keep the returned media reference), Pinterest board
   resolution, TikTok `privacyLevel` and `postMode` (`DIRECT_POST` publishes,
   `UPLOAD` is an inbox draft), LinkedIn geo and @mention URN resolution.
5. **Create.** One `create_post` call with multiple targets when the copy is
   shared, rather than separate posts.
6. **Verify with read-back.** `list_posts`: confirm scheduled time, account,
   content preview, and provider post/job ID. Report `scheduled` or `published`
   only per the status the tool returns.
7. **Manage.** `update_post`, `reschedule_post`, or `delete_post` when the user
   changes their mind; re-verify after any change.

## Pitfalls

- Never invent account IDs, board IDs, or TikTok `privacyLevel` values; always
  read them from the tools or the server-served skill.
- Upload media before `create_post`; keep the returned media reference.
- `UPLOAD` mode on TikTok means an inbox draft, not a published post. Say so.
- A queued post is not published. Do not claim publication for posts that only
  show as scheduled or pending.
- API key quota is one shared pool between the HTTP OpenAPI endpoints and MCP
  `tools/call` + `tools/list` for that key. Handshake traffic (login, OAuth,
  `initialize`, `get-session`) does not consume quota.
- If `skills/get` fails, the connection may be stale; reconnect the MCP server
  before proceeding.

## Verification

`list_posts` shows the post with its scheduled time, account, and status, and
any provider post/job ID. Every scheduled slot must trace back to an approved
draft, and every `scheduled`/`published` claim must match the status field the
tool returned.
