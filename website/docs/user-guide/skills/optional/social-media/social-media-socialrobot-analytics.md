---
title: "Socialrobot Analytics — Report SocialRobot post, account, and audience analytics"
sidebar_label: "Socialrobot Analytics"
description: "Report SocialRobot post, account, and audience analytics"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Socialrobot Analytics

Report SocialRobot post, account, and audience analytics.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/socialrobot-analytics` |
| Path | `optional-skills/social-media/socialrobot-analytics` |
| Version | `0.1.0` |
| Author | Nicolas Torres (ntgussoni), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Social-Media`, `Analytics`, `Audience`, `MCP` |
| Related skills | [`socialrobot-scheduling`](/docs/user-guide/skills/optional/social-media/social-media-socialrobot-scheduling), [`social-media-content-calendar`](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SocialRobot Analytics Skill

Answers performance questions about the user's connected social accounts using
the analytics tools on the SocialRobot MCP server. It produces reports on post
and account performance, follower demographics, and analytics-derived best
times to post. It reports the numbers the tools return; it does not invent
metrics, extrapolate missing platforms, or turn data into marketing claims.

## When to Use

- "Which of my Instagram posts did best last month?"
- "What are the best times to post for my audience?"
- "How is my LinkedIn Company Page audience split by industry?"
- "Summarize last week's performance across accounts."

Don't use for: scheduling or creating posts (use `socialrobot-scheduling`), or
platforms the user has not connected (the tools only cover linked accounts).

## Prerequisites

- The `socialrobot` MCP server is installed and connected: `hermes mcp install
  socialrobot`. OAuth signs you in through a browser; an API key
  (`x-api-key` header) works for headless setups. No account yet?
  [Get a free account](https://socialrobot.io).
- Scopes granted include `analytics:read` and `audience:read`.

## How to Run

Pick the tool that matches the question, call it, and report what it returns
with the time range and account it covers. Tool signatures and fields live in
`tools/list`; this skill covers choosing the right tool and reporting the
results faithfully. All analytics tools require the account ID from
`list_connected_accounts` first.

## Quick Reference

`tools/list` is the source of truth for signatures and fields. The mapping from
question to tool:

| Tool | Question it answers |
|------|---------------------|
| `get_account_analytics` | How did this account perform over a period? |
| `get_post_analytics` | How did one post perform? |
| `get_posts_with_analytics` | Which recent posts did well, in one call? |
| `get_follower_demographics` | Who follows this account? |
| `instagram_best_post_times` | When does this account's audience engage? |

## Procedure

1. **Discover accounts.** Call `list_connected_accounts` and use the returned
   account IDs. Do not guess.
2. **Scope the question.** Account-level: `get_account_analytics`. Single post:
   `get_post_analytics`. Recent batch: `get_posts_with_analytics`. Audience:
   `get_follower_demographics`. Best times: `instagram_best_post_times`.
3. **Call the tool** with the account ID and, where supported, the time range
   the user asked about. Keep the returned period explicit in your report.
4. **Report faithfully.** Present the numbers the tool returned, the period,
   and the account. Note when a metric is unavailable or zero; do not fill gaps
   with estimates.
5. **Interpret, do not inflate.** Frame best times as analytics-derived windows
   for this account, not universal guarantees. Tie post-level numbers to the
   content and date when useful.

## Pitfalls

- Never invent or round up metrics the tools did not return. If a field is
  missing, say it is missing.
- Coverage varies by platform: LinkedIn Company Pages expose demographics and
  page analytics; TikTok account analytics plus `get_posts_with_analytics` cover
  public videos (requires `video.list`). State the coverage for each account.
- `instagram_best_post_times` reflects analytics for the connected account;
  it is not a claim about other accounts or platforms.
- Always attach the time range to the numbers you report. Bare numbers without
  a period are unverifiable.
- Do not use the analytics output to make forward-looking performance claims in
  marketing copy; that is a separate editorial decision for the user.

## Verification

The response includes the account ID, the time range queried, and non-empty
metric fields for at least the primary metric asked about. If the user asked
for a comparison, the numbers must come from the same tool and same period so
they are comparable.
