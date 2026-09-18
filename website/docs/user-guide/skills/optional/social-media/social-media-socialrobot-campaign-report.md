---
title: "Socialrobot Campaign Report — Compare campaign performance across platforms and recap"
sidebar_label: "Socialrobot Campaign Report"
description: "Compare campaign performance across platforms and recap"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Socialrobot Campaign Report

Compare campaign performance across platforms and recap.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/socialrobot-campaign-report` |
| Path | `optional-skills/social-media/socialrobot-campaign-report` |
| Version | `0.1.0` |
| Author | Nicolas Torres (ntgussoni), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Social-Media`, `Analytics`, `Campaign`, `MCP` |
| Related skills | [`socialrobot-analytics`](/docs/user-guide/skills/optional/social-media/social-media-socialrobot-analytics), [`social-media-content-calendar`](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SocialRobot Campaign Report Skill

Builds a cross-platform performance report for a campaign or time window using
the SocialRobot MCP analytics tools, and writes a structured recap: what was
posted, how each platform performed, what stood out, and what to do next. It
compares platforms honestly, using platform-native metrics, and never invents
numbers or extrapolates beyond the window the tools returned.

## When to Use

- "How did the launch do across platforms?"
- "Compare last month on Instagram vs LinkedIn vs X."
- "Recap our Q3 social performance."
- "Which platform should we double down on next quarter?"

Don't use for: single-account metric questions (use `socialrobot-analytics`),
or scheduling (use `socialrobot-scheduling`).

## Prerequisites

- The `socialrobot` MCP server is installed and connected: `hermes mcp install
  socialrobot`. OAuth signs you in through a browser; an API key
  (`x-api-key` header) works for headless setups. No account yet?
  [Get a free account](https://socialrobot.io).
- Scopes granted include `analytics:read` and `audience:read`.

## How to Run

1. Define the window and scope with the user (campaign, platform, account).
2. Pull per-platform data with `get_posts_with_analytics` (needs `platform`
   and `accountId` from `list_connected_accounts`).
3. Aggregate, compare, and write the recap.

## Quick Reference

| Tool | Role in the report |
|------|--------------------|
| `list_connected_accounts` | Discover the account IDs to query. |
| `get_posts_with_analytics` | Per-platform post list with metrics and history for the window. |
| `get_account_analytics` | Account-level series over the window. |
| `get_follower_demographics` | Audience context (who the platform reached). |

## Procedure

1. **Scope the report.** Confirm the campaign name or window (start and end
   dates), and which platforms and accounts it covers. A report without an
   explicit window is not verifiable.
2. **Pull the data.** For each platform in scope: `get_posts_with_analytics`
   with the account ID and `startDate`/`endDate`. Note the platforms it covers
   (instagram, facebook, threads, pinterest, linkedin, tiktok) and state
   clearly when a platform is not covered.
3. **Aggregate honestly.** Sum or average only within comparable metric
   families. Platforms report different native metrics (for example reach vs
   impressions vs engagements); keep them labeled per platform instead of
   forcing one cross-platform number.
4. **Add context.** `get_account_analytics` for the window trend and
   `get_follower_demographics` for who the posts reached, where supported.
5. **Write the recap.** Structure: window and scope, per-platform performance,
   top posts (with ids), what worked and what did not, and a recommendation
   grounded in the numbers. Flag data gaps (platforms with no posts, metrics
   the platform did not return).

## Pitfalls

- Comparing incompatible metrics across platforms as if they were the same
  number. Label each metric with its platform.
- Claiming causation ("this post caused signups") from engagement data alone.
- Ignoring the window: numbers from outside the requested range do not belong
  in the report.
- Reporting for platforms with no posts in the window as if they underperformed;
  report them as not covered.
- Rounding or trimming outliers to make the story cleaner. Report what the
  tools returned.

## Verification

Every number in the recap traces to a tool call with an explicit platform,
account, and window. The report states per-platform coverage, and no metric is
presented without its platform label and time range.
