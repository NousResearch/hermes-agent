---
title: "Simplepost — Publish, schedule, and manage social posts safely"
sidebar_label: "Simplepost"
description: "Publish, schedule, and manage social posts safely"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Simplepost

Publish, schedule, and manage social posts safely.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/simplepost` |
| Path | `optional-skills/social-media/simplepost` |
| Version | `0.1.0` |
| Author | Vladimir Haltakov (haltakov), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `SimplePost`, `Social Media`, `Publishing`, `Scheduling`, `MCP` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SimplePost Skill

Use SimplePost to preview, draft, schedule, publish, inspect, edit, or discard posts through social accounts the user connected at `app.simplepost.social`. This skill preserves supplied copy, resolves real account IDs, and makes every write auditable; it does not connect accounts, invent social content, or provide engagement analytics.

## When to Use

- The user asks to publish supplied content now or at a future time.
- The user wants to preview, validate, save, inspect, edit, or discard a SimplePost draft or scheduled post.
- The user wants the same root post sent to multiple connected accounts, with optional account-specific overrides.
- The user asks about a prior SimplePost delivery result or queued post.
- Do not use this skill for generic copywriting without a SimplePost action.
- Do not use it to connect, disconnect, or reauthorize a social account; those actions happen in the SimplePost web app.

## Prerequisites

Hermes needs the hosted SimplePost MCP server. Once the catalog entry is available, install and authorize it through `terminal`:

```text
terminal(command="hermes mcp install simplepost", timeout=120)
terminal(command="hermes mcp login simplepost", timeout=315)
terminal(command="hermes mcp test simplepost", timeout=60)
```

Before the catalog entry is merged, add the same hosted server directly:

```text
terminal(command="hermes mcp add simplepost --url https://app.simplepost.social/mcp --auth oauth", timeout=30)
terminal(command="hermes mcp login simplepost", timeout=315)
```

Complete OAuth in the browser, then start a new Hermes session or reload MCP. Never paste a SimplePost access token into chat, source files, or configuration.

## How to Run

Invoke the skill with a normal request that states the action, content, targets, and timing. Examples:

- “Use SimplePost to preview this exact text for my connected X and LinkedIn accounts.”
- “Save this image and caption as a draft for my Instagram account.”
- “Schedule this exact thread for 2026-10-03 at 09:00 Europe/Berlin.”
- “Inspect my failed SimplePost posts from today and explain each returned error.”

Use the SimplePost MCP tools directly after loading the skill. Use `terminal` only for Hermes MCP installation, login, testing, or recovery.

## Quick Reference

| Need | SimplePost MCP tool | Writes data? |
| --- | --- | --- |
| Resolve connected targets | `list_accounts` | No |
| Validate without saving | `validate_post` | No |
| Preview without saving | `preview_post` | No |
| Render a post preview | `show_post_preview` | No |
| Publish, schedule, or draft | `create_post` | Yes |
| Find drafts, queued, or delivered posts | `inspect_posts` | No |
| Edit a draft or future scheduled post | `update_scheduled_post` | Yes |
| Discard a draft or future scheduled post | `discard_scheduled_post` | Yes |
| Inspect the posting calendar | `get_schedule` or `show_schedule` | No |
| Upload an attachment exposed by the client | `upload_media` | Yes |

Read [`references/mcp.md`](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/social-media/simplepost/references/mcp.md) for payload rules, result interpretation, media handling, TikTok options, and recovery behavior.

## Procedure

1. **Identify the exact action and content.** Distinguish publish-now, schedule, draft, preview, validate, inspect, edit, and discard. Preserve user-supplied text exactly unless the user asks for rewriting or platform-specific adaptation.

2. **Resolve real targets.** Call `list_accounts` before a write. Use returned `accountId` values and display names; never invent account IDs, aliases, board IDs, chat IDs, or post IDs. Check returned allowance or eligibility information and explain any block instead of retrying around it.

3. **Resolve only missing essentials.** A request to publish exact content to named connected accounts is authorization to publish; do not add another confirmation step. Ask only when a target, required media item, future time, timezone, or destructive target cannot be determined safely.

4. **Prepare media.** Pass public HTTPS URLs directly. Use `upload_media` only when Hermes exposes the attachment as a structured file parameter. Never reconstruct a file parameter from a path, filename, file ID, or base64 text. Preserve metadata and any `fittedMedia` returned by validation or preview.

5. **Choose the lightest preflight.** Use `preview_post` when the user wants to inspect the proposed result, `validate_post` when they ask for a check without saving, and `show_post_preview` when they want the rendered view. A direct, complete publishing request does not require an automatic preview.

6. **Normalize scheduling.** Convert relative language into a future ISO 8601 datetime with an explicit offset or `Z`. Ask for the timezone only if it cannot be inferred safely. Repeat the absolute time and timezone in the final response.

7. **Execute once.** For each new `create_post` intention, generate one unique `idempotencyKey`. If a call times out or returns an uncertain result, retry only with the same key and unchanged content. Never use a new key for an uncertain write because it can create a duplicate.

8. **Inspect the complete result.** A successful tool call can still contain per-account or per-thread failures. Read validation, summary, posting results, thread results, warnings, returned post IDs and URLs, and any automatic repost state.

9. **Make the outcome auditable.** Report the exact root text and ordered thread segments, selected accounts, absolute schedule, validation warnings, partial failures, and useful returned IDs or URLs. Clearly distinguish “draft saved” from “valid and ready to publish.”

## Pitfalls

- SimplePost can edit or discard drafts and future scheduled posts. It cannot edit, delete, or undo a post already published to a social platform.
- Drafts may be saved even when validation fails. A saved invalid draft is not ready to publish.
- `thread` follow-ups are supported on X, Bluesky, Threads, and Telegram. Other targets receive only the root and should surface a warning.
- MCP and hosted API calls target connected `accountIds`; do not use direct SDK platform credentials or payload shapes with these tools.
- Immediate publishing can take minutes. An uncertain result is not permission to submit again with a new idempotency key.
- `inspect_posts` reports SimplePost content and delivery state, not reach, clicks, impressions, or social engagement.
- Account connection and reauthorization happen at `https://app.simplepost.social`; do not search tools for unsupported account-management actions.
- TikTok available audiences and posting restrictions are account-specific. Never silently replace an unavailable requested audience with another.

## Verification

For a non-writing smoke test, call `list_accounts`, then `preview_post` or `validate_post` with clearly marked test content and confirm no post was created. For a writing smoke test, save a draft, inspect the returned ID with `inspect_posts`, render it with `show_post_preview`, and discard it only if the user authorized that cleanup.

For a real publish or schedule, verify every requested account has a corresponding result and report partial failures individually. Restart Hermes after the first OAuth setup and confirm the SimplePost MCP tools reconnect without requesting or exposing credentials.
