---
title: "Tiktok Publisher — Publish TikTok videos through MyBrandMetrics"
sidebar_label: "Tiktok Publisher"
description: "Publish TikTok videos through MyBrandMetrics"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Tiktok Publisher

Publish TikTok videos through MyBrandMetrics.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/tiktok-publisher` |
| Path | `optional-skills/social-media/tiktok-publisher` |
| Version | `1.0.0` |
| Author | Clawbus; Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `TikTok`, `Social Media`, `Publishing`, `MyBrandMetrics` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# TikTok Publisher Skill

Use this skill to publish TikTok videos through a preconfigured MyBrandMetrics
API connection. It supports hosted video URLs, local video files, privacy
settings, optional publish polling, and status checks for existing publish jobs.

## When to Use

- The user asks to publish, upload, or post a video to TikTok.
- The user provides a hosted video URL or local video file and wants it sent to
  TikTok.
- The user wants to create a private draft-like post with `SELF_ONLY` privacy.
- The user asks to check the status of an existing TikTok publish job.

## Prerequisites

- A readable local file path or direct public video URL.
- `TIKTOK_API_KEY` or `TIKTOK_AUTHORIZATION_TOKEN`, or a config file described in
  `references/configuration.md`.
- Network access to `https://api.mybrandmetrics.com`.
- A title and privacy level before publishing.

Supported privacy levels depend on the MyBrandMetrics/TikTok account
configuration. Use `SELF_ONLY` when the user wants a private post.

## References

- `references/configuration.md` — credential sources and config file shape.
- `references/publishing-examples.md` — concrete publish and status examples.

## How to Run

Use the `terminal` tool with the scripts:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/social-media/tiktok-publisher/scripts/publish_tiktok.py" \
  --source "https://example.com/video.mp4" \
  --title "My TikTok title" \
  --privacy-level "SELF_ONLY"
```

Check an existing publish job:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/social-media/tiktok-publisher/scripts/check_status.py" \
  --publish-id "PUBLISH_ID"
```

## Quick Reference

Publish a remote URL:

```bash
python3 "$HERMES_HOME/skills/social-media/tiktok-publisher/scripts/publish_tiktok.py" \
  --source "https://example.com/video.mp4" \
  --title "My TikTok Title" \
  --privacy-level "SELF_ONLY"
```

Publish a local file and wait:

```bash
python3 "$HERMES_HOME/skills/social-media/tiktok-publisher/scripts/publish_tiktok.py" \
  --source "/path/to/video.mp4" \
  --title "Hello from local file" \
  --privacy-level "PUBLIC" \
  --wait-for-published
```

Publish with polling options:

```bash
python3 "$HERMES_HOME/skills/social-media/tiktok-publisher/scripts/publish_tiktok.py" \
  --source "https://example.com/video.mp4" \
  --title "Scheduled check" \
  --privacy-level "SELF_ONLY" \
  --poll-interval 5000 \
  --poll-timeout 300000
```

Check status:

```bash
python3 "$HERMES_HOME/skills/social-media/tiktok-publisher/scripts/check_status.py" \
  --publish-id "abc123"
```

## Procedure

1. Determine whether the user wants a new publish request or a status check.
2. For new publishing, confirm `source`, `title`, and `privacy_level`.
3. If the source is local, verify the file exists before calling the API.
4. Load credentials from environment variables or a user-supplied config path.
5. Run `scripts/publish_tiktok.py` for publishing or `scripts/check_status.py`
   for status checks.
6. Return the real JSON response from the API. Do not fabricate publish IDs,
   TikTok URLs, or final statuses.

For destructive or public-facing posts, confirm with the user before publishing
with a public privacy level.

## Pitfalls

- Never commit or echo real TikTok/MyBrandMetrics API keys.
- `TIKTOK_API_KEY` and `TIKTOK_AUTHORIZATION_TOKEN` are both accepted; environment
  variables override config file values.
- Local uploads use multipart form data and require a readable local file.
- URL uploads send JSON and require the URL to be directly reachable by the API.
- Non-200 API responses are printed to stderr and the scripts exit non-zero.

## Verification

```bash
python3 "$HERMES_HOME/skills/social-media/tiktok-publisher/scripts/check_status.py" \
  --publish-id "PUBLISH_ID"
```
