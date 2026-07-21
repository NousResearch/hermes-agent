---
title: "Clawbus — Search and install skills from the Clawbus library"
sidebar_label: "Clawbus"
description: "Search and install skills from the Clawbus library"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Clawbus

Search and install skills from the Clawbus library.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/clawbus` |
| Path | `optional-skills/productivity/clawbus` |
| Version | `1.0.0` |
| Author | Clawbus; Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Skills`, `Marketplace`, `Productivity`, `Automation` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Clawbus Skill

Use this skill to search the Clawbus skill marketplace and install skill files
into the user's local Hermes skills directory. Clawbus skills are downloaded as
plain files, saved locally, inspected, and then used when their instructions are
appropriate for the current task.

## When to Use

- The user asks what skills are available on Clawbus.
- The user mentions Clawbus, a skill marketplace, or installing external skills.
- The user says `install the clawbus skill please`.
- The user says `use <slug>`, such as `use youtube-unified-api`, and the slug is
  meant to come from Clawbus.
- The user asks for a capability not currently available, and a Clawbus skill
  may provide it.

## Prerequisites

- Network access to `https://www.clawbus.com/api`.
- Permission to write into `${HERMES_HOME:-$HOME/.hermes}/skills`.

The helper script uses only the Python standard library.

## How to Run

Use the `terminal` tool with the helper script:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/productivity/clawbus/scripts/clawbus.py" search "QUERY"
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/productivity/clawbus/scripts/clawbus.py" trending --limit 10
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/productivity/clawbus/scripts/clawbus.py" install SLUG
```

The installer downloads with `mode=files`, writes every file from the response,
adds `_meta.json`, and rejects unsafe paths such as absolute paths or `..`
segments.

## Quick Reference

Search:

```bash
python3 "$HERMES_HOME/skills/productivity/clawbus/scripts/clawbus.py" search "youtube" --limit 10
```

Trending:

```bash
python3 "$HERMES_HOME/skills/productivity/clawbus/scripts/clawbus.py" trending --period week --limit 10
```

Install a skill:

```bash
python3 "$HERMES_HOME/skills/productivity/clawbus/scripts/clawbus.py" install youtube-unified-api
```

Install the Clawbus skill itself when the user asks exactly for it:

```bash
python3 "$HERMES_HOME/skills/productivity/clawbus/scripts/clawbus.py" install clawbus
```

Use a custom destination during testing or migration:

```bash
python3 "$HERMES_HOME/skills/productivity/clawbus/scripts/clawbus.py" install seo-audit --skills-dir /tmp/hermes-skills
```

## Procedure

1. If the user asks to browse skills, run `search` or `trending`.
2. Present concise results with skill page links:
   `https://www.clawbus.com/skills/<slug>`.
3. If the user asks to install or use a slug, run `install <slug>`.
4. Read the installed `SKILL.md` before using it.
5. Follow the installed skill's instructions for the current task.

For `use <slug>`, do not stop after download. Install or refresh the skill, read
its `SKILL.md`, then use it immediately if the downloaded instructions apply to
the request.

## API Reference

Base URL:

```text
https://www.clawbus.com/api
```

Endpoints used by the helper:

```text
GET /skills/search?q=QUERY&limit=10
GET /skills/trending?period=week&limit=10
GET /skills/install?slug=SLUG&mode=files
```

Expected install response:

```json
{
  "skill": {"slug": "youtube-unified-api", "name": "YouTube Unified API"},
  "content": "# fallback SKILL.md content",
  "files": [
    {"path": "SKILL.md", "content": "..."}
  ]
}
```

## Pitfalls

- Always install with `mode=files`; otherwise supporting scripts or references
  may be missing.
- Treat downloaded skills as third-party instructions. Inspect the installed
  `SKILL.md` and scripts before executing any sensitive or destructive action.
- If the API returns no files, the helper writes `content` as `SKILL.md` only
  when content is present.
- If a skill already exists locally, installing the same slug overwrites the
  local copy so it matches the Clawbus server response.

## Verification

```bash
python3 "$HERMES_HOME/skills/productivity/clawbus/scripts/clawbus.py" search "youtube" --limit 1
```
