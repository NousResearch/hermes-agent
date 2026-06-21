---
title: "Media Content Production"
sidebar_label: "Media Content Production"
description: "Media umbrella: YouTube transcript extraction, GIF search/download, audio feature visualization, music generation, and song prompt workflows"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Media Content Production

Media umbrella: YouTube transcript extraction, GIF search/download, audio feature visualization, music generation, and song prompt workflows.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/media/media-content-production` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Media Content Production

Use this skill for retrieving, transforming, analyzing, or generating media content.

## Universal workflow

1. Identify the media source, rights/usage constraints, target format, and quality requirements.
2. Prefer official APIs or deterministic CLI tools over browser scraping.
3. Download or generate a small sample first for long media jobs.
4. Verify the resulting file exists, opens, and has expected duration/format/metadata.

## YouTube content

- Fetch transcripts with metadata and timestamps when possible.
- Summaries should preserve source context, speaker/topic shifts, and timestamps for claims.
- If transcript extraction fails, report that directly and try alternate language/fallback methods.

## GIF search

- Use Tenor/API search with explicit query, rating/safety, and result size.
- Prefer preview/tinygif for chat reactions and full GIF for deliverables.
- Download and verify file type/size before attaching.

## Audio analysis

- Use spectrograms, mel/chroma/MFCC, and time slices to inspect audio structure.
- Report parameters (window, hop, sample rate, slice) with the visualization.

## Music generation and songwriting

- For lyrics: structure, rhyme/meter, emotional arc, and genre conventions matter more than long prompts.
- For generation tools: check hardware/API availability, generate a short sample first, and deliver prompt + audio artifact when possible.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
