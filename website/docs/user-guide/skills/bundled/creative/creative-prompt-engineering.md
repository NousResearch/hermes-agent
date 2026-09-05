---
title: "Prompt Engineering"
sidebar_label: "Prompt Engineering"
description: "Use when the user asks for a prompt for any AI generation tool (image, video, music, voice), or says 'I need a prompt'"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Prompt Engineering

Use when the user asks for a prompt for any AI generation tool (image, video, music, voice), or says 'I need a prompt'. Covers Nano Banana Pro 2, Veo 3.1, Kling 3.0/2.6/O1, Z Image Turbo, Flux, Krea 1, Elevenlabs TTS, and SUNO music.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/creative/prompt-engineering` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `creative`, `prompts`, `ai-generation`, `image`, `video`, `music`, `voice` |
| Related skills | [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music), [`heartmula`](/docs/user-guide/skills/bundled/media/media-heartmula), `image_gen` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Prompt Engineering for AI Generation

## Overview

Generate production-ready prompts for AI creative tools. Use when the user says "I need a prompt" or wants image, video, music, or voice generated. Think like a creative director, not a tag list.

## Tool Selection

| Goal | Tool | Goal | Tool |
|------|------|------|------|
| Best image | Nano Banana Pro 2 | Cinematic video | Veo 3.1 |
| Real-time image | Krea 1 | Character video | Kling 3.0 |
| Text in images | Flux / Z Image Turbo | Beat-synced video | Kling 2.6 |
| Speed | Z Image Turbo | UGC / product | Kling O1 |
| Full songs | SUNO | Expressive TTS | Elevenlabs |

## Universal Principles

1. **Specificity beats keywords** — Write like a creative director, not a tag list
2. **Front-load important elements** — Models weight early content heavily
3. **One change at a time** — Adjust single variables when iterating
4. **Motion needs verbs** — Video prompts require action, not static scenes
5. **Lock what stays** — Explicitly state preserved elements when editing

## Cross-Tool Pipelines

**Nano Banana → Kling (character video):** Generate 2-3 angle character sheet in Nano Banana, export at 1080p+, then Kling prompt describes **only motion** — never redescribe the subject. See [references/cross-tool.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/cross-tool.md).

## References

- [references/nano-banana.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/nano-banana.md) — Image (reasoning, edits, consistency)
- [references/veo.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/veo.md) — Video (cinematic, native audio, multi-shot)
- [references/kling.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/kling.md) — Video (3.0/2.6/O1, beat-matched)
- [references/flux.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/flux.md) — Image (text rendering, layered)
- [references/z-image.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/z-image.md) — Image (speed, bilingual)
- [references/krea.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/krea.md) — Image (real-time, artistic)
- [references/elevenlabs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/elevenlabs.md) — Voice (expression brackets)
- [references/suno.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/suno.md) — Music (lyrics, style JSON)
- [references/cross-tool.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/creative/prompt-engineering/references/cross-tool.md) — Pipelines and common pitfalls
