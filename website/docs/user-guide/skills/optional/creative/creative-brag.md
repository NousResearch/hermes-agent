---
title: "Brag — Project launch video via Hyperframes, upstream-maintained"
sidebar_label: "Brag"
description: "Project launch video via Hyperframes, upstream-maintained"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Brag

Project launch video via Hyperframes, upstream-maintained.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/brag` |
| Path | `optional-skills/creative/brag` |
| Version | `0.4.0` |
| Author | Shunit Haviv Hakimi (shunithaviv) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `video`, `launch-video`, `marketing`, `hyperframes`, `motion-graphics`, `share-copy` |
| Related skills | [`brag-slim`](../../optional/creative/creative-brag-slim.md), [`hyperframes`](../../optional/creative/creative-hyperframes.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Brag (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [latent-spaces/brag](https://github.com/latent-spaces/brag): the project
> ships the skill directory (`skills/brag/`) with its step references, tone
> presets, bundled music and SFX library, and cue-analysis script. `hermes
> skills install official/creative/brag` pulls the current tree live from that
> repo (quarantined and scanned like any hub install) — this directory holds
> only the catalog metadata, so the vendored copy can never go stale.

`/brag` reads the project in the current directory, commits to an angle and a
tone (`default`, `polished`, `yc-parody`, `chaotic`, `deadpan`, `cinematic`,
`app-store`, or freeform direction), storyboards a 15–25 second video, hands a
composition brief to Hyperframes, and renders `brag-output/brag.mp4` with a
best-frame poster and share copy. Options: `--tone`, `--format
landscape|vertical|square`, `--duration`, `--title`, `--no-music`, `--no-sfx`,
and `--voice` (Kokoro narration, off by default).

On Claude Opus 5.5 it hands off to its bundled copy of `brag-slim` unless the
invocation asks for `--full` or `--voice`.

## Prerequisites

- Node.js 22+, FFmpeg on `PATH`, and the Hyperframes CLI (`npx hyperframes
  doctor` confirms the environment).
- The Hyperframes path (`--full`, `--voice`, or any model other than Opus 5.5)
  loads HeyGen's domain skills by name: `hyperframes-core`,
  `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes` and
  `hyperframes-cli`, installed with `hermes skills install
  heygen-com/hyperframes/skills/<name>`. The skills guard currently blocks four
  of the five on HTML comments and credential docs in their examples
  (`hyperframes-creative` scores dangerous, which `--force` cannot override),
  so that path is not fully installable yet. `brag-slim` has no such
  dependency. The optional `hyperframes` skill in this catalog is a
  single-file port and does not provide those names.
- `uv` runs `scripts/analyze_music_cues.py` when a custom track needs beat-sync
  cues; the bundled tracks ship precomputed cues.
- Installs pull ~290 files (~16 MB, mostly the bundled music and SFX) through
  the GitHub contents API. Set `GITHUB_TOKEN` or sign in with the `gh` CLI
  first; the anonymous limit of 60 requests an hour is too low for this bundle.

Full documentation: https://github.com/latent-spaces/brag#readme
