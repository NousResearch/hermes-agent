---
title: "Meshy 3D Printing — Print-ready models and Creative Lab products with Meshy"
sidebar_label: "Meshy 3D Printing"
description: "Print-ready models and Creative Lab products with Meshy"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Meshy 3D Printing

Print-ready models and Creative Lab products with Meshy.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/meshy-3d-printing` |
| Path | `optional-skills/creative/meshy-3d-printing` |
| Version | `0.7.0` |
| Author | Xubin Du (Arlieeee), Meshy |
| License | MIT |
| Platforms | macos, linux, windows |
| Tags | `3d`, `3d-printing`, `multicolor`, `3mf`, `stl`, `slicer`, `meshy` |
| Related skills | [`meshy-3d-generation`](../../optional/creative/creative-meshy-3d-generation.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Meshy 3D Printing (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [meshy-dev/meshy-3d-agent](https://github.com/meshy-dev/meshy-3d-agent): the project ships the
> skill directory (`skills/meshy-3d-printing/`) with its printing recipes and the shared setup,
> delivery and troubleshooting references, released together with the Meshy CLI version it pins.
> `hermes skills install official/creative/meshy-3d-printing` pulls the current tree live from
> that repo (quarantined and scanned like any hub install) — this directory holds only the
> catalog metadata, so the vendored copy can never go stale.

The skill drives the [Meshy CLI](https://www.npmjs.com/package/meshy-cli) (`meshy`) to deliver
print files at the intended size: white models from text or a photo, printability analysis and
repair (with wall-thickness checks for a given printer and size), multicolour 3MF, splitting a
model into printable parts, and Creative Lab products (figures, lamps, keychains, fridge
magnets, keycaps, fidget toys). It can open the result in a detected slicer. Preparing an OBJ
the user already has, or opening an STL/3MF, runs locally with no account and no cost.

For digital assets (textures, rigging, animation, game-ready models), use
`meshy-3d-generation`.

## Prerequisites

- Node.js 22.12+ with npm on `PATH`. The skill uses an installed `meshy` of the pinned version,
  or runs that version temporarily with `npm exec`; a global install is optional.
- For cloud work, a Meshy account. Sign-in is the CLI's browser device flow
  (`meshy auth login --device`): the user approves in the browser and nothing secret is pasted
  into the chat. An existing `MESHY_API_KEY` is honoured instead.
- Generation, repair, multicolour, splitting and Creative Lab spend Meshy credits; analysis and
  local preparation are free. The skill asks before any paid stage the user has not agreed to.
- Network: the Meshy API and sign-in page, the media hosts of downloaded assets, and the npm
  registry when the CLI is installed or run temporarily. Slicer handoff launches an
  already-installed slicer.

Full documentation: https://github.com/meshy-dev/meshy-3d-agent#readme
