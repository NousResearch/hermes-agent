---
title: "Cad — Parametric CAD to STEP/STL/3MF/GLB, upstream-maintained"
sidebar_label: "Cad"
description: "Parametric CAD to STEP/STL/3MF/GLB, upstream-maintained"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Cad

Parametric CAD to STEP/STL/3MF/GLB, upstream-maintained.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/cad` |
| Path | `optional-skills/creative/cad` |
| Version | `0.7.20` |
| Author | Thompson Labs LLC (earthtojake) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `cad`, `3d-printing`, `build123d`, `step`, `stl`, `3mf`, `glb`, `mechanical-design` |
| Related skills | [`urdf`](../../optional/creative/creative-urdf.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# CAD (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad): the
> project ships a self-contained skill directory (`skills/cad/`) with its
> model contract and 12 topic references. `hermes skills install
> official/creative/cad` pulls the current tree live from that repo
> (quarantined and scanned like any hub install). This directory holds only
> the catalog metadata, so the copy can never go stale against the pinned
> `cadgen` runtime the upstream skill names.

The skill turns a text brief, a dimensioned drawing or a photo into a
parametric build123d model: a plain Python script whose decorated function
returns a shape and writes STEP (plus optional STL, 3MF or GLB) on every run.
It covers assemblies with source-defined placements, vendor STEP inputs,
kinematics and animation, geometry checks written against the saved STEP
(`read_step`/`read_scene`, bounding boxes, volumes, clearances), and a
mandatory rendered-snapshot review of every visible change. Prompt references
such as `/work/robot/STEP/assembly.step#o1.2.f7` resolve to exact faces and
edges in a saved document.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) on `PATH`. Every command runs as
  `uvx --no-config --managed-python --python 3.13 --from cadgen==<pin> ...`;
  the first run downloads that runtime, the first snapshot a ~115 MB headless
  Chromium. No system Python packages are needed.
- `cadgen` starts a warm build daemon on first build and leaves it running
  (`cadgen daemon status`); stop it by PID when the session's CAD work ends.
- **Telemetry is on by default** in `cadgen` (usage counts and crash
  locations under a random install id, never files, paths or prompts). Run
  `cadgen telemetry off` once, or set `CADGEN_TELEMETRY=0` (or
  `DO_NOT_TRACK=1`) on each command, unless the user wants to share stats.
- The upstream skill offers `cad_show` / `cad_view` tools when its host
  provides them. Hermes does not, so use the CLI path the skill documents:
  `cadgen viewer --host 127.0.0.1 --json --detach`, then hand the user
  `url?file=<absolute path>`. Headless sessions (cron, gateway) skip the
  viewer and report the output paths plus the snapshot review instead.
- Sibling skills it names with a `$` prefix (`$dxf`, `$step-parts`, `$srdf`)
  are not in the Hermes catalog yet; treat those mentions as optional.

Pair with `urdf` (same upstream, same runtime) when the parts become a robot
description.

Full documentation: https://github.com/earthtojake/text-to-cad#readme
