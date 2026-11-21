---
title: "Mod Any Game — Mod a PC game you own, idea to in-game clip, upstream-kept"
sidebar_label: "Mod Any Game"
description: "Mod a PC game you own, idea to in-game clip, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Mod Any Game

Mod a PC game you own, idea to in-game clip, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/gaming/mod-any-game` |
| Path | `optional-skills/gaming/mod-any-game` |
| Version | `0.2.0` |
| Author | rehan-remade |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `game-modding`, `modding`, `reverse-engineering`, `unity`, `unreal`, `godot`, `minecraft`, `terraria`, `sprites`, `showcase-video` |
| Related skills | [`minecraft-modpack-server`](../../optional/gaming/gaming-minecraft-modpack-server.md), [`dream-loop`](../../optional/creative/creative-dream-loop.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# mod-any-game (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [rehan-remade/universal-modder](https://github.com/rehan-remade/universal-modder):
> the project ships a hub skill directory (`skills/mod-any-game/`, a `SKILL.md`
> plus per-engine references, case studies and a safety sheet) next to a `um`
> CLI and a shared knowledge base of field notes. `hermes skills install
> official/gaming/mod-any-game` pulls the current hub tree live from that repo
> (quarantined and scanned like any hub install) — this directory holds only
> the catalog metadata, so the engine notes can never lag the tooling.

universal-modder is a loop for modding a PC game the user owns, from "add a
tactical nuke to Terraria" or "make a new civilization for Age of Empires II"
to a working mod verified in the real game and recorded as a clip. The hub skill
runs the same ten steps every time: search the knowledge base for prior art on
the game and engine; recon (engine, version, managed vs native code, mod
loaders, anti-cheat, save paths); pick a route (data mod, script mod, managed
patch, native hook, mashup); set up a safe lab with saves backed up; read the
actual game code instead of guessing; build one working vertical slice;
generate assets; verify in the running game; record a showcase; package; and
leave a field note for the next agent. The per-engine references cover Unity,
Unreal, Godot, Source, Bethesda, .NET/XNA, Genie (AoE2), Minecraft, native
binaries, retro decompilations and the big modding frameworks.

## Prerequisites

- Python 3.10+ and `ffmpeg` on `PATH`. The `um` CLI (engine scanner, save
  backups, sprite and 3D-to-sprite pipelines, window capture and input on
  Windows, showcase video cutting, knowledge-base search) installs with
  `uv tool install git+https://github.com/rehan-remade/universal-modder`
  (pin a commit with `@<sha>` when reproducibility matters; `pipx` also works).
  Blender is needed only for 3D-model-to-sprite renders (`um render3d`).
- The upstream hub is `SKILL.md` + `references/` (~55 KB, no scripts); the
  fetch is pinned to one tree SHA, recorded in the bundle metadata.
- Assets: upstream generates sprites, textures, 3D models, SFX and music
  through fal (`um fal <recipe>` or the fal MCP server, `FAL_KEY`). In Hermes
  use the native `image_generate` tool for 2D concept art, icons and sprite
  sources first; `FAL_KEY` is optional and only unlocks `um fal` for 3D, audio
  and video recipes. Every `um fal` call is billed to the user's fal account.
- The knowledge base (`um kb search "<game>"`) reads `knowledge/` from a clone
  of the repo when one is on the path, otherwise `um kb sync` mirrors it from
  GitHub into `~/.universal-modder` (override with `UM_HOME`); no auth needed.
- Windows games are driven natively or from WSL (`um win launch`, `um win
  shot`, `um win drive`, `um win record`); Linux and macOS hosts can still do
  recon, code reading, data and script mods, asset work and packaging.

## Companion skills (same repo, not pulled by this stub)

The hub names nine companion skills that live beside it upstream:
`game-recon`, `reverse-engineering`, `fal-assets`, `asset-pipeline`,
`game-automation`, `mashup-mods`, `showcase-video`, `publish-mod` and
`share-field-notes`. Install any of them straight from the repo when a phase
needs the deep version (community trust, scanned on install):

```bash
hermes skills install rehan-remade/universal-modder/skills/game-recon
hermes skills tap add rehan-remade/universal-modder   # then browse/search the rest
```

## Safety rules the hub enforces (keep them)

- Back up saves, profiles and config folders with `um backup create` before
  the first modded launch; keep the restore path written down.
- Driving input (`um win drive`) takes over the user's mouse and keyboard: ask
  before long runs, and kill game processes by exact PID (`um win kill <pid>`),
  never by name pattern.
- Single-player, offline, or servers the user runs — never the client of an
  online game behind anti-cheat (EasyAntiCheat, BattlEye, Vanguard, VAC on
  official servers), and never a bypass of anti-cheat, DRM or ownership
  checks; `references/safety.md` draws that line and the recon step settles
  online vs offline before any route is chosen.
- Ship no game files: packaged mods contain only the user's own code and
  generated assets, with tools, loaders and AI-generated assets credited.

## When to prefer it

- The user owns the game and wants new content or mechanics in it, a
  cross-game mashup, or an answer to "what engine is this and has anyone
  modded it?".
- Not for hosting or administering a Minecraft server — use
  `minecraft-modpack-server`. Not for stand-alone 3D visuals with no game
  attached — `dream-loop` covers that loop.

Full documentation: https://github.com/rehan-remade/universal-modder#readme
