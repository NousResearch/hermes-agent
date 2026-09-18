---
title: "Video Shotcraft — Cinematic product videos with Remotion shot recipes"
sidebar_label: "Video Shotcraft"
description: "Cinematic product videos with Remotion shot recipes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Video Shotcraft

Cinematic product videos with Remotion shot recipes.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/video-shotcraft` |
| Path | `optional-skills/creative/video-shotcraft` |
| Version | `0.1.0` |
| Author | Vincent Wei (Vincentwei1021), Hermes Agent |
| License | Apache-2.0 |
| Platforms | linux, macos |
| Tags | `Video`, `Remotion`, `Creative`, `Motion-Design` |
| Related skills | [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# video-shotcraft

Turn a frontend project or webpage into a cinematic product/promo video with
[Remotion](https://www.remotion.dev/): real page screenshots, 2.5D camera moves,
beat-synced cuts, and film-grade sound design, driven by a library of 152 shot
recipe cards and a production-ready template ("Ink Press"). This is a heavy
skill: it shallow-clones the upstream repository at use time rather than
vendoring its ~186MB of docs, demos, and audio assets. Upstream is
[Vincentwei1021/video-shotcraft](https://github.com/Vincentwei1021/video-shotcraft)
(Apache-2.0, credit to Vincentwei1021); this port carries its license.

## When to Use

- The user asks to turn a frontend project, web app, or webpage into a product,
  promo, launch, or demo video.
- The user names "video-shotcraft", the "Ink Press" template, or asks to
  reproduce that template's look for their own product.
- The user wants a single shot's motion effect (e.g. a card fly-in, a spotlight
  hero, a 3D ticker wall) reproduced from one of the shot recipe cards.
- The user wants a beat-synced cut of product footage to a chosen BGM.

Not for: general video editing of existing footage (use ffmpeg directly), or
AI-generated video (use the FLUX video tools if available).

## Prerequisites

- Node.js 18+ and npm (Remotion projects; renders run via `npx remotion`).
- ffmpeg (frame extraction for QA, contact sheets, audio work).
- A browser-screenshot capability for real page captures: the Hermes
  `browser_exec` tool works for full-page 2x screenshots, or install
  Playwright/Puppeteer via `terminal` inside the working project.
- ~1-2GB free disk: the upstream clone is large even shallow, and renders
  produce sizable mp4s.
- Optional: `librosa` (Python) for BGM beat analysis on beat-synced pieces;
  `three` + `@react-three/fiber` + `@remotion/three` for the 3D components.

## How to Run

All commands run through the Hermes `terminal` tool. Remotion renders of a full
promo take many minutes: run them with `background=true` and
`notify_on_complete=true`, then poll with the `process` tool instead of
blocking. Quick single-frame checks (`npx remotion still`) are fine in the
foreground with a generous timeout.

First, fetch the upstream library (docs + demo sources; skip heavy media):

```bash
git clone --depth 1 https://github.com/Vincentwei1021/video-shotcraft \
  ~/scratch/video-shotcraft-upstream
```

If bandwidth matters, use a blobless sparse clone instead and check out only
what you need (`SKILL.md`, `references/`, `template/`, `demos/`, `assets/lib/`,
`assets/audio/` when doing sound design):

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Vincentwei1021/video-shotcraft ~/scratch/video-shotcraft-upstream
cd ~/scratch/video-shotcraft-upstream
git sparse-checkout set SKILL.md README.md references template demos assets/lib
```

Then read, in the clone: `SKILL.md` (mode selection and core principles),
then the clone's `references/pipeline.md` (the eight-stage production
pipeline), and — per route — the clone's `template/TEMPLATE.md`,
`references/guided-free-creation.md` and `references/sound-design.md` in the clone,
`references/music-beat-sync.md` in the clone, plus the clone's
`references/aesthetic-rules.md` and `references/final-review.md` in the clone.

## The Three Modes

Pick (or confirm) exactly one mode before gathering material or writing video
code. If the user already chose, execute it without re-asking.

1. **Template replacement** — the user wants something close to the Ink Press
   template film. Read `template/TEMPLATE.md` in the clone and follow its
   swap guide: replace screenshots, copy, and brand tokens shot by shot.
2. **Autonomous free creation** — the user authorizes the agent to decide.
   Read `references/pipeline.md` in the clone and drive stages 0-7 end to end:
   product brief, visual direction, shot mapping, storyboard, capture,
   implementation, sound, final review — recording decisions instead of
   pausing for confirmation. Explicit user requirements remain hard
   constraints throughout.
3. **Guided co-creation** — the user wants to approve key decisions. Read
   `references/guided-free-creation.md` in the clone; confirm the product
   brief, requirement decisions, visual direction, shot mapping, and final
   storyboard with the user (1-3 questions per round), then continue from
   pipeline stage 4 (final capture) without re-litigating approved stages.

Two shortcuts from upstream: naming "Ink Press" selects mode 1 directly, and
naming specific shot cards fixes those cards as constraints (single shots can
proceed once the target material is clear). For a single shot, pick the card
under `references/shots/` in the clone, read it in full, and locate its exact
demo source via the card's reference-implementation pointer — the demo TSX,
not the card prose, holds the tuned parameters.

## Procedure

1. **Clone upstream** shallow into `~/scratch/video-shotcraft-upstream/` as
   shown above. Expect a large download even with `--depth 1`.
2. **Read the upstream docs**: `SKILL.md`, then `references/pipeline.md` in the
   clone, plus the mode-specific doc from The Three Modes.
3. **Inspect the product read-only**: positioning, key features, page states,
   design tokens (fonts, spacing, colors from source or computed styles), how
   to start a dev server, and data risk. Sensitive data (customer, personal,
   keys, live) must be faked or frozen before capture.
4. **Storyboard**: lock visual direction with a cheap HTML/CSS styleframe
   (2-3 static 1920x1080 keyframes, screenshot-verified) before any Remotion
   code; map features to shot cards by scanning the frontmatter of
   `references/shots/` in the clone; write the storyboard as the
   `| # | time | shot | key motion |` table upstream prescribes.
5. **Capture final material**: start the product's dev server and take
   full-page 2x screenshots plus element crops and a layout.json coordinate
   table (via `browser_exec` or Playwright). Capture only after the storyboard
   is final — wrong material discovered mid-implementation scraps whole scenes.
6. **Implement shots**: scaffold a Remotion project (upstream initializes new
   ones with `npx create-video@latest`, 30fps, 1920x1080), copy needed
   components from `assets/lib/` in the clone (PageCam, Caption, FlashCut,
   etc. — copy, don't import), and adapt each chosen card's demo source.
   Deterministic rendering only: no `Date.now()`/`Math.random()`, seed all
   pseudo-randomness.
7. **Render and QA**: verify each shot with a still, re-render the full piece
   after each feedback round, and extract frames with ffmpeg for review.
   Upstream's commands (see Pitfalls re: verification):
   `npx remotion still src/index.ts <Comp> out/qa/<name>.png` and
   `npx remotion render src/index.ts <Comp> out/promo.mp4`. The template
   renders with `npm install && npx remotion render src/index.ts AiflPromo
   out/promo.mp4`. Sound design follows the clone's `references/sound-design.md`;
   the bundled SFX under `assets/audio/` are free for commercial use
   (see its ATTRIBUTION.md). Finish with an independent final review against
   the clone's `references/final-review.md` and
   the clone's `references/aesthetic-rules.md`, in a fresh subagent context.

## Pitfalls

- **The repo is ~186MB even shallow.** Use `--depth 1`, expect a slow first
  clone, and prefer the sparse-checkout variant — `assets/audio/`, `gallery/`,
  and `demos/_textures/` are only needed for the phases that use them. Gallery
  preview mp4s are not in git at all; browse them online at
  https://vincentwei1021.github.io/video-shotcraft/ instead of running
  `gallery/fetch-media.sh`.
- **Render commands are unverified by this port.** They are copied verbatim
  from upstream's docs, which this skill's author read but did not execute
  (installing Remotion is heavy). If `npx remotion` flags have drifted, trust
  `npx remotion --help` and upstream's current docs over this file.
- **Upstream docs are Chinese-primary.** `SKILL.md`, `references/pipeline.md`,
  and most shot cards are written in Chinese; read them directly (translate
  key constraints into your working notes) rather than skipping them — the
  cards' "known pitfalls" callouts must not be degraded when adapting demos.
- **Full renders are slow.** Minutes to tens of minutes per pass; always run
  them backgrounded and iterate on stills, not full renders.
- **Don't hand-build UI for real-page shots.** Reproducing an existing page
  demands real screenshots (2x textures + element crops); hand-built UI is
  only for abstract/brand segments and must meet publication quality.
- **Some demos need extra deps**: `@remotion/motion-blur` for a few cards,
  `three`/`@react-three/fiber`/`@remotion/three` for FlatPanel and the camera
  helpers — check `demos/README.md` in the clone before copying.
- **Beat-synced pieces**: analyze the BGM first (BPM grid + drum-hit
  classification per `references/music-beat-sync.md` in the clone), pin sparse
  accents to real transients, and validate cut error ≤3 frames from the
  rendered audio track. Deliver both a with-BGM and a no-BGM (SFX-kept)
  version via a props file, as upstream specifies.

## Verification

- The clone exists and `SKILL.md`, `references/pipeline.md`, and the
  mode-specific doc were actually read before storyboarding.
- Every implemented shot has a QA still under `out/qa/`, and the final mp4
  plays end to end (probe with `ffprobe`; extract keyframes with ffmpeg and
  inspect entrance/hold/exit frames per shot).
- The final piece passes the checklist in the clone's `references/final-review.md`
  and `references/aesthetic-rules.md` from the clone: design-spec consistency,
  feature coverage, shot fidelity, audio/visual technical quality, and no
  sensitive data on screen.
