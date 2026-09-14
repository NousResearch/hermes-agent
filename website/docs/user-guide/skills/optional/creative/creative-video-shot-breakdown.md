---
title: "Video Shot Breakdown — Measured per-shot breakdown (拉片) of any video via ffmpeg"
sidebar_label: "Video Shot Breakdown"
description: "Measured per-shot breakdown (拉片) of any video via ffmpeg"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Video Shot Breakdown

Measured per-shot breakdown (拉片) of any video via ffmpeg.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/video-shot-breakdown` |
| Path | `optional-skills/creative/video-shot-breakdown` |
| Version | `1.0.0` |
| Author | eternityspring (adapted for Hermes Agent) |
| License | Apache-2.0 |
| Platforms | linux, macos, windows |
| Tags | `video`, `shot-list`, `film-analysis`, `ffmpeg`, `editing`, `拉片`, `creative` |
| Related skills | [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content), [`kanban-video-orchestrator`](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator), [`hyperframes`](/docs/user-guide/skills/optional/creative/creative-hyperframes) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Video Shot Breakdown (拉片) — measured cuts, model-judged shots, code-checked gates

Adapted from [reelbench-skills](https://github.com/eternityspring/reelbench-skills)
(Apache-2.0, upstream commit `75520c7b`) for Hermes Agent. Use when asked to
break a finished video into a per-shot analysis table (shot list, 拉片, 拆镜头,
shot durations, shot size / camera move / category per shot, cuts-per-minute,
pacing analysis, reference study of an ad or short film), or to export the
video with a synchronized shot-info panel (annotated shot video, 分镜视频).

Two stages, each a zero-dependency Node (>= 18) CLI plus `ffmpeg`/`ffprobe`:

| Stage | Reference dir | Output | What it does |
|---|---|---|---|
| ① `video-shots` | `references/video-shots/` | `shots.json`, `shots.md`, `shots-report.html`, `frames/`, `sheets/` | ffmpeg scene detection fixes the cut points, durations and per-shot motion; YOU fill size / category / camera / frame / rhythm from contact sheets; 15 deterministic quality gates check every claim |
| ② `video-sync` | `references/video-sync/` | `sync.mp4` | Composites the original video with a scrolling, auto-highlighting shot table (landscape → panel below, portrait → panel right); needs a Chrome/Chromium binary for the panel render |

The premise: **boundaries are measured, not eyeballed.** The model's least
reliable output on video is time, so cut points, durations and motion come
from code; the model only judges what only a model can (framing, category,
camera move, description, rhythm role), and code checks each judgment on the
spot. The hardest gate blocks "push/pan/track" claims when the measured
frame-to-frame change is near zero.

## Workflow (stage ①)

`{skillDir}` below = this skill's directory (wherever it is installed, typically
`~/.hermes/skills/creative/video-shot-breakdown`); `{baseDir}` = `{skillDir}/references/video-shots`.
Run everything with `terminal` from an empty output directory.

1. **Seed** (cuts are fixed here — never hand-edit `start`/`end` later):
   `node {baseDir}/scripts/video-shots.mjs seed <video> --track track.json --title "<title>" --lang en > shots.json`
   Read the stderr line (duration / fps / cuts / shots). Dark, low-contrast or
   short-form footage under-cuts at the default threshold: run `--threshold 0.15`
   as well and compare shot counts against a quick eyeball of the clip; far more
   shots than you'd count → `--threshold 0.4` or merge in step 4. Omit `--lang en`
   for a Chinese report; the flag propagates to every later command. Note that
   `--lang en` switches labels, gate names and violation text — the stderr
   progress lines and the final `N 镜 / N 秒` summary stay zh-CN by design.
2. **Frames + contact sheets:**
   `node {baseDir}/scripts/video-shots.mjs frames shots.json --video <video>`
   `node {baseDir}/scripts/video-shots.mjs sheet shots.json --cols 4 --rows 6`
   `node {baseDir}/scripts/video-shots.mjs sheet shots.json --cols 4 --rows 6 --pick b`
   Sheet **a** (15% into each shot) is content; sheet **b** (85%) is camera
   work — compare the same cell across both sheets to call the move.
3. **Annotate in batches of &lt;= 24 shots** (one 4x6 sheet): load
   `{baseDir}/references/taxonomy.md` (the four vocabularies + criteria) and
   `{baseDir}/references/analysis-pass.md`, then `vision_analyze` the
   sheet(s) — ask for one sentence per cell plus size and on-screen text — and
   fill `size` → `category` → `camera` → `frame` (→ optional `rhythm`) in
   `shots.json` with `execute_code`/`write_file`. Go back to a single
   `frames/S07a.jpg` only for the shots you cannot call from the sheet. When
   the a/b framing difference and the measured `motion` disagree, trust the
   measurement. Burned-in subtitles count as dialogue → `audio` with speaker.
   Declare a top-level `cast` array (`[{"id","name"}]`) so the subjects gate runs.
4. **Recut** missed/extra cuts (dissolves and dark-to-dark are missed; handheld
   shake and flashes over-cut):
   `node {baseDir}/scripts/video-shots.mjs recut shots.json --track track.json --split 63.5 --merge 45.97 > shots.new.json && mv shots.new.json shots.json`
   Rerun `frames`, re-annotate the cleared shots.
5. **Validate — do not skip:**
   `node {baseDir}/scripts/video-shots.mjs validate shots.json --track track.json --frames frames`
   Fix every violation and rerun until all 15 gates pass. A skipped gate
   (no `--track`, no `cast`, no frames dir) is not a pass — say so in the report.
6. **Render:**
   `node {baseDir}/scripts/video-shots.mjs render shots.json --md --track track.json > shots.md`
   `node {baseDir}/scripts/video-shots.mjs render shots.json --html --track track.json --video <video-relative-path> > shots-report.html`
   Report one line: shots, average length, cuts/min, dominant size and camera,
   longest/shortest shot, report path, plus recuts made and hints left.

Full upstream text (zh-CN, verbatim): `references/video-shots/video-shots.md`.

## Workflow (stage ②, optional)

`node {skillDir}/references/video-sync/scripts/video-sync.mjs export shots.json --video <video> --frames frames --lang en --chrome <chrome-binary> -o sync.mp4`
`plan` prints the geometry only; `panels` + `compose` are the two halves of
`export`. Small sources (640x360) need `--scale 2`. Layout lives entirely in
`{skillDir}/references/video-sync/scripts/panel.css`. Full text: `{skillDir}/references/video-sync/video-sync.md`.

## Hermes adaptations (read before following upstream text)

- Upstream text uses another harness's tool names: Read/Write/Edit → `read_file` /
  `write_file` / `patch`, Bash → `terminal`, "Read frames/S07a.jpg" → `vision_analyze`.
- Dense-CJK reference files may be misdetected as binary by `read_file` — read
  them via `terminal` `cat` or Python `open()` in `execute_code`.
- Contact sheets are the token-efficient path: one `vision_analyze` call covers
  up to 24 shots. Ask it for >= 8 English words per shot so the frame-description gate passes.
- Machine fields (`start`, `end`, `seconds`, `motion`, `seedCuts`, `meta`,
  `manualCuts`) are evidence — editing them by hand trips the boundary gate by design.
- Stage ② needs a Chrome/Chromium binary (`--chrome /usr/bin/google-chrome`, or
  the Playwright chromium under `~/.cache/ms-playwright`); stage ① does not.
- Downstream: `shots.json` is a reference study for storyboard/prompt work (e.g.
  `kanban-video-orchestrator`), or a cut list for `hyperframes`/ffmpeg re-edits.

## Verification

- `node references/video-shots/scripts/selftest.mjs` → "449 项断言全部通过";
  `node references/video-sync/scripts/selftest.mjs` → "122 项断言全部通过".
  Both run without a model, ffmpeg, or network.
- `validate` prints 15 ✅ lines and a summary (`13 镜 / 45.17 秒 / ...`).
- Sabotage check: set one static shot's `camera` to `push-in` and validate —
  the motion gate must fail with "measured frame change is only N".
- Bundled sample: `references/video-shots/examples/demo-shots.json` +
  `demo-track.json` (53 shots, 15 gates green) — `validate` it to sanity-check the toolchain.

## Pitfalls

- Node >= 18 and `ffmpeg`/`ffprobe` on PATH; no npm install ever.
- Long films = dozens of sheets; cut into chapters with ffmpeg first.
- No ASR: dialogue only from burned-in subtitles. Unsubtitled talking heads
  should be `subject`, not `dialogue` — the evidence gate forces the choice.
- Scene detection does not see dissolves; place the cut at the dissolve midpoint
  and set `transitionIn: "dissolve"`.
- Frame descriptions must be >= 8 English words (>= 12 CJK chars), unique across
  shots, and not start with "This shot".
- `render --html` inlines `report.css`/`report.js` from the scripts dir — keep
  the three files together.
