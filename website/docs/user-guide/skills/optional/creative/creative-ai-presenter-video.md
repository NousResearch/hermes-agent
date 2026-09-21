---
title: "Ai Presenter Video — Make a verified AI presenter video from script + image"
sidebar_label: "Ai Presenter Video"
description: "Make a verified AI presenter video from script + image"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ai Presenter Video

Make a verified AI presenter video from script + image.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/ai-presenter-video` |
| Path | `optional-skills/creative/ai-presenter-video` |
| Version | `1.0.0` |
| Author | cclank (https://github.com/cclank/lanshu-create-ai-presenter-video), ported by Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `video`, `presenter`, `avatar`, `lipsync`, `tts`, `captions`, `creative` |
| Related skills | [`hyperframes`](../../optional/creative/creative-hyperframes.md), [`kanban-video-orchestrator`](../../optional/creative/creative-kanban-video-orchestrator.md), [`comfyui`](../../optional/creative/creative-comfyui.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# AI Presenter Video Skill

This skill turns a script and one authorized adult presenter image into a
presenter-led video with deterministic editing and acceptance evidence. It can
resume or repair existing jobs, but it never bypasses image rights, adult
status, upload, voice-cloning, or paid-generation consent.

## When to Use

Use it to create, continue, revise, caption, repair lip sync, or re-export a
presenter-video job. The workflow is provider-neutral: choose only generation
capabilities actually available in the session, and preserve accepted work
rather than regenerating completed stages.

## Prerequisites

- Linux or macOS with `python3`, `ffmpeg`, and `ffprobe`. Final delivery also
  requires Bash, `jq`, and `awk`.
- One presenter image whose rights and adult status the user has confirmed.
- Optional narration through `text_to_speech`, word-timestamp ASR through the
  configured STT tooling or `faster-whisper`, and presenter generation through
  configured video tooling or an authorized avatar/lip-sync endpoint.
- Remote avatar or TTS generation can be billable. Before the first paid call,
  state the uploaded assets, requested seconds, known cost, pilot size, and
  retry ceiling, then obtain explicit approval. Never upload the presenter
  image until `input.remote_upload_approved` is true in `job.json`.

The human author and upstream source remain cclank's
`lanshu-create-ai-presenter-video` (MIT). The reference files preserve the
substantive upstream guidance; the deterministic local scripts use no network
or credentials.

## How to Run

Hermes expands `${HERMES_SKILL_DIR}` to the installed skill directory. Set it
again in each `terminal` call because shell variables do not persist between
tool calls:

```bash
SKILL_DIR="${HERMES_SKILL_DIR}"
```

Start a new job with:

```bash
python3 "$SKILL_DIR/scripts/init_job.py" \
  --job-dir ~/Videos/my-presenter-video \
  --presenter-image /path/to/presenter.png \
  --topic "explain context engineering in one minute" \
  --duration 60 --aspect 9:16 \
  --rights-confirmed --adult-presenter-confirmed
```

Use `--script` for an existing script file. Other flags include
`--voice-sample`, `--supporting-media`, `--width`, `--height`, `--fps`,
`--watermark`, and `--cta`. For an existing job, read `job.json` and its QA
reports, then resume from the earliest unfinished state.

## Quick Reference

- Voice generation maps to `text_to_speech`; presenter generation maps to the
  configured video tooling or an authorized avatar endpoint; word-timestamp
  ASR maps to configured STT tooling or `faster-whisper`.
- Deterministic composition uses `ffmpeg` filtergraphs, or the `hyperframes`
  skill when installed.
- Visual QA uses `vision_analyze` on the contact sheet and sampled frames for
  identity, mouth timing, hands, blinking, and continuity. Numeric checks come
  from the scripts' `ffprobe` output.
- Consent flags `rights_confirmed`, `adult_presenter_confirmed`,
  `remote_upload_approved`, and `voice_clone_approved` live under the `input`
  object. `manual_input_review.*` remains at the job root.
- Defaults are 9:16, 1080×1920, 30 fps, and 45–75 seconds for topic-derived
  videos. Use a stock voice when no authorized sample exists; use a hook,
  two to four beats, and a close; add no music or CTA unless requested.
- Read [references/generation.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/generation.md) for intake, voice,
  provider selection, presenter prompts, and paid generation;
  [references/editing.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/editing.md) for timeline, captions,
  HyperFrames, and exports; and
  [references/qa-recovery.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/qa-recovery.md) for acceptance and
  recovery paths.

## Procedure

### 1. Review the inputs

Actually inspect the presenter image with `vision_analyze` and listen to any
voice sample. Record the findings by updating the `manual_input_review`
booleans in `job.json`:

```bash
python3 - <<'PY'
import json
import os
p = os.path.expanduser("~/Videos/my-presenter-video/job.json")
j = json.load(open(p))
j["manual_input_review"].update(image_viewed=True, single_clear_face=True,
                                image_has_no_unwanted_text=True)
json.dump(j, open(p, "w"), indent=2)
PY
```

### 2. Run the gate

```bash
python3 "$SKILL_DIR/scripts/preflight.py" ~/Videos/my-presenter-video/job.json
```

Proceed locally only when `ok: true`, and perform remote generation only when
`remote_ready: true`. `preflight.py` distinguishes `errors`, which block all
work, from `remote_blockers`, which block only remote generation. It also
updates `job.json` in place with the report path, so re-read the file afterward
instead of editing a stale copy.

### 3. Lock content and audio

Read [references/generation.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/generation.md). Generate the full
narration with `text_to_speech`, ASR-verify it against the script, and record
real durations. Locked audio is the master clock for every later stage.

### 4. Plan and generate the presenter

Follow [references/generation.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/generation.md). Generate a short,
low-cost pilot first; begin the full run only after the pilot passes identity
and mouth-timing review.

### 5. Edit

Follow [references/editing.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/editing.md). Drive the deterministic
timeline from locked audio, and add captions and keyword callouts only after
audio and media are final.

### 6. Finalize delivery

Read [references/qa-recovery.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/ai-presenter-video/references/qa-recovery.md), render, then run:

```bash
bash "$SKILL_DIR/scripts/finalize_delivery.sh" \
  ~/Videos/my-presenter-video/renders/rendered.mp4 \
  ~/Videos/my-presenter-video/outputs my-video
```

The finalizer preserves aspect ratio, runs two-pass loudness normalization
(program approximately −16 LUFS), produces master and share encodes,
decode-verifies both, writes a delivery report JSON, and emits a nine-frame
contact sheet.

### Operating rules

- Confirm image rights, adult status, remote-upload approval, and
  voice-cloning authorization before the relevant remote action.
- Never infer or clone a real person's voice from an image; use an authorized
  sample or a stock TTS voice.
- Lock the complete narration before presenter generation, caption timing, or
  final scene boundaries.
- Mute video sources in the final composition; only the approved narration
  and intentional mix tracks carry audio.
- Preserve provider request bodies and task IDs (minus credentials/expiring
  URLs). Poll interrupted work before resubmitting — avoid double billing.
- Stop after three rejected paid candidates and summarize the failure mode.
- Do not claim completion until the final files fully decode and the contact
  sheet or full playback has been reviewed.

## Pitfalls

- `preflight.py` requires ffprobe; on a bare box install ffmpeg first.
- The consent booleans set by init flags land under `input.*`; editing them
  at the job-json root silently does nothing (preflight keeps blocking).
- `finalize_delivery.sh` needs bash + jq + awk and a fully decodable input —
  a truncated render fails the decode check by design, not by accident.
- Long avatar clips drift: prefer one continuous presenter source sliced on
  the audio timeline over many regenerated chapter clips (identity drift
  across regenerations is the #1 visual-QA failure).
- FAL i2v endpoints cap duration (typically 5–15s); plan chapter-level
  presenter segments accordingly and reuse the pilot's seed/params for
  consistency where the endpoint supports it.

## Verification

Inspect the final contact sheet with `vision_analyze` before claiming
completion. Confirm the master and share files fully decode, compare captions
and lip sync at normal speed, and retain the delivery report.

Hands-on validation from August 2026 covered `init_job.py` producing the
expected `job.json` state machine; `preflight.py` blocking unreviewed input,
becoming `ok: true` after manual review, and keeping `remote_ready: false`
until `input.remote_upload_approved`; and `finalize_delivery.sh` producing a
decode-verified master and share encode, delivery report, and nine-frame
contact sheet from a synthetic five-second 1080×1920 render with exit 0.
