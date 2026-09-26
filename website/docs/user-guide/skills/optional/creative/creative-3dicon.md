---
title: "3Dicon — Animate a prompt or icon into a looping transparent WebP"
sidebar_label: "3Dicon"
description: "Animate a prompt or icon into a looping transparent WebP"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 3Dicon

Animate a prompt or icon into a looping transparent WebP.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/3dicon` |
| Path | `optional-skills/creative/3dicon` |
| Version | `1.0.0` |
| Author | samyost1 (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `icon`, `animation`, `webp`, `alpha`, `image-generation`, `video-generation`, `seedance`, `motion` |
| Related skills | [`dream-loop`](../../optional/creative/creative-dream-loop.md), [`pixel-art`](../../optional/creative/creative-pixel-art.md), [`hyperframes`](../../optional/creative/creative-hyperframes.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# 3dicon Skill

Ported from samyost1/3dicon (MIT) at 9539d3a; iconloop/ vendored verbatim.

Turns a prompt (or an existing still) into a seamlessly looping 3D icon with
real soft alpha: one still from an image model, image-to-video with the same
frame as first and last frame so the loop closes, exact-unpremultiply matting
against a known grey backing, then an animated WebP that `expo-image`, Chrome
and Safari render natively. It does **not** do camera moves, objects entering
or leaving frame, transparent MP4, or reduced-motion — see `references/limitations.md`.

## When to Use

- "Animate this app icon", "make a 3D icon loop", "transparent animated asset".
- Add motion to flat icon art the user already has (feed it in with `--still`).
- Not for video editing, long clips, or anything with a moving camera.

## Prerequisites

- `ffmpeg` on PATH (the CLI refuses to start without it).
- Python deps from the skill's own `scripts/requirements.txt` (pillow, numpy,
  rembg, onnxruntime; everything else is stdlib). The first `matte` run
  downloads a ~180 MB rembg model (`isnet-general-use`).
- API keys, read from the environment — Hermes already exports these when
  configured: `OPENAI_API_KEY` (GPT Image), `GOOGLE_API_KEY` (Gemini image
  models), `OPENROUTER_API_KEY` (unified image API **and** the default video
  path — one OpenRouter key covers the whole pipeline). `REPLICATE_API_TOKEN`
  only for `--via replicate`. `ICONLOOP_IMAGE_BACKEND` = `openai` (default) |
  `gemini` | `openrouter` selects the still backend. Optional overrides are
  listed in `scripts/env.example`; copy it to `scripts/.env` only if you would
  rather not export variables (env vars always win; nothing is ever printed).
- Check the keys are present *before* running — `animate` costs money, and a
  run that dies halfway has already spent it.
- `HERMES_SKILL_DIR` is this skill's directory; if unset, use the directory
  containing this SKILL.md. Install: `hermes skills install official/creative/3dicon`.

## How to Run

One-time setup (creates a private venv beside the vendored package):

```bash
uv venv "${HERMES_SKILL_DIR}/scripts/.venv"
uv pip install --python "${HERMES_SKILL_DIR}/scripts/.venv/bin/python" -r "${HERMES_SKILL_DIR}/scripts/requirements.txt"
```

Every stage is run from the `scripts/` directory with an **absolute** `--out`
working directory (the package is not installed, so `cd` is required):

```bash
cd "${HERMES_SKILL_DIR}/scripts"
.venv/bin/python -m iconloop --out "$WORK" still   --prompt "a 3D stopwatch, sage green and cream, soft matte plastic"
.venv/bin/python -m iconloop --out "$WORK" animate --strategy event --emit --motion "the stopwatch rocks gently while its hand sweeps clockwise"
.venv/bin/python -m iconloop --out "$WORK" matte
.venv/bin/python -m iconloop --out "$WORK" encode  --sweep
.venv/bin/python -m iconloop --out "$WORK" encode  --size 288 --name timer
.venv/bin/python -m iconloop --out "$WORK" verify
```

`run` is just `still` plus the reminder to stop and ask. The stages are
separate because each one is a place to look before spending the next thing.
`animate --dry-run` composes the full motion prompt and spends nothing.

**Alternative still source.** Hermes' native `image_generate` tool can make the
still instead of the `still` stage. `still` has no image-input flag; the hook
is on the next stage: `animate --still /abs/path/art.png`. The PNG must be a
centred object on a **transparent** background (the pipeline flattens it onto
its grey backing itself); a baked-in background defeats the matting. Ask the
image model for the same look the `still` stage bakes in (stylised 3D icon,
soft matte, transparent background, no text/shadow/ground plane).

## Quick Reference

| Stage | Reads → writes | Cost (upstream's numbers) |
|---|---|---|
| `still` / `run` | prompt → `still.png` | ≈ $0.13, seconds |
| `animate` | `still.png` → `sent_to_kling.png`, `render.mp4` | rest of ≈ $0.48 total, ~4 min |
| `matte` | `render.mp4` → `src/`, `master/*.png` | free, local CPU, slow |
| `encode` | `master/` → `<name>.webp` (+`.webm`, `.mp4` opt.) | free, instant |
| `verify` | `master/`, `*.webp` → report, `contact_sheet.png`; exit 1 on fail | free |

Full flag list with defaults: `references/cli-flags.md`.

## Procedure

1. **Always stop after the still.** Generate exactly ONE still (`still` or
   `run`). Never generate several takes to choose from; `--variants` exists but
   stays off unless the user asks to compare. The still is 13 cents and the
   full run is 48 cents and four minutes, and every later stage inherits the
   still's object, colour and weight — animating unapproved art means paying twice.
2. **Show it and ask.** Report only the file path (on Desktop/Telegram the
   image renders inline); use `vision_analyze` yourself only if you need to
   check it. Say exactly:
   > **Still ready.** *(the image)* Happy with it, or change something?
   Do not explain the prompt, list what you did, or describe the image. If they
   are not happy, adjust the prompt and generate one replacement, not a grid.
3. **Propose the motion and stop again.** Approval of the still is not approval
   of the motion — the motion is the second decision and the one that costs
   four minutes. Propose exactly one, in this shape:
   > **Motion** — *(one plain sentence: what physically happens)*
   > `event` · `lively` · `--emit`
   > Agree, or describe the motion you want?
   One sentence and the flags. No reasoning, no alternatives, no flag glossary.
4. **Translate their words into flags** if they describe their own motion:
   - *what does this object do when left alone?* → `--strategy`
   - *how much should the object itself move?* → `--energy`
   - *would the action throw something off?* → `--emit`
   - *what literally happens to the material?* → `--motion`
   Show the translated proposal in the same format and confirm before running.
   Camera moves, things entering the frame, the object travelling away: say
   in one line that it is not supported and offer the nearest thing that is.
   Use `animate --dry-run` to see what the flags expand to; never paste it.

   | left alone, the object… | `--strategy` | what you get |
   |---|---|---|
   | moves by itself — flows, burns, breathes, ticks | `native` | continuous motion from its own physics |
   | does nothing; inert until used | `event` | performs its function once, then rests |
   | does nothing, but a part is loose, hinged or light | `part` | body anchors, one small piece moves |
   | does nothing and has no moving parts | `surface` | light or material travels across a fixed form |

   Most requested objects are **inert** — reach for `event` first. Add `--emit`
   when the action would realistically throw off a fragment, droplet, spark or
   glint; at icon size that emission is often the whole reason it reads.
   `event` and `part` also allow the object to be temporarily altered (opened,
   split, filled) as long as it returns to its opening state.

   | `--energy` | the object |
   |---|---|
   | `still` | holds completely rigid |
   | `calm` | leans, settles, breathes a little |
   | `lively` | squashes, tilts, recoils, shakes, springs back |
   | `playful` | anticipates, overshoots, wobbles, hops in place |

   Defaults follow the strategy — `native`/`event` → `lively`, `part` → `calm`,
   `surface` → `still`; explicit `--energy` wins. Lifeless result: raise energy
   first. Only whole-object rotation and drifting are forbidden at every level.
   Small and many beats large and one; repeated elements move out of step;
   everything that goes out comes back; animate the effect, not its cause.
   `--motion` states the specific action in plain physical language — what
   happens to the material, not how the animation should feel.
5. **Only on a yes**, run `animate`, then `matte`, `encode --sweep`, `encode`
   with the chosen size/fps, then `verify`.
6. **Report the result** in one line plus the file:
   > **Done** — `flame.webp` · 857 KB · 122f @ 24fps *(contact sheet path)*
   If `verify` fails, one line on what failed and one proposal:
   > **Motion too weak** — step 0.9, needs 1.0. Retry as `event` with `--emit`?
   **Never** paste raw CLI output, stage-by-stage progress, the composed motion
   prompt, or a pipeline summary. A successful stage's output is its file.

## Pitfalls

1. **No human present (cron, headless, batch): generate the still and STOP.**
   Never run `animate` without an explicit yes in the conversation; leave the
   `still.png` path and the proposed motion for the user to approve later.
2. **Resolution is the quality lever, not `--quality`.** Quality 50→90 cut
   colour error from 2.96 to 1.84/255 and added 500 KB — invisible. Raise
   `--size` before `--quality`, never above the 512 px master (only upscales).
3. **Encode at the source frame rate** (24 fps). Downsampling to 8 fps for size
   produces judder that looks like bad rendering or matting. `--sweep` prints
   what each rate costs; choose with the numbers in front of you.
4. **The last frame is set to the first frame** in `kling.py` on both
   backends. That closes the loop; do not remove it and cross-fade instead.
5. **Video model:** default is Seedance 2.0 via OpenRouter. `--model` accepts
   Seedance 2.5, Veo 3.1, Hailuo 3, Wan 2.7 (all take first+last frame). Not
   Kling 3.0 — 2500-char prompt cap, the composed clause is longer, it rejects.
   `--via replicate` is the fallback and stops at Kling 2.5.
6. **The backing colour is load-bearing** (mid-grey 158,158,158). The matte
   solves `C = (F - (1-a)*BG)/a` exactly — edge colour error 26.9 → 15.9. Do
   not switch to chroma green/magenta; they spill onto glossy edges.
7. **Believe the verifier, not the contact sheet.** A render can look animated
   in stills and be static (1 moving frame of 121). If `verify` says static,
   re-prompt; do not encode it.
8. **Never temporally smooth alpha** — it smears a ghost ring around motion.
9. Keys missing → the CLI exits naming the variable before spending anything;
   fix the env rather than editing `scripts/iconloop/` (vendored verbatim).
10. **An unset key is not a spend guard.** The CLI walks *up* from the cwd
    looking for `.env`/`.env.local` and will find `~/.hermes/.env` (where Hermes
    keeps its keys) when run from an installed skill. Convenient — no `.env`
    to copy — but `env -u OPENAI_API_KEY` does **not** stop a paid call. Treat
    every `still`/`run`/`animate` invocation as paid; only `--dry-run` is free.
11. `--preset camera` crashes with `NameError: ACCENT` at upstream 9539d3a
    (vendored as-is). Use `--strategy part --motion "..."` for a camera instead;
    the other ten presets compose fine.

## Verification

- `cd "${HERMES_SKILL_DIR}/scripts" && .venv/bin/python -m iconloop --help`
  lists `still animate matte encode verify run`.
- `animate --dry-run --strategy event --emit --motion "..."` prints the
  composed clause and spends nothing (needs an existing `still.png`).
- After a full run: `verify` exits 0, and `<name>.webp` opens as an animated
  image with transparent corners in a browser.

## References

- `references/cli-flags.md` — every stage's flags with defaults, presets,
  `--feel` values, env variables; load when translating a request into flags.
- `references/limitations.md` — measured limits (shadow loss, size vs frames,
  no transparent MP4, reduced-motion, rembg model comparison); load when a
  user asks "why does it…" or before promising a format.
