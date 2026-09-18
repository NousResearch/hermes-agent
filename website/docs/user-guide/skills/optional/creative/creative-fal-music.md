---
title: "Fal Music — Generate music and sound effects via fal.ai hosted models"
sidebar_label: "Fal Music"
description: "Generate music and sound effects via fal.ai hosted models"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Fal Music

Generate music and sound effects via fal.ai hosted models.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/fal-music` |
| Path | `optional-skills/creative/fal-music` |
| Version | `1.0.0` |
| Author | Hermes Agent (agent-tools-scout) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `music`, `audio`, `sound-effects`, `fal`, `elevenlabs`, `minimax`, `lyria`, `stable-audio`, `ace-step`, `generation` |
| Related skills | [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music), [`heartmula`](/docs/user-guide/skills/optional/creative/creative-heartmula), [`audiocraft-audio-generation`](/docs/user-guide/skills/optional/creative/creative-audiocraft-audio-generation) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# fal-music — hosted music and sound-effect generation

One script, one key, eight hosted models. Use this when the user wants an actual
audio file (a track, a jingle, a full song with lyrics, a sound effect) and has no
GPU for `heartmula` / `audiocraft-audio-generation`. Cloud generation takes
15–120 s and costs cents; no local model download.

## When to Use

- "Make me a lofi track / cinematic score / synthpop song with these lyrics"
- Background music or SFX for a video (`hyperframes`, `kanban-video-orchestrator` pipelines)
- A quick sound effect ("door creak", "8-bit coin pickup", "rain loop")
- The user has a fal.ai account (`FAL_KEY`) — the same key the `image_generate`
  and `video_generate` FAL backends use

Do NOT use for text-to-speech (use the `text_to_speech` tool) or for local/offline
generation (use `heartmula` or `audiocraft-audio-generation`).

## Models

| id | endpoint | length | lyrics | pick it when |
|----|----------|--------|--------|--------------|
| `elevenlabs-music` (default) | `elevenlabs/music/v2.5` | 10–300 s | no (prompt describes vocals) | best general quality, vocal or instrumental; $0.60 / output minute, rounded up |
| `minimax-music-3` | `minimax/music-3` | 1–300 s | yes, `[verse]`/`[chorus]` tags | full songs with your lyrics, WAV out, seedable |
| `minimax-music-2.6` | `fal-ai/minimax-music/v2.6` | model-decided | yes | cheaper MiniMax tier, `--instrumental` flag |
| `lyria3` / `lyria3-pro` | `fal-ai/lyria3[/pro]` | ~30 s fixed | no | Google Lyria clips; prompt only |
| `stable-audio-3` | `fal-ai/stable-audio-3/medium/text-to-audio` | 1–380 s | no | long instrumentals, licensed training data, `--negative-prompt`, `--seed` |
| `ace-step` | `fal-ai/ace-step` | 5–240 s | yes | cheapest ($0.0002 / s ≈ 83 min per $1); prompt is a comma-separated tag list |
| `elevenlabs-sfx` | `fal-ai/elevenlabs/sound-effects/v2` | 0.5–22 s | n/a | sound effects, `--loop` for seamless loops |

Prices come from fal's model pages at authoring time and change; `--dry-run` never
bills. Only elevenlabs-music and ace-step publish a per-unit price in fal's model
API; the rest bill per fal's dashboard rate card.

## Procedure

1. Locate the script (skills install under `~/.hermes/skills/`; profiles under
   `~/.hermes/profiles/<name>/skills/`):
   ```bash
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/fal-music/SKILL.md' 2>/dev/null | head -1)")
   [ -f "$SKILL_DIR/SKILL.md" ] || echo "fal-music is not installed: hermes skills install official/creative/fal-music"
   ```
2. Confirm prerequisites once per machine. Use the interpreter Hermes runs on
   (`~/.hermes/hermes-agent/.venv/bin/python` on a source install; a bare `python`
   on a PEP 668 host may be the wrong one):
   ```bash
   python -c "import fal_client" 2>/dev/null || pip install fal-client==0.13.1
   test -n "$FAL_KEY" || echo "FAL_KEY missing — https://fal.ai/dashboard/keys"
   ```
   Hermes sessions load `~/.hermes/.env`, so a `FAL_KEY=` line there is enough;
   a plain shell needs `set -a; . ~/.hermes/.env; set +a` first.
3. Turn the request into a style prompt. Genre, mood, tempo/BPM, instrumentation,
   vocal type, era, and "instrumental" when no vocals are wanted. The
   `songwriting-and-ai-music` skill's craft (structure tags, style painting)
   transfers directly to `--lyrics` / `--prompt` here.
4. Preview the payload, then generate:
   ```bash
   python "$SKILL_DIR/scripts/fal_music.py" --model elevenlabs-music \
     --prompt "warm lofi hip hop, dusty drums, mellow Rhodes, vinyl crackle, 80 BPM" \
     --duration 45 --instrumental --dry-run
   python "$SKILL_DIR/scripts/fal_music.py" --model elevenlabs-music \
     --prompt "..." --duration 45 --instrumental -o /tmp/lofi.mp3
   ```
   Lyrics-driven song:
   ```bash
   python "$SKILL_DIR/scripts/fal_music.py" --model minimax-music-3 \
     --prompt "upbeat synthpop, bright female vocals, 120 BPM" \
     --lyrics $'[verse]\nNeon lights on empty streets\n[chorus]\nRun with me tonight' \
     --duration 90 -o /tmp/song.wav
   ```
   Sound effect:
   ```bash
   python "$SKILL_DIR/scripts/fal_music.py" --model elevenlabs-sfx \
     --prompt "heavy wooden door creaks open slowly, stone hallway reverb" --duration 4 -o /tmp/door.mp3
   ```
5. The script prints JSON with `output`, `url`, and any extra fields the endpoint
   returned (`seed`, `duration`, `lyrics`). Return the file with `MEDIA:/tmp/lofi.mp3`
   and mention the seed when the model returned one so the user can iterate.
6. Iterate by changing one thing at a time (prompt wording, duration, seed). For a
   longer piece than the model allows, generate sections and concatenate with
   ffmpeg (`ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp3`).

## Pitfalls

- **`--duration` is clamped, never rejected** — 500 s on `minimax-music-3` becomes
  300 s. Models with no length knob (`lyria3`, `minimax-music-2.6`) drop it and
  the script prints a `note: ... ignored` line on stderr; the `--list` table shows
  each model's range. `elevenlabs-sfx` without `--duration` lets the model infer
  length from the prompt.
- **`minimax-music-3` requires lyrics**; the script sends `[inst]` when you pass
  none, which makes the track instrumental. Do not pass prose as lyrics — use the
  bracketed structure tags.
- **`ace-step`'s prompt is a tag list** (`"lofi, chill, jazzy piano, 85 bpm"`), not
  a sentence. Sentences still run but steer poorly.
- **ElevenLabs music bills per started minute** — 61 s costs two minutes. Round
  durations down to the minute boundary when cost matters.
- **Copyrighted references** ("sounds like &lt;artist>", song titles) are refused or
  stripped by ElevenLabs and MiniMax; describe the style instead.
- **`fal_client` errors with "User is locked. Reason: Exhausted balance"** means
  the fal account needs a top-up, not a code problem. Surface the message.
- The script only forwards keys each endpoint's OpenAPI schema declares; `--seed`
  on a model without a seed knob is dropped with a stderr note, not an error.

## Verification

- `python fal_music.py --list` prints all eight models.
- `--dry-run` shows the exact endpoint + payload before any billable call.
- After generation, `ffprobe -hide_banner /tmp/out.mp3` reports a duration close to
  the requested one (ElevenLabs/MiniMax may land ±10 %).
