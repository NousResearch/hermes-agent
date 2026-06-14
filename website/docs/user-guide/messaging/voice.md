---
title: "Voice"
description: "Call your Hermes Agent over WebRTC — real-time, full-duplex voice"
---

# Voice Platform

The voice platform turns your agent into something you can **call**. It joins a
[Daily](https://daily.co) (WebRTC) room as a participant, listens with
[Deepgram Flux](https://developers.deepgram.com/docs/flux) (streaming
speech-to-text with model-integrated turn detection), thinks with your agent,
and speaks back with [Cartesia](https://cartesia.ai) (streaming text-to-speech)
— all in real time, with **in-process barge-in**: interrupting the agent cancels
its actual generation, not just the audio.

Unlike an external voice orchestrator that wraps the model as a black box, this
loop runs *inside* the agent session — so a barge-in stops the LLM mid-thought,
and voice shares the same agent, tools, and memory as every other channel.

## What you need

Install the optional dependencies:

```bash
uv sync --extra voice-platform   # daily-python + websockets
```

Set four keys in your `.env` (all three services have free tiers):

| Key | Service | Used for |
| --- | --- | --- |
| `DAILY_API_KEY` | [Daily](https://dashboard.daily.co) | WebRTC transport (the agent mints its own room) |
| `DEEPGRAM_API_KEY` | [Deepgram](https://console.deepgram.com) | Flux streaming STT + turn detection |
| `CARTESIA_API_KEY` | [Cartesia](https://play.cartesia.ai) | streaming TTS |
| a voice model key | your provider | generation (see `voice_model:` below) |

## A faster model for voice (optional)

Real-time voice is latency-critical, so you can point voice turns at a fast,
non-reasoning model while your main model stays whatever you normally use. Add a
top-level `voice_model:` block to `config.yaml` — a sibling to `model:` and
`fallback_model:`:

```yaml
voice_model:
  provider: groq
  model: meta-llama/llama-4-scout-17b-16e-instruct
```

Leave it out (or set `provider: auto`) and voice rides your main model.

## Configuration

The plugin works with no extra config. To tune it, add a `platforms.voice.extra`
block to `config.yaml`:

```yaml
platforms:
  voice:
    extra:
      cartesia_voice_id: ""        # Cartesia voice id (or set CARTESIA_VOICE_ID)
      cartesia_model: sonic-3.5
      eot_threshold: 0.7           # Flux end-of-turn confidence (higher = waits longer)
      eager_eot_threshold: 0.5     # Flux eager/speculative end-of-turn
      allow_interruptions: true    # barge-in when the caller starts speaking
      tts_speed: ""                # optional Cartesia speed (e.g. 1.0)
      greeting_prompt: ""          # optional: how the agent opens the call
```

## Making a call

With the keys set and the plugin enabled, start the gateway. The agent creates
its own private Daily room, joins it, and **prints the join URL on startup** —
watch the gateway output for:

```
voice: standalone call ready — open this URL to talk to your agent (keep the token private):
    https://<you>.daily.co/<room>?t=<token>
```

Open that URL in a browser, allow the microphone, and talk. The room is private,
so the URL carries a short-lived owner **token** — that token is the join
permission, so share the URL only with people you want to reach your agent. The
room is short-lived (~1 hour), and the agent ends the call automatically when the
caller leaves, the room expires, or a maximum-duration cap is reached.

## Notes

- One active call per agent process (the WebRTC virtual audio devices are
  process-level singletons).
- Per-call cost telemetry (STT seconds, timing legs) is written to
  `voice-telemetry.jsonl` on the agent's data volume.
