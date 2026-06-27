---
sidebar_position: 11
title: "Wake Word"
description: "Hands-free 'Hey Hermes' wake word — start a voice session by speaking, the 'Hey Siri' way"
---

# Wake Word ("Hey Hermes")

The wake word turns Hermes into a hands-free assistant across the CLI, TUI, and
desktop app: with one setting on, Hermes listens in the background for a spoken
trigger phrase. Say it, and Hermes starts a fresh session, opens the microphone,
captures your command via the normal [voice pipeline](/user-guide/features/voice-mode),
and answers — exactly like "Hey Siri" or "Alexa". Use `surface` to pick which
one listens.

Detection runs **entirely on-device**. The always-on listener only watches for
the wake phrase; no audio leaves your machine until you actually speak a command
to the agent.

## How it works

1. With `wake_word.enabled: true` (or after `/wake on`), a lightweight hotword
   detector listens on your default microphone.
2. When it hears the wake phrase it pauses itself (freeing the mic), starts a new
   session, and records one utterance with voice mode's silence detection.
3. Your speech is transcribed and sent to the agent. After it replies, the
   listener resumes automatically and waits for the next wake word.

It is **off by default** — nothing listens until you turn it on.

## Engines

| Engine | Cost | API key | Notes |
|--------|------|---------|-------|
| **openWakeWord** (default) | Free | None | Local ONNX models. Ships with `hey_jarvis`, `alexa`, `hey_mycroft`, … |
| **Porcupine** | Free tier / paid | `PORCUPINE_ACCESS_KEY` | Picovoice engine; built-in keywords + custom `.ppn` files |

Both are lazy-installed the first time you enable the wake word. To install ahead
of time:

```bash
uv pip install 'hermes-agent[wake]'   # or: pip install 'hermes-agent[wake]'
```

## Quick start

```bash
# In an interactive `hermes` session:
/wake on        # start listening (installs the engine on first use)
/wake status    # show phrase, provider, and state
/wake off       # stop listening
```

Or enable it permanently in `~/.hermes/config.yaml`:

```yaml
wake_word:
  enabled: true
```

## Configuration

```yaml
wake_word:
  enabled: false
  surface: auto               # which surface owns the listener: "auto" | "cli" | "tui" | "gui"
  provider: openwakeword      # "openwakeword" (free, local) | "porcupine"
  phrase: "hey jarvis"        # cosmetic label only — detection is keyed by the model/keyword below
  sensitivity: 0.5            # 0.0-1.0 — raise to reduce false triggers
  start_new_session: true     # start a fresh session on wake vs. continue the current one
  openwakeword:
    model: hey_jarvis         # built-in name OR path to a custom .onnx/.tflite
    inference_framework: onnx # "onnx" | "tflite"
  porcupine:
    keyword: jarvis           # built-in keyword OR path to a custom .ppn
```

`sensitivity`, `phrase`, and `start_new_session` apply to both engines. The
`openwakeword` and `porcupine` blocks select the actual detection model.

### Surfaces (CLI, TUI, GUI)

The wake word works in all three Hermes surfaces, and `surface` picks which one
owns the listener and opens the new session when it fires:

| `surface` | Behavior |
|-----------|----------|
| `auto` (default) | Whichever surface you launch arms the listener. |
| `cli` | Only the classic `hermes` CLI. |
| `tui` | Only `hermes --tui`. |
| `gui` | Only the desktop app. |

The detector is on-device and single-mic, so only one surface listens at a time
— `surface` is how you pin it. The TUI and desktop GUI share the same Python
backend (`tui_gateway`), which runs the detector server-side and yields the mic
to voice capture while a command records.

## Using a real "Hey Hermes"

The bundled openWakeWord models do **not** include "hey hermes" — `hey_jarvis`
is the free, instantly-working default. To detect the literal phrase you supply
your own model and point the config at it:

### Option A — openWakeWord (free)

Train a custom model (≈75–90 min on a free/Colab GPU), then drop the `.onnx`
file somewhere and reference it:

```yaml
wake_word:
  enabled: true
  provider: openwakeword
  phrase: "hey hermes"
  openwakeword:
    model: ~/.hermes/wakewords/hey_hermes.onnx
```

Training references:

- openWakeWord — <https://github.com/dscripka/openWakeWord>
- 2026 training Colab — <https://github.com/alfiedennen/openwakeword-colab-2026>

:::tip Pick a distinctive phrase
Wake phrases that don't collide with everyday speech generalize best. Two
syllables with an uncommon word ("hermes" qualifies) beat common words like
"hello" or "stop".
:::

### Option B — Porcupine (custom keyword in seconds)

Create a "Hey Hermes" keyword in the [Picovoice Console](https://console.picovoice.ai/),
download the `.ppn`, and:

```yaml
wake_word:
  enabled: true
  provider: porcupine
  phrase: "hey hermes"
  porcupine:
    keyword: ~/.hermes/wakewords/hey_hermes.ppn
```

Set your access key in `~/.hermes/.env`:

```bash
PORCUPINE_ACCESS_KEY=your-key-here
```

## Requirements

- A working microphone and the `sounddevice` + `numpy` audio stack (shared with
  voice mode).
- An STT provider for transcribing the spoken command — local `faster-whisper`
  works out of the box; see [Voice Mode](/user-guide/features/voice-mode) for the
  full provider list.
- The wake engine deps (auto-installed, or `hermes-agent[wake]`).

`/wake status` reports exactly what's missing if the listener won't start.

## Notes & limits

- **Local surfaces only.** The wake word runs in the CLI, TUI, and desktop GUI —
  wherever a local microphone is available. It does not run in the messaging
  gateway (Telegram, Discord, …), which has no mic.
- **One mic at a time.** The detector releases the microphone while a command is
  recording and reclaims it once the turn ends, so it won't fight voice capture.
- **Privacy.** Hotword detection is local. Set `sensitivity` higher if you get
  false triggers, lower if it misses you.
