---
sidebar_position: 11
title: "Realtime Voice"
description: "Speech-to-speech conversations with Hermes over a realtime voice provider — hermes realtime"
---

# Realtime Voice

`hermes realtime` opens a live, bidirectional voice session: your microphone streams
to a realtime speech-to-speech model, its voice streams back to your speaker, you can
interrupt it mid-sentence, and it can call Hermes tools while you talk.

This is different from [Voice Mode](/user-guide/features/voice-mode), which records a
turn, transcribes it, runs the normal text agent, and reads the answer back with TTS.
In a realtime session the **provider's model is the one talking**; Hermes owns the tools,
the approval prompts, and the session around it. Voice Mode stays the right choice when
you want your configured chat model to do the thinking.

## Requirements

- A realtime voice provider. The bundled backend is **OpenAI Realtime** (`gpt-realtime`
  family) and needs `OPENAI_API_KEY` (or `VOICE_TOOLS_OPENAI_KEY`) in `~/.hermes/.env`,
  or an `openai-api` credential from `hermes auth add`.
- A microphone, a speaker, and the `sounddevice` package (the same audio dependency as
  Voice Mode: `pip install sounddevice numpy`, plus PortAudio on Linux —
  `apt install libportaudio2`).

Check what is registered and whether it is configured:

```bash
hermes realtime --list
```

```text
openai           ready        OpenAI Realtime (default model gpt-realtime-2.1)
```

## Usage

```bash
hermes realtime                          # OpenAI, default model + voice, hermes-cli tools
hermes realtime --voice cedar            # pick a provider voice
hermes realtime --model gpt-realtime-mini
hermes realtime --toolset hermes-cli     # which Hermes toolset the model may call
hermes realtime --no-tools               # conversation only
hermes realtime --tool-timeout 20        # per-call tool timeout in seconds (default 60)
hermes realtime --provider gemini        # any provider a plugin registered
```

Speak when you see `realtime: connected`. Transcripts print as `You:` and `Hermes:`
lines; tool calls print as `→ tool: <name>`. **Ctrl+C** hangs up.

:::tip Use headphones
The microphone stays open while the assistant speaks — that is what makes interrupting
work. With open speakers the model can hear itself and cut its own answer short.
:::

### Interrupting

Start talking over the assistant. The provider's server-side turn detection reports the
interruption; Hermes stops playback immediately, cancels the in-flight response, tells the
provider exactly how many milliseconds of the answer you actually heard (so the model's
memory of what it said matches yours), and cancels any tool calls still running for that
turn. Providers that cannot truncate on the wire get the local stop only — nothing is faked.

### Tools and approvals

Tools are the same registry entries a chat session gets (`--toolset` picks the set, default
`hermes-cli`). A call runs through Hermes' normal dispatch — hooks, budgets, and the
dangerous-command approval prompt for `terminal` — on a background thread, so the audio
never stalls. Approval prompts appear in the terminal; answer them with the keyboard. Every
call is bounded by `--tool-timeout`: a slow tool answers the model with an honest "still
running" result instead of holding the conversation. Each call is answered exactly once,
even if the provider redelivers it.

## Plugins and other providers

Any plugin can register a provider with `ctx.register_realtime_voice_provider(provider)`;
`hermes realtime --provider <name>` then drives it with the same orchestrator. See
[Register a realtime voice provider](/developer-guide/plugins#register-a-realtime-voice-provider).

## Limitations

- Realtime transcripts are not saved to session history or memory yet.
- MCP servers are not started for a realtime session; only registry tools are exposed.
- Reconnecting with a provider's resumption handle is not implemented; a dropped connection
  ends the session with an error.
