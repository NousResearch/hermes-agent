---
sidebar_label: "Voice Calling"
title: "Voice Calling Plugin"
description: "Guarded outbound and inbound phone calls through Twilio and a FastAPI voice service"
---

# Voice Calling Plugin

The bundled `voice_call` plugin lets Hermes place and manage phone calls through a separate FastAPI service.

It exposes one model tool, `voice_call`, with four actions:

- `call(to, purpose, context, escalation_policy)`
- `hangup(call_id)`
- `transfer_to_jason(call_id)`
- `get_transcript(call_id)`

The tool is disabled until `VOICE_CALL_SERVICE_URL` or `voice_call.service_url` is configured. Twilio credentials are checked by the service before a real outbound call is placed.

## Safety rules

`voice_call` is intentionally strict:

- `call` requires a destination, explicit `purpose`, and recipient-specific `context`.
- `escalation_policy` must be one of:
  - `no_escalation`
  - `transfer_on_request`
  - `transfer_if_blocked`
  - `take_message`
- Phone numbers are lightly normalized and redacted in tool output.
- `voice_call.allowed_prefixes` can restrict destinations, for example to `+1`.
- `voice_call.blocked_prefixes` can deny destinations before the service is contacted.
- Secrets stay in environment variables. Caller identity and behavior settings can live in `config.yaml`.

## Configuration

Add non-secret behavior settings to `config.yaml`:

```yaml
voice_call:
  service_url: "https://voice.example.com"
  timeout_seconds: 8
  allowed_prefixes: ["+1"]
  blocked_prefixes: []
  caller_profile:
    on_behalf_of: "Jason Lai"
    assistant_name: "Hermes"
    callback_number: "+1..."
    disclosure: "This is Hermes, an AI assistant calling on behalf of Jason Lai."
    transfer_number: "+1..."
```

Environment variables:

```bash
# Required for Hermes tool availability
VOICE_CALL_SERVICE_URL=https://voice.example.com

# Required by the FastAPI service for real outbound Twilio calls
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
VOICE_CALL_PUBLIC_BASE_URL=https://voice.example.com

# Optional policy/profile overrides
VOICE_CALL_ALLOWED_PREFIXES=+1
VOICE_CALL_BLOCKED_PREFIXES=+1900,+1976
VOICE_CALL_ON_BEHALF_OF="Jason Lai"
VOICE_CALL_ASSISTANT_NAME=Hermes
VOICE_CALL_CALLBACK_NUMBER=+1...
VOICE_CALL_TRANSFER_NUMBER=+1...
VOICE_CALL_DISCLOSURE="This is Hermes, an AI assistant calling on behalf of Jason Lai."
```

## Run the service

```bash
uvicorn plugins.voice_call.service:app --host 0.0.0.0 --port 8765
```

Point Twilio webhooks at the public URL:

- Outbound/inbound TwiML: `/twilio/voice/inbound`
- Media Stream WebSocket: `/twilio/media-stream`
- Status callback: `/twilio/status`
- SMS webhook: `/sms/inbound`

Hermes initiates outbound calls by posting JSON to `/twilio/voice/outbound`. If Twilio credentials or `VOICE_CALL_PUBLIC_BASE_URL` are missing, the service returns a `dry_run: true` call record and does not dial.

## Provider seams

The MVP service returns TwiML, accepts Twilio Media Streams, and keeps in-memory call transcripts. The WebSocket route is async and ready for streaming provider implementations:

- STT: Groq, Deepgram, or OpenAI
- TTS: ElevenLabs first, Kokoro optional
- Agent backend: Hermes API or local `hermes chat`

Production deployments should replace the in-memory `CallStore` if transcripts need to survive service restarts.
