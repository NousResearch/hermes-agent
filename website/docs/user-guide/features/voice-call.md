---
title: Voice Calls
description: Make and receive real phone calls through Telnyx (default), Twilio, or Plivo — outbound notifications, two-way conversations, an inbound allowlist, and optional realtime speech-to-speech.
sidebar_label: Voice Calls
---

# Voice Calls

The `voice_call` platform plugin lets Hermes place and receive real phone
calls (PSTN) through a carrier API. It supports:

- **Outbound calls** — the agent dials a number via the `voice_call` tool
  (or you do, with `hermes voicecall call ...` / `/voicecall call ...`), in
  `notify` mode (speak a message, hang up) or `conversation` mode (two-way
  dialog driven by the agent).
- **Inbound calls** — people can call your number and talk to Hermes,
  gated by an allowlist.
- **Providers** — Telnyx (default), Twilio, Plivo, and a credential-free
  `mock` provider for trying everything locally.

## Why the gateway must be running

Inbound calls arrive as carrier webhooks. The plugin runs a webhook
server for the **gateway's lifetime** — it starts with `hermes gateway run`
and stops on shutdown. If the gateway isn't running, outbound tool calls
fail with a clear error and inbound calls won't reach Hermes.

## Quick start with the mock provider

No credentials, no network — calls are simulated:

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    voice_call:
      enabled: true
      extra:
        provider: mock
        from_number: "+15555550000"
        inbound_policy: allowlist
        allow_from: ["+15555550001"]
```

```bash
hermes gateway run
hermes voicecall call --to +15555550001 --message "Hello from Hermes"
hermes voicecall status
```

## Telnyx setup (default provider)

1. In the [Telnyx portal](https://portal.telnyx.com), create a **Call
   Control application**, attach a phone number to it, and note the
   **connection ID**. Copy the application's **public key** (used to verify
   webhook signatures).
2. Put credentials in `~/.hermes/.env`:

   ```text
   TELNYX_API_KEY=KEY_xxxxxxxx
   TELNYX_CONNECTION_ID=1234567890
   TELNYX_PUBLIC_KEY=BASE64_PUBLIC_KEY
   ```

3. Configure the platform. Carriers must be able to reach your webhook, so
   set an explicit `public_url` (your reverse proxy / static tunnel) or let
   the plugin manage an ngrok tunnel:

   ```yaml
   gateway:
     platforms:
       voice_call:
         enabled: true
         extra:
           provider: telnyx
           from_number: "+15551234567"     # your Telnyx number
           inbound_policy: allowlist
           allow_from: ["+15557654321"]    # numbers allowed to call in
           tunnel:
             provider: ngrok               # or set public_url: https://...
   ```

   With ngrok, add `NGROK_AUTHTOKEN` to `.env` (and optionally a reserved
   `NGROK_DOMAIN`). Point your Telnyx application's webhook URL at
   `<public-url>/voice/webhook`.

4. Verify and run:

   ```bash
   hermes voicecall doctor
   hermes gateway run
   hermes voicecall call --to +1555... --message "Hello from Hermes" --mode notify
   ```

Twilio (`TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN`) and Plivo
(`PLIVO_AUTH_ID` / `PLIVO_AUTH_TOKEN`) work the same way with
`provider: twilio` / `provider: plivo`.

## How conversations work

When a caller speaks, the transcript becomes a normal Hermes message from
their phone number — the agent has its usual session, memory, and tools.
The agent's reply is spoken back on the call (markdown and URLs are
stripped automatically; the agent is prompted to answer in short spoken
sentences).

Sessions are keyed `per-phone` by default — the same caller keeps one
ongoing conversation across calls. Set `session_scope: per-call` for a
fresh session on every call.

## Control surfaces

- **Model tool** `voice_call`: `initiate_call`, `continue_call`,
  `speak_to_user`, `send_dtmf`, `end_call`, `get_status`.
- **CLI**: `hermes voicecall status [--json] | call | speak | continue |
  dtmf | end | doctor | tail [-f]`. `call` defaults to **conversation**
  mode (pass `--mode notify` for speak-and-hang-up), and `-t/--to` is
  optional when `to_number` is configured — `hermes voicecall call -m
  "call me when you see this"` just works.
- **Slash command**: `/voicecall status`, `/voicecall call --to +1555...
  --message "..."`, `/voicecall end --call-id ID`.

## Configuration reference

```yaml
gateway:
  platforms:
    voice_call:
      enabled: true
      extra:
        provider: telnyx              # mock | telnyx | twilio | plivo
        from_number: "+15551234567"
        to_number: "+15557654321"     # optional default destination for outbound calls
        session_scope: per-phone      # per-phone | per-call
        inbound_policy: allowlist     # disabled | allowlist | open
        allow_from: ["+15557654321"]
        inbound_greeting: "Hello, this is Hermes. How can I help?"
        serve:
          bind: "127.0.0.1"
          port: 3334
          path: "/voice/webhook"
        public_url: null              # explicit public base URL, if you have one
        tunnel:
          provider: none              # none | ngrok | tailscale-serve | tailscale-funnel
        outbound:
          default_mode: notify        # notify | conversation
          notify_hangup_delay_s: 3
        timeouts:
          max_call_s: 600
          ring_s: 45
          silence_s: 30
        security:
          skip_signature_verification: false   # dev only — logs a warning
        realtime:
          enabled: false              # speech-to-speech (telnyx/twilio only)
          provider: openai            # openai | gemini
```

### Environment variables

| Variable | Used for |
|---|---|
| `TELNYX_API_KEY`, `TELNYX_CONNECTION_ID`, `TELNYX_PUBLIC_KEY` | Telnyx |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Twilio |
| `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN` | Plivo |
| `VOICE_CALL_FROM_NUMBER` | caller ID if not in config |
| `VOICE_CALL_TO_NUMBER` | default outbound destination if not in config |
| `VOICE_CALL_ALLOWED_NUMBERS` | inbound allowlist (comma-separated) |
| `VOICE_CALL_HOME_NUMBER` | cron/notification call target |
| `NGROK_AUTHTOKEN`, `NGROK_DOMAIN` | ngrok tunnel |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | realtime voice models |

## Realtime voice (speech-to-speech)

Turn-based calls use carrier TTS and transcription — reliable, but each
turn takes a few seconds. Realtime mode instead bridges the call's audio
directly to a speech-to-speech model (OpenAI Realtime or Gemini Live) for
natural, low-latency conversation with barge-in (you can interrupt the
assistant mid-sentence).

```yaml
gateway:
  platforms:
    voice_call:
      enabled: true
      extra:
        provider: telnyx            # realtime works on telnyx and twilio
        # ... telnyx setup as above ...
        realtime:
          enabled: true
          provider: openai          # openai | gemini
          model: gpt-realtime       # optional override
          voice: marin
```

Add `OPENAI_API_KEY` (or `GEMINI_API_KEY` with `provider: gemini`) to
`~/.hermes/.env`.

How it works: when a realtime call is dialed or answered, the carrier
opens a media WebSocket back to Hermes (authenticated with a one-shot
token), and the plugin pipes audio between the phone line (µ-law 8 kHz)
and the model (PCM 16/24 kHz). During the call the realtime model can ask
the full Hermes agent questions through an `agent_consult` tool — "let me
check that" — so calendar, memory, and live data stay available without
leaving the audio loop. Transcripts are still recorded to the call
history.

Conversation-mode calls use the realtime bridge; `notify` calls keep
plain carrier TTS (cheaper and sufficient for one-shot messages).

## Security notes

- **Webhook signatures are verified by default** (Telnyx Ed25519, Twilio
  HMAC-SHA1, Plivo HMAC-SHA1 v3). `skip_signature_verification: true` is
  for local development only and logs a warning at every startup.
- **Inbound calls are deny-by-default**: `allowlist` only admits numbers in
  `allow_from`; `open` accepts anyone and must be set explicitly.
- The webhook server also enforces body-size limits, per-IP request
  limits, and replay deduplication.
- Keep credentials in `~/.hermes/.env`. They are never logged, and
  `hermes voicecall doctor` reports presence only — never values.

## Troubleshooting

- **"runtime is not running"** — start `hermes gateway run` with the
  platform enabled.
- **Inbound calls never arrive** — check the carrier's webhook URL points
  at your `public_url`/tunnel + `/voice/webhook`, and that
  `hermes voicecall doctor` shows the tunnel resolved.
- **Caller gets rejected** — their number isn't in `allow_from`
  (E.164 format, e.g. `+15557654321`).
- **Port already in use** — change `serve.port`; the gateway retries
  automatically once the conflict clears.
- **Calls drop after 10 minutes** — that's `timeouts.max_call_s`; raise it.
