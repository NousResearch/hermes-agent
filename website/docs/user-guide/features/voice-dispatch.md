---
sidebar_position: 11
title: "Grok Voice Dispatch"
description: "Realtime Grok voice control for dispatching work to isolated Hermes delegates from the web dashboard"
---

# Grok Voice Dispatch

Grok Voice Dispatch is a dashboard voice interface for controlling Hermes Agent with realtime speech. It is different from [Voice Mode](./voice-mode.md): Voice Mode adds speech input/output to the CLI and messaging platforms, while Grok Voice Dispatch runs in the web dashboard and uses Grok Voice as a fast conversational controller for isolated Hermes delegates.

The safety model is simple:

```text
Browser microphone
  → xAI realtime voice session using a short-lived client secret
  → narrow voice tool calls only
  → Hermes dashboard bridge
  → isolated Hermes delegate
  → normal Hermes tools, sessions, approvals, and logs
```

Grok Voice does **not** receive direct shell, file, browser, MCP, or credential tools. It can start, check, and stop delegate work. Hermes still performs the work under the normal Hermes execution and approval model.

## Setup

Run the dedicated setup section:

```bash
hermes setup voice
```

The setup flow is disabled by default. It only enables realtime voice dispatch when you explicitly opt in.

When enabled, setup writes non-secret settings to `~/.hermes/config.yaml` and stores the xAI API key in `~/.hermes/.env`:

```bash
XAI_API_KEY=...
```

The setup wizard never prints the key. If `XAI_API_KEY` already exists, Hermes detects it without echoing the value and asks before replacing it.

## Configuration

Default config values:

```yaml
voice:
  realtime:
    enabled: false
    provider: xai
    model: grok-voice-latest
    voice: eve
    ephemeral_token_ttl_seconds: 300
    turn_detection:
      type: server_vad
      threshold: 0.85
      silence_duration_ms: 900
      prefix_padding_ms: 333
    audio:
      input_rate: 24000
      output_rate: 24000
    dispatch:
      max_active_delegates: 1
      default_toolsets: []
      summarize_events_for_voice: true
```

You can edit these with:

```bash
hermes config edit
```

or by re-running:

```bash
hermes setup voice
```

## Start the dashboard

Install the web dashboard dependencies if needed:

```bash
pip install 'hermes-agent[web]'
```

Start the local dashboard:

```bash
hermes dashboard
```

Open the **Voice** tab at `/voice`.

The dashboard protects voice endpoints with the dashboard session token. The browser requests a short-lived xAI realtime client secret from Hermes; the long-lived `XAI_API_KEY` stays server-side in `.env`.

## How dispatch works

Voice Dispatch exposes only narrow control functions to the realtime voice session:

- `start_delegate` — start an isolated Hermes delegate for a user task
- `get_delegate_status` — check active or specified delegate status
- `stop_delegate` — request cancellation/stop for a delegate

The delegate runs as its own Hermes agent session. That keeps the main dashboard/chat session free while the work runs.

Delegate status and events are written to a local SQLite ledger under the Hermes home directory. Event previews are redacted before persistence so obvious secret-shaped values are not stored as raw audit output.

## Approvals

Voice Dispatch preserves Hermes approvals.

If a delegate hits an action that needs approval, Grok Voice may tell you approval is required, but it cannot approve privileged actions by itself. You approve or deny through the Hermes UI, with the actual command/tool/action details visible before you click.

Voice Dispatch v1 does not offer voice-mediated `always` approvals. Available approval choices are limited to:

- approve once
- approve for this session
- deny

This prevents a remote realtime voice model from permanently widening local permissions.

## Phone or remote access

The safest default is local-only dashboard access:

```bash
hermes dashboard --host 127.0.0.1
```

For phone access, prefer a private network or authenticated tunnel:

- Tailscale
- Cloudflare Tunnel with Access
- ngrok with authentication

Do not expose the dashboard directly on a public interface. Avoid `--insecure` unless you understand the risk and have separate network controls in place.

## Troubleshooting

### Voice tab says disabled

Run:

```bash
hermes setup voice
```

and explicitly enable Grok Voice Dispatch.

### Missing xAI credentials

Set the key through setup:

```bash
hermes setup voice
```

or edit `~/.hermes/.env`:

```bash
XAI_API_KEY=...
```

Restart the dashboard after changing `.env`.

### Browser microphone fails

If the config endpoint and xAI client-secret endpoint work but audio fails, check browser and operating-system microphone permissions. In automated browsers, `getUserMedia` often fails with permission denied even when the backend is configured correctly.

### Delegate finishes but voice does not announce it

Keep the `/voice` page open while the delegate runs. The dashboard watches active delegate status and pushes concise completion summaries into the realtime voice conversation while the WebSocket is live.
