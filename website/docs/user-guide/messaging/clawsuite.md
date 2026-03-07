---
sidebar_position: 7
title: "ClawSuite"
description: "Connect ClawSuite desktop app to Hermes Agent via WebSocket"
---

# ClawSuite Setup

[ClawSuite](https://github.com/outsourc-e/hermes-agent) is an Electron-based desktop management UI that connects to hermes-agent over a local WebSocket connection. It gives you a native desktop interface for managing your agent, monitoring sessions, and chatting — without opening a terminal.

## Architecture

```
┌─────────────────┐        WebSocket (ws://127.0.0.1:18789)        ┌──────────────────────┐
│  ClawSuite App  │ ◄────────────────────────────────────────────► │  hermes-agent gateway │
│  (Electron UI)  │        JSON protocol over WS                   │  ClawSuiteAdapter     │
└─────────────────┘                                                 └──────────────────────┘
```

The ClawSuiteAdapter starts a local WebSocket server inside the hermes-agent gateway process. ClawSuite connects to it on startup and receives streaming agent responses in real time.

## Quick Setup

### 1. Enable ClawSuite in your environment

Add to `~/.hermes/.env`:

```bash
# Enable the ClawSuite WebSocket gateway
CLAWSUITE_ENABLED=true

# Optional: protect with a shared secret (recommended if port is reachable externally)
CLAWSUITE_TOKEN=your-secret-token

# Optional: customize port/host (defaults: 18789, 127.0.0.1)
# CLAWSUITE_PORT=18789
# CLAWSUITE_HOST=127.0.0.1
```

### 2. Install the websockets dependency

```bash
pip install websockets
```

Or if you installed via the project extras:

```bash
pip install 'hermes-agent[clawsuite]'
```

### 3. Start the gateway

```bash
hermes gateway
```

You should see:

```
[ClawSuite] Listening on ws://127.0.0.1:18789 (auth=token)
```

### 4. Launch ClawSuite

Open the ClawSuite desktop app. It will auto-connect to `ws://127.0.0.1:18789` on startup. If you set a token, enter it in ClawSuite's connection settings.

## Protocol Reference

All messages are JSON over WebSocket.

### Connect Handshake

**Client → Server** (first message after connecting):
```json
{
  "type": "connect",
  "device": { "id": "desktop-01", "nonce": "abc123" },
  "token": "your-secret-token"
}
```

**Server → Client** (challenge):
```json
{
  "type": "event",
  "event": "connect.challenge",
  "payload": { "nonce": "abc123" }
}
```

**Server → Client** (confirmed):
```json
{
  "type": "event",
  "event": "connected",
  "payload": {
    "sessionKey": "desktop-01",
    "deviceId": "desktop-01",
    "server": "hermes-agent",
    "version": "1.0"
  }
}
```

### Chat

**Client → Server**:
```json
{
  "type": "chat",
  "message": "What's in my todo list?",
  "sessionKey": "main"
}
```

**Server → Client** (streaming chunks while agent thinks):
```json
{
  "type": "event",
  "event": "agent",
  "payload": { "stream": "chunk", "text": "Here are your " }
}
```

**Server → Client** (final response):
```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "stream": "done",
    "state": "final",
    "message": {
      "id": "msg-uuid",
      "text": "Here are your current todo items:\n...",
      "role": "assistant"
    }
  }
}
```

### Other Client Messages

| Type | Description |
|------|-------------|
| `{"type":"ping"}` | Keep-alive ping — server replies `{"type":"pong"}` |
| `{"type":"disconnect"}` | Graceful disconnect request |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAWSUITE_ENABLED` | `false` | Set to `true` to enable the adapter |
| `CLAWSUITE_TOKEN` | _(none)_ | Shared secret for authentication. When unset, any client can connect (local-only). |
| `CLAWSUITE_PORT` | `18789` | WebSocket server port |
| `CLAWSUITE_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` only on secured networks. |
| `CLAWSUITE_ALLOW_ALL` | `true` (when no token) | Override auth check |
| `CLAWSUITE_ALLOWED_DEVICES` | _(none)_ | Comma-separated device IDs allowlist |

## Security Notes

- By default ClawSuite binds to `127.0.0.1` — only local processes can connect.
- Set `CLAWSUITE_TOKEN` to prevent unauthorized local processes from connecting.
- Never expose port 18789 to the internet without a reverse proxy with TLS.

## Troubleshooting

**"websockets package not installed"**
```bash
pip install websockets
```

**ClawSuite can't connect**
- Confirm hermes gateway is running: `hermes gateway`
- Check the port matches (`CLAWSUITE_PORT` in both `.env` and ClawSuite settings)
- Check firewall isn't blocking localhost connections

**Authentication failed**
- Ensure `CLAWSUITE_TOKEN` in `.env` matches the token configured in ClawSuite
