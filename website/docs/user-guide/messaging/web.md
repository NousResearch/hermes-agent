---
sidebar_position: 8
title: "Web UI"
description: "Access Hermes from any browser on your network — phone, tablet, or desktop"
---

# Web UI Setup

Access Hermes from any browser on your local network. Open the URL on your phone, tablet, or another computer — no app install, no third-party account needed.

:::info No External Dependencies
The Web adapter uses `aiohttp`, which is already included in the `[messaging]` extra. No additional packages or external services are required.
:::

## Overview

| Component | Value |
|-----------|-------|
| **Library** | `aiohttp` (HTTP + WebSocket) |
| **Connection** | Local network (LAN) |
| **Auth** | Token-based (auto-generated or custom) |
| **Features** | Markdown, code highlighting, voice messages, images, mobile responsive |

---

## Quick Start

### Option 1: On-Demand via Command

Start the gateway normally, then type from any connected platform (Telegram, Discord, etc.):

```
/remote-control
```

The bot replies with the URL and access token. Open the URL on your phone.

You can also specify a custom port and token:

```
/remote-control 9000 mysecrettoken
```

### Option 2: Auto-Start with Gateway

Add to `~/.hermes/.env`:

```bash
WEB_UI_ENABLED=true
WEB_UI_PORT=8765          # default: 8765
WEB_UI_TOKEN=mytoken      # auto-generated if empty
```

Start the gateway:

```bash
hermes gateway
```

The web UI starts automatically alongside your other platforms.

---

## Step 1: Configure

Add to `~/.hermes/.env`:

```bash
# Enable Web UI
WEB_UI_ENABLED=true

# Port to listen on (default: 8765)
WEB_UI_PORT=8765

# Bind address (default: 0.0.0.0 = all interfaces, for LAN access)
# Set to 127.0.0.1 for localhost-only access
WEB_UI_HOST=0.0.0.0

# Access token (leave empty to auto-generate on each startup)
WEB_UI_TOKEN=your-secret-token
```

## Step 2: Start the Gateway

```bash
hermes gateway
```

You'll see output like:

```
[Web] Web UI: http://192.168.1.106:8765
[Web] Access token: your-secret-token
```

## Step 3: Open in Browser

1. Open the URL shown in the console on any device on the same network
2. Enter the access token
3. Start chatting

---

## Features

### Markdown & Code Highlighting

Bot responses render full GitHub-flavored Markdown with syntax-highlighted code blocks powered by highlight.js.

### Voice Messages

Click the microphone button to record a voice message. The audio is transcribed via Whisper STT and sent to the agent. If voice mode is enabled (`/voice tts`), the bot replies with audio playback in the browser.

### Images & Files

- Images display inline in the chat
- Documents show as download links
- Generated images from the agent appear automatically

### Mobile Responsive

The UI adapts to phone screens — full chat experience with touch-friendly input and buttons.

### Typing Indicator

Shows an animated indicator while the agent is processing your message.

### Auto-Reconnect

If the connection drops (server restart, network change), the client automatically reconnects with exponential backoff.

---

## Firewall & Network

### macOS Firewall

macOS may block incoming connections by default. If devices on your network can't connect:

1. **System Settings** > **Network** > **Firewall**
2. Either disable the firewall temporarily, or add Python to the allowed apps

### Localhost Only

To restrict access to the local machine only:

```bash
WEB_UI_HOST=127.0.0.1
```

### Remote Access (Outside LAN)

The Web UI is designed for local network access. For access from outside your network, use a tunnel:

```bash
# Using ngrok
ngrok http 8765

# Using Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8765

# Using Tailscale (recommended — encrypted, no port forwarding)
# Install Tailscale on both devices, then access via Tailscale IP
```

---

## Security

- **Token authentication** — every WebSocket connection must authenticate with the correct token before sending messages
- **No data leaves your network** — the server runs locally, chat data stays on your machine
- **No HTTPS by default** — traffic is unencrypted on the LAN. Use a reverse proxy or tunnel for encryption
- **File uploads** require the auth token in the `Authorization` header
- **Media cleanup** — uploaded and generated files are automatically deleted after 24 hours

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_UI_ENABLED` | `false` | Enable the web gateway |
| `WEB_UI_PORT` | `8765` | HTTP server port |
| `WEB_UI_HOST` | `0.0.0.0` | Bind address (`0.0.0.0` = LAN, `127.0.0.1` = localhost) |
| `WEB_UI_TOKEN` | (auto) | Access token. Auto-generated if empty. |

---

## Troubleshooting

### "Server not found" on phone

- Verify both devices are on the **same WiFi network**
- Check if macOS firewall is blocking (see Firewall section above)
- Try the IP address shown in console output, not `localhost`
- If using VPN, the console shows all available IPs — try each one

### Port already in use

Change the port in `.env`:

```bash
WEB_UI_PORT=9000
```

### Voice recording not working

- Browser must support `MediaRecorder` API (Chrome, Firefox, Safari 14.5+)
- HTTPS is required for microphone access on non-localhost origins
- On localhost (`127.0.0.1`), HTTP works fine for microphone

### CDN resources not loading

The UI loads `marked.js` and `highlight.js` from CDN. If you're offline or behind a restrictive proxy, markdown rendering and code highlighting won't work but basic chat still functions.
