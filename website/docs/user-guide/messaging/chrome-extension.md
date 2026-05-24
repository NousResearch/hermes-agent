---
sidebar_position: 9
title: "Chrome Extension"
description: "Connect the Hermes Chrome extension to the API server"
---

# Chrome Extension Integration

The Hermes Chrome extension is a browser-native frontend for Hermes Agent. It runs in Chrome's Side Panel and connects to Hermes through the OpenAI-compatible API server, so the same agent, tools, memory, and skills available to Open WebUI are available from the browser.

Extension source: [hermes-extension](https://github.com/NousResearch/hermes-extension).

## Prerequisites

Enable the API server in `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=<generate with: openssl rand -hex 32>
API_SERVER_HOST=0.0.0.0   # only if Hermes runs on a remote server
```

Restart the gateway after changing `.env`:

```bash
hermes gateway stop && hermes gateway
```

The extension sends requests to `/v1/chat/completions` with streaming enabled and includes `X-Hermes-Session-Id` so one browser conversation maps to one Hermes session.

## CORS Configuration

Because the Chrome extension calls Hermes directly from a browser context, the API server must allow the extension origin.

For local development with an unpacked extension, the extension ID may change. Use wildcard CORS only while developing:

```bash
API_SERVER_CORS_ORIGINS=*
```

For production, use the stable Chrome extension origin:

```bash
API_SERVER_CORS_ORIGINS=chrome-extension://<your-stable-extension-id>
```

You can get a stable extension ID from a Chrome Web Store listing or by adding Chrome's `key` field to `manifest.json` after you have a key pair. See the extension repo README for the exact workflow.

:::warning
Do not expose a remote Hermes API server to the internet with `API_SERVER_CORS_ORIGINS=*`. Set a specific `chrome-extension://...` origin before using a public or shared network.
:::

## Local vs Remote Hermes

### Local

When Chrome and Hermes run on the same machine, keep the default host binding:

```bash
API_SERVER_HOST=127.0.0.1
API_SERVER_CORS_ORIGINS=chrome-extension://<extension-id>
```

No firewall rule is required. In the extension settings, use:

```text
http://127.0.0.1:8642
```

### Remote

When Hermes runs on a remote machine, bind the API server to all interfaces and allow the extension origin:

```bash
API_SERVER_HOST=0.0.0.0
API_SERVER_CORS_ORIGINS=chrome-extension://<extension-id>
```

Open the API server port, `8642` by default, in the remote host firewall or AWS security group. Use HTTPS or a trusted private network when exposing Hermes beyond localhost.

In the extension settings, use the remote API URL:

```text
http://<server-hostname-or-ip>:8642
```

## Quick Start

1. Generate an API key:

   ```bash
   openssl rand -hex 32
   ```

2. Add API server settings to `~/.hermes/.env`:

   ```bash
   API_SERVER_ENABLED=true
   API_SERVER_KEY=<your-generated-key>
   API_SERVER_CORS_ORIGINS=*
   ```

3. Restart Hermes:

   ```bash
   hermes gateway stop && hermes gateway
   ```

4. Build and install the extension:

   ```bash
   cd ../hermes-extension
   npm install
   npm run build
   ```

   In Chrome, open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**, and select the extension's `dist/` directory.

5. Click the Hermes extension icon to open the Side Panel.

6. Open **Settings** and enter:

   ```text
   API URL: http://127.0.0.1:8642
   API Key: <your-generated-key>
   ```

7. Start chatting. Use the `@ page` button to attach the current page to your next message. Use `/new` to clear the browser-side conversation and start a new Hermes session.

## Setup Wizard

You can configure the required API server settings with:

```bash
hermes gateway setup chrome-extension
```

During development, leave the extension ID blank and the wizard will set:

```bash
API_SERVER_CORS_ORIGINS=*
```

For production, enter the stable Chrome extension ID and the wizard will set:

```bash
API_SERVER_CORS_ORIGINS=chrome-extension://<extension-id>
```
