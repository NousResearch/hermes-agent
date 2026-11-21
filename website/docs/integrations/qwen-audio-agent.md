---
sidebar_position: 5
title: "qwen-audio-agent Integration"
description: "Drive Hermes hands-free with realtime full-duplex voice over ACP — barge-in, a local wake word, and background task readback"
---

# qwen-audio-agent Integration

[qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent) is an open-source realtime full-duplex voice frontend for ACP agents. It spawns Hermes over stdio via `hermes acp` and turns a session into a hands-free voice conversation — in a macOS desktop orb, a terminal TUI, or a browser.

## What it adds

| Capability | Behavior |
|---|---|
| **Full-duplex voice with barge-in** | Talk while Hermes is speaking; the interruption stops generation immediately |
| **Local wake word** | Wake-word detection runs fully on-device (sherpa-onnx); nothing leaves your machine for activation |
| **Background task readback** | Long jobs run async; results return to the conversation when they finish |
| **Voice permission confirmations** | Approve or deny Hermes' permission requests by voice |

## Setup

1. Install Hermes and complete authentication:

   ```bash
   hermes setup --portal
   ```

2. Install qwen-audio-agent and let it fill in any missing pieces:

   ```bash
   npm install -g qwen-audio-agent
   qwenaudio install hermes
   ```

3. Verify, then start:

   ```bash
   qwenaudio setup --backend hermes
   qwenaudio --backend hermes
   ```

Hermes is a natively supported backend in qwen-audio-agent, so your Hermes configuration — model, tools, MCP servers, skills — carries over unchanged.
