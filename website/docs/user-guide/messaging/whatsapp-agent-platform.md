---
sidebar_position: 6
title: "WhatsApp Agent Platform"
description: "Connect a WhatsApp agent to Hermes with an API key, from the terminal or Desktop"
---

# WhatsApp Agent Platform Setup

Use this route if you created an agent in **WhatsApp → Settings → Agents** and have its **API key**. Hermes connects through Meta's official Agent Platform API using the community **WhatsApp Agent Platform** plugin by [Ahtisham Dilawar](https://github.com/ahtishamdilawar/hermes-whatsapp-agent-platform). The plugin is independent of Meta and Nous Research.

You will message Hermes in the **chat for that agent**. You do not need to link your personal account with a QR code, register a business number, or expose a public webhook. If you want to link a phone account or serve business customers instead, use the [WhatsApp integration chooser](./whatsapp.md#choose-your-integration).

## Before you start

- Use **Hermes 0.21.4 or newer** and configure a working [model provider](../../getting-started/quickstart.md). The catalog plugin provides the WhatsApp connection; Hermes still needs a model to answer.
- Your WhatsApp account must have **Agents** available. Availability is subject to Meta's rollout. If **Settings → Agents** is absent, this setup cannot create that feature for your account.
- In WhatsApp, choose **Settings → Agents → Create an agent**. Open its chat, then **Chat info → API key**, and copy that agent's key. These steps are described in [Meta's developer manual](https://www.whatsapp.com/developer/WhatsApp-Agent-Platform-Developer-Manual.pdf).
- Decide which Hermes **backend and profile** will own the connection. Use that same profile for installation, credentials, gateway status, and restarts.

:::warning Keep the key private
Enter the key in Hermes's credential prompt or Desktop's credential field. Do not paste it into a WhatsApp conversation, screenshots, logs, or a public issue. Regenerating the key in WhatsApp invalidates the previous key; update Hermes afterward.
:::

## Set up from the terminal

Run these commands on the machine running your Hermes gateway. They work in PowerShell and Unix shells.

### 1. Select the profile

For the default profile, find its settings files:

```bash
hermes config path
hermes config env-path
```

These commands print paths, not credentials. The active home can differ from `~/.hermes`, especially on Windows, with `HERMES_HOME`, or with named profiles. See [Profiles](../profiles.md).

For a named profile, use `-p` on **every** command in this guide. For example:

```bash
hermes -p work config path
hermes -p work config env-path
hermes -p work plugins install whatsapp-agent-platform
hermes -p work gateway setup
hermes -p work gateway status
```

### 2. Install and enable the plugin

```bash
hermes plugins install whatsapp-agent-platform
```

Accept the prompt to enable it. If it is already installed but disabled:

```bash
hermes plugins enable whatsapp-agent-platform
```

The package name is **`whatsapp-agent-platform`**. The gateway platform ID is **`whatsapp_agent_platform`**. The [catalog entry](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/whatsapp-agent-platform.yaml) pins the reviewed plugin release; the plugin's [catalog page](/docs/plugins/whatsapp-agent-platform) contains its full configuration and media reference.

### 3. Enter the agent's API key

```bash
hermes gateway setup
```

Choose **WhatsApp Agent Platform**, then enter the key from the intended agent chat. This saves **`WHATSAPP_AGENT_PLATFORM_API_KEY`** in the selected profile's credential store.

If that option is missing, check that the plugin is installed and enabled for this profile and that Hermes meets its version requirement. The ordinary **WhatsApp** option is the Baileys phone-linking integration; it cannot authenticate this API key.

### 4. Connect the gateway

```bash
hermes gateway status
```

If a gateway already serves this profile, restart **that gateway**. For a CLI-managed service:

```bash
hermes gateway restart
```

If Desktop manages it, use Desktop's gateway restart control. If you run it in a terminal, stop that foreground process and run it again in the same profile. If no gateway is running, start one:

```bash
hermes gateway
```

Keep that terminal open. To run in the background, use the OS-specific [Gateway Service instructions](./index.md#service-management). Do not start a second gateway or another client polling the same key: Meta reports **409 Conflict** when pollers compete.

### 5. Send the first message

Check `hermes gateway status` for **`whatsapp_agent_platform`** and its live connection state. Then open the **agent chat you created in WhatsApp** and send a **new** message, such as “Hello, Hermes.” Confirm a reply there.

The plugin skips old updates on first activation, so a message sent before connection may not trigger a reply. **Saved**, **enabled**, and **Connected** are separate states; only a fresh message and reply prove the complete conversation path.

## Set up from Hermes Desktop

1. Select the **backend** that runs Hermes and the **profile** you want to use. On a remote backend, installing the plugin locally does not install it on that remote machine.
2. Open **Capabilities → Plugins**. Find **WhatsApp Agent Platform** in the catalog and install it. Enable its **Agent** switch for the intended profile. It provides a gateway adapter and does not need a Desktop plugin switch.
3. Open **Messaging** and select **WhatsApp Agent Platform**. Enter the key from the agent's **Chat info → API key** in the required API-key field, then click **Save changes**.
4. Enable the platform. When the **Restart now** banner appears, restart the gateway and wait for the platform to report **Connected**. A saved credential indicator alone does not prove authentication.
5. In WhatsApp, send a **fresh message in that agent's chat** and confirm that Hermes replies.

If the platform does not appear, check installation and the Agent switch on the selected backend/profile, then reopen Messaging. If the app or backend is older and does not expose these controls, use the terminal steps on that backend with the same profile. See the [Desktop messaging guide](../desktop.md#whatsapp-in-desktop).

The Desktop chat backend (`hermes serve`) and the messaging gateway are separate processes. A working Desktop chat does not prove that the WhatsApp gateway is running.

## API authentication and sender approval

The Agent API key connects the **created agent** to Hermes. It is separate from both a WhatsApp **Linked Devices** QR code and a Hermes **DM approval code**.

The plugin verifies the agent's creator through Meta before dispatching messages. The creator does not need the generic Hermes DM pairing handshake. Under the current API restriction, the agent can reply **only to its creator**. Adding extra inbound allowed users cannot grant an outbound capability that Meta does not offer. See the [plugin's authorization reference](https://github.com/ahtishamdilawar/hermes-whatsapp-agent-platform/blob/4d8fc1c482ed06a34b6609b1dcdf59e370de7d40/README.md#who-can-talk-to-your-hermes).

If ordinary contacts receive Hermes pairing requests, check which platform is enabled. Those requests can come from a separate Baileys bot connection. For a private Agent Platform setup, disable the unused **WhatsApp** platform in Messaging on the same profile. Unlink a phone session only if you intend to remove that connection.

## Features and limits

The catalog's v0.2.0 plugin supports text, quoted replies, inbound photos/voice notes/files, outbound media attachments, typing indicators, and scheduled messages to the creator. Voice transcription needs Hermes speech-to-text configuration. See the [plugin media guide](https://github.com/ahtishamdilawar/hermes-whatsapp-agent-platform/blob/4d8fc1c482ed06a34b6609b1dcdf59e370de7d40/README.md#media) for formats and limits.

This route has no message editing or streaming edits, groups, native polls/buttons, or native slash-command menu. Typed commands such as `/help` still work. Outgoing audio is a file attachment; incoming reactions go to plugin hooks and do not trigger an agent answer.

Meta documents separate method budgets: **12 requests/minute** for messages, statuses, and each media method, and **15/minute** for update polling. **429** means the affected budget is exhausted; reduce bursts and wait for retries. These limits are separate from the Business Cloud API's customer-service window. See [Meta's manual](https://www.whatsapp.com/developer/WhatsApp-Agent-Platform-Developer-Manual.pdf).

To reduce status chatter, put this in the `config.yaml` printed by `hermes config path`, preserving your other display settings:

```yaml
display:
  platforms:
    whatsapp_agent_platform:
      tool_progress: "off"
      interim_assistant_messages: false
      long_running_notifications: false
      busy_ack_detail: false
```

Agent chats are **not end-to-end encrypted**. Review [WhatsApp's third-party agent terms](https://www.whatsapp.com/legal/third-party-agents-terms) before sending sensitive content. Official API access remains subject to Meta's terms and enforcement.

## Troubleshooting

| What you see | What to check |
|---|---|
| **Agents is missing in WhatsApp** | Account/region rollout. Hermes cannot enable the WhatsApp feature. Choose another [integration](./whatsapp.md#choose-your-integration) if needed. |
| **WhatsApp Agent Platform is missing in Hermes** | Install and enable `whatsapp-agent-platform` on the selected backend/profile, check the Hermes version, and reopen setup or Messaging. |
| **Saved, but disconnected** | Check platform enablement, the live gateway, selected profile, and authentication errors. Restart the existing gateway after saving a key. |
| **Invalid API key / 400 with code 100** | Copy the key from the intended agent's Chat info. If regenerated, replace the stored key and restart. |
| **401 Unauthorized** | Check the configured credential and authentication error; share only redacted diagnostics. |
| **409 Conflict** | Another client is polling that key. Keep one gateway/poller for this agent. |
| **429 Too Many Requests** | Wait for retries; reduce rapid messages and optional status chatter. |
| **Connected, but no answer** | Send a new message after connection in the correct agent chat, using the creator's account. Check the model/provider if intake succeeds but generation fails. |
| **Pairing codes in unrelated personal chats** | Check whether the Baileys **WhatsApp** platform is enabled separately; its sender policy is independent of Agent Platform authentication. |

For deeper diagnostics, use the [plugin troubleshooting reference](https://github.com/ahtishamdilawar/hermes-whatsapp-agent-platform/blob/4d8fc1c482ed06a34b6609b1dcdf59e370de7d40/README.md#troubleshooting). Keep keys and message contents out of public reports.

## References

- WhatsApp. (2026, August 25). [*WhatsApp Agent Platform developer manual* (Version 1)](https://www.whatsapp.com/developer/WhatsApp-Agent-Platform-Developer-Manual.pdf).
- WhatsApp. (2026, August 25). [*Third-party agent terms*](https://www.whatsapp.com/legal/third-party-agents-terms).
- Dilawar, A. (n.d.). [*Hermes WhatsApp Agent Platform* (v0.2.0 documentation)](https://github.com/ahtishamdilawar/hermes-whatsapp-agent-platform/blob/4d8fc1c482ed06a34b6609b1dcdf59e370de7d40/README.md). GitHub.
- Nous Research. (n.d.). [*WhatsApp Agent Platform catalog entry*](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/whatsapp-agent-platform.yaml). GitHub.
