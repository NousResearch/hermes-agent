# WhatsApp Agent Platform

The WhatsApp Agent Platform creates a separate private chat for a personal agent. This is different from the WhatsApp Web bridge and from the WhatsApp Business Cloud API. See the [official developer manual](https://www.whatsapp.com/developer/WhatsApp-Agent-Platform-Developer-Manual.pdf).

## Setup

1. In WhatsApp, open **Settings → Agents**, create an agent, then copy its API key from the agent chat's info screen.
2. Run `hermes config` and enter the key for **WhatsApp Agent**. The key is stored as `WHATSAPP_AGENT_TOKEN` in your Hermes `.env`. You can also set that secret directly.
3. Enable the `whatsapp_agent` platform in `config.yaml` if the config tool has not enabled it:

   ```yaml
   platforms:
     whatsapp_agent:
       enabled: true
   ```

4. Restart the gateway, then send a message in the new agent chat. The first run skips messages sent before the gateway started, so send a new message after startup.

The platform allows the agent to message only its creator. Hermes stores the update cursor in `HERMES_HOME/gateway` so restarts resume from the last processed batch. Only text messages are currently handled. The upstream API limits update polls to 15/minute and outbound messages to 12/minute; replies longer than 4096 characters are split.

Agent chats are not end-to-end encrypted; review WhatsApp's [agent terms](https://www.whatsapp.com/legal/third-party-agents-terms). Treat the key as a secret. Reset it in WhatsApp if it has been shared, then update Hermes's `.env` and restart the gateway.
