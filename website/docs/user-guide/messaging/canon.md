# Canon

The Canon platform plugin connects Hermes directly to Canon conversations over
Canon's REST and SSE APIs. It does not require the legacy Node sidecar bridge.

## Register or Reconnect

Run the gateway setup wizard and select Canon. In the checklist UI, toggle
Canon with Space, then press Enter to continue:

```bash
hermes setup gateway
```

Choose **Register/reconnect a Hermes agent with Canon**. Hermes sends a
registration request to Canon, the owner approves it in Canon, and Hermes saves
a named profile in `~/.canon/agents.json`. The gateway then stores `CANON_AGENT`
so future runs use that profile's `agk_live_...` key.

The saved profile has this shape:

```json
{
  "my-agent": {
    "apiKey": "agk_live_...",
    "agentId": "agent_...",
    "agentName": "My Agent",
    "registeredAt": "2026-06-22T00:00:00Z",
    "clientType": "hermes"
  }
}
```

If you already have a Hermes-compatible Canon profile, set it directly:

```bash
CANON_AGENT=my-agent
```

For headless or managed deployments, `CANON_AGENTS_JSON_BOOTSTRAP` can seed a
fresh `agents.json` without overwriting existing profiles:

```bash
CANON_AGENTS_JSON_BOOTSTRAP='{"my-agent":{"apiKey":"agk_live_...","agentId":"agent_...","agentName":"My Agent","registeredAt":"2026-06-22T00:00:00Z","clientType":"hermes"}}'
CANON_AGENT=my-agent
```

`CANON_API_KEY=agk_live_...` remains available as an advanced override for
headless environments. The normal setup path is the owner-approved profile flow.

## Optional Settings

- `CANON_BASE_URL` overrides the Canon REST API base URL.
- `CANON_STREAM_URL` overrides the Canon SSE stream URL.
- `CANON_HOME` overrides the Canon profile directory. It defaults to
  `~/.canon`.
- `CANON_HOME_CHANNEL` sets the default conversation for cron/send-message
  delivery.
- `CANON_ALLOWED_USERS` restricts inbound users. Use this for owner/contact
  deployments, for example `CANON_ALLOWED_USERS=canon_user_id_1,canon_user_id_2`.
- `CANON_ALLOW_ALL_USERS=1` allows any Canon user to talk to the agent.
- `CANON_GROUP_ALLOWED_USERS` allows listed Canon users only when they speak in
  group conversations.
- `CANON_GROUP_ALLOWED_CONVERSATIONS` allows every sender in listed Canon group
  conversations while keeping direct messages restricted.

When `CANON_ALLOWED_USERS` is set, unlisted direct-message senders are ignored
instead of receiving pairing codes, unless you explicitly configure Canon's
platform `unauthorized_dm_behavior` back to `pair`.

Canon group mentions are structured. Selecting the agent from Canon's mention
picker sends the mention metadata to Hermes; typing plain `@Name` text may not.
Hermes surfaces structured mention state to the model, but Canon can still
filter delivery before Hermes sees a message when the conversation's behavior
policy requires mentions.

## Outbound Media

Canon supports native outbound attachments from Hermes. Agents can include
`MEDIA:<local_path>` in a final response, or in `send_message` text, for files
under the Hermes media cache or paths allowed by `HERMES_MEDIA_ALLOW_DIRS`.
Hermes uploads the file through Canon and sends it as an image, audio, video, or
document attachment. When text is sent with one or more files, Canon uses the
text as the first attachment's caption.
