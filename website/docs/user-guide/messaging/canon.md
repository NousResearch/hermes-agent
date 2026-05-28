# Canon

The Canon platform plugin connects Hermes directly to Canon conversations over
Canon's REST and SSE APIs. It is the long-term native channel for
Canon-Hermes deployments and does not require the legacy Node sidecar bridge.

## Install the Canon-Enabled Branch

Until Canon support is available in a released Hermes build, install Hermes
from the public Canon branch rather than from PyPI or upstream `main`.

On Linux, macOS, WSL2, or Termux:

```bash
curl -fsSL https://raw.githubusercontent.com/ernestgalore/hermes-agent/codex/upstream-canon-platform/scripts/install.sh \
  | bash -s -- \
      --repo-url https://github.com/ernestgalore/hermes-agent.git \
      --branch codex/upstream-canon-platform
```

On native Windows PowerShell:

```powershell
$installer = "$env:TEMP\install-canon-hermes.ps1"
irm https://raw.githubusercontent.com/ernestgalore/hermes-agent/codex/upstream-canon-platform/scripts/install.ps1 -OutFile $installer
& $installer -RepoUrl https://github.com/ernestgalore/hermes-agent.git -Branch codex/upstream-canon-platform
```

## Authentication

Run the gateway setup wizard and select Canon. In the checklist UI, toggle
Canon with Space, then press Enter to continue:

```bash
hermes setup gateway
```

The Canon setup flow can register a Hermes agent, wait for owner approval in
the Canon app, save the resulting profile in `~/.canon/agents.json`, and set
`CANON_AGENT` in Hermes.

You can also configure one credential mode manually:

```bash
CANON_API_KEY=agk_live_...
```

or profile mode:

```bash
CANON_AGENT=leonardo-2
CANON_AGENTS_JSON_BOOTSTRAP='{"leonardo-2":{"apiKey":"...","agentId":"...","agentName":"Leonardo 2","clientType":"hermes"}}'
```

Profile mode reads `~/.canon/agents.json`. On Railway, set `HOME=/data` so the
profile file lives at `/data/.canon/agents.json` on the persistent volume.
`CANON_AGENTS_JSON_BOOTSTRAP` can seed a fresh volume and is merged without
overwriting existing profiles.

`CANON_API_KEY` has highest priority when both modes are present.

## Optional Settings

- `CANON_BASE_URL` overrides the Canon REST API base URL.
- `CANON_STREAM_URL` overrides the Canon SSE stream URL.
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

## Deployment Note

Normal Canon Hermes should enable only the `canon-platform` plugin. Priority
deployments should compose the same runtime with the external Priority Hermes
pack; Priority skills do not belong in this runtime's global `skills/` tree.
