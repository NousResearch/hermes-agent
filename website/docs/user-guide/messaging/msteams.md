---
sidebar_position: 4
title: "Microsoft Teams"
description: "Set up Hermes Agent as a Microsoft Teams bot with Bot Framework and Microsoft Graph"
---

# Microsoft Teams

Hermes includes a native Microsoft Teams gateway adapter with Bot Framework webhook ingress, Microsoft Graph enrichments, Teams mention handling, Adaptive Cards, polls, file delivery, and direct-send support.

## What works

| Capability | Status |
|------------|--------|
| DMs | ✅ |
| Group chats | ✅ |
| Channel conversations | ✅ |
| Mention gating / reply style overrides | ✅ |
| Typing indicator | ✅ |
| Adaptive Cards | ✅ |
| Polls | ✅ |
| Image sends | ✅ |
| DM file sends (FileConsentCard) | ✅ |
| Group/channel file sends (SharePoint) | ✅ |
| Inbound attachment download | ✅ |
| Graph fallback for hosted contents / reference attachments | ✅ |
| Message actions: read / edit / delete / pins / reactions / search / member-info / channel-list / channel-info | ✅ |

## Required credentials

Put these in `~/.hermes/.env`:

```bash
MSTEAMS_APP_ID=...
MSTEAMS_APP_PASSWORD=...
MSTEAMS_TENANT_ID=...
```

These are the Azure Bot / Entra app credentials used by Hermes for Bot Framework auth and Microsoft Graph client-credentials flows.

## Minimal setup

```bash
hermes gateway setup
```

Or configure manually:

```bash
MSTEAMS_APP_ID=...
MSTEAMS_APP_PASSWORD=...
MSTEAMS_TENANT_ID=...
MSTEAMS_WEBHOOK_HOST=0.0.0.0
MSTEAMS_WEBHOOK_PORT=3978
MSTEAMS_WEBHOOK_PATH=/api/messages
```

Then start the gateway:

```bash
hermes gateway start
hermes gateway status
```

## Teams app / Azure requirements

You need:

1. An Azure Bot / Microsoft App registration
2. Microsoft Teams channel enabled on the bot
3. A public HTTPS messaging endpoint that points at Hermes:
   - `https://your-domain.example/api/messages`
4. A Teams app manifest installed into personal / group / team scopes as needed

For local development, expose Hermes with a tunnel such as ngrok or Tailscale Funnel.

## Direct sends and cron delivery

Hermes can proactively send to Teams from `send_message`, cron jobs, and home-channel delivery **after** the bot has already seen that conversation and persisted its conversation reference.

If Hermes has never received an inbound event from the target DM/group/channel, direct delivery will fail because Bot Framework proactive sends need a stored conversation reference.

## Graph-powered features

To reach full Teams functionality, Hermes uses Microsoft Graph for:

- user lookup / member info
- channel thread context fetch
- hosted contents recovery
- SharePoint / OneDrive reference attachment recovery
- channel info / channel listing
- pins / reactions / search

If Graph permissions are incomplete, core text messaging still works, but some attachment/history/action features may degrade.

## File handling

### DMs
Hermes uses the Teams **FileConsentCard** flow for document-style file sends in personal chats.

### Group chats / channels
Hermes uses **SharePoint upload + Teams file info card** when `MSTEAMS_SHAREPOINT_SITE_ID` is configured.

```bash
MSTEAMS_SHAREPOINT_SITE_ID=contoso.sharepoint.com,guid1,guid2
```

## Important runtime controls

Common optional variables:

```bash
MSTEAMS_REQUIRE_MENTION=true
MSTEAMS_REPLY_STYLE=thread
MSTEAMS_DM_POLICY=pairing
MSTEAMS_GROUP_POLICY=allowlist
MSTEAMS_ALLOW_FROM=aad-id-1,aad-id-2
MSTEAMS_GROUP_ALLOW_FROM=aad-id-3
MSTEAMS_TEXT_CHUNK_LIMIT=4000
MSTEAMS_SHAREPOINT_SITE_ID=contoso.sharepoint.com,guid1,guid2
MSTEAMS_MEDIA_ALLOW_HOSTS=graph.microsoft.com,sharepoint.com,teams.microsoft.com
MSTEAMS_MEDIA_AUTH_ALLOW_HOSTS=graph.microsoft.com,api.botframework.com
```

## Reply style

Hermes supports:

- `thread` — default, reply in-thread when possible
- `top-level` — send as top-level post / linear message

You can override reply style globally or per team/channel via Teams config JSON.

## Security notes

Recommended defaults:

- Keep DM policy at `pairing` or `allowlist`
- Keep group policy at `allowlist` unless you explicitly want open channels
- Keep `MSTEAMS_MEDIA_AUTH_ALLOW_HOSTS` narrow
- Only expose the webhook over HTTPS

## Troubleshooting

### Bot receives messages but cannot fetch media
Check:

- Graph permissions are granted with admin consent
- `MSTEAMS_MEDIA_ALLOW_HOSTS` includes the attachment host
- `MSTEAMS_MEDIA_AUTH_ALLOW_HOSTS` includes hosts that require bearer-auth retry

### Group/channel file sends do not work
Check:

- `MSTEAMS_SHAREPOINT_SITE_ID` is configured
- the app has the required SharePoint / Graph application permissions
- admin consent has been granted

### Bot does not answer in channels
Check:

- `MSTEAMS_GROUP_POLICY`
- `MSTEAMS_REQUIRE_MENTION`
- any team/channel overrides in `MSTEAMS_TEAMS_JSON`
- whether the sender is covered by `MSTEAMS_GROUP_ALLOW_FROM`

## Related docs

- [Messaging Gateway](/docs/user-guide/messaging)
- [Environment Variables](/docs/reference/environment-variables)
