---
sidebar_position: 6
title: "Teams Meetings"
description: "Set up the Microsoft Teams meeting summary pipeline with Microsoft Graph webhooks"
---

# Microsoft Teams Meetings

Use the Teams meeting pipeline when you want Hermes to ingest Microsoft Graph meeting events, fetch transcripts first, fall back to recordings plus STT when needed, and deliver a structured summary to downstream sinks.

This page focuses on setup and enablement:
- Graph credentials
- webhook listener configuration
- Teams delivery modes
- pipeline config shape

For day-2 operations, go-live checks, and the operator worksheet, use the dedicated guide: [Operate the Teams Meeting Pipeline](/docs/guides/operate-teams-meeting-pipeline).

## What This Feature Does

The pipeline:
1. receives Microsoft Graph webhook events
2. resolves the meeting and prefers transcript artifacts first
3. falls back to recording download plus STT when no usable transcript is available
4. stores durable job state and sink records locally
5. can write summaries to Notion, Linear, and Microsoft Teams

Operator actions stay in the CLI:

```bash
hermes teams-pipeline validate
hermes teams-pipeline list
hermes teams-pipeline maintain-subscriptions
```

## Prerequisites

Before enabling the meetings pipeline, make sure you have:

- a working Hermes install
- the existing [Microsoft Teams bot setup](/docs/user-guide/messaging/teams) if you want Teams outbound delivery
- Microsoft Graph application credentials with the permissions required for the meeting resources you plan to subscribe to
- a public HTTPS URL that Microsoft Graph can call for webhook delivery
- `ffmpeg` installed if you want recording-plus-STT fallback

## Step 1: Add Microsoft Graph Credentials

Add Graph app-only credentials to `~/.hermes/.env`:

```bash
MSGRAPH_TENANT_ID=<tenant-id>
MSGRAPH_CLIENT_ID=<client-id>
MSGRAPH_CLIENT_SECRET=<client-secret>
```

These credentials are used by:
- the Graph client foundation
- subscription maintenance commands
- meeting resolution and artifact fetches
- Graph-based Teams outbound delivery when you do not provide a dedicated Teams access token

## Step 2: Enable the Graph Webhook Listener

The webhook listener is a gateway platform named `msgraph_webhook`. At minimum, enable it and set a client state value:

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<random-shared-secret>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

The listener exposes:
- `/msgraph/webhook` for Graph notifications
- `/health` for a simple health check

You need to route your public HTTPS endpoint to that listener. For example, if your public domain is `https://ops.example.com`, your Graph notification URL would typically be:

```text
https://ops.example.com/msgraph/webhook
```

## Step 3: Configure Teams Delivery and Pipeline Behavior

The meeting pipeline reads its runtime config from the existing `teams` platform entry. Pipeline-specific knobs live under `teams.extra.meeting_pipeline`. Teams outbound delivery stays on the normal Teams platform config surface.

Example `~/.hermes/config.yaml`:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      port: 8646
      client_state: "replace-me"
      accepted_resources:
        - "communications/onlineMeetings"

  teams:
    enabled: true
    extra:
      client_id: "your-teams-client-id"
      client_secret: "your-teams-client-secret"
      tenant_id: "your-teams-tenant-id"

      # outbound summary delivery
      delivery_mode: "graph" # or incoming_webhook
      team_id: "team-id"
      channel_id: "channel-id"
      # incoming_webhook_url: "https://..."

      meeting_pipeline:
        transcript_min_chars: 80
        transcript_required: false
        transcription_fallback: true
        ffmpeg_extract_audio: true
        notion:
          enabled: false
        linear:
          enabled: false
```

## Teams Delivery Modes

The pipeline supports two Teams summary-delivery modes inside the existing Teams plugin.

### `incoming_webhook`

Use this when you want a simple webhook post into Teams without channel-message creation through Graph.

Required config:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "incoming_webhook"
      incoming_webhook_url: "https://..."
```

### `graph`

Use this when you want Hermes to post the summary through Microsoft Graph into a Teams chat or channel.

Supported targets:
- `chat_id`
- `team_id` + `channel_id`
- `team_id` + `home_channel` fallback for the existing Teams platform

Example:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "graph"
      team_id: "team-id"
      channel_id: "channel-id"
```

## Step 4: Start the Gateway

Start Hermes normally after updating config:

```bash
hermes gateway run
```

Or, if you run Hermes in Docker, start the gateway the same way you already do for your deployment.

Check the listener:

```bash
curl http://localhost:8646/health
```

## Step 5: Create Graph Subscriptions

Use the plugin CLI to create and inspect subscriptions.

Examples:

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllRecordings \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"
```

:::warning Graph subscriptions expire in 72 hours

Microsoft Graph caps webhook subscriptions at 72 hours and will not auto-renew them. You MUST schedule `hermes teams-pipeline maintain-subscriptions` before going live, or notifications will silently stop three days after any manual subscription creation. See [Automating subscription renewal](/docs/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production) in the operator runbook — three options (Hermes cron, systemd timer, plain crontab).

:::

For subscription maintenance and day-2 operator flows, continue with the guide: [Operate the Teams Meeting Pipeline](/docs/guides/operate-teams-meeting-pipeline).

## Validation

Run the built-in validation snapshot:

```bash
hermes teams-pipeline validate
```

Useful companion checks:

```bash
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
```

## Troubleshooting

| Problem | What to check |
|---------|---------------|
| Graph webhook validation fails | Confirm the public URL is correct and reachable, and that Graph is calling the exact `/msgraph/webhook` path |
| Jobs do not appear in `hermes teams-pipeline list` | Confirm `msgraph_webhook` is enabled and that subscriptions point at the right notification URL |
| Transcript-first never succeeds | Check Graph permissions for transcript resources and whether the transcript artifact exists for that meeting |
| Recording fallback fails | Confirm `ffmpeg` is installed and the Graph app can access recording artifacts |
| Teams summary delivery fails | Re-check `delivery_mode`, target IDs, and Teams auth config |

## Related Docs

- [Microsoft Teams bot setup](/docs/user-guide/messaging/teams)
- [Operate the Teams Meeting Pipeline](/docs/guides/operate-teams-meeting-pipeline)
