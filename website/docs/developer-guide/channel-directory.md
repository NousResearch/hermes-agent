---
sidebar_position: 12
title: "Channel Directory"
description: "How Hermes discovers, serializes, and resolves messaging targets"
---

# Channel Directory

The gateway builds `~/.hermes/channel_directory.json` at startup and refreshes it periodically. `send_message(action="list")` uses it to display discovered channels and contacts; named targets are resolved through it before delivery.

## Serialized target form

For session-derived entries, the directory stores the platform in the top-level map and serializes the target as:

```json
{
  "id": "<chat_id>[:<thread_id>]",
  "name": "<display name>",
  "type": "<display type>",
  "thread_id": "<thread_id or null>"
}
```

`name` and `type` are discovery and display metadata. Resolution returns the stored `id`; the send path parses that value into `chat_id` and, for platforms with supported thread syntax, `thread_id`.

## Slack workspaces

Slack discovery calls `users.conversations` for every connected workspace client. The current directory entry stores the Slack conversation ID, name, and display type, but it does not serialize the Slack workspace (`team_id`).

Consequently, in a multi-workspace Slack setup, a Directory entry is a discovery record rather than a documented guarantee of workspace-scoped outbound routing. After name resolution, the generic send path carries a raw Slack conversation ID and optional thread timestamp. The live Slack adapter can select a workspace from explicit outbound metadata or its process-local channel-to-workspace knowledge; neither is populated by the Directory entry itself.

This is a description of the current contract boundary, not a routing-policy change. Callers that require deterministic multi-workspace Slack routing must use a delivery interface that carries workspace scope explicitly.

## Discord

Discord entries may also include `guild` so names such as `Guild/channel` can be resolved unambiguously. The outbound Discord sender receives the resolved channel ID and optional thread ID; guild is not an outbound transport argument.
