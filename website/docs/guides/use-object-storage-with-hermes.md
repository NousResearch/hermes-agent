---
sidebar_position: 7
title: "Use Object Storage with Hermes"
description: "Give Hermes Agent durable file storage through an MCP server: setup, tool scoping, and patterns for artifacts that outlive the sandbox"
---

# Use Object Storage with Hermes

Hermes runs on laptops, VPSes, containers, and serverless infrastructure — and
several of those are ephemeral by design. Anything the agent writes to its
working directory disappears with the sandbox. Object storage gives the agent
a durable place for the files it produces: reports, generated assets,
downloads, data it should find again next session, and files it wants to hand
to you as a link.

This guide connects Hermes to [Tigris](https://www.tigrisdata.com/), an
S3-compatible object storage service with a hosted MCP server, so the setup
is config-only. The same pattern applies to any storage provider that ships
an MCP server.

## When should you use this?

Use object storage when:

- Hermes runs on ephemeral or remote compute and its output needs to survive
- you talk to Hermes from Telegram or another chat surface and want files
  handed back as links rather than trapped on the host
- multiple Hermes deployments (or you and the agent) need to see the same
  files
- the agent produces artifacts on a schedule (see
  [Automate with Cron](./automate-with-cron.md)) that something else consumes

Do not use it when:

- everything the agent touches is local and the host is durable — the
  filesystem MCP server or built-in tools are simpler
- you're tempted to put Hermes's own home directory in a bucket. Session
  state lives in SQLite under `~/.hermes`, and SQLite on remote storage
  corrupts. Buckets are for the agent's files, not its internals.

## Step 1: connect the Tigris MCP server

If your Hermes includes the `tigris` catalog entry, install it from the
catalog:

```bash
hermes mcp install tigris
```

Otherwise, add it to `~/.hermes/config.yaml` yourself:

```yaml
mcp_servers:
  tigris:
    url: "https://mcp.storage.dev/mcp"
    auth: oauth
```

It's a hosted MCP server with OAuth 2.1, so there is nothing to run locally.
On first connect, Hermes opens the browser authorization flow and caches the
token under `~/.hermes/mcp-tokens/`. If Hermes runs on a remote or headless
host, complete the flow with the paste-back prompt or a port forward — see
[OAuth over SSH / Remote Hosts](./oauth-over-ssh.md#mcp-servers).

Tigris uses a single global endpoint, so there is no region to configure;
the same config works wherever the agent runs.

## Step 2: scope the tool surface

The server exposes bucket management (list, create, delete), object
operations (upload, download, list, delete, create folders and text files),
and presigned share links. As with any MCP server, connect the smallest
useful surface. For an agent that should file artifacts and share them — but
never delete anything or touch other buckets — filter at install time (the
catalog checklist) or in config:

```yaml
mcp_servers:
  tigris:
    url: "https://mcp.storage.dev/mcp"
    auth: oauth
    tools:
      include:
        - list_objects
        - get_object
        - put_object
        - create_presigned_url
```

A dedicated bucket per agent keeps the blast radius small: a confused agent
can only rearrange the bucket it works in.

Tigris adds two operator-side controls that extend this. Bucket
[snapshots](https://www.tigrisdata.com/docs/snapshots/) capture the bucket at
a point in time, so an agent mistake is a rollback rather than a loss.
[Forks](https://www.tigrisdata.com/docs/forks/) create a
zero-copy branch of a bucket, so you can point an experimental agent, or a
new version of your prompts, at a fork of the real artifact history and throw
it away afterwards — testing against real accumulated state instead of
fixtures. Both are managed from the Tigris CLI or console, not by the agent
itself. If you use a different storage provider, check what versioning
features it offers.

## Step 3: put it to work

Tell Hermes once where its files go — it persists knowledge across sessions,
so this tends to stick:

```text
Store everything you produce in the bucket hermes-artifacts. When you finish
a deliverable, give me a presigned link to it.
```

Then use it like any other capability:

```text
Summarize this week's monitoring reports, write the summary to
hermes-artifacts as reports/week-29.md, and send me a share link.
```

The share link is a presigned URL — it works from your phone, expires on its
own, and requires no account on the receiving end. Paired with a
[cron job](./automate-with-cron.md), this is the standard shape for a nightly
report bot: generate, upload, drop the link in chat.

## Verify

Ask Hermes to list the bucket, or check directly:

```bash
aws s3 ls s3://hermes-artifacts/ --endpoint-url https://t3.storage.dev
```

If the tools don't appear in the session, confirm the `mcp_servers` block is
present, restart the session, and check that the OAuth flow completed —
`hermes mcp login tigris` re-runs it.
