---
sidebar_position: 16
title: "Manage Hermes Cloud with MCP"
description: "Connect Hermes Agent to the Nous Portal MCP server so your local agent can list, start, stop, and manage your Hermes Cloud instances conversationally"
---

# Manage Hermes Cloud with MCP

[Hermes Cloud](https://portal.nousresearch.com/cloud) runs hosted Hermes Agent instances for you. Normally you manage them from the `/agents` page in the [Nous Portal](/integrations/nous-portal). This guide connects your **local** Hermes Agent to the Portal's MCP server so you can manage those cloud instances by just asking — "list my cloud agents", "restart the stopped one", "what's it costing me" — without leaving your terminal.

It's a standard [MCP](/user-guide/features/mcp) server hosted by Nous Research, gated by the same OAuth login you already use for the Portal. Once connected, Hermes gets two tools it can call on your behalf.

## What you can do with it

Once connected, the model can call these on your Hermes Cloud org:

| Ask for… | Under the hood |
|----------|----------------|
| "List my cloud agents" | `agents` (list) |
| "What's the status of `<name>`?" | `agents` (get / status) |
| "Roughly what is this instance costing?" | `agents` (cost_estimate) |
| "Start / stop / restart `<name>`" | `agent` (start / stop / restart) |
| "Spin up a new instance called `<name>`" | `agent` (create) |
| "Destroy `<name>`" | `agent` (destroy) |
| "Update the env / image on `<name>`" | `agent` (update_env / update_image) |

Every call runs against **your** org with your Portal identity, and membership is re-checked on each call — the connection can only touch instances you already control from the web UI.

## Prerequisites

- A [Nous Portal](/integrations/nous-portal) account with [Hermes Cloud](https://portal.nousresearch.com/cloud) access (at least one instance, or the ability to create one).
- MCP support installed. If you used the standard install script it's already there; otherwise:

  ```bash
  cd ~/.hermes/hermes-agent
  uv pip install -e ".[mcp]"
  ```

You do **not** need a separate API key or client secret — the server uses OAuth with PKCE, and the login is a browser round-trip.

## Step 1: add the server

```bash
hermes mcp add --url https://portal.nousresearch.com/mcp --auth oauth hermes-cloud
```

`--auth oauth` tells Hermes this is an OAuth-protected HTTP server. On first connect Hermes:

1. Discovers the server's OAuth endpoints automatically (RFC 9728 / 8414 metadata).
2. Registers itself as a client (RFC 7591 Dynamic Client Registration) — no secret to copy.
3. Opens your browser to the Portal to sign in and authorize.
4. Stores the resulting token under `~/.hermes/mcp-tokens/` and reuses it (refresh is automatic).

### Choosing an organization

If your Portal account belongs to **more than one organization**, the browser shows an **org picker** during authorization — pick which org this connection should manage. The choice is made once, in the browser; there's nothing to pass on the command line. Single-org accounts skip this step and bind automatically.

If you ever need to point the connection at a different org, remove and re-add the server (`hermes mcp remove hermes-cloud`, then the `add` command again) and pick the other org in the browser.

## Step 2: verify it connected

```bash
hermes mcp test hermes-cloud
```

Then start (or reload) a session:

```bash
hermes chat
```

```text
/reload-mcp
```

Ask a read-only question to confirm the tools are live:

```text
List my Hermes Cloud agents and their current status.
```

You should get back the same instances you see on the Portal's `/agents` page.

## Step 3: use it

Read-only questions are always safe:

```text
Which of my cloud agents is currently running, and roughly what is each one costing?
```

Lifecycle actions map to plain requests:

```text
Restart the instance called research-bot.
```

```text
Create a new Hermes Cloud instance named scratch, then tell me when it's ready.
```

Hermes reports what each tool returned — the instance list, the new status, the created instance's details — so you can confirm the action landed.

## Configuration

After `hermes mcp add`, the server lives in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  hermes-cloud:
    url: "https://portal.nousresearch.com/mcp"
    auth: oauth
```

No credentials go in `config.yaml` — the OAuth token is kept separately under `~/.hermes/mcp-tokens/`, the same way the Portal refresh token stays out of your config.

### Limiting the tool surface

The server exposes both read (`agents`) and mutating (`agent`) tools. If you want the connection to be **read-only** — list and inspect, but never start/stop/create/destroy — restrict it to the `agents` tool:

```yaml
mcp_servers:
  hermes-cloud:
    url: "https://portal.nousresearch.com/mcp"
    auth: oauth
    tools:
      include: [agents]
```

Run `/reload-mcp` after changing the config. See [Use MCP with Hermes](/guides/use-mcp-with-hermes) for the full filtering model (`include`/`exclude`, `prompts`, `resources`).

## Troubleshooting

### The browser shows an org picker and I'm not sure which to choose

You belong to multiple Portal organizations. Pick the org whose Hermes Cloud instances you want to manage from this connection. If you're unsure, it's the org that owns the instances you see on the Portal `/agents` page. You can re-choose later by removing and re-adding the server.

### "invalid_client" or "unknown client" on connect

The stored client registration no longer matches the server (for example, you connected to a different environment previously). Clear this server's cached OAuth state and re-add it:

```bash
hermes mcp remove hermes-cloud
rm -f ~/.hermes/mcp-tokens/hermes-cloud.*
hermes mcp add --url https://portal.nousresearch.com/mcp --auth oauth hermes-cloud
```

### The tools aren't showing up after adding the server

Reload MCP inside the session and re-check:

```text
/reload-mcp
```

```text
Tell me which MCP-backed tools are available right now.
```

If they're still missing, run `hermes mcp test hermes-cloud` to see the connection error directly.

### It asks me to log in again

OAuth tokens refresh automatically, but if the Portal invalidates your session (password change, revoke, expiry) the next call asks you to re-authorize. Re-run the `hermes mcp add` command — the browser flow re-mints a token.

### Headless / SSH / remote host

The OAuth browser callback runs on the machine where Hermes is running. On a remote host, forward the loopback port over SSH — the same pattern as any other OAuth login. See [OAuth over SSH / Remote Hosts](/guides/oauth-over-ssh).

## See also

- **[Nous Portal](/integrations/nous-portal)** — the subscription, models, and Tool Gateway behind the same login
- **[Use MCP with Hermes](/guides/use-mcp-with-hermes)** — connecting and filtering MCP servers in general
- **[MCP feature overview](/user-guide/features/mcp)** — what MCP is and how Hermes uses it
- **[MCP configuration reference](/reference/mcp-config-reference)** — every `mcp_servers` field, including `auth: oauth`
- **[OAuth over SSH](/guides/oauth-over-ssh)** — logging in from remote or browser-only environments
