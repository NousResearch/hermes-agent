---
sidebar_position: 7
title: "List Hermes as an MCP server"
description: "Submission checklist and launch copy for listing Hermes Agent's built-in MCP server in registries and directories."
---

# List Hermes as an MCP server

Hermes can run as an MCP server with `hermes mcp serve`, exposing its cross-platform conversation bridge to MCP clients. This guide keeps the listing process repeatable for registry submissions, community directories, and launch posts.

Use it when submitting Hermes to the official MCP Registry, Glama, mcpservers.org, MCP.so, Docker MCP Catalog, or similar directories.

## Listing facts

| Field | Value |
| --- | --- |
| Product | Hermes Agent |
| MCP server command | `hermes mcp serve` |
| Transport | stdio |
| Package | `hermes-agent` on PyPI |
| Recommended install | `pipx install "hermes-agent[mcp]"` or Hermes install script |
| Source | `https://github.com/NousResearch/hermes-agent` |
| Docs | `https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp#running-hermes-as-an-mcp-server` |
| Category | agent runtime, messaging bridge, automation, personal AI assistant |
| License | MIT |

## Short description

Hermes Agent is a self-improving AI agent that also runs as an MCP server, letting MCP clients read conversation history, list channels, send messages, and manage approvals across connected platforms like Discord, Slack, Telegram, email, and more.

## Directory description

Hermes Agent is an open-source AI agent framework that can run in terminals, messaging platforms, IDEs, and as an MCP server. In MCP server mode, `hermes mcp serve` exposes a stdio bridge for connected clients to list conversations, read recent message history, poll for events, list available channels, send platform messages through Hermes, and respond to pending approval requests.

Use Hermes when you want one local MCP server to bridge multiple messaging and automation surfaces while preserving Hermes's existing approvals, gateway configuration, sessions, and tool ecosystem.

## Key use cases

- Let Claude Code, Cursor, Codex, and other MCP clients read recent Hermes conversations.
- Use MCP clients to draft and operator-review messages before sending them to Discord, Slack, Telegram, email, or other configured Hermes gateway targets.
- Poll for near-real-time conversation events from a coding agent or operator cockpit.
- Centralize approval workflows so external MCP clients can see and resolve pending Hermes approvals.
- Reuse one Hermes gateway instead of configuring every MCP client for every messaging platform.

## Capabilities to mention

Hermes's MCP server currently exposes these tool families:

- conversation discovery and details
- message history reads
- attachment metadata fetching
- event polling and long polling
- outbound message sending through configured platforms
- channel/target listing
- pending approval listing and response

Do not describe it as a general-purpose remote code execution MCP server. Hermes itself has tools for code and automation, but this MCP server surface is specifically the Hermes conversation, channel, event, and approval bridge.

## Install and client config

For clients that understand PyPI extras, prefer the package with the MCP extra:

```bash
pipx install "hermes-agent[mcp]"
```

Then configure the MCP client to launch Hermes with fixed arguments:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

If a client or registry cannot parse PyPI extras, inject the same pinned MCP runtime dependencies that the `[mcp]` extra declares before running the same command:

```bash
pipx install hermes-agent
pipx inject hermes-agent mcp==1.26.0 starlette==1.0.1
```

For source installs, run:

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
hermes mcp serve
```

## Official MCP Registry checklist

Before publishing to `registry.modelcontextprotocol.io`:

- Add or verify a root `server.json` that uses the official schema URL.
- Use the canonical GitHub namespace `io.github.nousresearch/hermes-agent` unless a domain namespace has been authenticated.
- Reference the PyPI package `hermes-agent` with `registryType: "pypi"` and fixed package arguments `mcp` and `serve`.
- Include the PyPI package ownership marker in the package README: `mcp-name: io.github.nousresearch/hermes-agent`.
- Validate `server.json` against the declared schema.
- Confirm `hermes mcp serve` starts from an installed wheel or editable install with the `[mcp]` extra.
- Document the fallback install path for clients that cannot parse PyPI extras.

## Submission notes by directory

### Official MCP Registry

Submit only after the root `server.json` and README ownership marker have merged and a release including both has been published to PyPI. Use the package arguments `mcp` and `serve` so clients do not invoke plain `hermes`.

### Glama

Use the directory description above. Emphasize that Hermes is an MCP server for messaging, channel discovery, events, and approvals. Link to the MCP guide and GitHub repository.

### mcpservers.org and MCP.so

Use the short description plus the key use cases. Pick categories such as `Automation`, `Communication`, `Agent Framework`, or `Developer Tools` depending on the directory taxonomy.

### Docker MCP Catalog

Submit only after there is an official container image with the required MCP server annotation matching the server name. Until then, list the PyPI package path instead of claiming Docker availability.

## Launch post copy

### Show HN title

Show HN: Hermes Agent — an open-source self-improving agent that can also run as an MCP server

### Show HN body

Hermes Agent is an open-source AI agent framework from Nous Research. It runs in your terminal, messaging platforms, and IDEs, and now it can also run as an MCP server with `hermes mcp serve`.

The MCP server mode lets MCP clients such as Claude Code, Cursor, or Codex use Hermes as a conversation and channel bridge: list conversations, read recent message history, poll for events, list available channels, send messages through configured platforms, and respond to pending approval requests.

The goal is to make Hermes a safe operator layer between coding agents and the places humans actually coordinate: Discord, Slack, Telegram, email, and other gateway platforms.

Repo: https://github.com/NousResearch/hermes-agent
Docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp#running-hermes-as-an-mcp-server

### Product Hunt tagline

Self-improving AI agent for terminal, messaging, IDEs, and MCP clients.

### Product Hunt description

Hermes Agent is an open-source agent runtime that works across your terminal, messaging platforms, and IDEs. Its MCP server mode lets other MCP clients use Hermes as a conversation, channel, event, and approval bridge for connected platforms like Slack, Discord, Telegram, and email.

## Support CTA placeholder

Do not include commercial-support copy in directory submissions until the canonical support policy from #31292 has merged and been published. If a directory requires a support field before then, link to the GitHub repository issues/discussions only and avoid promises about paid support, SLAs, managed hosting, security reviews, or production hardening.

## Safety notes

- Do not submit paid promotions or paid directory placements without explicit approval.
- Do not mass-post launch copy across communities.
- Tailor each post to the community and disclose affiliation.
- For outbound messages, draft first and require human review before sending.
- Do not claim support for platforms or transports beyond the current MCP server implementation.
