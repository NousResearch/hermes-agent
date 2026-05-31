---
sidebar_position: 7
title: "Run Hermes as an MCP server"
description: "Expose Hermes Agent conversations, messaging, approvals, and events to MCP clients"
---

# Run Hermes as an MCP server

Hermes can run as a local MCP server so other MCP-capable clients can talk to a running Hermes workspace.

Use this when you want another agent or IDE to:

- list and inspect Hermes conversations
- send messages through Hermes-supported platforms
- poll gateway events
- hand work to Hermes without leaving your MCP client
- keep Hermes' memory, skills, cron jobs, and approval flows as the operating layer

## Start the server

From a machine with Hermes installed by the standard installer, MCP support is already included.
If you installed from PyPI manually, include the MCP extra:

```bash
pip install "hermes-agent[mcp]"
```

Then start the stdio server:

```bash
hermes mcp serve
```

For debugging:

```bash
hermes mcp serve --verbose
```

The server uses stdio transport. Configure your MCP client to launch `hermes mcp serve` as the command.

## Example client configuration

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

If Hermes is installed in a virtual environment or custom location, use the absolute path to the `hermes` executable.

## Package registry metadata

This repository includes `server.json` for MCP registry discovery. The package is published on PyPI as `hermes-agent`, and the metadata pins the fixed `mcp serve` package arguments so clients launch the server instead of plain Hermes chat. The intended runtime command is:

```bash
hermes mcp serve
```

`server.json` uses `"identifier": "hermes-agent[mcp]"` so Python package runners that understand PyPI extras install the optional MCP SDK before launching Hermes. If you install Hermes from PyPI yourself, include the MCP extra before configuring a client:

```bash
pip install "hermes-agent[mcp]"
```

If your MCP client cannot install PyPI extras automatically, install that extra first and configure the client with the explicit command shown above.

When publishing or validating registry metadata, use the server name:

```text
io.github.nousresearch/hermes-agent
```

## Safety notes

Hermes can bridge to real messaging platforms and tools. Treat an MCP client connected to Hermes as a privileged operator:

- only connect trusted MCP clients
- keep command approvals enabled for risky actions
- prefer dry-run or preview flows before sending messages
- review outbound communication policies for shared team workspaces
- avoid exposing secrets through environment variables unless they are required

## Commercial support

Teams that want Hermes deployed with MCP, messaging gateways, internal tools, or custom skills can use the optional [commercial support](https://github.com/NousResearch/hermes-agent/blob/main/COMMERCIAL_SUPPORT.md) path described in the repository.
