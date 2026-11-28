---
sidebar_position: 6
title: "Tutorial: Use Lightpanda with Hermes"
description: "Connect the Lightpanda browser as an MCP server in Hermes Agent, verify the integration, and run a practical website-audit workflow"
---

# Tutorial: Use Lightpanda with Hermes

This guide shows how to connect the [Lightpanda](https://lightpanda.io) browser to Hermes Agent through MCP, verify that the tools load correctly, and use the integration for practical website analysis.

If you are new to MCP in Hermes, read [Use MCP with Hermes](/docs/guides/use-mcp-with-hermes) first. This page is the concrete Lightpanda-specific setup.

## What you get

After finishing this guide, Hermes can use Lightpanda-backed MCP tools such as:

- `mcp_lightpanda_goto`
- `mcp_lightpanda_markdown`
- `mcp_lightpanda_links`
- `mcp_lightpanda_semantic_tree`
- `mcp_lightpanda_interactiveElements`
- `mcp_lightpanda_structuredData`
- `mcp_lightpanda_evaluate`

That gives you a strong workflow for:

- website audits
- structured content extraction
- link and navigation analysis
- UX and QA reviews
- SEO and metadata checks
- agent-driven web research

## Why MCP is the right integration path

Lightpanda can expose multiple interfaces, but for Hermes the best starting point is MCP:

- Hermes stays the agent and orchestration layer
- Lightpanda contributes browser and extraction tools
- Hermes discovers those tools through MCP
- you can keep the integration local and self-hosted

A Hermes skill can document a workflow on top of the integration, but the actual capability comes from the MCP server configuration.

## Prerequisites

You need:

- a working Hermes installation
- MCP support enabled in Hermes
- a Lightpanda binary available locally
- permission to edit your Hermes config file

On a typical local setup, useful Lightpanda binary locations may look like:

- `/usr/local/bin/lightpanda`
- `/home/user/lightpanda`
- `/path/to/browser/zig-out/bin/lightpanda`

## Step 1: verify the Lightpanda binary

First confirm the binary exists and is executable:

```bash
file /path/to/lightpanda
test -x /path/to/lightpanda && echo EXECUTABLE
```

If the second command prints `EXECUTABLE`, the binary is usable.

## Step 2: back up your Hermes config

Before editing your config, create a backup:

```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-$(date +%Y%m%d)-lightpanda
```

## Step 3: add a Lightpanda MCP server

Open `~/.hermes/config.yaml` and add a new `mcp_servers` entry.

### Minimal configuration

```yaml
mcp_servers:
  lightpanda:
    command: "/path/to/lightpanda"
    args: ["mcp"]
    timeout: 120
    connect_timeout: 60
```

### Example with a concrete local path

```yaml
mcp_servers:
  lightpanda:
    command: "/home/user/lightpanda"
    args: ["mcp"]
    timeout: 120
    connect_timeout: 60
```

:::warning
If your config already contains an `mcp_servers:` block, merge the new `lightpanda` entry into the existing block. Do not create a second top-level `mcp_servers:` key.
:::

## Step 4: restart Hermes or reload MCP

After saving the config, restart Hermes.

If you are already inside Hermes and your setup supports it, you can also run:

```text
/reload-mcp
```

## Step 5: verify that Lightpanda tools loaded

A simple validation prompt is:

```text
Tell me which MCP-backed tools are available right now.
```

You should see Lightpanda-related tools in the available tool list.

You can also test with a direct task:

```text
Use the Lightpanda MCP tools to open https://example.com and summarize the page structure.
```

## Step 6: run a practical website audit

A good first workflow looks like this:

1. open the page
2. extract readable page content
3. inspect the semantic tree
4. extract links
5. inspect interactive elements
6. extract structured data
7. run focused JavaScript only if needed

In tool terms, the progression is usually:

1. `mcp_lightpanda_goto`
2. `mcp_lightpanda_markdown` or `mcp_lightpanda_semantic_tree`
3. `mcp_lightpanda_links`
4. `mcp_lightpanda_interactiveElements`
5. `mcp_lightpanda_structuredData`
6. `mcp_lightpanda_evaluate` when you need page-specific JavaScript inspection

### Example prompt

```text
Audit https://example.com using the Lightpanda MCP tools. Summarize:
1. the page purpose
2. top navigation and key links
3. main calls to action
4. structured data and metadata
5. obvious UX or SEO issues
```

## Optional: keep the exposed tool surface narrow

If you later want a more restrictive policy, you can add a tool filter:

```yaml
mcp_servers:
  lightpanda:
    command: "/path/to/lightpanda"
    args: ["mcp"]
    timeout: 120
    connect_timeout: 60
    tools:
      include:
        - goto
        - markdown
        - links
        - semantic_tree
        - interactiveElements
        - structuredData
        - evaluate
```

This is useful when you want the smallest practical browser tool surface.

## Common use cases

Lightpanda + Hermes is especially useful for:

- company website research
- sales and lead qualification
- job and career page monitoring
- content inventory work
- QA walkthroughs for public websites
- SEO audits and schema checks

A very good first project is a company website analysis workflow that produces a structured report for each domain.

## Troubleshooting

### The MCP server does not load

Check:

- the Lightpanda binary path is correct
- the binary is executable
- the YAML is valid
- you did not accidentally duplicate `mcp_servers:`
- Hermes was restarted or MCP was reloaded after the change

### The tools do not appear

Check:

- whether the server actually started successfully
- whether the tools were filtered by `tools.include` or `tools.exclude`
- whether you are looking for the Hermes-exposed names such as `mcp_lightpanda_goto`

### Some websites fail or behave strangely

Lightpanda is still evolving. Some sites may not work correctly because of incomplete browser API coverage, anti-bot controls, or heavy client-side behavior.

Start with simpler public sites first and expand from there.

## Safe usage notes

- back up `~/.hermes/config.yaml` before editing it
- keep your MCP tool surface as small as is practical
- start with public websites before using the setup for sensitive workflows
- remember that MCP configuration enables the integration; docs and skills only describe how to use it

## Related docs

- [Use MCP with Hermes](/docs/guides/use-mcp-with-hermes)
- [MCP (Model Context Protocol)](/docs/user-guide/features/mcp)
- [Tools Reference](/docs/reference/tools-reference)
- [MCP Config Reference](/docs/reference/mcp-config-reference)
