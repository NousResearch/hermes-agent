---
sidebar_position: 5
title: "Baizhi Cloud Agent Toolkit"
description: "Connect Baizhi Cloud's hosted Agent Toolkit to Hermes through MCP with a profile-scoped API key"
---

# Baizhi Cloud Agent Toolkit

[Baizhi Cloud Agent Toolkit](https://baizhi.cloud/landing/agent-toolkit)
(百智云) provides hosted tools for search, webpage and document processing,
images, and other tasks through one remote MCP endpoint. This integration
adds external tools to Hermes; it does not change
your model provider or replace Hermes' built-in tools.

| Setting | Value |
|---------|-------|
| Catalog/server name | `agent-toolkit` |
| Endpoint | `https://agent-toolkit.app.baizhi.cloud/mcp` |
| Transport | Streamable HTTP |
| Authentication | API key sent as an `Authorization: Bearer` header |

## Before connecting

Create an API key in the [Agent Toolkit console](https://agent-toolkit.app.baizhi.cloud/).
Grant only the service permissions you need, and review your account's usage
limits and charges: tool calls may consume credits. Calls run on Baizhi Cloud,
and tool inputs are sent to that service; do not send private documents or other sensitive data without the
appropriate permission. Tool availability depends on the service and your
key's permissions.

Choose the Hermes [profile](../user-guide/profiles.md) that should use the
service. The commands below operate on the active profile; use
`hermes -p PROFILE ...` to select another one. In graphical clients, select the
intended profile before installing. To locate its configuration files:

```bash
hermes config path
hermes config env-path
```

This is a profile-scoped integration, not a project-local MCP installation.
It is optional: installing or updating Hermes does not automatically connect
your account or enable this service.

## Install from the MCP catalog

First check whether your installed Hermes catalog includes `agent-toolkit`:

```bash
hermes mcp catalog
```

The catalog entry is available only in Hermes versions that include it. If it
is absent, update to a version containing the entry, or use the
[manual configuration](#manual-configuration) below. Do not assume an older
release has the entry just because it appears in the online documentation.

### CLI

Run `hermes mcp`, select **agent-toolkit**, and follow the credential prompt,
or install it directly:

```bash
hermes mcp install agent-toolkit
```

Enter the **API key only**, without `Bearer`, quotes, or the full header.
Input is hidden. If `MCP_AGENT_TOOLKIT_API_KEY` is already set, Hermes reuses
it instead of prompting again. After connecting, the interactive CLI offers
a checklist of discovered tools. The catalog preselects only
`websearch_search`, `web_scrape`, and `web_extract`; additional tools require
your selection. Reinstalling preserves an existing explicit `tools.include`
selection; review the selection again if you previously enabled all tools.

### Desktop and Web Dashboard

- **Desktop:** open the MCP tab in **Capabilities**, find **agent-toolkit**
  in the catalog, and click **Install**. The first click reveals the API-key
  password field; enter the raw key and click **Install** again.
- **Web Dashboard:** open the **MCP** page, find **agent-toolkit** under
  **Catalog**, and click **Install**. Enter the raw key in the credential
  dialog and confirm **Install**.

These entry points use the same catalog installer. The key is stored in the
selected profile's `.env` as `MCP_AGENT_TOOLKIT_API_KEY`; `config.yaml` keeps
an environment-variable reference rather than the key itself. The `.env`
file is plaintext, not an encrypted credential vault. Restrict access to it
and its backups, and never commit it or paste its contents into chat,
screenshots, issues, or pull requests.

## Verify and choose tools

An **Installed** message confirms that configuration was saved, not that
authentication or every tool call succeeded. Test the connection explicitly:

```bash
hermes mcp test agent-toolkit
```

The output lists the tools discovered from the server. The Web Dashboard's
**Test connection** action provides the same discovery check. Select only
the tools you need in an interactive terminal:

```bash
hermes mcp configure agent-toolkit
```

Start a **new Hermes session** after installation or tool-selection changes.
Ask Hermes to use a specific discovered Agent Toolkit tool for a small,
non-sensitive task, then check that the call succeeds and review its usage
in Baizhi Cloud. Discovery alone does not prove that a billable operation is
authorized or that the account has sufficient credit.

If all tools are selected, Hermes stores no filter and newly added server
tools may also become available. For a restricted set, keep an explicit
selection; see [MCP tool selection](../user-guide/features/mcp.md#tool-selection-at-install-time).

## Manual configuration

Use this path if your version has HTTP MCP support but does not contain the
catalog entry. In the profile's `.env` file shown by `hermes config env-path`,
add the following variable and replace the example value locally:

```dotenv
MCP_AGENT_TOOLKIT_API_KEY=replace-with-your-api-key
```

Merge this entry into the existing `mcp_servers` mapping in the file shown by
`hermes config path`. Preserve every other server and setting:

```yaml
mcp_servers:
  agent-toolkit:
    url: https://agent-toolkit.app.baizhi.cloud/mcp
    headers:
      Authorization: "Bearer ${MCP_AGENT_TOOLKIT_API_KEY}"
    enabled: true
    tools:
      include:
        - websearch_search
        - web_scrape
        - web_extract
```

Do not put the real key in this YAML or in the endpoint URL. Hermes uses
Streamable HTTP for this configuration; do not add `transport: sse` or
`auth: oauth`. Then follow [Verify and choose tools](#verify-and-choose-tools).

## Rotate, disable, or remove

- **Rotate the key:** create a replacement in Baizhi Cloud, update
  `MCP_AGENT_TOOLKIT_API_KEY` in the same profile's `.env` using a local
  editor, and restart the Hermes process using that profile. Test the new
  key before revoking the old one. Reinstalling the catalog entry reuses an
  existing key; `hermes mcp login` and `reauth` are OAuth flows, not API-key
  rotation commands.
- **Disable:** use `hermes mcp` to select the installed entry and disable it,
  use the graphical client's enable/disable control, or set
  `mcp_servers.agent-toolkit.enabled` to `false` in the profile config.
  Start a new session for the change to take effect.
- **Remove:** run `hermes mcp remove agent-toolkit`. This removes the server
  configuration but does not revoke the cloud key. If no longer needed,
  remove the local value with
  `hermes config unset MCP_AGENT_TOOLKIT_API_KEY` and revoke the key in
  Baizhi Cloud. Restart any Hermes processes that had loaded that key.

## Troubleshooting

- **Entry not in the catalog:** use a Hermes version containing this entry,
  or configure the endpoint manually as above.
- **Unauthorized / 401:** check that the key is valid, belongs to the active
  profile, and contains neither a `Bearer` prefix nor extra whitespace.
  An unresolved `${MCP_AGENT_TOOLKIT_API_KEY}` means the variable was not
  available to the Hermes process making the connection.
- **Tools listed but a call fails:** check the key's service permissions,
  account credit, usage limits, and the specific tool's required inputs in
  Baizhi Cloud. Do not repeatedly retry an operation that may incur charges.
- **Connection or discovery fails:** confirm outbound HTTPS access to the
  endpoint, then rerun `hermes mcp test agent-toolkit`. Redact credentials
  before sharing diagnostics.

For transport settings and filtering details, see the
[MCP configuration reference](../reference/mcp-config-reference.md).
