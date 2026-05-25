---
sidebar_position: 7
title: "HermesShare MCP Server"
description: "Configure the restricted Local Files / SMB Share MCP server for HermesShare"
---

# HermesShare Local Files / SMB Share MCP Server

Hermes ships a restricted MCP stdio server for exposing a single local fileshare, such as Brandon's `/home/hermes/HermesShare`, to Hermes sessions as MCP tools. It is intentionally narrower than a general filesystem MCP server: every request is resolved under one configured root, absolute paths are rejected, `..` traversal is rejected, symlink escapes are rejected, and hidden files are excluded by default.

## Tools

When configured as the `hermesshare` MCP server, Hermes discovers tools with the usual `mcp_` prefix:

- `mcp_hermesshare_list_share_files` — list files/directories below the share root.
- `mcp_hermesshare_search_share` — search text-like files and return compact snippets.
- `mcp_hermesshare_read_shared_doc` — read UTF-8 text-like documents with byte limits.
- `mcp_hermesshare_write_shared_doc` — write UTF-8 text documents when writes are enabled.
- `mcp_hermesshare_get_recent_files` — return recently modified files.
- `mcp_hermesshare_sync_status` — report root readability/writability and a compact freshness sample.

## Hermes config snippet

Add this to `~/.hermes/config.yaml` and restart Hermes:

```yaml
mcp_servers:
  hermesshare:
    command: "hermes-share-mcp"
    args: ["--root", "/home/hermes/HermesShare"]
    timeout: 30
    connect_timeout: 30
```

For read-only use:

```yaml
mcp_servers:
  hermesshare:
    command: "hermes-share-mcp"
    args: ["--root", "/home/hermes/HermesShare", "--read-only"]
    timeout: 30
    connect_timeout: 30
```

Environment overrides are also supported for service managers:

- `HERMES_SHARE_MCP_ROOT` — defaults to `/home/hermes/HermesShare`.
- `HERMES_SHARE_MCP_ALLOW_WRITE` — defaults to `true`; set `false` to disable writes.
- `HERMES_SHARE_MCP_INCLUDE_HIDDEN` — defaults to `false`.
- `HERMES_SHARE_MCP_MAX_READ_BYTES`, `HERMES_SHARE_MCP_MAX_SEARCH_BYTES`, `HERMES_SHARE_MCP_MAX_WRITE_BYTES` — positive integer byte caps.

## Verification

1. Confirm the package includes the entry point:

   ```bash
   hermes-share-mcp --print-config-snippet
   ```

2. Add the config snippet, then restart Hermes or run `/reload-mcp` from an existing session.

3. Confirm discovery:

   ```bash
   hermes mcp list
   hermes mcp test hermesshare
   ```

4. Ask Hermes to call `mcp_hermesshare_sync_status`, then list or search a known file under `/home/hermes/HermesShare`.

5. Safety smoke checks: attempts to read `/etc/passwd`, `../outside.md`, a hidden file such as `.secret.md`, or a symlink pointing outside the share root should return `ok: false` with `unsafe_path`.
