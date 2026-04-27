---
title: "Siyuan"
sidebar_label: "Siyuan"
description: "SiYuan Note API for searching, reading, creating, and managing blocks and documents in a self-hosted knowledge base via curl"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Siyuan

SiYuan Note API for searching, reading, creating, and managing blocks and documents in a self-hosted knowledge base via curl.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/siyuan` |
| Path | `optional-skills/productivity/siyuan` |
| Version | `1.0.0` |
| Author | FEUAZUR |
| License | MIT |
| Tags | `SiYuan`, `Notes`, `Knowledge Base`, `PKM`, `API` |
| Related skills | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SiYuan Note API

Use the [SiYuan](https://github.com/siyuan-note/siyuan) kernel API via curl to search, read, create, update, and delete blocks and documents in a self-hosted knowledge base. No extra tools needed -- just curl and an API token.

## Prerequisites

1. Install and run SiYuan (desktop or Docker)
2. Get your API token: **Settings > About > API token**
3. Store it in `~/.hermes/.env`:
   ```
   SIYUAN_TOKEN=your_token_here
   SIYUAN_URL=http://127.0.0.1:6806
   ```
   `SIYUAN_URL` defaults to `http://127.0.0.1:6806` if not set.

## API Basics

All SiYuan API calls are **POST with JSON body**. Every request follows this pattern:

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/..." \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'
```

Responses are JSON with this structure:
```json
{"code": 0, "msg": "", "data": { ... }}
```
`code: 0` means success. Any other value is an error -- check `msg` for details.

**ID format:** SiYuan IDs look like `20210808180117-6v0mkxr` (14-digit timestamp + 7 alphanumeric chars).

## Quick Reference

| Operation | Endpoint |
|-----------|----------|
| Full-text search | `/api/search/fullTextSearchBlock` |
| SQL query | `/api/query/sql` |
| Read block | `/api/block/getBlockKramdown` |
| Read children | `/api/block/getChildBlocks` |
| Get path | `/api/filetree/getHPathByID` |
| Get attributes | `/api/attr/getBlockAttrs` |
| List notebooks | `/api/notebook/lsNotebooks` |
| List documents | `/api/filetree/listDocsByPath` |
| Create notebook | `/api/notebook/createNotebook` |
| Create document | `/api/filetree/createDocWithMd` |
| Append block | `/api/block/appendBlock` |
| Update block | `/api/block/updateBlock` |
| Rename document | `/api/filetree/renameDocByID` |
| Set attributes | `/api/attr/setBlockAttrs` |
| Delete block | `/api/block/deleteBlock` |
| Delete document | `/api/filetree/removeDocByID` |
| Export as Markdown | `/api/export/exportMdContent` |

## Common Operations

### Search (Full-Text)

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/search/fullTextSearchBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "meeting notes", "page": 0}' | jq '.data.blocks[:5]'
```

### Search (SQL)

Query the blocks database directly. Only SELECT statements are safe.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/query/sql" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stmt": "SELECT id, content, type, box FROM blocks WHERE content LIKE '\''%keyword%'\'' AND type='\''p'\'' LIMIT 20"}' | jq '.data'
```

Useful columns: `id`, `parent_id`, `root_id`, `box` (notebook ID), `path`, `content`, `type`, `subtype`, `created`, `updated`.

### Read Block Content

Returns block content in Kramdown (Markdown-like) format.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/getBlockKramdown" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data.kramdown'
```

### Read Child Blocks

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/getChildBlocks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### Get Human-Readable Path

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/getHPathByID" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### Get Block Attributes

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/attr/getBlockAttrs" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### List Notebooks

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/lsNotebooks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.data.notebooks[] | {id, name, closed}'
```

### List Documents in a Notebook

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/listDocsByPath" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"notebook": "NOTEBOOK_ID", "path": "/"}' | jq '.data.files[] | {id, name}'
```

### Create a Document

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/createDocWithMd" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notebook": "NOTEBOOK_ID",
    "path": "/Meeting Notes/2026-03-22",
    "markdown": "# Meeting Notes\n\n- Discussed project timeline\n- Assigned tasks"
  }' | jq '.data'
```

### Create a Notebook

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/createNotebook" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My New Notebook"}' | jq '.data.notebook.id'
```

### Append Block to Document

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/appendBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "parentID": "DOCUMENT_OR_BLOCK_ID",
    "data": "New paragraph added at the end.",
    "dataType": "markdown"
  }' | jq '.data'
```

Also available: `/api/block/prependBlock` (same params, inserts at the beginning) and `/api/block/insertBlock` (uses `previousID` instead of `parentID` to insert after a specific block).

### Update Block Content

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/updateBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "BLOCK_ID",
    "data": "Updated content here.",
    "dataType": "markdown"
  }' | jq '.data'
```

### Rename a Document

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/renameDocByID" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID", "title": "New Title"}'
```

### Set Block Attributes

Custom attributes must be prefixed with `custom-`:

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/attr/setBlockAttrs" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "BLOCK_ID",
    "attrs": {
      "custom-status": "reviewed",
      "custom-priority": "high"
    }
  }'
```

### Delete a Block

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/deleteBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "BLOCK_ID"}'
```

To delete a whole document: use `/api/filetree/removeDocByID` with `{"id": "DOC_ID"}`.
To delete a notebook: use `/api/notebook/removeNotebook` with `{"notebook": "NOTEBOOK_ID"}`.

### Export Document as Markdown

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/export/exportMdContent" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID"}' | jq -r '.data.content'
```

## Block Types

Common `type` values in SQL queries:

| Type | Description |
|------|-------------|
| `d` | Document (root block) |
| `p` | Paragraph |
| `h` | Heading |
| `l` | List |
| `i` | List item |
| `c` | Code block |
| `m` | Math block |
| `t` | Table |
| `b` | Blockquote |
| `s` | Super block |
| `html` | HTML block |

## Pitfalls

- **All endpoints are POST** -- even read-only operations. Do not use GET.
- **SQL safety**: only use SELECT queries. INSERT/UPDATE/DELETE/DROP are dangerous and should never be sent.
- **ID validation**: IDs match the pattern `YYYYMMDDHHmmss-xxxxxxx`. Reject anything else.
- **Error responses**: always check `code != 0` in responses before processing `data`.
- **Large documents**: block content and export results can be very large. Use `LIMIT` in SQL and pipe through `jq` to extract only what you need.
- **Notebook IDs**: when working with a specific notebook, get its ID first via `lsNotebooks`.

## Alternative: MCP Server

If you prefer a native integration instead of curl, install the SiYuan MCP server:

```yaml
# In ~/.hermes/config.yaml under mcp_servers:
mcp_servers:
  siyuan:
    command: npx
    args: ["-y", "@porkll/siyuan-mcp"]
    env:
      SIYUAN_TOKEN: "your_token"
      SIYUAN_URL: "http://127.0.0.1:6806"
```
