# Feishu/Lark Open Platform Tools

Hermes includes an opt-in `feishu` toolset for automating Feishu/Lark office workflows beyond chat messaging. The tools operate on Docx cloud documents, Bitable/Base project tables, Drive files/folders, and Wiki nodes through Feishu Open Platform APIs.

These tools are intentionally separate from the Feishu messaging gateway:

- `gateway/platforms/feishu.py` handles IM messages, images, files, cards, reactions, and message-resource downloads.
- `tools/feishu_*.py` handles business-office APIs such as documents, bases, Drive metadata, folders, comments, and Wiki nodes.

## Contents

- [Architecture](#architecture)
- [Installation and configuration](#installation-and-configuration)
- [Permissions](#permissions)
- [Tool inventory](#tool-inventory)
- [Usage examples](#usage-examples)
- [Ownership and collaboration](#ownership-and-collaboration)
- [Testing](#testing)
- [Operational notes](#operational-notes)
- [Security considerations](#security-considerations)

## Architecture

| File | Responsibility |
| --- | --- |
| `tools/feishu_openapi.py` | Shared Feishu/Lark client helper, lazy `lark_oapi` import, environment loading, path/query/body request handling, and normalized Open Platform error handling. |
| `tools/feishu_doc_tool.py` | Docx read/create/plain-text append/Markdown-to-native-block append/Markdown-to-native-block replace. |
| `tools/feishu_bitable_tool.py` | Bitable/Base app creation, table rename, field schema management, and record CRUD/search. |
| `tools/feishu_drive_tool.py` | Drive metadata, search, folder creation, comments, and comment replies. |
| `tools/feishu_wiki_tool.py` | Wiki spaces, nodes, node metadata, and node creation. |
| `toolsets.py` | Registers the opt-in `feishu` toolset. |
| `model_tools.py` | Imports the tool modules so the registry can discover their schemas. |
| `skills/productivity/feishu-project-workspace-management/SKILL.md` | Reusable workflow for Feishu project workspaces: Drive folders, native Docx documents, and Bitable project management. |

The tools use Hermes' standard registry pattern (`tools.registry.registry.register`) and return JSON strings via `tool_result` / `tool_error`.

## Installation and configuration

Install Hermes with the Feishu optional dependency:

```bash
uv pip install -e ".[feishu]"
# or, for a full local development environment
uv pip install -e ".[all,dev]"
```

Configure credentials in the environment or `~/.hermes/.env`:

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
FEISHU_DOMAIN=feishu      # use "lark" for https://open.larksuite.com
```

Supported domain values:

| Value | Base URL |
| --- | --- |
| `feishu` | `https://open.feishu.cn` |
| `lark` | `https://open.larksuite.com` |

Enable the `feishu` toolset in the CLI/tool configuration when you want these tools available to the model.

## Permissions

A chat bot permission alone is not sufficient. The Feishu/Lark app must have the relevant Open Platform scopes for the resources it will operate on.

Recommended permission groups:

- Docs/Docx: create, read, update document blocks, raw content read.
- Bitable/Base: create apps, manage tables/fields, read/write records.
- Drive: search files, metadata, create folders, comments, permissions/members if ownership transfer is required.
- Wiki: list spaces/nodes and create/read nodes.

Target resources must also be visible to the app. For user-owned resources, share the document/base/folder/wiki space with the app or place new resources in a folder the app can access.

## Tool inventory

### Docx

| Tool | Purpose |
| --- | --- |
| `feishu_doc_read` | Read raw content from a Docx cloud document. |
| `feishu_doc_create` | Create a Docx cloud document, optionally under a Drive folder token. |
| `feishu_doc_append_text` | Append plain paragraph text. Use only when plain text is desired. |
| `feishu_doc_append_markdown` | Convert Markdown to native Docx blocks and append it. Preferred for formatted content. |
| `feishu_doc_replace_markdown` | Delete current document body and replace it with Markdown converted to native Docx blocks. |

### Bitable/Base

| Tool | Purpose |
| --- | --- |
| `feishu_bitable_create_app` | Create a Base app, optionally under a Drive folder token. |
| `feishu_bitable_list_tables` | List tables in a Base. |
| `feishu_bitable_update_table` | Rename/update a table. |
| `feishu_bitable_get_fields` | Read a table's field schema. |
| `feishu_bitable_create_field` | Create a field. Common type codes: `1` Text, `3` SingleSelect, `5` DateTime, `17` Attachment. |
| `feishu_bitable_update_field` | Update a field's name/type/property. |
| `feishu_bitable_search_records` | Search records with optional view/filter/sort/field selection. |
| `feishu_bitable_create_record` | Create one record. |
| `feishu_bitable_update_record` | Update one record. |
| `feishu_bitable_delete_record` | Delete one record by explicit record ID. |

### Drive

| Tool | Purpose |
| --- | --- |
| `feishu_drive_get_meta` | Batch-query Drive metadata for a token/type. |
| `feishu_drive_search_files` | Search Drive files by keyword. |
| `feishu_drive_create_folder` | Create a Drive folder. |
| `feishu_drive_list_comments` | List comments for a Drive-backed file. |
| `feishu_drive_reply_comment` | Reply to a comment. |

### Wiki

| Tool | Purpose |
| --- | --- |
| `feishu_wiki_list_spaces` | List visible Wiki spaces. |
| `feishu_wiki_list_nodes` | List nodes in a Wiki space. |
| `feishu_wiki_get_node` | Get metadata for a Wiki node. |
| `feishu_wiki_create_node` | Create a Wiki node, optionally mounting an existing object token. |

## Usage examples

### Create a formatted Docx document

Use native Docx block conversion for professional documents. Do **not** write Markdown headings with `feishu_doc_append_text`; that produces literal `#` text instead of real headings.

```python
from tools import feishu_doc_tool as doc

created = doc._handle_doc_create({"title": "Project Overview"})
# extract document_id/doc_token from the JSON result

doc._handle_doc_replace_markdown({
    "doc_token": "DOCX_TOKEN",
    "markdown": """
# Project Overview

## Goals

- Define the MVP
- Ship the first milestone
""".strip(),
})
```

Implementation details:

- Markdown is converted via `POST /open-apis/docx/v1/documents/blocks/convert`.
- Converted blocks are reordered by `first_level_block_ids` before insertion.
- Body replacement deletes children via `/children/batch_delete`.
- Block insertion is chunked to respect the 50-child-per-request Feishu limit.

### Create a Bitable project management base

A project management base typically includes:

- `任务名称` — Text
- `状态` — SingleSelect: 未开始, 进行中, 阻塞, 待评审, 已完成
- `计划完成日期` — DateTime
- `模块` — SingleSelect
- `优先级` — SingleSelect: P0, P1, P2, P3
- `负责人` — Text
- `任务说明` — Text
- `验收标准` — Text
- `风险/阻塞` — Text
- `最近更新` — DateTime

Use `feishu_bitable_create_app`, then rename the default table, create/update fields, and seed records.

### Build a full project workspace

Use the `feishu-project-workspace-management` skill for repeatable project setup:

1. Create a Drive project folder and subfolders.
2. Create native Docx project docs (overview, PRD, architecture, meeting notes).
3. Create a Bitable/Base for task/project management.
4. Transfer ownership to the target project owner and keep the app as editor.
5. Verify by reading back docs, listing Bitable tables/fields/records, and checking Drive metadata.

## Ownership and collaboration

For Drive-backed documents and bases, the live-validated owner-transfer flow is:

1. Grant the target user edit access:

```http
PATCH /open-apis/drive/v1/permissions/:token/members/:member_id?type=<docx|bitable>&member_type=openid
{"perm": "edit"}
```

2. Transfer ownership:

```http
POST /open-apis/drive/v1/permissions/:token/members/transfer_owner?type=<docx|bitable>&need_notification=false
{"member_type": "openid", "member_id": "<open_id>"}
```

3. Verify metadata:

```http
POST /open-apis/drive/v1/metas/batch_query
{"request_docs": [{"doc_token": "...", "doc_type": "docx"}]}
```

`owner_id` should equal the target user's `open_id`.

## Testing

Focused test suite:

```bash
source venv/bin/activate
python -m pytest \
  tests/tools/test_feishu_openapi.py \
  tests/tools/test_feishu_doc_tool.py \
  tests/tools/test_feishu_bitable_tool.py \
  tests/tools/test_feishu_drive_tool.py \
  tests/tools/test_feishu_wiki_tool.py \
  -q
```

Tool discovery smoke test:

```bash
source venv/bin/activate
python - <<'PY'
import model_tools
from toolsets import resolve_toolset
required = {
    "feishu_doc_read", "feishu_doc_create", "feishu_doc_append_text",
    "feishu_doc_append_markdown", "feishu_doc_replace_markdown",
    "feishu_bitable_create_app", "feishu_bitable_list_tables",
    "feishu_bitable_update_table", "feishu_bitable_get_fields",
    "feishu_bitable_create_field", "feishu_bitable_update_field",
    "feishu_bitable_search_records", "feishu_bitable_create_record",
    "feishu_bitable_update_record", "feishu_bitable_delete_record",
    "feishu_drive_get_meta", "feishu_drive_search_files",
    "feishu_drive_create_folder", "feishu_drive_list_comments",
    "feishu_drive_reply_comment", "feishu_wiki_list_spaces",
    "feishu_wiki_list_nodes", "feishu_wiki_get_node", "feishu_wiki_create_node",
}
assert required <= set(model_tools.get_all_tool_names())
assert required <= set(resolve_toolset("feishu"))
print("schema discovery ok")
PY
```

## Operational notes

- `doc_token`, `file_token`, `app_token`, `table_id`, `record_id`, and Wiki node tokens are different identifiers and are not interchangeable.
- Feishu IM attachment `file_key` values are not Drive file tokens.
- Bitable date/datetime record values should use millisecond timestamps.
- Read Bitable field schemas before writing records; field values are type-specific.
- Keep tool modules import-safe without credentials. `lark_oapi` is imported lazily inside the shared OpenAPI helper.

## Security considerations

- Do not log `FEISHU_APP_SECRET` or tenant access tokens.
- Avoid default-enabling the `feishu` toolset for users who have not configured Feishu credentials.
- Prefer explicit record IDs for destructive operations such as record deletion.
- Report permission/ownership transfer failures explicitly; do not silently leave app-owned project artifacts when a human owner is required.