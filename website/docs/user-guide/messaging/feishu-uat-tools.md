---
sidebar_position: 12
title: "Feishu User Identity Tools (UAT)"
description: "Use Feishu APIs with your personal user identity instead of the bot's app identity"
---

# Feishu User Identity Tools (UAT)

Beyond the standard bot identity, Hermes can make Feishu API calls using your personal user credentials — a feature called User-Access-Token (UAT) authentication. This unlocks 30+ additional tools that access your calendar, documents, tasks, and more using your own identity.

## Why Dual Identity?

Feishu has two types of credentials:

| Identity Type | Who | Access Level | Use Case |
|---|---|---|---|
| **Tenant-Access-Token (TAT)** | The bot app | App-scoped permissions | Respond to chats, send messages, read what users share |
| **User-Access-Token (UAT)** | You (the user) | Your personal permissions | Access your own calendar, documents, tasks, files |

With UAT enabled, the agent gains 30+ additional tools to:
- Read/write your calendar events
- Create and edit Feishu documents
- Manage Feishu bitables (databases)
- Access your Google Sheets via Feishu integration
- Create and update tasks
- Search your workspace
- And more — using your identity

All UAT calls are audited by Feishu (you can see them in Feishu's audit log). Your personal credentials are stored locally in `~/.hermes/feishu_uat.json` with mode `0600` (readable only by you).

## Setup (5 Minutes)

### Step 1: Create a Feishu OAuth App (Recommended Scopes)

First, create or configure a Feishu app to support user identity. Open the Feishu Developer Console:

- **Feishu China:** [https://open.feishu.cn/](https://open.feishu.cn/)
- **Lark International:** [https://open.larksuite.com/](https://open.larksuite.com/)

In your app's **Permissions & Scopes** section, enable these recommended scopes:

```
calendar:calendar                — Read/write calendar events
calendar:freebusy:readonly       — Check calendar free/busy
drive:drive                      — Access files and folders
docs:document:readonly           — Read documents
docs:document                    — Edit documents
bitable:app                      — Access Feishu bitables
wiki:wiki:readonly               — Read wiki pages
sheets:spreadsheet               — Read/write spreadsheets
task:task:write                  — Create and edit tasks
task:task:read                   — Read tasks
im:message:send_as_user          — Send messages as you
im:chat:readonly                 — Read chat info
search:search                    — Search workspace
authen:user.employee_id:read     — Get your user info
contact:user.base:readonly       — Access contact info
```

Not all tools require all scopes — see the per-tool list below.

### Step 2: Run the Device Flow Setup

Run this command:

```bash
hermes setup feishu-uat --scope "calendar:calendar drive:drive docs:document bitable:app wiki:wiki:readonly sheets:spreadsheet task:task:write im:message:send_as_user im:chat:readonly search:search"
```

Replace the scope list with your actual scopes if you customized them.

Alternatively, if `FEISHU_APP_ID` is already set in your environment:

```bash
hermes setup feishu-uat
```

This uses the app ID from your Feishu bot configuration.

### Step 3: Scan QR Code + Authorize

The setup command will:
1. Display a QR code in the terminal
2. Show a user code and link as fallback
3. Prompt you to scan the QR code with your **Feishu mobile app**

Scan the QR code using Feishu (not WeChat). A browser window will open. Click **Authorize** to grant Hermes access to the requested scopes.

### Step 4: Verify the Token File

After successful authorization, check that the token was saved:

```bash
ls -la ~/.hermes/feishu_uat.json
```

If it exists and is readable, you're done. The file contains your access and refresh tokens (encrypted, readable only by your user).

```json
{
  "app_id": "cli_xxx",
  "user_open_id": "ou_xxx",
  "access_token": "***",
  "refresh_token": "***",
  "expires_at": 1234567890000,
  "refresh_expires_at": 1234567890000,
  "scope": "calendar:calendar drive:drive ...",
  "granted_at": 1234567890000
}
```

## Available UAT Tools

Hermes exposes 30+ UAT tools across 11 families. Each tool automatically uses your user identity when called.

### Calendar (4 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_calendar_list_events` | `calendar:calendar` | List your calendar events for a date range |
| `feishu_calendar_get_event` | `calendar:calendar` | Get details of a specific event |
| `feishu_calendar_create_event` | `calendar:calendar` | Create a new calendar event |
| `feishu_calendar_freebusy` | `calendar:freebusy:readonly` | Check your free/busy availability |

### Bitable (6 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_bitable_list_apps` | `bitable:app` | List all Feishu bitables you have access to |
| `feishu_bitable_list_tables` | `bitable:app` | List tables in a specific bitable |
| `feishu_bitable_list_records` | `bitable:app` | Query records from a table with filters |
| `feishu_bitable_search_records` | `bitable:app` | Full-text search within a table |
| `feishu_bitable_create_record` | `bitable:app` | Create a new record in a table |
| `feishu_bitable_update_record` | `bitable:app` | Update an existing record |

### Drive (5 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_drive_list_files` | `drive:drive` | List files in a folder |
| `feishu_drive_get_file_meta` | `drive:drive` | Get metadata about a file/folder |
| `feishu_drive_move_file` | `drive:drive` | Move a file to a different folder |
| `feishu_drive_delete_file` | `drive:drive` | Delete a file from your drive |
| `feishu_drive_copy_file` | `drive:drive` | Copy a file to a new location |

### Docs (1 tool)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_doc_read` | `docs:document:readonly` | Read the full content of a Feishu document |

### Instant Messaging (2 tools — send as you)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_send_message_as_user` | `im:message:send_as_user` | Send a message to a user/group as yourself |
| `feishu_reply_message_as_user` | `im:message:send_as_user` | Reply to a thread as yourself |

### Chat (2 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_chat_get_info` | `im:chat:readonly` | Get info about a chat (members, name, etc.) |
| `feishu_chat_list_members` | `im:chat:readonly` | List all members in a chat |

### Search (2 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_search_global` | `search:search` | Global workspace search across all documents |
| `feishu_search_message` | `search:search` | Search messages in chats |

### Sheets (3 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_sheets_read_range` | `sheets:spreadsheet` | Read a range of cells from a sheet |
| `feishu_sheets_write_range` | `sheets:spreadsheet` | Write values to a range in a sheet |
| `feishu_sheets_append_rows` | `sheets:spreadsheet` | Append new rows to the bottom of a sheet |

### Task (5 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_task_list` | `task:task:read` | List all your tasks with optional filters |
| `feishu_task_get` | `task:task:read` | Get details of a specific task |
| `feishu_task_create` | `task:task:write` | Create a new task |
| `feishu_task_update` | `task:task:write` | Update task title, status, or assignee |
| `feishu_task_add_comment` | `task:task:write` | Add a comment to a task |

### Wiki (2 tools)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_wiki_search` | `wiki:wiki:readonly` | Search your wiki space |
| `feishu_wiki_get_node` | `wiki:wiki:readonly` | Retrieve the full content of a wiki page |

### User Info (1 tool)

| Tool | Scope Required | Description |
|------|---|---|
| `feishu_get_my_user_info` | `authen:user.employee_id:read` | Get your Feishu user profile (name, ID, email) |

**Total: 33 UAT tools**

## Switching Between Identities

Internally, tools use `feishu_oapi_client.FeishuClient` to switch between identities:

```python
# Bot identity (TAT) — always available
client = FeishuClient.for_tenant()
# Makes calls with the app's access token

# User identity (UAT) — only if ~/.hermes/feishu_uat.json exists
client = FeishuClient.for_user()
# Reads UAT from disk and makes calls with your token
```

Most UAT tools automatically handle this. If you need to make raw API calls, use:

```python
client = FeishuClient.for_user()
client.do_request("GET", "/open-apis/calendar/v4/calendars", ...)
```

## Error Handling

The UAT system exposes four semantic error classes to help you understand what went wrong:

| Error | Cause | How to Fix |
|---|---|---|
| `NeedAuthorizationError` | No UAT file found; user never authorized | Run `hermes setup feishu-uat` and scan the QR code |
| `AppScopeMissingError` | (errcode 99991672) App lacks an API scope | Add the missing scope in Feishu Developer Console, then re-run setup |
| `UserAuthRequiredError` | (errcode 99991679) User auth expired or insufficient scopes | Run `hermes setup feishu-uat` to re-authorize |
| `UserScopeInsufficientError` | Token valid but missing specific scopes | Re-run setup with additional scopes |

When a tool fails with one of these errors, the agent will typically show you the error message and next steps.

## Streaming Cards (Optional)

When enabled, Hermes can send interactive update cards that stream the agent's response as it's being generated, rather than waiting for the full response to complete.

### Enable Streaming Cards

Add this to your `~/.hermes/config.yaml`:

```yaml
platforms:
  feishu:
    extra:
      streaming_card:
        enabled: true
```

Default is `false` (disabled). Streaming cards are opt-in and don't affect existing `update_message` behavior.

### How It Works

1. Hermes sends an initial interactive card to the chat
2. As the LLM generates tokens, the card updates in real-time
3. Users see the response evolving before their eyes
4. When complete, the card is finalized

This is purely visual — the actual tool execution and permissions remain unchanged.

## Troubleshooting

### UAT token expired

```
UserAuthRequiredError: User 'ou_xxx' missing scopes [...] for 'feishu_calendar_list_events'
```

**Fix:** Re-authorize with fresh scopes:

```bash
hermes setup feishu-uat --force
```

### App scope missing

```
AppScopeMissingError: App 'cli_xxx' missing scopes [drive:drive] for API 'feishu_drive_list_files'
```

**Fix:** 
1. Open Feishu Developer Console
2. Go to your app's **Permissions & Scopes**
3. Enable the missing scope (e.g., `drive:drive`)
4. Re-run setup:

```bash
hermes setup feishu-uat --force
```

### No UAT file / Setup failed

```bash
ls -la ~/.hermes/feishu_uat.json
```

If the file doesn't exist:
1. Ensure `FEISHU_APP_ID` is set (from your bot credentials)
2. Run setup again with QR code scan
3. Check that you scanned with the **Feishu mobile app**, not WeChat

### Token file permissions issue

If you see permission errors:

```bash
chmod 600 ~/.hermes/feishu_uat.json
```

### Tool not found / unavailable

If a tool is not available in your agent:

1. Check that the corresponding scope is enabled in your UAT
2. Check that your Feishu app has the scope enabled
3. Verify the UAT is not expired:

```bash
cat ~/.hermes/feishu_uat.json | grep expires_at
```

Compare `expires_at` (in milliseconds) to the current time.

## Security Notes

- **Token storage:** UAT tokens are stored in `~/.hermes/feishu_uat.json` with mode `0600` (owner-only readable)
- **Audit trail:** All UAT calls are logged in your Feishu workspace's audit log
- **Scope limits:** The agent can only access APIs for which you granted scopes
- **Refresh tokens:** Hermes stores both access and refresh tokens; access tokens are automatically refreshed if they expire (within the refresh token's 30-day TTL)
- **Local-only:** Tokens never leave your machine; Hermes communicates directly with Feishu's OAPI endpoints

## Related

- [Feishu / Lark Setup](./feishu.md) — Configure the bot's TAT identity
- [Feishu Platform Adapter](./feishu.md) — Full gateway documentation
- [Feishu Developer Docs](https://open.feishu.cn/document) — Official Feishu API reference
