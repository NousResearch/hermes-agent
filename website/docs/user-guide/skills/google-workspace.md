---
sidebar_position: 2
sidebar_label: "Google Workspace"
title: "Google Workspace — Gmail, Calendar, Drive, Sheets & Docs"
description: "Send email, manage calendar events, search Drive, read/write Sheets, and access Docs — all through OAuth2-authenticated Google APIs"
---

# Google Workspace Skill

Gmail, Calendar, Drive, Contacts, Sheets, and Docs integration for Hermes. Uses OAuth2 with automatic token refresh. Prefers the [Google Workspace CLI (`gws`)](https://github.com/nicholasgasior/gws) when available for broader coverage, and falls back to Google's Python client libraries otherwise.

**Skill path:** `skills/productivity/google-workspace/`

## Setup

The setup is fully agent-driven — ask Hermes to set up Google Workspace and it walks you through each step. The flow:

1. **Create a Google Cloud project** and enable the required APIs (Gmail, Calendar, Drive, Sheets, Docs, People)
2. **Create OAuth 2.0 credentials** (Desktop app type) and download the client secret JSON
3. **Authorize** — Hermes generates an auth URL, you approve in the browser, paste back the redirect URL
4. **Done** — token auto-refreshes from that point on

:::tip Email-only users
If you only need email (no Calendar/Drive/Sheets), use the **himalaya** skill instead — it works with a Gmail App Password and takes 2 minutes. No Google Cloud project needed.
:::

## Gmail

### Searching

```bash
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"
```

Returns JSON with `id`, `from`, `subject`, `date`, `snippet`, and `labels` for each message.

### Reading

```bash
$GAPI gmail get MESSAGE_ID
```

Returns the full message body as text (prefers plain text, falls back to HTML).

### Sending

```bash
# Basic send
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"

# HTML email
$GAPI gmail send --to user@example.com --subject "Report" \
  --body "<h1>Q4 Results</h1><p>Details here</p>" --html

# Custom From header (display name + email)
$GAPI gmail send --to user@example.com --subject "Hello" \
  --from '"Research Agent" <user@example.com>' --body "Message text"

# With CC
$GAPI gmail send --to user@example.com --cc "team@example.com" \
  --subject "Update" --body "FYI"
```

### Custom From Header

The `--from` flag lets you customize the sender display name on outgoing emails. This is useful when multiple agents share the same Gmail account but you want recipients to see different names:

```bash
# Agent 1
$GAPI gmail send --to client@co.com --subject "Research Summary" \
  --from '"Research Agent" <shared@company.com>' --body "..."

# Agent 2  
$GAPI gmail send --to client@co.com --subject "Code Review" \
  --from '"Code Assistant" <shared@company.com>' --body "..."
```

**How it works:** The `--from` value is set as the RFC 5322 `From` header on the MIME message. Gmail allows customizing the display name on your own authenticated email address without any additional configuration. Recipients see the custom display name (e.g. "Research Agent") while the email address stays the same.

**Important:** If you use a *different email address* in `--from` (not the authenticated account), Gmail requires that address to be configured as a [Send As alias](https://support.google.com/mail/answer/22370) in Gmail Settings → Accounts → Send mail as.

The `--from` flag works on both `send` and `reply`:

```bash
$GAPI gmail reply MESSAGE_ID \
  --from '"Support Bot" <shared@company.com>' --body "We're on it"
```

### Replying

```bash
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
```

Automatically threads the reply (sets `In-Reply-To` and `References` headers) and uses the original message's thread ID.

### Labels

```bash
# List all labels
$GAPI gmail labels

# Add/remove labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

## Calendar

```bash
# List events (defaults to next 7 days)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (timezone required)
$GAPI calendar create --summary "Team Standup" \
  --start 2026-03-01T10:00:00-07:00 --end 2026-03-01T10:30:00-07:00

# With location and attendees
$GAPI calendar create --summary "Lunch" \
  --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z \
  --location "Cafe" --attendees "alice@co.com,bob@co.com"

# Delete event
$GAPI calendar delete EVENT_ID
```

:::warning
Calendar times **must** include a timezone offset (e.g. `-07:00`) or use UTC (`Z`). Bare datetimes like `2026-03-01T10:00:00` are ambiguous and will be treated as UTC.
:::

## Drive

```bash
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5
```

## Sheets

```bash
# Read a range
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write to a range
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

## Docs

```bash
$GAPI docs get DOC_ID
```

Returns the document title and full text content.

## Contacts

```bash
$GAPI contacts list --max 20
```

## Output Format

All commands return JSON. Key fields per service:

| Command | Fields |
|---------|--------|
| `gmail search` | `id`, `threadId`, `from`, `to`, `subject`, `date`, `snippet`, `labels` |
| `gmail get` | `id`, `threadId`, `from`, `to`, `subject`, `date`, `labels`, `body` |
| `gmail send/reply` | `status`, `id`, `threadId` |
| `calendar list` | `id`, `summary`, `start`, `end`, `location`, `description`, `htmlLink` |
| `calendar create` | `status`, `id`, `summary`, `htmlLink` |
| `drive search` | `id`, `name`, `mimeType`, `modifiedTime`, `webViewLink` |
| `contacts list` | `name`, `emails`, `phones` |
| `sheets get` | 2D array of cell values |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup (ask Hermes to set up Google Workspace) |
| `REFRESH_FAILED` | Token revoked — re-run authorization steps |
| `HttpError 403: Insufficient Permission` | Missing scope — revoke and re-authorize with the right services |
| `HttpError 403: Access Not Configured` | API not enabled in Google Cloud Console |
| `ModuleNotFoundError` | Run setup script with `--install-deps` |
