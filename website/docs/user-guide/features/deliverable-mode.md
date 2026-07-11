---
title: Deliverable Mode (Artifacts in Chat)
sidebar_label: Deliverable Mode
description: How the agent ships generated charts, PDFs, spreadsheets, and other files as native attachments in messaging platforms.
---

# Deliverable Mode

When Hermes Agent runs inside a messaging gateway (Slack, Discord, Telegram,
WhatsApp, Signal, etc.), it can deliver generated files directly into the
chat — not as paths the user has to copy, but as native attachments.

A chart shows up as an inline image. A PDF report shows up as a file
download. A spreadsheet uploads as `.xlsx`. The agent does not need to
write a `MEDIA:` tag or do anything special — it just generates the file
and mentions its absolute path in the response. The gateway picks the path
out of the text, removes it from the visible message, and uploads the
file natively.

## How it works

Three pieces fit together:

1. **The agent has tools that produce files.** `execute_code` for charts via
   matplotlib, the `latex-pdf-report` skill for PDFs, the `powerpoint` skill
   for decks, `image_generate` for images, `text_to_speech` for audio, and so
   on.

2. **The gateway scans agent responses for file paths.** Any absolute path
   (`/tmp/...`) or home-relative path (`~/...`) ending in a supported
   extension gets extracted. Paths inside code blocks and inline code are
   ignored so code samples are never mutilated.

3. **The gateway dispatches by file type.** Images embed inline where the
   platform supports it; videos embed inline; audio routes to voice/audio
   attachments; everything else uploads as a file attachment.

## Supported file extensions

| Category | Extensions | Delivery |
|---|---|---|
| Images | `.png .jpg .jpeg .gif .webp .bmp .tiff .svg` | Inline embed |
| Video | `.mp4 .mov .avi .mkv .webm` | Inline embed (where supported) |
| Audio | `.mp3 .wav .ogg .m4a .flac` | Voice / audio attachment |
| Documents | `.pdf .docx .doc .odt .rtf .txt .md` | File upload |
| Data | `.xlsx .xls .csv .tsv .json .xml .yaml .yml` | File upload |
| Presentations | `.pptx .ppt .odp` | File upload |
| Archives | `.zip .tar .gz .tgz .bz2 .7z` | File upload |
| Web | `.html .htm` | File upload |

`.py`, `.log`, and other source-file extensions are intentionally excluded so
the agent doesn't auto-ship arbitrary source files; if you want to send code
to the user, use a code block.

## Encouraging the agent to produce artifacts

The agent doesn't reach for artifacts by default — it has to know to.
Two ways to nudge it:

**Per-session:** ask explicitly ("send me the comparison as a chart",
"return the data as a CSV") or write your own custom-instructions /
personality entry that biases toward artifact-style replies on
messaging platforms.

**Project-level:** add the bias to `AGENTS.md` / `CLAUDE.md` /
`.cursorrules` in a project the agent works from, to your global
persona in `~/.hermes/SOUL.md`, or as a named preset under
`agent.personalities` in `~/.hermes/config.yaml` (switchable per session
via `/personality`).

The mechanic the agent has to use is simple: render the file to an
absolute path (e.g. `/tmp/q3-revenue.png`) and mention that path as
plain text in the reply. The gateway does the rest. Paths inside
fenced code blocks or backticks are ignored so code samples are never
mutilated.

## Kanban: artifacts ride completion notifications

If you use Hermes' kanban multi-agent workflow, workers can attach
deliverable files to their `kanban_complete` call:

```python
kanban_complete(
    summary="rendered Q3 revenue chart and report",
    artifacts=[
        "/tmp/q3-revenue.png",
        "/tmp/q3-report.pdf",
    ],
)
```

When the gateway notifier delivers the "task completed" message to whoever
subscribed to the task in Slack/Telegram/etc., it also uploads each artifact
as a native attachment to that chat. The human gets the deliverable and the
summary in one place.

Paths inside the task's managed scratch workspace must reference existing
regular files when `kanban_complete` runs. Hermes copies them to durable task
attachment storage before deleting that workspace and rewrites the completion
metadata to the stored paths. A missing, unreadable, non-file, or larger than
25 MiB scratch artifact rejects completion and keeps the workspace so the
worker can correct the path and retry. External paths remain unchanged and are
silently skipped if they are unavailable when the notifier runs.

## Connecting more services with MCP

Beyond the artifact-delivery pipeline, the agent can reach into other
services via MCP (Model Context Protocol). The MCP ecosystem ships
community servers for most popular tools — install whichever you need:

| Service | What it unlocks |
|---|---|
| **Notion** | Read/write Notion pages, databases, query workspace |
| **GitHub** | Issues, PRs, comments, repo search beyond the gh CLI |
| **Linear** | Tickets, projects, cycles |
| **Slack** | Workspace-wide search, read other channels |
| **Gmail** | Inbox triage, send mail, label management |
| **Salesforce** | Leads, opportunities, account data |
| **Snowflake / BigQuery** | SQL against data warehouses |
| **Google Drive** | File search, contents, share management |

Install MCP servers via `~/.hermes/config.yaml` under the `mcp_servers`
section. See [MCP integration](./mcp.md) for the full setup guide.

## Comparison to Perplexity Computer in Slack

Perplexity Computer's Slack integration is built around the same idea:
the agent generates a deliverable (chart, PDF, slide deck) and posts it
back into the thread as a native attachment. Hermes Agent's deliverable
mode provides the same user-facing pattern locally:

- Generation happens in the user's own venv / sandbox (no remote tenant).
- Files land in the chat via the same Slack `files.uploadV2` API.
- Connector breadth comes via MCP rather than a curated catalog of 400
  hosted integrations — install the ones you actually use.

OAuth tokens stay on the user's machine in `auth.json` / `.env`. No hosted
token storage. No multi-tenant microVM. Same end result.
