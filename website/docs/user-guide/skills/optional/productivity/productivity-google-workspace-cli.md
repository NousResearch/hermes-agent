---
title: "Google Workspace CLI — Use Google Workspace CLI without Google Cloud setup"
sidebar_label: "Google Workspace CLI"
description: "Use Google Workspace CLI without Google Cloud setup"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Google Workspace CLI

Use Google Workspace CLI without Google Cloud setup.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/google-workspace-cli` |
| Path | `optional-skills/productivity/google-workspace-cli` |
| Version | `1.0.0` |
| Author | MyBrandMetrics; Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `Google`, `Calendar`, `Drive`, `Sheets`, `CLI`, `MyBrandMetrics`, `OAuth` |
| Related skills | [`google-workspace`](/user-guide/skills/bundled/productivity/productivity-google-workspace) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Google Workspace CLI Skill

Use this skill to manage Google Calendar, Sheets, and Drive through the `gws`
CLI with short-lived OAuth access tokens issued by MyBrandMetrics. It is built
for users who want Google Workspace automation without creating a Google Cloud
project, downloading OAuth client secrets, or running an MCP server.

The setup is intentionally simpler than MCP-based Google Workspace tools: the
user connects Google once through MyBrandMetrics, then Hermes uses the API key
to fetch scoped access tokens when it needs to run `gws`.

## When to Use

- The user needs Calendar, Sheets, or Drive operations from Hermes.
- The user wants a CLI workflow instead of an MCP server.
- The user does not want to create a Google Cloud project or manage OAuth client
  credentials.
- The user already connected Google by logging in through MyBrandMetrics.
- The user has a MyBrandMetrics API key and wants external token provisioning
  for `gws`.

## Why This Is Easier

Compared with Google Workspace MCP setups, this skill keeps the agent-side
surface small:

- No Google Cloud project setup.
- No OAuth client ID or client secret file.
- No local MCP server to install, configure, or keep running.
- No redirect URI or Desktop OAuth app setup inside Hermes.
- Google authorization happens with a normal Google login through
  MyBrandMetrics.
- Hermes only needs the MyBrandMetrics API key plus the lightweight `gws`
  wrapper.

## Prerequisites

- A MyBrandMetrics API key with the `wk_...` format.
- A Google account already connected in MyBrandMetrics by Google login.
- Network access to `https://api.mybrandmetrics.com`.
- The `gws` CLI available on `PATH`, or installable through the helper script.

Prefer `GWS_SKILL_API_KEY` for the API key. If the user cannot set environment
variables in their Hermes runtime, save the key at `~/.google_workspace_api_key`
with mode `0600`.

Install `gws` when missing:

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace-cli/scripts/install_gws.sh"
```

## How to Run

Run all Google Workspace actions through the `terminal` tool and the wrapper:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" <service> <resource> <method> [args]
```

Supported services are `calendar`, `sheets`, and `drive`. The wrapper maps them
to MyBrandMetrics token sources, injects the token into
`GOOGLE_WORKSPACE_CLI_TOKEN`, and then runs `gws`.

## Quick Reference

Calendar:

```bash
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" calendar events list --params '{"calendarId":"primary","maxResults":10}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" calendar events insert --params '{"calendarId":"primary"}' --json '{"summary":"Title","start":{"dateTime":"2026-05-21T10:00:00Z"},"end":{"dateTime":"2026-05-21T10:30:00Z"}}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" calendar events patch --params '{"calendarId":"primary","eventId":"EVENT_ID"}' --json '{"summary":"Updated title"}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" calendar events delete --params '{"calendarId":"primary","eventId":"EVENT_ID"}'
```

Sheets:

```bash
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" sheets spreadsheets values get --params '{"spreadsheetId":"ID","range":"Sheet1!A1:D10"}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" sheets spreadsheets values update --params '{"spreadsheetId":"ID","range":"Sheet1!A1:C2","valueInputOption":"USER_ENTERED"}' --json '{"values":[["A1","B1","C1"],["A2","B2","C2"]]}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" sheets spreadsheets values append --params '{"spreadsheetId":"ID","range":"Sheet1!A1","valueInputOption":"USER_ENTERED"}' --json '{"values":[["New row"]]}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" sheets spreadsheets batchUpdate --params '{"spreadsheetId":"ID"}' --json '{"requests":[{"addSheet":{"properties":{"title":"New Sheet"}}}]}'
```

Drive:

```bash
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" drive files list --params '{"pageSize":10,"q":"trashed = false"}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" drive files create --json '{"name":"New Folder","mimeType":"application/vnd.google-apps.folder"}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" drive files update --params '{"fileId":"ID","addParents":"NEW_FOLDER_ID","removeParents":"OLD_FOLDER_ID"}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" drive permissions create --params '{"fileId":"ID"}' --json '{"role":"writer","type":"user","emailAddress":"user@example.com"}'
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" drive +upload /local/path/file.pdf --parent FOLDER_ID --name file.pdf
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" drive files export --params '{"fileId":"ID","mimeType":"application/pdf"}' --output doc.pdf
```

## Procedure

1. Check whether the key is already available:

   ```bash
   test -n "$GWS_SKILL_API_KEY" || test -s "$HOME/.google_workspace_api_key"
   ```

2. If the key is missing, ask the user for their MyBrandMetrics API key. Store
   it only after the user provides it explicitly:

   ```bash
   umask 077
   printf '%s' 'wk_REPLACE_ME' > "$HOME/.google_workspace_api_key"
   ```

3. Install `gws` if needed:

   ```bash
   bash "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/install_gws.sh"
   ```

4. Run the requested Calendar, Sheets, or Drive command through
   `scripts/gws_wrapper.py`.

5. For destructive Drive or Calendar actions, confirm the target ID and action
   with the user before running the command.

## Positioning Notes

- Use this skill when the user values fast setup and CLI execution.
- Use MCP-based Google Workspace tools only when the user specifically needs MCP
  resources/tools exposed to an MCP client.
- Use the bundled `google-workspace` skill when the user wants Hermes to manage
  direct Google OAuth credentials locally.

## Pitfalls

- This skill does not manage Google OAuth consent. MyBrandMetrics must already
  be authorized for the requested Google service through Google login.
- `gws` command shapes can differ by version. If a command fails with usage
  output, retry using the subcommand names printed by the installed `gws`.
- The wrapper times out token requests after 15 seconds to keep agent turns from
  hanging on network failures.
- Keep the MyBrandMetrics API key out of command output and logs.

## Verification

```bash
python3 "$HERMES_HOME/skills/productivity/google-workspace-cli/scripts/gws_wrapper.py" calendar events list --params '{"calendarId":"primary","maxResults":1}'
```
