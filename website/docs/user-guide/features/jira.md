# Jira

Hermes can manage Jira Cloud issues, projects, and comments directly — search with JQL, create and update tickets, transition statuses, and add comments — using the Jira REST API v3 with API token authentication. Tokens are stored in `~/.hermes/auth.json`; you only authenticate once per machine.

Unlike Spotify (which requires PKCE OAuth), Jira uses long-lived API tokens. There is no browser redirect and no client ID to register. Authentication takes about 30 seconds.

## Prerequisites

- A Jira Cloud account (hosted at `*.atlassian.net`). Self-hosted Jira Server/Data Center is not supported.
- An Atlassian API token — generate one at [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens).
- Hermes Agent installed.

## Setup

### One-shot: `hermes tools`

The fastest path. Run:

```bash
hermes tools
```

Scroll to `🎫 Jira`, press space to enable it, then `s` to save. Hermes drops you straight into the authentication prompt — enter your domain, email, and API token. Once complete, the toolset is enabled and authenticated in one pass.

### Two-step flow

#### 1. Enable the toolset

```bash
hermes tools
```

Toggle `🎫 Jira` on and save. Dismiss the inline auth prompt if you prefer to authenticate separately.

#### 2. Run the auth command

```bash
hermes auth jira
```

Hermes prompts for three values:

| Prompt | Example | Notes |
|--------|---------|-------|
| Jira domain | `mycompany.atlassian.net` | Without `https://` — Hermes strips it automatically |
| Atlassian email | `you@example.com` | The email you use to log in to Jira |
| API token | `ATATT3x…` | From the Atlassian account settings page above; hidden during input |

Hermes validates the token by calling `GET /rest/api/3/myself` before saving. If the credentials are wrong you'll see an error immediately — nothing is written to disk.

On success, credentials are stored under `providers.jira` in `~/.hermes/auth.json`.

## Verify

```bash
hermes auth status jira
```

Shows whether credentials are present, which domain and email are stored, and confirms the auth type is `api_token`.

## Using it

Once authenticated, the agent has 4 Jira tools available. Talk to the agent naturally:

```
> What's the status of PROJ-42?
> Create a bug in project WEBAPP: "Login fails on Safari when using SSO"
> Search for open high-priority issues assigned to me
> Move PROJ-10 to In Progress
> Add a comment to PROJ-7: "Investigated — root cause is a race condition in session refresh"
> List all projects I have access to
> Show me the last 10 issues updated in the ENG project
```

### Tool reference

#### `jira_issue`

The main workhorse. One tool, five actions:

| Action | Purpose | Required args |
|--------|---------|---------------|
| `get` | Fetch full issue details | `issue_key` (e.g. `PROJ-42`) |
| `create` | Create a new issue | `project_key`, `issuetype`, `summary` |
| `update` | Update fields on an existing issue | `issue_key` + any of `summary`, `description`, `assignee_id`, `priority`, `labels` |
| `transitions` | List available status transitions | `issue_key` |
| `transition` | Move issue to a new status | `issue_key`, `transition_id` |

**Creating an issue:**
```
Create a Story in project ENG titled "Add dark mode to dashboard" with priority Medium
```
Common issue types: `Bug`, `Task`, `Story`, `Epic`, `Sub-task`. The agent matches whatever your Jira project supports.

**Transitioning status — why two steps?**
Jira transition IDs differ between projects and instances. The agent always calls `transitions` first to get the valid IDs for that issue, then calls `transition`. You never need to know the IDs yourself.

#### `jira_search`

JQL (Jira Query Language) search across all issues you have access to.

```
Search for all open bugs in the WEBAPP project assigned to me
→ jql: project = WEBAPP AND issuetype = Bug AND status != Done AND assignee = currentUser()

Show issues updated in the last 3 days
→ jql: updated >= -3d ORDER BY updated DESC

Find all issues with the label "backend" in active sprints
→ jql: labels = "backend" AND sprint in openSprints()
```

Returns a compact list: key, summary, status, issuetype, priority, assignee. Default limit: 20 results (max 50 per call).

#### `jira_project`

| Action | Purpose |
|--------|---------|
| `list` | All projects you have access to (key, name, type) |
| `get` | Details for a specific project (`project_key` required) |

Use `list` to discover project keys before creating issues.

#### `jira_comment`

| Action | Purpose | Required args |
|--------|---------|---------------|
| `list` | Fetch comments on an issue | `issue_key` |
| `add` | Post a new comment | `issue_key`, `body` |

Plain text is accepted for `body`; the tool converts it to Atlassian Document Format automatically.

## JQL quick reference

| Goal | JQL |
|------|-----|
| My open issues | `assignee = currentUser() AND status != Done` |
| Active sprint | `sprint in openSprints()` |
| High/Urgent bugs | `issuetype = Bug AND priority in (High, Highest)` |
| Created this week | `created >= startOfWeek()` |
| Updated today | `updated >= startOfDay()` |
| Text search | `text ~ "login error"` |
| No assignee | `assignee is EMPTY` |
| By label | `labels = "frontend"` |
| Sub-tasks of issue | `parent = PROJ-10` |

Combine filters with `AND`; use `ORDER BY created DESC` / `updated DESC` for recency.

## Sign out

```bash
hermes auth logout jira
```

Removes credentials from `~/.hermes/auth.json`. Your Jira account and API token are not affected — to revoke the token itself, delete it at [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens).

## Troubleshooting

**"Authentication failed: invalid email or API token"** — the email or token is wrong. Double-check that you copied the token correctly (they're long and easy to truncate). Generate a fresh token if needed.

**`400 Bad Request` when creating an issue** — usually a missing required field or an issue type your project doesn't support. Ask the agent to list projects (`jira_project list`) and try a common type like `Task`.

**`404 Not Found`** — the issue key doesn't exist in your Jira instance, or you don't have permission to view it.

**Transition not working** — the transition name you described doesn't exist for that issue's current status. The agent always fetches valid transitions before applying one; if none match what you want, the status flow for that issue type may differ. Ask the agent to show available transitions first.

**Auth not configured (tools return an error)** — run `hermes auth jira` to authenticate.

## Where things live

| Location | Contents |
|----------|----------|
| `~/.hermes/auth.json` → `providers.jira` | domain, email, Basic Auth token, account ID |
| Atlassian account settings | the API token itself (managed at id.atlassian.com) |
