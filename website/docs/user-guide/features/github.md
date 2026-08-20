# GitHub App Connector

Hermes can operate GitHub issues and pull requests directly — create issues,
comment on threads, submit PR reviews, and merge PRs — using the **GitHub App
bot identity** when configured, so every agent action is attributed to the
bot (`<app-slug>[bot]`), never to your personal account.

That attribution is the connector's core value: with a plain personal token
(PAT), Hermes's comments and reviews are indistinguishable from yours. With a
GitHub App, they're visibly separate — you can tell at a glance which action
came from Hermes and which came from you. Every tool result includes an
`attribution` block (`auth_method` + `actor`) so the agent and the user
always know who acted.

## Prerequisites

- A GitHub App installed on the repos you want to operate. See
  [Creating a GitHub App](https://docs.github.com/en/apps/creating-github-apps)
  — the app needs `Issues: Read and write` and `Pull requests: Read and
  write` permissions. Install it on your account/repos, then note the App ID,
  Installation ID, and the private key `.pem` path.
- Hermes Agent installed and running.

## Setup

### 1. Enable the toolset

```bash
hermes tools
```

Toggle `🐙 GitHub App Connector` on and save. When GitHub credentials are
present (any of the options below), the toolset also auto-enables on first
run without visiting `hermes tools`.

### 2. Configure credentials

The connector resolves credentials in this order — the first one present
wins:

| Priority | Method | Env vars | Attribution |
|---|---|---|---|
| 1 | **GitHub App** (recommended) | `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY_PATH`, `GITHUB_APP_INSTALLATION_ID` | `<slug>[bot]` |
| 2 | Personal token | `GITHUB_TOKEN` or `GH_TOKEN` | your account |
| 3 | gh CLI | `gh auth login` | your account |

Secrets live in `~/.hermes/.env` (profile-scoped):

```bash
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY_PATH=/path/to/your-app.pem
GITHUB_APP_INSTALLATION_ID=789012
```

Installation tokens are minted automatically (JWT → installation token,
cached ~50 min, refreshed on 401) — no manual token rotation.

## Tools

| Tool | Purpose |
|---|---|
| `github_identity` | Verify which identity Hermes acts as (bot vs account) + accessible repos |
| `github_create_issue` | Create an issue (title, body, labels, assignees) |
| `github_comment_issue` | Comment on an issue or PR thread |
| `github_list_issues` | List issues with filters (state, labels, assignee, creator, sort) |
| `github_get_issue` | Fetch issue/PR detail, optionally with comments |
| `github_review_pr` | Submit a PR review: approve, request changes, or comment (with optional inline comments) |
| `github_merge_pr` | Merge a PR (squash by default, or merge/rebase) |

## Examples

```text
> Comment on issue #42 in himanusia/plus1: "This is fixed in #43, please re-test."

github_comment_issue(repo="himanusia/plus1", number=42, body="This is fixed in #43, please re-test.")
→ {"success": true, "author": "jarpis-bot[bot]", "attribution": {"auth_method": "github-app", "actor": "jarpis-bot[bot]"}}
```

```text
> Approve PR #7 in himanusia/plus1

github_review_pr(repo="himanusia/plus1", number=7, event="APPROVE", body="LGTM — CI green, tests pass.")
→ {"success": true, "state": "APPROVED", "attribution": {"auth_method": "github-app", "actor": "jarpis-bot[bot]"}}
```

## Attribution guarantees

- **With a GitHub App configured, every write action is attributed to the
  bot** — the `actor` in tool results is `<app-slug>[bot]`, and GitHub shows
  the bot as the author of comments/reviews/merges. You can always tell an
  agent action from your own.
- **With only a PAT or gh CLI, actions are attributed to your account** —
  the connector reports `auth_method: pat` / `auth_method: gh-cli` and the
  account login, so the distinction is explicit even when you choose not to
  run a bot.
- `github_identity` is the read-only check: run it first to confirm which
  identity Hermes will act as before commenting or reviewing.

## Event-driven flow (optional)

Pair the connector with Hermes's webhook subscriptions for event-driven
GitHub workflows — e.g. auto-reply to `issue_comment` events on your repos:

```bash
hermes webhook subscribe gh-issue-comments --events "issue_comment,issues" --deliver origin
```

See [Webhooks](../reference/webhooks) for details.
