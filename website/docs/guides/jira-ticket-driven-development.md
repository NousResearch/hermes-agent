---
title: "Jira ticket-driven development"
description: "Use Jira issue URLs or keys as the source of truth for Hermes implementation work."
---

# Jira ticket-driven development

Hermes can load Jira issue content when you give it a full issue URL or an issue key such as `CPG-1489`. When the `jira_get_issue` tool is available, ask Hermes to complete the ticket and include the Jira reference:

```text
Complete https://example.atlassian.net/browse/CPG-1489 and open PRs in the affected repos.
```

```text
Complete CPG-1489. Read the Jira ticket first, then implement the acceptance criteria.
```

The tool returns normalized issue content: summary, description, labels, components, status, priority, issue links, subtasks, attachments, and custom fields whose names look like acceptance criteria, repository, team, component, epic, or environment metadata.

## Setup

Add Jira credentials to `~/.hermes/.env` or your deployment secret manager:

```bash
JIRA_SITE_URL=https://example.atlassian.net
JIRA_EMAIL=dev@example.com
JIRA_API_TOKEN=atlassian-api-token
```

For deployments that use bearer auth, set `JIRA_BEARER_TOKEN` instead of `JIRA_EMAIL` and `JIRA_API_TOKEN`.

Use least-privilege Jira credentials. The built-in Jira tool is read-only and only calls Jira's issue read endpoint; it does not transition, comment on, or edit issues.

## Operational limits and conventions

- Full Jira URLs include the site, so `JIRA_SITE_URL` is optional for those. Bare issue keys require `JIRA_SITE_URL` or `ATLASSIAN_SITE_URL`.
- Hermes relies on the ticket fields, linked issues, labels, and components to infer the relevant repository or repositories. If multiple repos are involved, mention them in the ticket description, labels, components, or a repository-related custom field.
- Branch names should include the ticket key when practical, for example `feat/cpg-1489-jira-ticket-tool`.
- Keep one PR per affected repository unless the team explicitly asks for a different shape.
- Very large attachments are not downloaded automatically; the tool exposes attachment metadata and URLs so the agent can decide whether to fetch a specific file with an appropriate tool.
- Jira write-back is intentionally out of scope for the first integration. If your team wants a Jira comment, remote link, or transition, ask explicitly and provide/write-enable a separate integration.
- Repository access, maximum concurrent repos, and CI requirements are governed by the normal terminal/GitHub credentials available to the Hermes runtime. If a ticket spans more repos than the runtime can access, Hermes should open PRs for accessible repos and report the blocked repos clearly.

## Security notes

- Store Jira tokens only in `~/.hermes/.env`, the gateway's secret store, or your orchestrator's secret manager. Do not put them in prompts, tickets, or repo files.
- Prefer project-scoped or read-only Jira tokens where Atlassian permissions allow it.
- The tool never prints the token and returns only normalized issue data.
