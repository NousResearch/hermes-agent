---
sidebar_position: 16
title: "Contribute compute (autodevelop)"
description: "BYOK community agents drain a maintainer-curated GitHub issue queue"
---

# Contribute compute, not just patches

`hermes autodevelop` is the OSS on-ramp for people who already have coding agents and **bring their own keys**. You clone an opted-in repo, point Hermes at the maintainer-curated queue, and walk away. Hermes uses **your** tokens and compute — never the project's inference budget.

This is community compute plus community judgment (what maintainers put in the queue). It is not drive-by spam against every unlabeled issue.

## One-time setup

```bash
gh auth login
hermes setup          # configure your BYOK models
export GITHUB_TOKEN   # or GH_TOKEN; required for live GitHub API calls
```

`gh` login alone is not enough for this command — the CLI talks to `api.github.com` with `GITHUB_TOKEN` / `GH_TOKEN`.

## Contributor loop

```bash
cd your-checkout
hermes autodevelop list --repo NousResearch/hermes-agent
hermes autodevelop prompt --repo NousResearch/hermes-agent 12345
hermes autodevelop run --repo NousResearch/hermes-agent --max-issues 3
```

Default `run` / `claim` / `park` are **dry-run** (no issue comments). Pass `--commit` only when you intend to post a claim and assign yourself.

Unattended implement (chose these defaults because the issue allows a thin first slice and `--commit` must not surprise a dry-run):

```bash
hermes autodevelop run --repo owner/name --execute --require-tests
# open a draft PR only after a real claim:
hermes autodevelop run --repo owner/name --commit --execute --require-tests --open-pr
```

`--open-pr` is refused without `--commit`. Claims per login are persisted under `$HERMES_HOME/autodevelop/claims.json` and capped at 2 open issues per repo.

| Step | What happens |
| --- | --- |
| Resolve queue | Open issues with the queue label (`agent-ready` by default), excluding locked / assigned / in-progress claims, large epics, and `no-human-gate: false` unless you opt in |
| Atomic claim | Idempotent comment with a reclaim TTL (default 8h), same collision story as kanban claims |
| Search-first | The printed prompt requires `gh search issues` / `gh search prs` before coding ([CONTRIBUTING](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#before-you-start-search-first), #38284) |
| Ship or park | Draft PR by default, or `hermes autodevelop park --reason "…"` to release the claim |
| Next | `--max-issues` / `--budget` stop the loop |

Credit on the PR stays with the **human contributor**.

Optional Desktop: run the same CLI from a Desktop-integrated terminal. There is no separate Desktop control surface.

## Maintainer queue contract

Not every open issue is safe for unattended agents. Publish the contract with labels and issue-body fields:

| Field | Purpose |
| --- | --- |
| label `agent-ready` (or `--label`) | Opt-in to the queue |
| `scope: small\|medium\|large` | `large` is skipped unless `--include-large` |
| acceptance checklist (`- [ ] …`) | Done-when, copied into the prompt |
| `touches:` paths | Routing and the sensitive-path refuse list |
| `no-human-gate: true\|false` | Product decisions still need a person when false |
| label `autodevelop-allow-sensitive` | Required before `touches` may include secrets / auth / release |

Hermes dogfood: keep a slice of small bugs `agent-ready` so external BYOK contributors can help without a babysitter on every session.

## Safety defaults

- **BYOK only** — contributor keys; never maintainer inference
- **Draft PRs by default**; no auto-merge
- **Rate limits** — `--max-issues` / `--budget`, and a per-invocation claim cap
- **Path allowlists** — refuse secrets, release, and auth `touches` (and matching `CODEOWNERS` paths) unless `autodevelop-allow-sensitive`
- **No force-push to foreign branches** — standard fork PR flow in the prompt
- **Idempotent claims** — stale claim reclaim after `--ttl-hours`
- **Human override** — park on ambiguity; dry-run unless `--commit`

## Kanban mirror (optional)

```bash
hermes autodevelop sync-kanban --repo NousResearch/hermes-agent
# or
hermes autodevelop run --repo NousResearch/hermes-agent --sync-kanban
```

Each claimable issue becomes a local kanban task with idempotency key `github:owner/repo#n`. The existing dispatcher owns spawn / heartbeat / `failure_limit`. This is a thin GitHub → board sync, not a replacement for `hermes kanban`.

Schedule an unattended drain with [cron](./cron.md):

```bash
hermes cron create "0 */4 * * *" \
  "Run: hermes autodevelop run --repo NousResearch/hermes-agent --max-issues 1" \
  --name "autodevelop drain"
```

## Related

- [Kanban](./kanban.md) — multi-agent claim / dispatch / failure_limit
- [CLI commands](../../reference/cli-commands.md) — `hermes autodevelop` reference
- [CONTRIBUTING](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) — search-first and credit the human
