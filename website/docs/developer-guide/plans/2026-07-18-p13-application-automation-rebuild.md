# P13 application automation rebuild — implementation plan

> **For Hermes:** Execute in isolated worktree `/home/kensei/worktrees/p13-application-automation-rebuild`; do not touch live cron stores or activate jobs.

**Goal:** Rebuild Mossy, SahilBlog and mailbox-cleaner into bounded, testable, disabled-by-default scheduled candidates, with controlled dry-run evidence only.

**Locked decisions:**
- No job activation in this tranche.
- Tests plus one controlled live dry-run per pipeline after fixture proof.
- No publishing, mailbox mutation, branch deletion, force-push, auto-merge, or blind reviewer instruction application.
- Register all new jobs atomically disabled; never create-then-pause.

## Phase 1 — shared disabled registration

Add `hermes cron create --disabled` through CLI/API persistence. It must persist `enabled=false` atomically, never appear eligible in due-job selection, and be visible in `cron list --all`.

**Proof:** failing-first focused tests for create, due selection and readback. No live registration until Phase 4.

## Phase 2 — Mossy feedback loop

Create a side-effect-controlled domain module plus read-only watcher and context renderer. Normalise reviews, inline comments and PR conversation comments; reject bots/self; dedupe by namespaced GitHub IDs; persist isolated atomic state/queue/ledger/rules.

Implement policy classification only: routine patch, reply-only, clarification, or Sahil decision required. Teknium is priority only. Add pre-push rule selection/reporting, but do not implement push/reply/resolve automation in this disabled proof tranche.

**Proof:** fixture GitHub payloads, state migration, bot filtering, dedupe, authority matrix, rule promotion and strict dry-run mutation denylist.

## Phase 3 — SahilBlog single owner

Create `blog.scheduler_runner` with a non-blocking single-flight lock, run ID, atomic state/terminal record, hard timeout and JSON terminal output. It may produce or retry at most one draft per invocation and at most three images; it stages `approved:false` only.

Replace attachment-heavy approval requester output with tracker reconciliation against MDX, text-only stable batches, no local paths, no media, and 15-line maximum. Legacy detached wrappers become explicit deprecated non-runners; no background shell process remains.

**Proof:** temp-root lock/timeout/state tests; bounded pipeline invocation; stale tracker cleanup; compact output tests; static wrapper no-detach test.

## Phase 4 — mailbox cleaner read-only replacement

Create a deterministic `scripts/mailbox_cleaner/` package and three thin flow entrypoints. Use explicit-account MCP read adapters only. Scheduled flow may not parse/refresh OAuth caches, call direct Graph/Gmail APIs, or write credentials. It fails closed on missing read-only credentials; preparation stays an operator-only non-cron tool.

Encode account safety policy as code. State/report outputs are atomic, 0600, metadata-only and bounded. No mutation operation may exist in the scheduled model.

**Proof:** fake MCP clients, network/OAuth write denial, account-policy cases, urgent dedupe, HTML escaping and truthful degraded output.

## Phase 5 — integration and evidence

Run focused tests then affected regression suites. Independently review specification and code quality. Create the three new scheduler records disabled only after the disabled-create primitive passes. Run one manual controlled dry-run per pipeline with explicit temporary state roots; collect command, exit status, state and no-mutation evidence. No activation.

## Acceptance criteria

- All added tests demonstrate RED then GREEN evidence in execution logs.
- `git diff` is limited to the feature branch/worktree.
- No legacy detached wrapper can be scheduled successfully.
- No scheduled mailbox path imports direct OAuth/cache-write or mail mutation code.
- Mossy’s read-only proof cannot call a mutation command.
- Blog controlled run cannot publish, approve, commit, push or deliver Discord content.
- New cron registrations are atomically disabled and read back disabled.
