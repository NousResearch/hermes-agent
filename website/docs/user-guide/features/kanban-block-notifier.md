# Kanban blocker notifications without cron polling

This note documents the supported path for notifying a human when a Kanban task
enters a human-facing blocked state (`needs_input` / `capability`) without a
separate cron job that repeatedly scans blocked tasks.

## Current architecture

Kanban state is stored in SQLite board databases:

- the legacy default board: `<hermes-root>/kanban.db`;
- project boards: `<hermes-root>/kanban/boards/<slug>/kanban.db`;
- task transition history: `task_events`;
- gateway subscriptions: `kanban_notify_subs`.

The gateway already has a notifier watcher that tails `task_events` for
subscribed tasks and deduplicates delivery with `kanban_notify_subs.last_event_id`.
That closes the loop for tasks created from a chat, but it does not guarantee a
user-visible ping for every worker-originated `kanban_block(kind="needs_input")`
when no chat subscription exists.

Hermes also exposes Kanban lifecycle plugin hooks from `hermes_cli.kanban_db`:

- `kanban_task_claimed`;
- `kanban_task_completed`;
- `kanban_task_blocked`.

Those hooks fire after the SQLite transaction commits, so observers never hold
the board write lock and always observe durable state.

## Chosen approach

Use an opt-in plugin, `kanban-block-notifier`, that listens to
`kanban_task_blocked` and sends a one-shot message via `hermes send` for only
human-facing blockers.

Why this survives `hermes update`:

- the integration uses the public plugin and lifecycle-hook surface;
- it does not patch a local installed copy of core files;
- if shipped with Hermes, it is enabled through `plugins.enabled`; if installed
  as a user plugin, it lives in `<HERMES_HOME>/plugins/`, which updates do not
  overwrite.

Why it is not a cron/watchdog:

- no periodic scan of all `blocked` tasks is performed;
- the plugin runs only when a block transition happens;
- deduplication is persisted in a tiny SQLite sidecar so repeated identical
  blocker events do not spam the user.

## Routing rules

The plugin notifies targets configured under
`plugins.entries.kanban-block-notifier.targets` (default: `telegram`, meaning the
Telegram home target known to `hermes send`). It sends only when:

- `task.block_kind` is `needs_input` or `capability`; or
- a legacy untyped block reason contains strong human-input keywords such as
  `secret`, `token`, `access`, `DNS`, `Cloudflare`, `HTTPS`, or `decision`.

It suppresses:

- `review-required` blocks, which should route to an internal reviewer gate;
- `dependency` blocks;
- `transient` blocks.

Secret-related reasons are redacted before delivery. If the reason looks like a
secret/token/access request, the message tells the user not to paste secrets into
chat and includes `secure_drop_url` when configured; otherwise it explicitly says
that secure-drop is not configured.

## Configuration

Enable the plugin:

```bash
hermes plugins enable kanban-block-notifier
```

Optional config:

```yaml
plugins:
  enabled:
    - kanban-block-notifier
  entries:
    kanban-block-notifier:
      targets:
        - telegram
      notify_kinds:
        - needs_input
        - capability
      suppress_review_required: true
      secure_drop_url: "https://your-secure-drop.example/one-time"
      # Optional override; default is <kanban-home>/kanban/kanban-block-notifier.sqlite3
      # state_db: /path/to/kanban-block-notifier.sqlite3
```

`hermes send --to telegram` must work for the gateway/home Telegram target. Test
it directly before enabling the plugin in production:

```bash
hermes send --to telegram "Kanban notifier smoke test"
```

## Smoke/regression test

On a temporary board/home, create a task, block it with `kind="needs_input"`, and
assert exactly one send is attempted for two identical hook invocations. The unit
suite covers this behavior in:

```bash
scripts/run_tests.sh tests/plugins/test_kanban_block_notifier.py
```

Manual production smoke test after review:

1. Enable the plugin in the Telegram-orchestrator profile.
2. Create a throwaway task on a non-production board.
3. Run a worker or CLI action that blocks it with `kind="needs_input"`.
4. Confirm a single Telegram message arrives.
5. Re-run the same blocker and confirm no duplicate message arrives.
6. Disable the plugin with `hermes plugins disable kanban-block-notifier` if it
   misbehaves; existing Kanban behavior remains unchanged.
