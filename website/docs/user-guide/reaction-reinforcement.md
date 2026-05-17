---
sidebar_position: 14
---

# Reaction Reinforcement

Hermes lives on Telegram, Discord, and Slack, where users naturally react to
messages with emoji.  Historically those reactions just disappeared into
the void; with **reaction reinforcement** (issue
[#27438](https://github.com/NousResearch/hermes-agent/issues/27438)) Hermes
captures `👍 / ❤️ / 👎 / 💩 / …` and turns them into a lightweight feedback
signal it can act on -- without any extra effort from you.

## What this is, and what it is not

This is **not** model fine-tuning -- that's the territory of
[#498 (Conversational RL Personalization)](https://github.com/NousResearch/hermes-agent/issues/498).
Reaction reinforcement is a small, in-process mechanism that:

1. Listens to message-reaction updates on supported platforms.
2. Maps each emoji to a polarity (positive / negative / neutral) and a
   weight.
3. Persists the events to a SQLite database under your profile's
   `HERMES_HOME` so future Hermes features can read them.

The captured events are a **foundation**.  Consumers -- memory weighting,
skill confidence scoring, response-style tuning, preference extraction --
will land in follow-up changes; this page documents the capture layer
that's available today.

:::note Privacy
Reaction events are stored in your local `reactions.db` under
`HERMES_HOME`.  Nothing is uploaded, shared, or used as telemetry.  When
profiles are active each profile has its own DB, so a reaction in your
`work` profile never bleeds into `personal`.
:::

## Default emoji → signal mapping

The defaults track the table from
[#27438](https://github.com/NousResearch/hermes-agent/issues/27438):

| Reaction        | Polarity | Weight | Label          |
| --------------- | -------- | ------ | -------------- |
| ❤️  heart        | positive | +2.0   | `heart`        |
| 👍 thumbs up    | positive | +1.0   | `thumbs_up`    |
| 😂 laugh        | positive | +0.8   | `laugh`        |
| 🙌 raised hands | positive | +1.0   | `raised_hands` |
| 👎 thumbs down  | negative | -1.0   | `thumbs_down`  |
| 😢 cry          | negative | -1.5   | `cry`          |
| 💩 poo          | negative | -2.0   | `poo`          |
| 😡 angry        | negative | -2.0   | `angry`        |

Emoji outside this table are dropped by default.  Set
`HERMES_REACTION_INCLUDE_UNKNOWN=true` to also record them with neutral
weight `0.0` (useful for engagement telemetry).

## Enabling reaction reinforcement

The feature is strictly **opt-in**.  Enable via either env or
`config.yaml`:

```bash
# .env or your shell
export HERMES_REACTION_SIGNALS_ENABLED=true
```

```yaml
# config.yaml
reaction_signals:
  enabled: true
  min_signal_threshold: 0.5
  decay_days: 30
  include_unknown: false
```

After flipping the switch, restart the gateway (`hermes gateway restart`).
The next time someone reacts to a Hermes message, the event lands in
`$HERMES_HOME/reactions.db`.

:::caution Telegram-only in v1
The v1 capture layer wires up the Telegram
`MessageReactionHandler`.  Discord (`on_raw_reaction_add`) and Slack
(`reaction_added`) adapters expose the same hook on
`BasePlatformAdapter.handle_reaction()`, but the platform-side handlers
will be added in follow-up PRs.
:::

### Telegram-specific setup

Telegram only delivers `message_reaction` updates when the bot is set up
correctly:

- **Direct messages**: works out of the box.
- **Groups / supergroups**: the bot must be an **admin** in the chat.
  This is a Telegram-side restriction; without admin rights the bot
  receives no reaction updates at all.

Anonymous group reactions (where Telegram doesn't include the
reacting user) are deliberately dropped -- attributing them to "unknown"
would pollute per-user signal.

## Configuration reference

Each YAML key has a matching env var (env wins).  All keys are
documented in detail in
[Environment Variables → Emoji Reaction Reinforcement](../reference/environment-variables.md#emoji-reaction-reinforcement).

| YAML key                     | Env var                            | Default | Purpose                                                                  |
| ---------------------------- | ---------------------------------- | ------- | ------------------------------------------------------------------------ |
| `enabled`                    | `HERMES_REACTION_SIGNALS_ENABLED`  | `false` | Master switch                                                            |
| `min_signal_threshold`       | `HERMES_REACTION_MIN_SIGNAL`       | `0.5`   | Magnitude below which the aggregated signal is treated as noise          |
| `decay_days`                 | `HERMES_REACTION_DECAY_DAYS`       | `30`    | Retention horizon for `ReactionStore.prune_older_than()` (no auto cron yet) |
| `include_unknown`            | `HERMES_REACTION_INCLUDE_UNKNOWN`  | `false` | Record unrecognised emoji with neutral weight 0.0                        |

This block is intentionally **separate** from the per-platform
`reactions:` toggles (`telegram.reactions`, `discord.reactions`,
`slack.reactions`) which control the outgoing `👀/✅/❌` lifecycle
reactions on the user's trigger message.  Those continue to behave
exactly as before.

## Where the data lives

```
$HERMES_HOME/reactions.db
```

WAL-mode SQLite, schema version `1`.  Indexes are tuned for the three
read patterns we expect early consumers to use:

- per message (memory weighting, response-style tuning)
- per user (preference extraction)
- by timestamp (decay / pruning)

The schema is identical across platforms, so when Discord and Slack
capture layers land they will append to the same table.

### Inspecting events

You can read the DB directly with the `sqlite3` CLI:

```bash
sqlite3 ~/.hermes/reactions.db \
  "SELECT ts, platform, emoji, label, weight, added \
     FROM reaction_events \
    ORDER BY ts DESC LIMIT 10;"
```

Or from Python:

```python
from gateway.reaction_store import get_reaction_store

store = get_reaction_store()
summary = store.aggregate_for_message(
    platform="telegram",
    channel_id="42",
    target_message_id="123",
)
print(summary)
# -> {'net_weight': 2.0, 'positive': 1, 'negative': 0,
#     'neutral': 0, 'unique_users': 1, 'sample_count': 1, ...}
```

### Pruning history

By default events are kept forever.  To prune older than the configured
retention horizon:

```python
from gateway.reaction_store import get_reaction_store
deleted = get_reaction_store().prune_older_than(days=30)
```

A scheduled-cron integration may land in a follow-up; for now run
`prune_older_than` from a shell job if you want automatic cleanup.

## Per-user weight overrides

Some users react with everything; some reserve `❤️` for things that
genuinely changed their day.  The capture layer already supports custom
weight tables via `ReactionConfig.from_env(overrides=…)`; the user-facing
YAML surface for per-user overrides is intentionally deferred to a
follow-up PR so this initial change stays small and reviewable.

## Troubleshooting

**I enabled it but no rows show up in `reactions.db`.**

1. Restart the gateway after changing env or config.
2. On Telegram groups: make the bot an admin.  Without admin rights
   Telegram silently drops `message_reaction` updates server-side.
3. Confirm the master flag: `echo $HERMES_REACTION_SIGNALS_ENABLED`
   should be `true`.  Anything else is treated as disabled.
4. Custom emoji (premium / animated) and unicode outside the default
   weight table are dropped unless you set
   `HERMES_REACTION_INCLUDE_UNKNOWN=true`.

**My DB is on NFS / SMB / WSL1.**

The reaction store reuses Hermes' standard WAL-with-DELETE fallback, so
operations work but concurrency degrades.  This is the same trade-off
as `state.db` and `kanban.db`; see
[SQLite docs](https://www.sqlite.org/wal.html) for the underlying
reason.
