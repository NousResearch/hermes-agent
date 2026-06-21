---
sidebar_position: 4
title: "Self-Evolution Observability"
description: "Inspect local memory and skill evolution events with hermes evolution"
---

# Self-Evolution Observability

Self-evolution observability records successful durable learning changes that Hermes makes through its memory and skill tools. It gives you a local timeline of what changed, where it changed, and why the agent made the update.

This feature is disabled by default. Enable it explicitly when you want a local review trail for agent-managed memory and skill changes.

```bash
hermes evolution enable
hermes evolution disable
```

## What It Records

When enabled, Hermes records successful mutations from these agent tools:

- `memory.add`, `memory.replace`, `memory.remove`
- `skill.create`, `skill.patch`, `skill.edit`, `skill.delete`
- `skill.write_file`, `skill.remove_file`

Each event includes an event ID, UTC timestamp, target path, target kind/name, summary, optional reason, unified diff, redaction status, and truncation status.

Events are stored locally in the active profile:

```text
$HERMES_HOME/evolution/events.jsonl
```

Running `hermes evolution enable` creates the `evolution/` directory. The `events.jsonl` file is created only when the first event is recorded.

## What It Does Not Record

This is not a complete audit log. It does not record config changes, model/provider changes, profile changes, ordinary session messages, ordinary tool calls, failed mutation attempts, direct shell/editor edits, or Hub skill install/update/uninstall commands.

Curator event recording is deferred to a future version. PR1 records memory and `skill_manage` events only.

## Privacy

Evolution logging is local-only and is not telemetry. Events are not uploaded to Nous or any remote service.

When `evolution.redact` is true, Hermes redacts sensitive text before writing event summaries, reasons, and diffs. When a diff exceeds `evolution.max_diff_chars`, Hermes truncates it and records that truncation in event metadata.

Default config:

```yaml
evolution:
  enabled: false
  record_diff: true
  redact: true
  max_diff_chars: 20000
```

## Commands

Show recent events:

```bash
hermes evolution list
hermes evolution timeline
```

Filter events:

```bash
hermes evolution list --days 7
hermes evolution list --type skill
hermes evolution list --type memory.add
hermes evolution list --target USER
```

Show one event by full ID or short ID:

```bash
hermes evolution show a1b2c3
```

Show aggregate activity:

```bash
hermes evolution stats --days 30
```

Clear older events safely:

```bash
hermes evolution clear --older-than 90
hermes evolution clear --older-than 90 --yes
```

Without `--yes`, `clear` only previews how many events would be deleted. PR1 does not include automatic retention or `--all` cleanup.

## Limits

Evolution logs are not a backup system. Some destructive events omit content, including `skill.delete`, which records `[skill deleted: content omitted]` instead of the deleted skill body.

Evolution logs are also not a complete audit log. They focus on durable self-evolution changes made by Hermes through memory and skill tools.
