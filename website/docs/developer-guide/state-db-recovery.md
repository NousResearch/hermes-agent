---
title: "State DB Recovery"
description: "How Hermes recovers state.db when the FTS index or the file itself is corrupt"
---

# State database and FTS recovery

For the user-facing walkthrough (what to stop, what the files beside `state.db` are, why
maintenance commands refuse while a writer is live) see
[Session storage recovery](../user-guide/session-storage-recovery.md).

`state.db` stores two different data classes:

- `sessions` and `messages` are the canonical transcript.
- `messages_fts*` tables and their sync triggers are derived search indexes.

The derived indexes may be detached temporarily. They must not turn a live
message write or search into an unbounded full-transcript rebuild.

## Durable-redaction storage sanitation (schema v33)

Opening an existing database below schema v33 redacts the durable message,
provenance, routing-display, and prompt projections and rebuilds available FTS
indexes. The migration enables SQLite `secure_delete`, commits those logical
changes, then requires `wal_checkpoint(TRUNCATE)`, `VACUUM`, and a final
`wal_checkpoint(TRUNCATE)`. The schema marker advances to v33 only after physical
sanitation succeeds and all applicable FTS migrations complete. A blocked
checkpoint, failed rewrite, or non-empty freelist fails startup with the prior
schema version intact so a later sole-opener can retry. An incomplete FTS
migration also holds the prior version, even when sanitation succeeds.

Versions 31 and 32 also existed upstream without durable redaction, so they
must pass the same sanitation gate. The v31 legacy import-identity reset is
recorded atomically and runs only once: retries must not erase newly keyed
identities after their raw provenance has already been projected. Databases
already stamped v33 are not automatically re-sanitized on each open.

After a clean close, this removes legacy raw token bytes from the migrated
projection fields in the active `state.db` and its active `state.db-wal`, provided
those bytes are not also retained in an intentionally literal field. Canonical
tool-call operands are not scrubbed: a token present in an operand can still be
recovered from the active database and primary FTS. This is not a blanket
no-secrets-in-storage guarantee. It does **not** erase bytes from backups, prior
copied or external journal files, filesystem snapshots, storage-device remanence,
or other media outside the active SQLite image.

## Durable-redaction boundary and exceptions

Durable redaction covers display, history, audit, and provenance projections:
session/message display fields and FTS, the legacy `sessions/sessions.json`
mirror's display/provenance/metadata fields, and A2A audit/conversation JSONL.
Existing legacy mirrors are rewritten when loaded, and recovery applies the
same fail-closed checkpoint/VACUUM/freelist verification before accepting an
output database. These protections apply to redacted projections, not all data
reachable through a transcript read or export.

Canonical `messages.tool_calls` arguments remain literal for execution and replay.
Their secrets can appear in primary FTS search snippets, session exports, and
recovery/move export payloads. Display redaction must not be mistaken for an
access-control boundary around those interfaces. Protecting arbitrary operand
secrets at rest requires a separate non-retention or encrypted-replay design.

This is not general metadata encryption at rest. The documented raw
operational exceptions are restart-critical routing/session/model fields:
`gateway_routing.session_key`; `sessions.json` session keys and IDs, timestamps,
origin routing IDs (`platform`, chat/user/thread/scope/profile IDs), resume and
active-turn state, counters/cost state, reset state, and model overrides; and
state-db session routing/peer identity, workspace/Git/model/tool restoration,
and structured lifecycle fields. `delivery_obligations.content` and all of its
routing/delivery state remain raw so owed deliveries can be dispatched after a
restart. A2A audit envelope IDs (`peer`, `task_id`) and conversation
context/task/input admission IDs also remain raw routing metadata; only their
summary/transcript text is redacted. Owner-maintained peer/admission identity
and retry state likewise remain operational, not display projections. Encryption
at rest for operational metadata is a separate future capability.

## Live behavior when FTS is corrupt

If an FTS write or search reports the corruption error class, `SessionDB`:

1. records the durable `fts_stale` marker;
2. removes the FTS sync triggers in the same transaction;
3. retries canonical writes without the derived-index sinks; and
4. serves searches from canonical rows through the `LIKE` fallback.

The failing live operation never runs `FTS5('rebuild')`. Existing recovery
ownership remains unchanged: a later `SessionDB` open may rebuild under the
cross-process admission lock and foreign-holder guard. If that guarded rebuild
cannot run, FTS remains detached, canonical writes stay available, and
`hermes doctor` reports the explicit repair command.

## Live behavior when the file itself is corrupt

If a live write reports bare `SQLITE_CORRUPT` / `SQLITE_NOTADB` (`database
disk image is malformed`, `file is not a database`) with no FTS provenance,
the damage is in a canonical B-tree, the schema, or the freelist. `SessionDB`
then quarantines that handle (`StateDbCorruptError`):

1. the failing write propagates the typed error and nothing is retried;
2. later writes on the handle fail immediately without touching the file;
3. the handle never reopens its connection after `close()`; and
4. `close()` skips its explicit WAL checkpoint.

Stopping the writes is the protection. In the field, a handle that kept
writing for ~50 minutes after the first structural error checkpointed 15
pages under the wrong page numbers on shutdown (page 1 received a
`messages_fts_trigram_data` leaf) and turned a damaged-but-readable file into
one that no longer opened at all. Skipping the explicit checkpoint is the
second line of defence; on Python 3.12+ the quarantine also disables
SQLite's own last-connection checkpoint (`SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`),
so the `-wal` sidecar survives `close()` for forensics. On Python 3.11 that
switch is unavailable and SQLite may still checkpoint once on close, so copy
`state.db`, `state.db-wal` and `state.db-shm` together before restarting
anything.

The gateway and the agent flush path treat the quarantine like a replaced
file: pending transcripts go to `sessions/<id>.jsonl` and the gateway
`pending_messages/` spool instead of the retry queue, and the FTS one-shot
rebuild never runs on the damaged file. The quarantine is per process — the
shared handle stays poisoned for every holder until the process restarts on a
repaired or restored file. Do not run `hermes doctor --fix` while the gateway
is still up. Next steps:

```bash
hermes gateway stop
HERMES_HOME="$HOME/.hermes" hermes sessions recover --source "$HOME/.hermes/state.db" --inspect-only
# if recoverable:
HERMES_HOME="$HOME/.hermes" hermes sessions recover --source "$HOME/.hermes/state.db" --output "$HOME/recovered-state.db"
```

or restore the newest snapshot from `state-snapshots/`.

## Explicit repair

Stop every process that can open the profile database before repairing it.
Keep them stopped for the complete repair and verification window.

```bash
hermes gateway stop
HERMES_HOME="$HOME/.hermes" hermes sessions repair --check-only
HERMES_HOME="$HOME/.hermes" hermes sessions repair
```

`sessions repair` creates a SQLite backup by default and performs structural
work through the repository's guarded snapshot-and-promotion path. Do not copy
`state.db`, `state.db-wal`, and `state.db-shm` independently with `cp`; those
files are one live SQLite image.

After repair, verify the health probe, stale marker, trigger set, and canonical
row counts before restarting the gateway:

```bash
HERMES_HOME="$HOME/.hermes" hermes sessions repair --check-only
sqlite3 "$HOME/.hermes/state.db" \
  "SELECT key, value FROM state_meta WHERE key = 'fts_stale';"
sqlite3 "$HOME/.hermes/state.db" \
  "SELECT type, name FROM sqlite_master WHERE name IN
   ('messages_fts_insert','messages_fts_update','messages_fts_delete')
   ORDER BY name;"
sqlite3 "$HOME/.hermes/state.db" \
  "SELECT 'sessions', COUNT(*) FROM sessions
   UNION ALL SELECT 'messages', COUNT(*) FROM messages;"
```

The marker query should return no row, the expected FTS triggers should be
present, and canonical row counts must not decrease. If repair fails, preserve
both the live database and the reported backup; never delete canonical rows to
make a derived-index error disappear.
