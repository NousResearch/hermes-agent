---
title: Exact session attachment
description: Non-executing stored-session attachment and its recovery fence
---

`session.attach` is the gateway's non-executing attachment primitive. It takes
`{session_id, profile}` with the full stored primary key and canonical profile
name. Both are required. A title, prefix, runtime ID, profile alias or missing
row is refused. Compression descendants are never selected implicitly; the
returned `parent_session_id` is informational lineage only.

The result identifies the requested and bound stored ID, runtime ID, profile,
runtime incarnation, observed owner incarnation and disposition. Repeated calls
reuse the same runtime while it survives, including after a lost RPC reply.
After runtime loss a new incarnation is returned for the same stored identity.
An ambiguous local runtime, closing/rotated watcher, settling disconnect or
foreign live lease is refused without replacement, interruption or takeover.

For an exactly matching surviving runtime, attachment uses the existing additive
transport membership machinery. Already authorized work, queued inputs and open
requests are left untouched. Existing live `session.activate` remains available.

A cold attachment registers an **inert** runtime record. It does not build an
agent, hydrate or repair a transcript, reopen/end a stored row, retire an
interrupted-turn marker, start session services, schedule auto-continuation,
execute prompts/tools/goals/notifications, drain a queue, or answer a request.
History remains in the existing paged store. Closing/reaping the inert record
does not finalize the stored conversation or unregister another owner's notifier.

Cold legacy work is `interrupted` only when a strictly valid marker identifies a
writer that is known dead. Current markers carry a writer PID and, when available,
its process start time; an alive or unprovable writer leaves the disposition
`unknown`. Older markers without writer identity also remain unknown. A missing,
stale or disabled marker is not a durable terminal receipt. Unreadable, malformed,
duplicate-key or unsupported marker state stays unknown and fenced. The recorded
prompt is never returned in the descriptor.

The **execution fence** persists for the life of that inert record. Prompt
submission and execution-dependent RPCs return `4091` with
`reason: attachment_execution_fenced` before agent construction, ownership
admission or prompt persistence. Queue/notification admission, direct synthesized
turns, isolated compute dispatch and auto-continuation also refuse. Calling
legacy `session.resume` or `session.activate` on it cannot remove the fence or
present it as an ordinary idle runtime. Retrying attachment does not settle old
work. There is deliberately no generic “clear fence” flag: until a separate
explicit disposition contract exists, start a distinct new conversation for new
work. This preserves the old marker, transcript and execution uncertainty.

Identity and local runtime disposition are captured under the registry/reattach
and history locks. The descriptor does **not** claim an atomic cut across the
transcript, replay ring and request registry: `recovery.complete`,
`events_complete` and `execution_complete` are false; history is not loaded;
request lifecycle is unknown; history/request revisions, stream incarnation and
replay high-water are null. `replay_epoch` identifies the server process only.
These unknowns are intentional, not empty successful recovery. Obtain supported
live snapshots separately, and do not infer durable request outcomes or complete
event history from attachment. Durable interaction/outcome recovery and
revision-conditional settlement remain separate capabilities.

This is additive to existing create/resume/activate APIs. Their normal live
behavior is unchanged; the fence applies only to records created by the new
cold-attachment path. No database schema, model tool, client-specific behavior
or auto-continue configuration change is introduced.
