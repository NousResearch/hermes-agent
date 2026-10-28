---
title: Peer session messaging
---

# Coordinate two local sessions

Peer messaging lets independent CLI, TUI, or Desktop conversations exchange
short project updates without sharing their conversation histories or being
converted into subagents. It is opt-in through **hermes tools → Peer Messaging**
in both sessions. Start new sessions after enabling it so their tool sets stay
stable for prompt caching.

The toolset provides `peer_sessions`, `peer_send`, and `peer_receive`.
`peer_receive` is resolved through the existing tool registry's toolset union;
none of these tools is added to the default core bundle.

## Example: avoid overlapping edits

Tell both sessions that they are working together and should agree on file
ownership before editing. This is coordination, not a filesystem lock: a message
cannot prevent another agent from editing a file before it reads the update.

1. Session A calls `peer_sessions()` to discover other local conversations in the
   same profile and project. The grouping key is the stored Git root, falling
   back to the exact working directory. `same_project=false` searches the same
   profile without the project filter. An open owner and recent activity do not
   establish that a model turn is running.
2. A calls `peer_send(target_session_id="...", message="I am changing the API schema; please handle the UI.")`.
   It receives a `message_id` and `status="queued"`, not a claim that B read it.
3. B receives a fixed inbox hint while active and calls `peer_receive()`.
   The tool result contains A's session ID, the message, and its stable ID.
4. B replies with `peer_send(target_session_id="<from_session_id>", message="Understood; I will change only the UI.", in_reply_to="<message_id>")`,
   then acknowledges the original with `peer_receive(action="ack", message_ids=["<message_id>"])`.
5. A reads and acknowledges B's reply. Neither session needs to copy its full
   conversation to the other.

For a brief rendezvous, `peer_receive(wait_seconds=10)` waits for incoming
messages, up to 30 seconds. Prefer useful work over repeated polling. A wait does
not start the other session or guarantee that it will reply.

## Delivery contract

- Reads return at most five messages, each at most 8,000 characters. `has_more`
  indicates that another page remains; acknowledge handled IDs to advance.
- Reading does **not** consume the envelope. Failed hints, failed tool-result
  delivery, and a receiver restart can therefore read the same ID again.
- An explicit recipient ACK removes only the named messages from its own inbox.
  Repeating an ACK is safe; `not_pending` means absent, not proof of prior
  delivery. ACK means mailbox consumption, **not task completion**.
- The same-process-shared lock used elsewhere in Hermes serializes publication
  and consumption. Each inbox admits at most 50 pending messages, including
  concurrent senders. Invalid envelopes are discarded; temporary I/O errors
  leave valid messages available for retry.
- Compression continuations can read their ancestors' queued mail. Old target
  IDs resolve along the existing compression chain; user-created branches do
  not inherit access to their parent's inbox. Ended or archived targets are
  refused rather than reopened.

This is **at-least-once reading**, not exactly-once execution. A process can stop
after doing work but before acknowledging the message. Use the stable ID and
actual project state when reconciling a repeat; do not blindly repeat effects.
ACKs are not durable execution receipts or independently verified evidence that
the model understood the content.

## Boundaries

Peer bodies appear only in normal tool results. The existing activity hook
passes a fixed content-free inbox hint through steering; peer-controlled text,
names, IDs, roles, and authorization flags are never injected into a user row.
Messages remain lower-authority peer data, not user approval or permission to
change the recipient's task.

This increment is for independent local CLI/Desktop/TUI conversations in one
profile home. It does not grant access to managed messaging sessions, Bot rooms,
cron jobs, delegated workers, or other profiles. Both participants must enable
the toolset; a queued message does not enable a disabled recipient.

There is no automatic wake of an idle session, session spawning, broadcast,
shared task board, file locking, or new scheduler. An idle session reads queued
mail when it next runs. A2A and Bot Chat retain their own authorization and
execution-receipt contracts. The mailbox assumes the existing same-OS-user,
profile-home trust boundary; it is not an isolation mechanism against another
process running as that user.
