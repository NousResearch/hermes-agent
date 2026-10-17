# Durable input-to-output transfer

The normalized gateway event is persisted before ordinary-message admission. Its
original platform identity, reply anchor, routing and incoming attachments survive
queueing and batching. This boundary is after platform normalization: it does not
claim to make a platform's polling acknowledgement transactional with admission.

## Ownership invariant

One logical response has one authoritative delivery-ledger row. Every contributing
input has a transactional link to that row before its spool record is retired.
Already-transferred input does not invoke the model again; unfinished input still
recovers. A fresh identity with identical text remains a new request. Mixed batches
are rebuilt from their untransferred members rather than assigning old members to
a second output. There is no text-similarity classifier or alternate sender.

The output retains normalized text, ordered attachment dispatch units, reply/routing
metadata and per-component completion. Normal, queued and already-streamed final
paths use the existing delivery owner. Recovery skips confirmed components and
reuses attachment dispatch. Failed turns do not upload generated attachments.
Missing or denied paths remain failed rather than silently completing the output.

Successful platform receipt IDs are associated with the authoritative output.
Fresh reply ingress resolves these IDs within session/platform/chat/thread scope.
Unknown or ambiguous receipts stay unknown. Typed original identities and prior
output disposition enter the new turn and existing transcript display metadata;
cached prompts and old messages are not rewritten.

## Persistence and rollback

Schema initialization adds an input-link table, a nullable output payload and a
receipt table. Legacy text-only output rows remain readable. Receipt rows expire
with their output; input-link tombstones prevent re-execution after content
retention and currently have no pruning policy.

Older code can open the extended database but cannot recover attachment manifests
or honor component completion. Drain or explicitly reconcile pending manifests
before downgrade. Preserve spool/database evidence for reconciliation.

## Limits and verification

A send accepted by the platform before its receipt is persisted is ambiguous.
Retry may repeat accepted content without rerunning the model. Partial success
inside a multi-chunk send or image batch is also ambiguous. This is not exactly-once
external delivery. Attachment paths are retained, not immutable byte snapshots;
deleted or replaced files require explicit handling. Automatic TTS is not covered
by a complete crash-recovery guarantee.

Synthetic disk-backed tests use real gateway/ledger/compaction implementations and
fake transports. They cover duplicate and mixed-batch admission, fresh identical
text, crash/record/send boundaries, failed attachment recovery, exact-route receipt
lookup, and retained identity through compaction and reopening. Fake provider
responses establish plumbing only, not semantic behavior of a language model.
Policy-level quality of answers to follow-ups, repeated URLs and complaints needs
separate model evaluation. Runtime rollout and historical reconciliation are
operator decisions, not consequences of applying this source patch.
