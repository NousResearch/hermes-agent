# Webhook Bot Chat persistence boundary

Webhook events execute in their own producer session. The canonical recipient Bot
Chat is resolved by the native owner, by title, and receives an automation input
through its existing FIFO. The producer never executes an event prompt in the Bot
Chat, reopens a recipient writer, or treats a queued receipt as settled inference.

## Owners

- `gateway/webhooks/`: target addressing/pins, committed-final presentation,
  prepared direct/hook origins, verified completion/process lineage and bounded
  reconciliation on the existing gateway loop.
- `persistence.authority.webhooks`: the typed ledger/record/incarnation contract.
  `Persistence.webhook_deliveries` is the only producer ledger door. SQLite earns
  `webhook_delivery_fences`; an unsupported backend refuses that capability.
- `persistence/sqlite/authority/`: ledger transaction bodies and retirement policy.
  Source deletion and the existing housekeeping expiry invoke that policy inside
  their own transactions. No gateway SQL callback receives a connection.
- `hermes_cli/webhooks/inspection.py`: bounded, allowlisted public projection,
  through a non-creating reader. HTTP profile authorization and MCP server-route
  restrictions still belong to their existing public entry points.

## Transaction boundaries

`publish` receives a closed `WebhookLedger` and a `WebhookRecord` with the revision
it observed. It checks the current epoch, the owner-memory gate, record revision,
immutable identity/destination and the irreversible payload-compaction marker in
one writer transaction. First publication and each pre-dial claim also check the
source admission, linked admission and webhook root against their live sessions
in that snapshot. Normal one-shot ending does not revoke the source; deletion does.

The claim commits uncertainty **before** physical I/O. Deletion that wins before
that claim refuses delivery. Deletion after the handoff starts retains its receipt
custody; the observer is still tracked and cannot be mistaken for a never-sent
request. Missing admitted evidence and uncertain post-dial failures never authorize
another native admission. A proven withdrawn pre-admission pin may retry the same
stable delivery identity. Reconciliation observes existing receipts before discovery.

Deletion stamps associated original/completion/linked/process producer copies.
Expiry uses the source result's existing retry horizon; standalone preparation uses
its creation time, and legacy standalone rows start aging on first maintenance.
Compaction drops message/native envelopes but retains identity, digest, pins,
outcome and evidence flags. A stale observer cannot restore that payload, even
when it finishes a physical handoff after expiry. Recipient mailbox retention is
not changed.

## Recipient incarnation

An expected incarnation contains the canonical home, directory identity, file
identity and store instance ID. Capture is read-only and never creates or migrates
a target. A missing first store remains explicit; it can be frozen once under the
original directory after its owner initializes it. A pin is never recaptured to
read back a duplicate obligation after the recipient was removed.

The native mailbox checks the expected incarnation before resolving its chat. The
**typed `incarnation` admission extra** repeats the open-writer check in the FIFO
insert transaction, alongside (not instead of) the existing gate, supersede/fold,
hosted-member and generation fences. This compares the held writer generation and
its own connection's store instance, not just a fresh connection at the path.
Directory/file replacement refuses old obligations; callers omitting an expected
incarnation retain their native contract. An unpinned legacy obligation may only
observe a receipt proven by the same native admission/message, never send anew.

## Regression ownership

- `tests/contracts/authority/test_webhook_deliveries_contract.py`: capability door,
  publication revision/epoch/owner fences, source/root lifetime and bounded reads.
- `tests/persistence_sqlite/authority/test_webhook_deliveries.py`: transactional
  writer-incarnation admission and irreversible compaction.
- `tests/gateway/test_webhook_*`: signed ingress, presentation, native receipt
  identity, direct/hook preparation, coalescing, completions/processes,
  cross-profile lifecycle, crash recovery, retention and cancellation custody.
- CLI/HTTP/MCP parity and actual Web/Desktop JSX tests own their separate public
  contracts. Generated canonical wire artifacts come only from
  `scripts/gen_gateway_contracts.py`.

A static source check or generated-contract match is not a behavioral receipt.
Execute the focused files with the repository runner in an isolated test environment;
run real process/model-loopback lanes separately from controlled transport tests.
