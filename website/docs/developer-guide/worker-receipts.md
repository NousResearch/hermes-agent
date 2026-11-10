# Worker receipt acknowledgement and recovery

Managed workers persist through the authenticated profile owner, not a local SQLite fallback.
`worker.persist` commits its operation and `(execution_id, sequence, payload_digest)` receipt
atomically. The worker waits **90 seconds**: the owner's transcript write-lock patience is
60 seconds, leaving 30 seconds for scheduling and transport. This is a bounded delivery budget,
not a guarantee that arbitrarily slow storage will complete.

## Uncertain acknowledgements

A receive timeout does not prove rollback. The worker drops the uncertain connection and sends
one separately bounded (10-second receive budget) `worker.receipt` request to the authenticated
owner with the exact scope, operation and payload. This request only reads a committed receipt;
it never resends or executes the mutation and never creates a receipt. Connection discovery and
handshake retain their own existing bounds.

The owner verifies the same profile, capability, principal, adoption secret, live PID/birth,
assignment and generation as persistence. A single read snapshot checks the runtime/worker epoch,
generation, sequence and payload digest before returning the original result. An exact committed
compression publication may resolve its original parent assignment, allowing the outbox to move
to the recorded child. Foreign, conflicting, missing and retired receipts fail closed: the exact
entry remains in the private pending journal and failure stays sticky. Explicit retry/adoption
continues to use the existing durable replay protocol; automatic timeout reconciliation does not.

## Commit is the acknowledgement boundary

Periodic PASSIVE WAL checkpoints and bounded FTS merges run on a SessionDB-owned background
worker, not inline before returning the committed result. There is at most one active pass and
one pending bit per maintenance kind, rather than an unbounded executor queue. Idle workers exit.
The spawning context preserves the owning profile. Close stops new scheduling and drains pending
maintenance before retiring SQLite; a legitimate late reopen creates a new maintenance lifecycle.
Thread-start failures preserve due work for the next write or the close-time drain. Maintenance
never reopens a closed connection, and existing quarantine/generation checks still apply.

## Local diagnostics

`worker_receipts.created_at` is a nullable insertion-time Unix timestamp. Writable open performs
the existing additive column reconciliation; historical rows remain **NULL**, since their commit
time is unknown. An exact replay preserves the original timestamp.

Debug logging records fixed stage names and numeric durations: owner receive, SQLite lock wait,
commit, and response send completion (`ack`), plus worker receive duration. These new records
contain no request IDs, execution/session/profile identifiers, payloads, prompts or secrets and
are not outbound telemetry. Owner `ack` means send completion, not proof that a remote caller
consumed the frame. Stages deliberately have no identifying correlation token.

## Provider outcomes

Terminal API results carry the classifier's numeric `failure_status_code` through the bounded
managed-worker result frame. Exhausted/non-retryable 4xx refusals use the provider exit policy
and Kanban records `provider_rejected`, not `crashed`. Retryable provider failures retain their
existing requeue policy. In particular, HTTP 429 alone is not evidence of a permanent failure:
rate limits and quota/billing holds retain `rate_limited`, the classifier's retry verdict/reset
metadata, and do not blindly trip the task-failure breaker. Genuine process loss remains a crash;
interruption and persistence failures retain their separate handling.
