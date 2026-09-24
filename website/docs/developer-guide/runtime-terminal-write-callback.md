# Runtime terminal-write callback

The canonical admission terminal transitions accept a private optional
`_terminal_write(conn, admission, outcome, result)` callback:

- `settle_session_input` passes the validated started admission, its requested terminal
  outcome, and the structured result (or `None`).
- `cancel_session_input` passes a validated queued admission, `"cancelled"`, and `None`.
- `resolve_unknown_session_input` passes a validated unknown admission,
  `"interrupted"`, and `None`.
- `gateway.session_results.finish_result` forwards the callback only when it performs
  first settlement. Its existing-terminal branch reads the already committed result and
  does not invoke the callback again.

The callback runs after the applicable current-epoch, admission-status, owner, and
generation fences and inside the same `SessionDB._execute_write` transaction as the
terminal admission, result, worker-retirement, and runtime-revision updates. Raising from
the callback aborts and rolls back that entire transaction. Omitting it preserves the
ordinary terminal transition behavior.

## Caller contract

This is a trusted in-process metadata seam, not a public RPC, plugin API, or external
execution-authority boundary. A callback must:

- use only the supplied SQLite connection for bounded metadata writes;
- never call `commit`, `rollback`, or a self-committing `SessionDB` method;
- never perform filesystem cleanup, network requests, process control, or other physical
  effects from the transaction;
- make its SQL safe to rerun, because `SessionDB` may roll back and invoke the whole write
  callback again after a retryable SQLite contention error; and
- treat the admission, outcome, and result as transaction-local inputs, not as authority to
  execute or publish work.

Terminal replay must recover existing durable state rather than repeat callback work or
rewrite the result. Consumers that need post-commit physical effects must perform them only
after the terminal transition returns successfully and own their separate idempotence and
recovery policy.
