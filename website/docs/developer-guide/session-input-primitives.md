# Session input primitives

Three generic seams let a plugin batch inbound input, hand it to a running turn
at a checkpoint, and render it — without core learning any profile or window
policy. With no provider registered, every function is a no-op and admission,
claiming and the turn loop behave exactly as before.

| Primitive | Where | What core owns |
| --- | --- | --- |
| `session_input_policy` | `session_input_primitives.policy_for` | Validates the profile-scoped window/eligibility snapshot before the write transaction; folds a follower into the FIFO tail in the same transaction as its retry alias. |
| `session_input_admitted` + transfer | `gateway/session_input_transfer.transfer_after_submit` | After the commit: registration, the durable `steer_session_input` write, delivery to the adopted agent, requeue when the turn refuses it. The submitting viewer disconnecting never aborts a committed hand-over. |
| `checkpoint_input` | `session_input_checkpoint.checkpoint_input` | Appends already-ticketed text at the tool-batch boundary (`before_tools`) or when a model request is rebuilt (`before_request`), keeping the cached transcript prefix intact. |

## Provider protocol

A plugin registers one provider (`register_provider(provider)` /
`unregister_provider`). All hooks are optional; an exception disables that hook
and the core keeps running.

```python
policy(profile_id, payload) -> None | {
    "eligible": bool,              # may this payload batch at all?
    "first_window_seconds": float, # (0, 60]
    "quiet_seconds": float,        # (0, 60], restarted by each arrival
    "hard_cap_seconds": float,     # >= max(first, quiet)
    "signature": str,              # only equal signatures share one batch
}
render(parts) -> str               # text a batch commits with
format_part(text, *, count, arrived) -> str  # one ticket appended mid-turn
admitted(snapshot) -> bool         # offer this committed batch to the turn now?
transfer_mode(phase) -> "steer" | "redirect" | "queue"
```

Providers never receive a SQLite connection, a live agent or a lock; `snapshot`
and `phase` are immutable DTOs, and the policy snapshot is frozen before the
admission transaction opens.

## Durability

A batch record is versioned in the admission payload
(`session_input_batch_v1`) and each follower's retry alias lives in
`state_meta`, both committed atomically with the admission. The claim path
honors the record's deadline outside the writer lock. Accepted text therefore
stays retryable, visible and settleable when the plugin is disabled, crashes, or
is removed — only *new* batching stops.

## Pre-existing records (opt-in)

Core reads exactly the names it writes. A deployment that migrated rows written
before this module existed opts in to reading them, so core itself never names a
concrete legacy key:

```python
session_input_primitives.register_legacy_names(
    batch_keys=[...],          # payload keys an older writer used, oldest first
    alias_prefixes=[...],      # state_meta prefixes its retry aliases used
)
```

Registered names are read, never written; `clear_legacy_names()` returns to the
default, where nothing but `session_input_batch_v1` and `session.input.alias.v1.`
is recognised. Nothing else in the primitive depends on the registration.

## See also

- `plugins/session-input-inbox-example/` — a worked provider (default off).
- `website/docs/user-guide/features/hooks.md` — why the existing plugin hooks
  cannot express this: `pre_gateway_dispatch` does not run on Local/WS input,
  and `post_gateway_admission` cannot fold payloads atomically or transfer a
  committed admission into the running generation.
