# Native workspace API (protocol 1)

This opt-in API operates in the configured local terminal workspace. Enable
`gateway.api_server.workspace.enabled: true` and set `terminal.cwd` to the intended
workspace directory. An API key is required. The selected profile owns the configuration;
requests cannot select another backend, root, or owner. Other terminal backends retain
their existing behavior and report `unsupported_backend` for this API.

Read `GET /v1/capabilities` → `features.workspace_io` first. Use its authoritative `root`
when constructing command paths; `/workspace` is not an alias. Shell commands have the
same authority as native terminal commands. File path containment is not a shell sandbox.

All workspace routes use POST, reject query parameters, and require JSON `session_id`
for an existing session. Native authentication and profile middleware apply. Original
`Idempotency-Key` is required for write, remove, process start, cancellation and acknowledgement.
Its format is `<admission_epoch>:<opaque suffix>` (at most 256 characters total); read the
32-hex-character epoch from the capability before assigning a key to new work. Persist the
original key with the operation. Never replace the key when retrying an uncertain request. Replaying
an identical mutation returns its receipt; a different payload returns 409. A cancel sent
before start reserves a tombstone, so the later original-key start does not execute.

| Path under `/api/workspace` | Additional request fields |
| --- | --- |
| `/files/read` | `path`, optional `max_bytes` |
| `/files/write` | `path`, `data_base64` |
| `/files/remove` | `path`, optional `force`, `recursive` |
| `/files/resolve` | `path` |
| `/processes` | `command`, optional `cwd`, `env`, `timeout_seconds`, `max_output_bytes` |
| `/processes/cancel` | original start key in the header |
| `/processes/{operation_id}/status` | none |
| `/processes/{operation_id}/output` | optional `stdout_cursor`, `stderr_cursor`, `max_bytes` |
| `/operations/{operation_id}/acknowledge` | original mutation key in the header |

File reads return `{kind:"file",data_base64,size_bytes}` or `{kind:"missing"}` only for
missing paths. Writes create parents and atomically replace the destination. Paths are
relative to root or absolute beneath it. Traversal and intermediate symlinks are denied.
Removal never follows a final symlink. Resolve is lexical; actual I/O validates each path.

Receipts have `operation_id`, `session_id`, `kind`, `status`, `reason`, `process_id`,
`exit_code`, `termination_confirmed`, `output_available`, `stdout_bytes`, `stderr_bytes`,
`stdout_truncated`, `stderr_truncated`, and `acknowledged`. Consumers must not treat `stopping` as finished.
Output responses contain independent `stdout` and `stderr` objects, each with
`data_base64`, `next_cursor`, `truncated`, plus top-level `complete`, `status`, `operation_id`.
Cursors count bytes. Retained output is a bounded prefix; truncation is explicit.

Limits: 262,144 bytes per file, 131,072 retained bytes per output stream, 16,384 bytes per
stream per output response, eight active processes, and 300 seconds per process.
Child environment overrides apply only after native runtime selection; runtime selector
variables are rejected. They do not mutate gateway environment or persistent terminal state.

New mutations are rejected during drain. Existing receipts, output, reads and cancellation
remain available. Admitted work stays visible to drain until it finishes. A disconnected
HTTP request does not stop admitted work; clients cancel using the original key.

Receipts and bounded output snapshots survive gateway restart. Terminal output and its receipt
commit atomically. Output returned while running is also committed before the response. Incomplete operations become
`interrupted`, never automatically re-execute, and do not claim confirmed process termination.
Interrupted output is labeled `partial:true`; an unconfirmed process remains visible to drain.
The ledger stores fingerprints, receipts and output bytes, not command bodies or environment
values. Command output may contain sensitive data and the ledger is profile-private.

The server retains at most 4,096 unacknowledged operations. Admission epochs rotate after
256 reservations. Known receipts replay across epochs. Unknown keys bearing a retired epoch
return HTTP 410 (`workspace_key_expired`) and never execute. A client can fetch a new epoch
for genuinely new work, but must resolve an uncertain original request using its original key.

Acknowledge only **after** the caller durably records the complete outcome. Acknowledgement
before returning from a non-durable method can lose an outcome and is not supported. The
endpoint requires the same session, operation ID and original key. It rejects active or
termination-unconfirmed operations with 409. It returns the receipt with `acknowledged:true`
and `output_available:false`; output then returns 410 `output_acknowledged`. Duplicate
acknowledgements return that receipt while its tombstone remains. Acknowledged tombstones
from retired epochs are compacted automatically; an acknowledgement retry with its original
key then returns 410 `workspace_key_expired`, which means no new effect occurred. An operation
ID alone may return 404 after compaction. Unacknowledged output is never silently evicted.
The finite unacknowledged limit applies backpressure until the caller safely acknowledges.

Local HTTP, native process capture, restart recovery, and retention contracts have focused tests.
This is not a released-container qualification claim; external-backend workspace execution is
unsupported in v1. The supported client contract is also exported as `workspace-api.schema.json`.
