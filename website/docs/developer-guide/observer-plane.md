# Hermes observer plane

Hermes exposes a receive-only observer source under `/observer/v1`. It is a
separate route family from `/api/ws` and is enabled only when the process has a
separate `HERMES_OBSERVER_SOURCE_TOKEN` bearer. The observer token is not a
controller token and cannot be used with JSON-RPC methods.

The source captures gateway events as a second sink; it never replaces or
rebinds `session["transport"]`. Events are projected by allowlist before they
are placed in the journal or sent to an observer. Hidden and unsupported source
events become content-free cursor advancement, preserving sequence continuity
without exposing raw payloads.

Each logical session has a random stream epoch and contiguous sequence. The
in-memory replay journal retains at most 10,000 projected records and 15
minutes, whichever is smaller. A cursor below the retained floor is reported as
`retention_exceeded`; an old stream epoch is reported as `stream_restarted`.
Observers should rehydrate from the snapshot after either gap. Observer queues
are bounded and a full/failed observer queue is dropped silently from the
source path; it cannot delay the primary controller sink.

Snapshot hydration records a stream head before reading durable transcript rows.
The returned `through_seq` is the barrier watermark; events after it belong to
the live/replay phase and must not be treated as part of the durable snapshot.
Only safe user/assistant transcript rows and coarse activity/status metadata are
projected. Tool arguments/results, reasoning, request bodies, credentials,
local-read/setup payloads, and unknown raw bodies are excluded.
