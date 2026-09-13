# Hosted Room Bot Artifacts

Bot file handoff is scoped to an admitted Group Chat task. The existing
`share_group_file` tool is registered in `bot_room`, injected at the room-service
boundary, and refuses calls without an internal artifact scope. It is not added
to the core toolset or ordinary chat tools.

## Ownership

- `gateway/hosted_room_artifacts.py` owns the private outbox, exact task and
  execution-generation scope, no-follow file reads, bounded receipts, and
  retirement evidence.
- `tools/hosted_room_artifact.py` copies allowed local or execution-backend files
  into that outbox. It imports path resolution from `tools/file_tools_paths.py`.
- `tui_gateway/hosted_room_artifact_service.py` imports verified output into the
  existing Files store, publishes through the canonical cursor fence, then ACKs
  the source. Private/unpublished output is retired without becoming history.
- `gateway/hosted_room_attachments.py` retains its Files/catalog implementation.
  The artifact addition is `abort_unpublished_event`, which cannot revoke a
  durable owner event. Existing viewer-read and atomic commitment gates remain.
- `gateway/hosted_room_discussion.py` preserves admission-time recipients and
  passes published member attachments to subsequent Bots. Older tasks without
  recipient snapshots can replay text but cannot publish files using a new roster.
- `tui_gateway/hosted_room_server_rpc.py` binds the local task's artifact scope
  and finalizes it while profile context is active. `methods_prompt.py` validates
  the complete internal proof; external callers cannot supply it.
- `gateway/platforms/api_server_runs.py` binds room source/policy/artifact context
  for authenticated peer runs and retains output manifests for replay. Reserved
  run metadata is rejected by `api_server_room_dispatch.py` for ordinary auth.
- `gateway/platforms/api_server_room_artifacts.py` serves exact scoped artifact
  GET/ACK/discard operations; `api_server_room_grants.py` registers them and
  performs best-effort private cleanup after authority revocation.
- `tui_gateway/hosted_room_peer_http.py` uses one URL builder for JSON, input PUT,
  and artifact GET, preserving profile prefixes and mounted base paths.

## Recovery Boundaries

The room log remains authoritative. Publication precedes ACK; a lost ACK can be
retried without republishing. Competing rollback cannot revoke a committed
publication. Superseded or non-visible terminal output stays private. Retry
metadata keeps its driver task alive until the existing finite cleanup boundary.

Fresh preflight rejection is only invocation-local evidence. After lease loss,
a successor may have accepted the same input batch. Error finalization drops
local staged bytes but never eagerly sends a remote DELETE based on preflight
evidence; remote expiry remains the cleanup backstop. The accepted-input schedule
is exercised by `test_peer_preflight_accepted_batch.py`.

Route hydration, observed-grant error fencing, passive replicas, and rejecting
probe cooldown belong to the route layer. Artifact import and retirement call
those real owners; there is no cached-route substitute or compatibility shim.

## Provenance

This ports the existing #99159 implementation from David Dudok de Wit's
`4fdca7f680dbf54310829cd609899852e685a75e`, preserving the original
`03ec5da8b1` handoff, `457933e397` artifact-owner extraction, `397c8e9e03`
forged-metadata guard, `bedf9e257b` output recovery, `ae06c7e56f` route fencing,
`c82b3e18ca` binary profile routing, and `4fdca7f680` accepted-input correction.
Existing Files/catalog owners are reconciled, not replaced by donor snapshots.
