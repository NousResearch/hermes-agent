# Hosted rooms across local profile owners

Hosted room execution uses the canonical session authority. Each profile runs its
own ordinary gateway process and owns only its own state database. The source
room driver contacts destination owners through their private OS-authenticated
control sockets; it never launches a legacy session worker or opens a destination
database.

Configure explicit destinations in the source profile's configuration:

```yaml
hosted_rooms:
  profiles:
    helper: /absolute/path/to/profiles/helper
```

The name must match the destination profile name. The destination profile needs a
live owner: its own standalone gateway, or the default multiplexer serving it under
`gateway.multiplex_profiles` (each served profile gets its own authority and
database there). Directory existence alone does not authorize execution.
Configuration entries are exposed as roster metadata without reading the
destination config or sessions. Hosted-room execution across two profiles served
by the same multiplexer is not live-certified yet.

The destination reverse-checks current room ownership, membership, exact durable
task identity, hosted generation and frozen prompt with the source at submit and
again before dequeue. It retains a source-namespaced binding in its own database.
An unavailable source or destination fails closed; retained work is not silently
rerouted. This private producer interface is not available to public WebSocket
clients. Local same-OS-user trust applies to the control sockets.

Startup publishes adapters before hosted recovery. Canonical supervision replaces
the legacy room driver, avoiding competing leases. Completed retries recover the
same receipt without new inference. Started work after an owner crash is unknown
and pauses followers. Retry cannot replay unknown work. The owning user must use
Discard with the exact room, member, task and hosted generation; canonical unknown
resolution commits before the source driver's cancellation receipt. Repeating that
exact discard recovers a lost acknowledgement.

Desktop discovers canonical groups and sends controls to the captured connection,
profile and room. Unknown work offers confirmed Discard, never Retry. Existing
renderer-only groups require explicit new canonical-room creation; history is not
automatically replayed as new work. The canonical workspace currently handles text
and execution controls; attachment publication remains a separate room capability.


## Cross-gateway document input

Document-capable RoomLink clients opt in with `Hermes-Room-Features: document-input-v1` when reading authenticated `/v1/room-members/capabilities`. Support is returned as a `document_inputs` sibling, outside the legacy catalog. The old catalog, its digest and text-only dispatch mapping remain unchanged. Generic `attachments` stays false; the document feature advertises only `file` and `pdf` kinds and its bounded limits.

A document dispatch binds each input’s source event, attachment identity, recipient member, kind, basename, MIME type, size and SHA-256. This immutable manifest is part of the signed request and logical run fingerprint. Only base64 transfer bytes are excluded from that fingerprint. The source retains the authorized manifest before dispatch. A manifest-only run request first checks accepted canonical evidence; only the authenticated, pre-admission `room_document_input_required` response starts lazy byte fulfillment. Accepted replay does not depend on source bytes or current ingress limits. Missing or ambiguous evidence is never permission to start the work again.

The receiver verifies the complete batch before entering existing verified-document preparation and canonical accepting-write custody. The prepared payload includes the complete `api_turn_v1` admission, so its retained copies are bound to the correct session, request and payload digest. No second staging store or file-download authority is created. Interrupted preparation uses the existing logical-attempt and reclamation mechanisms.

The feature permits at most 8 inputs, 5,000,000 bytes per input and 6,000,000 bytes per batch. A smaller positive `gateway.max_inbound_media_bytes` limit applies; zero or negative keeps the feature’s own fixed bounds. The signed proof plaintext budget is 9,000,000 bytes plus the 16-byte AES-GCM tag, below the unchanged generic 10 MB API body limit. This input capability is independent of the document-output capability below. Authority-host takeover remains separate work.

## Cross-gateway document output

Document-output clients negotiate `document-output-v1` through the same scoped capabilities endpoint. The optional `document_output` contract is frozen in the signed dispatch before submission; absent consent keeps the legacy wire shape. The receiving canonical admission binds the existing `share_group_file` tool and private outbox to the exact room, member, task, execution generation and source installation. The feature uses the same 8-document, 5,000,000-byte per-file and 6,000,000-byte per-turn bounds, without changing local file output.

The group host uses the existing durable output obligation and per-thread publication hold. It verifies transferred bytes, commits the Bot’s reply and attachments, checks the published copies, and only then acknowledges the source. Source reads require scoped status permission; acknowledgement and discard also require control permission. Each disposition is checked against the canonical accepted Run and its immutable output scope and manifest. Network work stays outside the group’s policy lock, so a delayed transfer does not block Stop.

Published output enters the normal group attachment feed. A later addressed Bot receives the exact version through the existing local custody or peer document-input path; no secondary invitation publication service is introduced. Closing Desktop only detaches the viewer. Missing receipts, unresolved execution, damaged publication evidence or unavailable source authority do not prove that cleanup completed.
