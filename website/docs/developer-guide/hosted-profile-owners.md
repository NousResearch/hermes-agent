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
