# Hosted rooms across local profile owners

Hosted room execution uses the canonical session authority. Each profile runs its
own ordinary gateway process and owns only its own state database. The source
room driver contacts destination owners through their private OS-authenticated
control sockets; it never launches a legacy session worker or opens a destination
database.

A room may seat the profiles its gateway serves: the gateway's own profile and,
under `gateway.multiplex_profiles`, every profile the multiplexer currently serves
(each served profile gets its own authority and database there). For a host-scoped
connection (Desktop's shared socket, the dashboard) these need no configuration;
this is the path Desktop's New Group Chat takes. A connection bound to one profile
seats that profile plus the profiles its `hosted_rooms.profiles` entries name, as
before; an entry naming a served profile must point at its served home. A profile
that another local gateway owns joins only as an explicit destination in the source
profile's configuration:

```yaml
hosted_rooms:
  profiles:
    helper: /absolute/path/to/profiles/helper
```

The name must match the destination profile name, and the destination needs a live
owner: its own gateway (a `gateway.standalone` profile) or another local
multiplexer that serves it. One home answers to one name. On a host-scoped
connection a served profile keeps its served home whatever its entry says. An entry
that names a home already seated under another name, or is malformed, is ignored
with one log line, and only a new member naming it is refused; a room that already
seats such an alias keeps running its other members. Directory existence alone does
not authorize execution, so a profile that is installed but not served here
(parked, or owned by a gateway this one has no entry for) is refused at room
creation, with the member named in the refusal. A served profile whose
`profiles/<name>` directory is a link is refused too: the member transport cannot
reach it under its name yet. Configuration entries are exposed as roster metadata
without reading the destination config or sessions.

Execution between two profiles served by the same multiplexer is covered by
in-process tests that drive the real private control socket. It is not yet certified
on a live multi-profile install, and members whose profile directory is a link cannot
run turns.

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
