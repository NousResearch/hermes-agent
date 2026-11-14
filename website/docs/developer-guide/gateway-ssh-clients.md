---
title: "Gateway over SSH: client contract"
description: "How an SSH-tunnel client (Desktop or mobile SSH mode) reaches the gateway with hermes gateway attach and hermes gateway tickets"
---

# Gateway over SSH: client contract

This page is for authors of a client that reaches a Hermes host over SSH: it runs commands on
the host, forwards the gateway's loopback port, and then talks to the gateway as a local client
would. Two commands on the host give it everything it needs:

| Command | Gives the client |
|---------|------------------|
| [`hermes gateway attach --json`](./gateway-internals.md#hermes-gateway-attach) | The gateway's loopback endpoint and one single-use WebSocket ticket. With `--start` it starts a gateway when none is running. |
| `hermes gateway tickets --probe --json` | Whether this host has the `tickets` command, which ticket purposes it mints, and whether this login is a Windows service session. |
| `hermes gateway tickets --serve --json` | Single-use tickets over one long-lived SSH channel: one per HTTP request, and (on a host that lists `interactive`) the WebSocket's ticket for every profile. |

Both commands work for the profile the command runs in (`HERMES_HOME`, or `hermes -p <name>`).
A `--serve` request can name another profile that the same gateway serves.
A non-interactive SSH command often does not have `hermes` on its `PATH`; the client has to
locate the binary.

## `hermes gateway tickets --probe --json`

Starts nothing and contacts no gateway. Exits `0` and prints one JSON line:

```json
{"protocol": 1, "purposes": ["native-http", "native-profile-create", "native-profile-registry", "interactive"],
 "windows_service_session": false}
```

(Shown wrapped here; on the wire it is one line.) `protocol` is the version of the `--serve` line
protocol below. `purposes` lists the ticket purposes `--serve` mints. Read them and ignore keys you
do not know: later versions add keys.

A host that lists `native-profile-registry` also accepts the `profile` field of a request (see
below). A host that lists only `native-http` ignores `profile` and mints for the profile the
command runs in, so do not send `profile` to it.

A host that lists `interactive` mints the WebSocket's ticket too (see
[The WebSocket ticket](#the-websocket-ticket)). A host that does not list it answers
`unsupported_purpose`; dial the WebSocket with the ticket `attach` returned there.

`windows_service_session` is `true` when this login runs elevated in Windows session 0 (an
administrator's OpenSSH login); it is `false` everywhere else, and a host that does not report the
key is treated as `false`. It is the value `attach --json` reports in
`start_context.windows_service_session`. See client rule 2.

A host whose Hermes predates this command exits non-zero and prints no JSON on stdout. That is
the signal to use the client's previous SSH path.

## `hermes gateway tickets --serve --json`

Run it once per connection and keep its SSH channel open. It finds the running gateway for the
profile as `attach` without flags does. It never starts one: run `attach --start` first.

All lines are single-line JSON objects, one per line, on stdout (replies) and stdin (requests).

**First line.** When a gateway is ready:

```json
{"type": "hello", "protocol": 1,
 "endpoint": {"api_origin": "http://127.0.0.1:<port>", "profile_id": "...", "instance_id": "...",
              "authority_epoch": 1, "runtime_protocol": 1, "supervisor": "none",
              "capabilities": ["..."], "control_home": null, "code_sha": "..."}}
```

(Shown wrapped here; on the wire it is one line.) `endpoint` is the same object `attach --json` returns in its `endpoint` field. Check that its
`instance_id` equals the one your `attach` returned: if it differs, the gateway was replaced in
between, so attach again.

When there is no gateway to use, the first line is an error and the command exits non-zero with
the exit code `hermes gateway ensure` gives that verdict (`5` when absent):

```json
{"type": "error", "error": {"code": "absent", "reason_code": null, "message": "gateway absent: not ready"}}
```

`code` is the discovery state (`absent`, `starting`, `draining`, `incompatible`, `inaccessible`,
`conflict`), `reason_code` is the reason `attach` gives for it (for example
`gateway_other_session`) or `null`, and `message` is the text `attach` shows for the same verdict.

**Requests.** After the hello, write one line per ticket you need:

```json
{"id": 7, "purpose": "native-http"}
```

`id` is an integer or a string of your choice. `purpose` is one of the probe's `purposes`.
An optional `"profile": "<name>"` asks for a ticket for another profile of the same gateway (see
[Profiles](#profiles)); without it the ticket is for the profile the command runs in. Leave the
key out to mean that: `"profile": null` is refused, not read as absent. A request line may be at most 65536 bytes (64 KiB),
not counting its line ending (LF or CRLF); a longer line is discarded as it is read and gets one `invalid_request`
reply. Each request line gets exactly one reply line, in order, carrying your `id`:

```json
{"id": 7, "ok": true, "ticket": "<ticket>", "expires_in": 30}
{"id": 7, "ok": false, "error": {"code": "ticket_unavailable", "message": "..."}}
```

| `error.code` | Meaning |
|--------------|---------|
| `invalid_request` | The line is not a JSON object with an integer or string `id`, is nested too deeply to parse, or is longer than 65536 bytes (64 KiB, line ending excluded). The reply's `id` is `null`. The channel stays open. |
| `invalid_request` (with your `id`) | `profile` is present and is not a string holding a valid profile name (`null` included), or `profile` is present at all on an `interactive` request. |
| `unsupported_purpose` | `purpose` is missing or not in the probe's `purposes`. |
| `unknown_profile` | No profile of that name exists on the host. |
| `profile_not_served` | The profile exists, but the running gateway does not serve it (it has a gateway of its own, or is parked). Only answered when the same gateway, still `ready`, lists the profiles it serves and this one is not among them; otherwise the refusal is `ticket_unavailable`. The channel is still good for other profiles. |
| `ticket_unavailable` | The gateway did not mint the ticket: it stopped, was replaced, or refused. Close the channel, run `attach` again and open a new one. |

Close stdin to end the command; it exits `0`.

### Profiles

One gateway usually serves every profile of the host, and each ticket is bound to one of them.
The gateway answers `403` with `{"detail": "profile_scope_mismatch"}` when a request selects a
profile other than its ticket's. So name, in the ticket request, the profile the HTTP request
selects:

| HTTP request | `purpose` | `profile` |
|--------------|-----------|-----------|
| Selects a profile with `?profile=<name>`, a JSON body `"profile": "<name>"`, a `/api/profiles/<name>/...` path, or `recents_profile=<name>` on `/api/profiles/sessions/sidebar` | `native-http` | `<name>` |
| Selects none | `native-http` | omit |
| `POST /api/profiles` (create a profile) | `native-profile-create` | omit |
| `PATCH` or `DELETE /api/profiles/<name>` (rename or delete) | `native-profile-registry` | `<name>` |

`profile` is the profile's name as `GET /api/profiles` lists it; the root profile is `default`.
These are the tickets the Desktop app uses for the same requests on its local connection, minted
the same way, with the same limits:

- A `native-http` ticket never creates, renames or deletes a profile (`403`).
- A `native-profile-create` ticket is accepted by `POST /api/profiles` only, and a
  `native-profile-registry` ticket by `PATCH`/`DELETE` of the one profile it names only. Every
  other route answers `401` to them.
- The profile the gateway itself runs in cannot be deleted or moved. Renaming `default` only sets
  its display name.

Creating, renaming or deleting a profile makes the gateway rescan what it serves. For a moment
after it, a ticket request can fail with `ticket_unavailable` or `profile_not_served`, or a
request with `401`; retry.

### The WebSocket ticket

```json
{"id": 8, "purpose": "interactive"}
{"id": 8, "ok": true, "ticket": "<ticket>", "expires_in": 30}
```

An `interactive` ticket opens the gateway's WebSocket (`/api/ws`), as the ticket `attach` returns
does: dial `attach`'s `url` with the subprotocols `attach`'s `protocol` and
`hermes-gateway-ticket.<ticket>`. The difference is its reach. The ticket `attach` returns is bound
to the profile `attach` ran in, and the gateway answers `profile_mismatch` to any frame on that
socket whose `profile` names another one. An `interactive` ticket from `--serve` is the one the
Desktop app asks for on its local connection: one socket on which each frame's `profile` selects
any profile that gateway serves (a profile it does not serve is still `profile_mismatch`). So one
connection is enough for every profile.

It takes no `profile`: a request that carries the key is refused with `invalid_request`. Like every
ticket it is single-use (it opens one socket), expires `expires_in` seconds after it was minted,
and is bound to that gateway instance. The HTTP API answers `401` to it. Request a new one for
every dial, including a reconnect.

**Using an HTTP ticket.** Send it on one HTTP request to the forwarded `api_origin`, in the
`X-Hermes-Gateway-Ticket` header, with no `Origin` header. A ticket is single-use: it
authenticates exactly one request, expires `expires_in` seconds after it was minted, and is bound
to one profile and to that gateway instance. Request a fresh ticket for every HTTP request. The
ticket `attach` returns is for the WebSocket only; the HTTP API answers `401` to it.

Tickets are credentials. The command writes them to its stdout only, never to stderr or a log;
a client must not log them either.

## Client rules

1. **Probe first.** Run `hermes gateway tickets --probe --json` (one SSH exec). If it fails or
   does not print JSON, use your previous SSH path unchanged. Once the probe has succeeded, never
   fall back: show a gateway refusal (authorization, draining, another session, profile) to the user.
2. **A Windows service session only attaches.** If the probe reports
   `"windows_service_session": true`, run `attach --json` only, never `--start`. If the gateway
   is absent, tell the user to start Hermes from a normal (non-administrator) window on that
   machine's desktop. Do not suggest `hermes gateway install`: it can start the gateway elevated,
   which that user's own windows are then refused by. A host that does not report the key is
   treated as `false`.
3. **Otherwise run `attach --start --json` on every connect and reconnect.** A gateway started
   this way exits after about 10 idle minutes. Allow a long first timeout: a first run after an
   update can take most of a minute.
4. **Forward the returned loopback port** over SSH and dial the WebSocket, as in
   [`hermes gateway attach`](./gateway-internals.md#hermes-gateway-attach). When the probe's
   `purposes` list `interactive`, dial with an `interactive` ticket from your `--serve` channel
   ([The WebSocket ticket](#the-websocket-ticket)), so the one socket serves every profile;
   otherwise dial with the ticket `attach` returned, which serves the profile `attach` ran in.
5. **Keep one `hermes gateway tickets --serve --json` channel open** and, for HTTP, request one
   ticket per request, with the purpose and profile [that request needs](#profiles). A new SSH exec per ticket costs seconds; a request on an
   open channel does not.
6. **One path per connection.** Never mix the gateway path with your previous one in the same
   connection.
