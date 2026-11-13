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
| `hermes gateway tickets --probe --json` | Whether this host has the `tickets` command, and which ticket purposes it mints. |
| `hermes gateway tickets --serve --json` | Single-use HTTP tickets, one per request, over one long-lived SSH channel. |

Both commands work for the profile the command runs in (`HERMES_HOME`, or `hermes -p <name>`).
A non-interactive SSH command often does not have `hermes` on its `PATH`; the client has to
locate the binary.

## `hermes gateway tickets --probe --json`

Starts nothing and contacts no gateway. Exits `0` and prints one JSON line:

```json
{"protocol": 1, "purposes": ["native-http"]}
```

`protocol` is the version of the `--serve` line protocol below. `purposes` lists the ticket
purposes `--serve` mints. Read both and ignore keys you do not know: later versions add keys.

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

`id` is an integer or a string of your choice. A request line may be at most 65536 bytes (64 KiB),
not counting its newline; a longer line is discarded as it is read and gets one `invalid_request`
reply. Each request line gets exactly one reply line, in order, carrying your `id`:

```json
{"id": 7, "ok": true, "ticket": "<ticket>", "expires_in": 30}
{"id": 7, "ok": false, "error": {"code": "ticket_unavailable", "message": "..."}}
```

| `error.code` | Meaning |
|--------------|---------|
| `invalid_request` | The line is not a JSON object with an integer or string `id`, is nested too deeply to parse, or is longer than 65536 bytes (64 KiB, newline excluded). The reply's `id` is `null`. The channel stays open. |
| `unsupported_purpose` | `purpose` is missing or not in the probe's `purposes`. |
| `ticket_unavailable` | The gateway did not mint the ticket: it stopped, was replaced, or refused. Close the channel, run `attach` again and open a new one. |

Close stdin to end the command; it exits `0`.

**Using a ticket.** Send it on one HTTP request to the forwarded `api_origin`, in the
`X-Hermes-Gateway-Ticket` header, with no `Origin` header. A ticket is single-use: it
authenticates exactly one request, expires `expires_in` seconds after it was minted, and is bound
to the profile and to that gateway instance. Request a fresh ticket for every HTTP request. The
ticket `attach` returns is for the WebSocket only; the HTTP API answers `401` to it.

Tickets are credentials. The command writes them to its stdout only, never to stderr or a log;
a client must not log them either.

## Client rules

1. **Probe first.** Run `hermes gateway tickets --probe --json` (one SSH exec). If it fails or
   does not print JSON, use your previous SSH path unchanged. Once the probe has succeeded, never
   fall back: show a gateway refusal (authorization, draining, another session, profile) to the user.
2. **A Windows service session only attaches.** If the probe reports
   `"windows_service_session": true`, run `attach --json` only, never `--start`. If the gateway
   is absent, tell the user to start or install it on that machine. A host that does not report
   the key is treated as `false`.
3. **Otherwise run `attach --start --json` on every connect and reconnect.** A gateway started
   this way exits after about 10 idle minutes. Allow a long first timeout: a first run after an
   update can take most of a minute.
4. **Forward the returned loopback port** over SSH and dial the WebSocket with the `attach`
   ticket, as in [`hermes gateway attach`](./gateway-internals.md#hermes-gateway-attach).
5. **For HTTP, keep one `hermes gateway tickets --serve --json` channel open** and request one
   `native-http` ticket per request. A new SSH exec per ticket costs seconds; a request on an
   open channel does not.
6. **One path per connection.** Never mix the gateway path with your previous one in the same
   connection.
