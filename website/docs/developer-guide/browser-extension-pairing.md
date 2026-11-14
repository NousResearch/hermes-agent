# Hermes Browser Extension Pairing

The gateway exposes a loopback-only pairing flow so the Hermes Browser
extension (or any local extension) can obtain a **scoped bearer token** with
one approval click. The token is scoped to the default profile and grants
access only to local chat, model-list, and browser-context requests — it never
exposes the raw `API_SERVER_KEY`.

## Flow

```
Extension                    Gateway (127.0.0.1)
    |  POST /api/browser-extension/pair/start      |
    |---------------------------------------------->|  mint short-lived pairing (180s)
    |  {pairing_id, approval_url, ttl_seconds}      |
    |<----------------------------------------------|
    |  user opens approval_url (approve page)       |
    |                                              |
    |  POST /pair/grant/{id}   (Approve button)     |  status -> approved, mint token
    |  POST /pair/deny/{id}    (Deny button)        |  status -> denied
    |                                              |
    |  GET  /pair/status/{id}  (poll until terminal)|
    |<----------------------------------------------|  {status, token} once approved
    |  use token as Bearer for /v1/* requests       |
```

## Endpoints

All pairing endpoints are **loopback-only** (403 for any non-loopback peer):

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/browser-extension/pair/start` | Mint a pending pairing (180s TTL) |
| GET | `/api/browser-extension/pair/approve/{id}` | Branded approval page (Approve/Deny buttons) |
| POST | `/api/browser-extension/pair/grant/{id}` | Approve; marks the pairing approved and mints the token |
| POST | `/api/browser-extension/pair/deny/{id}` | Reject the pairing |
| GET | `/api/browser-extension/pair/status/{id}` | Poll: `pending` → `approved` (with token) or 410 `denied`/`expired` |

## Approving inside the extension panel

`pair/start` accepts `extensionOrigin`, the exact origin of the extension that
is asking (`chrome-extension://<id>`, `moz-extension://<uuid>` or
`safari-web-extension://<uuid>`; a Firefox install gets a random one, so it
cannot be guessed). When the value is that and nothing else (no wildcard, port,
path or whitespace) and, if the request carries an `Origin` header, matches it,
the pairing is *embeddable* and the response says `"embeddable": true`.

For an embeddable pairing only, the approval page and its Approved and Denied
results are sent with `frame-ancestors <that origin>` and without
`X-Frame-Options`, so that extension can show them in an iframe in its own
panel. Every other page, and every pairing that does not name an origin, keeps
`frame-ancestors 'none'` and `X-Frame-Options: DENY`. A gateway that predates
this omits `embeddable`, which tells the extension to open the approval page in
a tab as before.

A framed page is only a few hundred pixels wide, so the extension asks for the
compact layout with `?embed=1`: a slim header band, the prompt and the two
buttons, nothing to scroll. The flag is honoured only for an embeddable
pairing, and the Approve and Deny forms carry it so the result pages stay
compact too. Every pairing page scrolls with the portal's own scrollbar.

## Who may press Approve and Deny

Grant and deny are plain form posts, so they check where the post came from:
the browser-set `Origin` must be the gateway's own loopback address (and
`Sec-Fetch-Site`, when sent, `same-origin`). A missing or `null` origin, another
extension's or page's origin (even one on `API_SERVER_CORS_ORIGINS`), or a
hostname rebound to 127.0.0.1 gets a 403 and leaves the pairing pending. The
approval page therefore sends `Referrer-Policy: same-origin`; the default
`no-referrer` would make browsers post `Origin: null`.

This stops an extension or web page approving its own pairing with a fetch. It
does not stop an extension that has host access to the gateway from scripting
the page it shows, so the page still names the extension asking.

`/v1/capabilities` advertises `browserPairing: true` and the
`browser_pair_*` endpoints so fresh browser installs can discover pairing
before they hold any token.

## Opening the Webapp without a terminal

A gateway that serves the Webapp (`webapp.enabled: true`, see `gateway/webapp_listener.py`) signs a browser
in with a credential that only a one-use ticket can hand over, and `hermes webapp` was the only way to get
one. Two more loopback routes let the paired extension ask for those tickets itself. Both answer a `POST`,
only to a loopback peer (403 otherwise), and need `Authorization: Bearer <paired token or API key>` (401
otherwise, even when the server has no API key). Every reply is `Cache-Control: no-store`.

| Method | Path | Answers |
| --- | --- | --- |
| POST | `/api/browser-extension/webapp-ticket?profile=<name>` | `{ticket, webapp_url, expires_in_seconds}`: a one-use, non-launch Webapp window ticket and the Webapp's own origin |
| POST | `/api/browser-extension/webapp-socket-ticket?profile=<name>` | `{ticket, ws_url, ttl_seconds}`: a one-use ticket for the Webapp's `/api/ws` gateway socket |

The extension opens `<webapp_url>/webapp/launch#ticket=<ticket>&query=profile%3D<name>` in a tab; the launch
page spends the ticket from the fragment, keeps the browser credential in that tab's `sessionStorage` and
enters the Webapp. For the socket, it dials `ws_url` offering `hermes-gateway-v1` and
`hermes-gateway-ticket.<ticket>` as subprotocols, which gives it the session-authority socket (Projects,
context compaction, session image attach, the rich roster) without ever holding the long-lived credential.
`profile` is optional; with one, the socket ticket answers for that profile, as the Webapp's own
`/api/webapp/socket-ticket` does.

The tickets are minted in the gateway process, through the Webapp's own stores and under the browser
credential in use (`gateway/browser_webapp_tickets.py`), so they expire in seconds (30 s for the window
ticket and the socket ticket) and `hermes webapp --sign-out` voids them and closes the sockets they opened.

Refusals carry `error.code`:

| Status | `error.code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_profile` | `profile` is not a profile name |
| 403 | `webapp_ticket_loopback_only` | The peer is not on the loopback interface |
| 404 | `webapp_off` | `webapp.enabled` is not true |
| 404 | `webapp_unavailable` | The gateway has no Webapp, it could not start, or its bundle is not built (the message says why) |
| 404 | `webapp_not_private` | The Webapp is behind sign-in or served at a public address; no ticket opens it |
| 404 | `profile_not_found` | A socket ticket was asked for a profile that does not exist |
| 429 | `webapp_tickets_exhausted` | Too many window tickets are pending |
| 503 | `webapp_starting` | The Webapp is still starting; try again |

A gateway that predates these routes answers a plain 404 with no `error.code`, which tells the extension to
fall back to the credential a signed-in tab holds.

**What a paired token can now reach.** The Webapp's session grants host files and a shell. A paired
extension was scoped to chat, model-list and browser-context requests; with these routes it can open a
signed-in Webapp. Approve a pairing only for an extension you trust, and note that the routes exist only
while the Webapp is on and private (a loopback bind with no sign-in gate and no public URL).

## Tokens

- Minted by `gateway/browser_pairing.py` (`BrowserPairingStore`) at grant time.
- Persisted under `HERMES_HOME/state/browser_pairing.json`, so pairings and
  tokens survive gateway restarts.
- Accepted by the API-server auth check **only for the default profile** — a
  named profile can never be reached with a token minted against the listener
  key.
- Max token age is 365 days; tokens are not revocable per-session (the
  browser clears them locally when it disconnects).

## Extension contract

The approval page keeps a stable DOM contract for automated tests:

- `#approveButton` submits `POST /api/browser-extension/pair/grant/{id}`
- `#denyButton` submits `POST /api/browser-extension/pair/deny/{id}`

The page ships in the gateway wheel under `gateway/assets/browser_pairing/`
(fonts + artwork loaded as data URIs at request time). If the assets are
missing the page degrades gracefully to fallback fonts with no artwork.
