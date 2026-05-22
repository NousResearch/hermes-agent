# XMPP

[XMPP](https://xmpp.org/) (Extensible Messaging and Presence Protocol) is an open, federated chat protocol. Hermes connects to any XMPP server, hosted or self-hosted, including [Prosody](https://prosody.im/) and [ejabberd](https://www.ejabberd.im/), and relays messages between XMPP contacts or MUC rooms and the agent.

## Prerequisites

- An XMPP account on a server you control or have access to (the bot logs in with its own JID and password)
- Python package **slixmpp** (`pip install slixmpp`)

## Configure Hermes

### Via setup wizard

```bash
hermes setup gateway
```

Select **XMPP** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```
XMPP_JID=hermes@chat.example.org
XMPP_PASSWORD=********
XMPP_ALLOWED_USERS=alice@example.org,bob@example.org
XMPP_ROOMS=ops@conference.example.org
XMPP_HOME_CHANNEL=alice@example.org
```

| Variable | Required | Description |
|---|---|---|
| `XMPP_JID` | Yes | Bare JID the bot logs in as |
| `XMPP_PASSWORD` | Yes | Password for that account |
| `XMPP_HOST` | No | Server host override. Default: SRV lookup on the JID domain |
| `XMPP_PORT` | No | Server port override. Default: `5222` |
| `XMPP_FORCE_STARTTLS` | No | Require STARTTLS. Default: `true` |
| `XMPP_NICKNAME` | No | MUC nickname. Default: the JID's local part |
| `XMPP_ROOMS` | No | Comma-separated MUC JIDs to auto-join |
| `XMPP_ALLOWED_USERS` | Recommended | Comma-separated bare JIDs allowed to DM the bot |
| `XMPP_ALLOW_ALL_USERS` | No | Set `true` to disable the allowlist (dev only) |
| `XMPP_HOME_CHANNEL` | No | Default JID for cron delivery (prefix MUC targets with `muc:`) |
| `XMPP_HOME_CHANNEL_NAME` | No | Human label for the home channel |
| `XMPP_UPLOAD_SERVICE` | No | Pin the HTTP Upload component JID. Default: auto-discover |
| `XMPP_HTML_FORMATTING` | No | Emit XHTML-IM dual body for rich rendering. Default: `true` |

## Run a local Prosody for testing

[Prosody](https://prosody.im/) is the simplest self-hosted server to try the integration against.

```bash
docker run -d --name prosody \
  -p 5222:5222 -p 5269:5269 -p 5280:5280 -p 5281:5281 \
  -e LOCAL=hermes \
  -e DOMAIN=localhost \
  -e PASSWORD=hermes-dev-only \
  prosody/prosody
```

Then set:

```
XMPP_JID=hermes@localhost
XMPP_PASSWORD=hermes-dev-only
XMPP_HOST=127.0.0.1
XMPP_FORCE_STARTTLS=false
```

`XMPP_FORCE_STARTTLS=false` is acceptable on `localhost` for development. Leave it `true` for any non-loopback deployment.

## Authorization

By default **all JIDs are denied** — set `XMPP_ALLOWED_USERS` to the comma-separated bare JIDs that should be able to talk to the bot. Resources (the `/laptop` part of `alice@example.org/laptop`) are stripped before the allowlist check.

For MUC rooms, the bot only replies when addressed by its nickname (e.g. `hermes: status?`). Unaddressed room chatter is ignored.

## Cron delivery

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="xmpp",          # uses XMPP_HOME_CHANNEL
    prompt="Summarise overnight alerts."
)
```

Target a specific JID or MUC room directly:

```python
send_message(target="xmpp:alice@example.org", message="Done!")
send_message(target="xmpp:muc:ops@conference.example.org", message="Deploy finished.")
```

## Media attachments (HTTP Upload, XEP-0363)

When the agent emits `MEDIA:/absolute/path/to/file`, the adapter uploads the file to the server's HTTP Upload component (auto-discovered on connect, or pinned via `XMPP_UPLOAD_SERVICE`) and delivers the resulting HTTPS URL as a stanza with an XEP-0066 OOB extension. XMPP clients that understand OOB (Conversations, Dino, Gajim, Profanity) render the file inline; clients that don't still see the URL in the message body and can tap through.

Requirements:

- The server needs an HTTP Upload component. Prosody enables it with `Component "uploads.example.org" "http_file_share"` in `prosody.cfg.lua`. ejabberd ships `mod_http_upload`.
- The file must be reachable to the bot's local filesystem at the absolute path the agent emitted. The adapter rejects relative paths and missing files before contacting the server.
- The file must fit under the server's advertised upload size limit. Larger files arrive as a plain-text "📎 filename (upload failed: …)" line, with the reason logged in `gateway.log`.

Disable file uploads entirely with `XMPP_UPLOAD_SERVICE=disabled` (any unreachable JID also works) — the adapter then falls back to text-only attachments for every media send.

## Rich text (XHTML-IM, XEP-0071)

Markdown in the agent's response is sent in two parallel forms:

- `<body>` — markdown stripped to plain text, for clients that read only the body.
- `<html>` — an XHTML-IM fragment with `<strong>`, `<em>`, `<code>`, `<pre>`, `<a>`, `<ul>`, `<ol>`, and `<blockquote>`, for clients that opt in to XEP-0071.

Dual-body emission only happens for messages that fit in a single stanza; chunked responses fall back to plain text because splitting XHTML across stanzas produces invalid fragments most clients reject. Set `XMPP_HTML_FORMATTING=false` to opt out entirely (some MUC components strip the `<html>` element and dual-body looks redundant in that case).

## Limitations

- **OMEMO / OpenPGP encryption is not handled by the adapter.** Use a server you trust and STARTTLS for transport security; end-to-end encryption requires a client that speaks OMEMO and is out of scope here.
- **Image preview for HTTP URLs:** the adapter sets the OOB extension for `https://` URLs the agent emits inline via markdown (`![alt](url)`); plain `http://` URLs are *not* OOB-flagged to keep tappable links from auto-rendering as attachments on hostile pages.

## Troubleshooting

**`'slixmpp' package not installed`** — Run `pip install slixmpp`.

**Authentication failed** — Verify the JID and password by logging into the same account from any XMPP client (e.g. Gajim, Conversations, Dino).

**Connect timed out** — Check `XMPP_HOST` and `XMPP_PORT`. If your server requires a non-default port (Snikket on 5223, etc.), set both. If your server has no SRV record, set `XMPP_HOST` explicitly.

**Bot ignores room messages** — The bot only replies when addressed by its nickname. Send `hermes: hello` (replace `hermes` with `XMPP_NICKNAME` if you overrode it).
