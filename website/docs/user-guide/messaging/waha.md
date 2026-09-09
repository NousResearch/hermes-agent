# WhatsApp via WAHA

Run Hermes on WhatsApp through [WAHA](https://waha.devlike.pro/) (WhatsApp HTTP API) —
a self-hosted gateway with a plain REST API. This is the third WhatsApp transport:
it shares the same gating, mention rules, allow-lists, and markdown formatting as the
[Baileys bridge](whatsapp) and [Cloud API](whatsapp-cloud) adapters, but needs no
Meta business verification and no linked-device subprocess on the Hermes host.

## When to pick WAHA

| | Baileys bridge | Cloud API | **WAHA** |
|---|---|---|---|
| Setup | QR scan, local Node subprocess | Meta business verification | QR scan in the WAHA dashboard |
| Runs where | Hermes host | Meta cloud | Anywhere (own server/container) |
| Multi-account | One per bridge process | One per Meta app | Multiple WAHA sessions |
| Streaming edits | Yes | Yes | Yes (NOWEB/WEBJS/GOWS) |

> [!IMPORTANT]
> Enable **only one** WhatsApp transport per profile (bridge, Cloud API, or WAHA).
> Two enabled transports on the same number deliver every message twice and race
> for the same chats. Pick one per profile; run additional transports in separate
> profiles with separate numbers.

## Prerequisites

1. A running WAHA instance (Docker is the usual way):

   ```bash
   docker run -d --name waha \
     -p 3000:3000 \
     -e WHATSAPP_API_KEY=your-api-key \
     -v waha-sessions:/app/.sessions \
     devlikeapro/waha:noweb
   ```

2. A working session: open the WAHA dashboard (`http://localhost:3000/dashboard`),
   create a session (e.g. `hermes`), scan the QR, and wait for status **WORKING**.

## Step 1: Configure Hermes

Add to `~/.hermes/.env`:

```bash
WAHA_ENABLED=true
WAHA_BASE_URL=http://127.0.0.1:3000     # your WAHA instance
WAHA_API_KEY=your-api-key               # matches WHATSAPP_API_KEY on WAHA
WAHA_SESSION=hermes                     # the session name from the dashboard
```

Optional behavior settings in `~/.hermes/config.yaml`:

```yaml
waha:
  dm_policy: allowlist
  allow_from: ["6281234567890@c.us"]    # numbers that may talk to the bot
  group_policy: allowlist
  group_allow_from: ["120363001234567890@g.us"]
  require_mention: true                 # groups: only respond when addressed
  observe_unmentioned_group_messages: true
  send_read_receipts: false
```

### Managing access from chat: `/access`

Once you are an admin (`allow_admin_from` / `group_allow_admin_from` in the platform
block, same shape as Telegram's), you can manage the lists from the chat itself — no
config edit or restart:

```
/access list                        show both allowlists
/access allow user 0812-3456-789    local, national or +international formats all work
/access allow user @Niken           PushName, resolved against the chat's participants
/access deny  user 6281234567890    remove again
/access allow group "Kelas XI-C"    group name, or the raw 123…@g.us id
/access deny  group "Kelas XI-C"
```

Ambiguous names reply with the matching candidates instead of guessing. Changes take
effect immediately (the running adapter's lists are updated in place) and are persisted
to `config.yaml` — writes go to the top-level block the loader reads, so twin
`platforms.*` entries are folded in rather than left to diverge. Every change is
appended to `access_audit.jsonl` beside the profile config.

Two honest refusals instead of silent dead entries: an unresolvable name/number gets
`Could not resolve …` (nothing is stored — paste the full id), and edits while the
policy is not `allowlist` get `Stored nothing …` (the list is not consulted, so the
edit would change nothing). With no admin list configured at all, `/access` refuses
everything — fail closed, so strangers can never rewrite the allowlists.

## Step 2: Register the WAHA webhook

WAHA pushes inbound messages to Hermes over HTTP. Point the session webhook at the
adapter's receiver (default port `8655`, path `/webhooks/waha`):

```bash
curl -X PUT "$WAHA_BASE_URL/api/sessions/hermes" \
  -H "X-Api-Key: your-api-key" -H "Content-Type: application/json" \
  -d '{
    "config": {
      "webhooks": [{
        "url": "http://<hermes-host>:8655/webhooks/waha",
        "events": ["message"],
        "customHeaders": [{"name": "X-Hermes-Token", "value": "<shared-secret>"}],
        "retries": {"policy": "constant", "delaySeconds": 2, "attempts": 15}
      }]
    }
  }'
```

- `<hermes-host>`: when WAHA and Hermes share a host, the Docker gateway IP
  (e.g. `10.89.0.1`) or the host LAN IP works.
- `<shared-secret>`: set the same value as `WAHA_WEBHOOK_SECRET` in Hermes' `.env`
  so the receiver can authenticate WAHA's POSTs (timing-safe token compare).
- `events: ["message"]` only — ack/reaction events are acked but unused.

## Step 3: Start the gateway

```bash
hermes gateway
```

`hermes gateway status` shows the WAHA platform once the env vars are set.

## Group behavior

Identical to the other WhatsApp transports: with `require_mention: true` the bot
responds only to `@mentions`, replies to its own messages, slash commands, or
configured mention patterns (keywords). With
`observe_unmentioned_group_messages: true`, unmentioned chatter from allowlisted
groups is stored as observed context (no dispatch, no API cost) so a later mention
sees the full conversation flow. See [WhatsApp groups](whatsapp#group-chats-mentions-and-observed-context).

## Cron delivery

Set `WAHA_HOME_CHANNEL` (a chat JID) in `.env` to route `deliver=waha` cron jobs;
out-of-process cron sends go through the same REST API.

## Troubleshooting

- **`session 'x' is STARTING/SCAN_QR_CODE`** — the WhatsApp session is not
  authenticated; scan the QR in the WAHA dashboard. Hermes refuses to start the
  adapter until the session reports **WORKING**.
- **No inbound messages** — check the webhook registration (`GET /api/sessions/hermes`
  → `config.webhooks`), the receiver port reachability from the WAHA container, and
  `WAHA_WEBHOOK_SECRET` matching the webhook's `X-Hermes-Token` header.
- **Media not arriving** — WAHA must be able to reach Hermes' webhook port, and
  Hermes must be able to reach WAHA's media URLs (same network by default).
