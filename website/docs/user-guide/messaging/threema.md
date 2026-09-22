# Threema

[Threema](https://threema.ch/) is a Swiss end-to-end encrypted messenger that needs no phone
number and no email address. Hermes talks to it through the [Threema
Gateway](https://gateway.threema.ch/), in **end-to-end mode** — the only mode that can receive
messages at all.

> Run `hermes gateway setup` and pick **Threema** for a guided walk-through.

## How it differs from every other platform

Threema is the one platform where the connection runs *inward*: Hermes does not hold a socket
open to Threema, Threema POSTs each incoming message to a callback URL you host.

That has three consequences worth knowing before you start:

1. **You need a public HTTPS endpoint with a publicly trusted certificate.** Threema refuses
   self-signed certificates, and there is no way around it from the Hermes side. A reverse proxy
   with Let's Encrypt, a Cloudflare Tunnel, or `ngrok` for development all work.
2. **The callback URL is set by hand** in the Gateway administration panel. Threema publishes no
   API for it.
3. **Messages cost money.** Roughly CHF 0.01 per message, prepaid. One credit per message sent,
   plus one more per file upload — so an image costs two. Hermes splits a long answer into
   3500-byte parts, and *each part is a credit*: keep the agent's replies short.

## Prerequisites

- A **Threema Gateway ID in end-to-end mode** (8 characters, starts with `*`). Register at
  [gateway.threema.ch](https://gateway.threema.ch/). Basic mode cannot receive messages; if your
  ID was set up in basic mode, ask Threema support to switch it.
- The **private key** the panel shows you when the ID is created. It is shown once — save it.
- The **API secret** from the same panel.
- A **public HTTPS URL** that reaches this machine.
- **PyNaCl** for the encryption. Hermes installs it on demand; `pip install pynacl` also works.
  The wheels bundle libsodium, so there is no system package to install.

## Configure

Put the credentials in `~/.hermes/.env`:

```bash
THREEMA_GATEWAY_ID=*MYBOT
THREEMA_API_SECRET=your-api-secret
THREEMA_PRIVATE_KEY_PATH=~/.hermes/threema_private.key
THREEMA_PUBLIC_URL=https://hermes.example.com
THREEMA_ALLOWED_USERS=ECHOECHO,ABCD1234
```

The private key file holds the single line the panel gave you:

```
private:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`chmod 600` it. Hermes reads the key from the file rather than the environment so it does not sit
in the process environment of every child process.

Or in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    threema:
      enabled: true
      extra:
        gateway_id: "*MYBOT"
        private_key_path: "~/.hermes/threema_private.key"
        public_url: "https://hermes.example.com"
        callback_port: 8647
        callback_path: "/threema/callback"
```

The API secret stays in `.env` — `config.yaml` is for settings, not secrets.

## Point Threema at your callback

In the Gateway panel, set the callback URL to:

```
https://<your-public-host>/threema/callback
```

A minimal nginx front:

```nginx
location /threema/callback {
    proxy_pass http://127.0.0.1:8647;
    proxy_set_header Host $host;
}
```

Start the gateway and check it is listening:

```bash
hermes gateway start
curl -s http://127.0.0.1:8647/health
# {"status": "ok", "platform": "threema", "identity": "*MYBOT"}
```

Then message your Gateway ID from the Threema app. Hermes verifies the callback's MAC, decrypts
the box with your private key, and answers.

## What works

| Direction | Supported |
|-----------|-----------|
| Inbound | text, files, images, audio, video, location |
| Outbound | text (auto-split at 3500 bytes), images, files, audio, video |
| Cron delivery | yes — `deliver=threema` works even when the job runs detached from the gateway |

Not available, because the Gateway API has no such concept: threads, reactions, message editing,
typing indicators, and group messages (Threema's group support is provisional and needs the
group's control-message state machine).

Delivery receipts that arrive from the recipient are recognised and dropped — they are
checkmarks, not user turns, and would otherwise wake the agent for every read confirmation.
Hermes does not send receipts back, because each one would cost a credit.

## Troubleshooting

**Nothing arrives.** Almost always the callback URL. Check the certificate is publicly trusted
(`curl https://<host>/threema/callback` from another machine — a certificate warning means
Threema will not deliver either), and that the URL saved in the panel matches the path the
gateway logs on startup. Threema retries a failed callback three times at five-minute intervals
and then discards the message.

**`Callback MAC verification failed`.** `THREEMA_API_SECRET` does not match the ID in the panel.
The MAC is computed over the raw POST fields with that secret, so a stale secret fails every
message.

**`HTTP 402` when sending.** The account is out of credits. `/credits` is checked at startup and
the count is logged.

**`Could not decrypt message`.** The private key does not belong to this Gateway ID. Hermes acks
such a callback anyway — retrying a box this key can never open just repeats the failure four
times.

**A long answer arrives in pieces.** Expected: Threema caps a message at 3500 *bytes*, and emoji
cost four bytes each. Each part is billed separately.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `THREEMA_GATEWAY_ID` | Gateway ID in end-to-end mode (`*MYBOT`) |
| `THREEMA_API_SECRET` | API secret from the Gateway panel |
| `THREEMA_PRIVATE_KEY_PATH` | File holding `private:<hex>` (preferred) |
| `THREEMA_PRIVATE_KEY` | The key inline, if you cannot use a file |
| `THREEMA_PUBLIC_URL` | Public HTTPS base URL, used to warn about a non-TLS callback |
| `THREEMA_CALLBACK_PORT` | Local listener port (default 8647) |
| `THREEMA_CALLBACK_PATH` | Callback path (default `/threema/callback`) |
| `THREEMA_HOME_CHANNEL` | Threema ID for cron and notification delivery |
| `THREEMA_ALLOWED_USERS` | Comma-separated Threema IDs allowed to talk to the bot |
| `THREEMA_ALLOW_ALL_USERS` | Allow anyone — every inbound message costs credits |
