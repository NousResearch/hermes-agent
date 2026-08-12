# QQ (OneBot)

The OneBot adapter connects Hermes to QQ through the **OneBot 11 protocol**, compatible with [NapCat](https://napneko.github.io/), [Lagrange](https://github.com/LagrangeDev/Lagrange.Core), LLOneBot, and go-cqhttp. Instead of the official QQ Bot platform (which requires a Tencent-approved app), OneBot drives a regular QQ account via a local bridge — useful for personal bots and groups the official platform can't reach.

It supports **private chats and group chats**, mention gating in groups, allowlists, inbound image download (for the vision tool), long-reply splitting at sentence boundaries, and text-image rendering for very long replies.

> Run `hermes gateway setup` and pick **QQ (OneBot)** for a guided walk-through.

## Prerequisites

- A running OneBot-compatible bridge (e.g. NapCat) connected to the QQ account you want to use
- The bridge must be reachable from the Hermes host (typically the same machine, or over your LAN)

## Configure Hermes

Add the platform to the `gateway` block in `~/.hermes/gateway-config.yaml`:

```yaml
gateway:
  platforms:
    onebot:
      enabled: true
      extra:
        mode: reverse              # reverse | forward
        host: "0.0.0.0"            # reverse: address to listen on
        port: 8643                 # reverse: listen port
        # url: "ws://127.0.0.1:3001"   # forward: bridge ws endpoint
        # access_token: ""         # must match the bridge's token, if set
        # bot_qq: ""               # optional; auto-learned from meta events
        require_mention: true      # groups: only reply when @'d
        dm_policy: open            # open | allowlist | disabled
        allow_from: []             # user ids when dm_policy=allowlist
        group_policy: open         # open | allowlist | disabled
        group_allow_from: []       # group ids when group_policy=allowlist
        split_length: 100          # long replies split at this many chars
        text_image_threshold: 150  # longer replies render as a text image
```

### Connection modes

| Mode | Description |
|------|-------------|
| `reverse` (default) | Hermes hosts a WebSocket server; the bridge's **ws-reverse** client dials in (`ws://<hermes-host>:8643/ws`). One connection carries both events and actions. |
| `forward` | Hermes dials the bridge's WebSocket server (`ws://<bridge-host>:3001` for NapCat's default). |

If the bridge uses an access token, set the same value in `access_token` (Hermes sends it as `Authorization: Bearer <token>` on the reverse connection; forward mode includes it in the handshake headers).

## Authorizing users

Like other platforms, inbound users must be authorized before they can talk to the bot:

| Env var | Meaning |
|---------|---------|
| `ONEBOT_ALLOWED_USERS` | Comma-separated QQ user ids allowed to chat (e.g. `841859784`) |
| `ONEBOT_ALLOW_ALL_USERS` | `true` allows anyone (dev only) |
| `GATEWAY_ALLOW_ALL_USERS` | Global allow-all for every platform |

Unauthorized users in DMs receive a pairing code; in groups they are silently ignored.

## Group mentions

With `require_mention: true` (default), the bot only responds in groups when it is explicitly @'d or when the message replies to an existing message. Set it to `false` to respond to every group message (noisy — not recommended for large groups). When no `bot_qq` is configured the bot learns its own id from OneBot meta events, so mention detection works out of the box.

## Long replies

- **≤ `split_length`** (default 100) characters: sent as a single text message.
- **`split_length` – `text_image_threshold`** (default 150): split into multiple messages, breaking at sentence boundaries (`。！？!?；;\n`) so sentences are never cut in half.
- **> `text_image_threshold`**: rendered as a black-on-white text image (720 px wide, CJK-aware font fallback chain) and sent as a single image message. Falls back to text chunks if rendering fails.

Set `text_image_threshold: 0` to disable the image path.

## Markdown & voice

- **Markdown is stripped** before delivery — QQ does not render it, so `**bold**`, headings, lists, and tables are converted to readable plain text (headings → `【…】`, lists → `•`, tables → spaced cells, fenced code blocks → bordered boxes). This runs before splitting and text-image rendering, so images are clean too.
- **Inbound voice messages are transcribed**: the adapter downloads the clip and converts it with `ffmpeg` to 16 kHz mono WAV, then hands it to Hermes' STT pipeline (local whisper / Groq / OpenAI — same as other platforms). Requires `ffmpeg` on the Hermes host; without it, voice clips degrade to a `[语音]` marker.

## Notes

- Outbound messages use the OneBot segment-array format (not CQ-code strings) — required for NapCat's message handling.
- Replies are sent as plain text without quoting the triggering message.
- Inbound images are downloaded to a temp directory and exposed to the vision tool via `media_urls`; undownloadable images degrade to a `[图片]` placeholder.
- QQ faces map to common emoji; unknown faces collapse to `[表情]`.
- Long replies may take a few seconds to render — the gateway shows a typing indicator where supported.
