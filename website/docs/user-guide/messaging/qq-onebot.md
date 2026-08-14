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
        image_max_size: 2048       # downscale inbound images (0 = keep as-is)
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
| `ONEBOT_ALLOWED_USERS` | Comma-separated QQ user ids allowed to chat (e.g. `123456789`) |
| `ONEBOT_ALLOW_ALL_USERS` | `true` allows anyone (dev only) |
| `GATEWAY_ALLOW_ALL_USERS` | Global allow-all for every platform |

Non-admin users are silently rejected in DMs (no pairing flow); in groups, unauthorized senders are ignored.

## Group mentions

With `require_mention: true` (default), the bot only responds in groups when it is explicitly @'d or when the message replies to an existing message. Set it to `false` to respond to every group message (noisy — not recommended for large groups). When no `bot_qq` is configured the bot learns its own id from OneBot meta events, so mention detection works out of the box.

## Long replies

- **≤ `split_length`** (default 100) characters: sent as a single text message.
- **`split_length` – `text_image_threshold`** (default 150): split into multiple messages, breaking at sentence boundaries (`。！？!?；;\n`) so sentences are never cut in half.
- **> `text_image_threshold`**: rendered as a black-on-white text image (800 px wide, CJK-aware font fallback chain) and sent as a single image message. Falls back to text chunks if rendering fails.

Set `text_image_threshold: 0` to disable the image path.

The text-image renderer is an AstrBot-style **element-based Markdown renderer**: bold / italic / strikethrough / inline code / code blocks / headers / quotes / lists and **tables** (AstrBot itself has no table element) are all drawn natively. Chinese typography rules are honored — punctuation never starts a line (行首禁则), inline styles wrap as a whole line, literal `\n` in plain text becomes a real line break (inside inline code it becomes a space; `\\n` is kept), and inline code uses a light-blue pill with a monospace font for Latin/digits and glyph-level fallback for CJK. When the reply target's nickname is known, the card gets an AstrBot-style blue top bar (`To <nickname>`, Klein blue #002FA7, white text at **twice the body font size**, ~68 px tall), matching AstrBot's card header proportions.

## Markdown & voice

- **Markdown is stripped** before delivery — QQ does not render it, so `**bold**`, headings, lists, and tables are converted to readable plain text (headings → `【…】`, lists → `•`, tables → spaced cells, fenced code blocks → bordered boxes). This runs before splitting and text-image rendering, so images are clean too.
- **Inbound voice messages are transcribed**: the adapter downloads the clip and converts it with `ffmpeg` to 16 kHz mono WAV, then hands it to Hermes' STT pipeline (local whisper / Groq / OpenAI — same as other platforms). Requires `ffmpeg` on the Hermes host; without it, voice clips degrade to a `[语音]` marker.

## Images

- Inbound messages are parsed as the **OneBot segment array** (`message` field) when available — image/voice/at/face/video/file segments are handled structurally; text-format clients fall back to CQ-code string parsing. CQ entity escaping (`&amp;` → `&`, `&#91;` → `[`, …) is reversed before any URL is used, so CDN links with `&` parameters download correctly.
- Images are downloaded to a temp directory and exposed to the vision tool via `media_urls`; undownloadable images degrade to a `[图片]` placeholder. If the image segment only carries a `file` hash (no URL), the adapter calls the OneBot `get_image` action to resolve the real URL; `base64://` and `file://` forms are handled directly.
- Images larger than `image_max_size` (default **2048** px on the long edge) are downscaled with Pillow before the LLM sees them — high-resolution QQ photos otherwise make vision calls slow or time out. RGBA stays PNG, everything else becomes JPEG (q85); animated GIFs collapse to their first frame. Set `image_max_size: 0` to keep originals untouched.

## Outbound media

The adapter implements the gateway's native media senders as OneBot segments, so the agent can deliver rich media through the standard `MEDIA:` / markdown-image mechanism:

| Capability | OneBot segment | Notes |
|---|---|---|
| Image URL (direct) | `image` with `url` | Markdown image URLs from the agent are sent as-is; the bridge downloads them (no local file needed). |
| Local image | `image` with `base64://` | Up to 8 MB. |
| Batch images | multiple `image` segments | One message, max 9 images per message; URL + local mixed. |
| Voice | `record` with `base64://` | Up to 20 MB; the bridge transcodes to silk. |
| Video | `video` with `base64://` | Up to 20 MB. |
| File | `file` with `base64://` + `name` | Up to 20 MB. |
| Forwarded messages | `send_forward_msg` with `node`s | Group chats only. |

**Merged forwarding** is triggered by an agent-side block:

```
[[qq_forward]]
<display name>
<message text>
---
<display name>
<message text>
[[/qq_forward]]
```

Each `---`-separated block becomes one forwarded node (name + text, 500-char cap per node). In private chats the marker is ignored and the block degrades to plain text.

## Replying to a message (quote)

When the user replies to (quotes) a previous message, the adapter calls the OneBot `get_msg` API to fetch the original message and:

- prefixes the original text with `[引用]` so the agent sees what was quoted
- attaches any image / voice / video from the original message as media (voice goes through the STT pipeline, video is downloaded for frame extraction)

This works for both segment-array and CQ-string payloads. If `get_msg` fails the current message is delivered unchanged.

## Loop-message merge (interim commentary folding)

During a multi-tool turn the gateway sends interim commentary messages ("Using tool X…") followed by the final response. To save chat space the adapter buffers interim text messages per chat and, when the final message arrives, merges them into a single QQ forwarded-message and retracts the originals:

- group chats use `send_forward_msg`, private chats use `send_private_forward_msg`
- merge happens only when ≥2 interim messages are buffered
- originals are retracted (`delete_msg`) only after the forward succeeds; on failure they are kept
- the buffer is cleared on any new inbound user message

This relies on the gateway marking commentary sends with `interim: True` in the stream-consumer metadata (see `gateway/stream_consumer.py`).

## Hot reload

`onebot_utils.py` (pure helpers: CQ parsing, splitting, markdown stripping, emoji map) and `t2i_render.py` (text-image renderer) are hot-reloaded on every use: the adapter stats the file mtime and calls `importlib.reload` when it changed, so style/rule tweaks apply without a gateway restart. Changes to `adapter.py` itself still require a restart.

## Access tiers (admin / restricted member)

Group chats can be opened to all members of allowlisted groups while keeping
privileged operations admin-only. The adapter enforces its own access policy
(`enforces_own_access_policy`), so the gateway trusts its allowlist decisions.

| Role | Who | Group @ | DM | Capabilities |
|---|---|---|---|---|
| admin | `extra.admin_users` (falls back to `ONEBOT_ALLOWED_USERS`) | full | allowed | everything incl. slash commands |
| member | any other user in an allowlisted group | restricted | rejected | quick Q&A, image analysis, group summaries only |
| unauthorized | outside allowlisted groups / DM allowlist | blocked | rejected (pairing disabled) | — |

Configuration (`platforms.onebot.extra`):

```yaml
dm_policy: allowlist          # DM only for allow_from (pairing entry closed)
allow_from: [<admin_qq>]
group_policy: allowlist       # open groups by id
group_allow_from: [<group_id>]
admin_users: [<admin_qq>]     # optional; falls back to ONEBOT_ALLOWED_USERS
```

Enforcement points:
- member group messages get a `[受限用户:仅问答]` text prefix so the agent
  applies the soft restriction (quick Q&A only, no file/terminal/config/HA/
  cross-platform/cron actions — declared in the platform hint)
- member slash commands (`/new`, `/model`, `/help`, …) are dropped before a
  `MessageEvent` is constructed; path-like text (`/tmp/x`) is not affected
- member DMs are rejected (no pairing flow)
- outbound replies to member chats are scanned against sensitive-intent
  keywords and logged with a WARNING (audit, not hard blocking)

## Notes

- Outbound messages use the OneBot segment-array format (not CQ-code strings) — required for NapCat's message handling.
- Replies are sent as plain text without quoting the triggering message.
- QQ faces map to common emoji; unknown faces collapse to `[表情]`. Voice without a downloadable link degrades to `[语音]`; inbound video/file segments degrade to `[视频]` / `[文件:name]` placeholders.
- In **private chats** the bot shows QQ's native "typing…" bubble while the agent generates (via NapCat's `set_input_status` extension). Group chats have no typing indicator on QQ.
- Long replies may take a few seconds to render — the gateway shows a typing indicator where supported.
- Cron / scheduled deliveries cannot attach media to OneBot yet (the core `send_message_tool` media whitelist covers telegram, discord, matrix, weixin, signal, yuanbao, feishu, whatsapp and slack only) — interactive replies are unaffected.
