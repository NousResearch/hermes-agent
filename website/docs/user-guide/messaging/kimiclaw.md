---
sidebar_position: 18
title: "KimiClaw"
description: "Set up Hermes Agent as a KimiClaw bot on Kimi (Moonshot AI's consumer chat platform)"
---

# KimiClaw Setup

Run Hermes Agent as a [KimiClaw](https://www.kimi.com/) bot through the bundled `kimiclaw` platform plugin. KimiClaw is Moonshot AI's agentic bot system on kimi.com — distinct from the `kimi` LLM provider (which is the Moonshot inference API, configured under `providers:` not `platforms:`).

The adapter lives under `plugins/platforms/kimiclaw/` and is discovered by the platform-plugin loader, so there is no `Platform.KIMICLAW` enum edit or gateway factory edit.

KimiClaw bridges direct messages and group rooms under one bot identity. Direct messages use the Zed ACP path over a persistent WebSocket with sentinel session `im:kimi:main`; group rooms use the Connect RPC `Subscribe` server-stream over an HTTP/1.1 long-poll with `room:<uuid>` chat IDs.

## How the bot responds

| Context | Behavior |
|---------|----------|
| **Direct message** (`im:kimi:main`) | Uses the Zed ACP channel and responds as the bot user. |
| **Group room** (`room:<uuid>`) | Uses the Connect RPC channel and follows the gateway mention / free-response policy. |
| **Cron or notification delivery** | Routes through `KIMI_HOME_CHANNEL` when set. |
| **Allowlisted users** (`km_u_*`) | `KIMI_ALLOWED_USERS` restricts who can talk to the bot unless `KIMI_ALLOW_ALL_USERS=true`. |

Kimi user IDs use the `km_u_*` shape. Bot tokens use the `km_b_prod_*` shape. (Env-var names stay `KIMI_*` because the credentials are issued by Kimi.com itself.)

---

## Step 1: Create a KimiClaw bot and copy the token

1. Open [kimi.com](https://www.kimi.com/) and sign in.
2. Navigate **Kimi Claw → Add Claw → Link existing OpenClaw**. Kimi displays a one-shot install command of the form:

   ```bash
   bash <(curl -fsSL https://cdn.kimi.com/kimi-claw/claw-install.sh) --bot-token km_b_prod_<your_token>
   ```

3. **Copy the `km_b_prod_<your_token>` value.** That is the bot credential Hermes Agent needs. You do **not** need to run the install command itself — Hermes Agent connects to the KimiClaw wire protocol directly without a local OpenClaw runtime (see [Deployment model](#deployment-model) below). Keep the token private; Hermes sends it as `X-Kimi-Bot-Token` when connecting.
4. After clicking **Done** in the Kimi UI to bind your bot identity to the token, you can proceed to Step 2.

### Deployment model

KimiClaw's official install flow expects you to run `claw-install.sh` to set up a local OpenClaw runtime that owns the wire connection. The Hermes adapter takes a different path: it speaks the KimiClaw wire protocol (DM ACP WebSocket and Connect-RPC long-poll) directly from Python, with no local OpenClaw process. The adapter does send the OpenClaw-shaped runtime-metadata headers that kimi.com gates group-room participation on — declaring the minimum protocol version (`X-Kimi-OpenClaw-Version: 2026.3.13`, matching kimi.com's documented "OpenClaw 3.13 or above" gate). Bot identity is attributed honestly: the `X-Kimi-Claw-ID` header (`hermes-kimi-<hex>`) is set on both the DM WebSocket upgrade and the group HTTP path, and the group HTTP path additionally sets `User-Agent: hermes-kimi-adapter/1.0` (the WebSocket upgrade does not carry a `User-Agent` header). All five OpenClaw headers are overridable via `config.extra` (`claw_version`, `openclaw_version`, `claw_id`, `openclaw_plugins`, `openclaw_skills`).

---

## Step 2: Configure Hermes

Add to `~/.hermes/.env`:

```env
KIMI_BOT_TOKEN=km_b_prod_<your_token>

# Optional allowlist. Use Kimi user IDs, comma-separated.
KIMI_ALLOWED_USERS=km_u_<uuid>

# Development only: bypass the allowlist.
KIMI_ALLOW_ALL_USERS=false

# Optional default delivery target for cron / notifications.
# Must be a room:<uuid> — see "Known limitations" for why DM cron
# is not supported through the standalone send path.
KIMI_HOME_CHANNEL=room:<uuid>
```

Then enable the platform in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    kimiclaw:
      enabled: true
```

For profile-local configuration without a per-profile `.env`, the adapter also reads a top-level `kimiclaw:` block and maps it to the same `KIMI_*` environment values:

```yaml
kimiclaw:
  bot_token: km_b_prod_<your_token>
  home_channel: room:<uuid>
  allowed_users:
    - km_u_<uuid>
  allow_all_users: false
```

`platforms.kimiclaw.extra` still works for raw adapter extras, but the top-level `kimiclaw:` block is the bridge that expands into the standard env-var path.

---

## Step 3: Run the gateway

```bash
hermes gateway start
```

On startup, the plugin loader discovers `plugins/platforms/kimiclaw/` and registers the `kimiclaw` platform. A healthy connection logs that KimiClaw connected as the bot identity.

Send a message to the bot from Kimi. For group rooms, use the `room:<uuid>` value from the gateway inbound-message log when you need to set `KIMI_HOME_CHANNEL` or room-specific policy.

---

## Known limitations

**`output_mode: tool_only` and `send_message_tool`.** Against current upstream Hermes, `send_message_tool._send_via_adapter` prefers a live in-process adapter over the standalone-sender fallback. In `tool_only` mode the adapter suppresses all `send()` calls (it cannot yet distinguish "streaming prose" from "explicit tool send" at the gate), so explicit tool sends from an in-process gateway are silently dropped alongside the prose. Stay on the default `passthrough` for interactive DMs; `tool_only` is safe only for cron-driven or out-of-process deployments where dispatch falls through to `standalone_sender_fn` (which is not gated by `output_mode`).

**Cron home channel is group-rooms-only.** `KIMI_HOME_CHANNEL` accepts both `room:<uuid>` and `im:kimi:main` syntactically, but the standalone sender used by cron currently only delivers to `room:<uuid>` — DM cron delivery requires an active WebSocket session and falls through to the live adapter path. Configure `KIMI_HOME_CHANNEL` to a `room:<uuid>` for predictable cron delivery.

**`validate_config` token sources.** `validate_config(config)` returns true when any of `KIMI_BOT_TOKEN` (env), `PlatformConfig.token` (`platforms.kimiclaw.token` YAML), or `PlatformConfig.extra["bot_token"]` resolves to a non-empty string — mirroring the three sources the adapter `__init__` consults. The `apply_yaml_config_fn` hook also bridges top-level `kimiclaw.bot_token` YAML into `extra`, so both the top-level and `platforms.kimiclaw.*` YAML layouts are honored.

---

## Troubleshooting

**`KIMI_BOT_TOKEN` is missing or rejected.** Regenerate the token in Kimi and verify it starts with `km_b_prod_`. If you put the token in YAML, prefer the top-level `kimiclaw.bot_token` bridge or a real `KIMI_BOT_TOKEN` env var.

**Cron delivery does not reach Kimi.** Set `KIMI_HOME_CHANNEL` to the target `room:<uuid>` (cron uses the standalone send path, which currently only supports group rooms — see [Known limitations](#known-limitations) for the DM caveat). If the bot token was regenerated, the old room can become stale because the bot identity changed; run `/sethome` again from the live target chat or update the env var manually.

**The bot is silent for a user.** Add that user ID to `KIMI_ALLOWED_USERS`, or set `KIMI_ALLOW_ALL_USERS=true` only for a trusted development setup. User IDs have the `km_u_*` shape.

**Confused with the `kimi` LLM provider?** They're separate integrations. The Moonshot `kimi-for-coding` LLM provider (profile in `plugins/model-providers/kimi-coding/`, with aliases registered in `hermes_cli/providers.py`) talks to Moonshot's inference API and supplies LLM completions. The `kimiclaw` platform (this plugin) talks to kimi.com's bot endpoints (DM via WebSocket, group rooms via Connect RPC over HTTP long-poll) and runs as a chat bot on the consumer Kimi product. You can use either, both, or neither independently.
