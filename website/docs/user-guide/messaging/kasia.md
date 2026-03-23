---
sidebar_position: 6
title: "Kasia"
description: "Set up Hermes Agent on Kasia via the built-in Kaspa bridge"
---

# Kasia Setup

Hermes connects to Kasia through a built-in local Node bridge. The bridge derives a dedicated Hermes identity from your configured Kaspa seed phrase, polls a Kasia indexer for inbound handshakes and messages, and submits outbound transactions through a Kaspa wRPC node.

Kasia support in Hermes is **text-only**. Direct messages and announcement-style broadcast channels are supported.

:::warning Use a dedicated wallet
Use a dedicated Kaspa wallet for Hermes, not your primary wallet. The bridge needs enough mature KAS to pay network fees for outbound DMs, handshake replies, and broadcast sends. If the wallet is unfunded or low on funds, Kasia delivery can fail.
:::

## Prerequisites

- **Node.js v18+** and **npm** - the Kasia bridge runs as a local Node.js process
- **A 12- or 24-word Kaspa seed phrase** for a dedicated Hermes wallet
- **A Kasia indexer URL** and **Kaspa wRPC node URL**
- Optional: a custom **KNS API URL** if you do not want to use the default for your network

## Step 1: Run the Setup Wizard

```bash
hermes kasia
```

The wizard will:

1. Save `KASIA_ENABLED=true`
2. Prompt for your seed phrase
3. Prompt for indexer URL, node URL, and network
4. Ask whether to allow all users or configure an allowlist
5. Optionally save a Kasia home channel for cron delivery and cross-platform messages

Recommended mainnet defaults:

```bash
KASIA_INDEXER_URL=https://indexer.kasia.fyi
KASIA_NODE_WBORSH_URL=wss://wrpc.kasia.fyi
KASIA_NETWORK=mainnet
```

## Step 2: Configure Hermes

Basic example for `~/.hermes/.env`:

```bash
KASIA_ENABLED=true
KASIA_SEED_PHRASE="word1 word2 word3 ..."
KASIA_INDEXER_URL=https://indexer.kasia.fyi
KASIA_NODE_WBORSH_URL=wss://wrpc.kasia.fyi
KASIA_NETWORK=mainnet
KASIA_ALLOWED_USERS=kaspa:qpeeraddress
KASIA_HOME_CHANNEL=kaspa:qhomeaddress
```

Useful optional settings:

```bash
KASIA_INDEXER_URLS=https://indexer-a.example.com,https://indexer-b.example.com
KASIA_NODE_WBORSH_URLS=wss://node-a.example.com,wss://node-b.example.com
KASIA_KNS_URL=https://api.knsdomains.org/mainnet/api/v1
KASIA_FEE_POLICY=auto
KASIA_BRIDGE_PORT=3010
KASIA_SEND_WAIT_MS=5000
```

Notes:

- `KASIA_ALLOW_ALL_USERS=true` opens Hermes to any Kasia address. This is usually not what you want on an agent with terminal access.
- If you leave `KASIA_ALLOWED_USERS` empty and do not enable allow-all, the gateway denies incoming Kasia messages until you allowlist or pair users.
- If the gateway's unauthorized-DM behavior is `pair` (the default), unknown Kasia DMs can receive pairing instructions instead of being silently ignored.
- `KASIA_HOME_CHANNEL` can also be set later from the Kasia chat with `/sethome`.

## Step 3: Start the Gateway

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

The gateway starts the Kasia bridge automatically and keeps its state under `~/.hermes/kasia`.

## Broadcast Channels (Optional)

Hermes can also work with Kasia announcement-style broadcast channels:

```bash
KASIA_ALLOWED_BROADCAST_CHANNELS=alerts,ops
KASIA_BROADCAST_SUBSCRIPTIONS=news=kaspa:qpub1|kaspa:qpub2;alerts=kaspa:qpub3
```

- `KASIA_ALLOWED_BROADCAST_CHANNELS` lets Hermes publish to named channels.
- `KASIA_BROADCAST_SUBSCRIPTIONS` subscribes Hermes to channels from specific publisher addresses.
- The subscription format is `channel=publisher1|publisher2;channel2=publisher3`.
- Broadcast channels are not group chats. They are one-to-many announcement feeds.

## Diagnostics

```bash
hermes kasia doctor
```

The doctor checks:

- required env vars
- Node availability
- bridge health
- active indexer and node URLs
- wallet funding status
- configured access and home channel state

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Bridge does not start** | Confirm Node.js is on your PATH and that `scripts/kasia-bridge/node_modules` can be installed. |
| **Messages fail with funding errors** | Send more mature KAS to the Hermes wallet and wait for it to become spendable. |
| **No inbound messages arrive** | Verify your indexer URL, node URL, and that the sender is allowlisted or paired. |
| **KNS names do not resolve** | Set `KASIA_KNS_URL` explicitly or verify that `KASIA_NETWORK` matches the KNS endpoint you want. |
| **Broadcast sends are rejected** | Add the channel to `KASIA_ALLOWED_BROADCAST_CHANNELS` or enable `KASIA_ALLOW_ALL_BROADCAST_CHANNELS=true`. |

## Security

- Use a dedicated seed phrase and wallet for Hermes.
- Keep `KASIA_ALLOWED_USERS` tight unless you intentionally want open access.
- Treat `KASIA_SEED_PHRASE` like a secret.
- Review `~/.hermes/kasia/bridge.log` and `~/.hermes/kasia/state.json` carefully before sharing logs, since they may contain operational metadata.
