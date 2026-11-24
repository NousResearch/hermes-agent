---
title: "QQ Bot"
description: "QQ Bot moved out of Hermes core into the qqbot Hermes Labs plugin"
---

# QQ Bot

:::warning Hermes Labs plugin
QQ Bot is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.
:::

The [QQ Bot](https://q.qq.com) gateway platform (Official QQ Bot API v2: C2C, group @-mentions, guild and direct messages, voice transcription) is no longer part of Hermes core. It ships as the `qqbot` plugin in [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/qqbot), which carries the full setup guide.

## Install

```bash
hermes plugins install qqbot
```

Plugins are installed per profile; use `hermes -p <profile> plugins install qqbot` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `QQ_*` / `QQBOT_*` environment variables (`QQ_APP_ID`, `QQ_CLIENT_SECRET`, `QQ_ALLOWED_USERS`, `QQBOT_HOME_CHANNEL`, ...), the `platforms.qqbot` block in `config.yaml`, the `qqbot` platform name and the `hermes-qqbot` toolset (it resolves again once the plugin registers the platform).

:::info Upgrading from a release that bundled QQ Bot
Nothing to do. Every profile that was already using QQ Bot (`QQ_APP_ID` plus `QQ_CLIENT_SECRET` in its `.env`, `app_id` and `client_secret` in `platforms.qqbot.extra`, `platforms.qqbot` enabled in `config.yaml`, or the `hermes-qqbot` toolset listed for another platform) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A failed attempt is retried with a backoff; `hermes update` always retries. This happens once per profile: if you later remove the plugin (`hermes plugins remove qqbot`), it stays removed.
:::
