---
title: "QQ Bot"
description: "QQ Bot moved out of Hermes core into the official qqbot plugin, installed by default"
---

# QQ Bot

The [QQ Bot](https://q.qq.com) gateway platform (Official QQ Bot API v2: C2C, group @-mentions, guild and direct messages, voice transcription) is no longer part of Hermes core. It ships as the official `qqbot` plugin in [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/qqbot), maintained by Nous Research, which carries the full setup guide.

## Install

Nothing to do: the `qqbot` plugin is installed automatically on every Hermes home, and it stays idle until `QQ_APP_ID` and `QQ_CLIENT_SECRET` are set. To reinstall it (for example after `hermes plugins remove qqbot`):

```bash
hermes plugins install qqbot
```

Plugins are installed per profile; use `hermes -p <profile> plugins install qqbot` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `QQ_*` / `QQBOT_*` environment variables (`QQ_APP_ID`, `QQ_CLIENT_SECRET`, `QQ_ALLOWED_USERS`, `QQBOT_HOME_CHANNEL`, ...), the `platforms.qqbot` block in `config.yaml`, the `qqbot` platform name and the `hermes-qqbot` toolset (it resolves again once the plugin registers the platform).

:::info Upgrading from a release that bundled QQ Bot
Nothing to do. Every profile that was already using QQ Bot (`QQ_APP_ID` plus `QQ_CLIENT_SECRET` in its `.env`, `app_id` and `client_secret` in `platforms.qqbot.extra`, `platforms.qqbot` enabled in `config.yaml`, or the `hermes-qqbot` toolset listed for another platform) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A failed attempt is retried with a backoff; `hermes update` always retries. Either way it is installed once per profile: if you later remove the plugin (`hermes plugins remove qqbot`), it stays removed.
:::
