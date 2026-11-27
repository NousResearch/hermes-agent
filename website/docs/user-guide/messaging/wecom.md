---
sidebar_position: 14
title: "WeCom (Enterprise WeChat)"
description: "WeCom moved out of Hermes core into the wecom Hermes Labs plugin"
---

# WeCom (Enterprise WeChat)

:::warning Hermes Labs plugin
WeCom is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.
:::

The [WeCom](https://work.weixin.qq.com/) (Enterprise WeChat) gateway platform is no longer part of Hermes core. It ships as the `wecom` Hermes Labs plugin in [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/wecom), which carries the full setup guide. The same plugin serves both WeCom modes: the AI Bot WebSocket gateway (this page) and the [self-built app callback](wecom-callback.md).

## Install

```bash
hermes plugins install wecom
```

Plugins are installed per profile; use `hermes -p <profile> plugins install wecom` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `WECOM_*` environment variables (`WECOM_BOT_ID`, `WECOM_SECRET`, `WECOM_WEBSOCKET_URL`, `WECOM_ALLOWED_USERS`, `WECOM_HOME_CHANNEL`, `WECOM_DM_POLICY`, `WECOM_GROUP_POLICY`, ...), the `platforms.wecom` block in `config.yaml` (`extra.bot_id`, `extra.secret`, `dm_policy`, `group_policy`, `allow_from`, `groups`, ...) and the `hermes-wecom` toolset name.

:::info Upgrading from a release that bundled WeCom
Nothing to do. Every profile that was already using WeCom (`WECOM_BOT_ID` plus `WECOM_SECRET` in its `.env`, or `platforms.wecom.extra.bot_id` in `config.yaml`) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A failed attempt is retried with a backoff; `hermes update` always retries. This happens once per profile: if you later remove the plugin (`hermes plugins remove wecom`), it stays removed.

A profile that sets WeCom up after upgrading (credentials, or only `enabled: true`) gets nothing installed: Hermes Labs code is installed only when you ask, so the agent and gateway print one line naming `hermes plugins install wecom`. Run it (or use the WeCom card's install action in the dashboard or Desktop, where offered) to install and enable the plugin.

"After upgrading" means the profile's `config.yaml` carries the new schema stamp (`_config_version` 55 or later) and recorded no WeCom use when it reached it. A profile with WeCom settings only in `.env` and no stamped `config.yaml` looks exactly like one from before the move, so it is migrated automatically.
:::
