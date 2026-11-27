---
sidebar_position: 15
title: "WeCom Callback (Self-Built App)"
description: "WeCom callback mode moved out of Hermes core into the wecom Hermes Labs plugin"
---

# WeCom Callback (Self-Built App)

:::warning Hermes Labs plugin
WeCom is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.
:::

The WeCom (Enterprise WeChat) self-built app callback platform (`wecom_callback`) is no longer part of Hermes core. It ships in the `wecom` Hermes Labs plugin in [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/wecom), together with the [WeCom AI Bot](wecom.md) mode; the plugin carries the full setup guide.

## Install

```bash
hermes plugins install wecom
```

Plugins are installed per profile; use `hermes -p <profile> plugins install wecom` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `WECOM_CALLBACK_*` environment variables (`WECOM_CALLBACK_CORP_ID`, `WECOM_CALLBACK_CORP_SECRET`, `WECOM_CALLBACK_AGENT_ID`, `WECOM_CALLBACK_TOKEN`, `WECOM_CALLBACK_ENCODING_AES_KEY`, `WECOM_CALLBACK_HOST`, `WECOM_CALLBACK_PORT`, `WECOM_CALLBACK_ALLOWED_USERS`, ...), the `platforms.wecom_callback` block in `config.yaml` (`extra.corp_id`, `extra.apps` for multi-corp routing, `extra.port`, ...) and the `hermes-wecom-callback` toolset name. The callback listener still defaults to port 8645.

:::info Upgrading from a release that bundled WeCom
Nothing to do. Every profile that was already using WeCom callback mode (`WECOM_CALLBACK_CORP_ID` plus `WECOM_CALLBACK_CORP_SECRET` in its `.env`, or `platforms.wecom_callback.extra.corp_id` / `extra.apps` in `config.yaml`) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A profile using both modes gets the one plugin once. A failed attempt is retried with a backoff; `hermes update` always retries. If you later remove the plugin (`hermes plugins remove wecom`), it stays removed.

A profile that sets WeCom up after upgrading (credentials, or only `enabled: true`) gets nothing installed: Hermes Labs code is installed only when you ask, so the agent and gateway print one line naming `hermes plugins install wecom`. Run it (or use the WeCom card's install action in the dashboard or Desktop, where offered) to install and enable the plugin.

"After upgrading" means the profile's `config.yaml` carries the new schema stamp (`_config_version` 55 or later) and recorded no WeCom use when it reached it. A profile with WeCom settings only in `.env` and no stamped `config.yaml` looks exactly like one from before the move, so it is migrated automatically.
:::
