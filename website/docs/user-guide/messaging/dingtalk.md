---
sidebar_position: 10
title: "DingTalk"
description: "DingTalk moved out of Hermes core into the dingtalk Hermes Labs plugin"
---

# DingTalk

:::warning Hermes Labs plugin
DingTalk is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.
:::

The [DingTalk](https://www.dingtalk.com/) gateway platform (Stream Mode bot, AI Cards, QR-code setup) is no longer part of Hermes core. It ships as the `dingtalk` plugin in [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/dingtalk), which carries the full setup guide.

## Install

```bash
hermes plugins install dingtalk
```

Plugins are installed per profile; use `hermes -p <profile> plugins install dingtalk` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `DINGTALK_*` environment variables (`DINGTALK_CLIENT_ID`, `DINGTALK_CLIENT_SECRET`, `DINGTALK_ALLOWED_USERS`, `DINGTALK_HOME_CHANNEL`, ...), the `platforms.dingtalk` block in `config.yaml`, the `dingtalk` platform name for cron and webhook delivery, and the `hermes-dingtalk` toolset.

:::info Upgrading from a release that bundled DingTalk
Nothing to do for a profile whose `config.yaml` is from before the move. Every such profile that was already using DingTalk (`DINGTALK_CLIENT_ID` plus `DINGTALK_CLIENT_SECRET` in its `.env`, or `platforms.dingtalk` enabled in `config.yaml`) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A failed attempt is retried with a backoff; `hermes update` always retries. This happens once per profile: if you later remove the plugin (`hermes plugins remove dingtalk`), it stays removed. A webhook-only profile (`DINGTALK_WEBHOOK_URL` for cron delivery) counts as using DingTalk too.

A profile that sets DingTalk up after upgrading gets nothing installed: Hermes Labs code is installed only when you ask, so the agent and gateway print one line naming `hermes plugins install dingtalk`. Run it to install and enable the plugin. The dashboard and Desktop Channels pages do not offer Hermes Labs platforms in their list of platforms to add, but once the plugin is installed DingTalk appears there and is configured like any other channel.

"From before the move" means the profile's `config.yaml` carries a schema stamp below 55 (`_config_version`), or recorded DingTalk use when it reached 55. A profile with DingTalk settings only in `.env` and no stamped `config.yaml` looks exactly like a new setup, so it is told the command instead: if you ran DingTalk that way before upgrading, run `hermes plugins install dingtalk` once.
:::
