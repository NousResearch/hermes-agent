---
sidebar_position: 11
title: "Feishu / Lark"
description: "Feishu / Lark moved out of Hermes core into the official feishu plugin, installed by default"
---

# Feishu / Lark

The [Feishu / Lark](https://open.feishu.cn/) gateway platform and its document-comment tools (`feishu_doc_read`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment`, `feishu_drive_add_comment`) are no longer part of Hermes core. They ship as the official `feishu` plugin in [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/feishu), which carries the full setup guide (WebSocket and webhook modes, QR onboarding, group policies, document comments, meeting invites).

## Install

Nothing to do: the `feishu` plugin is installed automatically on every Hermes home, and it stays idle until `FEISHU_APP_ID` and `FEISHU_APP_SECRET` are set (or `app_id` / `app_secret` under `platforms.feishu`). Until then the gateway starts no Feishu adapter and the `feishu_doc` / `feishu_drive` tools are not offered to the model; they appear on the next session once the credentials are there. To reinstall it (for example after `hermes plugins remove feishu`):

```bash
hermes plugins install feishu
```

Plugins are installed per profile; use `hermes -p <profile> plugins install feishu` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `FEISHU_*` environment variables (`FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_DOMAIN`, `FEISHU_CONNECTION_MODE`, `FEISHU_ENCRYPT_KEY`, `FEISHU_VERIFICATION_TOKEN`, `FEISHU_HOME_CHANNEL`, ...), the `platforms.feishu` block in `config.yaml`, `feishu_comment_rules.json`, the `feishu_doc` / `feishu_drive` toolsets and the tool names.

:::info Upgrading from a release that bundled Feishu / Lark
Nothing to do. Every profile that was already using Feishu (`FEISHU_APP_ID` plus `FEISHU_APP_SECRET` in its `.env`, `platforms.feishu` enabled or carrying an `app_id` in `config.yaml`, or the `feishu_doc` / `feishu_drive` / `hermes-feishu` toolset listed for another platform) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A failed attempt is retried with a backoff; `hermes update` always retries. Either way it is installed once per profile: if you later remove the plugin (`hermes plugins remove feishu`), it stays removed. That includes the idle copy Hermes installed by default before Feishu left core: if you removed it, Feishu is off after the update, and starts say so with the command to get it back (`hermes plugins install feishu`).

One exception: a profile that sets `plugins.isolation: host` loads catalog plugins in the plugin host, where platform adapters cannot run. The plugin installs, but Feishu stays off there until that profile loads plugins in-process; the upgrade notice says so.
:::
