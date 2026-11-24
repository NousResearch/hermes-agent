---
sidebar_position: 11
title: "Feishu / Lark"
description: "Feishu / Lark moved out of Hermes core into the feishu Hermes Labs plugin"
---

# Feishu / Lark

:::warning Hermes Labs plugin
Feishu / Lark is now a Hermes Labs plugin: built by Nous Research, not supported. Fixes from the community are welcome.
:::

The [Feishu / Lark](https://open.feishu.cn/) gateway platform and its document-comment tools (`feishu_doc_read`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment`, `feishu_drive_add_comment`) are no longer part of Hermes core. They ship as the `feishu` plugin in [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/feishu), which carries the full setup guide (WebSocket and webhook modes, QR onboarding, group policies, document comments, meeting invites).

## Install

```bash
hermes plugins install feishu
```

Plugins are installed per profile; use `hermes -p <profile> plugins install feishu` for another profile. Restart the gateway afterwards so it serves the platform.

Every name stays the same: the `FEISHU_*` environment variables (`FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_DOMAIN`, `FEISHU_CONNECTION_MODE`, `FEISHU_ENCRYPT_KEY`, `FEISHU_VERIFICATION_TOKEN`, `FEISHU_HOME_CHANNEL`, ...), the `platforms.feishu` block in `config.yaml`, `feishu_comment_rules.json`, the `feishu_doc` / `feishu_drive` toolsets and the tool names.

:::info Upgrading from a release that bundled Feishu / Lark
Nothing to do. Every profile that was already using Feishu (`FEISHU_APP_ID` plus `FEISHU_APP_SECRET` in its `.env`, `platforms.feishu` enabled or carrying an `app_id` in `config.yaml`, or the `feishu_doc` / `feishu_drive` / `hermes-feishu` toolset listed for another platform) gets the plugin installed automatically from the catalog by `hermes update`, or the first time the profile starts (agent or gateway start; this honours `security.allow_lazy_installs`). A failed attempt is retried with a backoff; `hermes update` always retries. This happens once per profile: if you later remove the plugin (`hermes plugins remove feishu`), it stays removed.
:::
