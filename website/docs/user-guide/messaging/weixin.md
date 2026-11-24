---
sidebar_position: 15
title: "Weixin (WeChat)"
description: "Weixin (personal WeChat) moved out of Hermes core into the official weixin plugin, installed by default"
---

# Weixin (WeChat)

Hermes connects to [WeChat](https://weixin.qq.com/) (微信) personal accounts through Tencent's **iLink Bot API**. This is not WeCom (Enterprise WeChat); for that, see the [WeCom adapter](./wecom.md).

The Weixin gateway platform is no longer part of Hermes core. It ships as the official `weixin` plugin in [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/weixin), maintained by Nous Research. The plugin repository has the full guide: access policies, media, context tokens and troubleshooting.

## Install

Nothing to do: the `weixin` plugin is installed automatically on every Hermes home. To reinstall it (for example after `hermes plugins remove weixin`):

```bash
hermes plugins install weixin
```

Plugins are installed per profile; use `hermes -p <profile> plugins install weixin` for another profile. Restart the gateway afterwards so it serves the platform.

## Setup

Setup is unchanged. Run `hermes gateway setup`, pick **Weixin**, and scan the QR code with WeChat to log in. The wizard saves the iLink credentials to your `.env`.

Every name stays the same: the `WEIXIN_*` environment variables (`WEIXIN_ACCOUNT_ID`, `WEIXIN_TOKEN`, `WEIXIN_BASE_URL`, `WEIXIN_DM_POLICY`, `WEIXIN_HOME_CHANNEL`, ...), the `platforms.weixin` block in `config.yaml`, the saved accounts under `weixin/accounts/` in your Hermes home, and the `hermes-weixin` toolset.

:::info Upgrading from a release that bundled Weixin
Nothing to do. `hermes update`, or the first start of each profile, installs the plugin from the catalog. Existing credentials, access policies and home channels keep working.
:::
