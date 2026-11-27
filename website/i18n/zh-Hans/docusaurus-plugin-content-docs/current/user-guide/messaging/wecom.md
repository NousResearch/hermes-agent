---
sidebar_position: 14
title: "WeCom（企业微信）"
description: "WeCom 已从 Hermes 核心移出，改为 wecom Hermes Labs 插件"
---

# WeCom（企业微信）

:::warning Hermes Labs 插件
WeCom is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.

企业微信（WeCom）现在是 Hermes Labs 插件：由 Nous Research 构建，不提供支持，今后很可能不再维护。欢迎社区提交修复。
:::

[WeCom](https://work.weixin.qq.com/)（企业微信）网关平台已不再是 Hermes 核心的一部分，现由 [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/wecom) 中的 `wecom` Hermes Labs 插件提供，完整的配置指南也在该插件中。同一个插件同时支持两种 WeCom 模式：AI Bot WebSocket 网关（本页）和[自建应用回调](wecom-callback.md)。

## 安装

```bash
hermes plugins install wecom
```

插件按 profile 安装；为其他 profile 安装请使用 `hermes -p <profile> plugins install wecom`。安装后重启网关即可提供该平台。

所有名称保持不变：`WECOM_*` 环境变量（`WECOM_BOT_ID`、`WECOM_SECRET`、`WECOM_WEBSOCKET_URL`、`WECOM_ALLOWED_USERS`、`WECOM_HOME_CHANNEL`、`WECOM_DM_POLICY`、`WECOM_GROUP_POLICY` 等）、`config.yaml` 中的 `platforms.wecom` 配置块（`extra.bot_id`、`extra.secret`、`dm_policy`、`group_policy`、`allow_from`、`groups` 等）以及 `hermes-wecom` 工具集名称。

:::info 从内置 WeCom 的版本升级
无需任何操作。已在使用 WeCom 的 profile（`.env` 中有 `WECOM_BOT_ID` 和 `WECOM_SECRET`，或 `config.yaml` 中有 `platforms.wecom.extra.bot_id`）会在 `hermes update` 时，或该 profile 首次启动时（agent 或网关启动；遵循 `security.allow_lazy_installs`），自动从插件目录安装该插件。每个 profile 只执行一次：安装失败会按退避策略重试；`hermes update` 总会重试。如果之后移除了插件（`hermes plugins remove wecom`），它将保持移除状态。
:::
