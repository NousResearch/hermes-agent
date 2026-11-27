---
sidebar_position: 15
title: "WeCom 回调（自建应用）"
description: "WeCom 回调模式已从 Hermes 核心移出，改为 wecom Hermes Labs 插件"
---

# WeCom 回调（自建应用）

:::warning Hermes Labs 插件
WeCom is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.

企业微信（WeCom）现在是 Hermes Labs 插件：由 Nous Research 构建，不提供支持，今后很可能不再维护。欢迎社区提交修复。
:::

企业微信（WeCom）自建应用回调平台（`wecom_callback`）已不再是 Hermes 核心的一部分，现与 [WeCom AI Bot](wecom.md) 模式一起由 [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/wecom) 中的 `wecom` Hermes Labs 插件提供，完整的配置指南也在该插件中。

## 安装

```bash
hermes plugins install wecom
```

插件按 profile 安装；为其他 profile 安装请使用 `hermes -p <profile> plugins install wecom`。安装后重启网关即可提供该平台。

所有名称保持不变：`WECOM_CALLBACK_*` 环境变量（`WECOM_CALLBACK_CORP_ID`、`WECOM_CALLBACK_CORP_SECRET`、`WECOM_CALLBACK_AGENT_ID`、`WECOM_CALLBACK_TOKEN`、`WECOM_CALLBACK_ENCODING_AES_KEY`、`WECOM_CALLBACK_HOST`、`WECOM_CALLBACK_PORT`、`WECOM_CALLBACK_ALLOWED_USERS` 等）、`config.yaml` 中的 `platforms.wecom_callback` 配置块（`extra.corp_id`、用于多企业路由的 `extra.apps`、`extra.port` 等）以及 `hermes-wecom-callback` 工具集名称。回调监听端口默认仍为 8645。

:::info 从内置 WeCom 的版本升级
`config.yaml` 来自迁移前的 profile 无需任何操作。这类已在使用 WeCom 回调模式的 profile（`.env` 中有 `WECOM_CALLBACK_CORP_ID` 和 `WECOM_CALLBACK_CORP_SECRET`，或 `config.yaml` 中有 `platforms.wecom_callback.extra.corp_id` / `extra.apps`）会在 `hermes update` 时，或该 profile 首次启动时（遵循 `security.allow_lazy_installs`），自动从插件目录安装该插件。同时使用两种模式的 profile 只会安装一次该插件。安装失败会按退避策略重试；`hermes update` 总会重试。如果之后移除了插件（`hermes plugins remove wecom`），它将保持移除状态。

升级之后才配置 WeCom 的 profile（填写了凭据，或只设置了 `enabled: true`）不会自动安装任何内容：Hermes Labs 代码只在你要求时安装，agent 和网关会输出一行提示 `hermes plugins install wecom`。运行该命令（或在 dashboard / Desktop 提供时使用 WeCom 卡片上的安装操作）即可安装并启用插件。

“来自迁移前”指该 profile 的 `config.yaml` 带有低于 55 的版本标记（`_config_version`），或升级到 55 时记录了 WeCom 的使用。只在 `.env` 中配置了 WeCom、没有带版本标记的 `config.yaml` 的 profile 与新配置无法区分，因此只会收到安装命令的提示：如果你在升级前就以这种方式使用 WeCom，请运行一次 `hermes plugins install wecom`。
:::
