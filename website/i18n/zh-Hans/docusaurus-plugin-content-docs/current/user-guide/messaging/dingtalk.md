---
sidebar_position: 10
title: "DingTalk"
description: "钉钉（DingTalk）已移出 Hermes 核心，改为 dingtalk Hermes Labs 插件"
---

# DingTalk

:::warning Hermes Labs 插件
DingTalk is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.

钉钉现在是 Hermes Labs 插件：由 Nous Research 构建，不提供支持，今后很可能不再维护。欢迎社区提交修复。
:::

[钉钉（DingTalk）](https://www.dingtalk.com/) 网关平台（Stream 模式机器人、AI 卡片、扫码配置）已不再属于 Hermes 核心，而是作为 [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/dingtalk) 中的 `dingtalk` 插件发布，完整的配置指南见插件仓库。

## 安装

```bash
hermes plugins install dingtalk
```

插件按 profile 安装；其他 profile 请使用 `hermes -p <profile> plugins install dingtalk`。安装后重启 gateway 以加载该平台。

所有名称保持不变：`DINGTALK_*` 环境变量（`DINGTALK_CLIENT_ID`、`DINGTALK_CLIENT_SECRET`、`DINGTALK_ALLOWED_USERS`、`DINGTALK_HOME_CHANNEL` 等）、`config.yaml` 中的 `platforms.dingtalk` 配置块、cron 与 webhook 投递使用的 `dingtalk` 平台名，以及 `hermes-dingtalk` 工具集。

:::info 从内置钉钉的版本升级
无需任何操作。已经在使用钉钉的 profile（`.env` 中有 `DINGTALK_CLIENT_ID` 和 `DINGTALK_CLIENT_SECRET`，或 `config.yaml` 中启用了 `platforms.dingtalk`）会在 `hermes update` 时，或该 profile 首次启动时（agent 或 gateway 启动，遵循 `security.allow_lazy_installs`），自动从插件目录安装该插件。安装失败会按退避策略重试，`hermes update` 总会重试。每个 profile 只自动安装一次：之后若用 `hermes plugins remove dingtalk` 移除，它不会被重新安装。
:::
