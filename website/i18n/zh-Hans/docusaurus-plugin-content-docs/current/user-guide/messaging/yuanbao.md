---
sidebar_position: 16
title: "Yuanbao"
description: "元宝（Yuanbao）已移出 Hermes 核心，改为 yuanbao Hermes Labs 插件"
---

# Yuanbao

:::warning Hermes Labs 插件
Yuanbao is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not maintained going forward. Fixes from the community are welcome.

元宝现在是 Hermes Labs 插件：由 Nous Research 构建，不提供支持，今后很可能不再维护。欢迎社区提交修复。
:::

[元宝（Yuanbao）](https://yuanbao.tencent.com/) 网关平台及其 `yb_*` 工具（`yb_query_group_info`、`yb_query_group_members`、`yb_send_dm`、`yb_search_sticker`、`yb_send_sticker`）已不再属于 Hermes 核心，而是作为 [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/yuanbao) 中的 `yuanbao` 插件发布，完整的配置指南见插件仓库。

## 安装

```bash
hermes plugins install yuanbao
```

插件按 profile 安装；其他 profile 请使用 `hermes -p <profile> plugins install yuanbao`。安装后重启 gateway 以加载该平台。

所有名称保持不变：`YUANBAO_*` 环境变量（`YUANBAO_APP_ID` 或旧名 `YUANBAO_APP_KEY`、`YUANBAO_APP_SECRET`、`YUANBAO_HOME_CHANNEL` 等）、`config.yaml` 中的 `platforms.yuanbao`（以及旧的 `gateway.yuanbao`）配置块、`yuanbao` 工具集和各工具名称。

:::info 从内置元宝的版本升级
无需任何操作。已经在使用元宝的 profile（`.env` 中有 `YUANBAO_APP_ID` 或 `YUANBAO_APP_KEY` 以及 `YUANBAO_APP_SECRET`、`config.yaml` 中启用了 `platforms.yuanbao`，或为其他平台选择了 `yuanbao` / `hermes-yuanbao` 工具集）会在 `hermes update` 时，或该 profile 首次启动时（agent 或 gateway 启动，遵循 `security.allow_lazy_installs`），自动从插件目录安装该插件。安装失败会按退避策略重试，`hermes update` 总会重试。每个 profile 只自动安装一次：之后若用 `hermes plugins remove yuanbao` 移除，它不会被重新安装。
:::
