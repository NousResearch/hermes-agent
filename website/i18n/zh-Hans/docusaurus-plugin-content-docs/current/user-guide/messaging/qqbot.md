---
title: "QQ Bot"
description: "QQ Bot 已移出 Hermes 核心，改为默认安装的官方 qqbot 插件"
---

# QQ Bot

[QQ Bot](https://q.qq.com) 网关平台（官方 QQ Bot API v2：私聊（C2C）、群组 @-提及、频道及直接消息、语音转写）已不再属于 Hermes 核心，而是作为 [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/qqbot) 中的官方 `qqbot` 插件发布，由 Nous Research 维护，完整的配置指南见插件仓库。

## 安装

无需任何操作：每个 Hermes home 都会自动安装 `qqbot` 插件，在设置 `QQ_APP_ID` 和 `QQ_CLIENT_SECRET` 之前它不会做任何事。如需重新安装（例如执行过 `hermes plugins remove qqbot` 之后）：

```bash
hermes plugins install qqbot
```

插件按 profile 安装；其他 profile 请使用 `hermes -p <profile> plugins install qqbot`。安装后重启 gateway 以加载该平台。

所有名称保持不变：`QQ_*` / `QQBOT_*` 环境变量（`QQ_APP_ID`、`QQ_CLIENT_SECRET`、`QQ_ALLOWED_USERS`、`QQBOT_HOME_CHANNEL` 等）、`config.yaml` 中的 `platforms.qqbot` 配置块、`qqbot` 平台名称以及 `hermes-qqbot` 工具集（插件注册该平台后即可再次解析）。

:::info 从内置 QQ Bot 的版本升级
无需任何操作。已经在使用 QQ Bot 的 profile（`.env` 中有 `QQ_APP_ID` 和 `QQ_CLIENT_SECRET`、`platforms.qqbot.extra` 中有 `app_id` 和 `client_secret`、`config.yaml` 中启用了 `platforms.qqbot`，或为其他平台选择了 `hermes-qqbot` 工具集）会在 `hermes update` 时，或该 profile 首次启动时（agent 或 gateway 启动，遵循 `security.allow_lazy_installs`），自动从插件目录安装该插件。安装失败会按退避策略重试，`hermes update` 总会重试。无论哪种方式，每个 profile 只自动安装一次：之后若用 `hermes plugins remove qqbot` 移除，它不会被重新安装。
:::
