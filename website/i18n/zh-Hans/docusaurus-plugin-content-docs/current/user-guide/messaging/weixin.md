---
sidebar_position: 15
title: "Weixin（微信）"
description: "Weixin（个人微信）已移出 Hermes 核心，改为默认安装的官方 weixin 插件"
---

# Weixin（微信）

Hermes 通过腾讯的 **iLink Bot API** 连接[微信](https://weixin.qq.com/)个人账号。这与企业微信（WeCom）不同；企业微信请参阅 [WeCom 适配器](./wecom.md)。

Weixin 网关平台已不再属于 Hermes 核心，而是作为 [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/weixin) 中的官方 `weixin` 插件发布，由 Nous Research 维护。访问策略、媒体、context token 和故障排查等完整指南见插件仓库。

## 安装

无需任何操作：每个 Hermes home 都会自动安装 `weixin` 插件。如需重新安装（例如执行过 `hermes plugins remove weixin` 之后）：

```bash
hermes plugins install weixin
```

插件按 profile 安装；其他 profile 请使用 `hermes -p <profile> plugins install weixin`。安装后重启 gateway 以加载该平台。

## 配置

配置方式不变：运行 `hermes gateway setup`，选择 **Weixin**，用微信扫描二维码登录。向导会把 iLink 凭据保存到 `.env`。

所有名称保持不变：`WEIXIN_*` 环境变量（`WEIXIN_ACCOUNT_ID`、`WEIXIN_TOKEN`、`WEIXIN_BASE_URL`、`WEIXIN_DM_POLICY`、`WEIXIN_HOME_CHANNEL` 等）、`config.yaml` 中的 `platforms.weixin` 配置块、Hermes home 下 `weixin/accounts/` 中保存的账号，以及 `hermes-weixin` 工具集。

:::info 从内置 Weixin 的版本升级
无需任何操作。`hermes update` 或每个 profile 首次启动时会从插件目录安装该插件，已有的凭据、访问策略和 home channel 继续有效。
:::
