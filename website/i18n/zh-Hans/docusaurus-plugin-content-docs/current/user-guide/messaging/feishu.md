---
sidebar_position: 11
title: "飞书 / Lark"
description: "飞书 / Lark 已从 Hermes 核心移出，成为默认安装的官方 feishu 插件"
---

# 飞书 / Lark

[飞书 / Lark](https://open.feishu.cn/) 网关平台及其文档评论工具（`feishu_doc_read`、`feishu_drive_list_comments`、`feishu_drive_list_comment_replies`、`feishu_drive_reply_comment`、`feishu_drive_add_comment`）已不再属于 Hermes 核心。它们作为 [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/feishu) 中的官方 `feishu` 插件发布，完整的配置指南也在那里。

## 安装

无需任何操作：每个 Hermes home 都会自动安装 `feishu` 插件，在设置 `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET`（或 `platforms.feishu` 下的 `app_id` / `app_secret`）之前它保持空闲：网关不会启动飞书适配器，模型也看不到 `feishu_doc` / `feishu_drive` 工具；凭据就绪后，下一个会话即可使用这些工具。如需重新安装（例如执行过 `hermes plugins remove feishu` 之后）：

```bash
hermes plugins install feishu
```

插件按 profile 安装；其他 profile 请使用 `hermes -p <profile> plugins install feishu`。安装后重启网关即可提供该平台。

所有名称保持不变：`FEISHU_*` 环境变量、`config.yaml` 中的 `platforms.feishu` 配置块、`feishu_comment_rules.json`、`feishu_doc` / `feishu_drive` 工具集以及工具名称。

:::info 从内置飞书的版本升级
无需任何操作。已在使用飞书的每个 profile（`.env` 中有 `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET`、`config.yaml` 中启用了 `platforms.feishu` 或其中有 `app_id`，或为其他平台列出了 `feishu_doc` / `feishu_drive` / `hermes-feishu` 工具集）都会在 `hermes update` 时，或该 profile 首次启动时（agent 或网关启动；遵循 `security.allow_lazy_installs`）自动从插件目录安装该插件。失败会按退避重试；`hermes update` 总会重试。无论哪种方式，每个 profile 只安装一次：之后若移除插件（`hermes plugins remove feishu`），它不会再被安装。

例外：设置了 `plugins.isolation: host` 的 profile 会在插件宿主进程中加载目录插件，而平台适配器无法在其中运行。插件会安装，但在该 profile 改为进程内加载插件之前，飞书保持关闭；升级提示会说明这一点。
:::
