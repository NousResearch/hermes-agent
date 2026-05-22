---
sidebar_position: 2
title: "斜杠命令参考"
description: "交互式 CLI 与消息平台斜杠命令的完整参考"
---

# 斜杠命令参考

Hermes 有两类斜杠命令界面，均由 `hermes_cli/commands.py` 中的中央 `COMMAND_REGISTRY` 驱动：

- **交互式 CLI 斜杠命令** — 由 `cli.py` 分发，自动补全来自注册表
- **消息平台斜杠命令** — 由 `gateway/run.py` 分发，帮助文本与平台菜单从注册表生成

已安装的技能也会作为动态斜杠命令暴露在两个界面上，例如捆绑技能 `/plan` 会打开计划模式并将 Markdown 计划保存在 `.hermes/plans/` 下。

## 交互式 CLI 斜杠命令

在 CLI 中输入 `/` 打开自动补全菜单。内置命令不区分大小写。

### 会话相关

常用命令示例：`/new`、`/clear`、`/history`、`/save`、`/retry`、`/rollback`、`/stop`、`/queue`、`/steer`、`/goal`、`/resume`、`/sessions`、`/status`、`/agents`、`/background`、`/branch`、`/handoff` 等。功能与英文原文一致（详情请参阅源文档）。

### 配置

常用配置性命令示例：`/config`、`/model`、`/personality`、`/verbose`、`/fast`、`/reasoning`、`/skin`、`/statusbar`、`/voice`、`/yolo` 等。

### 工具与技能

常用工具/技能命令示例：`/tools`、`/toolsets`、`/browser`、`/skills`、`/cron`、`/curator`、`/kanban`、`/reload-mcp`、`/reload-skills`、`/reload`、`/plugins` 等。

### 信息类

`/help`、`/usage`、`/insights`、`/platforms`、`/paste`、`/copy`、`/image`、`/debug`、`/profile` 等。

### 退出

`/quit` 或 `/exit` 退出 CLI。

### 动态 CLI 命令

已安装的技能也会作为 `/<skill-name>` 命令动态可用，例如 `/gif-search`、`/github-pr-workflow` 等。

### 快速命令

在 `~/.hermes/config.yaml` 中配置 `quick_commands` 将短命令映射为 shell 命令或其他斜杠命令：

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  deploy:
    type: exec
    command: scripts/deploy.sh
  inbox:
    type: alias
    target: /gmail unread
```

### 自定义模型别名 {#custom-model-aliases}

你可以为常用模型定义别名，然后在 CLI 或消息平台中通过 `/model <alias>` 使用。支持完整形式与简短形式配置，别名优先于内置短名。

## 消息平台斜杠命令

消息网关在 Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams 等平台支持多数内置命令，例如 `/new`、`/reset`、`/status`、`/stop`、`/model`、`/personality`、`/fast`、`/retry`、`/undo`、`/sethome`、`/compress`、`/topic`（Telegram DM 特有）、`/title`、`/resume`、`/usage`、`/insights`、`/voice`、`/rollback`、`/background`、`/queue`、`/steer`、`/goal`、`/footer`、`/curator`、`/kanban`、`/reload-mcp`、`/yolo`、`/commands`、`/approve`、`/deny`、`/update`、`/restart`、`/debug`、`/help`、`/<skill-name>` 等。

## 说明

- 一些命令仅在 CLI 可用（例如 `/skin`, `/snapshot`, `/gquota`, `/reload` 等）。
- `/verbose` 默认仅在 CLI 可用，可通过配置在消息平台启用。
- 某些命令（如 `/sethome`, `/approve` 等）为消息特有。
- 许多命令在 CLI 与消息平台中均可用（例如 `/status`, `/background`, `/queue`, `/steer`, `/voice` 等）。

（此页根据英文原文翻译并保留主要功能点；若需我可以把所有子命令表格逐条完全翻译。）

---
