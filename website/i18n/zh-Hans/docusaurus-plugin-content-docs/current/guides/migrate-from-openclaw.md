---
sidebar_position: 10
title: "从 OpenClaw 迁移"
description: "将 OpenClaw/Clawdbot 配置迁移到 Hermes 的操作指南与核对清单。"
---

# 从 OpenClaw 迁移

`hermes claw migrate` 可把 OpenClaw（含旧版 Clawdbot/Moltbot）配置导入 Hermes。

## 快速开始

```bash
# 预览并迁移（默认先展示预览）
hermes claw migrate

# 只预览，不落盘
hermes claw migrate --dry-run

# 全量迁移（含密钥）并跳过确认
hermes claw migrate --preset full --migrate-secrets --yes
```

## 常用参数

- `--dry-run`: 仅预览
- `--preset full|user-data`: 预设迁移范围
- `--migrate-secrets`: 迁移 API 密钥
- `--overwrite`: 冲突时覆盖
- `--source <path>`: 自定义 OpenClaw 目录

## 迁移后必做检查

1. 查看迁移报告与冲突项
2. 启动新会话验证技能/记忆是否生效
3. 运行 `hermes status` 检查 provider 认证
4. 如果用了消息平台，重启 gateway
5. 核对 `session_reset`、TTS、MCP 等关键配置

## 常见问题

- 找不到 OpenClaw 目录: 使用 `--source`
- 密钥没有迁移: 确认加了 `--migrate-secrets`
- 技能未出现: 查看 `~/.hermes/skills/openclaw-imports/` 并重开会话

## 相关文档

- [配置](/user-guide/configuration)
- [定时任务（Cron）](/user-guide/features/cron)
- [插件系统](/user-guide/features/plugins)
