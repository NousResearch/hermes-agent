---
title: "Apple Reminders — 通过 remindctl 管理 Apple Reminders：添加、列出、完成"
sidebar_label: "Apple Reminders"
description: "通过 remindctl 管理 Apple Reminders：添加、列出、完成"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Reminders

通过 remindctl 管理 Apple Reminders：添加、列出、完成。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/apple/apple-reminders` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | macos |
| 标签 | `Reminders`, `tasks`, `todo`, `macOS`, `Apple` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Apple Reminders

使用 `remindctl` 直接从终端管理 Apple Reminders。任务通过 iCloud 在所有 Apple 设备间同步。

## 前提条件

- 安装了 Reminders.app 的 **macOS**
- 安装：`brew install steipete/tap/remindctl`
- 在提示时授予 Reminders 权限
- 检查：`remindctl status` / 请求授权：`remindctl authorize`

## 何时使用

- 用户提到"提醒"或"Reminders 应用"
- 创建带有截止日期且需同步到 iOS 的个人待办事项
- 管理 Apple Reminders 列表
- 用户希望任务出现在其 iPhone/iPad 上

## 何时不使用

- 调度 agent 提醒 → 改用 cronjob 工具
- 日历事件 → 使用 Apple Calendar 或 Google Calendar
- 项目任务管理 → 使用 GitHub Issues、Notion 等
- 用户说"提醒我"但意指 agent 提醒 → 先行确认

## 快速参考

### 查看提醒

```bash
remindctl                    # 今日提醒
remindctl today              # 今天
remindctl tomorrow           # 明天
remindctl week               # 本周
remindctl overdue            # 已逾期
remindctl all                # 全部
remindctl 2026-01-04         # 指定日期
```

### 管理列表

```bash
remindctl list               # 列出所有列表
remindctl list Work          # 显示指定列表
remindctl list Projects --create    # 创建列表
remindctl list Work --delete        # 删除列表
```

### 创建提醒

```bash
remindctl add "Buy milk"
remindctl add --title "Call mom" --list Personal --due tomorrow
remindctl add --title "Meeting prep" --due "2026-02-15 09:00"
```

### 完成 / 删除

```bash
remindctl complete 1 2 3          # 按 ID 完成
remindctl delete 4A83 --force     # 按 ID 删除
```

### 输出格式

```bash
remindctl today --json       # JSON 格式，用于脚本处理
remindctl today --plain      # TSV 格式
remindctl today --quiet      # 仅显示数量
```

## 日期格式

`--due` 及日期筛选器接受以下格式：
- `today`、`tomorrow`、`yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601（`2026-01-04T12:34:56Z`）

## 规则

1. 当用户说"提醒我"时，需确认：是 Apple Reminders（同步到手机）还是 agent cronjob 提醒
2. 创建提醒前始终确认提醒内容和截止日期
3. 使用 `--json` 进行程序化解析