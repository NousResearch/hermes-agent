---
title: "Apple Reminders — 通过 remindctl 管理 Apple 提醒：添加、列出、完成"
sidebar_label: "Apple Reminders"
description: "通过 remindctl 管理 Apple 提醒：添加、列出、完成"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Reminders

通过 remindctl 管理 Apple 提醒：添加、列出、完成。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/apple/apple-reminders` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | macos |
| 标签 | `Reminders`、`tasks`、`todo`、`macOS`、`Apple` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# Apple Reminders

使用 `remindctl` 直接从终端管理 Apple Reminders。任务通过 iCloud 在所有 Apple 设备间同步。

## 前置条件

- **macOS** 带有 Reminders.app
- 安装：`brew install steipete/tap/remindctl`
- 出现提示时授予 Reminders 权限
- 检查：`remindctl status` / 请求：`remindctl authorize`

## 使用场景

- 用户提到"提醒"或"Reminders 应用"
- 创建带截止日期的个人待办事项，同步到 iOS
- 管理 Apple Reminders 列表
- 用户希望任务出现在他们的 iPhone/iPad 上

## 不使用场景

- 调度代理警报 → 改用 cronjob 工具
- 日历事件 → 使用 Apple Calendar 或 Google Calendar
- 项目任务管理 → 使用 GitHub Issues、Notion 等
- 如果用户说"提醒我"但指的是代理警报 → 先澄清

## 快速参考

### 查看提醒

```bash
remindctl                    # 今天的提醒
remindctl today              # 今天
remindctl tomorrow           # 明天
remindctl week               # 本周
remindctl overdue            # 已过期
remindctl all                # 所有
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

### 完成/删除

```bash
remindctl complete 1 2 3          # 按 ID 完成
remindctl delete 4A83 --force     # 按 ID 删除
```

### 输出格式

```bash
remindctl today --json       # JSON 用于脚本
remindctl today --plain      # TSV 格式
remindctl today --quiet      # 仅计数
```

## 日期格式

`--due` 和日期过滤器接受：
- `today`、`tomorrow`、`yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601（`2026-01-04T12:34:56Z`）

## 规则

1. 当用户说"提醒我"时，澄清：Apple Reminders（同步到手机）vs 代理 cronjob 警报
2. 创建前始终确认提醒内容和截止日期
3. 使用 `--json` 用于程序化解析
