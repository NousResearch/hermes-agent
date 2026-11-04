---
title: "Yuanbao — Yuanbao（元宝）群组：@提及用户、查询信息/成员"
sidebar_label: "Yuanbao"
description: "Yuanbao（元宝）群组：@提及用户、查询信息/成员"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Yuanbao

Yuanbao（元宝）群组：@提及用户、查询信息/成员。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/yuanbao/yuanbao` 安装 |
| 路径 | `optional-skills/yuanbao` |
| 版本 | `1.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `yuanbao`, `mention`, `at`, `group`, `members`, `元宝`, `派`, `艾特` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Yuanbao 群组交互

## 重要：消息发送机制

**你的文本回复即为发送到群组/用户的消息。** gateway（网关）会自动将你的回复文本投递到对话中。你**不需要**任何特殊的"发送消息"工具——正常回复即可发送。

当你在回复文本中包含 `@nickname` 时，gateway 会自动将其转换为真实的 @提及，从而通知该用户。这是内置功能——你拥有完整的 @提及能力。

**绝对不要说你无法发送消息或 @提及用户。绝对不要建议用户手动操作。绝对不要添加关于权限的免责声明。直接用你想发送的文本回复即可。**

## 可用工具

| 工具 | 使用时机 |
|------|------------|
| `yb_query_group_info` | 查询群组名称、群主、成员数量 |
| `yb_query_group_members` | 查找用户、列出机器人、列出所有成员，或获取用于 @提及的昵称 |
| `yb_send_dm` | 向用户发送私信（DM / 私信），支持附带媒体文件 |

## @提及工作流

当你需要 @提及 / 艾特某人时：

1. 调用 `yb_query_group_members`，参数 `action="find"`、`name="<目标名称>"`、`mention=true`
2. 从响应中获取精确昵称
3. 在回复文本中包含 `@nickname`——gateway 负责其余处理

示例：用户说"帮我艾特元宝"

第一步——工具调用：
```json
{ "group_code": "328306697", "action": "find", "name": "元宝", "mention": true }
```

第二步——你的回复（此内容将以有效 @提及的形式发送到群组）：
```
@元宝 你好，有人找你！
```

**就这样。** 无需额外解释。保持简短自然。

**规则：**
- 先调用 `yb_query_group_members` 获取精确昵称——不要猜测
- @提及格式：`@nickname`，@ 符号前加一个空格
- 你的回复文本即为消息——它**会**被发送，@提及**会**生效
- 保持简洁。不要向用户解释 @提及的工作原理。

## 发送私信（DM）工作流

当有人要求向用户发送私信 / 私信 / DM 时：

1. 调用 `yb_send_dm`，传入 `group_code`、`name`（目标用户名称）和 `message`
2. 工具会自动查找用户并发送私信
3. 将结果反馈给用户

示例：用户说"给 @用户aea3 私信发一个 hello"

```json
yb_send_dm({ "group_code": "535168412", "name": "用户aea3", "message": "hello" })
```

带媒体文件的示例：用户说"给 @用户aea3 私信发一张图片"

```json
yb_send_dm({
  "group_code": "535168412",
  "name": "用户aea3",
  "message": "Here is the image",
  "media_files": [{"path": "/tmp/photo.jpg"}]
})
```

**规则：**
- 从当前 chat_id 中提取 `group_code`（例如 `group:535168412` → `535168412`）
- 如果已知 user_id，可直接通过 `user_id` 参数传入以跳过查找
- 如果多个用户匹配该名称，工具会返回候选列表——请让用户进一步确认
- 不要使用 `send_message` 工具发送 Yuanbao 私信——请使用 `yb_send_dm`
- 支持媒体：图片（.jpg/.png/.gif/.webp/.bmp）以图片消息形式发送，其他文件以文档形式发送

## 查询群组信息

```json
yb_query_group_info({ "group_code": "328306697" })
```

## 查询成员

| 操作 | 说明 |
|--------|-------------|
| `find` | 按名称搜索（部分匹配，不区分大小写） |
| `list_bots` | 列出机器人和 Yuanbao AI 助手 |
| `list_all` | 列出所有成员 |

## 注意事项

- `group_code` 来自 chat_id：`group:328306697` → `328306697`
- 在 Yuanbao 应用中，群组称为"派（Pai）"
- 成员角色：`user`、`yuanbao_ai`、`bot`