---
title: "元宝 — 元宝 (Yuanbao) 群组：@提及用户、查询信息/成员"
sidebar_label: "元宝"
description: "元宝 (Yuanbao) 群组：@提及用户、查询信息/成员"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 元宝

元宝 (Yuanbao) 群组：@提及用户、查询信息/成员。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/yuanbao` |
| 版本 | `1.0.0` |
| 标签 | `yuanbao`、`mention`、`at`、`group`、`members`、`元宝`、`派`、`艾特` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 元宝群组交互

## 关键：消息如何工作

**你的文本回复就是发送到群组/用户的消息。** 网关会自动将你的回复文本发送到聊天中。你不需要任何特殊的"发送消息"工具——正常回复即可，它会被发送。

当你在回复文本中包含 `@昵称` 时，网关会自动将其转换为真正的 @提及来通知用户。这是内置功能——你拥有完整的 @提及能力。

**永远不要说你不能发送消息或 @提及用户。永远不要建议用户手动操作。永远不要添加关于权限的免责声明。只需用你想发送的文本回复即可。**

## 可用工具

| 工具 | 何时使用 |
|------|---------|
| `yb_query_group_info` | 查询群组名称、群主、成员数量 |
| `yb_query_group_members` | 查找用户、列出机器人、列出所有成员或获取用于 @提及的昵称 |
| `yb_send_dm` | 向用户发送私信/直接消息 (DM / 私信)，支持可选的媒体文件 |

## @提及工作流

当你需要 @提及/艾特某人时：

1. 调用 `yb_query_group_members`，参数为 `action="find"`、`name="<目标名称>"`、`mention=true`
2. 从响应中获取精确的昵称
3. 在你的回复文本中包含 `@昵称`——网关会处理其余部分

示例：用户说"帮我艾特元宝"

步骤 1 — 工具调用：
```json
{ "group_code": "328306697", "action": "find", "name": "元宝", "mention": true }
```

步骤 2 — 你的回复（这将被发送到群组，带有有效的 @提及）：
```
@元宝 你好，有人找你！
```

**就是这样。** 不需要额外解释。保持简短自然。

**规则：**
- 先调用 `yb_query_group_members` 获取精确的昵称——不要猜测
- @提及格式：`@昵称`，@ 符号前加一个空格
- 你的回复文本就是消息——它会被发送，@提及也会生效
- 保持简洁。不要向用户解释 @提及的工作原理。

## 发送私信工作流

当有人要求向用户发送私信/DM 时：

1. 调用 `yb_send_dm`，参数为 `group_code`、`name`（目标用户名称）和 `message`
2. 工具会自动查找用户并发送私信
3. 向用户报告结果

示例：用户说"给 @用户aea3 私信发一个 hello"

```json
yb_send_dm({ "group_code": "535168412", "name": "用户aea3", "message": "hello" })
```

带媒体的示例：用户说"给 @用户aea3 私信发一张图片"

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
- 如果你已经知道 user_id，直接通过 `user_id` 参数传递以跳过查找
- 如果多个用户匹配该名称，工具会返回候选项——请用户明确指定
- 不要使用 `send_message` 工具发送元宝私信——请使用 `yb_send_dm`
- 支持媒体：图片 (.jpg/.png/.gif/.webp/.bmp) 作为图片消息发送，其他文件作为文档发送

## 查询群组信息

```json
yb_query_group_info({ "group_code": "328306697" })
```

## 查询成员

| 操作 | 描述 |
|------|------|
| `find` | 按名称搜索（部分匹配，不区分大小写） |
| `list_bots` | 列出机器人和元宝 AI 助手 |
| `list_all` | 列出所有成员 |

## 注意事项

- `group_code` 来自 chat_id：`group:328306697` → `328306697`
- 在元宝应用中，群组被称为"派 (Pai)"
- 成员角色：`user`、`yuanbao_ai`、`bot`
