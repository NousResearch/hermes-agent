---
title: "Notion — 通过 curl 使用 Notion API：页面、数据库、块、搜索"
sidebar_label: "Notion"
description: "通过 curl 使用 Notion API：页面、数据库、块、搜索"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Notion

通过 curl 使用 Notion API：页面、数据库、块、搜索。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/notion` |
| 版本 | `1.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 标签 | `Notion`, `生产力`, `笔记`, `数据库`, `API` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Notion API

通过 curl 使用 Notion API 创建、读取、更新页面、数据库（数据源）和块。无需额外工具 — 只需 curl 和 Notion API 密钥。

## 前提条件

1. 在 https://notion.so/my-integrations 创建集成
2. 复制 API 密钥（以 `ntn_` 或 `secret_` 开头）
3. 存储在 `~/.hermes/.env` 中：
   ```
   NOTION_API_KEY=ntn_your_key_here
   ```
4. **重要：** 在 Notion 中与你的集成共享目标页面/数据库（点击"..."→"连接到"→你的集成名称）

## API 基础

所有请求使用此模式：

```bash
curl -s -X GET "https://api.notion.com/v1/..." \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json"
```

`Notion-Version` 头是必需的。此技能使用 `2025-09-03`（最新版）。在此版本中，数据库在 API 中称为"data sources"。

## 常见操作

### 搜索

```bash
curl -s -X POST "https://api.notion.com/v1/search" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"query": "页面标题"}'
```

### 获取页面

```bash
curl -s "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 获取页面内容（块）

```bash
curl -s "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 在数据库中创建页面

```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"database_id": "xxx"},
    "properties": {
      "Name": {"title": [{"text": {"content": "新项目"}}]},
      "Status": {"select": {"name": "Todo"}}
    }
  }'
```

### 查询数据库

```bash
curl -s -X POST "https://api.notion.com/v1/data_sources/{data_source_id}/query" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"property": "Status", "select": {"equals": "Active"}},
    "sorts": [{"property": "Date", "direction": "descending"}]
  }'
```

### 创建数据库

```bash
curl -s -X POST "https://api.notion.com/v1/data_sources" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"page_id": "xxx"},
    "title": [{"text": {"content": "我的数据库"}}],
    "properties": {
      "Name": {"title": {}},
      "Status": {"select": {"options": [{"name": "Todo"}, {"name": "Done"}]}},
      "Date": {"date": {}}
    }
  }'
```

### 更新页面属性

```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"properties": {"Status": {"select": {"name": "Done"}}}}'
```

### 向页面添加内容

```bash
curl -s -X PATCH "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "children": [
      {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "来自 Hermes 的问候！"}}]}}
    ]
  }'
```

## 属性类型

数据库项的常用属性格式：

- **标题：** `{"title": [{"text": {"content": "..."}}]}`
- **富文本：** `{"rich_text": [{"text": {"content": "..."}}]}`
- **选择：** `{"select": {"name": "选项"}}`
- **多选：** `{"multi_select": [{"name": "A"}, {"name": "B"}]}`
- **日期：** `{"date": {"start": "2026-01-15", "end": "2026-01-16"}}`
- **复选框：** `{"checkbox": true}`
- **数字：** `{"number": 42}`
- **URL：** `{"url": "https://..."}`
- **邮箱：** `{"email": "user@example.com"}`
- **关联：** `{"relation": [{"id": "page_id"}]}`

## API 版本 2025-09-03 的关键差异

- **数据库 → 数据源：** 使用 `/data_sources/` 端点进行查询和检索
- **两个 ID：** 每个数据库同时拥有 `database_id` 和 `data_source_id`
  - 创建页面时使用 `database_id`（`parent: {"database_id": "..."}`）
  - 查询时使用 `data_source_id`（`POST /v1/data_sources/{id}/query`）
- **搜索结果：** 数据库以 `"object": "data_source"` 返回，带有其 `data_source_id`

## 注意

- 页面/数据库 ID 为 UUID（带或不带连字符）
- 速率限制：约每秒 3 个请求
- API 无法设置数据库视图过滤器 — 仅限 UI 操作
- 创建数据源时使用 `is_inline: true` 将其嵌入页面
- 在 curl 中添加 `-s` 标志以抑制进度条（Hermes 输出更整洁）
- 通过 `jq` 管道输出以获得可读的 JSON：`... | jq '.results[0].properties'`
