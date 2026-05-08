---
title: "Airtable — 通过 curl 使用 Airtable REST API"
sidebar_label: "Airtable"
description: "通过 curl 使用 Airtable REST API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Airtable

通过 curl 使用 Airtable REST API。记录增删改查、过滤、upsert。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/airtable` |
| 版本 | `1.1.0` |
| 作者 | community |
| 许可证 | MIT |
| 标签 | `Airtable`, `生产力`, `数据库`, `API` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Airtable — 基地、表格与记录

直接通过 `curl` 使用 Airtable 的 REST API 和 `terminal` 工具。无需 MCP 服务器、无需 OAuth 流程、无需 Python SDK — 只需 `curl` 和个人访问令牌。

## 前提条件

1. 在 https://airtable.com/create/tokens 创建**个人访问令牌 (PAT)**（令牌以 `pat...` 开头）。
2. 授予这些权限（最低要求）：
   - `data.records:read` — 读取行
   - `data.records:write` — 创建 / 更新 / 删除行
   - `schema.bases:read` — 列出基地和表格
3. **重要：** 在同一令牌 UI 中，将你要访问的每个基地添加到令牌的**访问**列表中。PAT 按基地限定范围 — 有效令牌在错误的基地上返回 `403`。
4. 将令牌存储在 `~/.hermes/.env` 中（或通过 `hermes setup`）：
   ```
   AIRTABLE_API_KEY=pat_your_token_here
   ```

> 注意：旧版 `key...` API 密钥已于 2024 年 2 月弃用。现在只有 PAT 和 OAuth 令牌有效。

## API 基础

- **端点：** `https://api.airtable.com/v0`
- **认证头：** `Authorization: Bearer $AIRTABLE_API_KEY`
- **所有请求** 使用 JSON（任何 POST/PATCH/PUT 请求体使用 `Content-Type: application/json`）。
- **对象 ID：** 基地 `app...`，表格 `tbl...`，记录 `rec...`，字段 `fld...`。ID 永不改变；名称可能改变。在自动化中优先使用 ID。
- **速率限制：** 每个基地每秒 5 个请求。`429` → 退避。单个基地的突发请求将被限流。

基础 curl 模式：
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=5" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

`-s` 抑制 curl 的进度条 — 每次调用都保持设置，以便工具输出保持干净供 Hermes 使用。通过 `python3 -m json.tool`（始终可用）或 `jq`（如已安装）管道输出以获得可读的 JSON。

## 字段类型（请求体格式）

| 字段类型 | 写入格式 |
|---|---|
| 单行文本 | `"Name": "hello"` |
| 长文本 | `"Notes": "multi\nline"` |
| 数字 | `"Score": 42` |
| 复选框 | `"Done": true` |
| 单选 | `"Status": "Todo"`（名称必须已存在，除非 `typecast: true`） |
| 多选 | `"Tags": ["urgent", "bug"]` |
| 日期 | `"Due": "2026-04-01"` |
| 日期时间 (UTC) | `"At": "2026-04-01T14:30:00.000Z"` |
| URL / 邮箱 / 电话 | `"Link": "https://…"` |
| 附件 | `"Files": [{"url": "https://…"}]`（Airtable 获取并重新托管） |
| 关联记录 | `"Owner": ["recXXXXXXXXXXXXXX"]`（记录 ID 数组） |
| 用户 | `"AssignedTo": {"id": "usrXXXXXXXXXXXXXX"}` |

在创建/更新请求体的顶层传递 `"typecast": true`，让 Airtable 自动转换值（例如即时创建新的选择选项，将 `"42"` → `42`）。

## 常见查询

### 列出令牌可见的基地
```bash
curl -s "https://api.airtable.com/v0/meta/bases" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 列出基地的表格 + 模式
```bash
curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
在修改之前使用此命令 — 确认确切的字段名称和 ID，显示选择字段的 `options.choices`，并显示主字段名称。

### 列出记录（前 10 条）
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=10" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 获取单条记录
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 过滤记录（filterByFormula）
Airtable 公式必须进行 URL 编码。让 Python 标准库来完成 — 永远不要手动编码：
```bash
FORMULA="{Status}='Todo'"
ENC=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$FORMULA")
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?filterByFormula=$ENC&maxRecords=20" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

常用公式模式：
- 精确匹配：`{Email}='user@example.com'`
- 包含：`FIND('bug', LOWER({Title}))`
- 多条件：`AND({Status}='Todo', {Priority}='High')`
- 或：`OR({Owner}='alice', {Owner}='bob')`
- 非空：`NOT({Assignee}='')`
- 日期比较：`IS_AFTER({Due}, TODAY())`

### 排序 + 选择特定字段
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?sort%5B0%5D%5Bfield%5D=Priority&sort%5B0%5D%5Bdirection%5D=asc&fields%5B%5D=Name&fields%5B%5D=Status" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
查询参数中的方括号必须进行 URL 编码（`%5B` / `%5D`）。

### 使用命名视图
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?view=Grid%20view&maxRecords=50" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
视图在服务端应用其保存的过滤 + 排序。

## 常见变更操作

### 创建记录
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Name":"New task","Status":"Todo","Priority":"High"}}' | python3 -m json.tool
```

### 一次调用创建最多 10 条记录
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "typecast": true,
    "records": [
      {"fields": {"Name": "Task A", "Status": "Todo"}},
      {"fields": {"Name": "Task B", "Status": "In progress"}}
    ]
  }' | python3 -m json.tool
```
批量端点每个请求最多 **10 条记录**。对于更大的插入，以 10 条为一批循环并短暂休眠以遵守 5 请求/秒/基地的速率限制。

### 更新记录（PATCH — 合并，保留未更改的字段）
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}' | python3 -m json.tool
```

### 通过合并字段进行 Upsert（无需 ID）
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "performUpsert": {"fieldsToMergeOn": ["Email"]},
    "records": [
      {"fields": {"Email": "user@example.com", "Status": "Active"}}
    ]
  }' | python3 -m json.tool
```
`performUpsert` 创建合并字段值为新的记录，修补合并字段值已存在的记录。适用于幂等同步。

### 删除记录
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 一次调用删除最多 10 条记录
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE?records%5B%5D=rec1&records%5B%5D=rec2" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

## 分页

列表端点每页最多返回 **100 条记录**。如果响应包含 `"offset": "..."`，在下一次调用时传回该值。循环直到该字段不存在：

```bash
OFFSET=""
while :; do
  URL="https://api.airtable.com/v0/$BASE_ID/$TABLE?pageSize=100"
  [ -n "$OFFSET" ] && URL="$URL&offset=$OFFSET"
  RESP=$(curl -s "$URL" -H "Authorization: Bearer $AIRTABLE_API_KEY")
  echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(r["id"], r["fields"].get("Name","")) for r in d["records"]]'
  OFFSET=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("offset",""))')
  [ -z "$OFFSET" ] && break
done
```

## 典型 Hermes 工作流

1. **确认认证。** `curl -s -o /dev/null -w "%{http_code}\n" https://api.airtable.com/v0/meta/bases -H "Authorization: Bearer $AIRTABLE_API_KEY"` — 期望 `200`。
2. **找到基地。** 列出基地（上面的步骤）或如果令牌缺少 `schema.bases:read` 权限，直接向用户索取 `app...` ID。
3. **检查模式。** `GET /v0/meta/bases/$BASE_ID/tables` — 在修改任何内容之前，在本地会话中缓存确切的字段名称和主字段名称。
4. **先读后写。** 对于"更新 Y 中的 X"，先使用 `filterByFormula` 解析 `rec...` ID，然后 `PATCH /v0/$BASE_ID/$TABLE/$RECORD_ID`。永远不要猜测记录 ID。
5. **批量写入。** 将相关的创建操作合并为一个 10 条记录的 POST，以保持在 5 请求/秒的预算内。
6. **破坏性操作。** 删除操作无法通过 API 撤销。如果用户说"删除所有 X"，先回显过滤条件 + 记录数量并在执行前确认。

## 注意事项

- **`filterByFormula` 必须进行 URL 编码。** 包含空格或非 ASCII 的字段名也需要编码（`{My Field}` → `%7BMy%20Field%7D`）。使用 Python 标准库（上面的模式）— 永远不要手动转义。
- **空字段在响应中省略。** 缺少 `"Assignee"` 键并不意味着该字段不存在 — 意味着此记录的值为空。在断定字段缺失之前检查模式（步骤 3）。
- **PATCH vs PUT。** `PATCH` 将提供的字段合并到记录中。`PUT` 完全替换记录并清除你未包含的任何字段。默认使用 `PATCH`。
- **单选选项必须已存在。** 当 `Shipping` 不在字段选项列表中时写入 `"Status": "Shipping"` 会报 `INVALID_MULTIPLE_CHOICE_OPTIONS` 错误，除非你传递 `"typecast": true`（自动创建选项）。
- **令牌按基地限定范围。** 一个基地返回 `403` 而另一个正常，意味着令牌的访问列表不包含该基地 — 不是权限或认证问题。引导用户到 https://airtable.com/create/tokens 授权。
- **速率限制按基地计算，而非按令牌。** `baseA` 上 5 请求/秒和 `baseB` 上 5 请求/秒是可以的；仅 `baseA` 上 6 请求/秒就会被限流。监控 `429` 响应上的 `Retry-After` 头。

## Hermes 重要提示

- **始终使用 `terminal` 工具配合 `curl`。** 不要使用 `web_extract`（无法发送认证头）或 `browser_navigate`（需要 UI 认证且速度慢）。
- **`AIRTABLE_API_KEY` 从 `~/.hermes/.env` 自动流入子进程**，当此技能加载时 — 无需在每次 `curl` 调用前重新导出。
- **小心转义公式中的花括号。** 在 heredoc 请求体中，`{Status}` 是字面值。在 shell 参数中，`{Status}` 在 `{...}` 花括号展开上下文之外是安全的 — 但在拼接到 URL 之前，通过 `python3 urllib.parse.quote` 传递动态字符串。
- **使用 `python3 -m json.tool` 美化输出**（始终可用）而非 `jq`（可选）。仅在需要过滤/投影时才使用 `jq`。
- **分页是按页的，不是全局的。** Airtable 的 100 条记录上限是硬性限制；无法提高。使用 `offset` 循环直到该字段不存在。
- **阅读非 2xx 响应的 `errors` 数组** — Airtable 返回结构化的错误代码如 `AUTHENTICATION_REQUIRED`、`INVALID_PERMISSIONS`、`MODEL_ID_NOT_FOUND`、`INVALID_MULTIPLE_CHOICE_OPTIONS`，告诉你具体出了什么问题。
