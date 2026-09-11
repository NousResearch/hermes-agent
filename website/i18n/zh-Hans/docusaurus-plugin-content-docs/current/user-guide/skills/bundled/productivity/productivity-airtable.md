---
title: "Airtable — 通过 curl 调用 Airtable REST API"
sidebar_label: "Airtable"
description: "通过 curl 调用 Airtable REST API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Airtable

通过 curl 调用 Airtable REST API。支持记录的增删改查、过滤和 upsert 操作。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/airtable` |
| 版本 | `1.1.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Airtable`, `Productivity`, `Database`, `API` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Airtable — Bases、Tables 与 Records

通过 `terminal` 工具，使用 `curl` 直接调用 Airtable 的 REST API。无需 MCP server，无需 OAuth 流程，无需 Python SDK——只需 `curl` 和一个个人访问令牌（PAT）。

## 前置条件

1. 在 https://airtable.com/create/tokens 创建一个**个人访问令牌（PAT）**（令牌以 `pat...` 开头）。
2. 授予以下权限范围（最低要求）：
   - `data.records:read` — 读取行
   - `data.records:write` — 创建 / 更新 / 删除行
   - `schema.bases:read` — 列出 bases 和 tables
3. **重要：** 在同一令牌 UI 中，将你需要访问的每个 base 添加到令牌的 **Access** 列表中。PAT 是按 base 划定范围的——有效令牌若未授权对应 base 会返回 `403`。
4. 将令牌存储在 `~/.hermes/.env` 中（或通过 `hermes setup` 配置）：
   ```
   AIRTABLE_API_KEY=pat_your_token_here
   ```

> 注意：旧版 `key...` API 密钥已于 2024 年 2 月弃用。目前仅支持 PAT 和 OAuth 令牌。

## API 基础

- **端点：** `https://api.airtable.com/v0`
- **认证头：** `Authorization: Bearer $AIRTABLE_API_KEY`
- **所有请求** 使用 JSON（POST/PATCH/PUT 请求体需设置 `Content-Type: application/json`）。
- **对象 ID：** base 为 `app...`，table 为 `tbl...`，record 为 `rec...`，field 为 `fld...`。ID 永不变更；名称可能变更。自动化流程中优先使用 ID。
- **速率限制：** 每个 base 每秒 5 次请求。收到 `429` 时需退避重试。单个 base 的突发请求会被限流。

基础 curl 模式：
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=5" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

`-s` 会抑制 curl 的进度条——每次调用都保持此设置，以确保工具输出对 Hermes 保持整洁。通过 `python3 -m json.tool`（始终可用）或 `jq`（若已安装）管道输出以获得可读的 JSON。

## 字段类型（请求体格式）

| 字段类型 | 写入格式 |
|---|---|
| 单行文本 | `"Name": "hello"` |
| 长文本 | `"Notes": "multi\nline"` |
| 数字 | `"Score": 42` |
| 复选框 | `"Done": true` |
| 单选 | `"Status": "Todo"`（选项名必须已存在，除非设置 `typecast: true`） |
| 多选 | `"Tags": ["urgent", "bug"]` |
| 日期 | `"Due": "2026-04-01"` |
| 日期时间（UTC） | `"At": "2026-04-01T14:30:00.000Z"` |
| URL / 邮箱 / 电话 | `"Link": "https://…"` |
| 附件 | `"Files": [{"url": "https://…"}]`（Airtable 会抓取并重新托管） |
| 关联记录 | `"Owner": ["recXXXXXXXXXXXXXX"]`（record ID 数组） |
| 用户 | `"AssignedTo": {"id": "usrXXXXXXXXXXXXXX"}` |

在创建/更新请求体的顶层传入 `"typecast": true`，可让 Airtable 自动强制转换值（例如动态创建新的单选选项，或将 `"42"` 转换为 `42`）。

## 常用查询

### 列出令牌可访问的 bases
```bash
curl -s "https://api.airtable.com/v0/meta/bases" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 列出某个 base 的 tables 及 schema
```bash
curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
在执行任何变更操作前先调用此接口——可确认精确的字段名和 ID，查看单选字段的 `options.choices`，并获取主字段名称。

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
Airtable 公式必须经过 URL 编码。使用 Python 标准库处理——切勿手动编码：
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

### 排序并选择特定字段
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
视图会在服务端应用其保存的过滤条件和排序规则。

## 常用变更操作

### 创建单条记录
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Name":"New task","Status":"Todo","Priority":"High"}}' | python3 -m json.tool
```

### 单次调用最多创建 10 条记录
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
批量端点每次请求上限为 **10 条记录**。对于更大批量的插入，需以 10 条为一批循环处理，并在每批之间短暂休眠，以遵守每 base 每秒 5 次的速率限制。

### 更新记录（PATCH——合并更新，保留未修改字段）
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}' | python3 -m json.tool
```

### 按合并字段 upsert（无需 ID）
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
`performUpsert` 会为合并字段值不存在的记录执行创建操作，为合并字段值已存在的记录执行更新操作。非常适合幂等同步场景。

### 删除单条记录
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 单次调用最多删除 10 条记录
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE?records%5B%5D=rec1&records%5B%5D=rec2" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

## 分页

列表端点每页最多返回 **100 条记录**。若响应中包含 `"offset": "..."`，需在下一次请求中传回该值。循环直至该字段不再出现：

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

## Hermes 典型工作流

1. **确认认证。** `curl -s -o /dev/null -w "%{http_code}\n" https://api.airtable.com/v0/meta/bases -H "Authorization: Bearer $AIRTABLE_API_KEY"` — 期望返回 `200`。
2. **找到 base。** 列出 bases（见上方步骤），或在令牌缺少 `schema.bases:read` 权限时直接向用户索取 `app...` ID。
3. **检查 schema。** `GET /v0/meta/bases/$BASE_ID/tables` — 在执行任何变更操作前，在会话中本地缓存精确的字段名和主字段名。
4. **写前先读。** 对于"更新满足条件 Y 的 X"类操作，先用 `filterByFormula` 解析出 `rec...` ID，再执行 `PATCH /v0/$BASE_ID/$TABLE/$RECORD_ID`。切勿猜测 record ID。
5. **批量写入。** 将相关的创建操作合并为一次 10 条记录的 POST 请求，以控制在每秒 5 次的速率预算内。
6. **破坏性操作。** 删除操作无法通过 API 撤销。若用户要求"删除所有 X"，先回显过滤条件和记录数量，确认后再执行。

## 注意事项

- **`filterByFormula` 必须进行 URL 编码。** 包含空格或非 ASCII 字符的字段名也需要编码（`{My Field}` → `%7BMy%20Field%7D`）。使用 Python 标准库（见上方模式）——切勿手动转义。
- **空字段不会出现在响应中。** 响应中缺少 `"Assignee"` 键并不意味着该字段不存在——而是表示该记录的值为空。在判断字段缺失之前，请先检查 schema（步骤 3）。
- **PATCH 与 PUT 的区别。** `PATCH` 将提供的字段合并到记录中。`PUT` 会完全替换记录，并清除所有未包含的字段。默认使用 `PATCH`。
- **单选选项必须已存在。** 若 `Shipping` 不在字段的选项列表中，写入 `"Status": "Shipping"` 会报错 `INVALID_MULTIPLE_CHOICE_OPTIONS`，除非传入 `"typecast": true`（会自动创建该选项）。
- **令牌的 base 范围限制。** 某个 base 返回 `403` 而其他 base 正常，说明该 base 未添加到令牌的 Access 列表中——而非权限范围或认证问题。请引导用户前往 https://airtable.com/create/tokens 授权。
- **速率限制是按 base 计算的，而非按令牌。** `baseA` 每秒 5 次、`baseB` 每秒 5 次是允许的；单独在 `baseA` 上每秒 6 次则会被限流。收到 `429` 时请监控 `Retry-After` 响应头。

## Hermes 重要说明

- **始终使用 `terminal` 工具配合 `curl`。** 不要使用 `web_extract`（无法发送认证头）或 `browser_navigate`（需要 UI 认证且速度慢）。
- **`AIRTABLE_API_KEY` 会在此 skill 加载时自动从 `~/.hermes/.env` 注入到子进程环境中**——每次 `curl` 调用前无需重新导出。
- **在公式中谨慎转义花括号。** 在 heredoc 请求体中，`{Status}` 是字面量。在 shell 参数中，`{Status}` 在 `{...}` 大括号展开上下文之外是安全的——但在拼接到 URL 之前，动态字符串应通过 `python3 urllib.parse.quote` 处理。
- **使用 `python3 -m json.tool` 格式化输出**（始终可用），而非 `jq`（可选）。仅在需要过滤/投影时才使用 `jq`。
- **分页是按页计算的，而非全局。** Airtable 的 100 条记录上限是硬性限制，无法调整。使用 `offset` 循环直至该字段不再出现。
- **读取非 2xx 响应中的 `errors` 数组**——Airtable 会返回结构化错误码，如 `AUTHENTICATION_REQUIRED`、`INVALID_PERMISSIONS`、`MODEL_ID_NOT_FOUND`、`INVALID_MULTIPLE_CHOICE_OPTIONS`，可精确定位问题所在。