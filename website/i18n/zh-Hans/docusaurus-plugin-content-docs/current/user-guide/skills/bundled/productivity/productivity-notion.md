---
title: "Notion — Notion API + ntn CLI：页面、数据库、Markdown、Workers"
sidebar_label: "Notion"
description: "Notion API + ntn CLI：页面、数据库、Markdown、Workers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Notion

Notion API + ntn CLI：页面、数据库、Markdown、Workers。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/notion` |
| 版本 | `2.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Notion`, `Productivity`, `Notes`, `Database`, `API`, `CLI`, `Workers` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Notion

通过两种方式与 Notion 交互。两种方式使用同一个集成 token——根据可用情况选择。

◆ **`ntn` CLI** — Notion 官方 CLI。语法更简洁，支持单行文件上传，Workers 必须使用此方式。截至 2026 年 5 月仅支持 macOS + Linux（Windows 支持"即将推出"）。**已安装时为默认方式。**
◆ **HTTP + curl** — 全平台可用，包括 Windows。**`ntn` 未安装时的默认回退方式。**

## 配置

### 1. 获取集成 token（两种方式均需要）

1. 在 https://notion.so/my-integrations 创建集成
2. 复制 API 密钥（以 `ntn_` 或 `secret_` 开头）
3. 存储到 `~/.hermes/.env`：
   ```
   NOTION_API_KEY=ntn_your_key_here
   ```
4. **在 Notion 中将目标页面/数据库共享给该集成：** 页面菜单 `...` → `Connect to` → 你的集成名称。若未执行此步骤，即使页面存在，API 也会返回 404。

### 2. 安装 `ntn`（macOS / Linux 上的首选方式）

```bash
# 推荐方式
curl -fsSL https://ntn.dev | bash

# 或通过 npm 安装（需要 Node 22+，npm 10+）
npm install --global ntn

ntn --version    # 验证安装
```

**跳过 `ntn login`——改用集成 token。** 此方式支持无头运行，无需浏览器：
```bash
export NOTION_API_TOKEN=$NOTION_API_KEY      # ntn 读取 NOTION_API_TOKEN
export NOTION_KEYRING=0                       # 不尝试使用系统密钥链
```

将上述 export 添加到你的 shell 配置文件（或 `~/.hermes/.env`），使每个会话都能继承这些变量。

### 3. 运行时选择路径

```bash
if command -v ntn >/dev/null 2>&1; then
  # 使用 ntn
else
  # 回退到 curl
fi
```

Windows 用户：在原生 `ntn` 发布之前完全跳过第 2 步——Path B 可正常使用。如果现在就想要 CLI 体验，可在 WSL2 中安装 `ntn`。

## API 基础

所有 HTTP 请求均需携带 `Notion-Version: 2025-09-03`。`ntn` 会自动处理此项。在此版本中，用户所称的"数据库"在 API 中称为 **data sources（数据源）**。

## Path A — `ntn` CLI（首选，macOS / Linux）

### 原始 API 调用（curl 的简写）
```bash
ntn api v1/users                                  # GET
ntn api v1/pages parent[page_id]=abc123 \         # POST，内联请求体
  properties[title][0][text][content]="Notes"
ntn api v1/pages/abc123 -X PATCH archived:=true   # PATCH；:= 表示非字符串类型（布尔/数字/null）
```

语法说明：
- `key=value` — 字符串字段
- `key[nested]=value` — 嵌套对象字段
- `key:=value` — 类型赋值（布尔值、数字、null、数组）

### 搜索
```bash
ntn api v1/search query="page title"
```

### 读取页面元数据
```bash
ntn api v1/pages/{page_id}
```

### 以 Markdown 格式读取页面（适合 agent 使用）
```bash
ntn api v1/pages/{page_id}/markdown
```

### 以块（block）形式读取页面内容
```bash
ntn api v1/blocks/{page_id}/children
```

### 从 Markdown 创建页面
```bash
ntn api v1/pages \
  parent[page_id]=xxx \
  properties[title][0][text][content]="Notes from meeting" \
  markdown="# Agenda

- Q3 roadmap
- Hiring"
```

### 用 Markdown 更新页面
```bash
ntn api v1/pages/{page_id}/markdown -X PATCH \
  markdown="## Update

Shipped the prototype."
```

### 查询数据库（data source）
```bash
ntn api v1/data_sources/{data_source_id}/query -X POST \
  filter[property]=Status filter[select][equals]=Active
```

对于包含 `sorts`、多个过滤条件或复合逻辑的复杂查询，通过管道传入 JSON：
```bash
echo '{"filter": {"property": "Status", "select": {"equals": "Active"}}, "sorts": [{"property": "Date", "direction": "descending"}]}' | \
  ntn api v1/data_sources/{data_source_id}/query -X POST --json -
```

### 文件上传（单行命令——CLI 最大优势）
```bash
ntn files create < photo.png
ntn files create --external-url https://example.com/photo.png
ntn files list
```

对比三步 HTTP 流程（创建上传 → PUT 字节 → 引用）。

### 常用环境变量
| 变量 | 作用 |
|---|---|
| `NOTION_API_TOKEN` | 认证 token（覆盖密钥链）——设置为你的集成 token |
| `NOTION_KEYRING=0` | 使用 `~/.config/notion/auth.json` 存储凭据，而非系统密钥链 |
| `NOTION_WORKSPACE_ID` | 跳过工作区选择提示 |

## Path B — HTTP + curl（跨平台，Windows 默认方式）

所有请求遵循以下模式：

```bash
curl -s -X GET "https://api.notion.com/v1/..." \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json"
```

Windows 10+ 自带的 `curl` 可直接使用。PowerShell 用户也可使用 `Invoke-RestMethod`。

### 搜索
```bash
curl -s -X POST "https://api.notion.com/v1/search" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"query": "page title"}'
```

### 读取页面元数据
```bash
curl -s "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 以 Markdown 格式读取页面（适合 agent 使用）

比块 JSON 更易于输入模型处理。

```bash
curl -s "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 以块形式读取页面内容（需要结构化数据时使用）
```bash
curl -s "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 从 Markdown 创建页面

`POST /v1/pages` 接受 `markdown` 请求体参数。

```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"page_id": "xxx"},
    "properties": {"title": [{"text": {"content": "Notes from meeting"}}]},
    "markdown": "# Agenda\n\n- Q3 roadmap\n- Hiring\n\n## Decisions\n- Ship MVP Friday"
  }'
```

### 用 Markdown 更新页面
```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"markdown": "## Update\n\nShipped the prototype."}'
```

### 在数据库中创建页面（带类型属性）
```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"database_id": "xxx"},
    "properties": {
      "Name": {"title": [{"text": {"content": "New Item"}}]},
      "Status": {"select": {"name": "Todo"}}
    }
  }'
```

### 查询数据库（data source）
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
    "title": [{"text": {"content": "My Database"}}],
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

### 向页面追加块
```bash
curl -s -X PATCH "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "children": [
      {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "Hello from Hermes!"}}]}}
    ]
  }'
```

### 文件上传（三步流程）
```bash
# 1. 创建上传
curl -s -X POST "https://api.notion.com/v1/file_uploads" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"filename": "photo.png", "content_type": "image/png"}'

# 2. 将字节 PUT 到上面返回的 upload_url
curl -s -X PUT "{upload_url}" --data-binary @photo.png

# 3. 在页面/块 payload 中引用 {file_upload_id}
```

## 属性类型

数据库条目的常用属性格式：

- **标题（Title）：** `{"title": [{"text": {"content": "..."}}]}`
- **富文本（Rich text）：** `{"rich_text": [{"text": {"content": "..."}}]}`
- **单选（Select）：** `{"select": {"name": "Option"}}`
- **多选（Multi-select）：** `{"multi_select": [{"name": "A"}, {"name": "B"}]}`
- **日期（Date）：** `{"date": {"start": "2026-01-15", "end": "2026-01-16"}}`
- **复选框（Checkbox）：** `{"checkbox": true}`
- **数字（Number）：** `{"number": 42}`
- **URL：** `{"url": "https://..."}`
- **邮箱（Email）：** `{"email": "user@example.com"}`
- **关联（Relation）：** `{"relation": [{"id": "page_id"}]}`

## API 版本 2025-09-03 — 数据库与 Data Sources

- **数据库已更名为 data sources。** 查询和检索请使用 `/data_sources/` 端点。
- **每个数据库有两个 ID：** `database_id` 和 `data_source_id`。
  - 创建页面时使用 `database_id`：`parent: {"database_id": "..."}`
  - 查询时使用 `data_source_id`：`POST /v1/data_sources/{id}/query`
- 搜索返回的数据库对象类型为 `"object": "data_source"`，包含 `data_source_id` 字段。

## Notion Workers（高级功能，需要 `ntn`）

Workers 是由 Notion 托管的 TypeScript 程序。一个 worker 可以暴露以下任意组合：
- **Syncs（同步）** — 按计划（默认 30 分钟）从外部 API 拉取数据到 Notion 数据库。
- **Tools（工具）** — 在 Notion 的 Custom Agents 中作为可调用工具出现。
- **Webhooks** — 接收来自外部服务（GitHub、Stripe 等）的 HTTP 事件并在 Notion 中执行操作。

**套餐/平台限制：**
- CLI 在所有套餐上均可使用。**部署 Workers 需要 Business 或 Enterprise 套餐。**
- 截至 2026 年 5 月，`ntn` 仅支持 macOS/Linux。Windows 用户需使用 WSL2 或等待原生支持。
- 2026 年 8 月 11 日前免费；之后按 Notion 积分计费。

### 最简 Worker

```bash
ntn workers new my-worker      # 脚手架
cd my-worker
# 编辑 src/index.ts
ntn workers deploy --name my-worker
```

`src/index.ts`：
```typescript
import { Worker } from "@notionhq/workers";

const worker = new Worker();
export default worker;

worker.tool("greet", {
  title: "Greet a User",
  description: "Returns a friendly greeting",
  inputSchema: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
  execute: async ({ name }) => `Hello, ${name}!`,
});
```

### Webhook 能力

```typescript
worker.webhook("onGithubPush", {
  title: "GitHub Push Handler",
  execute: async (events, { notion }) => {
    for (const event of events) {
      // event.body, event.rawBody（用于签名验证），event.headers
      console.log("got delivery", event.deliveryId);
    }
  },
});
```

部署后：`ntn workers webhooks list` 显示 Notion 生成的 URL。将该 URL 视为机密——除非添加签名验证，否则任何人都可以向其 POST 事件。

### Worker 生命周期命令

```bash
ntn workers deploy
ntn workers list
ntn workers exec <capability-key> -d '{"name": "world"}'
ntn workers sync trigger <key>            # 立即运行同步
ntn workers sync pause <key>
ntn workers env set GITHUB_WEBHOOK_SECRET=...
ntn workers runs list                     # 最近的调用记录
ntn workers runs logs <run-id>
ntn workers webhooks list
```

需要构建 Worker 时，使用 `ntn workers new` 创建脚手架，在 `src/index.ts` 中编写代码，通过 `ntn workers env set` 设置密钥，然后部署。Notion 文档 https://developers.notion.com/workers 涵盖完整 API 接口。

## Notion 风格 Markdown（用于 `/markdown` 端点）

标准 CommonMark 加上用于 Notion 特定块的类 XML 标签。缩进使用**制表符（tab）**。

**CommonMark 之外的块：**
```
<callout icon="🎯" color="blue_bg">
	Ship the MVP by **Friday**.
</callout>

<details color="gray">
<summary>Toggle title</summary>
	Children indented one tab
</details>

<columns>
	<column>Left side</column>
	<column>Right side</column>
</columns>

<table_of_contents color="gray"/>
```

**内联：**
- 提及（Mention）：`<mention-user url="..."/>`、`<mention-page url="...">Title</mention-page>`、`<mention-date start="2026-05-15"/>`
- 下划线：`<span underline="true">text</span>`
- 颜色：`<span color="blue">text</span>`，或块级别在第一行使用 `{color="blue"}`
- 数学公式：内联 `$x^2$`，块级 `$$ ... $$`
- 引用：`[^https://example.com]`

**颜色：** `gray brown orange yellow green blue purple pink red`，以及带 `*_bg` 后缀的背景色变体。

5/6 级标题会折叠为 H4。多个连续 `>` 行渲染为独立引用块——在单个 `>` 内使用 `<br>` 实现多行引用。

## 选择合适的路径

| 任务 | macOS / Linux | Windows |
|---|---|---|
| 读写页面、搜索、查询数据库 | `ntn api ...` | curl |
| 读取页面供 agent 摘要 | `ntn api v1/pages/{id}/markdown` | curl `/markdown` 端点 |
| 上传文件 | `ntn files create < file` | 三步 HTTP 流程 |
| 一次性 API 探索 | `ntn api ...` | curl |
| 构建由 Notion 托管的同步/webhook/agent 工具 | `ntn workers ...` | WSL2 + `ntn workers ...` |

## 注意事项

- 页面/数据库 ID 为 UUID 格式（带或不带连字符均可接受）。
- 速率限制：平均约 3 次请求/秒。CLI 不会绕过此限制。
- API 无法设置数据库**视图**过滤器——该功能仅限 UI 操作。
- 创建 data sources 时使用 `"is_inline": true` 可将其嵌入页面。
- 始终为 curl 传入 `-s` 以抑制进度条（使 agent 输出更整洁）。
- 读取数据时通过 `jq` 管道处理：`... | jq '.results[0].properties'`。
- Notion 现已推出 MCP 服务器（`Notion MCP`，在数据库操作上比上一版本的 token 效率提升约 91%）——如需在会话中进行流式 Notion 访问，可通过 Hermes 的 MCP 支持接入，但上述路径已足以应对大多数一次性任务。