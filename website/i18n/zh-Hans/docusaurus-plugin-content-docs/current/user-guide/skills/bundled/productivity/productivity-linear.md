---
title: "Linear — Linear: manage issues, projects, teams via GraphQL + curl"
sidebar_label: "Linear"
description: "Linear：通过 GraphQL + curl 管理 issues、项目和团队"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Linear

Linear：通过 GraphQL + curl 管理 issues、项目和团队。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/linear` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Linear`, `Project Management`, `Issues`, `GraphQL`, `API`, `Productivity` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Linear — Issue 与项目管理

直接通过 GraphQL API 使用 `curl` 管理 Linear 的 issues、项目和团队。无需 MCP server，无需 OAuth 流程，无需额外依赖。

## 配置

1. 从 **Linear 设置 > Account > Security & access > Personal API keys** 获取个人 API key（URL：https://linear.app/settings/account/security）。注意：组织级别的 *Settings > API* 页面仅显示 OAuth 应用和工作区成员 key，不显示个人 key。
2. 在环境中设置 `LINEAR_API_KEY`（通过 `hermes setup` 或你的环境配置）

## API 基础

- **端点：** `https://api.linear.app/graphql`（POST）
- **认证头：** `Authorization: $LINEAR_API_KEY`（API key 无需 "Bearer" 前缀）
- **所有请求均为 POST**，使用 `Content-Type: application/json`
- **UUID 和短标识符**（如 `ENG-123`）均可用于 `issue(id:)`

基础 curl 模式：
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ viewer { id name } }"}' | python3 -m json.tool
```

## Python 辅助脚本（更便捷的替代方案）

如需无需手写 GraphQL 的快速单行命令，此 skill 提供了一个基于标准库的 Python CLI，路径为 `scripts/linear_api.py`。零依赖，使用相同的认证方式（读取 `LINEAR_API_KEY`）。

```bash
SCRIPT=$(dirname "$(find ~/.hermes -path '*skills/productivity/linear/scripts/linear_api.py' 2>/dev/null | head -1)")/linear_api.py

python3 "$SCRIPT" whoami
python3 "$SCRIPT" list-teams
python3 "$SCRIPT" get-issue ENG-42
python3 "$SCRIPT" get-document 38359beef67c      # fetch a doc by slugId from the URL
python3 "$SCRIPT" raw 'query { viewer { name } }'
```

所有子命令：`whoami`、`list-teams`、`list-projects`、`list-states`、`list-issues`、`get-issue`、`search-issues`、`create-issue`、`update-issue`、`update-status`、`add-comment`、`list-documents`、`get-document`、`search-documents`、`raw`。运行时加 `--help` 查看参数说明。

适合使用脚本的场景：需要快速获取结果而不想编写 GraphQL。适合使用 curl 的场景：需要脚本未封装的查询，或需要内联组合过滤条件。

## 工作流状态

Linear 使用带有 `type` 字段的 `WorkflowState` 对象。**共 6 种状态类型：**

| 类型 | 描述 |
|------|-------------|
| `triage` | 待审核的新 issue |
| `backlog` | 已确认但尚未规划 |
| `unstarted` | 已规划/就绪但未开始 |
| `started` | 正在积极处理中 |
| `completed` | 已完成 |
| `canceled` | 不予处理 |

每个团队有其自己命名的状态（例如，"In Progress" 对应类型 `started`）。要更改 issue 的状态，需要目标状态的 `stateId`（UUID）——请先查询工作流状态。

**优先级值：** 0 = 无，1 = 紧急，2 = 高，3 = 中，4 = 低

## 常用查询

### 获取当前用户
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ viewer { id name email } }"}' | python3 -m json.tool
```

### 列出团队
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ teams { nodes { id name key } } }"}' | python3 -m json.tool
```

### 列出某团队的工作流状态
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ workflowStates(filter: { team: { key: { eq: \"ENG\" } } }) { nodes { id name type } } }"}' | python3 -m json.tool
```

### 列出 issues（前 20 条）
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issues(first: 20) { nodes { identifier title priority state { name type } assignee { name } team { key } url } pageInfo { hasNextPage endCursor } } }"}' | python3 -m json.tool
```

### 列出分配给我的 issues
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ viewer { assignedIssues(first: 25) { nodes { identifier title state { name type } priority url } } } }"}' | python3 -m json.tool
```

### 获取单个 issue（通过标识符如 ENG-123）
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issue(id: \"ENG-123\") { id identifier title description priority state { id name type } assignee { id name } team { key } project { name } labels { nodes { name } } comments { nodes { body user { name } createdAt } } url } }"}' | python3 -m json.tool
```

### 按文本搜索 issues
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issueSearch(query: \"bug login\", first: 10) { nodes { identifier title state { name } assignee { name } url } } }"}' | python3 -m json.tool
```

### 按状态类型过滤 issues
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issues(filter: { state: { type: { in: [\"started\"] } } }, first: 20) { nodes { identifier title state { name } assignee { name } } } }"}' | python3 -m json.tool
```

### 按团队和负责人过滤
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issues(filter: { team: { key: { eq: \"ENG\" } }, assignee: { email: { eq: \"user@example.com\" } } }, first: 20) { nodes { identifier title state { name } priority } } }"}' | python3 -m json.tool
```

### 列出项目
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ projects(first: 20) { nodes { id name description progress lead { name } teams { nodes { key } } url } } }"}' | python3 -m json.tool
```

### 列出团队成员
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ users { nodes { id name email active } } }"}' | python3 -m json.tool
```

### 列出标签
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issueLabels { nodes { id name color } } }"}' | python3 -m json.tool
```

## 常用变更操作

### 创建 issue
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation($input: IssueCreateInput!) { issueCreate(input: $input) { success issue { id identifier title url } } }",
    "variables": {
      "input": {
        "teamId": "TEAM_UUID",
        "title": "Fix login bug",
        "description": "Users cannot login with SSO",
        "priority": 2
      }
    }
  }' | python3 -m json.tool
```

### 更新 issue 状态
首先从上方的工作流状态查询中获取目标状态 UUID，然后：
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { issueUpdate(id: \"ENG-123\", input: { stateId: \"STATE_UUID\" }) { success issue { identifier state { name type } } } }"}' | python3 -m json.tool
```

### 分配 issue
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { issueUpdate(id: \"ENG-123\", input: { assigneeId: \"USER_UUID\" }) { success issue { identifier assignee { name } } } }"}' | python3 -m json.tool
```

### 设置优先级
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { issueUpdate(id: \"ENG-123\", input: { priority: 1 }) { success issue { identifier priority } } }"}' | python3 -m json.tool
```

### 添加评论
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { commentCreate(input: { issueId: \"ISSUE_UUID\", body: \"Investigated. Root cause is X.\" }) { success comment { id body } } }"}' | python3 -m json.tool
```

### 设置截止日期
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { issueUpdate(id: \"ENG-123\", input: { dueDate: \"2026-04-01\" }) { success issue { identifier dueDate } } }"}' | python3 -m json.tool
```

### 为 issue 添加标签
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { issueUpdate(id: \"ENG-123\", input: { labelIds: [\"LABEL_UUID_1\", \"LABEL_UUID_2\"] }) { success issue { identifier labels { nodes { name } } } } }"}' | python3 -m json.tool
```

### 将 issue 添加到项目
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "mutation { issueUpdate(id: \"ENG-123\", input: { projectId: \"PROJECT_UUID\" }) { success issue { identifier project { name } } } }"}' | python3 -m json.tool
```

### 创建项目
```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation($input: ProjectCreateInput!) { projectCreate(input: $input) { success project { id name url } } }",
    "variables": {
      "input": {
        "name": "Q2 Auth Overhaul",
        "description": "Replace legacy auth with OAuth2 and PKCE",
        "teamIds": ["TEAM_UUID"]
      }
    }
  }' | python3 -m json.tool
```

## 文档

Linear **Documents** 是与 issues 并列存储的文档（RFC、规范、笔记等）。它们有独立的 `documents` 根查询和 `document(id:)` 单条获取接口。

### 文档 URL 与 `slugId`

文档 URL 格式如下：
```
https://linear.app/<workspace>/document/<slug>-<hexSlugId>
```

末尾的十六进制段即为 `slugId`。示例：`https://linear.app/nousresearch/document/rfc-hermes-permission-gateway-discord-38359beef67c` → `slugId` 为 `38359beef67c`。

**重要 schema 细节：** Markdown 正文在 `content` 字段中。ProseMirror JSON 在 `contentState` 中（不是 `contentData`——该字段不存在，API 会返回 400）。

### 通过 slugId 获取文档

`document(id:)` 仅接受 UUID。若要通过 URL 中的十六进制 slug 获取，需过滤集合：

```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "query($s: String!) { documents(filter: { slugId: { eq: $s } }, first: 1) { nodes { id title content contentState slugId url creator { name } project { name } updatedAt } } }", "variables": {"s": "38359beef67c"}}' \
  | python3 -m json.tool
```

或通过 Python 辅助脚本：
```bash
python3 scripts/linear_api.py get-document 38359beef67c
```

### 通过 UUID 获取文档

```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ document(id: \"11700cff-b514-4db3-afcc-3ed1afacba1c\") { title content url } }"}' \
  | python3 -m json.tool
```

### 列出最近文档

```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ documents(first: 25, orderBy: updatedAt) { nodes { id title slugId url updatedAt project { name } } } }"}' \
  | python3 -m json.tool
```

### 按标题搜索文档

Linear 的 schema 没有 `searchDocuments` 根查询。请改用标题子字符串过滤：

```bash
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ documents(filter: { title: { containsIgnoreCase: \"RFC\" } }, first: 25) { nodes { title slugId url } } }"}' \
  | python3 -m json.tool
```

## 分页

Linear 使用 Relay 风格的游标分页：

```bash
# 第一页
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issues(first: 20) { nodes { identifier title } pageInfo { hasNextPage endCursor } } }"}' | python3 -m json.tool

# 下一页——使用上一响应中的 endCursor
curl -s -X POST https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ issues(first: 20, after: \"CURSOR_FROM_PREVIOUS\") { nodes { identifier title } pageInfo { hasNextPage endCursor } } }"}' | python3 -m json.tool
```

默认页大小：50。最大：250。始终使用 `first: N` 限制结果数量。

## 过滤参考

比较运算符：`eq`、`neq`、`in`、`nin`、`lt`、`lte`、`gt`、`gte`、`contains`、`startsWith`、`containsIgnoreCase`

使用 `or: [...]` 实现 OR 逻辑（filter 对象内默认为 AND）。

## 典型工作流

1. **查询团队**，获取团队 ID 和 key
2. **查询目标团队的工作流状态**，获取状态 UUID
3. **列出或搜索 issues**，找到需要处理的内容
4. **创建 issues**，提供团队 ID、标题、描述、优先级
5. **更新状态**，将 `stateId` 设置为目标工作流状态
6. **添加评论**，跟踪进度
7. **标记完成**，将 `stateId` 设置为团队的 "completed" 类型状态

## 速率限制

- 每个 API key 每小时 5,000 次请求
- 每小时 3,000,000 复杂度点
- 使用 `first: N` 限制结果数量以降低复杂度消耗
- 监控响应头 `X-RateLimit-Requests-Remaining`

## 重要说明

- 始终使用 `terminal` 工具配合 `curl` 进行 API 调用——不要使用 `web_extract` 或 `browser`
- 始终检查 GraphQL 响应中的 `errors` 数组——HTTP 200 仍可能包含错误
- 创建 issues 时若省略 `stateId`，Linear 默认使用第一个 backlog 状态
- `description` 字段支持 Markdown
- 使用 `python3 -m json.tool` 或 `jq` 格式化 JSON 响应以提高可读性