---
title: X (Twitter) 搜索
description: 使用 xAI 内置的 x_search Responses 工具在 agent 内搜索 X (Twitter) 帖子和话题串——支持 SuperGrok OAuth 登录或 XAI_API_KEY。
sidebar_label: X (Twitter) 搜索
sidebar_position: 7
---

# X (Twitter) 搜索

`x_search` 工具让 agent 可以直接搜索 X (Twitter) 的帖子、账号和话题串。其底层依托 xAI 在 Responses API（`https://api.x.ai/v1/responses`）上内置的 `x_search` 工具——Grok 在服务端执行搜索，并返回带有原始帖子引用的综合结果。

**当你明确需要 X 上的当前讨论、反应或观点时，请使用此工具而非 `web_search`。** 对于一般网页内容，继续使用 `web_search` / `web_extract`。

## 认证

满足以下**任一** xAI 凭据路径时，`x_search` 即会注册：

| 凭据 | 来源 | 配置方式 |
|------|------|---------|
| **SuperGrok / X Premium+ OAuth**（推荐） | 在 `accounts.x.ai` 浏览器登录，自动刷新 | `hermes auth add xai-oauth` — 参见 [xAI Grok OAuth (SuperGrok / X Premium+)](../../guides/xai-grok-oauth.md) |
| **`XAI_API_KEY`** | 付费 xAI API 密钥 | 在 `~/.hermes/.env` 中设置 |

两者使用相同的 endpoint 和相同的请求体，区别仅在于 bearer token。**当两者同时配置时，SuperGrok OAuth 优先**，x_search 将消耗你的订阅配额而非付费 API 用量。

工具的 `check_fn` 在每次重建模型工具列表时都会运行 xAI 凭据解析器。返回 `True` 表示 bearer token 可获取、非空，且（若已过期）已成功刷新。刷新失败的已撤销 token 会将该工具从 schema 中隐藏，模型将无法感知其存在。

## 启用工具

当 xAI 凭据（OAuth token 或 `XAI_API_KEY`）存在时自动启用。如不需要，可通过 `hermes tools` → Search → x_search 显式禁用。

```bash
hermes tools
# → 🐦 X (Twitter) Search   (press space to toggle on)
```

选择器提供两种凭据选项：

1. **xAI Grok OAuth (SuperGrok / Premium+)** — 若尚未登录，将打开浏览器跳转至 `accounts.x.ai`
2. **xAI API key** — 提示输入 `XAI_API_KEY`

任一选项均可满足门控条件。你可以使用已有的任意凭据，工具行为完全相同。若两者均已配置，调用时 OAuth 优先。

## 配置

```yaml
# ~/.hermes/config.yaml
x_search:
  # 用于 Responses 调用的 xAI 模型。
  # grok-4.5 是推荐的默认值；任何支持
  # x_search 工具访问权限的 Grok 模型均可使用。
  model: grok-4.5

  # 请求超时时间（秒）。复杂查询的 x_search 可能需要 60–120 秒，
  # 默认值较为宽松。最小值：30。
  timeout_seconds: 180

  # 遇到 5xx / ReadTimeout / ConnectionError 时的自动重试次数。
  # 每次重试按指数退避（1.5 倍尝试秒数，上限 5 秒）。
  retries: 2
```

## 工具参数

agent 调用 `x_search` 时使用以下参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | string（必填） | 在 X 上要查找的内容。 |
| `allowed_x_handles` | string 数组 | 可选，**仅**包含指定账号的列表（最多 10 个）。前缀 `@` 会被自动去除。 |
| `excluded_x_handles` | string 数组 | 可选，要排除的账号列表（最多 10 个）。与 `allowed_x_handles` 互斥。 |
| `from_date` | string | 可选，`YYYY-MM-DD` 格式的起始日期。 |
| `to_date` | string | 可选，`YYYY-MM-DD` 格式的结束日期。 |
| `enable_image_understanding` | boolean | 让 xAI 分析匹配帖子中附带的图片。 |
| `enable_video_understanding` | boolean | 让 xAI 分析匹配帖子中附带的视频。 |

工具返回的 JSON 包含：

- `answer` — Grok 生成的综合文本回答
- `citations` — Responses API 顶层字段返回的引用
- `inline_citations` — 从消息正文中提取的 `url_citation` 注释（每条包含 `url`、`title`、`start_index`、`end_index`）
- `degraded` — 当设置了任意缩小范围的过滤器（`allowed_x_handles`、`excluded_x_handles`、`from_date`、`to_date`）且两个引用渠道均返回空时为 `true`。此时 `answer` 是基于模型自身知识合成的，而非来自 X 索引，应视为无来源内容。否则为 `false`（包括"未设置过滤器"的情况——宽泛的无来源回答只是一个回答，而非过滤器未命中）
- `degraded_reason` — 列出哪些过滤器处于激活状态的简短字符串，当 `degraded` 为 `false` 时为 `null`
- `credential_source` — OAuth 解析成功时为 `"xai-oauth"`，API 密钥解析成功时为 `"xai"`
- `model`、`query`、`provider`、`tool`、`success`

### 日期验证

`from_date` / `to_date` 在发起 HTTP 调用前会在客户端进行验证：

- 若提供，两者均须能解析为 `YYYY-MM-DD` 格式。
- 当两者同时设置时，`from_date` 必须不晚于 `to_date`。
- `from_date` 不得晚于今天（UTC）——尚未开始的时间窗口内不可能存在帖子，调用必然返回零引用。
- `to_date` 允许为未来日期（调用方可能合理地请求"从昨天到明天"以捕获即将发布的帖子）。

验证失败会以结构化的 `{"error": "..."}` 工具结果返回，不会向 xAI 发起 HTTP 调用。

## 示例

与 agent 对话：

> X 上的人们对新的 Grok 图像功能有什么看法？重点关注 @xai 的回应。

agent 将：

1. 以 `query="reactions to new Grok image features"`、`allowed_x_handles=["xai"]` 调用 `x_search`
2. 获取综合回答及指向具体帖子的引用列表
3. 回复包含答案和参考来源

## 故障排查

### "No xAI credentials available"

当两种认证路径均失败时，工具会显示此错误。请在 `~/.hermes/.env` 中设置 `XAI_API_KEY`，或运行 `hermes auth add xai-oauth` 并完成浏览器登录。然后重启会话，让 agent 重新加载工具注册表。

### "`x_search` is not enabled for this model"

配置的 `x_search.model` 没有访问服务端 `x_search` 工具的权限。请切换至 `grok-4.5`（默认值）或其他支持该工具的 Grok 模型。当前支持列表请查阅 [xAI 文档](https://docs.x.ai/)。

### 工具未出现在 schema 中

可能有两个原因：

1. **工具集未启用。** 运行 `hermes tools`，确认 `🐦 X (Twitter) Search` 已勾选。
2. **无 xAI 凭据。** `check_fn` 返回 False，schema 保持隐藏。运行 `hermes auth status` 确认 xai-oauth 登录状态，并检查 `XAI_API_KEY` 是否已设置（如使用 API 密钥路径）。

### `degraded: true` — 回答无引用来源

当你使用了 `allowed_x_handles`、`excluded_x_handles` 或日期范围，且响应返回 `degraded: true` 时，说明 xAI 的 X 索引未找到匹配帖子，但 Grok 仍基于自身训练数据生成了综合回答。该回答无来源支撑——请勿将其视为真实的 X 内容。

值得排查的原因：

- **账号名拼写错误。** 去掉 `@`，仔细核对拼写，并确认该账号存在。
- **日期范围过窄**，或滑过了今日帖子；请扩大范围后重试。
- **xAI 索引缺口。** 部分活跃账号即使定期发帖，也会间歇性地无法在 `x_search` 中出现。请等待几分钟后重试，或在需要精确获取某账号时间线时使用 `xurl` 技能直接调用 X API。

## 另请参阅

- [xAI Grok OAuth (SuperGrok / Premium+)](../../guides/xai-grok-oauth.md) — OAuth 配置指南
- [Web 搜索与提取](web-search.md) — 用于一般（非 X）网页搜索
- [工具参考](../../reference/tools-reference.md) — 完整工具目录