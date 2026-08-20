---
title: 工具搜索
sidebar_position: 95
---

# 工具搜索（Tool Search）

当你在一个会话上附加了许多 MCP 服务器或非核心插件工具时，它们的 JSON schema 可能在每一轮次消耗大量上下文窗口 —— 即使只有少数工具与用户实际问的问题相关。

**工具搜索（Tool Search）** 是 Hermes 为解决这个问题而提供的可选渐进式披露层。激活后，MCP 和插件工具在模型可见的工具数组中被替换为三个桥接工具，模型按需加载每个具体工具的 schema。

:::info Hermes 内置工具从不延迟
构成 Hermes 核心能力集的工具（`terminal`、`read_file`、`write_file`、`patch`、`search_files`、`todo`、`memory`、`browser_*`、`web_search`、`web_extract`、`clarify`、`execute_code`、`delegate_task`、`session_search` 以及 `_HERMES_CORE_TOOLS` 中的其余工具）*始终*直接加载。只有 MCP 工具和非核心插件工具有资格被延迟。
:::

## 工作原理

当工具搜索在某一轮次激活时，模型会看到三个新工具替代了被延迟的工具：

```
tool_search(query, limit?)     — search the deferred-tool catalog
tool_describe(name)            — load the full schema for one tool
tool_call(name, arguments)     — invoke a deferred tool
```

典型的交互过程如下：

```
Model: tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }
Model: tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

当模型调用 `tool_call` 时，Hermes 会**解包桥接层**，以与模型直接调用相同的方式分派底层工具。工具调用前后的钩子、防护栏、审批提示都针对真实工具名运行 —— 而不是 `tool_call`。CLI 和网关中的活动流也会解包，让你看到底层工具而不是桥接层。

## 何时激活？

默认情况下工具搜索以 `auto` 模式运行：仅当延迟工具的 schema 将消耗当前模型上下文窗口至少 10% 时才激活。低于该阈值时，工具数组组装是纯粹的直通，你不会产生额外开销。

这个决策在每次构建工具数组时重新评估，因此：

- 只有少量 MCP 工具和长上下文模型的会话永远不会激活工具搜索。
- 附加了许多 MCP 服务器（通常 15+ 个工具）的会话会开始激活它。
- 在会话中移除 MCP 服务器会在下次组装时正确回到直接暴露。

## 配置

```yaml
tools:
  tool_search:
    enabled: auto       # auto (default), on, or off
    threshold_pct: 10   # percentage of context — only used in auto mode
    search_default_limit: 5
    max_search_limit: 20
```

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | `auto` | `auto` 在超过阈值时激活；`on` 只要有至少一个可延迟工具就激活；`off` 完全禁用。 |
| `threshold_pct` | `10` | `auto` 模式启动的上下文长度百分比。范围 0–100。 |
| `search_default_limit` | `5` | 模型调用 `tool_search` 未传 `limit` 时返回的结果数。 |
| `max_search_limit` | `20` | 模型可通过 `limit` 请求的硬上限。范围 1–50。 |

你也可以使用旧版布尔值格式：

```yaml
tools:
  tool_search: true   # equivalent to {enabled: auto}
```

## 何时不该使用

工具搜索以固定的每轮次 token 成本（三个桥接工具 schema，约 300 token）和至少一次额外往返（搜索 → 描述 → 调用）换取延迟 schema 的节省。当你工具很多而每轮次只用少量时，这是明确的收益；当工具总数很少时，这是开销。

`auto` 默认值会自动处理。如果你无条件设置 `enabled: on`，在小工具集上会预期有轻微的每轮次成本。

## 不可避免的权衡

这些来自提示缓存完整性不变式 —— 它们是任何渐进式披露设计固有的，而非本实现特有：

- **冷工具时多一次往返。** 模型第一次需要延迟工具时，会额外花费一两次模型调用来查找和加载 schema。静态方面的 token 节省是真实的，但一部分在运行时被偿还。
- **延迟 schema 没有缓存收益。** 加载的 `tool_describe` 结果进入对话历史（因此在后续轮次确实获得缓存），但它永远不会受益于系统提示缓存前缀。
- **依赖模型质量。** 工具搜索假设模型能为它想要的工具写出合理的搜索查询。较小的模型在这方面表现较差；Anthropic 发布的数据（Opus 4 有/无工具搜索时从 49% 提升到 74%）展示了上限，但也说明约 26 个百分点的准确度差距仍然是检索失败。
- **工具集编辑会使缓存失效。** 在会话中添加或移除工具会改变桥接工具的描述（包含延迟工具的计数）和目录，因此提示缓存会被失效。这与任何工具集编辑的权衡相同。

## 实现细节

- **检索：** 对分词后的工具名 + 描述 + 参数名进行 BM25。当 BM25 返回无正分结果时回退到工具名的字面子串匹配，防止零 IDF 退化情况（例如在所有工具名都包含 "github" 的目录中搜索 `"github"`）。
- **目录跨轮次无状态。** 每次组装都从当前工具定义列表重建 —— 没有会话键控的 `Map`。这避免了存储的目录与实时工具注册表不同步的那类 Bug。
- **目录限定于会话的工具集。** `tool_search`、`tool_describe` 和 `tool_call` 只能看到和调用会话实际授权的工具。子代理、看板工作者或限制于工具集子集的网关会话无法使用桥接层来发现或调用该子集之外的工具 —— 延迟目录是会话自身启用/禁用工具集的可延迟切片，而非整个进程注册表。
- **无 JS 沙箱。** Hermes 使用更简单的"结构化工具"模式（搜索/描述/调用作为普通函数）。某些其他实现提供的 JS 沙箱"代码模式"是一个很大的攻击面；我们跳过了它。

## 另请参阅

- `tools/tool_search.py` —— 实现代码
- `tests/tools/test_tool_search.py` —— 回归测试套件
- 原始实现 PR 中的 `openclaw-tool-search-report` PDF，包含塑造了该设计的研究
