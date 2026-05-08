# 上下文压缩和缓存

Hermes Agent 使用双重压缩系统和 Anthropic 提示缓存来在长对话中高效管理上下文窗口使用量。

源文件：`agent/context_engine.py`（ABC）、`agent/context_compressor.py`（默认引擎）、
`agent/prompt_caching.py`、`gateway/run.py`（会话卫生）、`run_agent.py`（搜索 `_compress_context`）


## 可插拔上下文引擎

上下文管理建立在 `ContextEngine` ABC（`agent/context_engine.py`）之上。内置的 `ContextCompressor` 是默认实现，但插件可以用替代引擎（如无损上下文管理）替换它。

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
  engine: "lcm"           # example — plugin providing lossless context
```

引擎负责：
- 决定何时应该触发压缩（`should_compress()`）
- 执行压缩（`compress()`）
- 可选地暴露代理可以调用的工具（如 `lcm_grep`）
- 跟踪 API 响应中的 token 使用量

选择通过 `config.yaml` 中的 `context.engine` 由配置驱动。解析顺序：
1. 检查 `plugins/context_engine/<name>/` 目录
2. 检查通用插件系统（`register_context_engine()`）
3. 回退到内置的 `ContextCompressor`

插件引擎**从不自动激活** — 用户必须明确将 `context.engine` 设置为插件的名称。默认的 `"compressor"` 始终使用内置引擎。

通过 `hermes plugins` → Provider 插件 → 上下文引擎配置，或直接编辑 `config.yaml`。

有关构建上下文引擎插件，参见[上下文引擎插件](/docs/developer-guide/context-engine-plugin)。

## 双重压缩系统

Hermes 有两个独立运行的压缩层：

```
                     ┌──────────────────────────┐
  Incoming message   │   Gateway Session Hygiene │  在上下文 85% 时触发
  ─────────────────► │   (pre-agent, rough est.) │  大会话的安全网
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │   Agent ContextCompressor │  在上下文 50% 时触发（默认）
                     │   (in-loop, real tokens)  │  正常上下文管理
                     └──────────────────────────┘
```

### 1. 网关会话卫生（85% 阈值）

位于 `gateway/run.py`（搜索 `Session hygiene: auto-compress`）。这是一个**安全网**，在代理处理消息之前运行。它防止在轮次之间（例如，Telegram/Discord 中的隔夜积累）会话增长太大时 API 失败。

- **阈值**：固定在模型上下文长度的 85%
- **Token 来源**：优先使用上一轮中 API 报告的实际 token；回退到粗略的基于字符的估计（`estimate_messages_tokens_rough`）
- **触发条件**：仅当 `len(history) >= 4` 且启用压缩时
- **目的**：捕获逃脱代理自身压缩器的会话

网关卫生阈值故意高于代理的压缩器。将其设置在 50%（与代理相同）会导致长网关会话中每轮都进行过早压缩。

### 2. 代理 ContextCompressor（50% 阈值，可配置）

位于 `agent/context_compressor.py`。这是在代理的工具循环中运行的**主要压缩系统**，可以访问准确的、API 报告的 token 计数。


## 配置

所有压缩设置从 `config.yaml` 中的 `compression` 键读取：

```yaml
compression:
  enabled: true              # Enable/disable compression (default: true)
  threshold: 0.50            # Fraction of context window (default: 0.50 = 50%)
  target_ratio: 0.20         # How much of threshold to keep as tail (default: 0.20)
  protect_last_n: 20         # Minimum protected tail messages (default: 20)

# Summarization model/provider configured under auxiliary:
auxiliary:
  compression:
    model: null              # Override model for summaries (default: auto-detect)
    provider: auto           # Provider: "auto", "openrouter", "nous", "main", etc.
    base_url: null           # Custom OpenAI-compatible endpoint
```

### 参数详情

| 参数 | 默认值 | 范围 | 描述 |
|-----------|---------|-------|-------------|
| `threshold` | `0.50` | 0.0-1.0 | 当提示词 token ≥ `threshold × context_length` 时触发压缩 |
| `target_ratio` | `0.20` | 0.10-0.80 | 控制尾部保护 token 预算：`threshold_tokens × target_ratio` |
| `protect_last_n` | `20` | ≥1 | 始终保留的最近消息的最小数量 |
| `protect_first_n` | `3` | （硬编码）| 系统提示词 + 第一次交换始终保留 |

### 计算值（200K 上下文模型默认值）

```
context_length       = 200,000
threshold_tokens     = 200,000 × 0.50 = 100,000
tail_token_budget    = 100,000 × 0.20 = 20,000
max_summary_tokens   = min(200,000 × 0.05, 12,000) = 10,000
```


## 压缩算法

`ContextCompressor.compress()` 方法遵循 4 阶段算法：

### 阶段 1：清除旧工具结果（便宜，无 LLM 调用）

受保护尾部之外的旧工具结果（>200 字符）替换为：
```
[Old tool output cleared to save context space]
```

这是一个便宜的预处理步骤，可以从冗长的工具输出（文件内容、终端输出、搜索结果）中节省大量 token。

### 阶段 2：确定边界

```
┌─────────────────────────────────────────────────────────────┐
│  Message list                                               │
│                                                             │
│  [0..2]  ← protect_first_n (system + first exchange)        │
│  [3..N]  ← middle turns → SUMMARIZED                        │
│  [N..end] ← tail (by token budget OR protect_last_n)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

尾部保护**基于 token 预算**：从末尾向后走，累积 token 直到预算耗尽。如果预算保护的消息比固定的 `protect_last_n` 计数少，则回退到后者。

边界对齐以避免拆分 tool_call/tool_result 组。`_align_boundary_backward()` 方法走过连续的工具结果以找到父助手消息，保持组的完整性。

### 阶段 3：生成结构化摘要

:::warning 摘要模型上下文长度
摘要模型必须有**至少与主代理模型一样大**的上下文窗口。整个中间部分在单个 `call_llm(task="compression")` 调用中发送给摘要模型。如果摘要模型的上下文较小，API 返回上下文长度错误 — `_generate_summary()` 捕获它，记录警告，并返回 `None`。压缩器然后**在没有摘要的情况下丢弃**中间轮次，静默丢失对话上下文。这是压缩质量下降最常见的原因。
:::

使用结构化模板通过辅助 LLM 对中间轮次进行摘要：

```
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work — specific file paths, commands run, results]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next]

## Critical Context
[Specific values, error messages, configuration details]
```

摘要预算随被压缩的内容量而扩展：
- 公式：`content_tokens × 0.20`（`_SUMMARY_RATIO` 常量）
- 最小值：2,000 token
- 最大值：`min(context_length × 0.05, 12,000)` token

### 阶段 4：组装压缩消息

压缩消息列表是：
1. 头消息（第一次压缩时在系统提示词附加注释）
2. 摘要消息（选择角色以避免连续相同角色违规）
3. 尾消息（未修改）

通过 `_sanitize_tool_pairs()` 清理孤立的 tool_call/tool_result 对：
- 引用已删除调用的工具结果 → 删除
- 结果已删除的工具调用 → 注入存根结果

### 迭代重新压缩

在后续压缩中，之前的摘要与 LLM 的**更新**指令一起传递，而不是从头摘要。这在多次压缩中保留信息 — 项目从"进行中"移动到"已完成"，添加新进度，删除过时信息。

压缩器实例上的 `_previous_summary` 字段存储上一次摘要文本以用于此目的。


## 压缩前后示例

### 压缩前（45 条消息，约 95K token）

```
[0] system:    "You are a helpful assistant..." (system prompt)
[1] user:      "Help me set up a FastAPI project"
[2] assistant: <tool_call> terminal: mkdir project </tool_call>
[3] tool:      "directory created"
[4] assistant: <tool_call> write_file: main.py </tool_call>
[5] tool:      "file written (2.3KB)"
    ... 30 more turns of file editing, testing, debugging ...
[38] assistant: <tool_call> terminal: pytest </tool_call>
[39] tool:      "8 passed, 2 failed\n..."  (5KB output)
[40] user:      "Fix the failing tests"
[41] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[42] tool:      "import pytest\n..."  (3KB)
[43] assistant: "I see the issue with the test fixtures..."
[44] user:      "Great, also add error handling"
```

### 压缩后（25 条消息，约 45K token）

```
[0] system:    "You are a helpful assistant...
               [Note: Some earlier conversation turns have been compacted...]"
[1] user:      "Help me set up a FastAPI project"
[2] assistant: "[CONTEXT COMPACTION] Earlier turns were compacted...

               ## Goal
               Set up a FastAPI project with tests and error handling

               ## Progress
               ### Done
               - Created project structure: main.py, tests/, requirements.txt
               - Implemented 5 API endpoints in main.py
               - Wrote 10 test cases in tests/test_api.py
               - 8/10 tests passing

               ### In Progress
               - Fixing 2 failing tests (test_create_user, test_delete_user)

               ## Relevant Files
               - main.py — FastAPI app with 5 endpoints
               - tests/test_api.py — 10 test cases
               - requirements.txt — fastapi, pytest, httpx

               ## Next Steps
               - Fix failing test fixtures
               - Add error handling"
[3] user:      "Fix the failing tests"
[4] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[5] tool:      "import pytest\n..."
[6] assistant: "I see the issue with the test fixtures..."
[7] user:      "Great, also add error handling"
```


## 提示缓存（Anthropic）

来源：`agent/prompt_caching.py`

通过缓存对话前缀，减少多轮对话的输入 token 成本约 75%。使用 Anthropic 的 `cache_control` 断点。

### 策略：system_and_3

Anthropic 允许每个请求最多 4 个 `cache_control` 断点。Hermes 使用"system_and_3"策略：

```
断点 1：系统提示词           （跨所有轮次稳定）
断点 2：倒数第 3 条非系统消息  ─┐
断点 3：倒数第 2 条非系统消息   ├─ 滚动窗口
断点 4：最后一条非系统消息     ─┘
```

### 工作原理

`apply_anthropic_cache_control()` 深拷贝消息并注入 `cache_control` 标记：

```python
# Cache marker format
marker = {"type": "ephemeral"}
# Or for 1-hour TTL:
marker = {"type": "ephemeral", "ttl": "1h"}
```

根据内容类型不同地应用标记：

| 内容类型 | 标记位置 |
|-------------|-------------------|
| 字符串内容 | 转换为 `[{"type": "text", "text": ..., "cache_control": ...}]` |
| 列表内容 | 添加到最后一个元素的字典 |
| None/空 | 添加为 `msg["cache_control"]` |
| 工具消息 | 添加为 `msg["cache_control"]`（仅原生 Anthropic） |

### 缓存感知设计模式

1. **稳定的系统提示词**：系统提示词是断点 1，跨所有轮次缓存。避免在对话中途修改它（压缩仅在第一次压缩时附加注释）。

2. **消息顺序很重要**：缓存命中需要前缀匹配。在中间添加或删除消息会使其后的所有内容的缓存失效。

3. **压缩缓存交互**：压缩后，被压缩区域的缓存失效，但系统提示词缓存存活。滚动的 3 消息窗口在 1-2 轮内重新建立缓存。

4. **TTL 选择**：默认为 `5m`（5 分钟）。对于用户在轮次之间休息的长时间运行会话，使用 `1h`。

### 启用提示缓存

以下情况下自动启用提示缓存：
- 模型是 Anthropic Claude 模型（通过模型名称检测）
- Provider 支持 `cache_control`（原生 Anthropic API 或 OpenRouter）

```yaml
# config.yaml — TTL 可配置（必须是 "5m" 或 "1h"）
prompt_caching:
  cache_ttl: "5m"
```

CLI 在启动时显示缓存状态：
```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```


## 上下文压力警告

中间上下文压力警告已被删除（参见 `run_agent.py` 中的迭代预算块，注释说："没有中间压力警告 — 它们导致模型在复杂任务上'过早放弃'"）。压缩在提示词 token 达到配置的 `compression.threshold`（默认 50%）时触发，没有先前的警告步骤；网关会话卫生在模型上下文窗口的 85% 时作为次要安全网触发。
