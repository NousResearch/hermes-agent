---
sidebar_position: 3
title: "Agent Loop 内部机制"
description: "AIAgent 执行流程、API 模式、工具、回调及回退行为的详细说明"
---

# Agent Loop 内部机制

核心编排引擎是 `run_agent.py` 中的 `AIAgent` 类——这是一个大型文件（15k+ 行），负责处理从 prompt（提示词）组装到工具分发再到 provider 故障转移的所有逻辑。

## 核心职责

`AIAgent` 负责：

- 通过 `prompt_builder.py` 组装有效的系统 prompt 和工具 schema
- 选择正确的 provider/API 模式（`chat_completions`、`codex_responses`、`anthropic_messages`）
- 发起支持取消操作的可中断模型调用
- 执行工具调用（顺序执行或通过线程池并发执行）
- 以 OpenAI 消息格式维护对话历史
- 处理压缩、重试和回退模型切换
- 跨父 agent 和子 agent 追踪迭代预算
- 在上下文丢失前将持久化内存刷写到磁盘

## 两个入口点

```python
# 简单接口——返回最终响应字符串
response = agent.chat("Fix the bug in main.py")

# 完整接口——返回包含消息、元数据、用量统计的 dict
result = agent.run_conversation(
    user_message="Fix the bug in main.py",
    system_message=None,           # 省略时自动构建
    conversation_history=None,      # 省略时自动从 session 加载
    task_id="task_abc123"
)
```

`chat()` 是对 `run_conversation()` 的轻量封装，从结果 dict 中提取 `final_response` 字段。

## API 模式

Hermes 支持三种 API 执行模式，通过 provider 选择、显式参数和 base URL 启发式规则来确定：

| API 模式 | 用途 | 客户端类型 |
|----------|------|-----------|
| `chat_completions` | 兼容 OpenAI 的端点（OpenRouter、自定义及大多数 provider） | `openai.OpenAI` |
| `codex_responses` | OpenAI Codex / Responses API | `openai.OpenAI`（使用 Responses 格式） |
| `anthropic_messages` | 原生 Anthropic Messages API | 通过适配器使用 `anthropic.Anthropic` |

模式决定了消息的格式化方式、工具调用的结构、响应的解析方式，以及缓存/流式传输的工作方式。三种模式在 API 调用前后均收敛到相同的内部消息格式（OpenAI 风格的 `role`/`content`/`tool_calls` dict）。

**模式解析顺序：**
1. 显式 `api_mode` 构造函数参数（最高优先级）
2. Provider 特定检测（例如 `anthropic` provider → `anthropic_messages`）
3. Base URL 启发式规则（例如 `api.anthropic.com` → `anthropic_messages`）
4. 默认：`chat_completions`

## 单轮生命周期

agent loop 的每次迭代按以下顺序执行：

```text
run_conversation()
  1. 若未提供则生成 task_id
  2. 将用户消息追加到对话历史
  3. 构建或复用已缓存的系统 prompt（prompt_builder.py）
  4. 检查是否需要预检压缩（上下文超过 50%）
  5. 从对话历史构建 API 消息
     - chat_completions：直接使用 OpenAI 格式
     - codex_responses：转换为 Responses API 输入项
     - anthropic_messages：通过 anthropic_adapter.py 转换
  6. 注入临时 prompt 层（预算警告、上下文压力提示）
  7. 若使用 Anthropic，应用 prompt 缓存标记
  8. 发起可中断的 API 调用（_interruptible_api_call）
  9. 解析响应：
     - 若有 tool_calls：执行工具，追加结果，回到步骤 5
     - 若为文本响应：持久化 session，按需刷写内存，返回
```

### 消息格式

所有消息在内部均使用兼容 OpenAI 的格式：

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

推理内容（来自支持扩展思考的模型）存储在 `assistant_msg["reasoning"]` 中，并可选择通过 `reasoning_callback` 展示。

### 消息交替规则

agent loop 强制执行严格的消息角色交替规则：

- 系统消息之后：`User → Assistant → User → Assistant → ...`
- 工具调用期间：`Assistant（含 tool_calls）→ Tool → Tool → ... → Assistant`
- **不允许**连续出现两条 assistant 消息
- **不允许**连续出现两条 user 消息
- **只有** `tool` 角色可以连续出现（并行工具结果）

Provider 会验证这些序列，并拒绝格式错误的历史记录。

## 可中断的 API 调用

API 请求被封装在 `_interruptible_api_call()` 中，该方法在后台线程中执行实际的 HTTP 调用，同时监听中断事件：

```text
┌────────────────────────────────────────────────────┐
│  主线程                        API 线程             │
│                                                    │
│   等待：                        HTTP POST           │
│    - 响应就绪          ───▶    发送至 provider       │
│    - 中断事件                                       │
│    - 超时                                          │
└────────────────────────────────────────────────────┘
```

当发生中断（用户发送新消息、`/stop` 命令或信号）时：
- API 线程被放弃（响应被丢弃）
- agent 可以处理新输入或干净地关闭
- 不会将部分响应注入对话历史

## 工具执行

### 顺序执行与并发执行

当模型返回工具调用时：

- **单个工具调用** → 直接在主线程中执行
- **多个工具调用** → 通过 `ThreadPoolExecutor` 并发执行
  - 例外：标记为交互式的工具（如 `clarify`）强制顺序执行
  - 无论完成顺序如何，结果均按原始工具调用顺序重新插入

### 执行流程

```text
for each tool_call in response.tool_calls:
    1. 从 tools/registry.py 解析处理器
    2. 触发 pre_tool_call 插件 hook
    3. 检查是否为危险命令（tools/approval.py）
       - 若危险：调用 approval_callback，等待用户确认
    4. 使用参数 + task_id 执行处理器
    5. 触发 post_tool_call 插件 hook
    6. 将 {"role": "tool", "content": result} 追加到历史
```

### Agent 级工具

部分工具在到达 `handle_function_call()` 之前，由 `run_agent.py` *提前*拦截：

| 工具 | 拦截原因 |
|------|---------|
| `todo` | 读写 agent 本地任务状态 |
| `memory` | 向持久化内存文件写入内容（有字符限制） |
| `session_search` | 通过 agent 的 session DB 查询 session 历史 |
| `delegate_task` | 以隔离上下文生成子 agent |

这些工具直接修改 agent 状态，并返回合成的工具结果，不经过注册表。

## 回调接口

`AIAgent` 支持平台特定的回调，用于在 CLI、gateway 和 ACP 集成中实现实时进度展示：

| 回调 | 触发时机 | 使用方 |
|------|---------|--------|
| `tool_progress_callback` | 每次工具执行前后 | CLI spinner、gateway 进度消息 |
| `thinking_callback` | 模型开始/停止思考时 | CLI "thinking..." 指示器 |
| `reasoning_callback` | 模型返回推理内容时 | CLI 推理展示、gateway 推理块 |
| `clarify_callback` | 调用 `clarify` 工具时 | CLI 输入提示、gateway 交互消息 |
| `step_callback` | 每次完整 agent 轮次结束后 | Gateway 步骤追踪、ACP 进度 |
| `stream_delta_callback` | 每个流式 token（启用时） | CLI 流式展示 |
| `tool_gen_callback` | 从流中解析出工具调用时 | CLI spinner 中的工具预览 |
| `status_callback` | 状态变更时（思考、执行等） | ACP 状态更新 |

## 预算与回退行为

### 迭代预算

agent 通过 `IterationBudget` 追踪迭代次数：

- 默认：90 次迭代（可通过 `agent.max_turns` 配置）
- 每个 agent 拥有独立预算。子 agent 获得独立预算，上限为 `delegation.max_iterations`（默认 50）——父 agent 与子 agent 的总迭代次数可超过父 agent 的上限
- 达到 100% 时，agent 停止并返回已完成工作的摘要

### 回退模型

当主模型失败时（429 限流、5xx 服务器错误、401/403 鉴权错误）：

1. 检查配置中的 `fallback_providers` 列表
2. 按顺序尝试每个回退 provider
3. 成功后，使用新 provider 继续对话
4. 遇到 401/403 时，在故障转移前尝试刷新凭据

回退系统也独立覆盖辅助任务——视觉、压缩和网页提取各自拥有独立的回退链，可通过 `auxiliary.*` 配置节进行配置。

## 压缩与持久化

### 压缩触发时机

- **预检**（API 调用前）：对话超过模型上下文窗口的 50%
- **Gateway 自动压缩**：对话超过 85%（更激进，在轮次之间运行）

### 压缩过程

1. 首先将内存刷写到磁盘（防止数据丢失）
2. 将中间对话轮次摘要为紧凑的摘要内容
3. 保留最后 N 条消息完整不变（`compression.protect_last_n`，默认：20）
4. 工具调用/结果消息对保持完整（不拆分）
5. 生成新的 session 血缘 ID（压缩会创建一个"子" session）

### Session 持久化

每轮结束后：
- 消息保存到 session 存储（通过 `hermes_state.py` 使用 SQLite）
- 内存变更刷写到 `MEMORY.md` / `USER.md`
- 可通过 `/resume` 或 `hermes chat --resume` 恢复 session

## 关键源文件

| 文件 | 用途 |
|------|------|
| `run_agent.py` | AIAgent 类——完整的 agent loop |
| `agent/prompt_builder.py` | 从内存、技能、上下文文件和个性组装系统 prompt |
| `agent/context_engine.py` | ContextEngine ABC——可插拔的上下文管理 |
| `agent/context_compressor.py` | 默认引擎——有损摘要算法 |
| `agent/prompt_caching.py` | Anthropic prompt 缓存标记和缓存指标 |
| `agent/auxiliary_client.py` | 用于辅助任务的辅助 LLM 客户端（视觉、摘要） |
| `model_tools.py` | 工具 schema 集合，`handle_function_call()` 分发 |

## 相关文档

- [Provider 运行时解析](./provider-runtime.md)
- [Prompt 组装](./prompt-assembly.md)
- [上下文压缩与 Prompt 缓存](./context-compression-and-caching.md)
- [工具运行时](./tools-runtime.md)
- [架构概览](./architecture.md)