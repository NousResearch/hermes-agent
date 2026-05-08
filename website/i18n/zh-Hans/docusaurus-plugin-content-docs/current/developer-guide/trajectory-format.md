---
sidebar_position: 10
title: "轨迹格式"
description: "ShareGPT 格式轨迹保存和加载的规范"
---

# 轨迹格式

Hermes Agent 将对话轨迹保存为 ShareGPT 格式的 JSONL（JSON Lines），用于 RL 训练、SFT 数据集和基准测试。

## 格式概览

```json
{
  "conversation_id": "sess_abc123def456",
  "conversations": [
    {"from": "human", "value": "How do I fix the bug in main.py?"},
    {"from": "gpt", "value": "<think>\nI'll help you fix the bug.\n</think>\nThe bug is in line 42..."},
    {"from": "human", "value": "<think>\nLet me check the file first\n</think>\nPlease read the file first."},
    {"from": "gpt", "value": "<tool_call>\n{\"tool_call_id\": \"call_abc123\", \"name\": \"read_file\", \"arguments\": {\"path\": \"main.py\"}}\n</tool_call>\n<tool_response>\n{\"tool_call_id\": \"call_abc123\", \"name\": \"read_file\", \"content\": \"Python 3.11.6\"}\n</tool_response>"},
  ],
  "system": "<system prompt>",
  "tool_stats": {
    "read_file": 3,
    "write_file": 1,
    "terminal": 2
  },
  "turns": 5,
  "reasoning_tokens": 1250,
  "completion_tokens": 890,
  "total_tokens": 3140,
  "timestamp": "2026-03-30T14:22:31.456789",
  "model": "anthropic/claude-sonnet-4.6",
  "completed": true
}
```

## 字段说明

### `conversation_id`

会话的唯一标识符，与 Hermes SessionDB 中的 `session_id` 对应。

### `conversations`

对话消息数组，按时间顺序。遵循 ShareGPT 格式：

| 角色 | 描述 |
|------|------|
| `human` | 用户消息 |
| `gpt` | 助手响应（包括 `<tool_call>` 和 `<tool_response>` XML 块） |

### `system`

在轨迹保存时生成的完整系统提示词。包含：

- 函数调用协议说明
- `<tools>` XML 块（JSON 工具定义）
- 函数调用示例

### `tool_stats`

工具使用统计 — 每个工具被调用的次数。

### `turns`

完整对话轮次数（用户-助手交互对）。

### Token 计数

- `reasoning_tokens`：推理内容 token 数（来自 `<think>` 块）
- `completion_tokens`：完成 token 数
- `total_tokens`：总 token 数

### `completed`

布尔值 — 对话是否自然完成。

- `true`：模型自然停止
- `false`：被中断、超时或达到最大轮次

## 对话格式详情

### `<tool_call>` 块

当模型请求工具调用时：

```
<tool_call>
{"tool_call_id": "call_abc123", "name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>
```

属性：
- `tool_call_id`：工具调用的唯一 ID
- `name`：工具名称
- `arguments`：JSON 格式的参数对象

### `<tool_response>` 块

工具执行后：

```
<tool_response>
{"tool_call_id": "call_abc123", "name": "terminal", "content": "Python 3.11.6"}
</tool_response>
```

一个 `<gpt>` 消息中可以有多个连续的 `<tool_call>` 和 `<tool_response>` 块。

## 规范化规则

### 推理内容标记

轨迹转换器将所有推理规范化为 `<think>` 标签，无论模型原始如何产生：

1. **原生推理 token**（来自 Anthropic、OpenAI o-series 等提供商的 `msg["reasoning"]` 字段）：包装为 `<think>\n{reasoning}\n</think>\n` 并预置到内容前。

2. **REASONING_SCRATCHPAD XML**（当原生推理禁用且模型通过系统提示词指令的 XML 推理时）：`<REASONING_SCRATCHPAD>` 标签通过 `convert_scratchpad_to_think()` 转换为 `<think>`。

3. **空 think 块**：每个 `gpt` 轮次保证有 `<think>` 块。如果没有产生推理，插入空块：`<think>\n\n</think>\n` — 这确保训练数据格式一致。

### 工具调用规范化

API 格式的工具调用（带有 `tool_call_id`、函数名、JSON 字符串参数）转换为 XML 包装的 JSON：

```
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>
```

- 参数从 JSON 字符串解析回对象（不是双重编码）
- 如果 JSON 解析失败（不应该发生 — 对话中已验证），使用空的 `{}` 并记录警告
- 一个助手轮次中的多个工具调用在一个 `gpt` 消息中产生多个 `<tool_call>` 块

### 工具响应规范化

跟随助手消息的所有工具结果分组到一个带有 XML 包装的 JSON 响应的单个 `tool` 轮次中：

```
<tool_response>
{"tool_call_id": "call_abc123", "name": "terminal", "content": "output here"}
</tool_response>
```

- 如果工具内容看起来像 JSON（以 `{` 或 `[` 开头），则解析它以便内容字段包含 JSON 对象/数组而不是字符串
- 多个工具结果用换行符连接在一个消息中
- 工具名称通过位置与父助手的 `tool_calls` 数组匹配

### 系统消息

系统消息在保存时生成（不是从对话中获取）。它遵循 Hermes 函数调用提示词模板，包含：

- 解释函数调用协议的前言
- 包含 JSON 工具定义的 `<tools>` XML 块
- `FunctionCall` 对象的模式参考
- `<tool_call>` 示例

工具定义包含 `name`、`description`、`parameters` 和 `required`（设置为 `null` 以匹配规范格式）。

## 加载轨迹

轨迹是标准 JSONL — 用任何 JSON-lines 阅读器加载：

```python
import json

def load_trajectories(path: str):
    """Load trajectory entries from a JSONL file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

# Filter to successful completions only
successful = [e for e in load_trajectories("trajectory_samples.jsonl")
              if e.get("completed")]

# Extract just the conversations for training
training_data = [e["conversations"] for e in successful]
```

### 加载用于 HuggingFace 数据集

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="trajectory_samples.jsonl")
```

规范化的 `tool_stats` 模式确保所有条目具有相同的列，防止在数据集加载期间出现 Arrow 模式不匹配错误。

## 控制轨迹保存

在 CLI 中，轨迹保存由以下控制：

```yaml
# config.yaml
agent:
  save_trajectories: true  # default: false
```

或通过 `--save-trajectories` 标志。当代理使用 `save_trajectories=True` 初始化时，每个对话轮次结束时调用 `_save_trajectory()` 方法。

批处理运行器始终保存轨迹（这是其主要目的）。

具有零推理的样本（跨所有轮次）在保存前被批处理运行器自动丢弃，以避免用非推理示例污染训练数据。
