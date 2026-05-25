---
sidebar_position: 5
title: "将 Hermes 作为 Python 库使用"
description: "将 AIAgent 嵌入你自己的 Python 脚本、Web 应用或自动化流水线——无需 CLI"
---

# 将 Hermes 作为 Python 库使用

Hermes 不仅仅是一个 CLI 工具。你可以直接导入 `AIAgent`，在自己的 Python 脚本、Web 应用或自动化流水线中以编程方式使用它。本指南将介绍具体方法。

---

## 安装

直接从仓库安装 Hermes：

```bash
pip install git+https://github.com/NousResearch/hermes-agent.git
```

或使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv pip install git+https://github.com/NousResearch/hermes-agent.git
```

也可以在 `requirements.txt` 中固定版本：

```text
hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git
```

:::tip
将 Hermes 作为库使用时，CLI 所需的环境变量同样必须设置。至少需要设置 `OPENROUTER_API_KEY`（若直接访问提供商，则设置 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`）。
:::

---

## 基本用法

使用 Hermes 最简单的方式是 `chat()` 方法——传入一条消息，返回一个字符串：

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)
response = agent.chat("What is the capital of France?")
print(response)
```

`chat()` 在内部处理完整的对话循环——工具调用、重试等一切事务——并仅返回最终的文本响应。

:::warning
将 Hermes 嵌入自己的代码时，务必设置 `quiet_mode=True`。否则，agent 会打印 CLI 的加载动画、进度指示器及其他终端输出，从而干扰你的应用输出。
:::

---

## 完整对话控制

如需对对话进行更精细的控制，可直接使用 `run_conversation()`。它返回一个包含完整响应、消息历史和元数据的字典：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="Search for recent Python 3.13 features",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"Messages exchanged: {len(result['messages'])}")
```

返回的字典包含：
- **`final_response`** — agent 的最终文本回复
- **`messages`** — 完整的消息历史（系统消息、用户消息、助手消息、工具调用）

（传入的 `task_id` 存储在 agent 实例上用于 VM 隔离，不会在返回字典中回显。）

你也可以传入自定义系统消息，覆盖该次调用的临时系统 prompt（提示词）：

```python
result = agent.run_conversation(
    user_message="Explain quicksort",
    system_message="You are a computer science tutor. Use simple analogies.",
)
```

---

## 配置工具集

使用 `enabled_toolsets` 或 `disabled_toolsets` 控制 agent 可访问的工具集：

```python
# 仅启用 Web 工具（浏览、搜索）
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    enabled_toolsets=["web"],
    quiet_mode=True,
)

# 启用除终端访问外的所有功能
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    disabled_toolsets=["terminal"],
    quiet_mode=True,
)
```

:::tip
当你需要一个功能最小化、受限的 agent 时（例如，仅用于研究机器人的 Web 搜索），使用 `enabled_toolsets`。当你需要大部分功能但需限制特定能力时（例如，在共享环境中禁用终端访问），使用 `disabled_toolsets`。
:::

---

## 多轮对话

通过将消息历史传回来维护多轮对话的状态：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

# 第一轮
result1 = agent.run_conversation("My name is Alice")
history = result1["messages"]

# 第二轮——agent 记住了上下文
result2 = agent.run_conversation(
    "What's my name?",
    conversation_history=history,
)
print(result2["final_response"])  # "Your name is Alice."
```

`conversation_history` 参数接受上一次结果的 `messages` 列表。agent 会在内部复制该列表，因此你的原始列表不会被修改。

---

## 保存轨迹数据

启用轨迹保存，以 ShareGPT 格式捕获对话——适用于生成训练数据或调试：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("Write a Python function to sort a list")
# 以 ShareGPT 格式保存到 trajectory_samples.jsonl
```

每次对话以单行 JSONL 的形式追加写入，便于从自动化运行中收集数据集。

---

## 自定义系统 Prompt

使用 `ephemeral_system_prompt` 设置自定义系统 prompt，用于引导 agent 的行为，但**不会**保存到轨迹文件中（保持训练数据的整洁）：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="You are a SQL expert. Only answer database questions.",
    quiet_mode=True,
)

response = agent.chat("How do I write a JOIN query?")
print(response)
```

这非常适合构建专用 agent——代码审查员、文档撰写员、SQL 助手——全部使用相同的底层工具。

---

## 批量处理

如需并行运行大量 prompt，Hermes 提供了 `batch_runner.py`，它可管理并发的 `AIAgent` 实例并进行适当的资源隔离：

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

每个 prompt 都有自己的 `task_id` 和隔离环境。如果需要自定义批处理逻辑，可以直接使用 `AIAgent` 构建：

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "Explain recursion",
    "What is a hash table?",
    "How does garbage collection work?",
]

def process_prompt(prompt):
    # 每个任务创建一个新的 agent 实例以保证线程安全
    agent = AIAgent(
        model="anthropic/claude-sonnet-4",
        quiet_mode=True,
        skip_memory=True,
    )
    return agent.chat(prompt)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

:::warning
务必为**每个线程或任务创建一个新的 `AIAgent` 实例**。agent 维护着内部状态（对话历史、工具会话、迭代计数器），这些状态不是线程安全的，不能共享。
:::

---

## 集成示例

### FastAPI 端点

```python
from fastapi import FastAPI
from pydantic import BaseModel
from run_agent import AIAgent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = "anthropic/claude-sonnet-4"

@app.post("/chat")
async def chat(request: ChatRequest):
    agent = AIAgent(
        model=request.model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    response = agent.chat(request.message)
    return {"response": response}
```

### Discord 机器人

```python
import discord
from run_agent import AIAgent

client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("!hermes "):
        query = message.content[8:]
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="discord",
        )
        response = agent.chat(query)
        await message.channel.send(response[:2000])

client.run("YOUR_DISCORD_TOKEN")
```

### CI/CD 流水线步骤

```python
#!/usr/bin/env python3
"""CI step: auto-review a PR diff."""
import subprocess
from run_agent import AIAgent

diff = subprocess.check_output(["git", "diff", "main...HEAD"]).decode()

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    disabled_toolsets=["terminal", "browser"],
)

review = agent.chat(
    f"Review this PR diff for bugs, security issues, and style problems:\n\n{diff}"
)
print(review)
```

---

## 关键构造函数参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `model` | `str` | `"anthropic/claude-opus-4.6"` | OpenRouter 格式的模型名称 |
| `quiet_mode` | `bool` | `False` | 抑制 CLI 输出 |
| `enabled_toolsets` | `List[str]` | `None` | 白名单指定工具集 |
| `disabled_toolsets` | `List[str]` | `None` | 黑名单指定工具集 |
| `save_trajectories` | `bool` | `False` | 将对话保存为 JSONL |
| `ephemeral_system_prompt` | `str` | `None` | 自定义系统 prompt（不保存到轨迹文件） |
| `max_iterations` | `int` | `90` | 每次对话的最大工具调用迭代次数 |
| `skip_context_files` | `bool` | `False` | 跳过加载 AGENTS.md 文件 |
| `skip_memory` | `bool` | `False` | 禁用持久化内存的读写 |
| `api_key` | `str` | `None` | API 密钥（回退到环境变量） |
| `base_url` | `str` | `None` | 自定义 API 端点 URL |
| `platform` | `str` | `None` | 平台提示（`"discord"`、`"telegram"` 等） |

---

## 重要说明

:::tip
- 如果不希望将工作目录中的 `AGENTS.md` 文件加载到系统 prompt 中，请设置 **`skip_context_files=True`**。
- 设置 **`skip_memory=True`** 可阻止 agent 读写持久化内存——推荐用于无状态 API 端点。
- `platform` 参数（如 `"discord"`、`"telegram"`）会注入平台特定的格式化提示，使 agent 适配其输出风格。
:::

:::warning
- **线程安全**：每个线程或任务创建一个 `AIAgent` 实例。切勿在并发调用中共享同一实例。
- **资源清理**：agent 在对话结束时会自动清理资源（终端会话、浏览器实例）。若在长期运行的进程中使用，请确保每次对话正常结束。
- **迭代限制**：默认的 `max_iterations=90` 较为宽松。对于简单的问答场景，建议适当降低该值（如 `max_iterations=10`），以防止工具调用循环失控并控制成本。
:::