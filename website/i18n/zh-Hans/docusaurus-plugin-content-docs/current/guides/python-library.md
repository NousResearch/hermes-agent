---
sidebar_position: 5
title: "将 Hermes 作为 Python 库使用"
description: "把 AIAgent 嵌入你自己的 Python 脚本、Web 应用或自动化流水线中 - 无需 CLI"
---

# 将 Hermes 作为 Python 库使用

Hermes 不只是一个 CLI 工具。你可以直接导入 `AIAgent`，并在自己的 Python 脚本、Web 应用或自动化流水线里以程序化方式使用它。本指南会带你完成整个过程。

---

## 安装

直接从仓库安装 Hermes：

```bash
pip install git+https://github.com/NousResearch/hermes-agent.git
```

或者使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv pip install git+https://github.com/NousResearch/hermes-agent.git
```

你也可以把它写进 `requirements.txt`：

```text
hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git
```

:::tip
当你把 Hermes 当作库来用时，CLI 使用的同一组环境变量也同样需要设置。至少要配置 `OPENROUTER_API_KEY`（如果你直接使用提供商访问，则可改为 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`）。
:::

---

## 基本用法

使用 Hermes 最简单的方式就是 `chat()` 方法 - 传入一条消息，拿回一个字符串：

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)
response = agent.chat("法国的首都是什么？")
print(response)
```

`chat()` 会在内部处理完整对话循环 - 工具调用、重试等等 - 只返回最终文本响应。

:::warning
当你把 Hermes 嵌入自己的代码时，一定要设置 `quiet_mode=True`。不然智能体会输出 CLI spinner、进度指示器和其他终端内容，干扰你应用本身的输出。
:::

---

## 完整的对话控制

如果你想对对话有更多控制，可以直接使用 `run_conversation()`。它会返回一个字典，包含完整响应、消息历史和元数据：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="搜索最近的 Python 3.13 新特性",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"Messages exchanged: {len(result['messages'])}")
```

返回的字典包含：
- **`final_response`** - 智能体最终的文本回复
- **`messages`** - 完整的消息历史（system、user、assistant、tool calls）

（你传入的 `task_id` 会存储在 agent 实例上，用于 VM 隔离，但不会出现在返回字典里。）

你也可以传入自定义系统提示词，用来覆盖这次调用的临时系统提示词：

```python
result = agent.run_conversation(
    user_message="解释快速排序",
    system_message="你是一名计算机科学导师。请使用简单类比。",
)
```

---

## 配置工具

可以使用 `enabled_toolsets` 或 `disabled_toolsets` 控制智能体能访问哪些工具集：

```python
# 只启用 web 工具（浏览、搜索）
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    enabled_toolsets=["web"],
    quiet_mode=True,
)

# 启用全部工具，但排除 terminal 访问
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    disabled_toolsets=["terminal"],
    quiet_mode=True,
)
```

:::tip
当你想要一个最小、锁定范围的智能体时，用 `enabled_toolsets`（例如只给研究机器人开放 web search）。当你想要大多数能力，但需要限制某些具体能力时，用 `disabled_toolsets`（例如在共享环境里禁止 terminal 访问）。
:::

---

## 多轮对话

通过把消息历史传回去，就能在多轮之间保留对话状态：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

# 第一轮
result1 = agent.run_conversation("我叫 Alice")
history = result1["messages"]

# 第二轮 - 智能体记住了上下文
result2 = agent.run_conversation(
    "我叫什么名字？",
    conversation_history=history,
)
print(result2["final_response"])  # "你的名字是 Alice。"
```

`conversation_history` 参数接受上一次结果里的 `messages` 列表。智能体会在内部复制它，所以你的原始列表不会被修改。

---

## 保存轨迹

启用轨迹保存，可以把对话记录成 ShareGPT 格式 - 适合生成训练数据或排查问题：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("写一个 Python 函数对列表排序")
# 以 ShareGPT 格式保存到 trajectory_samples.jsonl
```

每次对话都会以单独一行 JSONL 的形式追加，便于从自动化运行中收集数据集。

---

## 自定义系统提示词

使用 `ephemeral_system_prompt` 可以设置一个自定义系统提示词，引导智能体行为，但不会把它保存到轨迹文件中（这样可以保持训练数据干净）：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="你是一名 SQL 专家。只回答数据库问题。",
    quiet_mode=True,
)

response = agent.chat("我该如何写 JOIN 查询？")
print(response)
```

这非常适合构建专用智能体 - 代码审查员、文档撰写器、SQL 助手 - 都可以使用同一套底层工具。

---

## 批量处理

如果要并行运行很多提示词，Hermes 提供了 `batch_runner.py`。它会用合适的资源隔离来管理并发 `AIAgent` 实例：

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

每个提示词都会获得自己的 `task_id` 和隔离环境。如果你需要自定义批处理逻辑，也可以直接用 `AIAgent` 自己写：

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "解释递归",
    "什么是哈希表？",
    "垃圾回收是如何工作的？",
]

def process_prompt(prompt):
    # 为了线程安全，每个任务都创建一个新的 agent
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
务必为每个线程或任务创建一个**新的 `AIAgent` 实例**。智能体内部维护了对话历史、工具会话和迭代计数器，这些状态不能在线程之间共享。
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
"""CI 步骤：自动审查 PR diff。"""
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

## 关键构造参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `model` | `str` | `"anthropic/claude-opus-4.6"` | OpenRouter 格式的模型 |
| `quiet_mode` | `bool` | `False` | 抑制 CLI 输出 |
| `enabled_toolsets` | `List[str]` | `None` | 仅白名单中的工具集 |
| `disabled_toolsets` | `List[str]` | `None` | 黑名单中的工具集 |
| `save_trajectories` | `bool` | `False` | 将对话保存为 JSONL |
| `ephemeral_system_prompt` | `str` | `None` | 自定义系统提示词（不会保存到轨迹） |
| `max_iterations` | `int` | `90` | 每轮对话最大工具调用迭代次数 |
| `skip_context_files` | `bool` | `False` | 跳过加载 AGENTS.md 文件 |
| `skip_memory` | `bool` | `False` | 禁用持久化记忆的读写 |
| `api_key` | `str` | `None` | API key（会回退到环境变量） |
| `base_url` | `str` | `None` | 自定义 API 端点 URL |
| `platform` | `str` | `None` | 平台提示（例如 `"discord"`、`"telegram"` 等） |

---

## 重要说明

:::tip
- 如果你不想让当前工作目录下的 `AGENTS.md` 文件加载到系统提示词中，请设置 **`skip_context_files=True`**。
- 如果你希望智能体不读写持久化记忆，请设置 **`skip_memory=True`** - 这在无状态 API 端点里尤其推荐。
- `platform` 参数（例如 `"discord"`、`"telegram"`）会注入平台相关的格式提示，让智能体自动调整输出风格。
:::

:::warning
- **线程安全**：每个线程或任务都要创建一个 `AIAgent`。不要在并发调用之间共享同一个实例。
- **资源清理**：对话结束时，智能体会自动清理资源（终端会话、浏览器实例）。如果你运行的是长生命周期进程，请确保每个对话都能正常结束。
- **迭代上限**：默认的 `max_iterations=90` 已经很宽松。对于简单问答场景，可以考虑调低（例如 `max_iterations=10`），以防工具调用循环失控并控制成本。
:::