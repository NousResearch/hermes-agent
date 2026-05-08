---
sidebar_position: 5
title: "将 Hermes 作为 Python 库使用"
description: "在你的 Python 脚本、Web 应用或自动化管道中嵌入 AIAgent — 无需 CLI"
---

# 将 Hermes 作为 Python 库使用

Hermes 不只是 CLI 工具。你可以直接导入 `AIAgent` 并在自己的 Python 脚本、Web 应用或自动化管道中以编程方式使用。本指南展示如何操作。

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

你也可以在 `requirements.txt` 中固定：

```text
hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git
```

:::tip
使用 Hermes 作为库时需要与 CLI 相同的环境变量。至少设置 `OPENROUTER_API_KEY`（或如果使用直接提供商访问则设置 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`）。
:::

---

## 基本用法

使用 Hermes 最简单的方式是 `chat()` 方法 — 传入消息，获取字符串返回：

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)
response = agent.chat("法国的首都是什么？")
print(response)
```

`chat()` 在内部处理完整对话循环 — 工具调用、重试、一切 — 只返回最终文本响应。

:::warning
将 Hermes 嵌入自己的代码时始终设置 `quiet_mode=True`。没有它，智能体打印 CLI 旋转器、进度指示器和其他终端输出，会弄乱你的应用输出。
:::

---

## 完整对话控制

想要更多对话控制，直接使用 `run_conversation()`。它返回包含完整响应、消息历史和元数据的字典：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="搜索最近的 Python 3.13 功能",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"交换消息数: {len(result['messages'])}")
```

返回的字典包含：
- **`final_response`** — 智能体的最终文本回复
- **`messages`** — 完整消息历史（系统、用户、助手、工具调用）
- **`task_id`** — 用于 VM 隔离的任务标识符

你也可以传入自定义系统消息，覆盖该调用的临时系统 prompt：

```python
result = agent.run_conversation(
    user_message="解释快速排序",
    system_message="你是一位计算机科学导师。使用简单的类比。",
)
```

---

## 配置工具

使用 `enabled_toolsets` 或 `disabled_toolsets` 控制智能体可以访问哪些工具集：

```python
# 只启用 Web 工具（浏览、搜索）
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
当你想获得最小化、锁定的智能体时使用 `enabled_toolsets`（例如，仅用于网络搜索的研究机器人）。当你想获得大部分能力但需要限制特定功能时使用 `disabled_toolsets`（例如，在共享环境中无终端访问）。
:::

---

## 多轮对话

通过传回消息历史维护多轮对话状态：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

# 第一轮
result1 = agent.run_conversation("我的名字是 Alice")
history = result1["messages"]

# 第二轮 — 智能体记住上下文
result2 = agent.run_conversation(
    "我叫什么名字？",
    conversation_history=history,
)
print(result2["final_response"])  # "你的名字是 Alice。"
```

`conversation_history` 参数接受先前结果的 `messages` 列表。智能体在内部复制它，所以你的原始列表永远不会被修改。

---

## 保存轨迹

启用轨迹保存以 ShareGPT 格式捕获对话 — 对生成训练数据或调试有用：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("写一个 Python 函数排序列表")
# 保存到 trajectory_samples.jsonl，ShareGPT 格式
```

每个对话作为单个 JSONL 行追加，便于从自动化运行中收集数据集。

---

## 自定义系统 Prompt

使用 `ephemeral_system_prompt` 设置自定义系统 prompt，引导智能体行为但**不**保存到轨迹文件（保持训练数据干净）：

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="你是一位 SQL 专家。只回答数据库问题。",
    quiet_mode=True,
)

response = agent.chat("如何编写 JOIN 查询？")
print(response)
```

这非常适合构建专业智能体 — 代码审查员、文档编写者、SQL 助手 — 都使用相同的底层工具。

---

## 批量处理

要并行运行多个提示，Hermes 包含 `batch_runner.py`。它管理并发的 `AIAgent` 实例，具有适当的资源隔离：

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

每个提示获得自己的 `task_id` 和隔离环境。如果需要自定义批量逻辑，你可以直接使用 `AIAgent` 构建：

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "解释递归",
    "什么是哈希表？",
    "垃圾回收如何工作？",
]

def process_prompt(prompt):
    # 每个任务创建新智能体以保证线程安全
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
始终为每个线程或任务创建**新的 `AIAgent` 实例**。智能体维护内部状态（对话历史、工具会话、迭代计数器），跨并发调用共享不是线程安全的。
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

### CI/CD 管道步骤

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
    f"审查此 PR diff 的 bug、安全问题和样式问题：\n\n{diff}"
)
print(review)
```

---

## 主要构造函数参数

| 参数 | 类型 | 默认 | 说明 |
|-----------|------|---------|-------------|
| `model` | `str` | `"anthropic/claude-opus-4.6"` | OpenRouter 格式的模型 |
| `quiet_mode` | `bool` | `False` | 抑制 CLI 输出 |
| `enabled_toolsets` | `List[str]` | `None` | 白名单特定工具集 |
| `disabled_toolsets` | `List[str]` | `None` | 黑名单特定工具集 |
| `save_trajectories` | `bool` | `False` | 保存对话到 JSONL |
| `ephemeral_system_prompt` | `str` | `None` | 自定义系统 prompt（不保存到轨迹） |
| `max_iterations` | `int` | `90` | 每个对话的最大工具调用迭代次数 |
| `skip_context_files` | `bool` | `False` | 跳过加载 AGENTS.md 文件 |
| `skip_memory` | `bool` | `False` | 禁用持久记忆读写 |
| `api_key` | `str` | `None` | API key（回退到环境变量） |
| `base_url` | `str` | `None` | 自定义 API 端点 URL |
| `platform` | `str` | `None` | 平台提示（`"discord"`、`"telegram"` 等） |

---

## 重要说明

:::tip
- 如果不想将工作目录中的 `AGENTS.md` 文件加载到系统 prompt，设置 **`skip_context_files=True`**。
- 设置 **`skip_memory=True`** 防止智能体读取或写入持久记忆 — 推荐用于无状态 API 端点。
- `platform` 参数（例如 `"discord"`、`"telegram"`）注入平台特定格式提示，让智能体调整输出样式。
:::

:::warning
- **线程安全**：为每个线程或任务创建一个 `AIAgent`。永远不要跨并发调用共享实例。
- **资源清理**：对话结束时智能体自动清理资源（终端会话、浏览器实例）。如果你在长寿命进程中运行，确保每个对话正常完成。
- **迭代限制**：默认 `max_iterations=90` 很慷慨。对于简单问答用例，考虑降低它（例如 `max_iterations=10`）以防止失控的工具调用循环和控制成本。
:::
