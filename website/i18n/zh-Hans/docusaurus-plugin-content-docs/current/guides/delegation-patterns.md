---
sidebar_position: 13
title: "委托与并行工作"
description: "何时以及如何使用子代理委托——并行研究、代码审查和多文件工作的模式"
---

# 委托与并行工作

Hermes 可以生成独立的子代理来并行处理任务。每个子代理都有自己的对话、终端会话和工具集。只有最终的摘要会返回——中间的工具调用永远不会进入你的上下文窗口。

有关完整的功能参考，请参阅[子代理委托](/docs/user-guide/features/delegation)。

---

## 何时委托

**适合委托的场景：**
- 需要大量推理的子任务（调试、代码审查、研究综合）
- 会产生大量中间数据从而淹没你的上下文的任务
- 并行的独立工作流（同时研究 A 和 B）
- 需要全新上下文的任务，你希望代理不带偏见地处理

**使用其他方案的情况：**
- 单个工具调用 → 直接使用工具
- 步骤间有逻辑的机械性多步骤工作 → 使用 `execute_code`
- 需要用户交互的任务 → 子代理不能使用 `clarify`
- 快速文件编辑 → 直接编辑
- 必须超越当前轮次的持久长期工作 → 使用 `cronjob` 或 `terminal(background=True, notify_on_complete=True)`。`delegate_task` 是**同步的**：如果父轮次被中断，活跃的子代理会被取消，它们的工作会被丢弃。

---

## 模式：并行研究

同时研究三个主题并获取结构化摘要：

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

Focus on recent developments and key players.
```

在幕后，Hermes 使用：

```python
delegate_task(tasks=[
    {
        "goal": "Research WebAssembly outside the browser in 2025",
        "context": "Focus on: runtimes (Wasmtime, Wasmer), cloud/edge use cases, WASI progress",
        "toolsets": ["web"]
    },
    {
        "goal": "Research RISC-V server chip adoption",
        "context": "Focus on: server chips shipping, cloud providers adopting, software ecosystem",
        "toolsets": ["web"]
    },
    {
        "goal": "Research practical quantum computing applications",
        "context": "Focus on: error correction breakthroughs, real-world use cases, key companies",
        "toolsets": ["web"]
    }
])
```

所有三个任务并发运行。每个子代理独立搜索网络并返回摘要。父代理然后将它们综合成一份连贯的简报。

---

## 模式：代码审查

将安全审查委托给一个全新上下文的子代理，使其不带先入为主地审视代码：

```
Review the authentication module at src/auth/ for security issues.
Check for SQL injection, JWT validation problems, password handling,
and session management. Fix anything you find and run the tests.
```

关键是 `context` 字段——它必须包含子代理需要的一切：

```python
delegate_task(
    goal="Review src/auth/ for security issues and fix any found",
    context="""Project at /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt.
    Auth files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    Test command: pytest tests/auth/ -v
    Focus on: SQL injection, JWT validation, password hashing, session management.
    Fix issues found and verify tests pass.""",
    toolsets=["terminal", "file"]
)
```

:::warning 上下文问题
子代理**完全不**了解你的对话。它们从零开始。如果你委托"修复我们讨论的那个 bug"，子代理根本不知道你说的是什么 bug。始终显式传递文件路径、错误消息、项目结构和约束条件。
:::

---

## 模式：比较备选方案

并行评估同一问题的多种方法，然后选择最佳方案：

```
I need to add full-text search to our Django app. Evaluate three approaches
in parallel:
1. PostgreSQL tsvector (built-in)
2. Elasticsearch via django-elasticsearch-dsl
3. Meilisearch via meilisearch-python

For each: setup complexity, query capabilities, resource requirements,
and maintenance overhead. Compare them and recommend one.
```

每个子代理独立研究一个选项。由于它们是隔离的，不存在交叉污染——每个评估都基于其自身的优点。父代理获取所有三个摘要并进行比较。

---

## 模式：多文件重构

将大型重构任务分配给并行子代理，每个子代理处理代码库的不同部分：

```python
delegate_task(tasks=[
    {
        "goal": "Refactor all API endpoint handlers to use the new response format",
        "context": """Project at /home/user/api-server.
        Files: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        Old format: return {"data": result, "status": "ok"}
        New format: return APIResponse(data=result, status=200).to_dict()
        Import: from src.responses import APIResponse
        Run tests after: pytest tests/handlers/ -v""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Update all client SDK methods to handle the new response format",
        "context": """Project at /home/user/api-server.
        Files: sdk/python/client.py, sdk/python/models.py
        Old parsing: result = response.json()["data"]
        New parsing: result = response.json()["data"] (same key, but add status code checking)
        Also update sdk/python/tests/test_client.py""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Update API documentation to reflect the new response format",
        "context": """Project at /home/user/api-server.
        Docs at: docs/api/. Format: Markdown with code examples.
        Update all response examples from old format to new format.
        Add a 'Response Format' section to docs/api/overview.md explaining the schema.""",
        "toolsets": ["terminal", "file"]
    }
])
```

:::tip
每个子代理都有自己的终端会话。只要它们编辑不同的文件，就可以在同一个项目目录上工作而不会相互干扰。如果两个子代理可能触及同一个文件，请在并行工作完成后自己处理该文件。
:::

---

## 模式：先收集再分析

使用 `execute_code` 进行机械性的数据收集，然后委托需要大量推理的分析工作：

```python
# Step 1: Mechanical gathering (execute_code is better here — no reasoning needed)
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["AI funding Q1 2026", "AI startup acquisitions 2026", "AI IPOs 2026"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# Extract full content from top 5 most relevant
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# Save for the analysis step
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"Collected {len(results)} results, extracted {len(content['results'])} pages")
""")

# Step 2: Reasoning-heavy analysis (delegation is better here)
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M.""",
    toolsets=["terminal", "file"]
)
```

这通常是最高效的模式：`execute_code` 廉价地处理 10+ 个顺序工具调用，然后子代理用干净的上下文完成单个昂贵的推理任务。

---

## 工具集选择

根据子代理的需要选择工具集：

| 任务类型 | 工具集 | 原因 |
|-----------|----------|-----|
| 网络研究 | `["web"]` | 仅 web_search + web_extract |
| 代码工作 | `["terminal", "file"]` | Shell 访问 + 文件操作 |
| 全栈 | `["terminal", "file", "web"]` | 除消息传递外的所有功能 |
| 只读分析 | `["file"]` | 只能读取文件，无 Shell |

限制工具集可以保持子代理的专注并防止意外副作用（例如研究子代理运行 Shell 命令）。

---

## 约束

- **默认 3 个并行任务**：批处理默认使用 3 个并发子代理（可通过 config.yaml 中的 `delegation.max_concurrent_children` 配置，无硬性上限，只有 1 的下限）
- **嵌套委托是选择加入的**：叶子子代理（默认）不能调用 `delegate_task`、`clarify`、`memory`、`send_message` 或 `execute_code`。编排器子代理（`role="orchestrator"`）保留 `delegate_task` 以供进一步委托，但仅在 `delegation.max_spawn_depth` 提高到默认值 1 以上时（支持 1-3）；其他四个仍被阻止。通过 `delegation.orchestrator_enabled: false` 全局禁用。

### 调整并发和深度

| 配置 | 默认值 | 范围 | 效果 |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | 每次 `delegate_task` 调用的并行批处理大小 |
| `max_spawn_depth` | 1 | 1-3 | 可以产生进一步委托的委托层级数 |

示例：运行 30 个带嵌套子代理的并行工作器：

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **独立的终端**——每个子代理都有自己的终端会话，具有独立的工作目录和状态
- **无对话历史**——子代理只能看到父代理调用 `delegate_task` 时传递的 `goal` 和 `context`
- **默认 50 次迭代**——为简单任务设置较低的 `max_iterations` 以节省成本
- **不持久**——`delegate_task` 是同步的，在父轮次内运行。如果父轮次被中断（新用户消息、`/stop`、`/new`），所有活跃子代理都会被取消（`status="interrupted"`），它们的工作会被丢弃。对于必须超越当前轮次的工作，使用 `cronjob` 或 `terminal(background=True, notify_on_complete=True)`。

---

## 技巧

**在目标中要具体。**"修复 bug" 太模糊。"修复 api/handlers.py 第 47 行的 TypeError，其中 process_request() 从 parse_body() 接收 None" 给子代理提供了足够的工作信息。

**包含文件路径。** 子代理不知道你的项目结构。始终包含相关文件的绝对路径、项目根目录和测试命令。

**使用委托进行上下文隔离。** 有时你需要一个全新的视角。委托迫使你清晰地阐述问题，而子代理会以你的对话中积累的假设之外的角度来处理它。

**检查结果。** 子代理的摘要只是摘要。如果子代理说"已修复 bug 且测试通过"，请自己运行测试或读取差异来验证。

---

*有关完整的委托参考——所有参数、ACP 集成和高级配置——请参阅[子代理委托](/docs/user-guide/features/delegation)。*
