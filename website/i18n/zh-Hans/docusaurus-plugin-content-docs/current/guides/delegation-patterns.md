---
sidebar_position: 13
title: "委托与并行工作"
description: "何时以及如何使用子代理委托——并行研究、代码审查和多文件工作的模式"
---

# 委托与并行工作

Hermes 可以生成隔离的子代理来并行处理任务。每个子代理拥有独立的对话、终端会话和工具集。只有最终摘要会返回——中间工具调用不会进入你的上下文窗口。

完整功能参考，请参阅[子代理委托](/user-guide/features/delegation)。

---

## 何时委托

**适合委托的场景：**
- 推理密集型子任务（调试、代码审查、研究综合）
- 会用中间数据淹没上下文的任务
- 并行独立工作流（同时进行研究 A 和研究 B）
- 需要代理以无偏见方式处理的全新上下文任务

**使用其他方式的场景：**
- 单次工具调用 → 直接使用工具
- 步骤间有逻辑的机械性多步骤工作 → `execute_code`
- 需要用户交互的任务 → 子代理无法使用 `clarify`
- 快速文件编辑 → 直接操作
- 必须在当前轮次结束后继续运行的持久性长任务 → `cronjob` 或 `terminal(background=True, notify_on_complete=True)`。`delegate_task` 是**同步**的：若父轮次被中断，活跃的子代理将被取消，其工作将被丢弃。

---

## 模式：并行研究

同时研究三个主题并获取结构化摘要：

```
并行研究以下三个主题：
1. WebAssembly 在浏览器之外的现状
2. 2025 年 RISC-V 服务器芯片的采用情况
3. 量子计算的实际应用

重点关注近期进展和关键参与者。
```

在后台，Hermes 使用：

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

三个任务并发运行。每个子代理独立搜索网络并返回摘要。父代理随后将它们综合成一份连贯的简报。

---

## 模式：代码审查

将安全审查委托给一个全新上下文的子代理，让它以无先入之见的方式审查代码：

```
审查 src/auth/ 中的认证模块，检查安全问题。
检查 SQL 注入、JWT 验证问题、密码处理
和会话管理。修复发现的问题并运行测试。
```

关键在于 `context` 字段——它必须包含子代理所需的一切信息：

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
子代理对你的对话**一无所知**。它们从完全空白的状态开始。如果你委托"修复我们讨论的那个 bug"，子代理根本不知道你指的是哪个 bug。务必明确传递文件路径、错误信息、项目结构和约束条件。
:::

---

## 模式：比较备选方案

并行评估同一问题的多种解决方案，然后选出最佳方案：

```
我需要为 Django 应用添加全文搜索。并行评估三种方案：
1. PostgreSQL tsvector（内置）
2. 通过 django-elasticsearch-dsl 使用 Elasticsearch
3. 通过 meilisearch-python 使用 Meilisearch

对每种方案评估：配置复杂度、查询能力、资源需求
和维护开销。比较后推荐一种。
```

每个子代理独立研究一个选项。由于它们相互隔离，不存在交叉干扰——每项评估都基于自身的优缺点。父代理获取全部三份摘要后进行比较。

---

## 模式：多文件重构

将大型重构任务拆分给并行子代理，每个子代理负责代码库的不同部分：

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
每个子代理拥有独立的终端会话。只要它们编辑不同的文件，就可以在同一项目目录中工作而互不干扰。如果两个子代理可能修改同一文件，请在并行工作完成后自行处理该文件。
:::

---

## 模式：先收集后分析

使用 `execute_code` 进行机械性数据收集，然后委托推理密集型分析：

```python
# 第一步：机械性收集（此处 execute_code 更合适——无需推理）
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

# 第二步：推理密集型分析（此处委托更合适）
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M.""",
    toolsets=["terminal", "file"]
)
```

这通常是最高效的模式：`execute_code` 以低成本处理 10 余次顺序工具调用，然后子代理在干净的上下文中完成单次高成本推理任务。

---

## 工具集选择

根据子代理的需求选择工具集：

| 任务类型 | 工具集 | 原因 |
|-----------|----------|-----|
| 网络研究 | `["web"]` | 仅 web_search + web_extract |
| 代码工作 | `["terminal", "file"]` | Shell 访问 + 文件操作 |
| 全栈 | `["terminal", "file", "web"]` | 除消息功能外的全部工具 |
| 只读分析 | `["file"]` | 只能读取文件，无 Shell |

限制工具集可使子代理保持专注，并防止意外副作用（例如研究子代理执行 Shell 命令）。

---

## 约束条件

- **默认 3 个并行任务**：批次默认并发 3 个子代理（可通过 config.yaml 中的 `delegation.max_concurrent_children` 配置，无硬性上限，最低为 1）
- **嵌套委托需显式启用**：叶子子代理（默认）无法调用 `delegate_task`、`clarify`、`memory`、`send_message` 或 `execute_code`。编排器子代理（`role="orchestrator"`）保留 `delegate_task` 以支持进一步委托，但仅在 `delegation.max_spawn_depth` 高于默认值 1 时生效（支持 1-3）；其余四项仍被禁用。可通过 `delegation.orchestrator_enabled: false` 全局禁用。

### 调整并发数与深度

| 配置项 | 默认值 | 范围 | 效果 |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | 每次 `delegate_task` 调用的并行批次大小 |
| `max_spawn_depth` | 1 | 1-3 | 可进一步生成子代理的委托层级数 |

示例：运行 30 个并行 worker 并启用嵌套子代理：

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **独立终端** — 每个子代理拥有独立的终端会话，具有独立的工作目录和状态
- **无对话历史** — 子代理只能看到父代理调用 `delegate_task` 时传入的 `goal` 和 `context`
- **默认 50 次迭代** — 对简单任务设置较低的 `max_iterations` 以节省成本
- **非持久性** — `delegate_task` 是同步的，在父轮次内运行。若父轮次被中断（新用户消息、`/stop`、`/new`），所有活跃子代理将被取消（`status="interrupted"`），其工作将被丢弃。对于必须在当前轮次结束后继续运行的工作，请使用 `cronjob` 或 `terminal(background=True, notify_on_complete=True)`。

---

## 技巧

**目标要具体。** "修复 bug"过于模糊。"修复 api/handlers.py 第 47 行的 TypeError，该错误由 parse_body() 向 process_request() 返回 None 引起"才能给子代理足够的信息。

**包含文件路径。** 子代理不了解你的项目结构。务必提供相关文件的绝对路径、项目根目录和测试命令。

**利用委托实现上下文隔离。** 有时你需要全新的视角。委托迫使你清晰地阐述问题，而子代理会在没有对话中积累的假设前提下处理它。

**核验结果。** 子代理的摘要只是摘要。如果子代理说"修复了 bug 且测试通过"，请自行运行测试或查看 diff 来验证。

---

*完整的委托参考——所有参数、ACP 集成和高级配置——请参阅[子代理委托](/user-guide/features/delegation)。*