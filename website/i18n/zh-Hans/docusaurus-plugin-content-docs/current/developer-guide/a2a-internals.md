---
sidebar_position: 3
title: "A2A 内部原理"
description: "A2A 适配器的工作方式：生命周期、context 会话、事件桥接与 Agent Card"
---

# A2A 内部原理

A2A 适配器将 Hermes 的同步 `AIAgent` 包装成一个基于 [`a2a-sdk`](https://a2a-protocol.org) 的异步 JSON-RPC + SSE HTTP 服务器。它与 [ACP 适配器](./acp-internals.md) 同构：一个协议服务器类驱动相同的 `AIAgent` 回调接缝，只是把事件翻译为 A2A 而非 ACP。

关键实现文件：

- `plugins/platforms/a2a/adapter.py`
- `plugins/platforms/a2a/entry.py`
- `plugins/platforms/a2a/card.py`
- `plugins/platforms/a2a/executor.py`
- `plugins/platforms/a2a/sessions.py`
- `plugins/platforms/a2a/events.py`

## 启动流程

```text
独立：hermes-a2a / python -m plugins.platforms.a2a
  -> plugins.platforms.a2a.entry.main()
  -> 在服务器启动前解析 --version / --check
  -> 加载 ~/.hermes/.env
  -> 发现 MCP 工具（tools.mcp_tool.discover_mcp_tools）
  -> build_app()：AgentCard + DefaultRequestHandler + BoundedTaskStore
                  -> A2AStarletteApplication(...).build()
  -> uvicorn.run(app)

网关：发现插件 -> ctx.register_platform(name="a2a", ...)
  -> A2AAdapter.connect() -> uvicorn.Server.serve()
```

Agent Card 位于 `/.well-known/agent-card.json`；JSON-RPC 端点位于 `/`。

## 主要组件

### `HermesAgentExecutor`

`plugins/platforms/a2a/executor.py` 实现 a2a-sdk 的 `AgentExecutor` 接口（`execute` / `cancel`）。

`execute()`：

- 读取用户文本，并通过 `new_task` 解析（或创建）task
- 创建 `TaskUpdater`，将 task 标记为 `working`
- 解析 `contextId` 对应的 Hermes 会话
- 将 AIAgent 回调连接到 A2A 事件
- 在专用的有界工作线程池中运行 `AIAgent.run_conversation`
- 把最终响应作为产物发出，然后将 task 标记为 `completed`

`cancel()` 向会话发出信号并发出 `canceled` 状态。

### `ContextSessionStore`

`plugins/platforms/a2a/sessions.py` 将 `contextId` 映射到一个 `HermesSession`（一个 `AIAgent`、其滚动历史和一个取消事件）。它是线程安全的，惰性创建 agent，并接受 `agent_factory` 以便测试注入伪实现。真实构建会组合现有的编码/研究工具集，而不是增加 A2A 专用的核心 profile。

### 事件桥接

`plugins/platforms/a2a/events.py` 将 AIAgent 回调转换为 `TaskUpdater` 事件：

- `stream_delta_callback` -> 带文本块的 `working` 状态
- `tool_progress_callback` -> 标记 `hermes/kind=tool-call` 的 `working` 状态
- `step_callback` -> 标记 `hermes/kind=tool-result` 的 `working` 状态
- `reasoning_callback` -> 标记 `hermes/kind=reasoning` 的 `working` 状态
- 最终响应 -> `add_artifact(...)` + `complete()`

由于 `AIAgent` 运行在工作线程中，而 A2A 事件队列存在于服务器事件循环上，桥接使用以下方式编排每个异步更新：

```python
asyncio.run_coroutine_threadsafe(...)
```

并在其上短暂阻塞，从而使更新相对于 agent 自身进度保持有序（且所有 working 更新都在最终产物之前送达）。失败的更新会被记录并吞掉 —— 它绝不会中止该回合。

### Agent Card

`plugins/platforms/a2a/card.py` 根据 Hermes 版本加上精选的技能列表（通用 agent、研究）动态构建 `AgentCard`。与 ACP 签入的 `acp_registry/agent.json` 不同，A2A 的 card 在服务器启动时构建（没有需要保持同步的静态 JSON 清单），并在每次 `/.well-known/agent-card.json` 请求时重新序列化。

## 任务生命周期

```text
message/send | message/stream
  -> DefaultRequestHandler -> HermesAgentExecutor.execute()
     -> new_task()（若无当前 task）-> 入队 Task
     -> TaskUpdater.start_work()              [状态: working]
     -> to_thread(AIAgent.run_conversation)
          stream_delta / tool_progress / step -> working 状态更新
     -> add_artifact(final_response)          [artifact-update]
     -> complete()                            [状态: completed]
```

## 取消

`cancel()` 设置会话取消事件，并在可用时调用 `agent.interrupt()`，然后发出终态 `canceled` 状态。

## 当前限制

- 内存内的 task 存储和会话：两者在进程重启时都会丢失。
- 端点以未认证方式提供；请绑定 `127.0.0.1` 或在代理/认证层之后提供（见用户指南中的安全提示）。
- 推送通知、持久化 task 存储以及 gRPC / HTTP+JSON 传输不在本次范围内。（`tasks/resubscribe` 可通过 SDK 默认处理器路由，但本适配器的测试未覆盖。）
- 输入为纯文本；该接缝后续可接受更丰富的 part。

## 相关文件

- `tests/a2a/` —— A2A 测试套件
- `plugins/platforms/a2a/plugin.yaml` —— 内置平台清单
- `hermes_cli/config.py` —— 默认 `a2a` 配置值
- `pyproject.toml` —— `[a2a]` 可选依赖 + `hermes-a2a` 脚本
- `.plans/a2a-protocol.md` —— 设计与协议到 Hermes 的映射
