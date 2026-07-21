---
sidebar_position: 8
title: "程序化集成"
description: "从外部程序驱动 hermes-agent 的三种协议：ACP、TUI gateway JSON-RPC 以及兼容 OpenAI 的 HTTP API"
---

# 程序化集成

Hermes 提供三种协议，供外部程序驱动 agent——IDE 插件、自定义 UI、CI 流水线、嵌入式子 agent。根据你的传输方式和消费端选择合适的协议。

| 协议 | 传输方式 | 适用场景 | 定义位置 |
|----------|-----------|----------|------------|
| **ACP** | JSON-RPC over stdio | 已支持 [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol) 的 IDE 客户端（VS Code、Zed、JetBrains） | `acp_adapter/` |
| **TUI gateway** | JSON-RPC over stdio（或 WebSocket） | 需要精细控制会话、slash 命令、审批及流式事件的自定义宿主 | `tui_gateway/server.py` |
| **API server** | HTTP + Server-Sent Events | 兼容 OpenAI 的前端（Open WebUI、LobeHub、LibreChat……）及语言无关的 Web 客户端 | `gateway/platforms/api_server.py` |

三种协议均驱动同一个 `AIAgent` 核心，区别仅在于线路格式和所暴露的功能集。

---

## ACP（Agent Client Protocol）

`hermes acp` 启动一个基于 stdio 的 JSON-RPC 服务器，使用 ACP 协议。已在 VS Code（Zed Industries 的 ACP 扩展）、Zed 以及所有安装了 ACP 插件的 JetBrains IDE 中投入生产使用。

暴露的能力：会话创建、prompt（提示词）提交、流式 agent 消息块、工具调用事件、权限请求、会话 fork、取消及身份验证。工具输出会被渲染为 IDE 可理解的 ACP `Diff`/`ToolCall` 内容块。

完整生命周期、事件桥接及审批流程：[ACP 内部机制](./acp-internals)。

```bash
hermes acp                  # 在 stdio 上提供 ACP 服务
hermes acp --bootstrap      # 打印适用于支持 ACP 的 IDE 的安装代码片段
```

---

## TUI Gateway JSON-RPC

`tui_gateway/server.py` 是 Ink TUI（`hermes --tui`）和嵌入式仪表板 PTY 桥接所使用的协议。任何外部宿主均可通过 stdio（或经由 `tui_gateway/ws.py` 的 WebSocket）使用相同协议。

### 方法目录（精选）

```
prompt.submit           prompt.background       session.steer
session.create          session.list            session.interrupt
session.history         session.compress        session.branch
session.title           session.usage           session.status
clarify.respond         sudo.respond            secret.respond
approval.respond        config.set / config.get commands.catalog
command.resolve         command.dispatch        cli.exec
reload.mcp              reload.env              process.stop
delegation.status       subagent.interrupt      spawn_tree.save / list / load
terminal.resize         clipboard.paste         image.attach
```

### 流式返回的事件

`message.delta`、`message.complete`、`tool.start`、`tool.progress`、`tool.complete`、`approval.request`、`clarify.request`、`sudo.request`、`secret.request`、`gateway.ready`，以及会话生命周期和错误事件。

### Pi 风格 RPC 映射

Pi-mono RPC 规范（[issue #360](https://github.com/NousResearch/hermes-agent/issues/360)）中的每条命令均有对应的 TUI gateway 等价项：

| Pi 命令 | Hermes 等价项 |
|------------|-------------------|
| `prompt` | `prompt.submit`（或 ACP `session/prompt`） |
| `steer` | `session.steer` |
| `follow_up` | 在当前轮次结束后排队的 `prompt.submit` |
| `abort` | `session.interrupt` |
| `set_model` | 通过 `command.dispatch` 执行 `/model <provider:model>`（会话中途生效，持久化） |
| `compact` | `session.compress` |
| `get_state` | `session.status` |
| `get_messages` | `session.history` |
| `switch_session` | `session.resume` |
| `fork` | `session.branch` |
| `ui_request` / `ui_response` | `clarify.respond` / `sudo.respond` / `secret.respond` / `approval.respond` |

---

## 兼容 OpenAI 的 API Server

`gateway/platforms/api_server.py` 通过 HTTP 暴露 Hermes，供任何已支持 OpenAI 格式的客户端使用。适用于需要 Web 前端、curl 驱动的 CI 运行器或非 Python 消费端的场景。

端点：

```
POST /v1/chat/completions        OpenAI Chat Completions（通过 SSE 流式传输）
POST /v1/responses               OpenAI Responses API（有状态）
POST /v1/runs                    启动一次运行，返回 run_id（202）
GET  /v1/runs/{id}               运行状态
GET  /v1/runs/{id}/events        生命周期事件的 SSE 流
POST /v1/runs/{id}/approval      解决待处理的审批
POST /v1/runs/{id}/stop          中断运行
GET  /v1/capabilities            机器可读的功能标志
GET  /v1/models                  列出 hermes-agent
GET  /health, /health/detailed
```

配置、请求头（`X-Hermes-Session-Id`、`X-Hermes-Session-Key`）及前端接入：[API Server](../user-guide/features/api-server)。

---

## 该选哪个？

- **正在编写 IDE 插件，且 IDE 已支持 ACP** → 选 ACP。IDE 侧无需任何协议工作。
- **正在编写自定义桌面 / Web / TUI 宿主，且需要 Hermes 的全部功能**（slash 命令、审批、clarify、多 agent、会话分支）→ 选 TUI gateway JSON-RPC。
- **需要任意兼容 OpenAI 的前端、语言无关的 HTTP 客户端或 curl 驱动的自动化** → 选 API server。
- **需要在 Python 进程内嵌入，不想启动子进程** → 直接导入 `run_agent.AIAgent`。参见 [Agent Loop](./agent-loop)。

---

## 模型热切换

会话中途切换模型在所有接入方式上均可用——底层均为 `/model` slash 命令。

- **CLI / TUI：** `/model claude-sonnet-4` 或 `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI gateway RPC：** 使用 `{"command": "/model claude-sonnet-4"}` 调用 `command.dispatch`
- **ACP：** IDE 将 slash 命令作为 prompt 发送，agent 负责分发
- **API server：** 在请求体中包含 `model` 字段，或设置 `X-Hermes-Model`

内置 provider 感知解析（相同的模型名称会根据当前 provider 自动选择正确格式）。参见 `hermes_cli/model_switch.py`。

---

## 关于 `--mode rpc` 的说明

Hermes 没有 `--mode rpc` 标志。上述三种协议已覆盖所有使用场景——ACP 用于 IDE 协议客户端，TUI gateway 用于 stdio JSON-RPC 宿主，API server 用于 HTTP。如果你发现上述协议均无法满足的真实需求，请提交 issue 并说明你正在构建的具体消费端。