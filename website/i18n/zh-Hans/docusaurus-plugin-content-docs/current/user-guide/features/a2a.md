---
sidebar_position: 12
title: "A2A（Agent2Agent）服务器"
description: "将 Hermes Agent 作为 A2A 服务器运行，让其他 agent 通过 JSON-RPC + SSE 发现它并委派任务"
---

# A2A（Agent2Agent）服务器

Hermes Agent 可作为 [A2A](https://a2a-protocol.org) 服务器运行，让任何兼容 A2A 的客户端或对等 agent 通过 HTTP(S) 发现它并委派任务。MCP 把 agent 连接到 _工具_，而 A2A 把 agent 连接到 _其他 agent_ —— 因此 A2A 让 Hermes 成为 LangGraph、CrewAI、Google ADK、`a2a-inspector` 乃至另一个 Hermes 等编排器可调用的工作单元。

它是 [ACP](./acp.md)（通过 stdio 的编辑器集成）和 MCP 服务器（通过 MCP 暴露工具）的姊妹能力：A2A 是 **作为其他 agent 的远程 agent 的 Hermes**。

## Hermes 在 A2A 模式下暴露的内容

- 位于 `/.well-known/agent-card.json` 的 **Agent Card**，描述 Hermes 的名称、版本、能力（流式）和技能。
- `message/send` —— 同步请求/响应（返回已完成的 task）。
- `message/stream` —— 通过 Server-Sent Events 流式传输 task 状态更新和产物（artifact）。
- `tasks/get` 和 `tasks/cancel`。

每个回合会组合 Hermes 现有的编码/研究工具集（shell、文件系统、网页/浏览器、记忆、待办、skills、`execute_code`、`delegate_task`），无需增加 A2A 专用的核心工具集，也不会引入交互式消息或音频功能。

## 安装

正常安装 Hermes 后，添加 A2A 扩展：

```bash
pip install -e '.[a2a]'
```

这将安装 `a2a-sdk[http-server]` 依赖并启用：

- `hermes-a2a`
- `python -m plugins.platforms.a2a`
- 内置的 A2A 网关平台插件

## 启动 A2A 服务器

```bash
hermes-a2a
```

```bash
python -m plugins.platforms.a2a
```

默认情况下，服务器绑定 `127.0.0.1:9100`。Agent Card 位于 `/.well-known/agent-card.json`，JSON-RPC 端点位于 `/`。

```bash
hermes-a2a --host 127.0.0.1 --port 9100
hermes-a2a --public-url https://agents.example.com/hermes/   # 在 card 中公布的 URL
```

非交互式检查：

```bash
hermes-a2a --version
hermes-a2a --check
```

:::warning 暴露到网络
A2A 端点是 **未认证的**。绑定 `--host 0.0.0.0` 会把 Hermes —— 及其 shell/文件系统工具 —— 暴露给任何能访问该端口的对象。只能在你掌控的反向代理或认证层之后这样做。服务器在以 `0.0.0.0` 启动时会记录一条警告。
:::

## 与服务器通信

先获取 card，再发送消息。使用 `curl`：

```bash
curl http://127.0.0.1:9100/.well-known/agent-card.json

curl http://127.0.0.1:9100/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": "1", "method": "message/send",
  "params": {"message": {"role": "user", "kind": "message", "messageId": "m1",
    "parts": [{"kind": "text", "text": "总结这个仓库做了什么。"}]}}
}'
```

`message/stream` 使用相同的请求体，但 `"method": "message/stream"`（决定流式的是方法名；`Accept: text/event-stream` 头是客户端的惯例性礼貌）。响应是 `TaskStatusUpdateEvent`（working）和 `TaskArtifactUpdateEvent`（结果）的 SSE 流，最终以 `completed` 状态结束。

## 会话连续性

A2A 的 `contextId` 映射到一个持久的 Hermes 会话：每个 context 对应一个 `AIAgent` 及其滚动历史。复用同一 `contextId` 的后续消息会延续同一段对话。每个 `taskId` 是 context 内的一个回合。会话在服务器进程的生命周期内保存于内存中。

## 配置与凭据

A2A 模式使用与 CLI 相同的 Hermes profile 配置。行为设置位于 `~/.hermes/config.yaml`：

```yaml
a2a:
  enabled: false # true 时随 `hermes gateway` 启动
  host: 127.0.0.1
  port: 9100
  public_url: null
  max_concurrency: 16
  max_sessions: 512
  max_tasks: 2048
  max_task_history: 100
  tool_io: preview # preview | none | full

platform_toolsets:
  a2a: [web, terminal, file, vision, skills, browser, todo, memory,
        session_search, code_execution, delegation]
```

`tool_io: preview` 会限制对等 agent 可见的工具参数/结果，`none` 仅发送工具名，`full` 发送未截断值。独立启动时，命令行的 host、port 和 public URL 标志优先于配置文件。

`platform_toolsets.a2a` 使用与其他网关平台相同的工具配置机制。`agent.disabled_toolsets` 始终优先，因此运维人员可以全局移除 `terminal` 或 `file` 等敏感能力。由于 A2A 没有交互式审批往返，需要确认的危险命令会被拒绝，而不会从服务器标准输入读取。

也可以通过现有网关生命周期启动插件：

```bash
hermes config set a2a.enabled true
hermes gateway run
```

凭据仍保存在正常的密钥存储中：

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`

provider 解析使用 Hermes 正常的运行时解析器，因此 A2A 继承当前配置的 provider 和凭据。使用 `hermes model` 或编辑 `~/.hermes/.env` 来配置凭据。

## 故障排查

### 服务器启动但任务立即失败

验证依赖和 provider 设置：

```bash
hermes-a2a --check
hermes model
hermes doctor
```

### 客户端无法发现 agent

确认 card 可访问，且客户端指向基础 URL（而非 card URL）：

```bash
curl -fsS http://127.0.0.1:9100/.well-known/agent-card.json
```

## 另见

- [A2A 内部原理](../../developer-guide/a2a-internals.md)
- [ACP 编辑器集成](./acp.md)
- [Provider 运行时解析](../../developer-guide/provider-runtime.md)
