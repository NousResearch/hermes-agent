---
sidebar_position: 2
title: "ACP 内部机制"
description: "ACP 适配器的工作原理：生命周期、会话、事件桥接、审批流程与工具渲染"
---

# ACP 内部机制

ACP 适配器将 Hermes 的同步 `AIAgent` 封装为异步 JSON-RPC stdio 服务器。

关键实现文件：

- `acp_adapter/entry.py`
- `acp_adapter/server.py`
- `acp_adapter/session.py`
- `acp_adapter/events.py`
- `acp_adapter/permissions.py`
- `acp_adapter/tools.py`
- `acp_adapter/auth.py`
- `acp_registry/agent.json`

## 启动流程

```text
hermes acp / hermes-acp / python -m acp_adapter
  -> acp_adapter.entry.main()
  -> parse --version / --check / --setup before server startup
  -> load ~/.hermes/.env
  -> configure stderr logging
  -> construct HermesACPAgent
  -> acp.run_agent(agent, use_unstable_protocol=True)
```

Zed ACP Registry 路径通过 `uvx --from 'hermes-agent[acp]==<version>' hermes-acp` 启动同一适配器，指向 `hermes-agent` PyPI 发布包。

stdout 保留用于 ACP JSON-RPC 传输。人类可读的日志输出至 stderr。

## 主要组件

### `HermesACPAgent`

`acp_adapter/server.py` 实现 ACP agent 协议。

职责：

- 初始化 / 认证
- 新建/加载/恢复/fork/列出/取消会话方法
- prompt（提示词）执行
- 会话模型切换
- 将同步 AIAgent 回调接入 ACP 异步通知

### `SessionManager`

`acp_adapter/session.py` 跟踪活跃的 ACP 会话。

每个会话存储：

- `session_id`
- `agent`
- `cwd`
- `model`
- `history`
- `cancel_event`

管理器线程安全，支持：

- create
- get
- remove
- fork
- list
- cleanup
- cwd 更新

### 事件桥接

`acp_adapter/events.py` 将 AIAgent 回调转换为 ACP `session_update` 事件。

已桥接的回调：

- `tool_progress_callback`
- `thinking_callback`（当前在 ACP 桥接中设置为 `None`——推理内容通过 `step_callback` 转发）
- `step_callback`

由于 `AIAgent` 在工作线程中运行，而 ACP I/O 位于主事件循环，桥接使用：

```python
asyncio.run_coroutine_threadsafe(...)
```

### 权限桥接

`acp_adapter/permissions.py` 将危险终端审批 prompt 适配为 ACP 权限请求。

映射关系：

- `allow_once` -> Hermes `once`
- `allow_always` -> Hermes `always`
- 拒绝选项 -> Hermes `deny`

超时和桥接失败默认拒绝。

### 工具渲染辅助

`acp_adapter/tools.py` 将 Hermes 工具映射到 ACP 工具类型，并构建面向编辑器的内容。

示例：

- `patch` / `write_file` -> 文件 diff
- `terminal` -> shell 命令文本
- `read_file` / `search_files` -> 文本预览
- 大型结果 -> 截断文本块（保障 UI 安全）

## 会话生命周期

```text
new_session(cwd)
  -> create SessionState
  -> create AIAgent(platform="acp", enabled_toolsets=["hermes-acp"])
  -> bind task_id/session_id to cwd override

prompt(..., session_id)
  -> extract text from ACP content blocks
  -> reset cancel event
  -> install callbacks + approval bridge
  -> run AIAgent in ThreadPoolExecutor
  -> update session history
  -> emit final agent message chunk
```

### 取消

`cancel(session_id)`：

- 设置会话取消事件
- 在可用时调用 `agent.interrupt()`
- 使 prompt 响应返回 `stop_reason="cancelled"`

### Fork

`fork_session()` 将消息历史深拷贝至新的活跃会话，在保留对话状态的同时为 fork 分配独立的 session ID 和 cwd。

## Provider/认证行为

ACP 不实现自己的认证存储。

而是复用 Hermes 的运行时解析器：

- `acp_adapter/auth.py`
- `hermes_cli/runtime_provider.py`

因此 ACP 通告并使用当前配置的 Hermes provider/凭据。它还始终通告一个终端 setup 认证方法（`hermes-setup`，参数 `--setup`），以便首次运行的 registry 客户端在启动正常 ACP 会话前可以打开 Hermes 的交互式模型/provider 配置。

## 工作目录绑定

ACP 会话携带编辑器 cwd。

会话管理器通过任务作用域的终端/文件覆盖将该 cwd 绑定到 ACP session ID，使文件和终端工具相对于编辑器工作区运行。

## 重复同名工具调用

事件桥接按工具名称以 FIFO 队列跟踪工具 ID，而非每个名称仅保留一个 ID。这对以下场景很重要：

- 并行同名调用
- 单步内重复同名调用

若不使用 FIFO 队列，完成事件将附加到错误的工具调用上。

## 审批回调恢复

ACP 在 prompt 执行期间临时在终端工具上安装审批回调，执行完成后恢复之前的回调。这避免了将 ACP 会话特定的审批处理器永久全局安装。

## 当前限制

- ACP 会话持久化至共享的 `~/.hermes/state.db`（SessionDB），在进程重启后透明恢复；它们会出现在 `session_search` 中
- 非文本 prompt 块在请求文本提取时当前被忽略
- 编辑器特定的 UX 因 ACP 客户端实现而异

## 相关文件

- `tests/acp/` — ACP 测试套件
- `toolsets.py` — `hermes-acp` toolset 定义
- `hermes_cli/main.py` — `hermes acp` CLI 子命令
- `pyproject.toml` — `[acp]` 可选依赖 + `hermes-acp` 脚本