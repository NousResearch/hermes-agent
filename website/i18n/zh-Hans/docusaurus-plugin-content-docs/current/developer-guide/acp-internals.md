---
sidebar_position: 2
title: "ACP 内部原理"
description: "ACP 适配器的工作原理：生命周期、会话、事件桥接、审批和工具渲染"
---

# ACP 内部原理

ACP 适配器将 Hermes 同步的 `AIAgent` 封装在一个异步 JSON-RPC stdio 服务器中。

主要实现文件：

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
  -> load ~/.hermes/.env
  -> configure stderr logging
  -> construct HermesACPAgent
  -> acp.run_agent(agent, use_unstable_protocol=True)
```

Stdout 保留给 ACP JSON-RPC 传输。可读的日志输出到 stderr。

## 主要组件

### `HermesACPAgent`

`acp_adapter/server.py` 实现了 ACP 代理协议。

职责：

- 初始化 / 身份验证
- 新建/加载/恢复/分叉/列表/取消会话方法
- 提示词执行
- 会话模型切换
- 将同步 AIAgent 回调连接到 ACP 异步通知

### `SessionManager`

`acp_adapter/session.py` 跟踪活动的 ACP 会话。

每个会话存储：

- `session_id`
- `agent`
- `cwd`
- `model`
- `history`
- `cancel_event`

管理器是线程安全的，支持：

- 创建
- 获取
- 删除
- 分叉
- 列表
- 清理
- cwd 更新

### 事件桥接

`acp_adapter/events.py` 将 AIAgent 回调转换为 ACP `session_update` 事件。

桥接的回调：

- `tool_progress_callback`
- `thinking_callback`
- `step_callback`
- `message_callback`

由于 `AIAgent` 在工作线程中运行，而 ACP I/O 位于主事件循环上，桥接使用：

```python
asyncio.run_coroutine_threadsafe(...)
```

### 权限桥接

`acp_adapter/permissions.py` 将危险的终端审批提示适配为 ACP 权限请求。

映射：

- `allow_once` -> Hermes `once`
- `allow_always` -> Hermes `always`
- 拒绝选项 -> Hermes `deny`

超时和桥接失败默认拒绝。

### 工具渲染辅助函数

`acp_adapter/tools.py` 将 Hermes 工具映射到 ACP 工具类型并构建编辑器面向的内容。

示例：

- `patch` / `write_file` -> 文件差异
- `terminal` -> shell 命令文本
- `read_file` / `search_files` -> 文本预览
- 大结果 -> 为 UI 安全截断的文本块

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
- 导致提示响应返回 `stop_reason="cancelled"`

### 分叉

`fork_session()` 深拷贝消息历史到一个新的活动会话，保留对话状态的同时给予分叉其自己的会话 ID 和 cwd。

## Provider/Auth 行为

ACP 不实现自己的 auth store。

而是重用 Hermes 的运行时解析器：

- `acp_adapter/auth.py`
- `hermes_cli/runtime_provider.py`

因此 ACP 公布并使用当前配置的 Hermes provider/凭据。

## 工作目录绑定

ACP 会话携带编辑器 cwd。

会话管理器通过任务作用域的终端/文件覆盖将 cwd 绑定到 ACP 会话 ID，以便文件和终端工具相对于编辑器工作空间进行操作。

## 重复同名工具调用

事件桥接按工具名称跟踪工具 ID FIFO，而不仅仅是每个名称一个 ID。这对于以下情况很重要：

- 并行同名调用
- 一步中重复的同名调用

没有 FIFO 队列，完成事件会附加到错误的工具调用。

## 审批回调恢复

ACP 在提示执行期间临时在终端工具上安装审批回调，然后在之后恢复之前的回调。这避免了将 ACP 会话特定的审批处理器永久全局安装。

## 当前限制

- ACP 会话持久化到共享的 `~/.hermes/state.db`（SessionDB）并在进程重启时透明恢复；它们出现在 `session_search` 中
- 非文本提示块目前被忽略以进行请求文本提取
- 编辑器特定的 UX 因 ACP 客户端实现而异

## 相关文件

- `tests/acp/` — ACP 测试套件
- `toolsets.py` — `hermes-acp` 工具集定义
- `hermes_cli/main.py` — `hermes acp` CLI 子命令
- `pyproject.toml` — `[acp]` 可选依赖 + `hermes-acp` 脚本
