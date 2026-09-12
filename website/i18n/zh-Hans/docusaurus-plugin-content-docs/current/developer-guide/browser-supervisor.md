# Browser CDP Supervisor — 设计文档

**状态：** 已发布（PR 14540）
**最后更新：** 2026-04-23
**作者：** @teknium1

## 问题

原生 JS 对话框（`alert`/`confirm`/`prompt`/`beforeunload`）和 iframe 是我们浏览器工具中最大的两个缺口：

1. **对话框会阻塞 JS 线程。** 页面上的任何操作都会挂起，直到对话框被处理。在此工作之前，agent 无法感知对话框是否已打开——后续的工具调用会挂起或抛出不透明的错误。
2. **iframe 不可见。** Agent 可以在 DOM 快照中看到 iframe 节点；supervisor 为观察跨域（OOPIF）iframe 而附加 CDP 会话，但有意不暴露原始 CDP 或 JavaScript 执行路径。

[PR #12550](https://github.com/NousResearch/hermes-agent/pull/12550) 提出了一个无状态的 `browser_dialog` 包装器。该方案无法解决检测问题——它只是在 agent 已经（通过症状）知道对话框已打开时，提供了一个更简洁的 CDP 调用。已作为被取代方案关闭。

## 后端能力矩阵（2026-04-23 实测验证）

supervisor 会为任何附加 CDP URL 的浏览器任务保持持久 WebSocket，并将待处理对话框和框架结构加入 `browser_snapshot`。`browser_dialog` 和只读 `browser_cdp` 仅在会话启动时通过 `/browser connect` 或 `browser.cdp_url` 配置显式覆盖时注册；端点可托管在云端。provider 托管的每会话 CDP URL 不会自动作为该覆盖暴露。

| 后端 | 对话框检测 | 对话框响应 | 框架树 | 原始 CDP / eval 暴露 |
|---|---|---|---|---|
| 显式 `/browser connect` 或 `browser.cdp_url` 覆盖（本地或云端托管） | ✓ | ✓ 完整流程 | ✓ | 仅只读浏览器检查 |
| provider 托管的 Browserbase、Browser Use、Firecrawl 会话 CDP | 附加时 ✓ | ✗ 不会自动注册 | 附加时 ✓ | ✗ |
| Camofox / 默认本地 agent-browser | ✗ 无 CDP 端点 | ✗ | 通过 DOM 快照部分支持 | ✗ |

只读直接检查路径仅适用于显式配置的 CDP 传输。浏览器控制器协议（包括开发者模式）绝不协商或分派原始 CDP 或任意求值。

**云端会话边界。** 云端 provider 的每会话 CDP URL 可附加 supervisor 并添加快照字段，但不会自动注册直接工具。通过 `/browser connect` 或 `browser.cdp_url` 显式配置该 URL 才可选择只读检查和对话框响应工具。Camofox 没有 CDP 端点。

## 架构

### CDPSupervisor

每个 Hermes `task_id` 对应一个在后台守护线程中运行的 `asyncio.Task`。持有一个到后端 CDP 端点的持久 WebSocket 连接。维护：

- **对话框队列** — `List[PendingDialog]`，包含 `{id, type, message, default_prompt, session_id, opened_at}`
- **框架树** — `Dict[frame_id, FrameInfo]`，包含父子关系、URL、origin，以及是否存在用于 supervisor 观察的跨域子会话
- **会话映射** — `Dict[session_id, SessionInfo]`，供交互工具将操作路由到正确的已附加会话以执行 OOPIF 操作
- **近期控制台错误** — 最近 50 条的环形缓冲区（用于 PR 2 诊断）

附加时订阅：
- `Page.enable` — `javascriptDialogOpening`、`frameAttached`、`frameNavigated`、`frameDetached`
- `Runtime.enable` — `executionContextCreated`、`consoleAPICalled`、`exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — 暴露子 OOPIF target；supervisor 在每个上启用 `Page`+`Runtime`

通过快照锁实现线程安全的状态访问；工具处理器（同步）读取冻结快照，无需 await。

### 生命周期

- **启动：** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — 当任务附加显式覆盖或 provider 会话 CDP URL 时调用。幂等。
- **停止：** 会话拆除或 `/browser disconnect`。取消 asyncio task，关闭 WebSocket，丢弃状态。
- **重新绑定：** 若 CDP URL 变更（用户重新连接到新的 Chrome），停止旧 supervisor 并重新启动——绝不跨端点复用状态。

### 对话框策略

通过 `config.yaml` 中的 `browser.dialog_policy` 配置：

- **`must_respond`**（默认）— 捕获，在 `browser_snapshot` 中呈现，等待显式的 `browser_dialog(action=...)` 调用。在 300s 安全超时后若无响应，则自动关闭并记录日志。防止有缺陷的 agent 永久挂起。
- `auto_dismiss` — 记录并立即关闭；agent 事后通过 `browser_snapshot` 内的 `browser_state` 查看。
- `auto_accept` — 记录并接受（适用于用户希望干净导航离开时的 `beforeunload`）。

策略按 task 配置；v1 不支持按对话框覆盖。

## Agent 接口（PR 1）

### 一个新工具

```
browser_dialog(action, prompt_text=None, dialog_id=None)
```

- `action="accept"` / `"dismiss"` → 响应指定的或唯一待处理的对话框（必填）
- `prompt_text=...` → 向 `prompt()` 对话框提供的文本
- `dialog_id=...` → 当多个对话框排队时用于消歧（罕见）

该工具仅用于响应。Agent 在调用前从 `browser_snapshot` 输出中读取待处理对话框。

### `browser_snapshot` 扩展

当 supervisor 已附加时，在现有快照输出中新增三个可选字段：

```json
{
  "pending_dialogs": [
    {"id": "d-1", "type": "alert", "message": "Hello", "opened_at": 1650000000.0}
  ],
  "recent_dialogs": [
    {"id": "d-1", "type": "alert", "message": "...", "opened_at": 1650000000.0,
     "closed_at": 1650000000.1, "closed_by": "remote"}
  ],
  "frame_tree": {
    "top": {"frame_id": "FRAME_A", "url": "https://example.com/", "origin": "https://example.com"},
    "children": [
      {"frame_id": "FRAME_B", "url": "about:srcdoc", "is_oopif": false},
      {"frame_id": "FRAME_C", "url": "https://ads.example.net/", "is_oopif": true, "session_id": "SID_C"}
    ],
    "truncated": false
  }
}
```

- **`pending_dialogs`**：当前阻塞页面 JS 线程的对话框。Agent 必须调用 `browser_dialog(action=...)` 进行响应。

- **`recent_dialogs`**：最近关闭的最多 20 个对话框的环形缓冲区，带有 `closed_by` 标签——`"agent"`（我们响应了）、`"auto_policy"`（本地 auto_dismiss/auto_accept）、`"watchdog"`（must_respond 超时触发）或 `"remote"`（浏览器/后端主动关闭）。

- **`frame_tree`**：框架结构，包括跨域（OOPIF）子框架。上限为 30 条 + OOPIF 深度 2，以限制广告密集页面上的快照大小。当达到限制时，`truncated: true` 会出现；请使用快照和专用浏览器操作继续交互。

以上均不新增工具 schema 接口——agent 从其已请求的快照中读取。

### 可用性门控

`browser_cdp` 和 `browser_dialog` 均通过 `_browser_cdp_check` 门控：仅在会话启动时通过 `/browser connect` 或 `browser.cdp_url` 配置显式 CDP 覆盖时注册，端点可托管在云端。provider 托管会话 URL 本身不会注册它们。另一方面，任何附加 CDP 会话都可启动 supervisor，因此云端会话可能显示其快照字段；Camofox 和默认本地 agent-browser 会省略这些字段。

## 跨域 iframe 边界

supervisor 保留子 CDP 会话以观察对话框和帧结构，而不提供执行传输。`browser_cdp` 仅接受狭窄的只读浏览器级允许列表（`Browser.getVersion`、`Target.getTargets`），并拒绝 target 与 `frame_id`。`browser_console(expression=...)` 同样已禁用。这可防止附加或恶意页面借助 iframe 路由执行代码、访问凭据或访问内部服务。

## Camofox（后续跟进）

计划向 `jo-inc/camofox-browser` 提交 issue，添加：
- 每个会话的 Playwright `page.on('dialog', handler)`
- `GET /tabs/:tabId/dialogs` 轮询端点
- `POST /tabs/:tabId/dialogs/:id` 用于接受/关闭
- 框架树内省端点

## 涉及文件（PR 1）

### 新增

- `tools/browser_supervisor.py` — `CDPSupervisor`、`SupervisorRegistry`、`PendingDialog`、`FrameInfo`
- `tools/browser_dialog_tool.py` — `browser_dialog` 工具处理器
- `tests/tools/test_browser_supervisor.py` — 模拟 CDP WebSocket 服务器 + 生命周期/状态测试
- `website/docs/developer-guide/browser-supervisor.md` — 本文件

### 修改

- `toolsets.py` — 在 `browser`、`hermes-acp`、`hermes-api-server`、核心工具集中注册 `browser_dialog`（通过 CDP 可达性门控）
- `tools/browser_tool.py`
  - `browser_navigate` 启动钩子：若 CDP URL 可解析，调用 `SupervisorRegistry.get_or_start(task_id, cdp_url)`
  - `browser_snapshot`（约第 1536 行）：将 supervisor 状态合并到返回载荷
  - `/browser connect` 处理器：以新端点重启 supervisor
  - `_cleanup_browser_session` 中的会话拆除钩子
- `hermes_cli/config.py` — 向 `DEFAULT_CONFIG` 添加 `browser.dialog_policy` 和 `browser.dialog_timeout_s`
- 文档：`website/docs/user-guide/features/browser.md`、`website/docs/reference/tools-reference.md`、`website/docs/reference/toolsets-reference.md`

## 非目标

- Camofox 的检测/交互（上游缺口；单独跟踪）
- 向用户实时流式传输对话框/框架事件（需要 gateway 钩子）
- 跨会话持久化对话框历史（仅内存）
- 按 iframe 配置对话框策略（agent 可通过 `dialog_id` 表达）
- 将 `browser_cdp` 扩展到只读浏览器级允许列表之外

## 测试

单元测试使用 asyncio 模拟 CDP 服务器，该服务器实现了足够的协议子集，以覆盖所有状态转换：附加、启用、导航、对话框触发、对话框关闭、框架附加/分离、子 target 附加、会话拆除。手动端到端测试通过 `/browser connect` 连接到实时 Chromium 系浏览器，并运行上述对话框/框架测试用例。provider 托管的云端 CDP 单独作为快照-supervisor 路径覆盖；显式配置的云端端点与本地端点遵循相同的直接工具边界。