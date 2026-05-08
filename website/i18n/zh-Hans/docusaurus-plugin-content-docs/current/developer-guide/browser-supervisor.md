# Browser CDP Supervisor — 设计

**状态：** 已发货（PR 14540）
**最后更新：** 2026-04-23
**作者：** @teknium1

## 问题

原生 JS 对话框（`alert`/`confirm`/`prompt`/`beforeunload`）和 iframe 是我们浏览器工具中最大的两个缺口：

1. **对话框阻塞 JS 线程。** 页面上的任何操作都会停滞，直到对话框被处理。在此工作之前，代理无法知道对话框是否打开 — 后续工具调用会挂起或抛出不透明的错误。
2. **Iframe 不可见。** 代理可以在 DOM 快照中看到 iframe 节点，但无法在其中点击、输入或执行 eval — 特别是生活在独立 Chromium 进程中的跨域（OOPIF）iframe。

[PR #12550](https://github.com/NousResearch/hermes-agent/pull/12550) 提出了一个无状态的 `browser_dialog` 包装器。这并不能解决检测问题 — 当代理已经知道（通过症状）对话框已打开时，它是一个更干净的 CDP 调用。已关闭，被取代。

## 后端能力矩阵（经 2026-04-23 验证）

使用一次性探测脚本对一个数据 URL 页面进行测试，该页面在主框架和同源 srcdoc iframe 中触发警报，以及跨域 `https://example.com` iframe：

| 后端 | 对话框检测 | 对话框响应 | 框架树 | 通过 `browser_cdp(frame_id=...)` 的 OOPIF `Runtime.evaluate` |
|---|---|---|---|---|
| 本地 Chrome（`--remote-debugging-port`）/ `/browser connect` | ✓ | ✓ 完整工作流 | ✓ | ✓ |
| Browserbase | ✓（通过桥接） | ✓ 完整工作流（通过桥接） | ✓ | ✓（在真实跨域 iframe 上验证了 `document.title = "Example Domain"`） |
| Camofox | ✗ 无 CDP（仅 REST） | ✗ | 通过 DOM 快照部分 | ✗ |

**Browserbase 响应工作原理。** Browserbase 的 CDP 代理在内部使用 Playwright 并在约 10ms 内自动关闭原生对话框，因此 `Page.handleJavaScriptDialog` 无法跟上。为了解决这个问题，supervisor 通过 `Page.addScriptToEvaluateOnNewDocument` 注入一个桥接脚本，该脚本用同步 XHR 覆盖 `window.alert`/`confirm`/`prompt` 到一个魔法主机（`hermes-dialog-bridge.invalid`）。`Fetch.enable` 在它们触及网络之前拦截这些 XHR — 对话框成为 supervisor 捕获的 `Fetch.requestPaused` 事件，`respond_to_dialog` 通过 `Fetch.fulfillRequest` 用注入脚本解码的 JSON 体来完成。

净结果：从页面的角度来看，`prompt()` 仍然返回代理提供的字符串。从代理的角度来看，无论哪种方式都是相同的 `browser_dialog(action=...)` API。在真实的 Browserbase 会话上进行了端到端测试 — 4/4（alert/prompt/confirm-accept/confirm-dismiss）通过，包括值往返回页面 JS。

Camofox 在此 PR 中不受支持；计划在 `jo-inc/camofox-browser` 上提出后续上游问题，请求对话框轮询端点。

## 架构

### CDPSupervisor

每个 Hermes `task_id` 一个在后台守护线程中运行的 `asyncio.Task`。持有到后端 CDP 端点的持久 WebSocket。维护：

- **对话框队列** — `List[PendingDialog]`，包含 `{id, type, message, default_prompt, session_id, opened_at}`
- **框架树** — `Dict[frame_id, FrameInfo]`，包含父关系、URL、origin、是否跨域子 session
- **Session 映射** — `Dict[session_id, SessionInfo]`，以便交互工具可以路由到正确的附加 session 进行 OOPIF 操作
- **最近的控制台错误** — 最后 50 个的环形缓冲区（用于 PR 2 诊断）

附加时订阅：
- `Page.enable` — `javascriptDialogOpening`、`frameAttached`、`frameNavigated`、`frameDetached`
- `Runtime.enable` — `executionContextCreated`、`consoleAPICalled`、`exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — 浮现子 OOPIF 目标；supervisor 在每个上启用 `Page`+`Runtime`

通过快照锁进行线程安全的状态访问；工具处理器（同步）读取冻结的快照而不等待。

### 生命周期

- **启动：** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — 由 `browser_navigate`、Browserbase session 创建、`/browser connect` 调用。幂等。
- **停止：** session 拆除或 `/browser disconnect`。取消 asyncio 任务，关闭 WebSocket，丢弃状态。
- **重新绑定：** 如果 CDP URL 更改（用户重新连接到新的 Chrome），停止旧 supervisor 并重新开始 — 不要跨端点重用状态。

### 对话框策略

通过 `config.yaml` 中的 `browser.dialog_policy` 配置：

- **`must_respond`**（默认）— 捕获，在 `browser_snapshot` 中浮现，等待显式的 `browser_dialog(action=...)` 调用。300 秒安全超时后无响应，自动关闭并记录。防止有缺陷的代理永远停滞。
- `auto_dismiss` — 记录并立即关闭；代理通过 `browser_snapshot` 中的 `browser_state` 事后看到它。
- `auto_accept` — 记录并接受（用于用户想要干净导航离开的 `beforeunload`）。

策略按任务；v1 中没有按对话框覆盖。

## 代理界面（PR 1）

### 一个新工具

```
browser_dialog(action, prompt_text=None, dialog_id=None)
```

- `action="accept"` / `"dismiss"` → 响应指定的或唯一的待处理对话框（必需）
- `prompt_text=...` → 提供给 `prompt()` 对话框的文本
- `dialog_id=...` → 当多个对话框排队时消除歧义（罕见）

工具仅响应。代理在调用前从 `browser_snapshot` 输出中读取待处理的对话框。

### `browser_snapshot` 扩展

当 supervisor 附加时，向现有快照输出添加三个可选字段：

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

- **`pending_dialogs`**：当前阻塞页面 JS 线程的对话框。代理必须调用 `browser_dialog(action=...)` 来响应。在 Browserbase 上为空，因为他们的 CDP 代理在约 10ms 内自动关闭。

- **`recent_dialogs`**：最近关闭对话框的环形缓冲区，最多 20 个，带 `closed_by` 标记 — `"agent"`（我们响应了）、`"auto_policy"`（本地 auto_dismiss/auto_accept）、`"watchdog"`（must_respond 超时触发）或 `"remote"`（浏览器/后端关闭了它，例如 Browserbase）。这是 Browserbase 上的代理仍然可以了解发生了什么的方式。

- **`frame_tree`**：框架结构，包括跨域（OOPIF）子项。在广告密集型页面上，上限为 30 个条目 + OOPIF 深度 2 以限制快照大小。`truncated: true` 在达到限制时浮现；需要完整树的代理可以使用带 `Page.getFrameTree` 的 `browser_cdp`。

这些内容都没有新的工具模式界面 — 代理读取它已经请求的快照。

### 可用性门控

两个界面都基于 `_browser_cdp_check` 门控（supervisor 只能在 CDP 端点可达时运行）。在 Camofox / 无后端会话上，对话框工具被隐藏，快照省略新字段 — 没有模式膨胀。

## 跨域 iframe 交互

扩展对话框检测工作，`browser_cdp(frame_id=...)` 通过 supervisor 已连接的 WebSocket 使用 OOPIF 的子 `sessionId` 路由 CDP 调用（特别是 `Runtime.evaluate`）。代理从 `browser_snapshot.frame_tree.children[]` 中选取 `is_oopif=true` 的 frame_id，并将其传递给 `browser_cdp`。对于同源 iframe（没有专用 CDP session），代理使用顶层 `Runtime.evaluate` 中的 `contentWindow`/`contentDocument` 代替 — 当 `frame_id` 属于非 OOPIF 时，supervisor 浮现一个指向该后备的错误。

在 Browserbase 上，这是 iframe 交互的唯一可靠路径 — 无状态 CDP 连接（每次 `browser_cdp` 调用打开）会遇到签名 URL 过期，而 supervisor 的长期连接保持有效 session。

## Camofox（后续）

计划针对 `jo-inc/camofox-browser` 的问题，添加：
- 每个 session 的 Playwright `page.on('dialog', handler)`
- `GET /tabs/:tabId/dialogs` 轮询端点
- `POST /tabs/:tabId/dialogs/:id` 接受/关闭
- 框架树内省端点

## 涉及的文件（PR 1）

### 新建

- `tools/browser_supervisor.py` — `CDPSupervisor`、`SupervisorRegistry`、`PendingDialog`、`FrameInfo`
- `tools/browser_dialog_tool.py` — `browser_dialog` 工具处理器
- `tests/tools/test_browser_supervisor.py` — 模拟 CDP WebSocket 服务器 + 生命周期/状态测试
- `website/docs/developer-guide/browser-supervisor.md` — 此文件

### 修改

- `toolsets.py` — 在 `browser`、`hermes-acp`、`hermes-api-server`、核心工具集中注册 `browser_dialog`（基于 CDP 可达性门控）
- `tools/browser_tool.py`
  - `browser_navigate` 启动钩子：如果 CDP URL 可解析，`SupervisorRegistry.get_or_start(task_id, cdp_url)`
  - `browser_snapshot`（约第 1536 行）：将 supervisor 状态合并到返回负载中
  - `/browser connect` 处理器：用新端点重启 supervisor
  - `_cleanup_browser_session` 中的 Session 拆除钩子
- `hermes_cli/config.py` — 向 `DEFAULT_CONFIG` 添加 `browser.dialog_policy` 和 `browser.dialog_timeout_s`
- 文档：`website/docs/user-guide/features/browser.md`、`website/docs/reference/tools-reference.md`、`website/docs/reference/toolsets-reference.md`

## 非目标

- Camofox 的检测/交互（上游缺口；单独跟踪）
- 将对话框/框架事件实时流式传输给用户（需要网关钩子）
- 跨会话持久化对话框历史（仅内存）
- 按 iframe 的对话框策略（代理可以通过 `dialog_id` 表达）
- 替换 `browser_cdp` — 它作为长尾（cookies、视口、网络限速）的逃生舱留在那里

## 测试

单元测试使用异步模拟 CDP 服务器，该服务器说足够的协议来练习所有状态转换：附加、启用、导航、对话框触发、对话框关闭、框架附加/分离、子目标附加、session 拆除。真实后端 E2E（Browserbase + 本地 Chrome）是手动的；2026-04-23 调查中的探测脚本保留在仓库中 `scripts/browser_supervisor_e2e.py`，以便任何人可以在新的后端版本上重新验证。
