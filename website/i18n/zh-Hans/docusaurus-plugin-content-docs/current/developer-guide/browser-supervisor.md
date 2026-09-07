# Browser CDP Supervisor — 设计文档

**状态：** 已发布（PR 14540）
**最后更新：** 2026-04-23
**作者：** @teknium1

## 问题

原生 JS 对话框（`alert`/`confirm`/`prompt`/`beforeunload`）和 iframe 是我们浏览器工具中最大的两个缺口：

1. **对话框会阻塞 JS 线程。** 页面上的任何操作都会挂起，直到对话框被处理。在此工作之前，agent 无法感知对话框是否已打开——后续的工具调用会挂起或抛出不透明的错误。
2. **iframe 不可见。** Agent 可以在 DOM 快照中看到 iframe 节点，但无法在其中点击、输入或执行 eval——尤其是运行在独立 Chromium 进程中的跨域（OOPIF）iframe。

[PR #12550](https://github.com/NousResearch/hermes-agent/pull/12550) 提出了一个无状态的 `browser_dialog` 包装器。该方案无法解决检测问题——它只是在 agent 已经（通过症状）知道对话框已打开时，提供了一个更简洁的 CDP 调用。已作为被取代方案关闭。

## 后端能力矩阵（2026-04-23 实测验证）

使用一次性探测脚本，针对一个在主框架和同源 srcdoc iframe 中触发 alert 的 data-URL 页面，以及一个跨域 `https://example.com` iframe 进行测试：

| 后端 | 对话框检测 | 对话框响应 | 框架树 | OOPIF `Runtime.evaluate`（通过 `browser_cdp(frame_id=...)`） |
|---|---|---|---|---|
| 本地 Chrome（`--remote-debugging-port`）/ `/browser connect` | ✓ | ✓ 完整流程 | ✓ | ✓ |
| Browserbase | ✓（通过 bridge） | ✓ 完整流程（通过 bridge） | ✓ | ✓（`document.title = "Example Domain"` 已在真实跨域 iframe 上验证） |
| Camofox | ✗ 无 CDP（仅 REST） | ✗ | 通过 DOM 快照部分支持 | ✗ |

**Browserbase 响应的工作原理。** Browserbase 的 CDP 代理在内部使用 Playwright，并在约 10ms 内自动关闭原生对话框，因此 `Page.handleJavaScriptDialog` 无法跟上。为解决此问题，supervisor 通过 `Page.addScriptToEvaluateOnNewDocument` 注入一个 bridge 脚本，将 `window.alert`/`confirm`/`prompt` 覆盖为向魔法主机（`hermes-dialog-bridge.invalid`）发起的同步 XHR。`Fetch.enable` 在这些 XHR 触达网络之前将其拦截——对话框变成 supervisor 捕获的 `Fetch.requestPaused` 事件，`respond_to_dialog` 通过 `Fetch.fulfillRequest` 以 JSON 响应体完成请求，注入的脚本对其进行解码。

最终效果：从页面角度看，`prompt()` 仍然返回 agent 提供的字符串。从 agent 角度看，无论哪种方式，都是同一套 `browser_dialog(action=...)` API。已针对真实 Browserbase 会话进行端到端测试——4/4（alert/prompt/confirm-accept/confirm-dismiss）全部通过，包括值回传到页面 JS 的验证。

Camofox 在本 PR 中暂不支持；计划在 `jo-inc/camofox-browser` 提交上游 issue，请求添加对话框轮询端点。

## 架构

### CDPSupervisor

每个 Hermes `task_id` 对应一个在后台守护线程中运行的 `asyncio.Task`。持有一个到后端 CDP 端点的持久 WebSocket 连接。维护：

- **对话框队列** — `List[PendingDialog]`，包含 `{id, type, message, default_prompt, session_id, opened_at}`
- **框架树** — `Dict[frame_id, FrameInfo]`，包含父子关系、URL、origin，以及是否为跨域子会话
- **会话映射** — `Dict[session_id, SessionInfo]`，供交互工具将操作路由到正确的已附加会话以执行 OOPIF 操作
- **近期控制台错误** — 最近 50 条的环形缓冲区（用于 PR 2 诊断）

附加时订阅：
- `Page.enable` — `javascriptDialogOpening`、`frameAttached`、`frameNavigated`、`frameDetached`
- `Runtime.enable` — `executionContextCreated`、`consoleAPICalled`、`exceptionThrown`
- `Target.setAutoAttach {autoAttach: true, flatten: true}` — 暴露子 OOPIF target；supervisor 在每个上启用 `Page`+`Runtime`

通过快照锁实现线程安全的状态访问；工具处理器（同步）读取冻结快照，无需 await。

### 生命周期

- **启动：** `SupervisorRegistry.get_or_start(task_id, cdp_url)` — 由 `browser_navigate`、Browserbase 会话创建、`/browser connect` 调用。幂等。
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

- **`pending_dialogs`**：当前阻塞页面 JS 线程的对话框。Agent 必须调用 `browser_dialog(action=...)` 进行响应。在 Browserbase 上为空，因为其 CDP 代理会在约 10ms 内自动关闭对话框。

- **`recent_dialogs`**：最近关闭的最多 20 个对话框的环形缓冲区，带有 `closed_by` 标签——`"agent"`（我们响应了）、`"auto_policy"`（本地 auto_dismiss/auto_accept）、`"watchdog"`（must_respond 超时触发）或 `"remote"`（浏览器/后端主动关闭，例如 Browserbase）。这是 Browserbase 上的 agent 仍能了解发生了什么的方式。

- **`frame_tree`**：框架结构，包括跨域（OOPIF）子框架。上限为 30 条 + OOPIF 深度 2，以限制广告密集页面上的快照大小。当达到限制时，`truncated: true` 会出现；需要完整树的 agent 可使用 `browser_cdp` 配合 `Page.getFrameTree`。

以上均不新增工具 schema 接口——agent 从其已请求的快照中读取。

### 可用性门控

两个接口均通过 `_browser_cdp_check` 进行门控（supervisor 只能在 CDP 端点可达时运行）。在 Camofox / 无后端会话中，对话框工具被隐藏，快照省略新字段——不产生 schema 膨胀。

## 跨域 iframe 交互

在对话框检测工作的基础上，`browser_cdp(frame_id=...)` 通过 supervisor 已连接的 WebSocket，使用 OOPIF 的子 `sessionId` 路由 CDP 调用（尤其是 `Runtime.evaluate`）。Agent 从 `browser_snapshot.frame_tree.children[]` 中 `is_oopif=true` 的条目获取 frame_id，并将其传递给 `browser_cdp`。对于同源 iframe（无专用 CDP 会话），agent 改用顶层 `Runtime.evaluate` 中的 `contentWindow`/`contentDocument`——当 `frame_id` 属于非 OOPIF 时，supervisor 会返回指向该回退方案的错误。

在 Browserbase 上，这是 iframe 交互的**唯一**可靠路径——无状态 CDP 连接（每次 `browser_cdp` 调用时打开）会遭遇签名 URL 过期，而 supervisor 的长连接则保持有效会话。

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
- 替换 `browser_cdp`——它作为长尾场景（cookies、viewport、网络限速）的逃生舱口继续保留

## 测试

单元测试使用 asyncio 模拟 CDP 服务器，该服务器实现了足够的协议子集，以覆盖所有状态转换：附加、启用、导航、对话框触发、对话框关闭、框架附加/分离、子 target 附加、会话拆除。真实后端端到端测试（Browserbase + 本地 Chromium 系浏览器）为手动执行——通过 `/browser connect` 连接到实时 Chromium 系浏览器，并运行上述对话框/框架测试用例。