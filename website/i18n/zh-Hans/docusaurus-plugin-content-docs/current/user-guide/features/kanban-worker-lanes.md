# Kanban worker lanes

**Worker lane** 是 kanban dispatcher 可以将任务路由到的一类进程。每个 lane 都有一个身份（assignee 字符串）、一个 spawn 机制，以及一个关于生成后必须如何处理任务的契约。

本页面即该契约。它面向两类读者：

- **Operator**：选择将哪些 lane 接入 board（创建哪些 profile、使用哪些 assignee）。
- **插件 / 集成作者**：希望添加新的 lane 形态（CLI worker，如包装 Codex / Claude Code / OpenCode 的工具；容器化 review worker；通过 API 拉取任务的非 Hermes 服务）。

如果你编写的是 worker 代码本身——即在 lane 内部运行的 agent——请参见 [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill 以获取更深入的过程细节。

## 层级结构

```text
Hermes Kanban  =  规范的任务生命周期 + 审计追踪
Worker lane    =  单个已分配卡片的实现执行器
Reviewer       =  人或人代理，把关 "done"
GitHub PR      =  可上游化的产物（可选，用于代码 lane）
```

Hermes Kanban 拥有生命周期真相——`ready` → `running` → `blocked` / `done` / `archived`。Worker lane 执行工作，但从不拥有该真相；它们所做的一切都通过 `kanban_*` 工具（或者，对于非 Hermes 外部 worker，通过 API）回流到 kanban kernel。Reviewer 把关从"代码变更已编写"到"任务完成"的过渡。

## Lane 需要提供什么

要成为一个 kanban worker lane，一个集成必须提供三样东西：

### 1. Assignee 字符串

Dispatcher 将 `task.assignee` 与 Hermes profile 名称（默认 lane 形态）或已注册的非 spawnable 标识符（插件 lane 形态——见下方的[添加外部 CLI worker lane](#添加外部-cli-worker-lane)）进行匹配。无法解析 assignee 的任务将保留在 `ready` 状态，并触发 `skipped_nonspawnable` 事件，以便 board operator 修复；它们不会被静默丢弃或由任意 fallback 执行。

### 2. Spawn 机制

对于 Hermes profile lane，dispatcher 的 `_default_spawn` 在任务固定的工作空间内运行 `hermes -p <assignee> chat -q <prompt>`（当 `hermes` shim 不在 `$PATH` 上时，使用等效的 module 形式），并设置以下环境变量：

| Variable | 携带内容 |
|---|---|
| `HERMES_KANBAN_TASK` | worker 正在操作的任务 id |
| `HERMES_KANBAN_DB` | 每个 board 的 SQLite 文件的绝对路径 |
| `HERMES_KANBAN_BOARD` | board slug |
| `HERMES_KANBAN_WORKSPACES_ROOT` | board 工作空间树的根目录 |
| `HERMES_KANBAN_WORKSPACE` | *当前*任务工作空间的绝对路径 |
| `HERMES_KANBAN_RUN_ID` | 当前 run 的 id（用于生命周期门控） |
| `HERMES_KANBAN_CLAIM_LOCK` | claim lock 字符串（`<host>:<pid>:<uuid>`） |
| `HERMES_PROFILE` | worker 自己的 profile 名称（用于 `kanban_comment` 作者归属） |
| `HERMES_TENANT` | tenant 命名空间，如果任务有的话 |

对于非 Hermes lane（通过插件注册），插件提供自己的 `spawn_fn` callable，接收 `task`、`workspace` 和 `board`，并返回一个可选的 pid 用于崩溃检测。

### 3. 生命周期终结器

每次 claim 必须以以下三种方式之一结束：

- `kanban_complete(summary=..., metadata=...)` — 任务成功，状态翻转为 `done`。
- `kanban_block(reason=...)` — 任务等待人工输入，状态翻转为 `blocked`。当运行 `kanban_unblock` 时，dispatcher 会重新 spawn。
- Worker 进程退出且未调用工具。Kernel 会回收它并发出 `crashed`（PID 死亡）或 `gave_up`（连续失败断路器触发）或 `timed_out`（超过 max_runtime）。这是失败路径；健康的 worker 不会以此结束。

Kanban kernel 强制执行每次 run 只能由上述之一终结。一个既没有调用 complete 也没有调用 block 就正常退出的 worker 会被视为 crashed。

## 输出和 review-required 约定

对于大多数代码变更任务，worker 完成时工作并不真正算 *done*——它需要人工 reviewer。Kanban kernel 不强制执行这种区分（"代码变更任务"是模糊的，强制每个代码 worker 都 block 而不是 complete 会破坏不需要 review 的流程）。这是一个叠加在顶层的约定：

- **Block 而不是 complete**，`reason` 前缀为 `review-required: `，以便 dashboard / `hermes kanban show` 将该行显示为等待 review。
- **先将结构化元数据放入 `kanban_comment`**，因为 `kanban_block` 只携带人类可读的 `reason`。Comment 是持久的注释通道——每个与审计相关的字段（changed_files、tests_run、diff_path 或 PR url、decisions）都属于这里。
- **Reviewer 要么批准并 unblock**，这会重新 spawn worker 并附带 comment 线程以供跟进；要么通过另一条 comment 要求修改，下一次 worker run 会将其作为 `kanban_show` 上下文的一部分看到。

[`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill 提供了 `kanban_complete`（真正终结的任务——拼写错误修复、文档变更、研究报告）和 `review-required` block 模式的工作示例。

## 日志和审计追踪

Dispatcher 将每个任务的 worker stdout/stderr 写入 `<board-root>/logs/<task_id>.log`。日志可从 kanban 元数据中审计：

- `task_runs` 行携带 `log_path`、exit code（如果可用）、summary 和 metadata。
- `task_events` 行携带每个状态转换（`promoted`、`claimed`、`heartbeat`、`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`、`reclaimed`、`claim_extended`）。
- `kanban_show` 返回两者，因此 reviewer（或后续 worker）阅读任务时无需 dashboard 访问权限即可获得完整历史。

Dashboard 渲染 run 历史，包含 summary、metadata 块和 exit-status badge。CLI 用户可以运行 `hermes kanban tail <task_id>` 进行实时跟踪，或运行 `hermes kanban runs <task_id>` 查看历史尝试列表。

## 现有的 lane 形态

### Hermes profile lane（默认）

这是当今每个 kanban worker 采用的形态：assignee 是一个 profile 名称，dispatcher spawn `hermes -p <profile>`，worker 自动加载 [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill 以及 `KANBAN_GUIDANCE` system-prompt 块，并使用 `kanban_*` 工具终结 run。除了定义 profile 外无需额外设置。

当你为 fleet 创建 profile 时，选择名称时应匹配你希望 orchestrator 路由到的 *role*。Orchestrator（如果有的话）通过 `hermes profile list` 发现你的 profile 名称——系统没有固定的名册（参见 [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) skill 了解契约的 orchestrator 侧）。

### Orchestrator profile lane

Profile lane 的一种特化：orchestrator 是一个 Hermes profile，其工具集包含 `kanban` 但不包含用于实现的 `terminal` / `file` / `code` / `web`。它的工作是通过 `kanban_create` + `kanban_link` 将高层目标分解为子任务，然后退后。Orchestrator skill 编码了反诱惑规则。

## 添加外部 CLI worker lane

将非 Hermes CLI 工具（Codex CLI、Claude Code CLI、OpenCode CLI、本地 coding-model runner 等）作为 kanban worker lane 接入，*目前还不是一条铺好的路*。Dispatcher 的 spawn 函数是可插拔的（`spawn_fn` 是 `dispatch_once` 的参数），插件可以为自己的非 Hermes assignee 注册 `spawn_fn`，但 surrounding 集成工作——将 CLI 的 exit code 包装成 `kanban_complete` / `kanban_block` 调用、将 CLI 的工作空间/沙箱约定映射到 dispatcher 的 `HERMES_KANBAN_WORKSPACE` 环境变量、处理 auth 和每个 CLI 的策略——仍然是每个集成需要单独设计的工作。

如果你考虑添加一个 CLI lane，请开一个 issue 描述具体的 CLI 和你试图启用的 workflow。上方的契约是任何此类 lane 必须满足的约束；实现形态（每个 CLI 一个插件 vs 由 config 参数化的通用 CLI-runner 插件）是开放的。

相关的历史 issue 是 [#19931](https://github.com/NousResearch/hermes-agent/issues/19931)，以及已关闭未合并的 Codex 专用 PR [#19924](https://github.com/NousResearch/hermes-agent/pull/19924)——这些描述了最初的架构提案，但没有落地一个 runner。

## Dispatcher 处理的失败模式

以下是 lane 作者无需重新实现的：

- **Stale claim TTL** —— 一个 claim 后从未 heartbeat / complete / block 的 worker，在 `DEFAULT_CLAIM_TTL_SECONDS`（默认 15 分钟）后会被 reclaim——但仅当 worker 进程确实已死亡时。一个活着的 worker（慢模型在一个无工具的 LLM 调用中花费 20+ 分钟）会得到 claim *extended* 而不是被杀死；只有死亡的 PID 才会被 reclaim。
- **Crashed worker** —— 一个 host-local PID 已消失的 worker 会被 `detect_crashed_workers` 检测到并回收；任务增加 `consecutive_failures`，并可能在断路器触发时自动 block。
- **Run-level retry** —— 当任务被重试（post-block、post-crash、post-reclaim）时，worker 可以使用终结工具上的 `expected_run_id` 参数，在自己的 run 已被取代时快速失败。
- **Per-task max runtime** —— `task.max_runtime_seconds` 硬限制每次 run 的 wall-clock 时间，无论 PID 是否存活。捕获那些 live-PID extension 否则会一直运行的真正死锁 worker。
- **Stranded-task detection** —— 一个 assignee 在 `kanban.stranded_threshold_seconds`（默认 30 分钟）内从未产生 claim 的 ready 任务，会在 `hermes kanban diagnostics` 中显示为 `stranded_in_ready` 警告。严重度在 2x 阈值时升级为 error，在 6x 时升级为 critical。在一个信号中捕获拼写错误的 assignee、已删除的 profile 和宕机的外部 worker 池——与身份无关，无需维护 per-board allowlist。

## 相关

- [Kanban 概览](./kanban) —— 面向用户的介绍。
- [Kanban 教程](./kanban-tutorial) —— 打开 dashboard 的逐步演练。
- [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) —— worker 进程加载的 skill。
- [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) —— orchestrator 侧。
