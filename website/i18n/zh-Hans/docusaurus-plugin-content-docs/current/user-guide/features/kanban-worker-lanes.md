# Kanban worker lanes（工作者通道）

**worker lane**（工作者通道）是 kanban 调度器可以将任务路由到的一类进程。每个通道都有一个标识（assignee 字符串）、一个生成机制，以及一份关于生成后必须如何处理任务的契约。

本页即为该契约，面向两类读者：

- **运维人员**：选择将哪些通道接入看板（创建哪些 profile，使用哪些 assignee）。
- **插件/集成作者**：希望添加新的通道形态（封装 Codex / Claude Code / OpenCode 的 CLI worker、容器化审查 worker、通过 API 拉取任务的非 Hermes 服务）。

如果你编写的是 worker 代码本身——即运行在通道*内部*的 agent——请参阅 [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill，其中包含更深入的操作细节。

## 层级结构

```text
Hermes Kanban  =  规范的任务生命周期 + 审计追踪
Worker lane    =  某张已分配卡片的实现执行器
Reviewer       =  人工或人工代理，负责把关"完成"状态
GitHub PR      =  可上游的产物（可选，适用于代码通道）
```

Hermes Kanban 拥有生命周期的真实状态——`ready` → `running` → `blocked` / `done` / `archived`。Worker lane 执行工作，但从不拥有该真实状态；它们所做的一切都通过 `kanban_*` 工具回流至 kanban 内核（对于非 Hermes 外部 worker，则通过 API）。Reviewer 负责把关从"代码变更已写入"到"任务完成"的转换。

## 通道需提供的内容

要成为 kanban worker lane，集成必须提供三项内容：

### 1. assignee 字符串

调度器将 `task.assignee` 与 Hermes profile 名称（默认通道形态）或已注册的不可生成标识符（插件通道形态——见下文[添加外部 CLI worker 通道](#adding-an-external-cli-worker-lane)）进行匹配。assignee 无法解析的任务将保留在 `ready` 状态，并记录 `skipped_nonspawnable` 事件，以便看板运维人员修复；它们不会被静默丢弃，也不会由任意回退逻辑执行。

### 2. 生成机制

对于 Hermes profile 通道，调度器的 `_default_spawn` 会在任务固定的工作区内运行 `hermes -p <assignee> chat -q <prompt>`（或当 `hermes` shim 不在 `$PATH` 时使用等效的模块形式），并设置以下环境变量：

| 变量 | 携带内容 |
|---|---|
| `HERMES_KANBAN_TASK` | worker 正在操作的任务 id |
| `HERMES_KANBAN_DB` | 每个看板 SQLite 文件的绝对路径 |
| `HERMES_KANBAN_BOARD` | 看板 slug |
| `HERMES_KANBAN_WORKSPACES_ROOT` | 看板工作区树的根目录 |
| `HERMES_KANBAN_WORKSPACE` | *本*任务工作区的绝对路径 |
| `HERMES_KANBAN_RUN_ID` | 当前运行的 id（用于生命周期门控） |
| `HERMES_KANBAN_CLAIM_LOCK` | claim 锁字符串（`<host>:<pid>:<uuid>`） |
| `HERMES_PROFILE` | worker 自身的 profile 名称（用于 `kanban_comment` 作者归因） |
| `HERMES_TENANT` | 租户命名空间（如果任务有的话） |

对于非 Hermes 通道（通过插件注册），插件提供自己的 `spawn_fn` 可调用对象，接收 `task`、`workspace` 和 `board`，并返回可选的 pid 用于崩溃检测。

### 3. 生命周期终止器

每次 claim 必须以以下之一结束：

- `kanban_complete(summary=..., metadata=...)` — 任务成功，状态切换为 `done`。
- `kanban_block(reason=...)` — 任务等待人工输入，状态切换为 `blocked`。调度器在 `kanban_unblock` 运行时重新生成。
- worker 进程退出而未调用任何工具。内核回收该进程并发出 `crashed`（PID 已消亡）、`gave_up`（连续失败断路器触发）或 `timed_out`（超过 max_runtime）。这是失败路径；健康的 worker 不会在此结束。

kanban 内核强制要求每次运行恰好由其中一项终止。既未调用任何终止工具又正常退出的 worker 将被视为崩溃。

## 输出与 review-required 约定

对于大多数涉及代码变更的任务，worker 完成的那一刻并不意味着真正*完成*——还需要人工审查。kanban 内核不强制执行这一区分（"涉及代码变更的任务"定义模糊，且在每个代码 worker 上强制 block 而非 complete 会破坏不需要审查的流程）。这是叠加在上层的约定：

- **使用 block 而非 complete**，`reason` 以 `review-required: ` 为前缀，使仪表板 / `hermes kanban show` 将该行显示为等待审查。
- **先将结构化元数据写入 `kanban_comment`**，因为 `kanban_block` 只携带人类可读的 `reason`。Comment 是持久的注解通道——所有与审计相关的字段（changed_files、tests_run、diff_path 或 PR url、决策记录）都应放在这里。
- **Reviewer 批准并解除阻塞**，这将重新生成 worker 并附带 comment 线程用于后续跟进；或通过另一条 comment 要求修改，下一次 worker 运行时将通过 `kanban_show` 的上下文看到这些内容。

[`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill 中有 `kanban_complete`（真正终态的任务——拼写修复、文档变更、研究报告）和 `review-required` block 模式的完整示例。

## 日志与审计追踪

调度器将每个任务的 worker stdout/stderr 写入 `<board-root>/logs/<task_id>.log`。日志可通过 kanban 元数据进行审计：

- `task_runs` 行携带 `log_path`、退出码（如有）、摘要和元数据。
- `task_events` 行携带每次状态转换（`promoted`、`claimed`、`heartbeat`、`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`、`reclaimed`、`claim_extended`）。
- `kanban_show` 同时返回两者，因此 reviewer（或后续 worker）读取任务时无需访问仪表板即可获得完整历史。

仪表板以摘要、元数据块和退出状态徽章渲染运行历史。CLI 用户可运行 `hermes kanban tail <task_id>` 实时跟踪，或运行 `hermes kanban runs <task_id>` 查看历史尝试列表。

## 现有通道形态

### Hermes profile 通道（默认）

当前所有 kanban worker 采用的形态：assignee 是 profile 名称，调度器生成 `hermes -p <profile>`，worker 自动加载 [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill 以及 `KANBAN_GUIDANCE` 系统提示块，并使用 `kanban_*` 工具终止运行。除定义 profile 外无需任何额外配置。

为你的 fleet 创建 profile 时，选择与你希望 orchestrator 路由到的*角色*相匹配的名称。orchestrator（如果存在）通过 `hermes profile list` 发现你的 profile 名称——系统不假设固定的名单（orchestrator 侧的契约请参阅 [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) skill）。

### Orchestrator profile 通道

profile 通道的特化形态：orchestrator 是一个 Hermes profile，其工具集包含 `kanban`，但排除了用于实现的 `terminal` / `file` / `code` / `web`。其职责是通过 `kanban_create` + `kanban_link` 将高层目标分解为子任务，然后退出。orchestrator skill 编码了反诱惑规则。

## 添加外部 CLI worker 通道

将非 Hermes CLI 工具（Codex CLI、Claude Code CLI、OpenCode CLI、本地编码模型运行器等）接入 kanban worker 通道*尚未形成成熟路径*。调度器的 spawn 函数是可插拔的（`spawn_fn` 是 `dispatch_once` 的参数），插件可以为非 Hermes assignee 注册自己的 `spawn_fn`，但周边集成工作——将 CLI 的退出码封装为 `kanban_complete` / `kanban_block` 调用、将 CLI 的工作区/沙箱约定映射到调度器的 `HERMES_KANBAN_WORKSPACE` 环境变量、处理认证和每个 CLI 的策略——仍是每个集成各自的设计工作。

如果你考虑添加 CLI 通道，请提交一个 issue，描述具体的 CLI 以及你希望实现的工作流。上述契约是任何此类通道必须满足的约束；实现形态（每个 CLI 一个插件，还是通过配置参数化的通用 CLI 运行器插件）尚未确定。

相关历史 issue 为 [#19931](https://github.com/NousResearch/hermes-agent/issues/19931)，以及已关闭未合并的 Codex 专项 PR [#19924](https://github.com/NousResearch/hermes-agent/pull/19924)——这些描述了原始架构提案，但未落地运行器。

## 调度器处理的失败模式

通道作者无需重新实现以下逻辑：

- **Claim TTL 过期** — 已 claim 但从未心跳/完成/阻塞的 worker 在 `DEFAULT_CLAIM_TTL_SECONDS`（默认 15 分钟）后被回收——但仅当 worker 进程确实已死亡时。存活的 worker（慢速模型在一次无工具调用的 LLM 调用中耗时 20 分钟以上）会获得 claim *延期*而非被终止；只有 PID 已消亡时才会被回收。
- **Worker 崩溃** — 宿主本地 PID 已消失的 worker 由 `detect_crashed_workers` 检测并回收；任务的 `consecutive_failures` 递增，断路器触发时可能自动阻塞。
- **运行级重试** — 任务重试时（post-block、post-crash、post-reclaim），worker 可在终止工具上使用 `expected_run_id` 参数，在自身运行已被取代时快速失败。
- **每任务最大运行时间** — `task.max_runtime_seconds` 对每次运行的挂钟时间进行硬性限制，与 PID 存活状态无关。可捕获真正死锁的 worker——否则存活 PID 延期机制会让其持续运行。
- **滞留任务检测** — assignee 在 `kanban.stranded_threshold_seconds`（默认 30 分钟）内始终未产生 claim 的 ready 任务，会在 `hermes kanban diagnostics` 中显示为 `stranded_in_ready` 警告。严重程度在 2 倍阈值时升级为 error，在 6 倍时升级为 critical。可通过单一信号捕获拼写错误的 assignee、已删除的 profile 以及宕机的外部 worker 池——与标识无关，无需维护每个看板的白名单。

## 相关资源

- [Kanban 概览](./kanban) — 面向用户的介绍。
- [Kanban 教程](./kanban-tutorial) — 开启仪表板的完整演练。
- [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) — worker 进程加载的 skill。
- [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) — orchestrator 侧。