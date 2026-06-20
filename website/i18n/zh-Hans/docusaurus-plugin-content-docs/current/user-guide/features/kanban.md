---
sidebar_position: 12
title: "Kanban（多 Agent 看板）"
description: "基于 SQLite 的持久化任务看板，用于协调多个 Hermes 配置文件"
---

# Kanban — 多 Agent 配置文件协作

> **想要详细教程？** 请阅读 [Kanban 教程](./kanban-tutorial) —— 包含四个用户故事（独立开发者、批量任务、带重试的角色流水线、熔断器），并附有各场景的仪表盘截图。本页是参考文档，教程是叙述性说明。

Hermes Kanban 是一个持久化任务看板，在所有 Hermes 配置文件之间共享，允许多个具名 agent 协作完成工作，而无需脆弱的进程内子 agent 集群。每个任务都是 `~/.hermes/kanban.db` 中的一行记录；每次交接都是任何人都可以读写的一行记录；每个 worker 都是拥有独立身份的完整 OS 进程。

### 两个操作界面：模型通过工具交互，你通过 CLI 交互

看板有两个入口，均由同一个 `~/.hermes/kanban.db` 支撑：

- **Agent 通过专用 `kanban_*` 工具集驱动看板** —— `kanban_show`、`kanban_list`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`、`kanban_unblock`。调度器在 schema 中已内置这些工具来启动每个 worker；编排器（orchestrator）配置文件也可以通过 `kanban` 工具集显式启用。模型通过直接调用工具来读取和路由任务，*而不是*通过 shell 执行 `hermes kanban`。详见下方[Worker 如何与看板交互](#how-workers-interact-with-the-board)。
- **你（以及脚本和 cron）通过 CLI 上的 `hermes kanban …`、斜杠命令 `/kanban …` 或仪表盘驱动看板。** 这些界面面向人类和自动化场景——即没有工具调用模型的场合。

两个界面都通过同一个 `kanban_db` 层路由，因此读取视图一致，写入不会产生偏差。本页其余部分展示 CLI 示例，因为它们便于复制粘贴，但每个 CLI 动词都有模型使用的等效工具调用。

这种形态覆盖了 `delegate_task` 无法处理的工作负载：

- **研究分诊** —— 并行研究员 + 分析师 + 写作者，支持人工介入。
- **定时运维** —— 每日定期简报，逐周积累日志。
- **数字孪生** —— 持久化具名助手（`inbox-triage`、`ops-review`），随时间积累记忆。
- **工程流水线** —— 分解 → 在并行 worktree 中实现 → 审查 → 迭代 → PR。
- **批量任务** —— 一个专家管理 N 个对象（50 个社交账号、12 个监控服务）。

完整的设计原理、与 Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise 的对比分析，以及八种典型协作模式，请参阅仓库中的 `docs/hermes-kanban-v1-spec.pdf`。

## Kanban 与 `delegate_task` 的对比

两者看起来相似，但并非同一原语。

| | `delegate_task` | Kanban |
|---|---|---|
| 形态 | RPC 调用（fork → join） | 持久化消息队列 + 状态机 |
| 父级 | 阻塞直到子级返回 | `create` 后即发即忘 |
| 子级身份 | 匿名子 agent | 具有持久记忆的具名配置文件 |
| 可恢复性 | 无 —— 失败即失败 | 阻塞 → 解除阻塞 → 重新运行；崩溃 → 回收 |
| 人工介入 | 不支持 | 随时可评论 / 解除阻塞 |
| 每任务 agent 数 | 一次调用 = 一个子 agent | 任务生命周期内 N 个 agent（重试、审查、跟进） |
| 审计追踪 | 上下文压缩后丢失 | 永久保存在 SQLite 行中 |
| 协调方式 | 层级式（调用方 → 被调用方） | 对等式 —— 任意配置文件可读写任意任务 |

**一句话区别：** `delegate_task` 是函数调用；Kanban 是工作队列，每次交接都是任意配置文件（或人类）可见和编辑的一行记录。

**使用 `delegate_task` 的场景：** 父 agent 在继续之前需要一个简短的推理答案，无需人工介入，结果返回到父 agent 的上下文中。

**使用 Kanban 的场景：** 工作跨越 agent 边界、需要在重启后存活、可能需要人工输入、可能被不同角色接手，或需要事后可发现。

两者可以共存：kanban worker 在运行期间可以内部调用 `delegate_task`。

## 核心概念

- **Board（看板）** —— 一个独立的任务队列，拥有自己的 SQLite DB、工作区目录和调度器循环。单次安装可以有多个看板（例如每个项目、仓库或领域一个）；详见下方[看板（多项目）](#boards-multi-project)。单项目用户保持使用 `default` 看板，在本文档章节之外不会看到"board"这个词。
- **Task（任务）** —— 包含标题、可选正文、一个受让人（配置文件名称）、状态（`triage | todo | ready | running | blocked | done | archived`）、可选租户命名空间、可选幂等键（用于重试自动化的去重）的一行记录。
- **Link（链接）** —— `task_links` 行，记录父 → 子依赖关系。当所有父任务变为 `done` 时，调度器将 `todo → ready`。
- **Comment（评论）** —— agent 间协议。Agent 和人类追加评论；当 worker 被（重新）启动时，它将完整的评论线程作为上下文的一部分读取。
- **Workspace（工作区）** —— worker 操作的目录。三种类型：
  - `scratch`（默认）—— 在 `~/.hermes/kanban/workspaces/<id>/` 下（非默认看板为 `~/.hermes/kanban/boards/<slug>/workspaces/<id>/`）创建的临时目录。**任务完成时删除** —— scratch 是临时性的，worker（或 `hermes kanban complete <id>`）将任务标记为完成的那一刻，目录即被清除。如果想保留 worker 的输出，请使用 `worktree:` 或 `dir:<path>`。在某次安装中首次创建 scratch 工作区时，调度器会记录警告并在任务上发出 `tip_scratch_workspace` 事件（可通过 `hermes kanban show <id>` 查看）。
  - `dir:<path>` —— 现有的共享目录（Obsidian vault、邮件运维目录、每账号文件夹）。**必须是绝对路径。** 像 `dir:../tenants/foo/` 这样的相对路径在调度时会被拒绝，因为它们会相对于调度器碰巧所在的 CWD 解析，这是模糊的，也是混淆代理（confused-deputy）逃逸向量。路径本身是受信任的 —— 这是你的机器、你的文件系统，worker 以你的 uid 运行。这是受信任本地用户的威胁模型；kanban 设计为单主机。**完成时保留。**
  - `worktree` —— 用于编码任务的 git worktree，位于 `.worktrees/<id>/` 下。使用 `worktree:<path>` 固定确切的目标路径。Worker 端的 `git worktree add` 创建它，提供 `--branch` 时使用该分支。**完成时保留。**
- **Dispatcher（调度器）** —— 一个长期运行的循环，每 N 秒（默认 60 秒）执行一次：回收过期的认领、回收崩溃的 worker（PID 消失但 TTL 尚未过期）、推进就绪任务、原子性认领、启动已分配的配置文件。默认**在 gateway 内部运行**（`kanban.dispatch_in_gateway: true`）。每次 tick 一个调度器扫描所有看板；worker 启动时固定了 `HERMES_KANBAN_BOARD`，因此无法看到其他看板。在同一任务上连续启动失败 `kanban.failure_limit` 次（默认：2）后，调度器会以最后一个错误为原因自动阻塞该任务 —— 防止因配置文件不存在、工作区无法挂载等原因导致的反复抖动。
- **Tenant（租户）** —— 看板*内*的可选字符串命名空间。一个专家团队可以通过工作区路径和内存键前缀为多个业务提供数据隔离服务（`--tenant business-a`）。租户是软过滤器；看板是硬隔离边界。

## 看板（多项目） {#boards-multi-project}

看板让你将不相关的工作流分离到独立的队列中 —— 每个项目、仓库或领域一个。新安装只有一个名为 `default` 的看板（DB 位于 `~/.hermes/kanban.db`，保持向后兼容）。只需要一个工作流的用户无需了解看板；该功能是可选启用的。

每个看板的隔离是绝对的：

- 每个看板有独立的 SQLite DB（`~/.hermes/kanban/boards/<slug>/kanban.db`）。
- 独立的 `workspaces/` 和 `logs/` 目录。
- 为任务启动的 Worker 只能看到**其所在看板**的任务 —— 调度器在子进程环境中设置 `HERMES_KANBAN_BOARD`，worker 可访问的每个 `kanban_*` 工具都会读取它。
- 不允许跨看板链接任务（保持 schema 简单；如果确实需要跨项目引用，请使用自由文本提及并通过 id 手动查找）。

### 通过 CLI 管理看板

```bash
# 查看磁盘上的内容。全新安装只显示 "default"。
hermes kanban boards list

# 创建新看板。
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # 可选：将其设为活动看板

# 在不切换的情况下操作特定看板。
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# 更改后续调用的"当前"看板。
hermes kanban boards switch atm10-server
hermes kanban boards show             # 当前活动的是哪个？

# 重命名显示名称（slug 是不可变的 —— 它是目录名）。
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# 归档（默认）—— 将看板目录移动到 boards/_archived/<slug>-<ts>/。
# 可通过将目录移回来恢复。
hermes kanban boards rm atm10-server

# 硬删除 —— 对看板目录执行 `rm -rf`。无法恢复。
hermes kanban boards rm atm10-server --delete
```

看板解析顺序（优先级从高到低）：

1. CLI 调用中的显式 `--board <slug>`。
2. `HERMES_KANBAN_BOARD` 环境变量（调度器在启动 worker 时设置，因此 worker 无法看到其他看板）。
3. `~/.hermes/kanban/current` —— 由 `hermes kanban boards switch` 持久化的 slug。
4. `default`。

Slug 经过验证：小写字母数字 + 连字符 + 下划线，1-64 个字符，必须以字母数字开头。大写输入会自动转为小写。其他任何内容（斜杠、空格、点、`..`）在 CLI 层被拒绝，以防止路径遍历技巧命名看板。

### 通过仪表盘管理看板

`hermes dashboard` → Kanban 标签页在存在多个看板（或任何看板有任务）时，顶部会显示看板切换器。单看板用户只看到一个小的 `+ New board` 按钮；切换器在需要时才显示。

- **看板下拉菜单** —— 选择活动看板。你的选择保存在浏览器的 `localStorage` 中，因此在重新加载后仍然有效，不会影响你打开的终端中 CLI 的 `current` 指针。
- **+ New board** —— 打开一个模态框，询问 slug、显示名称、描述和图标。可选择自动切换到新看板。
- **Archive** —— 仅在非 `default` 看板上显示。确认后，将看板目录移动到 `boards/_archived/`。

所有仪表盘 API 端点接受 `?board=<slug>` 进行看板范围限定。事件 WebSocket 在连接时固定到一个看板；在 UI 中切换会针对新看板打开一个新的 WS。


## 快速开始

以下命令是**你**（人类）设置看板和创建任务的操作。一旦任务被分配，调度器就会将分配的配置文件作为 worker 启动，从那时起**模型通过 `kanban_*` 工具调用驱动任务，而不是 CLI 命令** —— 详见[Worker 如何与看板交互](#how-workers-interact-with-the-board)。

```bash
# 1. 创建看板（你）
hermes kanban init

# 2. 启动 gateway（托管内嵌调度器）
hermes gateway start

# 3. 创建任务（你 —— 或编排器 agent 通过 kanban_create）
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. 实时查看活动（你）
hermes kanban watch

# 5. 查看看板（你）
hermes kanban list
hermes kanban stats
```

当调度器接管 `t_abcd` 并启动 `researcher` 配置文件时，该 worker 的模型做的第一件事是调用 `kanban_show()` 读取其任务。它不会运行 `hermes kanban show t_abcd`。

### Gateway 内嵌调度器（默认）

调度器在 gateway 进程内运行。无需安装任何东西，无需管理单独的服务 —— 只要 gateway 运行，就绪任务会在下一个 tick（默认 60 秒）被接管。

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # 默认
  dispatch_interval_seconds: 60    # 默认
```

通过 `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0` 在运行时覆盖配置标志以进行调试。标准 gateway 监督适用：直接运行 `hermes gateway start`，或将 gateway 配置为 systemd 用户单元（参见 gateway 文档）。没有运行中的 gateway，`ready` 任务会保持原状，直到 gateway 启动 —— `hermes kanban create` 在创建时会对此发出警告。

将 `hermes kanban daemon` 作为单独进程运行已**弃用**；请使用 gateway。如果你确实无法运行 gateway（无头主机策略禁止长期运行的服务等），`--force` 逃生舱口在一个发布周期内保持旧的独立守护进程可用，但同时运行 gateway 内嵌调度器和针对同一 `kanban.db` 的独立守护进程会导致认领竞争，不受支持。

### 幂等创建（用于自动化 / webhook）

```bash
# 第一次调用创建任务。使用相同键的任何后续调用
# 返回现有任务 id 而不是重复创建。
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### 批量 CLI 动词

所有生命周期动词都接受多个 id，因此你可以在一个命令中清理一批任务：

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

## Worker 如何与看板交互 {#how-workers-interact-with-the-board}

**Worker 不会 shell 执行 `hermes kanban`。** 当调度器启动 worker 时，它在子进程环境中设置 `HERMES_KANBAN_TASK=t_abcd`，该环境变量在模型的 schema 中启用专用的 **kanban 工具集**。同一工具集也可供在工具集配置中启用 `kanban` 的编排器配置文件使用。这些工具通过 Python `kanban_db` 层直接读取和修改看板，与 CLI 的做法相同。运行中的 worker 像调用任何其他工具一样调用这些工具；它从不看到或需要 `hermes kanban` CLI。

| 工具 | 用途 | 必需参数 |
|---|---|---|
| `kanban_show` | 读取当前任务（标题、正文、先前尝试、父级交接、评论、完整预格式化的 `worker_context`）。默认使用环境变量中的任务 id。 | — |
| `kanban_list` | 列出带有 `assignee`、`status`、`tenant`、归档可见性和限制过滤器的任务摘要。供编排器发现看板工作使用。 | — |
| `kanban_complete` | 以 `summary` + `metadata` 结构化交接完成任务。 | `summary` / `result` 至少一个 |
| `kanban_block` | 以 `reason` 上报需要人工输入。 | `reason` |
| `kanban_heartbeat` | 在长时间操作期间发出存活信号。纯副作用。 | — |
| `kanban_comment` | 向任务线程追加持久化备注。 | `task_id`、`body` |
| `kanban_create` | （编排器）将任务扇出为带有 `assignee`、可选 `parents`、`skills` 等的子任务。 | `title`、`assignee` |
| `kanban_link` | （编排器）事后添加 `parent_id → child_id` 依赖边。 | `parent_id`、`child_id` |
| `kanban_unblock` | （编排器）将被阻塞的任务移回 `ready`。 | `task_id` |

典型的 worker 轮次如下所示：

```
# 模型的工具调用，按顺序：
kanban_show()                                     # 无参数 —— 使用 HERMES_KANBAN_TASK
# （模型读取返回的 worker_context，通过终端/文件工具完成工作）
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# （更多工作）
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

**编排器** worker 则进行扇出：

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
# → 返回 {"task_id": "t_r1", ...}
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → 返回 {"task_id": "t_r2", ...}
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # 两者都完成时推进到 ready
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

"（编排器）"工具 —— `kanban_list`、`kanban_create`、`kanban_link`、`kanban_unblock`，以及对外部任务的 `kanban_comment` —— 通过同一工具集提供；约定（由 `kanban-orchestrator` skill 强制执行）是 worker 配置文件不进行扇出或路由无关工作，编排器配置文件不执行实现工作。调度器启动的 worker 仍然针对破坏性生命周期操作限定在任务范围内，无法修改无关任务。

### 为什么使用工具而不是 shell 执行 `hermes kanban`

三个原因：

1. **后端可移植性。** 终端工具指向远程后端（Docker / Modal / Singularity / SSH）的 worker 会在容器*内部*运行 `hermes kanban complete`，而容器中没有安装 `hermes`，也没有挂载 `~/.hermes/kanban.db`。kanban 工具在 agent 自己的 Python 进程中运行，无论终端后端如何，始终能访问 `~/.hermes/kanban.db`。
2. **无 shell 引用脆弱性。** 通过 shlex + argparse 传递 `--metadata '{"files": [...]}'` 是潜在的隐患。结构化工具参数完全绕过了这个问题。
3. **更好的错误处理。** 工具结果是模型可以推理的结构化 JSON，而不是需要解析的 stderr 字符串。

**对普通会话零 schema 占用。** 普通的 `hermes chat` 会话在其 schema 中没有任何 `kanban_*` 工具，除非活动配置文件为编排器工作显式启用了 `kanban` 工具集。调度器启动的任务 worker 因为设置了 `HERMES_KANBAN_TASK` 而获得任务范围的工具；编排器配置文件通过配置获得更广泛的路由界面。对于从不使用 kanban 的用户，没有工具膨胀。

`kanban-worker` 和 `kanban-orchestrator` skill 教导模型何时调用哪个工具以及调用顺序。

### 推荐的交接证据

`kanban_complete(summary=..., metadata={...})` 是有意灵活的：summary 是人类可读的收尾说明，`metadata` 是机器可读的交接信息，下游 agent、审查者或仪表盘可以直接复用，无需从文本中提取。

对于工程和审查任务，推荐使用以下可选 metadata 格式：

```json
{
  "changed_files": ["path/to/file.py"],
  "verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
  "dependencies": ["parent task id or external issue, if any"],
  "blocked_reason": null,
  "retry_notes": "what failed before, if this was a retry",
  "residual_risk": ["what was not tested or still needs human review"]
}
```

这些键是约定，不是 schema 要求。有用的特性是每个 worker 留下足够的证据，让下一个读者能快速回答四个问题：

1. 改了什么？
2. 如何验证的？
3. 如果失败，什么可以解除阻塞或重试？
4. 什么风险是有意留下的？

不要将密钥、原始日志、token（令牌）、OAuth 材料和无关记录放入 `metadata`。改为存储指针和摘要。如果任务没有文件或测试，在 `summary` 中明确说明，并在 `metadata` 中放置确实存在的证据，例如来源 URL、issue id 或手动审查步骤。

### Worker skill

任何应该能够处理 kanban 任务的配置文件都必须加载 `kanban-worker` skill。它通过**工具调用**（而非 CLI 命令）教导 worker 完整的生命周期：

1. 启动时，调用 `kanban_show()` 读取标题 + 正文 + 父级交接 + 先前尝试 + 完整评论线程。
2. 通过终端工具执行 `cd $HERMES_KANBAN_WORKSPACE`，在那里完成工作。
3. 在长时间操作期间每隔几分钟调用一次 `kanban_heartbeat(note="...")`。**如果你的工作可能运行超过 1 小时，请至少每小时调用一次 `kanban_heartbeat`** —— 调度器会回收运行时间超过 `kanban.dispatch_stale_timeout_seconds`（默认 4 小时）且最近一小时内没有心跳的任务，认为 worker 在没有清理的情况下崩溃了。回收是无害的（任务返回 `ready` 重新调度，不增加失败计数器），但你会失去当前运行的进度。
4. 以 `kanban_complete(summary="...", metadata={...})` 完成，或在卡住时以 `kanban_block(reason="...")` 完成。

最终的 `kanban_complete` / `kanban_block` 调用是 worker 协议的一部分。如果 worker 进程以状态 0 退出而任务仍处于 `running` 状态，调度器将其视为协议违规，发出 `protocol_violation` 事件，并在下一个 tick 自动阻塞任务而不是重新启动它进入同一循环。这通常意味着模型写了一个纯文本答案并退出，而没有使用 Kanban 工具界面。

`kanban-worker` 是一个内置 skill，在安装和更新期间同步到每个配置文件 —— 无需单独的 Skills Hub 安装步骤。验证它是否存在于你用于 kanban worker 的配置文件中（`researcher`、`writer`、`ops` 等）：

```bash
hermes -p <your-worker-profile> skills list | grep kanban-worker
```

如果内置副本丢失，为该配置文件恢复它：

```bash
hermes -p <your-worker-profile> skills reset kanban-worker --restore
```

调度器在启动每个 worker 时也会自动传递 `--skills kanban-worker`，因此即使配置文件的默认 skills 配置不包含它，worker 也始终拥有该模式库。

### 为特定任务固定额外 skill

有时单个任务需要受让人配置文件默认不携带的专业上下文 —— 需要 `translation` skill 的翻译任务、需要 `github-code-review` 的审查任务、需要 `security-pr-audit` 的安全审计。与其每次都编辑受让人的配置文件，不如直接将 skill 附加到任务上。

**从编排器 agent**（常见情况 —— 一个 agent 将工作路由到另一个），使用 `kanban_create` 工具的 `skills` 数组：

```
kanban_create(
    title="translate README to Japanese",
    assignee="linguist",
    skills=["translation"],
)

kanban_create(
    title="audit auth flow",
    assignee="reviewer",
    skills=["security-pr-audit", "github-code-review"],
)
```

**从人类（CLI / 斜杠命令）**，为每个 skill 重复 `--skill`：

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**从仪表盘**，在内联创建表单的 **skills** 字段中以逗号分隔输入 skill 名称。

这些 skill 是对内置 `kanban-worker` 的**补充** —— 调度器为每个 skill（以及内置的）发出一个 `--skills <name>` 标志，因此 worker 启动时加载了所有这些 skill。skill 名称必须与受让人配置文件上实际安装的 skill 匹配（运行 `hermes skills list` 查看可用内容）；没有运行时安装。

### 编排器 skill

**行为良好的编排器不会自己做工作。** 它将用户的目标分解为任务，链接它们，将每个任务分配给你设置的配置文件之一，然后退后。`kanban-orchestrator` skill 将此编码为工具调用模式：反诱惑规则、Step-0 配置文件发现提示（调度器在未知受让人名称上静默失败，因此编排器必须将每张卡片落地到你机器上实际存在的配置文件），以及以 `kanban_create` / `kanban_link` / `kanban_comment` 为核心的分解手册。

典型的编排器轮次（两个并行研究员交接给一个写作者）：

```
# 来自用户的目标："draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # 两个研究员都完成时推进到 'ready'
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# 可选：事后发现的跨切依赖，无需重新创建任务
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

`kanban-orchestrator` 是一个内置 skill。它在安装和更新期间同步到每个配置文件，因此无需单独的 Skills Hub 安装步骤。验证它是否存在于你的编排器配置文件中：

```bash
hermes -p orchestrator skills list | grep kanban-orchestrator
```

如果内置副本丢失，为该配置文件恢复它：

```bash
hermes -p orchestrator skills reset kanban-orchestrator --restore
```

为获得最佳效果，将其与工具集限制为看板操作（`kanban`、`gateway`、`memory`）的配置文件配对，这样编排器即使尝试也无法执行实现任务。

## 仪表盘（GUI）

`/kanban` CLI 和斜杠命令足以无头运行看板，但可视化看板通常是人工介入的正确界面：分诊、跨配置文件监督、阅读评论线程以及在列之间拖动卡片。Hermes 将此作为**内置仪表盘插件**在 `plugins/kanban/` 中提供 —— 不是核心功能，不是单独的服务 —— 遵循[扩展仪表盘](./extending-the-dashboard)中描述的模型。

使用以下命令打开：

```bash
hermes kanban init      # 一次性：如果尚未创建 kanban.db
hermes dashboard        # 导航栏中出现 "Kanban" 标签页，位于 "Skills" 之后
```

### 插件提供的功能

- 一个 **Kanban** 标签页，每个状态显示一列：`triage`、`todo`、`ready`、`running`、`blocked`、`done`（开启切换时还有 `archived`）。
  - `triage` 是粗略想法的停车列。默认情况下（`kanban.auto_decompose: true`），调度器会自动对落在这里的任务运行**分解器** —— 编排器配置文件读取粗略想法，查看你的配置文件名册（含描述），并将任务扇出为路由到最合适专家的小型子任务图。原始任务作为每个子任务的父级保持存活，因此当所有子任务完成时，编排器会重新唤醒以判断完成情况，并在工作未完成时添加更多任务。点击页面顶部的 **Orchestration: Auto/Manual** 切换按钮（或设置 `kanban.auto_decompose: false`）切换到手动模式，在手动模式下分诊任务保持原位，直到你点击卡片上的 **⚗ Decompose** 或运行 `hermes kanban decompose <id>`。对于不需要扇出的任务（或没有编排器配置文件的设置），**✨ Specify** 按钮通过相同的 LLM 机制进行单任务规格重写（标题 + 正文，包含目标、方法、验收标准）。详见下方[自动与手动编排](#auto-vs-manual-orchestration)。
- 卡片显示任务 id、标题、优先级徽章、租户标签、分配的配置文件、评论/链接计数、**进度标签**（任务有依赖项时显示 `N/M` 子任务已完成）以及"N 前创建"。每张卡片的复选框启用多选。
- **Running 列内的按配置文件分组** —— 工具栏复选框切换 Running 列按受让人的子分组。
- **通过 WebSocket 实时更新** —— 插件以短轮询间隔追踪仅追加的 `task_events` 表；任何配置文件（CLI、gateway 或另一个仪表盘标签页）操作后，看板立即反映变化。重新加载经过防抖处理，因此一批事件只触发一次重新获取。
- **拖放**卡片在列之间更改状态。拖放操作发送 `PATCH /api/plugins/kanban/tasks/:id`，通过与 CLI 使用的相同 `kanban_db` 代码路由 —— 三个界面永远不会产生偏差。移动到破坏性状态（`done`、`archived`、`blocked`）时会提示确认。触摸设备使用基于指针的回退，因此看板可以在平板电脑上使用。
- **内联创建** —— 点击任意列标题上的 `+`，输入标题、受让人、优先级，以及（可选）从所有现有任务的下拉菜单中选择父任务。按 Enter 创建任务，Shift+Enter 在标题字段中插入换行，或按 Escape 取消。从 Triage 列创建会自动将新任务停放在分诊中。
- **多选与批量操作** —— shift/ctrl 点击卡片或勾选其复选框将其添加到选择中。顶部出现批量操作栏，包含批量状态转换、归档和重新分配（通过配置文件下拉菜单，或"（取消分配）"）。破坏性批量操作先确认。每个 id 的部分失败会被报告，不会中止其余操作。
- **点击卡片**（不按 shift/ctrl）打开侧边抽屉（按 Escape 或点击外部关闭），包含：
  - **可编辑标题** —— 点击标题进行重命名。
  - **可编辑受让人 / 优先级** —— 点击元数据行进行修改。
  - **可编辑描述** —— 默认以 markdown 渲染（标题、粗体、斜体、内联代码、围栏代码、`http(s)` / `mailto:` 链接、项目符号列表），带有"编辑"按钮可切换到文本区域。Markdown 渲染是一个微型、防 XSS 的渲染器 —— 每次替换都在 HTML 转义的输入上运行，只有 `http(s)` / `mailto:` 链接通过，并且始终设置 `target="_blank"` + `rel="noopener noreferrer"`。
  - **依赖编辑器** —— 父级和子级的芯片列表，每个都有 `×` 用于取消链接，加上所有其他任务的下拉菜单用于添加新的父级或子级。循环尝试在服务器端被拒绝并给出清晰的消息。
  - **状态操作行**（→ triage / → ready / → running / block / unblock / complete / archive），破坏性转换有确认提示。对于 **Triage** 列中的卡片，该行还提供两个 LLM 驱动的操作：**⚗ Decompose** 将任务扇出为路由到专家配置文件（按描述）的子任务图（编排器驱动路径），**✨ Specify** 进行单任务规格重写。当 LLM 判断任务不需要扇出时，Decompose 会回退到类似 specify 的推进，因此它是严格的超集。两者都可以从 CLI（`hermes kanban decompose <id>` / `specify <id>` / `--all`）、任何 gateway 平台（`/kanban decompose <id>`）以及通过 `POST /api/plugins/kanban/tasks/:id/decompose` 和 `…/specify` 以编程方式访问。在 `config.yaml` 的 `auxiliary.kanban_decomposer` 和 `auxiliary.triage_specifier` 下配置模型。
  - 结果部分（也以 markdown 渲染）、带 Enter 提交的评论线程、最近 20 个事件。
- **工具栏过滤器** —— 自由文本搜索、租户下拉菜单（默认为 `config.yaml` 中的 `dashboard.kanban.default_tenant`）、受让人下拉菜单、"显示已归档"切换、"按配置文件分组"切换，以及**推动调度器**按钮，这样你就不必等待下一个 60 秒 tick。

视觉上目标是熟悉的 Linear / Fusion 布局：深色主题、带计数的列标题、彩色状态点、优先级和租户的标签芯片。插件只读取主题 CSS 变量（`--color-*`、`--radius`、`--font-mono` 等），因此它会随活动的仪表盘主题自动重新换肤。

### 自动与手动编排 {#auto-vs-manual-orchestration}

看板有两种方式处理你放入 Triage 列的任务：

**自动（默认）** —— `kanban.auto_decompose: true`。Gateway 内嵌调度器在每个 tick 运行**分解器**，受 `kanban.auto_decompose_per_tick`（默认每 tick 3 个任务）限制，以防批量加载分诊任务时突发消耗辅助 LLM。分解器读取粗略想法，查看你安装的配置文件及其描述，并要求 LLM 生成 JSON 任务图：要启动哪些任务、分配给谁，以及哪些依赖哪些。原始分诊任务成为图中每个叶节点的父级，因此它保持存活直到整个图完成 —— 然后推进回 `ready`，让其受让人（编排器配置文件）判断完成情况，并在工作未完成时添加更多任务。这是"丢一行描述，走开"的流程。

**手动** —— `kanban.auto_decompose: false`。分诊任务保持在分诊中，直到你操作。点击卡片上的 **⚗ Decompose** 按钮，运行 `hermes kanban decompose <id>`（或 `--all`），或从聊天中使用 `/kanban decompose <id>`。这与看板的预分解器行为一致，适合需要完全控制运行时机的场景。

从 kanban 页面顶部的 **Orchestration: Auto/Manual** 切换按钮（翠绿色 = 自动，静音灰色 = 手动）在两种模式之间切换，或直接编辑 `config.yaml`。两种模式都与 `hermes kanban specify` 共存 —— 当你不想扇出时，它仍然可用作单任务规格重写。

分解器的路由决策依赖于配置文件描述，这是一个每配置文件的标签原语，通过 `hermes profile create --description "..."`、`hermes profile describe <name> --text "..."`、`hermes profile describe <name> --auto`（LLM 从配置文件安装的 skill + 模型自动生成），或仪表盘展开的 **Orchestration settings** 面板中的每配置文件编辑器来设置。没有描述的配置文件仍然出现在名册中 —— 它们可以按名称路由，只是精度较低。分解器**绝不**会将子任务落地为 `assignee=None`：当 LLM 选择未知配置文件时，子任务路由到 `kanban.default_assignee`（如果未设置，则路由到活动默认配置文件）。

配置项（均在 `~/.hermes/config.yaml` 的 `kanban:` 下）：

| 键 | 默认值 | 用途 |
|---|---|---|
| `auto_decompose` | `true` | 调度器每 tick 自动运行分解器。 |
| `auto_decompose_per_tick` | `3` | 每个调度器 tick 的分解上限。超出部分推迟到下一个 tick。 |
| `orchestrator_profile` | `""` | 拥有分解权的配置文件。空 = 回退到活动默认配置文件。 |
| `default_assignee` | `""` | LLM 选择未知配置文件时子任务的落地位置。空 = 回退到活动默认配置文件。 |
| `auto_subscribe_on_create` | `true` | 当 worker 在具有持久投递通道的会话（消息网关或 TUI）内调用 `kanban_create` 时，原始会话会自动订阅新任务的完成/阻塞事件。调度器仍负责驱动投递 —— 此设置只决定调用者的聊天/密钥是否出现在通知订阅表中。设为 `false` 则要求对每个任务显式调用 `kanban_notify-subscribe`。 |

以及两个辅助 LLM 槽：

| 键 | 用途 |
|---|---|
| `auxiliary.kanban_decomposer` | 生成任务图的模型（由 Decompose 调用）。设置 `provider`/`model` 以覆盖主聊天模型。 |
| `auxiliary.profile_describer` | 自动生成配置文件描述的模型（由 `hermes profile describe --auto` 调用）。 |

### 架构

GUI 严格是一个**通过 DB 读取 + 通过 kanban_db 写入**的层，没有自己的领域逻辑：

<!-- ascii-guard-ignore -->
```
┌────────────────────────┐      WebSocket (tails task_events)
│   React SPA (plugin)   │ ◀──────────────────────────────────┐
│   HTML5 drag-and-drop  │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     writes call kanban_db.*        │
│  FastAPI router        │     directly — same code path      │
│  plugins/kanban/       │     the CLI /kanban verbs use      │
│  dashboard/plugin_api.py                                    │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── append task_events ──────────┘
│  (WAL, shared)         │
└────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

### REST 接口

所有路由挂载在 `/api/plugins/kanban/` 下，并受仪表盘的临时会话 token 保护：

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | 按状态列分组的完整看板，加上用于过滤下拉菜单的租户和受让人 |
| `GET` | `/tasks/:id` | 任务 + 评论 + 事件 + 链接 |
| `POST` | `/tasks` | 创建（封装 `kanban_db.create_task`，接受 `triage: bool` 和 `parents: [id, …]`） |
| `PATCH` | `/tasks/:id` | 状态 / 受让人 / 优先级 / 标题 / 正文 / 结果 |
| `POST` | `/tasks/bulk` | 对 `ids` 中的每个 id 应用相同的补丁（状态 / 归档 / 受让人 / 优先级）。每个 id 的失败不会中止其他操作 |
| `POST` | `/tasks/:id/comments` | 追加评论 |
| `POST` | `/tasks/:id/specify` | 运行分诊规格器 —— 辅助 LLM 充实任务正文并将其从 `triage` 推进到 `todo`。返回 `{ok, task_id, reason, new_title}`；"不在分诊中" / 无辅助客户端 / LLM 错误时 `ok=false` 并附人类可读原因，返回 200 而非 4xx |
| `POST` | `/tasks/:id/decompose` | 运行 kanban 分解器 —— 辅助 LLM 生成任务图，辅助函数原子性创建子任务 + 链接根任务 + 翻转 `triage → todo`。返回 `{ok, task_id, reason, fanout, child_ids, new_title}`。与 `/specify` 相同的 LLM 错误返回 200 约定。 |
| `GET` | `/profiles` | 列出已安装的配置文件及其描述（供仪表盘的配置文件描述编辑器和编排器选择器使用）。 |
| `PATCH` | `/profiles/:name` | 设置或清除配置文件的描述（用户编写 —— `description_auto: false`）。返回 `{ok, profile, description}`。 |
| `POST` | `/profiles/:name/describe-auto` | 通过 `auxiliary.profile_describer` 为配置文件生成描述。以 `description_auto: true` 持久化，以便仪表盘可以显示"审查"徽章。 |
| `GET` | `/orchestration` | 读取 kanban 编排设置（`orchestrator_profile`、`default_assignee`、`auto_decompose`）以及回退后的*解析*有效值。 |
| `PUT` | `/orchestration` | 在 `config.yaml` 中更新三个编排键中的一个或多个。验证非空配置文件名实际存在。 |
| `POST` | `/links` | 添加依赖关系（`parent_id` → `child_id`） |
| `DELETE` | `/links?parent_id=…&child_id=…` | 删除依赖关系 |
| `POST` | `/dispatch?max=…&dry_run=…` | 推动调度器 —— 跳过 60 秒等待 |
| `GET` | `/config` | 从 `config.yaml` 读取 `dashboard.kanban` 偏好设置 —— `default_tenant`、`lane_by_profile`、`include_archived_by_default`、`render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events` 行的实时流 |

每个处理器都是一个薄封装 —— 插件约 700 行 Python（路由器 + WebSocket 追踪 + 批量处理器 + 配置读取器），不添加任何新的业务逻辑。一个微型 `_conn()` 辅助函数在每次读写时自动初始化 `kanban.db`，因此无论用户是先打开仪表盘、直接访问 REST API，还是运行 `hermes kanban init`，全新安装都能正常工作。

### 仪表盘配置

`~/.hermes/config.yaml` 中 `dashboard.kanban` 下的任何这些键都会更改标签页的默认值 —— 插件在加载时通过 `GET /config` 读取它们：

```yaml
dashboard:
  kanban:
    default_tenant: acme              # 预选租户过滤器
    lane_by_profile: true             # "按配置文件分组"切换的默认值
    include_archived_by_default: false
    render_markdown: true             # 设为 false 则使用纯 <pre> 渲染
```

每个键都是可选的，回退到所示的默认值。

### 安全模型

仪表盘的 HTTP 认证中间件[显式跳过 `/api/plugins/`](./extending-the-dashboard#backend-api-routes) —— 插件路由在设计上是未认证的，因为仪表盘默认绑定到 localhost。这意味着 kanban REST 接口可以从主机上的任何进程访问。

WebSocket 额外增加了一步：它要求仪表盘的临时会话 token 作为 `?token=…` 查询参数（浏览器无法在升级请求上设置 `Authorization`），与浏览器内 PTY 桥使用的模式一致。

如果你运行 `hermes dashboard --host 0.0.0.0`，每个插件路由 —— 包括 kanban —— 都可以从网络访问。**不要在共享主机上这样做。** 看板包含任务正文、评论和工作区路径；攻击者访问这些路由可以读取你整个协作界面，还可以创建 / 重新分配 / 归档任务。

`~/.hermes/kanban.db` 中的任务是有意与配置文件无关的（这是协调原语）。如果你用 `hermes -p <profile> dashboard` 打开仪表盘，看板仍然显示主机上任何其他配置文件创建的任务。同一用户拥有所有配置文件，但如果多个角色共存，这一点值得了解。

### 实时更新

`task_events` 是一个带有单调递增 `id` 的仅追加 SQLite 表。WebSocket 端点保存每个客户端最后看到的事件 id，并在新行到达时推送。当一批事件到达时，前端重新加载（非常廉价的）看板端点 —— 比尝试从每种事件类型修补本地状态更简单、更正确。WAL 模式意味着读取循环永远不会阻塞调度器的 `BEGIN IMMEDIATE` 认领事务。

### 扩展

插件使用标准的 Hermes 仪表盘插件契约 —— 完整的 manifest 参考、shell 槽、页面范围槽和 Plugin SDK，请参阅[扩展仪表盘](./extending-the-dashboard)。额外的列、自定义卡片样式、租户过滤布局或完整的 `tab.override` 替换都可以表达，无需 fork 此插件。

要禁用而不删除：在 `config.yaml` 中添加 `dashboard.plugins.kanban.enabled: false`（或删除 `plugins/kanban/dashboard/manifest.json`）。

### 范围边界

GUI 是刻意精简的。插件所做的一切都可以从 CLI 访问；插件只是让人类使用起来更舒适。自动分配、预算、治理门控和组织图视图仍然是用户空间 —— 一个路由器配置文件、另一个插件，或对 `tools/approval.py` 的复用 —— 正如设计规范的范围外章节所列。

## CLI 命令参考

这是**你**（或脚本、cron、仪表盘）用来驱动看板的界面。在调度器内部运行的 Worker 使用 `kanban_*` [工具界面](#how-workers-interact-with-the-board)进行相同的操作 —— 这里的 CLI 和那里的工具都通过 `kanban_db` 路由，因此两个界面在构造上是一致的。

```
hermes kanban init                                     # 创建 kanban.db + 打印守护进程提示
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|worktree:<path>|dir:<path>]
                                [--branch <name>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--max-retries N]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived] [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # 或 'none' 取消分配
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# 批量动词 —— 接受多个 id：
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban tail <id>                                # 跟踪单个任务的事件流
hermes kanban watch [--assignee P] [--tenant T]        # 将所有事件实时流式传输到终端
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # 长时间操作的 worker 存活信号
hermes kanban runs <id> [--json]                       # 尝试历史（每次运行一行）
hermes kanban assignees [--json]                       # 磁盘上的配置文件 + 每受让人任务计数
hermes kanban dispatch [--dry-run] [--max N]           # 单次扫描
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # 已弃用 —— 独立调度器（改用 `hermes gateway start`）
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # 每状态 + 每受让人计数
hermes kanban log <id> [--tail BYTES]                  # 来自 ~/.hermes/kanban/logs/ 的 worker 日志
hermes kanban notify-subscribe <id>                    # gateway 桥接钩子（由 gateway 中的 /kanban 使用）
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # worker 看到的内容
hermes kanban specify [<id> | --all] [--tenant T]      # 将分诊列的想法充实
        [--author NAME] [--json]                       #   为完整规格并推进到 todo
hermes kanban gc [--event-retention-days N]            # 工作区 + 旧事件 + 旧日志
        [--log-retention-days N]
```

所有命令也可以作为交互式 CLI 中的斜杠命令和消息 gateway 中使用（见下方[`/kanban` 斜杠命令](#kanban-slash-command)）。

`--max-retries` 是调度器的每任务熔断器覆盖。`--max-retries 1` 在第一次不成功的尝试后阻塞任务，而 `--max-retries 3` 允许两次重试并在第三次失败时阻塞。省略它则使用 `config.yaml` 中的 `kanban.failure_limit`，然后是内置默认值。

## `/kanban` 斜杠命令 {#kanban-slash-command}

每个 `hermes kanban <action>` 动词也可以作为 `/kanban <action>` 访问 —— 从交互式 `hermes chat` 会话内部**以及**从任何 gateway 平台（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、电子邮件、SMS）。两个界面都调用完全相同的 `hermes_cli.kanban.run_slash()` 入口点，该入口点复用 `hermes kanban` argparse 树，因此参数界面、标志和输出格式在 CLI、`/kanban` 和 `hermes kanban` 之间完全相同。你不必离开聊天来驱动看板。

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # 将分诊一行描述充实为真正的规格
/kanban specify --all --tenant engineering  # 一次性扫描某个租户中的所有分诊任务
```

以与 shell 相同的方式引用多词参数 —— `run_slash` 用 `shlex.split` 解析行的其余部分，因此 `"..."` 和 `'...'` 都有效。

### 运行中使用：`/kanban` 绕过运行中 agent 保护

Gateway 通常在 agent 仍在思考时将斜杠命令和用户消息排队 —— 这就是防止你在第一轮还在进行时意外启动第二轮的机制。**`/kanban` 被明确豁免于此保护。** 看板存在于 `~/.hermes/kanban.db` 中，而不是运行中 agent 的状态中，因此读取（`list`、`show`、`context`、`tail`、`watch`、`stats`、`runs`）和写入（`comment`、`unblock`、`block`、`assign`、`archive`、`create`、`link` 等）都会立即执行，即使在轮次进行中。

这就是分离的全部意义：

- Worker 阻塞等待对等方 → 你从手机发送 `/kanban unblock t_abcd`，调度器在下一个 tick 接管对等方。被阻塞的 worker 不会被中断 —— 它只是不再被阻塞。
- 你发现一张需要人工上下文的卡片 → `/kanban comment t_xyz "use the 2026 schema, not 2025"` 落在任务线程上，该任务的*下一次*运行将在 `kanban_show()` 中读取它。
- 你想知道你的团队在做什么而不停止编排器 → `/kanban list --mine` 或 `/kanban stats` 在不触及主对话的情况下检查看板。

### `/kanban create` 时自动订阅（仅限 gateway）

当你从 gateway 使用 `/kanban create "…"` 创建任务时，发起聊天（平台 + 聊天 id + 线程 id）会自动订阅该任务的终端事件（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）。每个终端事件你会收到一条消息回复 —— 包括 `completed` 时 worker 结果摘要的第一行 —— 无需轮询或记住任务 id。

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

订阅在任务达到 `done` 或 `archived` 后自动移除。如果你用 `--json`（机器输出）脚本化创建，则跳过自动订阅 —— 假设脚本化调用者希望通过 `/kanban notify-subscribe` 显式管理订阅。

### 消息中的输出截断

Gateway 平台有实际的消息长度限制。如果 `/kanban list`、`/kanban show` 或 `/kanban tail` 产生超过约 3800 个字符的输出，响应会被截断，并附上 `… (truncated; use \`hermes kanban …\` in your terminal for full output)` 页脚。CLI 界面没有此限制。

### 自动补全

在交互式 CLI 中，输入 `/kanban ` 并按 Tab 会循环显示内置子命令列表（`list`、`ls`、`show`、`create`、`assign`、`link`、`unlink`、`claim`、`comment`、`complete`、`block`、`unblock`、`archive`、`tail`、`dispatch`、`context`、`init`、`gc`）。上方 CLI 参考中列出的其余动词（`watch`、`stats`、`runs`、`log`、`assignees`、`heartbeat`、`notify-subscribe`、`notify-list`、`notify-unsubscribe`、`daemon`）也有效 —— 它们只是尚未出现在自动补全提示列表中。

## 协作模式

看板无需任何新原语即可支持以下八种模式：

| 模式 | 形态 | 示例 |
|---|---|---|
| **P1 扇出** | N 个同级，相同角色 | "并行研究 5 个角度" |
| **P2 流水线** | 角色链：侦察 → 编辑 → 写作 | 每日简报组装 |
| **P3 投票 / 法定人数** | N 个同级 + 1 个聚合器 | 3 个研究员 → 1 个审查者选择 |
| **P4 长期运行日志** | 相同配置文件 + 共享目录 + cron | Obsidian vault |
| **P5 人工介入** | worker 阻塞 → 用户评论 → 解除阻塞 | 模糊决策 |
| **P6 `@mention`** | 从文本内联路由 | `@reviewer look at this` |
| **P7 线程范围工作区** | 线程中的 `/kanban here` | 每项目 gateway 线程 |
| **P8 批量任务** | 一个配置文件，N 个对象 | 50 个社交账号 |
| **P9 分诊规格器** | 粗略想法 → `triage` → `hermes kanban specify` 扩展正文 → `todo` | "将这个一行描述变成规格化任务" |

每种模式的详细示例，请参阅 `docs/hermes-kanban-v1-spec.pdf`。

## 多租户使用

当一个专家团队为多个业务提供服务时，为每个任务添加租户标签：

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

Worker 接收 `$HERMES_TENANT` 并按前缀命名空间化其内存写入。看板、调度器和配置文件定义都是共享的；只有数据是有范围的。

## Gateway 通知

当你从 gateway（Telegram、Discord、Slack 等）运行 `/kanban create …` 时，发起聊天会自动订阅新任务。Gateway 的后台通知器每隔几秒轮询 `task_events`，并为每个终端事件（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）向该聊天发送一条消息。已完成的任务还会发送 worker `--result` 的第一行，这样你无需 `/kanban show` 就能看到结果。

你可以从 CLI 显式管理订阅 —— 当脚本 / cron 任务想要通知一个它不是从那里发起的聊天时很有用：

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

订阅在任务达到 `done` 或 `archived` 后自动移除；无需清理。

## 运行记录 —— 每次尝试一行

任务是一个逻辑工作单元；**运行**是执行它的一次尝试。当调度器认领一个就绪任务时，它在 `task_runs` 中创建一行，并将 `tasks.current_run_id` 指向它。当该尝试结束时 —— 完成、阻塞、崩溃、超时、启动失败、回收 —— 运行行以 `outcome` 关闭，任务的指针清除。被尝试三次的任务有三行 `task_runs`。

为什么用两张表而不是直接修改任务：你需要**完整的尝试历史**用于真实世界的事后分析（"第二次审查尝试到达批准，第三次合并"），你需要一个干净的地方挂载每次尝试的元数据 —— 哪些文件改变了、哪些测试运行了、审查者注意到了哪些发现。这些是运行事实，不是任务事实。

运行也是**结构化交接**所在的地方。当 worker 完成任务（通过 `kanban_complete(...)`）时，它可以传递：

- `summary`（工具参数）/ `--summary`（CLI）—— 人类交接；放在运行上；下游子任务在其 `build_worker_context` 中看到它。
- `metadata`（工具参数）/ `--metadata`（CLI）—— 运行上的自由格式 JSON 字典；子任务看到它与摘要一起序列化。
- `result`（工具参数）/ `--result`（CLI）—— 放在任务行上的简短日志行（遗留字段，保留向后兼容）。

下游子任务读取每个父任务最近完成运行的摘要 + 元数据。重试 worker 读取其自身任务上的先前尝试（结果、摘要、错误），以避免重复已经失败的路径。

```
# worker 实际做的事 —— agent 循环内的工具调用：
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

当你（人类）需要关闭 worker 无法关闭的任务时，同样的交接可以从 CLI 访问 —— 例如被放弃的任务，或你从仪表盘手动标记为完成的任务：

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# 查看重试任务的尝试历史：
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

运行在仪表盘上公开（抽屉中的运行历史部分，每次尝试一行彩色行）以及 REST API 上（`GET /api/plugins/kanban/tasks/:id` 返回 `runs[]` 数组）。带有 `{status: "done", summary, metadata}` 的 `PATCH /api/plugins/kanban/tasks/:id` 将两者都转发到内核，因此仪表盘的"标记完成"按钮等同于 CLI。`task_events` 行携带它们所属的 `run_id`，以便 UI 可以按尝试分组，`completed` 事件在其有效载荷中嵌入第一行摘要（上限 400 个字符），这样 gateway 通知器无需第二次 SQL 往返即可渲染结构化交接。

**批量关闭注意事项。** `hermes kanban complete a b c --summary X` 被拒绝 —— 结构化交接是每次运行的，因此将相同的摘要复制粘贴到 N 个任务几乎总是错误的。不带 `--summary` / `--metadata` 的批量关闭仍然适用于常见的"我完成了一堆管理任务"情况。

**状态变更导致的运行回收。** 如果你在仪表盘中将运行中的任务从 `running` 拖走（回到 `ready`，或直接到 `todo`），或归档仍在运行的任务，进行中的运行以 `outcome='reclaimed'` 关闭，而不是被孤立。当 `tasks.current_run_id` 为 `NULL` 时，`task_runs` 行始终处于终端状态，反之亦然 —— 该不变量在 CLI、仪表盘、调度器和通知器之间保持。

**从未认领的完成的合成运行。** 完成或阻塞从未被认领的任务（例如，人类从仪表盘关闭带摘要的 `ready` 任务，或 CLI 用户运行 `hermes kanban complete <ready-task> --summary X`）否则会丢失交接。相反，内核插入一个零持续时间运行行（`started_at == ended_at`），携带摘要 / 元数据 / 原因，以保持尝试历史完整。`completed` / `blocked` 事件的 `run_id` 指向该行。

**实时抽屉刷新。** 当仪表盘的 WebSocket 事件流报告用户当前正在查看的任务的新事件时，抽屉会重新加载自身（通过线程到其 `useEffect` 依赖列表中的每任务事件计数器）。不再需要关闭并重新打开才能看到运行的新行或更新的结果。

### 向前兼容性

`tasks` 上的两个可空列为 v2 工作流路由保留：`workflow_template_id`（此任务属于哪个模板）和 `current_step_key`（该模板中哪个步骤处于活动状态）。v1 内核忽略它们用于路由，但允许客户端写入它们，因此 v2 版本可以添加路由机制而无需另一次 schema 迁移。

## 事件参考

每次转换都向 `task_events` 追加一行。每行携带一个可选的 `run_id`，以便 UI 可以按尝试分组事件。类型分为三个集群，便于过滤（`hermes kanban watch --kinds completed,gave_up,timed_out`）：

**生命周期**（关于任务作为逻辑单元发生了什么变化）：

| 类型 | 有效载荷 | 时机 |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | 任务插入。`run_id` 为 `NULL`。 |
| `promoted` | — | 因所有父任务达到 `done` 而 `todo → ready`。`run_id` 为 `NULL`。 |
| `claimed` | `{lock, expires, run_id}` | 调度器原子性认领 `ready` 任务以启动。 |
| `completed` | `{result_len, summary?}` | Worker 写入 `--result` / `--summary` 且任务达到 `done`。`summary` 是第一行交接（400 字符上限）；完整版本存在于运行行上。如果在从未认领的任务上调用 `complete_task` 并带有交接字段，则合成零持续时间运行，以便 `run_id` 仍然指向某处。 |
| `blocked` | `{reason}` | Worker 或人类将任务翻转为 `blocked`。在带有 `--reason` 的从未认领任务上调用时合成零持续时间运行。 |
| `unblocked` | — | `blocked → ready`，手动或通过 `/unblock`。`run_id` 为 `NULL`。 |
| `archived` | — | 从默认看板中隐藏。如果任务仍在运行，携带作为副作用被回收的运行的 `run_id`。 |

**编辑**（不是转换的人类驱动变更）：

| 类型 | 有效载荷 | 时机 |
|---|---|---|
| `assigned` | `{assignee}` | 受让人更改（包括取消分配）。 |
| `edited` | `{fields}` | 标题或正文更新。 |
| `reprioritized` | `{priority}` | 优先级更改。 |
| `status` | `{status}` | 仪表盘拖放直接写入状态（例如 `todo → ready`）。从 `running` 拖走时携带被回收运行的 `run_id`；否则 `run_id` 为 NULL。 |

**Worker 遥测**（关于执行过程，而非逻辑任务）：

| 类型 | 有效载荷 | 时机 |
|---|---|---|
| `spawned` | `{pid}` | 调度器成功启动 worker 进程。 |
| `heartbeat` | `{note?}` | Worker 在长时间操作期间调用 `hermes kanban heartbeat $TASK` 发出存活信号。 |
| `reclaimed` | `{stale_lock}` | 认领 TTL 在完成前过期；任务返回 `ready`。 |
| `crashed` | `{pid, claimer}` | Worker PID 不再存活但 TTL 尚未过期。 |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | 超过 `max_runtime_seconds`；调度器发送 SIGTERM（5 秒宽限后发送 SIGKILL）并重新排队。 |
| `stale` | `{elapsed_seconds, last_heartbeat_at, heartbeat_age_seconds, timeout_seconds, pid, terminated}` | 任务运行时间超过 `kanban.dispatch_stale_timeout_seconds`（默认 4 小时）**且**最近一小时内没有 `kanban_heartbeat`。调度器向本地 worker（如有）发送 SIGTERM，将任务重置为 `ready` 重新调度。**不**增加失败计数器（stale 是调度器端的缺席检测，不是 worker 故障）。运行长时间操作的 Worker 应至少每小时调用一次 `kanban_heartbeat` 以避免此情况。 |
| `respawn_guarded` | `{reason}` | 调度器拒绝在本 tick 重新启动此就绪任务。原因：`blocker_auth`（上次失败是配额/认证/429 错误 —— 等待速率窗口重置）、`recent_success`（最近一小时内有完成的运行 —— 在重新运行前等待审查）、`active_pr`（最近的评论中出现 GitHub PR URL —— 先前的 worker 已经打开了 PR）。任务保持在 `ready`；下一个 tick 有另一次启动机会。如果底层条件持续存在，正常的 `consecutive_failures` 熔断器将在 `failure_limit` 次失败后通过 `gave_up` 自动阻塞。 |
| `spawn_failed` | `{error, failures}` | 一次启动尝试失败（PATH 缺失、工作区无法挂载等）。计数器递增；任务返回 `ready` 重试。 |
| `protocol_violation` | `{pid, claimer, exit_code}` | Worker 在任务仍处于 `running` 状态时成功退出，通常是因为它回答了问题而没有调用 `kanban_complete` 或 `kanban_block`。调度器还会立即发出 `gave_up` 并自动阻塞，而不是重试。 |
| `gave_up` | `{failures, effective_limit, limit_source, error}` | N 次连续不成功尝试后熔断器触发。任务以最后一个错误自动阻塞。有效限制解析为任务 `max_retries`，然后是调度器 `failure_limit` / `kanban.failure_limit`，然后是内置默认值。 |

`hermes kanban tail <id>` 显示单个任务的这些事件。`hermes kanban watch` 在整个看板范围内流式传输它们。

## 范围之外

Kanban 是刻意单主机的。`~/.hermes/kanban.db` 是本地 SQLite 文件，调度器在同一台机器上启动 worker。不支持跨两台主机运行共享看板 —— 没有"主机 A 上的 worker X，主机 B 上的 worker Y"的协调原语，崩溃检测路径假设 PID 是主机本地的。如果你需要多主机，每台主机运行独立的看板，并使用 `delegate_task` / 消息队列来桥接它们。

## 设计规范

完整的设计 —— 架构、并发正确性、与其他系统的比较、实现计划、风险、开放问题 —— 存在于 `docs/hermes-kanban-v1-spec.pdf` 中。在提交任何行为变更 PR 之前请先阅读它。