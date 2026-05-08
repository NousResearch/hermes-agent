# Kanban 教程

Hermes Kanban 系统设计的四个用例的完整演练，仪表板在浏览器中打开。如果您尚未阅读 [Kanban 概述](./kanban)，请先阅读——这假设您知道什么是任务、运行、受理人和调度器。

## 设置

```bash
hermes kanban init           # 可选；首次 `hermes kanban <anything>` 自动初始化
hermes dashboard             # 在浏览器中打开 http://127.0.0.1:9119
# 点击左侧导航中的 Kanban
```

仪表板是**您**观看系统的最舒适位置。调度器生成的 agent 工作线程永远看不到仪表板或 CLI——它们通过专用 `kanban_*` [工具集](./kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`）驱动看板。三个界面——仪表板、CLI、工作线程工具——都通过同一个每看板 SQLite DB（默认看板在 `~/.hermes/kanban.db`，后续创建的任何看板在 `~/.hermes/kanban/boards/<slug>/kanban.db`），因此无论变更来自栅栏哪一侧，每个看板都保持一致。

本教程全程使用 `default` 看板。如果您想要多个隔离队列（每个项目/repo/域一个），请参见概述中的[看板（多项目）](./kanban#boards-multi-project)——相同的 CLI / 仪表板 / 工作线程流程适用于每个看板，工作线程在物理上无法看到其他看板上的任务。

在整个教程中，**标记为 `bash` 的代码块是您运行的命令。** 标记为 `# worker tool calls` 的代码块是生成的工作线程模型发出的工具调用——在此展示以便您可以看到端到端循环，而不是因为您会自己运行它们。

## 看板一览

![Kanban board overview](/img/kanban-tutorial/01-board-overview.png)

六列，从左到右：

- **Triage** — 原始想法，规格器会在任何人处理之前将规格细化。点击任何分类卡上的 **✨ Specify** 按钮（或从聊天中运行 `hermes kanban specify <id>` / `/kanban specify <id>`）让辅助 LLM 将一句话扩展为完整规格（目标、方法、验收标准）并一次性提升到 `todo`。在 `config.yaml` 的 `auxiliary.triage_specifier` 下配置运行它的模型。
- **Todo** — 已创建但等待依赖项，或尚未分配。
- **Ready** — 已分配并等待调度器认领。
- **In progress** — 工作线程正在积极运行任务。开启"Lanes by profile"（默认）时，此列按受理人分组显示，让您一目了然每个工作线程在做什么。
- **Blocked** — 工作线程请求人工输入，或断路器跳闸。
- **Done** — 已完成。

顶部栏有搜索、租户和受理人过滤器，加上 `Lanes by profile` 切换和 `Nudge dispatcher` 按钮，立即运行一次调度 tick 而不是等待守护进程的下一次间隔。点击任何卡片在右侧打开其抽屉。

### 扁平视图

如果画像lane太吵，关闭"Lanes by profile"，"In Progress"列折叠为按认领时间排序的单一列表：

![Board with lanes by profile off](/img/kanban-tutorial/02-board-flat.png)

## 故事 1 — 独立开发发布功能

您正在构建功能。经典流程：设计 schema、实现 API、编写测试。三个有父子依赖的任务。

```bash
SCHEMA=$(hermes kanban create "Design auth schema" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --body "Design the user/session/token schema for the auth module." \
    --json | jq -r .id)

API=$(hermes kanban create "Implement auth API endpoints" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --parent $SCHEMA \
    --body "POST /register, POST /login, POST /refresh, POST /logout." \
    --json | jq -r .id)

hermes kanban create "Write auth integration tests" \
    --assignee qa-dev --tenant auth-project --priority 2 \
    --parent $API \
    --body "Cover happy path, wrong password, expired token, concurrent refresh."
```

因为 `API` 以 `SCHEMA` 为父，`tests` 以 `API` 为父，只有 `SCHEMA` 从 `ready` 开始。其他两个停留在 `todo` 直到其父完成。这是依赖提升引擎的工作——在有 API 可测试之前，其他工作线程不会拾取测试编写。

在下次调度器 tick（默认 60s，或如果您点击 **Nudge dispatcher** 立即）时，`backend-dev` 画像生成为工作线程，`HERMES_KANBAN_TASK=$SCHEMA` 在其环境中。以下是工作线程从 agent 内部视角看到的工具调用循环：

```python
# worker tool calls — NOT commands you run
kanban_show()
# → returns title, body, worker_context, parents, prior attempts, comments

# (worker reads worker_context, uses terminal/file tools to design the schema,
#  write migrations, run its own checks, commit — the real work happens here)

kanban_heartbeat(note="schema drafted, writing migrations now")

kanban_complete(
    summary="users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); "
            "refresh tokens stored as sessions with type='refresh'",
    metadata={
        "changed_files": ["migrations/001_users.sql", "migrations/002_sessions.sql"],
        "decisions": ["bcrypt for hashing", "JWT for session tokens",
                      "7-day refresh, 15-min access"],
    },
)
```

`kanban_show` 默认 `task_id` 为 `$HERMES_KANBAN_TASK`，因此工作线程不需要知道自己的 id。`kanban_complete` 将 summary + metadata 写入当前 `task_runs` 行，关闭该运行，并一次性原子跳转到 `kanban_db` 中的任务到 `done`——所有这些都在一次原子操作中。

当 `SCHEMA` 达到 `done` 时，依赖引擎自动将 `API` 提升到 `ready`。API 工作线程拾取时将调用 `kanban_show()`，看到附加到父交接的 `SCHEMA` 的 summary 和 metadata——因此它知道 schema 决策而无需重新阅读冗长的设计文档。

点击看板上已完成的 schema 任务，抽屉显示一切：

![Solo dev — completed schema task drawer](/img/kanban-tutorial/03-drawer-schema-task.png)

底部的 Run History 部分是关键添加。一次尝试：结果 `completed`，工作线程 `@backend-dev`，持续时间，时间戳，完整交接 summary。metadata blob（`changed_files`、`decisions`）也存储在该运行上，并被任何读取此父的下游工作线程获取。

您可以随时从终端检查相同数据——这些命令是**您**窥视看板，不是工作线程：

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## 故事 2 — 舰队耕种

您有三个工作线程（翻译员、转录员、文案）和一堆独立任务。您希望三个并行拉取并产生可见进度。这是最简单的 kanban 用例，也是原始设计优化的用例。

创建工作：

```bash
for lang in Spanish French German; do
    hermes kanban create "Translate homepage to $lang" \
        --assignee translator --tenant content-ops
done
for i in 1 2 3 4 5; do
    hermes kanban create "Transcribe Q3 customer call #$i" \
        --assignee transcriber --tenant content-ops
done
for sku in 1001 1002 1003 1004; do
    hermes kanban create "Generate product description: SKU-$sku" \
        --assignee copywriter --tenant content-ops
done
```

启动网关并离开——它托管嵌入式调度器，在同一个 kanban.db 上拾取所有三个专家画像的任务：

```bash
hermes gateway start
```

现在过滤看板到 `content-ops`（或只搜索"Transcribe"），您会看到：

![Fleet view filtered to transcribe tasks](/img/kanban-tutorial/07-fleet-transcribes.png)

两个转录完成，一个运行中，两个 ready 等待下次调度器 tick。"In Progress"列按画像分组（"Lanes by profile"默认），因此您可以看到每个工作线程的活动任务而不必扫描混合列表。调度器将在当前任务完成时立即将下一个 ready 任务提升到 running。三个守护进程在三个受理人池上并行工作，整个内容队列无需进一步人工输入即可排空。

**故事 1 中关于结构化交接的所有内容在这里仍然适用。** 完成调用的翻译员工作线程发出 `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})`——对分析和任何依赖此任务的下游任务都很有用。

## 故事 3 — 带重试的角色流水线

这才是 Kanban 比扁平 TODO 列表更有价值的地方。PM 编写规格。工程师实现它。审查员拒绝第一次尝试。工程师更改后重试。审查员批准。

过滤到 `auth-project` 的仪表板视图：

![Pipeline view for a multi-role feature](/img/kanban-tutorial/08-pipeline-auth.png)

三阶段链一目了然：`Spec: password reset flow`（已完成，pm）、`Implement password reset flow`（已完成，backend-dev）、`Review password reset PR`（Ready，reviewer）。每个在底部绿色显示其父，依赖项为子。

有趣的是实现任务，因为它被阻止并重试。以下是三个 agent 编排的完整工具调用序列，每个工作线程的模型：

```python
# --- PM worker spawns on $SPEC and writes the acceptance criteria ---
# worker tool calls
kanban_show()
kanban_complete(
    summary="spec approved; POST /forgot-password sends email, "
            "GET /reset/:token renders form, POST /reset applies new password",
    metadata={"acceptance": [
        "expired token returns 410",
        "reused last-3 password returns 400 with message",
        "successful reset invalidates all active sessions",
    ]},
)
# → $SPEC is done; $IMPL auto-promotes from todo to ready

# --- Engineer worker spawns on $IMPL (first attempt) ---
# worker tool calls
kanban_show()   # reads $SPEC's summary + acceptance metadata in worker_context
# (engineer writes code, runs tests, opens PR)
# Reviewer feedback arrives — engineer decides the concerns are valid and blocks
kanban_block(
    reason="Review: password strength check missing, reset link isn't "
           "single-use (can be replayed within 30min)",
)
# → $IMPL transitions to blocked; run 1 closes with outcome='blocked'
```

现在您（人类或单独的审查员画像）阅读阻止原因，决定修复方向明确，并从仪表板的"Unblock"按钮或 CLI / 斜杠命令解阻：

```bash
hermes kanban unblock $IMPL
# or from a chat: /kanban unblock $IMPL
```

调度器将 `$IMPL` 提升回 `ready`，并在下次 tick 时重新生成 `backend-dev` 工作线程。这是同一任务上的**新运行**：

```python
# --- Engineer worker spawns on $IMPL (second attempt) ---
# worker tool calls
kanban_show()
# → worker_context now includes the run 1 block reason, so this worker knows
#   which two things to fix instead of re-reading the whole spec
# (engineer adds zxcvbn check, makes reset tokens single-use, re-runs tests)
kanban_complete(
    summary="added zxcvbn strength check, reset tokens are now single-use "
            "(stored + deleted on success)",
    metadata={
        "changed_files": [
            "auth/reset.py",
            "auth/tests/test_reset.py",
            "migrations/003_single_use_reset_tokens.sql",
        ],
        "tests_run": 11,
        "review_iteration": 2,
    },
)
```

点击实现任务。抽屉显示**两次尝试**：

![Implementation task with two runs — blocked then completed](/img/kanban-tutorial/04b-drawer-retry-history-scrolled.png)

- **Run 1** — 由 `@backend-dev` `blocked`。审查反馈直接在结果下："password strength check missing, reset link isn't single-use (can be replayed within 30min)"。
- **Run 2** — 由 `@backend-dev` `completed`。新 summary，新 metadata。

每次运行都是 `task_runs` 中的一行，有自己的结果、summary 和 metadata。重试历史不是叠加在"最新状态"任务之上的概念花招——它是主要表示。当重试工作线程打开任务时，`build_worker_context` 向它展示先前的尝试，因此第二轮工作线程看到为什么第一轮被阻止并解决那些具体发现而不是从头重试。

审查员接下来拾取。当他们打开 `Review password reset PR` 时，看到：

![Reviewer's drawer view of the pipeline](/img/kanban-tutorial/09-drawer-pipeline-review.png)

父链接是已完成的实现。当审查员工作线程在 `Review password reset PR` 上生成并调用 `kanban_show()` 时，返回的 `worker_context` 包含父的最新完成运行的 summary + metadata——因此审查员阅读"added zxcvbn strength check, reset tokens are now single-use"，并在查看 diff 之前就已掌握更改文件列表。

## 故事 4 — 断路器和崩溃恢复

真实工作线程会失败。缺失凭证、OOM kill、瞬态网络错误。调度器有两道防线：断路器在 N 次连续失败后自动阻止以防止看板永远抖动，以及崩溃检测，在工作线程 PID 在 TTL 到期前消失时回收任务。

### 断路器 — 永久性失败

一个无法生成工作线程的部署任务，因为 `AWS_ACCESS_KEY_ID` 未在画像环境中设置：

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops
```

调度器尝试生成工作线程。生成失败（`RuntimeError: AWS_ACCESS_KEY_ID not set`）。调度器释放认领，递增失败计数器，并在下次 tick 重试。三次连续失败后（默认 `failure_limit`），电路跳闸：任务进入 `blocked`，结果 `gave_up`。直到人类解阻才重试。

点击被阻止的任务：

![Circuit breaker — 2 spawn_failed + 1 gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

三次运行，都有相同错误的 `error` 字段。前两次是 `spawn_failed`（可重试），第三次是 `gave_up`（终止）。上方事件日志显示完整序列：`created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`。

在终端：

```bash
hermes kanban runs t_ef5d
# #   OUTCOME        PROFILE        ELAPSED  STARTED
# 1   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 2   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 3   gave_up        deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
```

如果 Telegram / Discord / Slack 已接入，`gave_up` 事件上会触发网关通知，以便您无需检查看板也能听到中断。

### 崩溃恢复 — 工作线程中途死亡

有时生成成功但工作线程进程稍后死亡——segfault、OOM、`systemctl stop`。调度器轮询 `kill(pid, 0)` 并检测死 pid；认领释放，任务返回 `ready`，下次 tick 交给新工作线程。

种子数据中的一个例子是因内存不足而运行的迁移：

```bash
# Worker claims, starts scanning 2.4M rows, OOM kills it at ~2.3M
# Dispatcher detects dead pid, releases claim, increments attempt counter
# Retry with a chunked strategy succeeds
```

抽屉显示完整的两次尝试历史：

![Crash and recovery — 1 crashed + 1 completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

运行 1 — `crashed`，错误 `OOM kill at row 2.3M (process 99999 gone)`。运行 2 — `completed`，metadata 中有 `"strategy": "chunked with LIMIT + WHERE id > last_id"`。重试工作线程在上下文中看到运行 1 的崩溃并选择了更安全的策略；metadata 让未来观察者（或事后分析作者）清楚看到什么改变了。

## 结构化交接 — 为什么 `summary` 和 `metadata` 很重要

在上述每个故事中，工作线程在结束时调用 `kanban_complete(summary=..., metadata=...)`。这不是装饰——它是工作流各阶段之间的主要交接通道。

当工作线程在任务 B 上生成并调用 `kanban_show()` 时，它返回的 `worker_context` 包括：

- B 的**先前尝试**（先前运行：结果、summary、错误、metadata），以便重试工作线程不重复失败路径。
- **父任务结果** — 对于每个父，最近完成运行的 summary 和 metadata — 以便下游工作线程看到上游工作为什么及如何完成。

这取代了困扰扁平 kanban 系统的"翻阅评论和工作输出"舞蹈。PM 在规格的 metadata 中编写验收标准，工程师的工作线程在父交接中从结构上看到它们。工程师记录他们运行了哪些测试以及通过多少，审查员的工作线程在打开 diff 之前就在手中有了该列表。

批量关闭守卫存在是因为数据是按运行的。`hermes kanban complete a b c --summary X`（您从 CLI）被拒绝——将相同 summary 复制粘贴到三个任务几乎总是错误的。无语境标志的批量关闭仍然适用于常见的"我完成了一堆管理任务"用例。工具表面根本不暴露批量变体；`kanban_complete` 始终一次只处理一个任务，原因相同。

## 检查当前运行中的任务

为完整性——这是仍在飞行中的任务抽屉（在故事 1 的 API 实现，由 `backend-dev` 认领但尚未完成）：

![Claimed, in-flight task](/img/kanban-tutorial/10-drawer-in-flight.png)

状态为 `Running`。活动运行出现在 Run History 部分，结果 `active` 且无 `ended_at`。如果此工作线程死亡或超时，调度器用适当结果关闭此运行并在下次认领时打开新的——尝试行永远不会消失。

## 下一步

- [Kanban 概述](./kanban) — 完整数据模型、事件词汇和 CLI 参考。
- `hermes kanban --help` — 每个子命令，每个标志。
- `hermes kanban watch --kinds completed,gave_up,timed_out` — 跨整个看板的实时终端事件流。
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 当特定任务完成时获取网关 ping。
