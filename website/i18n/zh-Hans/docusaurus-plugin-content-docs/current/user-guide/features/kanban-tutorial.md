# Kanban 教程

本文将带你了解 Hermes Kanban 系统的四个典型使用场景，建议在浏览器中打开 dashboard 边读边操作。如果你还没看过 [Kanban 概览](./kanban)，建议先阅读——本文默认你已了解 task、run、assignee 和 dispatcher 的基本概念。

## 准备工作

```bash
hermes kanban init           # 可选；首次运行 hermes kanban <任意命令> 时会自动初始化
hermes dashboard             # 在浏览器中打开 http://127.0.0.1:9119
# 点击左侧导航栏的 Kanban
```

Dashboard 是**你**观察系统最舒适的地方。Dispatcher 派生的 agent worker 看不到 dashboard 或 CLI——它们通过专用的 `kanban_*` [工具集](./kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_list`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`、`kanban_unblock`）来操作 board。三个操作面——dashboard、CLI、worker 工具——都通过同一个 per-board SQLite DB 进行路由（默认 board 使用 `~/.hermes/kanban.db`，之后创建的 board 使用 `~/.hermes/kanban/boards/<slug>/kanban.db`），因此无论变更来自哪一侧，board 的状态始终保持一致。

本教程全程使用 `default` board。如果你需要多个隔离的队列（每个项目 / 仓库 / 域一个），请参阅概览中的 [Boards（多项目）](./kanban#boards-multi-project)——每个 board 都适用相同的 CLI / dashboard / worker 流程，且 worker 物理上无法看到其他 board 上的 task。

在本教程中，**标记为 `bash` 的代码块是*你*运行的命令。** 标记为 `# worker tool calls` 的代码块是派生 worker 的模型发出的工具调用——这里展示出来是为了让你看到完整的端到端循环，而不是因为你会亲自运行它们。

## 看板概览

![Kanban board overview](/img/kanban-tutorial/01-board-overview.png)

六列，从左到右依次为：

- **Triage** — 原始想法，specifier 会在任何人开始工作之前将其充实为完整 spec。点击任意 triage 卡片上的 **✨ Specify** 按钮（或在聊天中运行 `hermes kanban specify <id>` / `/kanban specify <id>`），即可让辅助 LLM 将一句话扩展为完整的 spec（目标、方法、验收标准），并一次性将其提升至 `todo`。在 `config.yaml` 的 `auxiliary.triage_specifier` 下配置运行它的模型。
- **Todo** — 已创建但正在等待依赖，或尚未分配。
- **Ready** — 已分配，等待 dispatcher 认领。
- **In progress** — worker 正在 actively 运行该 task。开启 "Lanes by profile"（默认开启）后，该列会按 assignee 子分组，让你一眼就能看到每个 worker 在做什么。
- **Blocked** — worker 请求人工输入，或 circuit breaker 触发。
- **Done** — 已完成。

顶部栏有搜索、tenant 和 assignee 的过滤器，以及一个 `Lanes by profile` 开关和一个 `Nudge dispatcher` 按钮，后者会立即执行一次 dispatch tick，而不是等待 daemon 的下一个间隔。点击任意卡片会在右侧打开其 drawer。

### 扁平视图

如果 profile lanes 显得杂乱，关闭 "Lanes by profile"，In Progress 列会折叠为按认领时间排序的单一扁平列表：

![Board with lanes by profile off](/img/kanban-tutorial/02-board-flat.png)

## 场景 1 — 单人开发交付功能

你正在开发一个功能。经典流程：设计 schema、实现 API、编写测试。三个具有 parent→child 依赖关系的 task。

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

因为 `API` 的 parent 是 `SCHEMA`，而 `tests` 的 parent 是 `API`，所以只有 `SCHEMA` 一开始会进入 `ready`。另外两个会停留在 `todo`，直到它们的 parent 完成。这就是 dependency promotion engine 在发挥作用——在有 API 可测试之前，不会有其他 worker 去执行测试编写任务。

在下一次 dispatcher tick（默认 60 秒，或如果你点击了 **Nudge dispatcher** 则立即执行）时，`backend-dev` profile 会作为 worker 被派生，其环境变量中带有 `HERMES_KANBAN_TASK=$SCHEMA`。以下是该 worker 从 agent 内部看到的工具调用循环：

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

`kanban_show` 默认将 `task_id` 设为 `$HERMES_KANBAN_TASK`，因此 worker 不需要知道自己的 id。`kanban_complete` 会将 summary + metadata 写入当前 `task_runs` 行，关闭该 run，并将 task 转移到 `done`——所有操作通过 `kanban_db` 一次性原子完成。

当 `SCHEMA` 进入 `done` 后，dependency engine 会自动将 `API` 提升至 `ready`。API worker 被派生后调用 `kanban_show()`，会看到 `SCHEMA` 的 summary 和 metadata 已附加到 parent handoff 中——因此它无需重新阅读冗长的设计文档，就能了解 schema 的决策。

点击 board 上已完成的 schema task，drawer 会展示所有信息：

![Solo dev — completed schema task drawer](/img/kanban-tutorial/03-drawer-schema-task.png)

底部的 Run History 是关键新增内容。一次尝试：outcome 为 `completed`，worker 为 `@backend-dev`，持续时间、时间戳，以及完整的 handoff summary。metadata blob（`changed_files`、`decisions`）也存储在 run 上，并会展示给任何读取该 parent 的下游 worker。

你可以随时从终端检查相同的数据——这些命令是**你**在查看 board，而不是 worker：

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## 场景 2 — 舰队并行处理

你有三个 worker（一个 translator、一个 transcriber、一个 copywriter）和一堆独立的 task。你希望三个 worker 并行工作并展示可见进度。这是最简单的 kanban 使用场景，也是原始设计优化的目标。

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

启动 gateway 即可离开——它托管 embedded dispatcher，
在同一个 kanban.db 上处理三个 specialist profile 的 task：

```bash
hermes gateway start
```

现在将 board 过滤到 `content-ops`（或搜索 "Transcribe"），你会看到：

![Fleet view filtered to transcribe tasks](/img/kanban-tutorial/07-fleet-transcribes.png)

两个 transcribe 已完成，一个正在运行，两个 ready 等待下一次 dispatcher tick。In Progress 列按 profile 分组（默认的 "Lanes by profile"），因此你可以直接看到每个 worker 的 active task，而无需扫描混合列表。Dispatcher 会在当前 task 完成后立即将下一个 ready task 提升至 running。三个 daemon 并行处理三个 assignee pool，整个内容队列无需进一步人工输入即可清空。

**场景 1 中关于结构化 handoff 的所有内容在这里同样适用。** 完成调用的 translator worker 会发出 `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})`——这对分析很有用，也对任何依赖此 task 的下游 task 有帮助。

## 场景 3 — 角色流水线与重试

这是 Kanban 相比扁平 TODO list 真正体现价值的地方。PM 编写 spec。工程师实现。Reviewer 拒绝了第一次尝试。工程师修改后再次尝试。Reviewer 批准。

Dashboard 视图，按 `auth-project` 过滤：

![Pipeline view for a multi-role feature](/img/kanban-tutorial/08-pipeline-auth.png)

三阶段链一目了然：`Spec: password reset flow`（DONE，pm）、`Implement password reset flow`（DONE，backend-dev）、`Review password reset PR`（READY，reviewer）。每个 task 底部都有其 parent（绿色），以及作为依赖的 children。

最值得关注的是 implementation task，因为它曾被 blocked 并重试过。以下是完整的三 agent 编排，展示为每个 worker 的模型所做的工具调用：

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

现在**你**（人类，或单独的 reviewer profile）阅读 block reason，判断修复方向明确，然后从 dashboard 的 "Unblock" 按钮解除 block——或从 CLI / slash command：

```bash
hermes kanban unblock $IMPL
# 或在聊天中: /kanban unblock $IMPL
```

Dispatcher 将 `$IMPL` 重新提升至 `ready`，并在下一次 tick 时重新派生 `backend-dev` worker。这次派生是该 task 的**新 run**：

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

点击 implementation task。Drawer 展示**两次尝试**：

![Implementation task with two runs — blocked then completed](/img/kanban-tutorial/04b-drawer-retry-history-scrolled.png)

- **Run 1** — `blocked`，由 `@backend-dev` 触发。Review feedback 直接显示在 outcome 下方："password strength check missing, reset link isn't single-use (can be replayed within 30min)"。
- **Run 2** — `completed`，由 `@backend-dev` 完成。全新的 summary，全新的 metadata。

每次 run 都是 `task_runs` 中的一行，拥有自己的 outcome、summary 和 metadata。Retry history 不是事后在 "latest state" task 之上叠加的概念——它就是主要的表现形式。当重试 worker 打开 task 时，`build_worker_context` 会展示之前的尝试，因此第二次 worker 能看到第一次被 block 的原因，并针对这些具体发现进行修复，而不是从头重新运行。

接下来 reviewer 接手。当他们打开 `Review password reset PR` 时，会看到：

![Reviewer's drawer view of the pipeline](/img/kanban-tutorial/09-drawer-pipeline-review.png)

Parent link 是已完成的 implementation。当 reviewer 的 worker 在 `Review password reset PR` 上被派生并调用 `kanban_show()` 时，返回的 `worker_context` 包含 parent 最新完成 run 的 summary + metadata——因此 reviewer 读到 "added zxcvbn strength check, reset tokens are now single-use"，并在查看 diff 之前就已掌握变更文件列表。

## 场景 4 — Circuit breaker 与崩溃恢复

真实的 worker 会失败。凭证缺失、OOM kill、瞬时网络错误。Dispatcher 有两道防线：在 N 次连续失败后自动 block 的 **circuit breaker**，防止 board 永远 thrash；以及 **crash detection**，可以回收 worker PID 在 TTL 到期前就已消失的 task。

### Circuit breaker — 看起来是永久性的失败

一个 deploy task 因为 profile 环境中未设置 `AWS_ACCESS_KEY_ID` 而无法派生 worker：

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops
```

Dispatcher 尝试派生 worker。Spawn 失败（`RuntimeError: AWS_ACCESS_KEY_ID not set`）。Dispatcher 释放 claim，增加失败计数器，并在下一次 tick 重试。三次连续失败后（默认 `failure_limit`），circuit 触发：task 进入 `blocked`，outcome 为 `gave_up`。在人工 unblock 之前不再重试。

点击 blocked task：

![Circuit breaker — 2 spawn_failed + 1 gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

三次 run，error 字段都是相同的错误。前两次是 `spawn_failed`（可重试），第三次是 `gave_up`（终止）。上方的 event log 展示完整序列：`created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`。

在终端上：

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

如果接入了 Telegram / Discord / Slack，`gave_up` 事件会触发 gateway 通知，让你无需查看 board 就能知晓故障。

### Crash recovery — worker 中途崩溃

有时 spawn 成功，但 worker 进程随后死亡——segfault、OOM、`systemctl stop`。Dispatcher 轮询 `kill(pid, 0)` 检测到 dead pid；释放 claim，task 回到 `ready`，下一次 tick 交给新的 worker。

Seed data 中的示例是一个因内存不足而崩溃的 migration：

```bash
# Worker claims, starts scanning 2.4M rows, OOM kills it at ~2.3M
# Dispatcher detects dead pid, releases claim, increments attempt counter
# Retry with a chunked strategy succeeds
```

Drawer 展示完整的两次尝试历史：

![Crash and recovery — 1 crashed + 1 completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

Run 1 — `crashed`，错误为 `OOM kill at row 2.3M (process 99999 gone)`。Run 2 — `completed`，metadata 中包含 `"strategy": "chunked with LIMIT + WHERE id > last_id"`。重试 worker 在其 context 中看到了 run 1 的 crash，因此选择了更安全的策略；metadata 让未来的观察者（或 postmortem 撰写者）一目了然地知道发生了什么变化。

## 结构化 handoff — 为什么 `summary` 和 `metadata` 很重要

在上述每个场景中，worker 最后都调用了 `kanban_complete(summary=..., metadata=...)`。这不是装饰——它是 workflow 各阶段之间的主要 handoff 通道。

当 task B 上的 worker 被派生并调用 `kanban_show()` 时，返回的 `worker_context` 包含：

- B 的**之前尝试**（previous runs：outcome、summary、error、metadata），因此重试 worker 不会重复失败的路径。
- **Parent task results** — 对于每个 parent，最新完成 run 的 summary 和 metadata —— 因此下游 worker 能看到上游工作是如何以及为什么完成的。

这取代了困扰扁平 kanban 系统的 "在评论和工作输出中翻找" 的麻烦。PM 在 spec 的 metadata 中编写 acceptance criteria，工程师的 worker 能在 parent handoff 中结构化地看到它们。工程师记录运行了哪些测试以及通过了多少，reviewer 的 worker 在打开 diff 之前就已掌握该列表。

Bulk-close guard 的存在正是因为这些数据是 per-run 的。`hermes kanban complete a b c --summary X`（你，从 CLI）会被拒绝——将相同的 summary 复制粘贴到三个 task 几乎总是错误的。不带 handoff flags 的 bulk close 仍然适用于常见的 "我完成了一堆 admin task" 场景。工具层面完全不暴露 bulk 变体；`kanban_complete` 始终是一次一个 task，原因相同。

## 查看正在运行的 task

作为补充——以下是仍在运行中的 task 的 drawer（来自场景 1 的 API implementation，已被 `backend-dev` 认领但尚未完成）：

![Claimed, in-flight task](/img/kanban-tutorial/10-drawer-in-flight.png)

Status 为 `Running`。Active run 出现在 Run History 部分，outcome 为 `active`，没有 `ended_at`。如果该 worker 死亡或超时，dispatcher 会以适当的 outcome 关闭此 run，并在下一次 claim 时开启新的 run——尝试行永远不会消失。

## 下一步

- [Kanban overview](./kanban) — 完整的数据模型、event vocabulary 和 CLI 参考。
- `hermes kanban --help` — 每个子命令，每个 flag。
- `hermes kanban watch --kinds completed,gave_up,timed_out` — 在终端上实时流式传输整个 board 的事件。
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 当特定 task 完成时接收 gateway ping。
