---
sidebar_position: 12
title: "看板 (多代理协作板)"
description: "基于 SQLite 的持久化任务看板，用于协调多个 Hermes 配置文件之间的协作"
---

# 看板 — 多配置文件协作

> **想要完整教程？** 请阅读 [看板教程](./kanban-tutorial) — 包含四个用户故事（独立开发者、批量处理、带重试的角色流水线、断路器）以及每个场景的仪表盘截图。本页是参考文档；教程是叙述性文档。

Hermes 看板是一个持久化任务看板，在所有 Hermes 配置文件之间共享，允许多个命名代理协作处理任务，而无需依赖脆弱的进程内子代理群。每个任务都是 `~/.hermes/kanban.db` 中的一行；每次交接都是任何人都可以读写的一行；每个工作进程都是具有独立身份的完整操作系统进程。

### 两个交互面：模型通过工具交互，你通过 CLI 交互

看板有两个入口，两者都基于同一个 `~/.hermes/kanban.db`：

- **代理通过专用的 `kanban_*` 工具集驱动看板** — `kanban_show`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`。调度器在生成每个工作进程时已将这些工具加入其 schema；模型通过直接调用它们来读取任务和交接工作，*而不是*通过 shell 调用 `hermes kanban`。请参阅下方的[工作进程如何与看板交互](#工作进程如何与看板交互)。
- **你（以及脚本、cron）通过 `hermes kanban …` 在 CLI、`/kanban …` 作为斜杠命令，或仪表盘来驱动看板。** 这些是面向人类和自动化的入口 — 背后没有工具调用模型的地方。

两个交互面都通过同一个 `kanban_db` 层路由，因此读取看到一致的视图，写入不会产生偏差。本页其余部分展示 CLI 示例，因为它们便于复制粘贴，但每个 CLI 动词都有模型使用的等效工具调用。

这是能够覆盖 `delegate_task` 无法处理的工作负载的形态：

- **研究分类** — 并行研究人员 + 分析师 + 作者，人工参与。
- **定时运维** — 每日简报，持续数周构建日志。
- **数字孪生** — 持久化命名助手（`inbox-triage`、`ops-review`），随时间积累记忆。
- **工程流水线** — 分解 → 并行工作树实现 → 审查 → 迭代 → PR。
- **批量工作** — 一个专家管理 N 个主题（50 个社交账户、12 个受监控服务）。

如需完整的设计原理、与 Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise 的对比分析以及八种标准协作模式，请参阅仓库中的 `docs/hermes-kanban-v1-spec.pdf`。

## 看板 vs. `delegate_task`

它们看起来相似；但它们不是同一个原语。

| | `delegate_task` | 看板 |
|---|---|---|
| 形态 | RPC 调用（fork → join） | 持久化消息队列 + 状态机 |
| 父进程 | 阻塞直到子进程返回 | `create` 后即发即忘 |
| 子进程身份 | 匿名子代理 | 具有持久记忆的命名配置文件 |
| 可恢复性 | 无 — 失败即失败 | 阻塞 → 解除阻塞 → 重新运行；崩溃 → 回收 |
| 人工参与 | 不支持 | 随时可以评论 / 解除阻塞 |
| 每个任务的代理数 | 一次调用 = 一个子代理 | 任务生命周期内 N 个代理（重试、审查、跟进） |
| 审计轨迹 | 上下文压缩时丢失 | 永久存储在 SQLite 中的持久化行 |
| 协调方式 | 层级式（调用者 → 被调用者） | 对等 — 任何配置文件可以读写任何任务 |

**一句话区别：** `delegate_task` 是一个函数调用；看板是一个工作队列，其中每次交接都是任何配置文件（或人类）可以看到和编辑的一行记录。

**使用 `delegate_task` 的场景：** 父代理在继续之前需要一个短推理答案，不涉及人类，结果返回到父代理的上下文中。

**使用看板的场景：** 工作跨代理边界，需要 survive 重启，可能需要人工输入，可能被不同角色接管，或需要事后可发现。

它们可以共存：一个看板工作进程在其运行期间内部可以调用 `delegate_task`。

## 核心概念

- **看板（Board）** — 一个独立的任务队列，拥有自己的 SQLite 数据库、工作空间目录和调度器循环。单个安装可以拥有多个看板（例如每个项目、仓库或领域一个）；请参阅下方的[多项目看板](#多项目看板)。单项目用户始终使用 `default` 看板，在本文档部分之外永远不会看到"看板"这个词。
- **任务（Task）** — 一行记录，包含标题、可选正文、一个被分配者（配置文件名称）、状态（`triage | todo | ready | running | blocked | done | archived`）、可选租户命名空间、可选幂等键（用于重试自动化的去重）。
- **链接（Link）** — `task_links` 中的一行，记录父 → 子依赖关系。当所有父任务为 `done` 时，调度器将 `todo` 提升为 `ready`。
- **评论（Comment）** — 代理间协议。代理和人类追加评论；当工作进程被（重新）生成时，它会读取完整的评论线程作为其上下文的一部分。
- **工作空间（Workspace）** — 工作进程操作的目录。三种类型：
  - `scratch`（默认） — 在 `~/.hermes/kanban/workspaces/<id>/`（或在非默认看板上的 `~/.hermes/kanban/boards/<slug>/workspaces/<id>/`）下的新临时目录。
  - `dir:<path>` — 现有的共享目录（Obsidian 仓库、邮件运维目录、按账户的文件夹）。**必须是绝对路径。** 像 `dir:../tenants/foo/` 这样的相对路径在调度时会被拒绝，因为它们会根据调度器碰巧所在的 CWD 解析，这是模糊的，并且是混淆代理的逃逸向量。路径其他方面是受信任的 — 这是你的机器，你的文件系统，工作进程以你的 uid 运行。这是受信任本地用户的威胁模型；看板在设计上是单主机的。
  - `worktree` — `.worktrees/<id>/` 下的 git worktree，用于编码任务。工作进程端的 `git worktree add` 会创建它。
- **调度器（Dispatcher）** — 一个长期运行的循环，每 N 秒（默认 60 秒）：回收过期的认领，回收崩溃的工作进程（PID 不存在但 TTL 尚未过期），提升就绪任务，原子认领，生成被分配的配置文件。默认在网关**内部运行**（`kanban.dispatch_in_gateway: true`）。每个 tick 一个调度器扫描所有看板；工作进程在生成时设置了 `HERMES_KANBAN_BOARD` 固定值，因此它们无法看到其他看板。在同一个任务上连续约 5 次生成失败后，调度器会自动将其阻塞，以最后一次错误为原因 — 防止在配置文件不存在、工作空间无法挂载等情况下反复尝试。
- **租户（Tenant）** — 看板*内部*的可选字符串命名空间。一个专家服务集群可以为多个业务（`--tenant business-a`）提供服务，通过工作空间路径和记忆键前缀实现数据隔离。租户是软过滤；看板是硬隔离边界。

## 多项目看板

看板允许你将不相关的工作流 — 每个项目、仓库或领域一个 — 分离到隔离的队列中。新安装恰好有一个名为 `default` 的看板（数据库位于 `~/.hermes/kanban.db`，用于向后兼容）。只需要一个工作流的用户永远不需要了解看板；该功能是可选的。

每个看板的隔离是绝对的：

- 每个看板有独立的 SQLite 数据库（`~/.hermes/kanban/boards/<slug>/kanban.db`）。
- 独立的 `workspaces/` 和 `logs/` 目录。
- 为任务生成的工作进程只能看到**它们自己看板**的任务 — 调度器在子进程环境中设置 `HERMES_KANBAN_BOARD`，工作进程可访问的每个 `kanban_*` 工具都会读取它。
- 不允许跨看板链接任务（保持 schema 简单；如果确实需要跨项目引用，使用自由文本提及并手动按 id 查找）。

### 从 CLI 管理看板

```bash
# 查看磁盘上有什么。全新安装只显示 "default"。
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

# 更改后续调用中哪个看板是"当前"的。
hermes kanban boards switch atm10-server
hermes kanban boards show             # 当前活跃的是什么？

# 重命名显示名称（slug 不可变 — 它是目录名）。
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# 归档（默认） — 将看板目录移动到 boards/_archived/<slug>-<ts>/。
# 可以通过移回目录来恢复。
hermes kanban boards rm atm10-server

# 硬删除 — 对看板目录执行 `rm -rf`。不可恢复。
hermes kanban boards rm atm10-server --delete
```

看板解析顺序（优先级从高到低）：

1. CLI 调用上的显式 `--board <slug>`。
2. `HERMES_KANBAN_BOARD` 环境变量（调度器在生成工作进程时设置，因此工作进程无法看到其他看板）。
3. `~/.hermes/kanban/current` — 由 `hermes kanban boards switch` 持久化的 slug。
4. `default`。

Slug 会被验证：小写字母数字 + 连字符 + 下划线，1-64 字符，必须以字母数字开头。大写输入会自动转为小写。任何其他内容（斜杠、空格、点、`..`）在 CLI 层被拒绝，以防止路径遍历技巧命名看板。

### 从仪表盘管理看板

`hermes dashboard` → 看板标签页在存在多个看板（或任何看板有任务）时，顶部会显示一个看板切换器。单看板用户只能看到一个小型的 `+ 新看板` 按钮；切换器在需要之前是隐藏的。

- **看板下拉菜单** — 选择活动看板。你的选择会保存到浏览器的 `localStorage`，因此在重新加载时保持不变，不会在你不注意的情况下移动 CLI 的 `current` 指针。
- **+ 新看板** — 打开一个模态框，要求输入 slug、显示名称、描述和图标。可以选择自动切换到新看板。
- **归档** — 仅在非 `default` 看板上显示。确认后，将看板目录移动到 `boards/_archived/`。

所有仪表盘 API 端点接受 `?board=<slug>` 进行看板范围限定。事件 WebSocket 在连接时固定到一个看板；在 UI 中切换会针对新看板打开新的 WS 连接。


## 快速开始

以下命令是**你**（人类）设置看板和创建任务的方式。一旦任务被分配，调度器会将被分配的配置文件生成为工作进程，从那时起**模型通过 `kanban_*` 工具调用来驱动任务，而不是 CLI 命令** — 请参阅[工作进程如何与看板交互](#工作进程如何与看板交互)。

```bash
# 1. 创建看板（你）
hermes kanban init

# 2. 启动网关（托管嵌入式调度器）
hermes gateway start

# 3. 创建任务（你 — 或编排代理通过 kanban_create）
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. 实时查看活动（你）
hermes kanban watch

# 5. 查看看板（你）
hermes kanban list
hermes kanban stats
```

当调度器接收 `t_abcd` 并生成 `researcher` 配置文件时，该工作进程的模型首先要做的就是调用 `kanban_show()` 来读取其任务。它不会运行 `hermes kanban show t_abcd`。

### 网关嵌入式调度器（默认）

调度器在网关进程内运行。无需安装任何东西，无需管理独立服务 — 如果网关在运行，就绪任务会在下一个 tick（默认 60 秒）被拾取。

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # 默认
  dispatch_interval_seconds: 60    # 默认
```

通过 `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0` 在运行时覆盖配置标志用于调试。标准的网关监控机制适用：直接运行 `hermes gateway start`，或将网关配置为 systemd 用户单元（参见网关文档）。没有运行中的网关时，`ready` 任务会保持原状，直到有网关启动 — `hermes kanban create` 在创建时会对此发出警告。

以独立进程运行 `hermes kanban daemon` 已**弃用**；请使用网关。如果你确实无法运行网关（无头主机策略禁止长时间运行的服务等），`--force` 逃生舱可以在一个发布周期内保持旧独立守护进程运行，但同时运行网关嵌入式调度器和独立守护进程对同一个 `kanban.db` 会导致认领竞争，不被支持。

### 幂等创建（用于自动化 / webhook）

```bash
# 第一次调用创建任务。使用相同键的后续调用返回现有任务 id，而不是重复创建。
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### 批量 CLI 动词

所有生命周期动词都接受多个 id，因此你可以一次清理一批：

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

## 工作进程如何与看板交互

**工作进程不会 shell 调用 `hermes kanban`。** 当调度器生成工作进程时，它在子进程环境中设置 `HERMES_KANBAN_TASK=t_abcd`，该环境变量会激活模型 schema 中的一组专用**看板工具集** — 七个直接通过 Python `kanban_db` 层读写看板的工具，与 CLI 使用的方式相同。运行中的工作进程像调用任何其他工具一样调用这些工具；它永远看不到也不需要 `hermes kanban` CLI。

| 工具 | 用途 | 必需参数 |
|---|---|---|
| `kanban_show` | 读取当前任务（标题、正文、先前尝试、父交接、评论、完整的预格式化 `worker_context`）。默认使用环境变量中的任务 id。 | — |
| `kanban_complete` | 使用 `summary` + `metadata` 结构化交接完成任务。 | 至少 `summary` / `result` 之一 |
| `kanban_block` | 使用 `reason` 升级请求人工输入。 | `reason` |
| `kanban_heartbeat` | 在长时间操作期间发送存活信号。纯副作用。 | — |
| `kanban_comment` | 向任务线程追加持久化备注。 | `task_id`、`body` |
| `kanban_create` | （编排器）通过 `assignee`、可选的 `parents`、`skills` 等扇出为子任务。 | `title`、`assignee` |
| `kanban_link` | （编排器）事后添加 `parent_id → child_id` 依赖边。 | `parent_id`、`child_id` |

一个典型的工作进程轮次如下：

```
# 模型的工具调用，按顺序：
kanban_show()                                     # 无参数 — 使用 HERMES_KANBAN_TASK
# （模型读取返回的 worker_context，通过终端/文件工具执行工作）
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# （更多工作）
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

一个**编排器**工作进程则扇出：

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
    parents=["t_r1", "t_r2"],                     # 当两者都完成时提升为 ready
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

三个"（编排器）"工具 — `kanban_create`、`kanban_link` 以及对外部任务的 `kanban_comment` — 对每个工作进程都可用；约定（由 `kanban-orchestrator` 技能强制执行）是工作配置文件不扇出，编排器配置文件不执行。

### 为什么使用工具而不是 shell 调用 `hermes kanban`

三个原因：

1. **后端可移植性。** 终端工具指向远程后端（Docker / Modal / Singularity / SSH）的工作进程会在容器*内部*运行 `hermes kanban complete`，那里没有安装 `hermes`，也没有挂载 `~/.hermes/kanban.db`。看板工具在代理自身的 Python 进程中运行，无论终端后端如何，都能访问 `~/.hermes/kanban.db`。
2. **没有 shell 引用的脆弱性。** 通过 shlex + argparse 传递 `--metadata '{"files": [...]}'` 是一个潜在的陷阱。结构化工具参数完全跳过了这个问题。
3. **更好的错误。** 工具结果是模型可以推理的结构化 JSON，而不是它必须解析的 stderr 字符串。

**对正常会话的零 schema 占用。** 普通 `hermes chat` 会话中零 `kanban_*` 工具。每个工具上的 `check_fn` 仅在设置了 `HERMES_KANBAN_TASK` 时返回 True，这只在调度器生成此进程时才会发生。对于从不接触看板的用户，没有工具膨胀。

`kanban-worker` 和 `kanban-orchestrator` 技能教模型何时调用哪个工具以及以什么顺序。

### 推荐的交接证据

`kanban_complete(summary=..., metadata={...})` 是故意设计为灵活的：summary 是人类可读的结案信息，`metadata` 是机器可读的交接信息，下游代理、审查者或仪表盘可以在不抓取散文的情况下重用。

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

这些键是约定，不是 schema 要求。有用的特性是每个工作进程留下足够的证据，使下一个读者能快速回答四个问题：

1. 改变了什么？
2. 如何验证的？
3. 如果失败了，什么可以解除阻塞或重试？
4. 什么风险被有意地保留？

将密钥、原始日志、令牌、OAuth 材料和无关的转录记录排除在 `metadata` 之外。存储指针和摘要。如果任务没有文件或测试，在 `summary` 中明确说明，并将 `metadata` 用于确实存在的证据，如源 URL、issue id 或手动审查步骤。

### 工作进程技能

任何应该能够处理看板任务的配置文件必须加载 `kanban-worker` 技能。它以**工具调用**（而非 CLI 命令）的形式教工作进程完整的生命周期：

1. 生成时，调用 `kanban_show()` 读取标题 + 正文 + 父交接 + 先前尝试 + 完整评论线程。
2. `cd $HERMES_KANBAN_WORKSPACE`（通过终端工具）并在那里工作。
3. 在长时间操作期间每隔几分钟调用 `kanban_heartbeat(note="...")`。
4. 使用 `kanban_complete(summary="...", metadata={...})` 完成，或使用 `kanban_block(reason="...")` 如果卡住了。

`kanban-worker` 是一个内置技能，在安装和更新时同步到每个配置文件 — 没有单独的技能中心安装步骤。验证你用于看板工作进程的配置文件中是否存在（`researcher`、`writer`、`ops` 等）：

```bash
hermes -p <your-worker-profile> skills list | grep kanban-worker
```

如果缺少内置副本，为该配置文件恢复：

```bash
hermes -p <your-worker-profile> skills reset kanban-worker --restore
```

调度器在生成每个工作进程时还会自动传递 `--skills kanban-worker`，因此即使配置文件的默认技能配置不包含它，工作进程也始终可以使用模式库。

### 为特定任务附加额外技能

有时单个任务需要被分配者配置文件默认不具备的专业上下文 — 一个需要 `translation` 技能的翻译任务，一个需要 `github-code-review` 的审查任务，一个需要 `security-pr-audit` 的安全审计。与其每次编辑被分配者的配置文件，不如将技能直接附加到任务上。

**从编排器代理**（通常情况 — 一个代理将工作路由给另一个代理），使用 `kanban_create` 工具的 `skills` 数组：

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

**从人类（CLI / 斜杠命令）**，为每个技能重复 `--skill`：

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**从仪表盘**，在行内创建表单的**技能**字段中输入逗号分隔的技能名称。

这些技能是内置 `kanban-worker` 的**附加项** — 调度器为每个（以及内置的）发出一个 `--skills <name>` 标志，因此工作进程在生成时加载了所有技能。技能名称必须匹配被分配者配置文件上实际安装的技能（运行 `hermes skills list` 查看可用内容）；没有运行时安装。

### 编排器技能

**行为良好的编排器不会自己执行工作。** 它将用户目标分解为任务，链接它们，将每个任务分配给专家，然后退后。`kanban-orchestrator` 技能将其编码为工具调用模式：反诱惑规则、标准专家名册（`researcher`、`writer`、`analyst`、`backend-eng`、`reviewer`、`ops`）以及以 `kanban_create` / `kanban_link` / `kanban_comment` 为键的分解剧本。

一个典型的编排器轮次（两个并行研究人员交接给作者）：

```
# 来自用户的目标："draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # 当两个研究人员完成时提升为 'ready'
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# 可选：在不重新创建任务的情况下添加后续发现的跨切面依赖
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

`kanban-orchestrator` 是一个内置技能。它在安装和更新时同步到每个配置文件，因此没有单独的技能中心安装步骤。验证你的编排器配置文件中是否存在：

```bash
hermes -p orchestrator skills list | grep kanban-orchestrator
```

如果缺少内置副本，为该配置文件恢复：

```bash
hermes -p orchestrator skills reset kanban-orchestrator --restore
```

为获得最佳效果，将其与工具集限于看板操作（`kanban`、`gateway`、`memory`）的配置文件配对，这样编排器即使尝试也无法执行实现任务。

## 仪表盘（GUI）

`/kanban` CLI 和斜杠命令足以无头运行看板，但可视化看板通常是人工参与的合适界面：分类、跨配置文件监督、阅读评论线程以及在列之间拖拽卡片。Hermes 将此作为 `plugins/kanban/` 处的**内置仪表盘插件**提供 — 不是核心功能，不是独立服务 — 遵循[扩展仪表盘](./extending-the-dashboard)中描述的模式。

打开方式：

```bash
hermes kanban init      # 一次性：创建 kanban.db（如果尚不存在）
hermes dashboard        # "看板" 标签页出现在导航中，在"技能"之后
```

### 插件提供的内容

- 一个**看板**标签页，按状态显示列：`triage`、`todo`、`ready`、`running`、`blocked`、`done`（开启切换时还有 `archived`）。
  - `triage` 是粗略想法的停放列，预期由指定者充实。使用 `hermes kanban create --triage`（或通过 Triage 列的行内创建）创建的任务会落在这里，调度器会将它们保持原样，直到人类或指定者将它们提升到 `todo` / `ready`。运行 `hermes kanban specify <id>` 让辅助 LLM 将 triage 任务扩展为具体规范（标题 + 正文含目标、方法、验收标准）并一次性提升到 `todo`；`--all` 一次性扫描所有 triage 任务。配置哪个模型运行指定者，在 `config.yaml` 的 `auxiliary.triage_specifier` 下。
- 卡片显示任务 id、标题、优先级徽章、租户标签、被分配的配置文件、评论/链接计数、一个**进度药丸**（`N/M` 子任务完成，当任务有依赖时）和"创建于 N 之前"。每张卡片有复选框支持多选。
- **运行中按配置文件分泳道** — 工具栏复选框切换运行中列按被分配者分组。
- **通过 WebSocket 实时更新** — 插件在短轮询间隔内跟踪仅追加的 `task_events` 表；看板在任何配置文件（CLI、网关或另一个仪表盘标签页）操作的瞬间反映变更。重新加载会防抖，因此突发事件只触发一次重新获取。
- **拖放**卡片在列之间更改状态。放置发送 `PATCH /api/plugins/kanban/tasks/:id`，通过与 CLI 相同的 `kanban_db` 代码路由 — 三个交互面永远不会偏离。移动到破坏性状态（`done`、`archived`、`blocked`）时会提示确认。触摸设备使用基于指针的后备方案，因此看板在平板上也可用。
- **行内创建** — 点击任何列标题上的 `+` 以输入标题、被分配者、优先级以及（可选的）从现有任务的下拉列表中选择父任务。从 Triage 列创建会自动将新任务停放在 triage 中。
- **多选批量操作** — shift/ctrl 点击卡片或勾选其复选框将其添加到选择中。顶部出现批量操作栏，包含批量状态转换、归档和重新分配（通过配置文件下拉列表，或"(取消分配)"）。破坏性批量操作先确认。每个 id 的部分失败在不中断其余部分的情况下报告。
- **点击卡片**（不带 shift/ctrl）打开侧边抽屉（Escape 或点击外部关闭），包含：
  - **可编辑标题** — 点击标题重命名。
  - **可编辑被分配者 / 优先级** — 点击元信息行修改。
  - **可编辑描述** — 默认 Markdown 渲染（标题、粗体、斜体、行内代码、围栏代码块、`http(s)` / `mailto:` 链接、无序列表），有一个"编辑"按钮切换到文本域。Markdown 渲染是一个微小的、XSS 安全的渲染器 — 每个替换在 HTML 转义后的输入上运行，只有 `http(s)` / `mailto:` 链接通过，且始终设置 `target="_blank"` + `rel="noopener noreferrer"`。
  - **依赖编辑器** — 父项和子项的芯片列表，每个有 `×` 取消链接，加上下拉列表覆盖每个其他任务以添加新的父项或子项。循环尝试在服务端以清晰消息拒绝。
  - **状态操作行**（→ triage / → ready / → running / block / unblock / complete / archive），带有破坏性转换的确认提示。对于**Triage** 列中的卡片，该行还暴露一个 **✨ 指定** 按钮，调用辅助 LLM（`config.yaml` 中的 `auxiliary.triage_specifier`）将单行文本扩展为具体规范（标题 + 正文含目标、方法、验收标准）并将任务提升到 `todo`。同样的行为可以通过 CLI（`hermes kanban specify <id>` / `--all`）、任何网关平台（`/kanban specify <id>`）以及通过 `POST /api/plugins/kanban/tasks/:id/specify` 编程方式实现。
  - 结果部分（也是 Markdown 渲染的）、带 Enter 提交的评论线程、最近 20 个事件。
- **工具栏过滤器** — 自由文本搜索、租户下拉（默认为 `config.yaml` 中的 `dashboard.kanban.default_tenant`）、被分配者下拉、"显示已归档"切换、"按配置文件分泳道"切换以及一个**推动调度器**按钮，这样你不必等待下一个 60 秒 tick。

视觉目标是最熟悉的 Linear / Fusion 布局：暗色主题、带计数的列标题、彩色状态点、优先级和租户的药丸芯片。插件只读取主题 CSS 变量（`--color-*`、`--radius`、`--font-mono`、...），因此它会根据活动中的任何仪表盘主题自动换肤。

### 架构

GUI 严格是一个**通过 DB 读取 + 通过 kanban_db 写入**层，没有自己的领域逻辑：

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

### REST 接口

所有路由挂载在 `/api/plugins/kanban/` 下，由仪表盘的临时会话令牌保护：

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | 按状态列分组的完整看板，加上租户 + 被分配者用于过滤器下拉 |
| `GET` | `/tasks/:id` | 任务 + 评论 + 事件 + 链接 |
| `POST` | `/tasks` | 创建（封装 `kanban_db.create_task`，接受 `triage: bool` 和 `parents: [id, …]`） |
| `PATCH` | `/tasks/:id` | 状态 / 被分配者 / 优先级 / 标题 / 正文 / 结果 |
| `POST` | `/tasks/bulk` | 对 `ids` 中的每个 id 应用相同的补丁（状态 / 归档 / 被分配者 / 优先级）。每个 id 的失败在不中断兄弟的情况下报告 |
| `POST` | `/tasks/:id/comments` | 追加评论 |
| `POST` | `/tasks/:id/specify` | 运行 triage 指定器 — 辅助 LLM 充实任务正文并将其从 `triage` 提升到 `todo`。返回 `{ok, task_id, reason, new_title}`；`ok=false` 并带有"不在 triage" / 无辅助客户端 / LLM 错误的人类可读原因是 200，不是 4xx |
| `POST` | `/links` | 添加依赖（`parent_id` → `child_id`） |
| `DELETE` | `/links?parent_id=…&child_id=…` | 移除依赖 |
| `POST` | `/dispatch?max=…&dry_run=…` | 推动调度器 — 跳过 60 秒等待 |
| `GET` | `/config` | 从 `config.yaml` 读取 `dashboard.kanban` 偏好设置 — `default_tenant`、`lane_by_profile`、`include_archived_by_default`、`render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events` 行的实时流 |

每个处理程序都是薄包装 — 插件大约 700 行 Python（路由器 + WebSocket 跟踪 + 批量批处理器 + 配置读取器），不添加新的业务逻辑。一个微小的 `_conn()` 辅助函数在每次读写时自动初始化 `kanban.db`，因此无论用户是先打开仪表盘、直接调用 REST API 还是运行 `hermes kanban init`，全新安装都能工作。

### 仪表盘配置

`~/.hermes/config.yaml` 中 `dashboard.kanban` 下的任何键都会更改标签页的默认值 — 插件在加载时通过 `GET /config` 读取它们：

```yaml
dashboard:
  kanban:
    default_tenant: acme              # 预选择租户过滤器
    lane_by_profile: true             # "按配置文件分泳道"切换的默认值
    include_archived_by_default: false
    render_markdown: true             # 设置为 false 使用纯 <pre> 渲染
```

每个键都是可选的，回退到显示的默认值。

### 安全模型

仪表盘的 HTTP 认证中间件[明确跳过 `/api/plugins/`](./extending-the-dashboard#backend-api-routes) — 插件路由在设计上不经过认证，因为仪表盘默认绑定到 localhost。这意味着看板 REST 接口可以从主机上的任何进程访问。

WebSocket 额外要求仪表盘的临时会话令牌作为 `?token=…` 查询参数（浏览器无法在升级请求上设置 `Authorization`），与浏览器内 PTY 桥使用的模式匹配。

如果你运行 `hermes dashboard --host 0.0.0.0`，每个插件路由（包括看板）都变得可从网络访问。**不要在共享主机上这样做。** 看板包含任务正文、评论和工作空间路径；攻击者访问这些路由可以获得你整个协作面的读取访问权限，还可以创建 / 重新分配 / 归档任务。

`~/.hermes/kanban.db` 中的任务是有意设计为配置文件无关的（这是协调原语）。如果你使用 `hermes -p <profile> dashboard` 打开仪表盘，看板仍然显示主机上任何其他配置文件创建的任务。同一用户拥有所有配置文件，但如果多个角色共存，这一点值得了解。

### 实时更新

`task_events` 是一个仅追加的 SQLite 表，具有单调递增的 `id`。WebSocket 端点保存每个客户端最后看到的事件 id 并在新行到达时推送。当突发事件到达时，前端重新加载（非常廉价的）看板端点 — 这比尝试从每种事件类型修补本地状态更简单也更正确。WAL 模式意味着读取循环永远不会阻塞调度器的 `BEGIN IMMEDIATE` 认领事务。

### 扩展它

插件使用标准的 Hermes 仪表盘插件契约 — 请参阅[扩展仪表盘](./extending-the-dashboard)获取完整的清单参考、shell 插槽、页面范围插槽和插件 SDK。额外列、自定义卡片装饰、租户过滤布局或完整的 `tab.override` 替换都可以在不 fork 此插件的情况下表达。

要禁用而不删除：在 `config.yaml` 中添加 `dashboard.plugins.kanban.enabled: false`（或删除 `plugins/kanban/dashboard/manifest.json`）。

### 范围边界

GUI 被故意设计得很薄。插件做的每件事都可以从 CLI 访问；插件只是让人类使用更舒适。自动分配、预算、治理门和组织架构图视图仍然是用户空间 — 路由器配置文件、另一个插件或 `tools/approval.py` 的重用 — 正如设计规范的范围外部分所列出的那样。

## CLI 命令参考

这是**你**（或脚本、cron、仪表盘）用来驱动看板的接口。在调度器内运行的工作进程使用 `kanban_*` [工具接口](#工作进程如何与看板交互) 执行相同操作 — 这里的 CLI 和那里的工具都通过 `kanban_db` 路由，因此两个接口在构造上是一致的。

```
hermes kanban init                                     # 创建 kanban.db + 打印守护进程提示
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|dir:<path>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived] [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # 或 'none' 取消分配
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# 批量动词 — 接受多个 id：
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban tail <id>                                # 跟踪单个任务的事件流
hermes kanban watch [--assignee P] [--tenant T]        # 实时流式传输所有事件到终端
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # 工作进程长时间操作的存活信号
hermes kanban runs <id> [--json]                       # 尝试历史（每次运行一行）
hermes kanban assignees [--json]                       # 磁盘上的配置文件 + 每个被分配者的任务计数
hermes kanban dispatch [--dry-run] [--max N]           # 单次通过
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # 已弃用 — 独立调度器（使用 `hermes gateway start` 代替）
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # 按状态 + 按被分配者的计数
hermes kanban log <id> [--tail BYTES]                  # 来自 ~/.hermes/kanban/logs/ 的工作进程日志
hermes kanban notify-subscribe <id>                    # 网关桥接钩子（网关中的 /kanban 使用）
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # 工作进程看到的内容
hermes kanban specify [<id> | --all] [--tenant T]      # 将 triage 列中的想法充实
        [--author NAME] [--json]                       #   为完整规范并提升到 todo
hermes kanban gc [--event-retention-days N]            # 工作空间 + 旧事件 + 旧日志
        [--log-retention-days N]
```

所有命令也可以作为交互式 CLI 和消息网关中的斜杠命令使用（请参阅下方的[`/kanban` 斜杠命令](#kanban-斜杠命令)）。

## `/kanban` 斜杠命令 {#kanban-斜杠命令}

每个 `hermes kanban <action>` 动词也可以作为 `/kanban <action>` 使用 — 从交互式 `hermes chat` 会话**内部**以及从任何网关平台（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、email、SMS）。两个接口都调用完全相同的 `hermes_cli.kanban.run_slash()` 入口点，该入口点重用 `hermes kanban` argparse 树，因此参数接口、标志和输出格式在 CLI、`/kanban` 和 `hermes kanban` 之间是相同的。你不必离开聊天就能驱动看板。

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # 将 triage 单行文本充实为真正的规范
/kanban specify --all --tenant engineering  # 一次性扫描一个租户的所有 triage 任务
```

引用多词参数的方式与 shell 相同 — `run_slash` 使用 `shlex.split` 解析行的其余部分，因此 `"..."` 和 `'...'` 都有效。

### 运行中使用：`/kanban` 绕过运行中代理守卫

网关通常在代理仍在思考时将斜杠命令和用户消息排队 — 这就是阻止你在第一个还在进行中时不小心开始第二个轮次的原因。**`/kanban` 被明确豁免于此守卫。** 看板存在于 `~/.hermes/kanban.db` 中，而不是运行中代理的状态中，因此读取（`list`、`show`、`context`、`tail`、`watch`、`stats`、`runs`）和写入（`comment`、`unblock`、`block`、`assign`、`archive`、`create`、`link`、…）都会立即通过，即使在轮次中间。

这就是分离的全部意义：

- 工作进程阻塞等待对等方 → 你从手机发送 `/kanban unblock t_abcd`，调度器在其下一个 tick 拾取对等方。被阻塞的工作进程不会被中断 — 它只是不再被阻塞。
- 你发现一张需要人工上下文的卡片 → `/kanban comment t_xyz "use the 2026 schema, not 2025"` 落在任务线程上，该任务的*下一次*运行将在 `kanban_show()` 中读取它。
- 你想知道你的集群在做什么而不停止编排器 → `/kanban list --mine` 或 `/kanban stats` 在不触及你主对话的情况下检查看板。

### `/kanban create` 时的自动订阅（仅限网关）

当你从网关使用 `/kanban create "…"` 创建任务时，源聊天（平台 + 聊天 id + 线程 id）会自动订阅该任务的终端事件（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）。你将在每个终端事件收到一条消息 — 包括 `completed` 时工作进程结果摘要的第一行 — 而无需轮询或记住任务 id。

```
你> /kanban create "transcribe today's podcast" --assignee transcriber
bot> 已创建 t_9fc1a3  (ready, assignee=transcriber)
     (已订阅 — 当 t_9fc1a3 完成或阻塞时你将收到通知)

… ~8 分钟后 …

bot> ✓ t_9fc1a3 已由 transcriber 完成
     转录了 42 分钟，保存到 podcast/2026-05-04.md
```

订阅在任务到达 `done` 或 `archived` 时自动移除。如果你使用 `--json`（机器输出）脚本化创建，自动订阅会被跳过 — 假设脚本调用者想通过 `/kanban notify-subscribe` 显式管理订阅。

### 消息中的输出截断

网关平台有实际的消息长度上限。如果 `/kanban list`、`/kanban show` 或 `/kanban tail` 产生超过约 3800 个字符的输出，响应会被截断并带有 `… (truncated; use \`hermes kanban …\` in your terminal for full output)` 页脚。CLI 接口没有这样的上限。

### 自动补全

在交互式 CLI 中，输入 `/kanban ` 并按 Tab 会在内置子命令列表中循环（`list`、`ls`、`show`、`create`、`assign`、`link`、`unlink`、`claim`、`comment`、`complete`、`block`、`unblock`、`archive`、`tail`、`dispatch`、`context`、`init`、`gc`）。上面 CLI 参考中列出的其余动词（`watch`、`stats`、`runs`、`log`、`assignees`、`heartbeat`、`notify-subscribe`、`notify-list`、`notify-unsubscribe`、`daemon`）也可以工作 — 只是还不在自动补全提示列表中。

## 协作模式

看板支持这八种模式而无需任何新原语：

| 模式 | 形态 | 示例 |
|---|---|---|
| **P1 扇出** | N 个兄弟，相同角色 | "并行研究 5 个角度" |
| **P2 流水线** | 角色链：侦察 → 编辑 → 作者 | 每日简报组装 |
| **P3 投票 / 仲裁** | N 个兄弟 + 1 个聚合者 | 3 个研究人员 → 1 个审查者选择 |
| **P4 长期日志** | 相同配置文件 + 共享目录 + cron | Obsidian 仓库 |
| **P5 人工参与** | 工作进程阻塞 → 用户评论 → 解除阻塞 | 模糊决策 |
| **P6 `@mention`** | 从散文中内联路由 | `@reviewer 看看这个` |
| **P7 线程范围工作空间** | 线程中的 `/kanban here` | 按项目的网关线程 |
| **P8 批量处理** | 一个配置文件，N 个主题 | 50 个社交账户 |
| **P9 Triage 指定器** | 粗略想法 → `triage` → `hermes kanban specify` 扩展正文 → `todo` | "把这单行文本变成有规范的任务" |

每种模式的实际示例，请参见 `docs/hermes-kanban-v1-spec.pdf`。

## 多租户使用

当一个专家服务集群为多个业务提供服务时，为每个任务标记一个租户：

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

工作进程接收 `$HERMES_TENANT` 并通过前缀命名空间化其记忆写入。看板、调度器和配置文件定义都是共享的；只有数据是范围的。

## 网关通知

当你从网关（Telegram、Discord、Slack 等）运行 `/kanban create …` 时，源聊天会自动订阅新任务。网关的后台通知器每几秒轮询 `task_events` 并向该聊天传递每个终端事件（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）的一条消息。完成的任务还会发送工作进程 `--result` 的第一行，这样你无需 `/kanban show` 就能看到结果。

你可以从 CLI 显式管理订阅 — 当脚本 / cron 任务想通知一个它不是源头的聊天时很有用：

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

订阅在任务到达 `done` 或 `archived` 时自动移除；无需清理。

## 运行记录 — 每次尝试一行

任务是一个逻辑工作单元；**运行**是执行它的一次尝试。当调度器认领一个就绪任务时，它创建一个 `task_runs` 行并将 `tasks.current_run_id` 指向它。当该尝试结束时 — 完成、阻塞、崩溃、超时、生成失败、回收 — 运行行以 `outcome` 关闭，任务的指针清除。一个被尝试了三次的任务有三个 `task_runs` 行。

为什么使用两个表而不是只修改任务：你需要**完整的尝试历史**用于实际的复盘（"第二次审查尝试得到了批准，第三次合并了"），你还需要一个干净的地方来放置每次尝试的元数据 — 哪些文件改变了，哪些测试运行了，审查者注意到了哪些发现。这些是运行事实，不是任务事实。

运行也是**结构化交接**的所在。当工作进程完成任务（通过 `kanban_complete(...)`）时，它可以传递：

- `summary`（工具参数）/ `--summary`（CLI） — 人类交接；放在运行记录上；下游子任务在它们的 `build_worker_context` 中看到它。
- `metadata`（工具参数）/ `--metadata`（CLI） — 运行记录上的自由格式 JSON 字典；子任务在摘要旁边看到它序列化后的内容。
- `result`（工具参数）/ `--result`（CLI） — 放在任务行上的短日志行（旧字段，保留用于向后兼容）。

下游子任务读取每个父任务的最近完成运行的摘要 + 元数据。重试的工作进程读取自己任务的先前尝试（outcome、summary、error），以便它们不会重复已经失败的路径。

```
# 工作进程实际做的事情 — 来自代理循环的一个工具调用：
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

当你（人类）需要结束一个工作进程无法完成的任务时 — 例如一个被放弃的任务，或你从仪表盘手动标记为完成的 — 从 CLI 可以达到同样的交接：

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

运行记录在仪表盘上暴露（抽屉中的运行历史部分，每次尝试一行彩色记录）和在 REST API 上（`GET /api/plugins/kanban/tasks/:id` 返回一个 `runs[]` 数组）。`PATCH /api/plugins/kanban/tasks/:id` 配合 `{status: "done", summary, metadata}` 会将两者转发到内核，因此仪表盘的"标记完成"按钮与 CLI 等效。`task_events` 行带有它们所属的 `run_id`，以便 UI 可以按尝试分组，并且 `completed` 事件在其负载中嵌入第一行摘要（上限 400 字符），因此网关通知器可以在没有第二次 SQL 往返的情况下渲染结构化交接。

**批量完成的注意事项。** `hermes kanban complete a b c --summary X` 会被拒绝 — 结构化交接是按运行记录的，因此将相同摘要复制粘贴到 N 个任务几乎总是错误的。不使用 `--summary` / `--metadata` 的批量完成对于常见的"我完成了一堆管理任务"场景仍然有效。

**从状态变更回收的运行。** 如果你在仪表盘中将运行中的任务从 `running` 拖出（回到 `ready`，或直接到 `todo`），或归档一个仍在运行的任务，进行中的运行以 `outcome='reclaimed'` 关闭而不是被孤立。当 `tasks.current_run_id` 为 `NULL` 时，`task_runs` 行始终处于终态，反之亦然 — 这个不变量在 CLI、仪表盘、调度器和通知器之间保持。

**从未认领完成的合成运行。** 完成或阻塞一个从未被认领的任务（例如人类从仪表盘使用摘要关闭一个 `ready` 任务，或 CLI 用户运行 `hermes kanban complete <ready-task> --summary X`）否则会丢失交接。内核改为插入一个零持续时间运行行（`started_at == ended_at`）携带摘要 / 元数据 / 原因，因此尝试历史保持完整。`completed` / `blocked` 事件的 `run_id` 指向该行。

**实时抽屉刷新。** 当仪表盘的 WebSocket 事件流报告用户当前查看的任务的新事件时，抽屉会重新加载自身（通过一个任务事件计数器线程到其 `useEffect` 依赖列表中）。不再需要关闭并重新打开来查看运行的新行或更新的结果。

### 前向兼容性

`tasks` 上的两个可空列保留用于 v2 工作流路由：`workflow_template_id`（此任务属于哪个模板）和 `current_step_key`（该模板中哪个步骤是活动的）。v1 内核在路由时忽略它们但允许客户端写入它们，因此 v2 发布可以在不需要另一个 schema 迁移的情况下添加路由机制。

## 事件参考

每次转换都会向 `task_events` 追加一行。每行携带一个可选的 `run_id`，以便 UI 可以按尝试分组事件。类型分为三个簇，便于过滤（`hermes kanban watch --kinds completed,gave_up,timed_out`）：

**生命周期**（关于任务作为逻辑单元发生了什么变化）：

| 类型 | 负载 | 何时 |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | 任务插入。`run_id` 为 `NULL`。 |
| `promoted` | — | `todo → ready`，因为所有父任务到达 `done`。`run_id` 为 `NULL`。 |
| `claimed` | `{lock, expires, run_id}` | 调度器原子认领一个 `ready` 任务以生成。 |
| `completed` | `{result_len, summary?}` | 工作进程写入了 `--result` / `--summary` 且任务到达 `done`。`summary` 是第一行交接（400 字符上限）；完整版本在运行行上。如果在从未认领的任务上带有交接字段调用 `complete_task`，会合成一个零持续时间运行，以便 `run_id` 仍然指向某个东西。 |
| `blocked` | `{reason}` | 工作进程或人类将任务翻转到 `blocked`。在从未认领的任务上带有 `--reason` 调用时合成零持续时间运行。 |
| `unblocked` | — | `blocked → ready`，手动或通过 `/unblock`。`run_id` 为 `NULL`。 |
| `archived` | — | 从默认看板隐藏。如果任务仍在运行，携带被回收的运行的 `run_id` 作为副作用。 |

**编辑**（非转换的人类驱动变更）：

| 类型 | 负载 | 何时 |
|---|---|---|
| `assigned` | `{assignee}` | 被分配者改变（包括取消分配）。 |
| `edited` | `{fields}` | 标题或正文更新。 |
| `reprioritized` | `{priority}` | 优先级改变。 |
| `status` | `{status}` | 仪表盘拖放直接写入状态（例如 `todo → ready`）。携带从 `running` 拖出时被回收的运行的 `run_id`；否则 `run_id` 为 NULL。 |

**工作进程遥测**（关于执行进程，不是逻辑任务）：

| 类型 | 负载 | 何时 |
|---|---|---|
| `spawned` | `{pid}` | 调度器成功启动了工作进程。 |
| `heartbeat` | `{note?}` | 工作进程调用 `hermes kanban heartbeat $TASK` 在长时间操作期间发送存活信号。 |
| `reclaimed` | `{stale_lock}` | 认领 TTL 到期但没有完成；任务回到 `ready`。 |
| `crashed` | `{pid, claimer}` | 工作进程 PID 不再存活但 TTL 尚未到期。 |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds` 超出；调度器 SIGTERM（然后在 5 秒宽限期后 SIGKILL）并重新排队。 |
| `spawn_failed` | `{error, failures}` | 一次生成尝试失败（缺少 PATH、工作空间无法挂载、…）。计数器递增；任务回到 `ready` 以重试。 |
| `gave_up` | `{failures, error}` | 在 N 次连续 `spawn_failed` 后触发断路器。任务以最后一个错误自动阻塞。默认 N = 5；通过 `--failure-limit` 覆盖。 |

`hermes kanban tail <id>` 显示单个任务的这些事件。`hermes kanban watch` 以看板为单位流式传输它们。

## 超出范围

看板被有意设计为单主机的。`~/.hermes/kanban.db` 是一个本地 SQLite 文件，调度器在同一台机器上生成工作进程。在两个主机上运行共享看板不受支持 — 没有"主机 A 上的工作进程 X，主机 B 上的工作进程 Y"的协调原语，并且崩溃检测路径假设 PID 是主机本地的。如果你需要多主机，请每台主机运行一个独立看板并使用 `delegate_task` / 消息队列来桥接它们。

## 设计规范

完整的设计 — 架构、并发正确性、与其他系统的比较、实现计划、风险、开放问题 — 存在于 `docs/hermes-kanban-v1-spec.pdf` 中。在提交任何行为变更 PR 之前请先阅读该文件。
