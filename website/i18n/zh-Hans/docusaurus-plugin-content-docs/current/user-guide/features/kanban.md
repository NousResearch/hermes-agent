---
sidebar_position: 12
title: "Kanban（多智能体看板）"
description: "用于协调多个 Hermes 配置文件的持久化 SQLite 支持的任务看板"
---

# Kanban —— 多配置文件协作

> **想要一个演练？** 阅读 [Kanban 教程](./kanban-tutorial) —— 四个用户故事（独立开发者、集群运维、带重试的角色流水线、断路器），每个都附有仪表板截图。本页是参考文档；教程是叙述性的。

Hermes Kanban 是一个持久的任务看板，在所有您的 Hermes 配置文件间共享，让多个命名智能体协作工作，而无需脆弱的进程内子智能体群。每个任务都是 `~/.hermes/kanban.db` 中的一行；每次交接都是任何人都可以读写的一行；每个工作者都是一个具有自己身份的完整操作系统进程。

### 两个界面：模型通过工具对话，您通过 CLI 对话

看板有两个前门，都由同一个 `~/.hermes/kanban.db` 支持：

- **智能体通过专用的 `kanban_*` 工具集驱动看板** —— `kanban_show`、`kanban_list`、`kanban_complete`、`kanban_block`、`kanban_heartbeat`、`kanban_comment`、`kanban_create`、`kanban_link`、`kanban_unblock`。调度器在生成每个工作者时，这些工具已经存在于其模式中；编排器配置文件也可以显式启用 `kanban` 工具集。模型通过直接调用工具来读取和路由任务，*而不是*通过 shell 执行 `hermes kanban`。请参阅下面的[工作者如何与看板交互](#how-workers-interact-with-the-board)。
- **您（以及脚本、cron）通过 CLI 上的 `hermes kanban …`、斜杠命令 `/kanban …` 或仪表板驱动看板。** 这些面向人类和自动化 —— 没有工具调用模型在背后操作的地方。

两个界面都通过相同的 `kanban_db` 层路由，因此读取看到一致的视图，写入不会漂移。本页的其余部分显示 CLI 示例，因为它们易于复制粘贴，但每个 CLI 动词都有一个模型使用的工具调用等效项。

这是涵盖 `delegate_task` 无法处理的工作负载的形态：

- **研究分类** —— 并行研究者 + 分析师 + 撰写者，人机协同。
- **计划运维** —— 在数周内构建日志的每日定期简报。
- **数字孪生** —— 随时间积累记忆的持久命名助手（`inbox-triage`、`ops-review`）。
- **工程流水线** —— 分解 → 在并行工作树中实现 → 审查 → 迭代 → PR。
- **集群工作** —— 一个专家管理 N 个主题（50 个社交账户、12 个监控服务）。

有关完整的设计原理、与 Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise 的比较分析，以及八种经典协作模式，请参阅仓库中的 `docs/hermes-kanban-v1-spec.pdf`。

## Kanban 与 `delegate_task`

它们看起来相似；它们不是同一个原语。

| | `delegate_task` | Kanban |
|---|---|---|
| 形态 | RPC 调用（fork → join） | 持久消息队列 + 状态机 |
| 父级 | 阻塞直到子级返回 | `create` 后即发即弃 |
| 子级身份 | 匿名子智能体 | 具有持久记忆的命名配置文件 |
| 可恢复性 | 无 —— 失败 = 失败 | 阻塞 → 解阻塞 → 重新运行；崩溃 → 回收 |
| 人机协同 | 不支持 | 随时评论 / 解阻塞 |
| 每任务智能体 | 一次调用 = 一个子智能体 | 任务生命周期内的 N 个智能体（重试、审查、跟进） |
| 审计追踪 | 上下文压缩时丢失 | SQLite 中持久的行，永久保留 |
| 协调 | 层级（调用者 → 被调用者） | 对等 —— 任何配置文件都可以读写任何任务 |

**一句话区别：** `delegate_task` 是一个函数调用；Kanban 是一个工作队列，其中每次交接都是任何配置文件（或人类）都可以查看和编辑的一行。

**在以下情况使用 `delegate_task`：** 父智能体需要在继续之前获得简短的推理答案，没有人类参与，结果返回到父级的上下文中。

**在以下情况使用 Kanban：** 工作跨越智能体边界，需要 survive 重启，可能需要人类输入，可能由不同角色接管，或需要在事后可发现。

它们共存：看板工作者可能在其运行期间内部调用 `delegate_task`。

## 核心概念

- **看板** —— 一个独立的任务队列，具有自己的 SQLite DB、工作区目录和调度器循环。单个安装可以有多个看板（例如每个项目、仓库或域一个）；请参阅下面的[看板（多项目）](#boards-multi-project)。单项目用户停留在 `default` 看板上，在此文档部分之外永远看不到"看板"这个词。
- **任务** —— 一行，包含标题、可选正文、一个被分配者（配置文件名）、状态（`triage | todo | ready | running | blocked | done | archived`）、可选租户命名空间、可选幂等键（用于重试自动化的去重）。
- **链接** —— `task_links` 行，记录父 → 子依赖关系。当所有父级都 `done` 时，调度器将 `todo → ready` 提升。
- **评论** —— 智能体间协议。智能体和人类追加评论；当工作者（重新）生成时，它会读取完整的评论线程作为其上下文的一部分。
- **工作区** —— 工作者操作的目录。三种类型：
  - `scratch`（默认）—— `~/.hermes/kanban/workspaces/<id>/` 下的全新临时目录（或在非默认看板上的 `~/.hermes/kanban/boards/<slug>/workspaces/<id>/`）。
  - `dir:<path>` —— 现有的共享目录（Obsidian vault、邮件运维目录、每账户文件夹）。**必须是绝对路径。** 像 `dir:../tenants/foo/` 这样的相对路径在调度时被拒绝，因为它们会针对调度器碰巧所在的任何 CWD 解析，这是模糊的，也是混乱的副手逃逸向量。该路径在其他方面是受信任的 —— 这是您的机器，您的文件系统，工作者以您的 uid 运行。这是受信任的本地用户威胁模型；kanban 按设计是单主机的。
  - `worktree` —— 编码任务在 `.worktrees/<id>/` 下的 git 工作树。工作者端的 `git worktree add` 创建它。
- **调度器** —— 一个长生命周期的循环，每 N 秒（默认 60）：回收过期的声明、回收崩溃的工作者（PID 已消失但 TTL 尚未过期）、提升就绪任务、原子声明、生成被分配的配置文件。**默认在网关内运行**（`kanban.dispatch_in_gateway: true`）。一个调度器每次扫描所有看板；工作者以固定的 `HERMES_KANBAN_BOARD` 生成，因此它们看不到其他看板。在同一任务上连续 `kanban.failure_limit` 次生成失败（默认：2）后，调度器自动将其阻塞，并将最后一个错误作为原因 —— 防止对配置文件不存在、工作区无法挂载等的任务进行抖动。
- **租户** —— 看板内可选的字符串命名空间。一个专家集群可以为多个企业（`--tenant business-a`）提供服务，通过工作区路径和内存键前缀进行数据隔离。租户是软过滤器；看板是硬隔离边界。

## 看板（多项目） {#boards-multi-project}

看板让您将不相关的工作流 —— 每个项目、仓库或域一个 —— 分离到隔离的队列中。新安装恰好有一个名为 `default` 的看板（为了向后兼容，DB 位于 `~/.hermes/kanban.db`）。只想要一个工作流的用户永远不需要了解看板；该功能是可选的。

每看板隔离是绝对的：

- 每看板独立的 SQLite DB（`~/.hermes/kanban/boards/<slug>/kanban.db`）。
- 独立的 `workspaces/` 和 `logs/` 目录。
- 为任务生成的工作者**只能**看到其看板的任务 —— 调度器在子环境中设置 `HERMES_KANBAN_BOARD`，工作者可以访问的每个 `kanban_*` 工具都会读取它。
- 不允许跨看板链接任务（保持模式简单；如果您确实需要跨项目引用，请使用自由文本提及并手动按 id 查找）。

### 从 CLI 管理看板

```bash
# 查看磁盘上的内容。新安装只显示 "default"。
hermes kanban boards list

# 创建新看板。
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # 可选：使其成为活动看板

# 在不切换的情况下操作特定看板。
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# 更改哪个看板是后续调用的"当前"看板。
hermes kanban boards switch atm10-server
hermes kanban boards show             # 当前哪个是活动的？

# 重命名显示名称（slug 是不可变的 —— 它是目录名）。
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# 归档（默认）—— 将看板的目录移动到 boards/_archived/<slug>-<ts>/。
# 通过移回目录可恢复。
hermes kanban boards rm atm10-server

# 硬删除 —— `rm -rf` 看板目录。不可恢复。
hermes kanban boards rm atm10-server --delete
```

看板解析顺序（优先级从高到低）：

1. CLI 调用上的显式 `--board <slug>`。
2. `HERMES_KANBAN_BOARD` 环境变量（调度器生成工作者时设置，因此工作者看不到其他看板）。
3. `~/.hermes/kanban/current` —— 由 `hermes kanban boards switch` 持久化的 slug。
4. `default`。

Slugs 经过验证：小写字母数字 + 连字符 + 下划线，1-64 个字符，必须以字母数字开头。大写输入自动转小写。其他字符（斜杠、空格、点、`..`）在 CLI 层被拒绝，因此路径遍历技巧无法命名看板。

### 从仪表板管理看板

`hermes dashboard` → Kanban 标签页在顶部显示一个看板切换器，一旦存在多个看板（或任何看板有任务）。单看板用户只看到一个小的 `+ New board` 按钮；切换器在需要之前隐藏。

- **看板下拉菜单** —— 选择活动看板。您的选择保存到浏览器的 `localStorage`，因此它在重新加载之间持久保存，而不会将 CLI 的 `current` 指针从您保持打开的终端下面移开。
- **+ New board** —— 打开一个模态框，要求输入 slug、显示名称、描述和图标。可选自动切换到新看板。
- **Archive** —— 仅在非 `default` 看板上显示。确认后，将看板目录移动到 `boards/_archived/`。

所有仪表板 API 端点都接受 `?board=<slug>` 进行看板范围限定。事件 WebSocket 在连接时固定到一个看板；在 UI 中切换会针对新看板打开一个新的 WS。


## 快速开始

下面的命令是**您**（人类）设置看板和创建任务。一旦任务被分配，调度器就会将分配的配置文件生成为工作者，从那时起**模型通过 `kanban_*` 工具调用驱动任务，而不是 CLI 命令** —— 请参阅[工作者如何与看板交互](#how-workers-interact-with-the-board)。

```bash
# 1. 创建看板（您）
hermes kanban init

# 2. 启动网关（托管嵌入式调度器）
hermes gateway start

# 3. 创建任务（您 —— 或通过 kanban_create 的编排器智能体）
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. 实时观看活动（您）
hermes kanban watch

# 5. 查看看板（您）
hermes kanban list
hermes kanban stats
```

当调度器拾取 `t_abcd` 并生成 `researcher` 配置文件时，该工作者的模型做的第一件事就是调用 `kanban_show()` 来读取其任务。它不会运行 `hermes kanban show t_abcd`。

### 网关嵌入式调度器（默认）

调度器在网关进程内运行。无需安装，无需管理单独的服务 —— 如果网关已启动，就绪任务会在下一次扫描时被拾取（默认 60 秒）。

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # 默认
  dispatch_interval_seconds: 60    # 默认
```

通过 `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0` 在运行时覆盖配置标志以进行调试。标准网关监管适用：直接运行 `hermes gateway start`，或将网关作为 systemd 用户单元连接（请参阅网关文档）。没有正在运行的网关，`ready` 任务会保持在原地，直到有一个启动 —— `hermes kanban create` 在创建时会对此发出警告。

将 `hermes kanban daemon` 作为单独进程运行已**弃用**；请使用网关。如果您确实无法运行网关（无头主机策略禁止长期运行的服务等），`--force` 逃生舱口会在一个发布周期内保留旧的独立守护进程，但同时针对同一个 `kanban.db` 运行网关嵌入式调度器和独立守护进程会导致声明竞争，不受支持。

### 幂等创建（用于自动化 / Webhook）

```bash
# 第一次调用创建任务。任何后续使用相同键的调用
# 返回现有任务 id 而不是重复创建。
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### 批量 CLI 动词

所有生命周期动词都接受多个 id，因此您可以在一个命令中批量清理：

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

## 工作者如何与看板交互 {#how-workers-interact-with-the-board}

**工作者不会通过 shell 执行 `hermes kanban`。** 当调度器生成工作者时，它在子环境中设置 `HERMES_KANBAN_TASK=t_abcd`，该环境变量在模型的模式中开启一个专用的 **kanban 工具集**。当编排器配置文件在其工具集配置中启用 `kanban` 时，也可以使用相同的工具集。这些工具通过 Python `kanban_db` 层直接读取和修改看板，与 CLI 相同。运行中的工作者像调用任何其他工具一样调用这些工具；它永远看不到或不需要 `hermes kanban` CLI。

| 工具 | 目的 | 必需参数 |
|---|---|---|
| `kanban_show` | 读取当前任务（标题、正文、先前尝试、父级交接、评论、完整预格式化的 `worker_context`）。默认为环境中的任务 id。 | — |
| `kanban_list` | 列出任务摘要，可按 `assignee`、`status`、`tenant`、归档可见性和限制过滤。供编排器发现看板工作。 | — |
| `kanban_complete` | 以 `summary` + `metadata` 结构化交接完成。 | `summary` / `result` 中至少一个 |
| `kanban_block` | 以 `reason` 上报等待人工输入。 | `reason` |
| `kanban_heartbeat` | 在长操作期间发出存活信号。纯副作用。 | — |
| `kanban_comment` | 向任务线程追加持久注释。 | `task_id`、`body` |
| `kanban_create` | （编排器）以 `assignee`、可选 `parents`、`skills` 等扇出到子任务。 | `title`、`assignee` |
| `kanban_link` | （编排器）事后添加 `parent_id → child_id` 依赖边。 | `parent_id`、`child_id` |
| `kanban_unblock` | （编排器）将阻塞的任务移回 `ready`。 | `task_id` |

一个典型的工作者轮次如下：

```
# 模型的工具调用，按顺序：
kanban_show()                                     # 无参数 —— 使用 HERMES_KANBAN_TASK
# （模型读取返回的 worker_context，通过 terminal/file 工具完成工作）
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# （更多工作）
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

一个**编排器**工作者改为扇出：

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
    parents=["t_r1", "t_r2"],                     # 当两者都完成时提升到 ready
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

"（编排器）"工具 —— `kanban_list`、`kanban_create`、`kanban_link`、`kanban_unblock` 以及对外部任务的 `kanban_comment` —— 可通过相同的工具集使用；约定（由 `kanban-orchestrator` 技能强制执行）是工作者配置文件不会扇出或路由不相关的工作，而编排器配置文件不会执行实现工作。调度器生成的工作者对于破坏性生命周期操作仍然是任务范围的，不能修改不相关的任务。

### 为什么使用工具而不是 shell 执行 `hermes kanban`

三个原因：

1. **后端可移植性。** 终端工具指向远程后端（Docker / Modal / Singularity / SSH）的工作者会在容器*内部*运行 `hermes kanban complete`，那里没有安装 `hermes`，`~/.hermes/kanban.db` 也没有挂载。kanban 工具在智能体自己的 Python 进程中运行，无论终端后端如何，始终能到达 `~/.hermes/kanban.db`。
2. **无 shell 引用脆弱性。** 通过 shlex + argparse 传递 `--metadata '{"files": [...]}'` 是一个潜在的隐患。结构化工具参数完全跳过它。
3. **更好的错误。** 工具结果是模型可以推理的结构化 JSON，而不是它必须解析的 stderr 字符串。

**正常会话的零模式占用。** 常规的 `hermes chat` 会话在其模式中零个 `kanban_*` 工具。每个工具上的 `check_fn` 仅在设置 `HERMES_KANBAN_TASK` 时才返回 True，这仅在调度器生成此进程时发生。对于从不接触 kanban 的用户，没有工具膨胀。

`kanban-worker` 和 `kanban-orchestrator` 技能教模型在何时以何种顺序调用哪个工具。

### 推荐的交接证据

`kanban_complete(summary=..., metadata={...})` 有意保持灵活：
summary 是人类可读的结案陈词，`metadata` 是机器可读的交接，下游智能体、审查者或仪表板可以重用，而无需从散文中抓取。

对于工程和审查任务，首选这种可选的 metadata 形状：

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

这些键是约定，不是模式要求。有用的属性是
每个工作者留下足够的证据，让下一个读者快速回答四个
问题：

1. 改变了什么？
2. 如何验证的？
3. 如果失败，什么可以解阻塞或重试？
4. 还有什么风险故意保留？

将秘密、原始日志、令牌、OAuth 材料和不相关的记录排除在
`metadata` 之外。改为存储指针和摘要。如果任务没有文件或
测试，在 `summary` 中明确说明，并为确实存在的证据使用 `metadata`，
例如源 URL、问题 id 或手动审查步骤。

### 工作者技能

任何应该能够处理 kanban 任务的配置文件都必须加载 `kanban-worker` 技能。它教工作者在**工具调用**中的完整生命周期，而不是 CLI 命令：

1. 生成时，调用 `kanban_show()` 读取标题 + 正文 + 父级交接 + 先前尝试 + 完整评论线程。
2. 通过终端工具 `cd $HERMES_KANBAN_WORKSPACE` 并在那里完成工作。
3. 在长操作期间每隔几分钟调用 `kanban_heartbeat(note="...")`。
4. 以 `kanban_complete(summary="...", metadata={...})` 完成，或如果卡住则以 `kanban_block(reason="...")` 阻塞。

`kanban-worker` 是一个捆绑技能，在安装和更新期间同步到每个配置文件 —— 没有单独的 Skills Hub 安装步骤。验证它存在于您用于 kanban 工作者（`researcher`、`writer`、`ops` 等）的任何配置文件中：

```bash
hermes -p <your-worker-profile> skills list | grep kanban-worker
```

如果捆绑副本缺失，为该配置文件恢复它：

```bash
hermes -p <your-worker-profile> skills reset kanban-worker --restore
```

调度器在生成每个工作者时还会自动传递 `--skills kanban-worker`，因此即使配置文件的默认技能配置不包含它，工作者也始终拥有模式库可用。

### 为特定任务固定额外技能

有时单个任务需要被分配者配置文件默认不携带的专业上下文 —— 需要 `translation` 技能的翻译工作，需要 `github-code-review` 的审查任务，需要 `security-pr-audit` 的安全审计。与其每次编辑被分配者的配置文件，不如直接将技能附加到任务。

**来自编排器智能体**（通常情况 —— 一个智能体将工作路由到另一个），使用 `kanban_create` 工具的 `skills` 数组：

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

**来自人类（CLI / 斜杠命令）**，为每个技能重复 `--skill`：

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**来自仪表板**，在内联创建表单的 **skills** 字段中以逗号分隔输入技能。

这些技能是**附加**到内置的 `kanban-worker` 的 —— 调度器为每个技能（以及内置的）发出一个 `--skills <name>` 标志，因此工作者以所有技能加载生成。技能名称必须与被分配者配置文件上实际安装的技能匹配（运行 `hermes skills list` 查看可用的内容）；没有运行时安装。

### 编排器技能

一个**行为良好的编排器不会自己做工作。** 它将用户的目标分解为任务，链接它们，将每个分配给您设置的配置文件之一，然后退后。`kanban-orchestrator` 技能将此编码为工具调用模式：反诱惑规则、Step-0 配置文件发现提示（调度器在未知的被分配者名称上静默失败，因此编排器必须将每张卡片锚定到您机器上实际存在的配置文件），以及基于 `kanban_create` / `kanban_link` / `kanban_comment` 的分解剧本。

一个经典的编排器轮次（两个并行研究者交接给撰写者）：

```
# 来自用户的目标："起草一篇关于 ICP 融资格局的发布文章"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # 当两个研究者都完成时提升到 'ready'
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# 可选：事后添加发现的跨领域依赖，无需重新创建任务
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

`kanban-orchestrator` 是一个捆绑技能。它在安装和更新期间同步到每个配置文件，因此没有单独的 Skills Hub 安装步骤。验证它存在于您的编排器配置文件中：

```bash
hermes -p orchestrator skills list | grep kanban-orchestrator
```

如果捆绑副本缺失，为该配置文件恢复它：

```bash
hermes -p orchestrator skills reset kanban-orchestrator --restore
```

为获得最佳效果，将其与工具集限制为看板操作（`kanban`、`gateway`、`memory`）的配置文件配对，以便编排器即使尝试也无法执行实现任务。

## 仪表板（GUI）

`/kanban` CLI 和斜杠命令足以无头运行看板，但可视化看板通常是人机协同的正确界面：分类、跨配置文件监管、阅读评论线程、在列之间拖动卡片。Hermes 将其作为 `plugins/kanban/` 的**捆绑仪表板插件**提供 —— 不是核心功能，不是单独的服务 —— 遵循 [扩展仪表板](./extending-the-dashboard) 中列出的模型。

打开方式：

```bash
hermes kanban init      # 一次性：如果尚未存在则创建 kanban.db
hermes dashboard        # "Kanban" 标签页出现在导航中，在 "Skills" 之后
```

### 插件提供的内容

- 一个 **Kanban** 标签页，每状态一列：`triage`、`todo`、`ready`、`running`、`blocked`、`done`（切换开启时加上 `archived`）。
  - `triage` 是规范者期望充实的大致想法的停放列。使用 `hermes kanban create --triage` 创建的任务（或通过 Triage 列的内联创建）落在这里，调度器在人工或规范者将其提升到 `todo` / `ready` 之前不会碰它们。运行 `hermes kanban specify <id>` 让辅助 LLM 将分类任务扩展为具体的规范（标题 + 带有目标、方法、验收标准的正文）并一次性提升到 `todo`；`--all` 一次性扫描每个分类任务。在 `config.yaml` 的 `auxiliary.triage_specifier` 下配置哪个模型运行规范者。
- 卡片显示任务 id、标题、优先级徽章、租户标签、分配的配置文件、评论/链接计数、**进度 pill**（任务有依赖项时完成的子项 `N/M`），以及"创建于 N 前"。每卡片复选框启用多选。
- **Running 内的每配置文件泳道** —— 工具栏复选框切换 Running 列按被分配者子分组。
- **通过 WebSocket 实时更新** —— 插件在短轮询间隔内跟踪追加的 `task_events` 表；看板在任何配置文件（CLI、网关或另一个仪表板标签页）操作的瞬间反映更改。重新加载会被防抖，因此事件突发触发单次重新获取。
- **拖放** 卡片在列之间更改状态。拖放发送 `PATCH /api/plugins/kanban/tasks/:id`，通过与 CLI 使用的相同的 `kanban_db` 代码路由 —— 三个界面永远不会漂移。移动到破坏性状态（`done`、`archived`、`blocked`）会提示确认。触摸设备使用基于指针的回退，因此看板在平板电脑上可用。
- **内联创建** —— 点击任何列标题上的 `+` 输入标题、被分配者、优先级和（可选）来自每个现有任务下拉菜单的父任务。从 Triage 列创建会自动将新任务停放在分类中。
- **多选与批量操作** —— shift/ctrl-点击卡片或勾选其复选框将其添加到选择中。顶部出现批量操作栏，带有批量状态转换、归档和重新分配（通过配置文件下拉菜单，或"(unassign)"）。破坏性批量操作先确认。每 id 部分失败会报告，不会中止其余操作。
- **点击卡片**（不带 shift/ctrl）打开侧边抽屉（Escape 或点击外部关闭），包含：
  - **可编辑标题** —— 点击标题重命名。
  - **可编辑被分配者 / 优先级** —— 点击元数据行重写。
  - **可编辑描述** —— 默认 markdown 渲染（标题、粗体、斜体、内联代码、围栏代码、`http(s)` / `mailto:` 链接、项目符号列表），带有"编辑"按钮可交换为文本区域。Markdown 渲染是一个微小的、XSS 安全的渲染器 —— 每次替换都在 HTML 转义的输入上运行，只有 `http(s)` / `mailto:` 链接通过，并且始终设置 `target="_blank"` + `rel="noopener noreferrer"`。
  - **依赖编辑器** —— 父级和子级的芯片列表，每个都有 `×` 取消链接，加上每个其他任务的下拉菜单以添加新的父级或子级。循环尝试在服务端被拒绝，并带有明确的消息。
  - **状态操作行**（→ triage / → ready / → running / block / unblock / complete / archive），破坏性转换带有确认提示。对于**Triage**列中的卡片，该行还暴露一个**✨ Specify**按钮，调用辅助 LLM（`config.yaml` 中的 `auxiliary.triage_specifier`）将单行扩展为具体规范（标题 + 带有目标、方法、验收标准的正文）并将任务提升到 `todo`。相同的行为可从 CLI（`hermes kanban specify <id>` / `--all`）、任何网关平台（`/kanban specify <id>`）和通过 `POST /api/plugins/kanban/tasks/:id/specify` 以编程方式访问。
  - 结果部分（也是 markdown 渲染）、评论线程带 Enter 提交、最近 20 个事件。
- **工具栏过滤器** —— 自由文本搜索、租户下拉菜单（默认为 `config.yaml` 中的 `dashboard.kanban.default_tenant`）、被分配者下拉菜单、"显示归档"切换、"按配置文件泳道"切换，以及一个 **Nudge dispatcher** 按钮，因此您不必等待下一个 60 秒扫描。

视觉上目标是熟悉的 Linear / Fusion 布局：深色主题、带计数的列标题、彩色状态点、优先级和租户的 pill 芯片。插件只读取主题 CSS 变量（`--color-*`、`--radius`、`--font-mono`、...），因此它会随活动仪表板主题自动换肤。

### 架构

GUI 严格是一个**通读 DB + 写穿 kanban_db** 层，没有自己的域逻辑：

```
┌────────────────────────┐      WebSocket（跟踪 task_events）
│   React SPA（插件）   │ ◀──────────────────────────────────┐
│   HTML5 拖放         │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     写入直接调用 kanban_db.*      │
│  FastAPI 路由器        │     相同的代码路径                │
│  plugins/kanban/       │     CLI /kanban 动词使用          │
│  dashboard/plugin_api.py                                    │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── 追加 task_events ──────────┘
│  (WAL, 共享)           │
└────────────────────────┘
```

### REST 接口

所有路由都挂载在 `/api/plugins/kanban/` 下，受仪表板的临时会话令牌保护：

| 方法 | 路径 | 目的 |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | 按状态列分组的完整看板，加上租户 + 被分配者用于过滤器下拉菜单 |
| `GET` | `/tasks/:id` | 任务 + 评论 + 事件 + 链接 |
| `POST` | `/tasks` | 创建（包装 `kanban_db.create_task`，接受 `triage: bool` 和 `parents: [id, …]`） |
| `PATCH` | `/tasks/:id` | 状态 / 被分配者 / 优先级 / 标题 / 正文 / 结果 |
| `POST` | `/tasks/bulk` | 将相同的补丁（状态 / 归档 / 被分配者 / 优先级）应用到 `ids` 中的每个 id。报告每 id 失败，不会中止兄弟操作 |
| `POST` | `/tasks/:id/comments` | 追加评论 |
| `POST` | `/tasks/:id/specify` | 运行分类规范者 —— 辅助 LLM 充实任务正文并将其从 `triage` 提升到 `todo`。返回 `{ok, task_id, reason, new_title}`；`ok=false` 带有人类可读的原因（"不在分类中" / 无 aux 客户端 / LLM 错误）是 200，不是 4xx |
| `POST` | `/links` | 添加依赖（`parent_id` → `child_id`） |
| `DELETE` | `/links?parent_id=…&child_id=…` | 移除依赖 |
| `POST` | `/dispatch?max=…&dry_run=…` | 推动调度器 —— 跳过 60 秒等待 |
| `GET` | `/config` | 从 `config.yaml` 读取 `dashboard.kanban` 首选项 —— `default_tenant`、`lane_by_profile`、`include_archived_by_default`、`render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events` 行的实时流 |

每个处理程序都是一个薄包装 —— 插件约 700 行 Python（路由器 + WebSocket 跟踪 + 批量批处理程序 + 配置读取器），不添加新的业务逻辑。一个微小的 `_conn()` 辅助程序在每次读写时自动初始化 `kanban.db`，因此无论用户是先打开仪表板、直接访问 REST API 还是运行 `hermes kanban init`，新安装都能工作。

### 仪表板配置

`~/.hermes/config.yaml` 中 `dashboard.kanban` 下的任何这些键都会更改标签页的默认值 —— 插件在加载时通过 `GET /config` 读取它们：

```yaml
dashboard:
  kanban:
    default_tenant: acme              # 预选租户过滤器
    lane_by_profile: true             # "按配置文件泳道"切换的默认值
    include_archived_by_default: false
    render_markdown: true             # 设置为 false 以使用纯 <pre> 渲染
```

每个键都是可选的，并回退到所示的默认值。

### 安全模型

仪表板的 HTTP 认证中间件[显式跳过 `/api/plugins/`](./extending-the-dashboard#后端-api-路由) —— 插件路由按设计未经认证，因为仪表板默认绑定到 localhost。这意味着 kanban REST 接口可从主机上的任何进程访问。

WebSocket 采取一个额外步骤：它需要仪表板的临时会话令牌作为 `?token=…` 查询参数（浏览器无法在升级请求上设置 `Authorization`），匹配浏览器内 PTY 桥接使用的模式。

如果您运行 `hermes dashboard --host 0.0.0.0`，每个插件路由 —— 包括 kanban —— 都会从网络可达。**不要在共享主机上这样做。** 看板包含任务正文、评论和工作区路径；攻击者访问这些路由会获得对整个协作表面的读取访问权限，还可以创建 / 重新分配 / 归档任务。

`~/.hermes/kanban.db` 中的任务按设计是与配置文件无关的（这就是协调原语）。如果您用 `hermes -p <profile> dashboard` 打开仪表板，看板仍会显示主机上任何其他配置文件创建的任务。同一用户拥有所有配置文件，但如果多个角色共存，这值得了解。

### 实时更新

`task_events` 是一个带有单调 `id` 的追加 SQLite 表。WebSocket 端点保存每个客户端最后看到的事件 id，并在新行到达时推送它们。当事件突发到达时，前端重新加载（非常便宜的）看板端点 —— 比尝试从每种事件类型修补本地状态更简单、更正确。WAL 模式意味着读取循环永远不会阻塞调度器的 `BEGIN IMMEDIATE` 声明事务。

### 扩展它

插件使用标准的 Hermes 仪表板插件合约 —— 请参阅 [扩展仪表板](./extending-the-dashboard) 了解完整的清单参考、shell 插槽、页面范围插槽和 Plugin SDK。额外的列、自定义卡片 chrome、租户过滤布局或完整的 `tab.override` 替换都可以在不 fork 此插件的情况下表达。

要禁用而不移除：将 `dashboard.plugins.kanban.enabled: false` 添加到 `config.yaml`（或删除 `plugins/kanban/dashboard/manifest.json`）。

### 范围边界

GUI 有意保持精简。插件做的所有事情都可以从 CLI 访问；插件只是让人类操作更舒适。自动分配、预算、治理门和组织结构图视图仍然是用户空间 —— 一个路由器配置文件、另一个插件或 `tools/approval.py` 的重用 —— 正如设计规范的超出范围部分所列。

## CLI 命令参考

这是**您**（或脚本、cron、仪表板）用来驱动看板的界面。在调度器内运行的工作者使用 `kanban_*` [工具界面](#how-workers-interact-with-the-board) 进行相同的操作 —— 这里的 CLI 和那里的工具都通过 `kanban_db` 路由，因此两个界面按构造一致。

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
hermes kanban assign <id> <profile>                    # 或 'none' 以取消分配
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
hermes kanban heartbeat <id> [--note "..."]            # 长操作的 worker 存活信号
hermes kanban runs <id> [--json]                       # 尝试历史（每次运行一行）
hermes kanban assignees [--json]                       # 磁盘上的配置文件 + 每被分配者任务计数
hermes kanban dispatch [--dry-run] [--max N]           # 一次性扫描
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # 已弃用 —— 独立调度器（改用 `hermes gateway start`）
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # 每状态 + 每被分配者计数
hermes kanban log <id> [--tail BYTES]                  # 来自 ~/.hermes/kanban/logs/ 的 worker 日志
hermes kanban notify-subscribe <id>                    # 网关桥接钩子（由网关中的 /kanban 使用）
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # worker 看到的内容
hermes kanban specify [<id> | --all] [--tenant T]      # 将分类列的想法充实
        [--author NAME] [--json]                       #   为完整规范并提升到 todo
hermes kanban gc [--event-retention-days N]            # 工作区 + 旧事件 + 旧日志
        [--log-retention-days N]
```

所有命令也可用作交互式 CLI 和消息网关中的斜杠命令（请参阅下面的 [`/kanban` 斜杠命令](#kanban-斜杠命令)）。

## `/kanban` 斜杠命令 {#kanban-斜杠命令}

每个 `hermes kanban <action>` 动词也可作为 `/kanban <action>` 访问 —— 在交互式 `hermes chat` 会话**内部**以及从任何网关平台（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost、电子邮件、SMS）。两个界面都调用完全相同的 `hermes_cli.kanban.run_slash()` 入口点，该入口点重用 `hermes kanban` argparse 树，因此参数界面、标志和输出格式在 CLI、`/kanban` 和 `hermes kanban` 之间完全相同。您不必离开聊天就能驱动看板。

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # 将分类单行充实为真实规范
/kanban specify --all --tenant engineering  # 一次性扫描一个租户中的每个分类任务
```

引用多字参数的方式与在 shell 上相同 —— `run_slash` 用 `shlex.split` 解析行的其余部分，因此 `"..."` 和 `'...'` 都有效。

### 运行中使用：`/kanban` 绕过运行中智能体守卫

网关通常会在智能体仍在思考时将斜杠命令和用户消息排队 —— 这就是阻止您意外地在第一次飞行中启动第二轮的原因。**`/kanban` 被显式免除于此守卫。** 看板存在于 `~/.hermes/kanban.db` 中，而不是运行中智能体的状态中，因此读取（`list`、`show`、`context`、`tail`、`watch`、`stats`、`runs`）和写入（`comment`、`unblock`、`block`、`assign`、`archive`、`create`、`link`、…）都会立即通过，即使在轮次中也是如此。

这就是分离的全部意义：

- 一个工作者阻塞等待对等者 → 您从手机发送 `/kanban unblock t_abcd`，调度器在下一次扫描时拾取对等者。被阻塞的工作者不会被中断 —— 它只是不再被阻塞。
- 您发现一张卡片需要人类上下文 → `/kanban comment t_xyz "use the 2026 schema, not 2025"` 落在任务线程上，该任务的*下一次*运行会在 `kanban_show()` 中读取它。
- 您想知道您的集群在做什么，而不停止编排器 → `/kanban list --mine` 或 `/kanban stats` 检查看板，而不触及您的主对话。

### 在 `/kanban create` 上自动订阅（仅限网关）

当您从网关使用 `/kanban create "…"` 创建任务时，原始聊天（平台 + 聊天 id + 线程 id）会自动订阅该任务的终端事件（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）。每个终端事件您都会收到一条消息 —— 包括 `completed` 上工作者结果摘要的第一行 —— 无需轮询或记住任务 id。

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

订阅在任务达到 `done` 或 `archived` 时自动移除。如果您用 `--json`（机器输出）编写创建脚本，则跳过自动订阅 —— 假设脚本调用者希望通过 `/kanban notify-subscribe` 显式管理订阅。

### 消息中的输出截断

网关平台有实际的消息长度上限。如果 `/kanban list`、`/kanban show` 或 `/kanban tail` 产生超过约 3800 个字符的输出，响应会被截断，并带有 `… (truncated; use \`hermes kanban …\` in your terminal for full output)` 页脚。CLI 界面没有这样的上限。

### 自动补全

在交互式 CLI 中，键入 `/kanban ` 并按 Tab 会循环浏览内置子命令列表（`list`、`ls`、`show`、`create`、`assign`、`link`、`unlink`、`claim`、`comment`、`complete`、`block`、`unblock`、`archive`、`tail`、`dispatch`、`context`、`init`、`gc`）。CLI 参考中列出的其余动词（`watch`、`stats`、`runs`、`log`、`assignees`、`heartbeat`、`notify-subscribe`、`notify-list`、`notify-unsubscribe`、`daemon`）也有效 —— 它们只是尚未在自动补全提示列表中。

## 协作模式

看板支持这八种模式，无需任何新原语：

| 模式 | 形态 | 示例 |
|---|---|---|
| **P1 扇出** | N 个同级，相同角色 | "并行研究 5 个角度" |
| **P2 流水线** | 角色链：侦察 → 编辑 → 撰写者 | 每日简报组装 |
| **P3 投票 / 法定人数** | N 个同级 + 1 个聚合器 | 3 个研究者 → 1 个审查者选择 |
| **P4 长期日志** | 相同配置文件 + 共享目录 + cron | Obsidian vault |
| **P5 人机协同** | 工作者阻塞 → 用户评论 → 解阻塞 | 模糊决策 |
| **P6 `@mention`** | 从散文中内联路由 | `@reviewer look at this` |
| **P7 线程范围工作区** | 线程中的 `/kanban here` | 每项目网关线程 |
| **P8 集群运维** | 一个配置文件，N 个主题 | 50 个社交账户 |
| **P9 分类规范者** | 粗略想法 → `triage` → `hermes kanban specify` 扩展正文 → `todo` | "将这个单行变成规范的任务" |

每种模式的工作示例，请参阅 `docs/hermes-kanban-v1-spec.pdf`。

## 多租户使用

当一个专家集群为多个企业服务时，为每个任务标记租户：

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

工作者接收 `$HERMES_TENANT` 并按前缀命名空间化他们的内存写入。看板、调度器和配置文件定义都是共享的；只有数据是范围限定的。

## 网关通知

当您从网关（Telegram、Discord、Slack 等）运行 `/kanban create …` 时，原始聊天会自动订阅新任务。网关的后台通知器每几秒轮询 `task_events` 一次，并向该聊天发送每个终端事件（`completed`、`blocked`、`gave_up`、`crashed`、`timed_out`）的一条消息。完成的任务还会发送工作者 `--result` 的第一行，因此您无需 `/kanban show` 就能看到结果。

您可以从 CLI 显式管理订阅 —— 当脚本 / cron 作业想要通知它未发起的聊天时很有用：

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

订阅在任务达到 `done` 或 `archived` 时自动移除；无需清理。

## 运行 —— 每次尝试一行

任务是逻辑工作单元；**运行**是执行它的一次尝试。当调度器声明就绪任务时，它在 `task_runs` 中创建一行，并将 `tasks.current_run_id` 指向它。当该尝试结束时 —— 完成、阻塞、崩溃、超时、生成失败、回收 —— 运行行以 `outcome` 关闭，任务指针清除。已被尝试三次的任务有三个 `task_runs` 行。

为什么需要两个表而不仅仅是修改任务：您需要**完整的尝试历史**来进行真实的事后分析（"第二次审查尝试到达批准，第三次合并"），并且您需要一个干净的地方来挂每尝试元数据 —— 哪些文件更改了、哪些测试运行了、审查者记录了哪些发现。这些是运行事实，不是任务事实。

运行也是**结构化交接**所在之处。当工作者完成任务（通过 `kanban_complete(...)`）时，它可以传递：

- `summary`（工具参数）/ `--summary`（CLI）—— 人类交接；放在运行上；下游子级在其 `build_worker_context` 中看到它。
- `metadata`（工具参数）/ `--metadata`（CLI）—— 运行上的自由格式 JSON 字典；子级看到它与摘要一起序列化。
- `result`（工具参数）/ `--result`（CLI）—— 放在任务行上的短日志行（遗留字段，为向后兼容保留）。

下游子级读取每个父级最近完成的运行的摘要 + 元数据。重试工作者读取他们自己任务上的先前尝试（结果、摘要、错误），因此他们不会重复已经失败的路径。

```
# 工作者实际做的 —— 来自智能体循环内的工具调用：
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

当工作者无法完成时，相同的交接也可从 CLI 访问 —— 例如被放弃的任务，或您从仪表板手动标记为完成的任务：

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# 审查重试任务上的尝试历史：
hermes kanban runs t_abcd
#   #  结果       配置文件           耗时   开始时间
#   1  blocked    worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed  worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

运行在仪表板（抽屉中的运行历史部分，每次尝试一行彩色行）和 REST API（`GET /api/plugins/kanban/tasks/:id` 返回 `runs[]` 数组）上暴露。带有 `{status: "done", summary, metadata}` 的 `PATCH /api/plugins/kanban/tasks/:id` 将两者都转发到内核，因此仪表板的"标记完成"按钮与 CLI 等效。`task_events` 行携带其所属的 `run_id`，因此 UI 可以按尝试分组，并且 `completed` 事件在其负载中嵌入第一行摘要（限制为 400 个字符），因此网关通知器可以在不进行第二次 SQL 往返的情况下渲染结构化交接。

**批量关闭注意事项。** `hermes kanban complete a b c --summary X` 被拒绝 —— 结构化交接是按运行的，因此将相同的摘要复制粘贴到 N 个任务几乎总是错误的。批量关闭*不带* `--summary` / `--metadata` 仍然适用于常见的"我完成了一堆管理任务"情况。

**从状态更改回收的运行。** 如果您在仪表板中将运行中的任务拖离 `running`（回到 `ready`，或直接到 `todo`），或归档仍在运行的任务，飞行中的运行以 `outcome='reclaimed'` 关闭，而不是被孤立。当 `tasks.current_run_id` 为 `NULL` 时，`task_runs` 行始终处于终端状态，反之亦然 —— 该不变式在 CLI、仪表板、调度器和通知器之间保持。

**从未声明完成的合成运行。** 完成或阻塞从未被声明的任务（例如，人类从仪表板关闭就绪任务并带有摘要，或 CLI 用户运行 `hermes kanban complete <ready-task> --summary X`）否则会丢弃交接。相反，内核插入一个零持续时间运行行（`started_at == ended_at`），携带摘要 / 元数据 / 原因，以便尝试历史保持完整。`completed` / `blocked` 事件的 `run_id` 指向该行。

**实时抽屉刷新。** 当仪表板的 WebSocket 事件流报告用户当前正在查看的任务的新事件时，抽屉会重新加载自身（通过线程到其 `useEffect` 依赖列表中的每任务事件计数器）。不再需要关闭和重新打开来查看运行的新行或更新的结果。

### 向前兼容

`tasks` 上的两个可空列保留给 v2 工作流路由：`workflow_template_id`（此任务属于哪个模板）和 `current_step_key`（该模板中哪个步骤是活动的）。v1 内核忽略它们进行路由，但允许客户端写入它们，因此 v2 发布可以添加路由机制而无需另一次模式迁移。

## 事件参考

每次转换都会向 `task_events` 追加一行。每行携带一个可选的 `run_id`，因此 UI 可以按尝试分组事件。类型分为三个集群，以便过滤容易（`hermes kanban watch --kinds completed,gave_up,timed_out`）：

**生命周期**（关于任务作为逻辑单元的变化）：

| 类型 | 负载 | 何时 |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | 任务插入。`run_id` 为 `NULL`。 |
| `promoted` | — | `todo → ready` 因为所有父级都达到 `done`。`run_id` 为 `NULL`。 |
| `claimed` | `{lock, expires, run_id}` | 调度器原子声明就绪任务以生成。 |
| `completed` | `{result_len, summary?}` | 工作者写入 `--result` / `--summary` 并且任务达到 `done`。`summary` 是第一行交接（400 字符上限）；完整版本存在于运行行上。如果在从未声明的任务上调用 `complete_task` 并带有交接字段，则合成一个零持续时间运行，以便 `run_id` 仍然指向某些内容。 |
| `blocked` | `{reason}` | 工作者或人类将任务翻转为 `blocked`。在从未声明的任务上调用并带有 `--reason` 时合成一个零持续时间运行。 |
| `unblocked` | — | `blocked → ready`，手动或通过 `/unblock`。`run_id` 为 `NULL`。 |
| `archived` | — | 从默认看板隐藏。如果任务仍在运行，则携带作为副作用被回收的运行的 `run_id`。 |

**编辑**（不是转换的人类驱动更改）：

| 类型 | 负载 | 何时 |
|---|---|---|
| `assigned` | `{assignee}` | 被分配者更改（包括取消分配）。 |
| `edited` | `{fields}` | 标题或正文更新。 |
| `reprioritized` | `{priority}` | 优先级更改。 |
| `status` | `{status}` | 仪表板拖放直接写入状态（例如 `todo → ready`）。从 `running` 拖离时携带被回收的运行的 `run_id`；否则 `run_id` 为 NULL。 |

**工作者遥测**（关于执行过程，而非逻辑任务）：

| 类型 | 负载 | 何时 |
|---|---|---|
| `spawned` | `{pid}` | 调度器成功启动工作者进程。 |
| `heartbeat` | `{note?}` | 工作者调用 `hermes kanban heartbeat $TASK` 以在长操作期间发出存活信号。 |
| `reclaimed` | `{stale_lock}` | 声明 TTL 过期而没有完成；任务回到 `ready`。 |
| `crashed` | `{pid, claimer}` | 工作者 PID 不再存活，但 TTL 尚未过期。 |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | 超过 `max_runtime_seconds`；调度器 SIGTERM（5 秒宽限期后 SIGKILL）并重新排队。 |
| `spawn_failed` | `{error, failures}` | 一次生成尝试失败（缺少 PATH、工作区无法挂载、…）。计数器递增；任务返回 `ready` 以重试。 |
| `gave_up` | `{failures, error}` | 在 N 次连续 `spawn_failed` 后断路器触发。任务自动阻塞，并带有最后一个错误。默认 N = 5；通过 `--failure-limit` 覆盖。 |

`hermes kanban tail <id>` 显示单个任务的这些。`hermes kanban watch` 板级流式传输它们。

## 超出范围

Kanban 按设计是单主机的。`~/.hermes/kanban.db` 是本地 SQLite 文件，调度器在同一台机器上生成工作者。跨两台主机运行共享看板不受支持 —— 没有"主机 A 上的工作者 X，主机 B 上的工作者 Y"的协调原语，崩溃检测路径假设 PID 是主机本地的。如果您需要多主机，请在每个主机上运行独立的看板，并使用 `delegate_task` / 消息队列来桥接它们。

## 设计规范

完整的设计 —— 架构、并发正确性、与其他系统的比较、实现计划、风险、开放问题 —— 位于 `docs/hermes-kanban-v1-spec.pdf`。在提交任何行为更改 PR 之前阅读该文档。
