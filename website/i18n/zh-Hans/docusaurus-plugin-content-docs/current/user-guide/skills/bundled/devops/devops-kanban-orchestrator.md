---
title: "Kanban Orchestrator"
sidebar_label: "Kanban Orchestrator"
description: "用于通过 Kanban 路由工作的编排器 profile 的任务分解手册及反诱惑规则"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Orchestrator

用于通过 Kanban 路由工作的编排器 profile 的任务分解手册及反诱惑规则。"不要自己执行工作"规则和基本生命周期会自动注入每个 kanban worker 的系统 prompt（提示词）中；本 skill 是当你专门扮演编排器角色时使用的更深层手册。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/devops/kanban-orchestrator` |
| 版本 | `3.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `kanban`, `multi-agent`, `orchestration`, `routing` |
| 相关 skill | [`kanban-worker`](/user-guide/skills/bundled/devops/devops-kanban-worker) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Kanban Orchestrator — 任务分解手册

> **核心 worker 生命周期**（包括 `kanban_create` 扇出模式和"分解而非执行"规则）通过 `KANBAN_GUIDANCE` 系统 prompt 块自动注入每个 kanban 进程。本 skill 是当你作为编排器 profile、整个职责就是路由时使用的更深层手册。

## Profile 由用户配置——不是固定名单

Hermes 的配置因人而异。有些用户运行单个 profile 处理所有事务；有些运行小型集群（`docker-worker`、`cron-worker`）；有些运行自己命名的精选专家团队。**没有默认的专家名单**——编排器 skill 不知道此机器上存在哪些 profile。

在扇出之前，你必须基于实际存在的 profile 来制定分解方案。调度器会静默地忽略无法识别的 assignee 名称——它不会自动纠正、不会建议、也不会回退。因此，在只有 `docker-worker` 的配置上，分配给 `researcher` 的卡片会永远停留在 `ready` 状态。

**第 0 步：在规划前发现可用的 profile。**

使用以下方法之一：

- `hermes profile list` — 打印此机器上已配置的 profile 表。如果有终端工具，通过终端工具运行；否则询问用户。
- `kanban_list(assignee="<some-name>")` — 验证单个名称。对于未知 assignee 返回空列表（而非报错），因此只能确认你已在考虑的名称。
- **直接询问用户。** 当目标需要多个专家时，"你配置了哪些 profile？"是一个合理的开场问题。

将结果缓存在工作记忆中供本次对话使用。每轮都重新询问会浪费工具调用。

## 何时使用看板（vs. 直接执行工作）

当以下任一条件成立时，创建 Kanban 任务：

1. **需要多个专家。** 研究 + 分析 + 写作需要三个 profile。
2. **工作应在崩溃或重启后继续存在。** 长期运行、周期性或重要的任务。
3. **用户可能需要介入。** 任意步骤需要人工参与。
4. **多个子任务可以并行运行。** 扇出以提高速度。
5. **预期需要审查/迭代。** 审查者 profile 循环处理起草者的输出。
6. **审计追踪很重要。** 看板行永久保存在 SQLite 中。

如果*以上均不适用*——这是一个小型一次性推理任务——改用 `delegate_task` 或直接回答用户。

## 反诱惑规则

你的职责描述是"路由，不执行"。执行该规则的约束：

- **不要自己执行工作。** 你受限的工具集通常甚至不包含用于实现的终端/文件/代码/网络工具。如果你发现自己在"快速修复这个"——停下来，为合适的专家创建任务。
- **对于任何具体任务，创建 Kanban 任务并分配它。** 每一次都如此。
- **在创建卡片之前拆分多通道请求。** 用户的一个 prompt 可能包含多个独立的工作流。先提取这些通道，然后每个通道创建一张卡片，而不是将不相关的工作打包到单个实现者卡片中。
- **并行运行独立通道。** 如果两张卡片不需要彼此的输出，不要链接它们，让调度器可以扇出处理。只链接真正的数据依赖。
- **永远不要将依赖工作创建为独立的 ready 卡片。** 如果一张卡片必须等待另一张卡片，在原始 `kanban_create` 调用中传入 `parents=[...]`。不要先创建再链接，也不要依赖卡片正文中的"等待 T1"之类的描述。
- **如果没有专家适合现有 profile，询问用户应创建哪个 profile 或使用哪个现有 profile。** 不要凭空发明 profile 名称；调度器会静默丢弃未知 assignee。
- **分解、路由、汇总——这就是全部工作。**

## 任务分解手册

### 第 1 步——理解目标

如果目标不明确，提出澄清性问题。询问的成本很低；派出错误的团队代价高昂。

### 第 2 步——草拟任务图

在创建任何内容之前，在回复用户时大声（在响应中）草拟任务图。将每个具体工作流视为候选卡片：

1. 从请求中提取通道。
2. 将每个通道映射到第 0 步中发现的某个 profile。如果某个通道不适合任何现有 profile，询问用户使用或创建哪个。
3. 决定每个通道是独立的还是受另一个通道门控的。
4. 将独立通道创建为无父链接的并行卡片。
5. 将综合/审查/集成卡片创建时带上其所依赖通道的父链接。使用未完成父任务创建的子任务从 `todo` 开始；调度器仅在每个父任务完成后才将其提升为 `ready`。

应该扇出的 prompt 示例（使用占位符 profile 名称——替换为用户配置中实际存在的名称）：

- "构建一个应用" → 一张卡片给面向设计的 profile 负责产品/UI 方向，一两张卡片给工程 profile 负责实现，如果用户有审查者 profile，再加一张后续的集成/审查卡片。
- "修复阻塞项并检查模型变体" → 一张实现卡片用于修复阻塞项，加一张发现/研究卡片用于配置/源码验证。最终的审查者卡片可以依赖两者。
- "研究文档并实现" → 文档研究卡片可以与代码库发现卡片并行运行；只有当实现真正需要这些发现时才等待。
- "分析这张截图并找到相关代码" → 一张卡片给具备视觉能力的 profile 进行视觉分析，同时另一张卡片搜索代码库。

"也"、"最后"或"和"等词语不自动意味着依赖关系。它们通常意味着"确保在汇报前涵盖这一点"。只有当一张卡片在另一张卡片的输出存在之前无法开始时，才链接任务。

在创建卡片之前将任务图展示给用户。让他们纠正——包括哪个实际 profile 名称应该负责每个通道。

### 第 3 步——创建任务并链接

使用第 0 步中的 profile 名称。以下示例使用占位符 `<profile-A>`、`<profile-B>`、`<profile-C>`——替换为用户实际拥有的名称。

```python
t1 = kanban_create(
    title="research: Postgres cost vs current",
    assignee="<profile-A>",  # whichever profile handles research on this setup
    body="Compare estimated infrastructure costs, migration costs, and ongoing ops costs over a 3-year window. Sources: AWS/GCP pricing, team time estimates, current Postgres bills from peers.",
    tenant=os.environ.get("HERMES_TENANT"),
)["task_id"]

t2 = kanban_create(
    title="research: Postgres performance vs current",
    assignee="<profile-A>",  # same profile, run in parallel
    body="Compare query latency, throughput, and scaling characteristics at our expected data volume (~500GB, 10k QPS peak). Sources: benchmark papers, public case studies, pgbench results if easy.",
)["task_id"]

t3 = kanban_create(
    title="synthesize migration recommendation",
    assignee="<profile-B>",  # whichever profile does synthesis/analysis
    body="Read the findings from T1 (cost) and T2 (performance). Produce a 1-page recommendation with explicit trade-offs and a go/no-go call.",
    parents=[t1, t2],
)["task_id"]

t4 = kanban_create(
    title="draft decision memo",
    assignee="<profile-C>",  # whichever profile drafts user-facing prose
    body="Turn the analyst's recommendation into a 2-page memo for the CTO. Match the tone of previous decision memos in the team's knowledge base.",
    parents=[t3],
)["task_id"]
```

`parents=[...]` 门控提升——子任务保持在 `todo` 状态，直到每个父任务达到 `done`，然后自动提升为 `ready`。无需手动协调；调度器和依赖引擎会处理这一切。

如果任务图有依赖关系，先创建父卡片，捕获其返回的 id，并在子卡片的 `kanban_create` 调用中将这些 id 包含在 `parents` 列表中。避免并行创建所有卡片后再链接；这会产生一个时间窗口，调度器可能在子任务的输入存在之前就认领它。

### 第 4 步——完成你自己的任务

如果你是作为任务被派生的（例如，规划者 profile 被分配了 `T0: "调查 Postgres 迁移"`），用你创建内容的摘要标记它为完成：

```python
kanban_complete(
    summary="decomposed into T1-T4: 2 research lanes in parallel, 1 synthesis on their outputs, 1 prose draft on the recommendation",
    metadata={
        "task_graph": {
            "T1": {"assignee": "<profile-A>", "parents": []},
            "T2": {"assignee": "<profile-A>", "parents": []},
            "T3": {"assignee": "<profile-B>", "parents": ["T1", "T2"]},
            "T4": {"assignee": "<profile-C>", "parents": ["T3"]},
        },
    },
)
```

### 第 5 步——向用户汇报

用简明的文字告诉他们你创建了什么，并说明你使用的实际 profile 名称：

> 我已排队 4 个任务：
> - **T1**（`<profile-A>`）：成本对比
> - **T2**（`<profile-A>`）：性能对比，与 T1 并行
> - **T3**（`<profile-B>`）：综合 T1 + T2 生成建议
> - **T4**（`<profile-C>`）：将 T3 转化为 CTO 备忘录
>
> 调度器现在将认领 T1 和 T2。T3 在两者完成后启动。T4 完成时你会收到 gateway 通知。使用仪表板或 `hermes kanban tail <id>` 跟踪进度。

## 常见模式

**扇出 + 扇入（研究 → 综合）：** N 张无父链接的研究类卡片，一张以所有研究卡片为父的综合卡片。

**并行实现 + 验证：** 一张实现者卡片进行变更，同时一张探索/研究卡片验证配置、文档或源码映射。审查者卡片可以依赖两者。不要因为用户在一句话中同时提到了两者，就让实现者承担不相关的验证工作。

**带门控的流水线：** `planner → implementer → reviewer`。每个阶段的 `parents=[previous_task]`。审查者阻塞或完成；如果审查者阻塞，操作员带着反馈解除阻塞并重新派发。

**同 profile 队列：** N 个任务，全部分配给同一个 profile，彼此之间无依赖。调度器串行处理——该 profile 按优先级顺序处理它们，在自己的记忆中积累经验。

**人工参与循环：** 任何任务都可以调用 `kanban_block()` 等待输入。调度器在 `/unblock` 后重新派发。评论线程携带完整上下文。

## 常见陷阱

**发明不存在的 profile 名称。** 调度器会静默地忽略无法识别的 assignee——卡片会永远停留在 `ready` 状态。始终从第 0 步发现的 profile 中分配；如果不确定，询问用户。

**将独立通道打包到一张卡片中。** 如果用户要求两个独立的结果，创建两张卡片。示例："修复阻塞项并检查模型变体"不是一个修复任务；为修复创建一张修复/工程卡片，为变体检查创建一张探索/研究卡片，然后可选地将审查门控在两者之上。

**因措辞而过度链接。** "最后检查 X"如果 X 是静态配置、文档或源码发现，仍然可以与实现并行。只有当检查依赖于实现结果时，才将其链接在实现之后。

**忘记依赖链接。** 如果任务图说 `research -> implement -> review`，不要将所有任务创建为独立的 ready 卡片。使用父链接，确保 implement/review 在其输入存在之前无法运行。

**重新分配 vs. 新任务。** 如果审查者以"需要修改"阻塞，创建一个从审查者任务链接的**新**任务——不要用严厉的眼神重新运行同一个任务。新任务分配给原始实现者 profile。

**链接的参数顺序。** `kanban_link(parent_id=..., child_id=...)` — 父任务在前。混淆顺序会将错误的任务降级为 `todo`。

**如果形状取决于中间发现，不要预先创建整个任务图。** 如果 T3 的结构取决于 T1 和 T2 的发现，让 T3 作为一个"综合发现"任务存在，其第一步是读取父任务的交接内容并规划其余部分。编排器可以派生编排器。

**Tenant 继承。** 如果你的环境中设置了 `HERMES_TENANT`，在每次 `kanban_create` 调用中传入 `tenant=os.environ.get("HERMES_TENANT")`，以确保子任务保持在同一命名空间中。

## 恢复卡住的 worker

当一个 worker profile 持续崩溃、产生幻觉或被自身错误阻塞时（通常是：错误的模型、缺少 skill、凭据损坏），kanban 仪表板会在任务上标记 ⚠ 徽章，并在抽屉中打开**恢复**部分。三个主要操作：

1. **Reclaim**（或 `hermes kanban reclaim <task_id>`）——立即中止正在运行的 worker 并将任务重置为 `ready`。现有认领 TTL 约为 15 分钟；这是最快的解决路径。
2. **Reassign**（或 `hermes kanban reassign <task_id> <new-profile> --reclaim`）——将任务切换到不同的 profile（此配置上存在的 profile）并让调度器用新 worker 认领它。
3. **更改 profile 模型**——仪表板会打印 `hermes -p <profile> model` 的复制粘贴提示，因为 profile 配置存储在磁盘上；在终端中编辑它，然后 Reclaim 以使用新模型重试。

当 worker 的 `kanban_complete(created_cards=[...])` 声明包含不存在或非该 worker profile 创建的卡片 id 时（门控会阻止完成），或者自由格式摘要引用了无法解析的 `t_<hex>` id 时（建议性文本扫描，非阻塞），会出现幻觉警告。两者都会产生审计事件，即使在恢复操作后也会持久保存——追踪记录保留用于调试。