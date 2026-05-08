---
title: "Kanban Orchestrator"
sidebar_label: "Kanban Orchestrator"
description: "分解手册 + 专家名册约定 + 反诱惑规则，用于通过 Kanban 路由工作的编排器配置文件"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Orchestrator

分解手册 + 专家名册约定 + 反诱惑规则，用于通过 Kanban 路由工作的编排器配置文件。"不要自己做工作"规则和基本生命周期通过 `KANBAN_GUIDANCE` 系统提示块自动注入到每个 kanban 工作者的系统提示中；当您专门扮演编排器角色时，此技能是更深入的手册。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/devops/kanban-orchestrator` |
| 版本 | `2.0.0` |
| 标签 | `kanban`, `multi-agent`, `orchestration`, `routing` |
| 相关技能 | [`kanban-worker`](/docs/user-guide/skills/bundled/devops/devops-kanban-worker) |

## 参考：完整 SKILL.md

:::info
以下是 Hermites 加载此技能时的完整技能定义。这是技能激活时 agent 看到的指令。
:::

# Kanban Orchestrator — 分解手册

> **核心工作者生命周期**（包括 `kanban_create` 扇出模式和"分解，不要执行"规则）通过 `KANBAN_GUIDANCE` 系统提示块自动注入到每个 kanban 流程。当你是其全部工作就是路由的编排器配置文件时，此技能是更深入的手册。

## 何时使用看板（与直接做工作相反）

在以下任一情况下创建 Kanban 任务：

1. **需要多个专家。** 研究 + 分析 + 写作是三个配置文件。
2. **工作应该能在崩溃或重启后存活。** 长时间运行、重复或重要的任务。
3. **用户可能想要插入。** 人类参与任何步骤。
4. **多个子任务可以并行运行。** 扇出以提高速度。
5. **预期需要审查/迭代。** 审阅者循环处理起草者输出。
6. **审计跟踪很重要。** 看板行永久保存在 SQLite 中。

如果以上都不适用 — 这是一个小的单次推理任务 — 使用 `delegate_task` 或直接回答用户。

## 反诱惑规则

您的工作描述是"路由，不要执行"。执行该规则的规则：

- **不要自己执行工作。** 您的受限工具集通常甚至不包括用于实现的 terminal/file/code/web。如果您发现自己"只是快速修复这个" — 停下来，为正确的专家创建一个任务。
- **对于任何具体任务，创建一个 Kanban 任务并分配它。** 每次都是如此。
- **如果没有专家适合，询问用户要创建哪个配置文件。** 不要在"差不多就行"下默认自己做。
- **分解、路由和总结 — 这就是全部工作。**

## 标准专家名册（约定）

除非用户的设置已自定义配置文件，否则假设这些存在。根据用户的实际情况进行调整 — 如果不确定，请询问。

| 配置文件 | 职责 | 典型工作区 |
|---|---|---|
| `researcher` | 阅读来源、收集事实、撰写发现 | `scratch` |
| `analyst` | 综合、排名、去重。消耗多个 `researcher` 输出 | `scratch` |
| `writer` | 以用户的声音起草散文 | `scratch` 或 `dir:` 进入他们的 Obsidian 保险库 |
| `reviewer` | 阅读输出、留下发现、审核批准 | `scratch` |
| `backend-eng` | 编写服务器端代码 | `worktree` |
| `frontend-eng` | 编写客户端代码 | `worktree` |
| `ops` | 运行脚本、管理服务、处理部署 | `dir:` 进入 ops 脚本仓库 |
| `pm` | 编写规格说明、验收标准 | `scratch` |

## 分解手册

### 第一步 — 理解目标

如果目标不明确，请提出澄清问题。廉价提问；生成错误舰队代价高昂。

### 第二步 — 勾勒任务图

在创建任何内容之前，大声勾勒出图表（例如，对用户）。"分析我们是否应该迁移到 Postgres"的示例：

```
T1  researcher        research: Postgres 成本对比现状
T2  researcher        research: Postgres 性能对比现状
T3  analyst           综合迁移建议                父母：T1, T2
T4  writer            起草决策备忘录              父母：T3
```

向用户展示。在创建任何内容之前让他们纠正。

### 第三步 — 创建任务并链接

```python
t1 = kanban_create(
    title="research: Postgres 成本对比现状",
    assignee="researcher",
    body="比较 3 年窗口内的估计基础设施成本、迁移成本和持续运营成本。来源：AWS/GCP 定价、团队时间估算、同行的当前 Postgres 账单。",
    tenant=os.environ.get("HERMES_TENANT"),
)["task_id"]

t2 = kanban_create(
    title="research: Postgres 性能对比现状",
    assignee="researcher",
    body="比较在我们预期数据量下的查询延迟、吞吐量和扩展特性（~500GB，10k QPS 峰值）。来源：基准测试论文、公开案例研究、如果容易的话 pgbench 结果。",
)["task_id"]

t3 = kanban_create(
    title="综合迁移建议",
    assignee="analyst",
    body="阅读 T1（成本）和 T2（性能）的发现。生成一页包含明确权衡和 go/no-go 决定的建议。",
    parents=[t1, t2],
)["task_id"]

t4 = kanban_create(
    title="起草决策备忘录",
    assignee="writer",
    body="将分析师的建议转化为给 CTO 的两页备忘录。使用团队知识库中先前决策备忘录的语调。",
    parents=[t3],
)["task_id"]
```

`parents=[...]` 门控提升 — 子任务保持在 `todo` 直到每个父任务达到 `done`，然后自动提升到 `ready`。无需手动协调；调度器和依赖引擎处理它。

### 第四步 — 完成您自己的任务

如果您是作为任务本身被生成的（例如 `planner` 配置文件被分配了 `T0: "调查 Postgres 迁移"`），用您创建的内容摘要标记完成：

```python
kanban_complete(
    summary="分解为 T1-T4：2 个研究者并行，1 个分析师处理他们的输出，1 个作者处理建议",
    metadata={
        "task_graph": {
            "T1": {"assignee": "researcher", "parents": []},
            "T2": {"assignee": "researcher", "parents": []},
            "T3": {"assignee": "analyst", "parents": ["T1", "T2"]},
            "T4": {"assignee": "writer", "parents": ["T3"]},
        },
    },
)
```

### 第五步 — 向用户报告

用简单的散文告诉他们您创建的内容：

> 我已排队 4 个任务：
> - **T1** (researcher)：成本比较
> - **T2** (researcher)：性能比较，与 T1 并行
> - **T3** (analyst)：综合 T1 + T2 为建议
> - **T4** (writer)：将 T3 转化为 CTO 备忘录
>
> 调度器将立即选取 T1 和 T2。两者完成后 T3 开始。T4 完成后您将收到 gateway ping。使用仪表板或 `hermes kanban tail <id>` 跟踪进度。

## 常见模式

**扇出 + 扇入（研究 → 综合）：** N 个 `researcher` 任务无父任务，一个 `analyst` 任务以所有任务为父任务。

**带门控的管道：** `pm → backend-eng → reviewer`。每个阶段的 `parents=[previous_task]`。审阅者阻塞或完成；如果审阅者阻塞，运营商用反馈解除阻塞并重新生成。

**同一配置文件队列：** 50 个任务，全部分配给 `translator`，它们之间无依赖。调度器串行化 — translator 按优先级顺序处理，在自己的记忆中积累经验。

**人类参与：** 任何任务都可以 `kanban_block()` 等待输入。调度器在 `/unblock` 后重新生成。评论线程携带完整上下文。

## 陷阱

**重新分配 vs 新任务。** 如果审阅者阻塞并说"需要更改"，创建一个从审阅者任务链接的新任务 — 不要用严肃的表情重新运行相同的任务。新任务分配给原始实现者配置文件。

**链接参数顺序。** `kanban_link(parent_id=..., child_id=...)` — 父在前。混淆它们会将错误的任务降级为 `todo`。

**如果形状取决于中间发现，不要预创建整个图。** 如果 T3 的结构取决于 T1 和 T2 的发现，让 T3 作为一个"综合发现"任务存在，其自身第一步是读取父级交接并计划其余部分。编排器可以生成编排器。

**租户继承。** 如果您的环境中设置了 `HERMES_TENANT`，在每个 `kanban_create` 调用上传递 `tenant=os.environ.get("HERMES_TENANT")`，这样子任务保持在相同的命名空间中。
