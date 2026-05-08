---
title: "Kanban Worker — Hermites Kanban 工作者的陷阱、示例和边缘情况"
sidebar_label: "Kanban Worker"
description: "Hermites Kanban 工作者的陷阱、示例和边缘情况"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Worker

Hermites Kanban 工作者的陷阱、示例和边缘情况。生命周期本身作为 KANBAN_GUIDANCE（来自 agent/prompt_builder.py）自动注入到每个工作者的系统提示中；当您想要更深入地了解特定场景时，加载此技能。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/devops/kanban-worker` |
| 版本 | `2.0.0` |
| 标签 | `kanban`, `multi-agent`, `collaboration`, `workflow`, `pitfalls` |
| 相关技能 | [`kanban-orchestrator`](/docs/user-guide/skills/bundled/devops/devops-kanban-orchestrator) |

## 参考：完整 SKILL.md

:::info
以下是 Hermites 加载此技能时的完整技能定义。这是技能激活时 agent 看到的指令。
:::

# Kanban Worker — 陷阱和示例

> 您看到此技能是因为 Hermites Kanban 调度器将您作为带有 `--skills kanban-worker` 的工作者生成 — 每个调度的工作者会自动加载它。**生命周期**（6 个步骤：定向 → 工作 → 心跳 → 阻塞/完成）也位于作为系统提示自动注入的 `KANBAN_GUIDANCE` 块中。此技能是更深入的细节：好的交接形状、重试诊断、边缘情况。

## 工作区处理

您的工作区类型决定了您应该如何处理 `$HERMES_KANBAN_WORKSPACE`：

| 类型 | 内容 | 如何工作 |
|---|---|---|
| `scratch` | 新鲜 tmp 目录，仅您使用 | 自由读写；任务归档时被 GC'd。 |
| `dir:<path>` | 共享持久目录 | 其他运行将读取您写的内容。将其视为长期状态。路径保证是绝对的（内核拒绝相对路径）。 |
| `worktree` | 解析路径处的 Git worktree | 如果 `.git` 不存在，先从主仓库运行 `git worktree add <path> <branch>`，然后正常 cd 并工作。在此提交。 |

## 租户隔离

如果设置了 `$HERMES_TENANT`，则任务属于租户命名空间。在读取或写入持久内存时，在内存条目前加上租户前缀，以免上下文泄漏到租户之间：

- 好：`business-a: Acme 是我们最大的客户`
- 坏（泄漏）：`Acme 是我们最大的客户`

## 好的摘要 + 元数据形状

`kanban_complete(summary=..., metadata=...)` 交接是下游工作者读取您所做工作的方式。有效的模式：

**编码任务：**
```python
kanban_complete(
    summary="已上线限速器 — 令牌桶，按 user_id 键控，带 IP 后备，14 个测试通过",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id 主要，未认证请求的 IP 后备"],
    },
)
```

**研究任务：**
```python
kanban_complete(
    summary="审查了 3 个竞争库；vLLM 在吞吐量上胜出，SGLang 在延迟上胜出，Tensorrt-LLM 在内存效率上胜出",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**审阅任务：**
```python
kanban_complete(
    summary="审查了 PR #123；发现 2 个阻塞问题（/search 中的 SQL 注入，/settings 缺少 CSRF）",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "原始 SQL 连接"},
            {"severity": "high", "file": "api/settings.py", "issue": "缺少 CSRF 中间件"},
        ],
        "approved": False,
    },
)
```

塑造 `metadata` 以便下游解析器（审阅者、聚合器、调度器）可以在不重读您的散文的情况下使用它。

## 快速回答的阻塞原因

坏：`"卡住了"` — 人类没有上下文。

好：一句话命名您需要的具体决定。将更长的上下文留作评论。

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="完整上下文：我有来自 Cloudflare 标头的用户 IP，但一些用户处于 NAT 后面，有数千个对等点。单独按 IP 键控会导致误报。",
)
kanban_block(reason="限速键选择：IP（简单，NAT 不安全）还是 user_id（需要认证，跳过匿名端点）？")
```

阻塞消息出现在仪表板 / gateway 通知器中。评论是人类打开任务时阅读的更深入上下文。

## 值得发送的心跳

好的心跳命名进度：`"epoch 12/50, loss 0.31"`、`"已扫描 1.2M/2.4M 行"`、`"已上传 47/120 个视频"`。

坏的心跳：`"仍在工作"`、空注释、亚秒间隔。最多每几分钟一次；对于约 2 分钟以下的任务完全跳过。

## 重试场景

如果您打开任务且 `kanban_show` 返回 `runs: [...]` 并带有一个或多个已关闭的运行，您是重试。先前运行的 `outcome` / `summary` / `error` 告诉您什么不起作用。不要重复那条路径。典型重试诊断：

- `outcome: "timed_out"` — 先前尝试达到 `max_runtime_seconds`。您可能需要将工作分块或缩短。
- `outcome: "crashed"` — OOM 或 segfault。减少内存占用。
- `outcome: "spawn_failed"` + `error: "..."` — 通常是配置文件问题（缺少凭证、PATH 错误）。通过 `kanban_block` 询问人类而不是盲目重试。
- `outcome: "reclaimed"` + `summary: "task archived..."` — 调度器在先前运行下归档了任务；您可能根本不应该运行，仔细检查状态。
- `outcome: "blocked"` — 先前尝试阻塞；解除阻塞评论应该在帖子中。

## 禁止事项

- 不要将 `delegate_task` 作为 `kanban_create` 的替代品。`delegate_task` 用于 YOUR 运行中的短期推理子任务；`kanban_create` 用于跨 agent 交接，超越一个 API 循环。
- 除非任务正文说要，否则不要修改 `$HERMES_KANBAN_WORKSPACE` 之外的文件。
- 不要创建分配给自己的后续任务 — 分配给正确的专家。
- 不要完成您实际未完成的任务。改为阻塞。

## 陷阱

**任务状态可能在调度和启动之间更改。** 在调度器声称和您的进程实际启动之间，任务可能已被阻塞、重新分配或归档。始终首先 `kanban_show`。如果它报告 `blocked` 或 `archived`，停止 — 您不应该运行。

**工作区可能有陈旧 artifacts。** 特别是 `dir:` 和 `worktree` 工作区可能包含先前运行的文件。阅读评论线程 — 它通常解释为什么您再次运行以及工作区处于什么状态。

**当 guidance 可用时不要依赖 CLI。** `kanban_*` 工具跨所有终端后端工作（Docker、Modal、SSH）。来自终端工具的 `hermes kanban <verb>` 在容器化后端中将失败，因为 CLI 未安装在那里。如有疑问，使用工具。

## CLI 后备（用于脚本）

每个工具都有等效的 CLI 用于人类操作员和脚本：
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- 等。

从 agent 内部使用工具；CLI 为终端的人类存在。
