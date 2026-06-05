---
title: "Kanban Worker — Hermes Kanban worker 的陷阱、示例与边界情况"
sidebar_label: "Kanban Worker"
description: "Hermes Kanban worker 的陷阱、示例与边界情况"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Worker

Hermes Kanban worker 的陷阱、示例与边界情况。生命周期本身会自动注入到每个 worker 的系统 prompt（提示词）中，作为 `KANBAN_GUIDANCE`（来自 `agent/prompt_builder.py`）；当你需要深入了解特定场景时，加载此 skill 即可。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/devops/kanban-worker` |
| 版本 | `2.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `kanban`, `multi-agent`, `collaboration`, `workflow`, `pitfalls` |
| 相关 skill | [`kanban-orchestrator`](/user-guide/skills/bundled/devops/devops-kanban-orchestrator) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Kanban Worker — 陷阱与示例

> 你看到此 skill，是因为 Hermes Kanban 调度器以 `--skills kanban-worker` 参数将你作为 worker 派生——它会为每个被派发的 worker 自动加载。**生命周期**（6 个步骤：orient → work → heartbeat → block/complete）也存在于自动注入到你系统 prompt 中的 `KANBAN_GUIDANCE` 块里。此 skill 是更深层的细节：良好的交接形式、重试诊断、边界情况。

## 工作区处理

你的工作区类型决定了你在 `$HERMES_KANBAN_WORKSPACE` 内部的行为方式：

| 类型 | 含义 | 操作方式 |
|---|---|---|
| `scratch` | 全新的临时目录，仅供你使用 | 自由读写；任务归档后会被 GC 回收。 |
| `dir:<path>` | 共享的持久化目录 | 其他运行实例会读取你写入的内容。将其视为长期状态。路径保证为绝对路径（内核拒绝相对路径）。 |
| `worktree` | 位于已解析路径的 Git worktree | 若 `.git` 不存在，先从主仓库执行 `git worktree add <path> <branch>`，然后 cd 进去正常工作。在此提交工作。 |

## 租户隔离

若 `$HERMES_TENANT` 已设置，则该任务属于某个租户命名空间。在读写持久化内存时，请为内存条目添加租户前缀，以防上下文跨租户泄漏：

- 正确：`business-a: Acme is our biggest customer`
- 错误（会泄漏）：`Acme is our biggest customer`

## 良好的 summary + metadata 形式

`kanban_complete(summary=..., metadata=...)` 的交接方式是下游 worker 读取你工作成果的途径。以下是有效的模式：

**编码任务：**
```python
kanban_complete(
    summary="shipped rate limiter — token bucket, keys on user_id with IP fallback, 14 tests pass",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    },
)
```

**需要人工审查的编码任务（review-required）：**

对于大多数涉及代码变更的任务，在人工审查者过目之前，工作并未真正*完成*。应使用 block 而非 complete，并在 `reason` 前加 `review-required: ` 前缀，以便仪表板将该行标记为待审查。先将结构化元数据（变更文件、测试计数、diff/PR url）写入 comment，因为 `kanban_block` 只携带人类可读的原因——comment 是持久化注释的渠道。审查者可执行 `hermes kanban unblock <id>` 批准（这会携带 comment 线程重新派生你以处理后续事项），或通过另一条 comment 要求修改。

```python
import json

kanban_comment(
    body="review-required handoff:\n" + json.dumps({
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "diff_path": "/path/to/worktree",  # or PR url if pushed
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    }, indent=2),
)
kanban_block(
    reason="review-required: rate limiter shipped, 14/14 tests pass — needs eyes on the user_id/IP fallback choice before merging",
)
```

仅在任务真正终结时使用 `kanban_complete`——例如单行拼写修复、无功能影响的文档变更，或产出物本身即为成果的研究任务。

**研究任务：**
```python
kanban_complete(
    summary="3 competing libraries reviewed; vLLM wins on throughput, SGLang on latency, Tensorrt-LLM on memory efficiency",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**审查任务：**
```python
kanban_complete(
    summary="reviewed PR #123; 2 blocking issues found (SQL injection in /search, missing CSRF on /settings)",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "raw SQL concat"},
            {"severity": "high", "file": "api/settings.py", "issue": "missing CSRF middleware"},
        ],
        "approved": False,
    },
)
```

请将 `metadata` 的结构设计为下游解析器（审查者、聚合器、调度器）无需重新阅读你的文字描述即可直接使用。

## 认领你实际创建的卡片

若你的运行产生了新的 kanban 任务（通过 `kanban_create`），请在 `kanban_complete` 的 `created_cards` 中传入这些 id。内核会验证每个 id 是否存在且由你的 profile 创建；任何幻构的 id 都会导致完成操作被阻断，并附带错误列表说明问题所在，且被拒绝的尝试会永久记录在任务的事件日志中。**只列出你从成功的 `kanban_create` 返回值中捕获的 id——绝不凭空捏造 id，绝不粘贴来自早期运行的 id，绝不认领其他 worker 创建的卡片。**

```python
# 正确 — 捕获返回值，然后认领。
c1 = kanban_create(title="remediate SQL injection", assignee="security-worker")
c2 = kanban_create(title="fix CSRF middleware", assignee="web-worker")

kanban_complete(
    summary="Review done; spawned remediations for both findings.",
    metadata={"pr_number": 123, "approved": False},
    created_cards=[c1["task_id"], c2["task_id"]],
)
```

```python
# 错误 — 认领没有捕获返回值的 id。
kanban_complete(
    summary="Created remediation cards t_a1b2c3d4, t_deadbeef",  # 幻构
    created_cards=["t_a1b2c3d4", "t_deadbeef"],                   # → 门控拒绝
)
```

若 `kanban_create` 调用失败（异常、tool_error），则卡片未被创建——不要为其包含幻构 id。重试创建，或省略该 id 并在 summary 中说明失败情况。散文扫描阶段也会捕获你自由格式 summary 中无法解析的 `t_<hex>` 引用；这些不会阻断完成操作，但会在仪表板的任务上显示为建议性警告。

## 能快速得到回应的 block 原因

差：`"stuck"` — 人类没有任何上下文。

好：一句话说明你需要的具体决策。将更长的上下文作为 comment 留下。

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="Full context: I have user IPs from Cloudflare headers but some users are behind NATs with thousands of peers. Keying on IP alone causes false positives.",
)
kanban_block(reason="Rate limit key choice: IP (simple, NAT-unsafe) or user_id (requires auth, skips anonymous endpoints)?")
```

block 消息是仪表板/gateway 通知器中显示的内容。comment 是人类打开任务时阅读的深层上下文。

## 值得发送的 heartbeat

好的 heartbeat 应说明进度：`"epoch 12/50, loss 0.31"`、`"scanned 1.2M/2.4M rows"`、`"uploaded 47/120 videos"`。

差的 heartbeat：`"still working"`、空 notes、亚秒级间隔。最多每隔几分钟发送一次；对于约 2 分钟以内的任务可完全跳过。

## 重试场景

若你打开任务后 `kanban_show` 返回的 `runs: [...]` 中包含一个或多个已关闭的运行，说明你是一次重试。先前运行的 `outcome` / `summary` / `error` 会告诉你哪里出了问题。不要重复那条路径。典型的重试诊断：

- `outcome: "timed_out"` — 上次尝试达到了 `max_runtime_seconds`。你可能需要将工作分块或缩短。
- `outcome: "crashed"` — OOM 或段错误。减少内存占用。
- `outcome: "spawn_failed"` + `error: "..."` — 通常是 profile 配置问题（缺少凭证、错误的 PATH）。通过 `kanban_block` 询问人类，而不是盲目重试。
- `outcome: "reclaimed"` + `summary: "task archived..."` — 操作员在上次运行期间将任务归档；你可能根本不应该在运行，请仔细检查状态。
- `outcome: "blocked"` — 上次尝试被阻断；解除阻断的 comment 现在应该已在线程中。

## 禁止事项

- 不要用 `delegate_task` 替代 `kanban_create`。`delegate_task` 用于你的运行内部的短期推理子任务；`kanban_create` 用于跨 agent 的、超出单次 API 循环的交接。
- 不要修改 `$HERMES_KANBAN_WORKSPACE` 之外的文件，除非任务正文明确要求。
- 不要创建分配给自己的后续任务——分配给合适的专家。
- 不要完成一个你实际上没有完成的任务。改为 block 它。

## 陷阱

**任务状态可能在调度与启动之间发生变化。** 从调度器认领任务到你的进程实际启动之间，任务可能已被 block、重新分配或归档。始终先执行 `kanban_show`。若其报告 `blocked` 或 `archived`，请停止——你不应该在运行。

**工作区可能存在过期产物。** 尤其是 `dir:` 和 `worktree` 工作区可能包含来自先前运行的文件。阅读 comment 线程——它通常会解释你为何再次运行以及工作区处于何种状态。

**当指导已可用时，不要依赖 CLI。** `kanban_*` 工具可在所有终端后端（Docker、Modal、SSH）上工作。从你的终端工具执行 `hermes kanban <verb>` 在容器化后端中会失败，因为 CLI 未安装在那里。如有疑问，使用工具。

## CLI 回退（用于脚本）

每个工具都有对应的 CLI 等价命令，供人工操作员和脚本使用：
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- 等等。

在 agent 内部使用工具；CLI 供终端前的人类使用。