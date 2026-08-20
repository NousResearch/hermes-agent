---
title: "Kanban Codex 实现通道"
sidebar_label: "Kanban Codex 实现通道"
description: "当 Hermes Kanban Worker 需要将 Codex CLI 作为独立实现通道运行，同时 Hermes 保持任务生命周期、协调、测试和交接的所有权时使用"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Codex 实现通道

当 Hermes Kanban Worker 需要将 Codex CLI 作为独立实现通道运行，同时 Hermes 保持任务生命周期、协调、测试和交接的所有权时使用。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/autonomous-ai-agents/kanban-codex-lane` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `kanban`、`codex`、`worktrees`、`autonomous-agents`、`prediction-market-bot` |
| 相关技能 | [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex)、[`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Kanban Codex 实现通道

## 概述

本技能定义了 Kanban Worker 的轻量级 Hermes+Codex 双通道约定。Hermes 始终是任务所有者：它调用 `kanban_show`，决定 Codex 是否适合，创建或选择隔离工作空间，启动和监控 Codex，协调任何差异，运行验证，并写入最终的 `kanban_complete` 或 `kanban_block` 交接。Codex 仅作为输入通道。Codex 输出不是任务完成信号，不是受信任的审核者，也不允许直接写入持久的 Kanban 状态。

该约定的存在使得 Hermes Worker 可以使用 Codex 进行有边界的实现辅助，而无需更改调度器。调度器仍必须生成 Hermes Worker。Worker 可以选择在其自身运行中生成 Codex，然后在独立审核和测试后接受、部分接受或拒绝该通道。

## 何时使用

在以下条件全部满足时使用 Codex 通道：

- Kanban 任务是具有明确验收标准的编码、重构、文档、测试或机械迁移任务。
- 有界差异可以在一次运行中由 Hermes 评估。
- 仓库可以在隔离的 git worktree/分支中复制或检出。
- Hermes 可以在 Codex 退出后自行运行相关测试。
- 提示可以声明所有安全约束和不得更改的文件。

在以下任一条件满足时不要使用 Codex 通道：

- 任务需要 Kanban 正文中尚未体现的人类判断。
- Worker 缺少仓库访问权限、Codex 认证或协调结果的时间。
- 更改涉及密钥、凭据存储、私有用户数据或生产订单录入系统。
- 直接进行小编辑比生成另一个代理更快更安全。
- 任务仅限研究，应产生书面交接而非差异。
- Worker 仅基于 Codex 自报就倾向于标记完成。

## 所有权规则

1. Hermes 拥有 Kanban 生命周期。Codex 绝不能调用 `kanban_complete`、`kanban_block`、`kanban_create`、gateway messaging 或任何 Hermes board CLI 作为 Worker 的替代。
2. Hermes 拥有最终验收。将 Codex commits/diffs 视为不受信任的补丁，直到经过审核和验证。
3. Hermes 拥有测试执行权。Codex 可以运行测试，但这些运行是建议性的；需要使用仓库的规范包装器从 Hermes 重复所需的验证。
4. Hermes 拥有安全控制权。如果 Codex 更改了安全边界、风险门控、实时交易行为或密钥处理，即使测试通过也要拒绝该通道。
5. Hermes 拥有清理权。终止卡住的 Codex 进程并在不再需要时移除临时 worktree。

## 必需的 Worktree 和分支模式

永远不要在共享的脏检出中直接运行 Codex。使用将通道与 Kanban 任务关联的分支/worktree 名称，并保持不受信任的编辑隔离。

推荐变量：

```bash
TASK_ID="${HERMES_KANBAN_TASK:-t_manual}"
REPO="/path/to/repo"
BASE="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
SAFE_TASK="$(printf '%s' "$TASK_ID" | tr -cd '[:alnum:]_-')"
BRANCH="codex/${SAFE_TASK}/$(date -u +%Y%m%d%H%M%S)"
WORKTREE="/tmp/${SAFE_TASK}-codex-lane"
```

创建隔离通道：

```bash
git -C "$REPO" fetch --all --prune
git -C "$REPO" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
git -C "$WORKTREE" status --short --branch
```

如果当前 Kanban 工作空间已经是为此任务创建的隔离 git worktree，仅当 `git status --short` 除有意的 Hermes 编辑外保持干净时，你才可以在其内部创建同级 Codex 分支。否则，创建一个单独的临时 worktree，并在协调后 cherry-pick 或复制已接受的提交回来。

协调后清理：

```bash
git -C "$REPO" worktree remove "$WORKTREE"
git -C "$REPO" branch -D "$BRANCH"  # 仅在已接受提交被复制/cherry-pick 或有意拒绝后
```

如果 worktree 需要作为审核制品保留，则保留它；在 `codex_lane.artifacts` 中记录它，并在交接中提及。

## Codex 能力检查

在生成 Codex 之前运行这些检查。缺少 Codex 是跳过通道的正常原因，而不是任务阻塞器——如果 Hermes 可以直接完成任务。

```bash
command -v codex
codex --version
codex features list | grep -i goals || true
```

如果需要 `/goal` 支持，仅在检查可用性后启用或启动该功能标志：

```bash
codex features enable goals || true
codex --enable goals --version
```

认证可通过 `OPENAI_API_KEY` 或 Codex CLI OAuth 状态（通常是 `~/.codex/auth.json`）进行。不要打印 token 文件。缺少 `OPENAI_API_KEY` 不是认证不可用的证明。

## 模式选择

对 Codex 应自行退出的有界一次性编辑使用 `codex exec`：

```python
terminal(
    command="codex exec --full-auto '$(cat /tmp/codex_prompt.md)'",
    workdir=WORKTREE,
    background=True,
    pty=True,
    notify_on_complete=True,
)
```

仅对受益于持久目标跟踪的更广泛多步骤工作使用 Codex `/goal`。在 PTY/tmux 会话中交互式启动，或在该功能默认禁用时使用 `codex --enable goals`。保持目标自包含：仓库路径、任务 ID、安全约束、允许范围、验收标准、测试和提交期望。

粘贴到 Codex 中的示例 `/goal` 目标文本：

```text
/goal Work in this repository only: <WORKTREE>. Task: <TASK_ID> <TITLE>.
Hermes owns the Kanban lifecycle; do not call Hermes kanban tools or messaging.
Create small commits on branch <BRANCH>. Follow the PMB safety constraints in the prompt.
Run the requested verification commands and report exact outputs. Stop after producing a diff and summary.
```

不要对 prediction-market-bot 或安全敏感仓库使用 `--yolo`。在隔离 worktree 内优先使用 `--full-auto`，然后依赖 Hermes 协调。

## 提示构建

对 prediction-market-bot 工作使用链接模板 `templates/pmb-codex-lane-prompt.md`。对于其他仓库，保持相同结构并将 PMB 特定安全块替换为仓库特定不变量。

每个 Codex 提示必须包含：

- `task_id`、标题和完整的 Kanban 验收标准。
- 仓库路径、worktree 路径、分支名称和允许的文件范围。
- 显式声明：Hermes 拥有 Kanban 生命周期；Codex 仅作为输入通道。
- 所需输出：简洁摘要、更改的文件、提交、运行的测试和已知风险。
- 禁止操作：密钥访问、外部消息、board 变更、无关重构（除非需要）。
- Codex 可以运行的验证命令以及 Hermes 之后将运行的命令。

对于 PMB，逐字包含这些强制安全约束：

```text
PMB safety constraints:
- live-SIM is paper-only; do not add or enable live REST order entry.
- Never use market orders.
- Do not add execution crossing or bypass price/risk checks.
- Do not fake passive fills, fills, PnL, order states, or reconciliation evidence.
- Do not weaken risk gates, limits, kill switches, or fail-closed behavior.
- Keep research/selection outside the C++ hot path unless explicitly requested.
- Do not read, print, write, or require secrets/tokens/credentials.
```

## 监控、超时和终止行为

在后台使用 PTY 和完成通知启动长 Codex 通道：

```python
result = terminal(
    command="codex exec --full-auto '$(cat /tmp/codex_prompt.md)'",
    workdir=WORKTREE,
    background=True,
    pty=True,
    notify_on_complete=True,
)
session_id = result["session_id"]
```

不干扰地监控：

```python
process(action="poll", session_id=session_id)
process(action="log", session_id=session_id, limit=200)
process(action="wait", session_id=session_id, timeout=300)
```

对于超过两分钟的通道，每隔几分钟发送一次 Kanban 心跳，例如 `kanban_heartbeat(note="Codex lane running in <WORKTREE>; waiting for tests/diff")`。

终止条件：

- 在任务剩余运行时间预算内没有有用的输出。
- Codex 请求密钥、生产凭据或外部权限。
- Codex 尝试修改 worktree 外的文件。
- Codex 开始无关的重写或依赖变更。
- Codex 在 Worker 超时前仍在运行且没有安全的部分制品。

终止命令：

```python
process(action="kill", session_id=session_id)
```

终止后，检查 `git status --short`，仅在安全时保留有用的补丁，并记录 `codex_lane.result: timed_out` 或 `rejected` 以及具体的 `rejected_reason`。

## 协调清单

Hermes 在接受任何 Codex 通道结果之前必须执行此清单：

- [ ] `git -C <WORKTREE> status --short --branch` 仅显示预期文件。
- [ ] `git -C <WORKTREE> diff --stat` 和 `git diff` 已由 Hermes 审核。
- [ ] 不包含密钥、凭据、生成的缓存、无关数据或本地制品。
- [ ] PMB 安全约束得到保留：没有实时 REST 订单录入、没有市价单、没有执行交叉、没有伪造被动成交/PnL、没有风险门控弱化、没有密钥。
- [ ] Codex 提交足够小，可以干净地 cherry-pick 或 squash。
- [ ] Hermes 自行运行了规范测试，对 Hermes Agent 使用 `scripts/run_tests.sh`，对其他仓库使用仓库的文档化包装器。
- [ ] Codex 运行的测试与 Hermes 运行的测试分开列出。
- [ ] 已接受的提交/差异已应用于 Hermes 拥有的工作空间/分支。
- [ ] 被拒绝或部分完成的工作有具体原因和制品路径（如果有用的话）。

接受结果：

- `accepted`：Codex diff/commits 已审核、应用并验证。
- `partial`：部分 Codex 工作在编辑或 cherry-pick 后被接受；拒绝的部分有文档记录。
- `rejected`：没有接受 Codex 更改；原因有文档记录。
- `timed_out`：Codex 超出通道预算；有用的制品可能存在也可能不存在。

## kanban_complete 元数据模式

对于考虑过通道的每个任务，在 `metadata.codex_lane` 下包含此对象。如果未使用 Codex，设置 `used: false` 并在 `rejected_reason` 或同级 `notes` 字段中解释原因。

```json
{
  "codex_lane": {
    "used": true,
    "mode": "exec | goal | skipped",
    "worktree": "/absolute/path/to/codex/worktree",
    "branch": "codex/t_caa69668/20260508100000",
    "command": "codex exec --full-auto ...",
    "result": "accepted | rejected | partial | timed_out",
    "accepted_commits": ["<sha1>", "<sha2>"],
    "rejected_reason": "empty when fully accepted; otherwise concrete reason",
    "tests_run": [
      {"command": "scripts/run_tests.sh tests/tools/test_x.py", "exit_code": 0, "owner": "hermes"},
      {"command": "codex-reported: npm test", "exit_code": 0, "owner": "codex"}
    ],
    "artifacts": ["/absolute/path/to/log-or-patch"]
  }
}
```

对于有意跳过 Codex 的任务：

```json
{
  "codex_lane": {
    "used": false,
    "mode": "skipped",
    "worktree": null,
    "branch": null,
    "command": null,
    "result": "rejected",
    "accepted_commits": [],
    "rejected_reason": "Direct Hermes edit was smaller and safer than spawning Codex.",
    "tests_run": [],
    "artifacts": []
  }
}
```

## 常见陷阱

1. 将 Codex 自报视为验证。始终检查差异并从 Hermes 重新运行测试。
2. 在用户的脏主检出中运行 Codex。始终在 worktree/分支中隔离。
3. 让 Codex 拥有 Kanban。Codex 可以总结进度，但 Hermes 编写 board 状态。
4. 忘记提示中的 PMB 安全不变量。缺少安全文本是通道设置失败。
5. 使用 `/goal` 进行快速编辑。除非需要持久的多步骤继续，否则优先使用 `codex exec`。
6. 在没有记录原因的情况下终止卡住的通道。`rejected_reason` 必须解释决策。
7. 接受广泛的无关清理（因为测试通过）。拒绝或仅 cherry-pick 有范围的更改。

## 验证清单

- [ ] Codex 被跳过，或仅在 `command -v codex`、`codex --version` 和可选的目标功能检查后启动。
- [ ] Codex 仅在隔离的 worktree/分支中运行。
- [ ] 提示包含任务范围、所有权规则、适用时的 PMB 安全约束和验证命令。
- [ ] Hermes 审核了 `git diff` 和安全敏感文件。
- [ ] Hermes 独立运行了规范测试。
- [ ] `kanban_complete.metadata.codex_lane` 遵循上述模式。
- [ ] 临时进程和不必要的 worktree 已被清理。
