---
sidebar_position: 3
sidebar_label: "Git Worktrees"
title: "Git Worktrees"
description: "使用 git worktree 和隔离检出在同一个仓库上安全地运行多个 Hermes agent"
---

# Git Worktrees

Hermes Agent 常用于大型、长寿命的仓库。当您想要：

- 在同一个项目上**并行运行多个 agent**，或
- 将实验性重构与主分支隔离，

Git **worktrees** 是为每个 agent 提供自己检出的最安全方式，而无需复制整个仓库。

本页面展示如何将 worktree 与 Hermes 结合，使每个会话都有一个干净、隔离的工作目录。

## 为什么将 Worktree 与 Hermes 一起使用？

Hermes 将**当前工作目录**视为项目根目录：

- CLI：运行 `hermes` 或 `hermes chat` 的目录
- 消息传递 gateway：由 `MESSAGING_CWD` 设置的目录

如果您在**同一个检出**中运行多个 agent，它们的更改可能会相互干扰：

- 一个 agent 可能删除或重写另一个正在使用的文件。
- 很难理解哪些更改属于哪个实验。

使用 worktree，每个 agent 获得：

- 自己的**分支和工作目录**
- 自己的**检查点管理器历史**，用于 `/rollback`

另请参阅：[检查点和 /rollback](./checkpoints-and-rollback.md)。

## 快速开始：创建 Worktree

从包含 `.git/` 的主仓库中，为功能分支创建新的 worktree：

```bash
# 从主仓库根目录
cd /path/to/your/repo

# 在 ../repo-feature 创建新分支和 worktree
git worktree add ../repo-feature feature/hermes-experiment
```

这会创建：

- 一个新目录：`../repo-feature`
- 一个新分支：`feature/hermes-experiment` 在该目录中检出

现在您可以 `cd` 进入新的 worktree 并在那里运行 Hermes：

```bash
cd ../repo-feature

# 在 worktree 中启动 Hermes
hermes
```

Hermes 会：

- 将 `../repo-feature` 视为项目根目录。
- 使用该目录进行上下文文件、代码编辑和工具。
- 使用**独立的检查点历史**，用于 `/rollback`，作用域限定在此 worktree。

## 并行运行多个 Agent

您可以创建多个 worktree，每个都有自己的分支：

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

在单独的终端中：

```bash
# 终端 1
cd ../repo-experiment-a
hermes

# 终端 2
cd ../repo-experiment-b
hermes
```

每个 Hermes 进程：

- 在自己的分支上工作（`feature/hermes-a` vs `feature/hermes-b`）。
- 在不同的 shadow repo hash 下写入检查点（从 worktree 路径派生）。
- 可以独立使用 `/rollback` 而不影响另一个。

这在以下情况下特别有用：

- 运行批量重构。
- 尝试同一任务的不同方法。
- 在同一个上游仓库上配对 CLI + gateway 会话。

## 安全清理 Worktree

当您完成实验时：

1. 决定是保留还是丢弃工作。
2. 如果要保留：
   - 像往常一样将分支合并到主分支。
3. 删除 worktree：

```bash
cd /path/to/your/repo

# 删除 worktree 目录及其引用
git worktree remove ../repo-feature
```

注意：

- `git worktree remove` 除非您强制执行，否则会拒绝删除有未提交更改的 worktree。
- 删除 worktree **不会**自动删除分支；您可以使用常规 `git branch` 命令删除或保留分支。
- 当您删除 worktree 时，`~/.hermes/checkpoints/` 下的 Hermes 检查点数据不会自动清理，但它通常非常小。

## 最佳实践

- **每个 Hermes 实验一个 worktree**
  - 为每个实质性更改创建专用分支/worktree。
  - 这使 diff 集中且 PR 较小且可审查。
- **用实验名称命名分支**
  - 例如 `feature/hermes-checkpoints-docs`、`feature/hermes-refactor-tests`。
- **频繁提交**
  - 使用 git 提交作为高级里程碑。
  - 在中间使用 [检查点和 /rollback](./checkpoints-and-rollback.md) 作为工具驱动编辑的安全网。
- **使用 worktree 时避免从裸仓库根目录运行 Hermes**
  - 优先使用 worktree 目录，这样每个 agent 都有明确的范围。

## 使用 `hermes -w`（自动 Worktree 模式）

Hermes 有一个内置的 `-w` 标志，**自动创建带有自己分支的临时 git worktree**。您不需要手动设置 worktree——只需 `cd` 进入您的仓库并运行：

```bash
cd /path/to/your/repo
hermes -w
```

Hermes 会：

- 在您仓库内的 `.worktrees/` 下创建一个临时 worktree。
- 检出隔离的分支（例如 `hermes/hermes-<hash>`）。
- 在该 worktree 内运行完整的 CLI 会话。

这是获得 worktree 隔离的最简单方法。您也可以将其与单次查询结合：

```bash
hermes -w -q "Fix issue #123"
```

对于并行 agent，打开多个终端并在每个中运行 `hermes -w`——每次调用都会自动获得自己的 worktree 和分支。

## 综合使用

- 使用 **git worktree** 为每个 Hermes 会话提供自己的干净检出。
- 使用 **分支** 捕获实验的高级历史。
- 使用 **检查点 + `/rollback`** 在每个 worktree 中从错误编辑中恢复。

这种组合为您提供：

- 强有力的保证，不同的 agent 和实验不会相互影响。
- 快速迭代周期，从错误编辑中轻松恢复。
- 干净、可审查的 pull requests。
