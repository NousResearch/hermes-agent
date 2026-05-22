---
sidebar_position: 3
sidebar_label: "Git Worktrees"
title: "Git Worktrees"
description: "Run multiple Hermes agents safely on the same repository using git worktrees and isolated checkouts"
---

# Git Worktrees

Hermes Agent 经常用于大型、长期维护的仓库。当你需要：

- 在同一个项目上**并行运行多个 agent**，或
- 把实验性重构与主分支隔离开来时，

Git **worktree** 是最安全的方式。它可以让每个 agent 都拥有自己独立的 checkout，而不必复制整个仓库。

本页演示如何把 worktree 和 Hermes 结合起来，让每个会话都拥有干净、隔离的工作目录。

## 为什么要和 Hermes 一起使用 worktree？

Hermes 把**当前工作目录**当作项目根目录：

- CLI：你运行 `hermes` 或 `hermes chat` 的目录
- 消息网关：由 `MESSAGING_CWD` 指定的目录

如果多个 agent 使用**同一个 checkout**，它们的改动会互相干扰：

- 一个 agent 可能删除或重写另一个 agent 正在使用的文件。
- 也更难判断哪部分改动属于哪个实验。

使用 worktree 后，每个 agent 都会拥有：

- **自己的分支和工作目录**
- **自己的 Checkpoint Manager 历史**，用于 `/rollback`

另见：[Checkpoints and /rollback](./checkpoints-and-rollback.md)。

## 快速开始：创建 worktree

从主仓库（包含 `.git/`）中为一个功能分支创建新的 worktree：

```bash
# 在主仓库根目录中执行
cd /path/to/your/repo

# 在 ../repo-feature 中创建一个新分支和 worktree
git worktree add ../repo-feature feature/hermes-experiment
```

这样会创建：

- 一个新目录：`../repo-feature`
- 一个新分支：`feature/hermes-experiment`，并在这个目录中检出

现在你可以进入新的 worktree，并在其中运行 Hermes：

```bash
cd ../repo-feature

# 在 worktree 中启动 Hermes
hermes
```

Hermes 会：

- 把 `../repo-feature` 视为项目根目录。
- 使用这个目录来读取上下文文件、编辑代码和调用工具。
- 为 `/rollback` 使用**独立的 checkpoint 历史**，且该历史只作用于这个 worktree。

## 并行运行多个 agent

你可以创建多个 worktree，每个都有自己的分支：

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

在不同终端里：

```bash
# 终端 1
cd ../repo-experiment-a
hermes

# 终端 2
cd ../repo-experiment-b
hermes
```

每个 Hermes 进程都会：

- 在自己的分支上工作（`feature/hermes-a` 与 `feature/hermes-b`）。
- 把 checkpoint 写到不同的 shadow repo hash 下（由 worktree 路径派生）。
- 可以彼此独立地使用 `/rollback`，而不会影响另一个。

这在以下场景尤其有用：

- 批量重构。
- 针对同一任务尝试不同方案。
- 在同一个上游仓库上并行使用 CLI + 网关会话。

## 安全清理 worktree

完成实验后：

1. 决定保留还是丢弃这次工作。
2. 如果要保留：
   - 按常规把分支合并回主分支。
3. 删除 worktree：

```bash
cd /path/to/your/repo

# 删除 worktree 目录及其引用
git worktree remove ../repo-feature
```

注意事项：

- 如果 worktree 里有未提交改动，`git worktree remove` 会拒绝删除，除非你强制执行。
- 删除 worktree **不会** 自动删除分支；你可以继续保留分支，或用普通 `git branch` 命令删除它。
- 删除 worktree 后，`~/.hermes/checkpoints/` 下与之对应的 Hermes checkpoint 数据不会自动清理，但通常体积很小。

## 最佳实践

- **每个 Hermes 实验一个 worktree**
  - 为每个大改动创建专属分支 / worktree。
  - 这样 diff 更集中，PR 更小，也更容易 review。
- **分支名体现实验内容**
  - 例如 `feature/hermes-checkpoints-docs`、`feature/hermes-refactor-tests`。
- **频繁提交**
  - 用 git commit 记录高层里程碑。
  - 中间的工具驱动编辑则用 [checkpoints 和 /rollback](./checkpoints-and-rollback.md) 兜底。
- **使用 worktree 时避免从裸仓库根目录运行 Hermes**
  - 优先进入 worktree 目录，这样每个 agent 的作用域更清晰。

## 使用 `hermes -w`（自动 worktree 模式）

Hermes 自带 `-w` 参数，可以**自动创建一个临时 git worktree** 并带上自己的分支。你不必手工设置 worktree - 只要进入仓库并运行：

```bash
cd /path/to/your/repo
hermes -w
```

Hermes 会：

- 在仓库内的 `.worktrees/` 下创建一个临时 worktree。
- 检出一个隔离分支（例如 `hermes/hermes-<hash>`）。
- 在该 worktree 中运行完整 CLI 会话。

这是最简单的 worktree 隔离方式。你也可以把它和单次提问结合起来：

```bash
hermes -w -q "Fix issue #123"
```

如果你要并行运行多个 agent，可以打开多个终端分别运行 `hermes -w` - 每次调用都会自动得到自己的 worktree 和分支。

## 放在一起看

- 使用 **git worktree** 给每个 Hermes 会话一个干净的 checkout。
- 使用 **branch** 记录实验的高层历史。
- 使用 **checkpoints + `/rollback`** 在每个 worktree 内恢复失误。

这个组合可以给你：

- 强保证，避免不同 agent 和实验互相踩踏。
- 快速迭代，并且坏改动也能轻松恢复。
- 干净、可 review 的 pull request。