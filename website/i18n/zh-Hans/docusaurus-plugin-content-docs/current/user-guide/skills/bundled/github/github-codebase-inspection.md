---
title: "代码库检查 — 使用 pygount 检查代码库：代码行数、语言、占比"
sidebar_label: "代码库检查"
description: "使用 pygount 检查代码库：代码行数、语言、占比"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 代码库检查

使用 pygount 检查代码库：代码行数、语言、占比。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/github/codebase-inspection` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `LOC`, `Code Analysis`, `pygount`, `Codebase`, `Metrics`, `Repository` |
| 相关 skill | [`github-repo-management`](/user-guide/skills/bundled/github/github-github-repo-management) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 使用 pygount 进行代码库检查

使用 `pygount` 分析仓库的代码行数、语言分布、文件数量及代码与注释的比例。

## 使用场景

- 用户请求统计 LOC（lines of code，代码行数）
- 用户需要仓库的语言分布情况
- 用户询问代码库的规模或组成
- 用户需要代码与注释的比例
- 一般性的"这个仓库有多大"问题

## 前置条件

```bash
pip install --break-system-packages pygount 2>/dev/null || pip install pygount
```

## 1. 基本摘要（最常用）

获取包含文件数量、代码行数和注释行数的完整语言分布：

```bash
cd /path/to/repo
pygount --format=summary \
  --folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,.eggs,*.egg-info" \
  .
```

**重要：** 始终使用 `--folders-to-skip` 排除依赖/构建目录，否则 pygount 会遍历这些目录，导致运行时间极长甚至卡死。

## 2. 常用目录排除项

根据项目类型进行调整：

```bash
# Python 项目
--folders-to-skip=".git,venv,.venv,__pycache__,.cache,dist,build,.tox,.eggs,.mypy_cache"

# JavaScript/TypeScript 项目
--folders-to-skip=".git,node_modules,dist,build,.next,.cache,.turbo,coverage"

# 通用兜底
--folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,vendor,third_party"
```

## 3. 按特定语言过滤

```bash
# 仅统计 Python 文件
pygount --suffix=py --format=summary .

# 仅统计 Python 和 YAML
pygount --suffix=py,yaml,yml --format=summary .
```

## 4. 逐文件详细输出

```bash
# 默认格式显示每个文件的详细信息
pygount --folders-to-skip=".git,node_modules,venv" .

# 按代码行数排序（通过管道传给 sort）
pygount --folders-to-skip=".git,node_modules,venv" . | sort -t$'\t' -k1 -nr | head -20
```

## 5. 输出格式

```bash
# 摘要表格（默认推荐）
pygount --format=summary .

# JSON 输出，适合程序化处理
pygount --format=json .

# 管道友好：语言、文件数、代码行、文档行、空行、字符串行
pygount --format=summary . 2>/dev/null
```

## 6. 结果解读

摘要表格各列说明：
- **Language** — 检测到的编程语言
- **Files** — 该语言的文件数量
- **Code** — 实际代码行数（可执行/声明性语句）
- **Comment** — 注释或文档行数
- **%** — 占总量的百分比

特殊伪语言：
- `__empty__` — 空文件
- `__binary__` — 二进制文件（图片、编译产物等）
- `__generated__` — 自动生成的文件（启发式检测）
- `__duplicate__` — 内容完全相同的文件
- `__unknown__` — 无法识别的文件类型

## 注意事项

1. **始终排除 .git、node_modules、venv** — 不使用 `--folders-to-skip` 时，pygount 会遍历所有内容，在大型依赖树上可能耗时数分钟甚至卡死。
2. **Markdown 显示 0 代码行** — pygount 将所有 Markdown 内容归类为注释而非代码，这是预期行为。
3. **JSON 文件代码行数偏低** — pygount 统计 JSON 行数时可能较为保守，如需精确统计 JSON 行数，请直接使用 `wc -l`。
4. **大型 monorepo** — 对于非常大的仓库，建议使用 `--suffix` 指定目标语言，而非扫描全部内容。