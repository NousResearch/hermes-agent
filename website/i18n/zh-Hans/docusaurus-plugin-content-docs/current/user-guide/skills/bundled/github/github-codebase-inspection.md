---
title: "代码库检查 — 使用 pygount 检查代码库：代码行数、语言、占比"
sidebar_label: "代码库检查"
description: "使用 pygount 检查代码库：代码行数、语言、占比"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 代码库检查

使用 pygount 检查代码库：代码行数、语言、占比。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/github/codebase-inspection` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `LOC`、`代码分析`、`pygount`、`代码库`、`指标`、`仓库` |
| 相关技能 | [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 使用 pygount 进行代码库检查

使用 `pygount` 分析仓库的代码行数、语言分布、文件数量和代码与注释比例。

## 使用场景

- 用户要求统计代码行数（LOC）
- 用户想要仓库的语言分布
- 用户询问代码库大小或组成
- 用户想要代码与注释的比例
- 一般的"这个仓库有多大"之类的问题

## 前置条件

```bash
pip install --break-system-packages pygount 2>/dev/null || pip install pygount
```

## 1. 基本摘要（最常用）

获取完整的语言分布，包含文件数量、代码行数和注释行数：

```bash
cd /path/to/repo
pygount --format=summary \
  --folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,.eggs,*.egg-info" \
  .
```

**重要：** 始终使用 `--folders-to-skip` 排除依赖/构建目录，否则 pygount 会遍历它们并花费很长时间或挂起。

## 2. 常见文件夹排除

根据项目类型调整：

```bash
# Python 项目
--folders-to-skip=".git,venv,.venv,__pycache__,.cache,dist,build,.tox,.eggs,.mypy_cache"

# JavaScript/TypeScript 项目
--folders-to-skip=".git,node_modules,dist,build,.next,.cache,.turbo,coverage"

# 通用排除
--folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,vendor,third_party"
```

## 3. 按特定语言筛选

```bash
# 仅统计 Python 文件
pygount --suffix=py --format=summary .

# 仅统计 Python 和 YAML
pygount --suffix=py,yaml,yml --format=summary .
```

## 4. 详细的逐文件输出

```bash
# 默认格式显示逐文件分布
pygount --folders-to-skip=".git,node_modules,venv" .

# 按代码行排序（通过管道传递给 sort）
pygount --folders-to-skip=".git,node_modules,venv" . | sort -t$'\t' -k1 -nr | head -20
```

## 5. 输出格式

```bash
# 摘要表（推荐默认使用）
pygount --format=summary .

# JSON 输出用于程序化处理
pygount --format=json .

# 适合管道传输：语言、文件数、代码、文档、空行、字符串
pygount --format=summary . 2>/dev/null
```

## 6. 结果解读

摘要表的列：
- **语言** — 检测到的编程语言
- **文件** — 该语言的文件数量
- **代码** — 实际代码行数（可执行/声明性）
- **注释** — 注释或文档行
- **%** — 占总数的百分比

特殊伪语言：
- `__empty__` — 空文件
- `__binary__` — 二进制文件（图片、编译文件等）
- `__generated__` — 自动生成的文件（启发式检测）
- `__duplicate__` — 内容完全相同的文件
- `__unknown__` — 未识别的文件类型

## 常见问题

1. **始终排除 .git、node_modules、venv** — 不使用 `--folders-to-skip`，pygount 会遍历所有内容，在大型依赖树上可能需要几分钟或挂起。
2. **Markdown 显示 0 行代码** — pygount 将所有 Markdown 内容分类为注释而非代码。这是预期行为。
3. **JSON 文件显示较低的代码行数** — pygount 可能保守地计算 JSON 行数。要获取精确的 JSON 行数，请直接使用 `wc -l`。
4. **大型 monorepo** — 对于非常大的仓库，考虑使用 `--suffix` 针对特定语言，而不是扫描所有内容。
