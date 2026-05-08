---
title: "Hermes Agent Skill Authoring — 编写仓库内 SKILL"
sidebar_label: "Hermes Agent Skill Authoring"
description: "编写仓库内 SKILL"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Hermes Agent 技能编写

编写仓库内 SKILL.md：frontmatter、验证器、结构。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/hermes-agent-skill-authoring` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `skills`, `authoring`, `hermes-agent`, `conventions`, `skill-md` |
| 相关技能 | [`writing-plans`](/docs/user-guide/skills/bundled/software-development/software-development-writing-plans), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 编写 Hermes-Agent 技能（仓库内）

## 概述

SKILL.md 可以存放在两个位置：

1. **用户本地：** `~/.hermes/skills/<maybe-category>/<name>/SKILL.md` — 个人的，不共享。通过 `skill_manage(action='create')` 创建。
2. **仓库内（此技能针对这种情况）：** `/home/bb/hermes-agent/skills/<category>/<name>/SKILL.md` — 提交到版本控制，随包发布。使用 `write_file` + `git add`。`skill_manage(action='create')` 不会针对此目录树。

## 使用时机

- 用户要求你"在这个分支/仓库/提交中"添加技能
- 你正在提交一个应该随 hermes-agent 发布的可重用工作流
- 你正在编辑 `skills/` 下的现有技能（小编辑用 `patch`，重写用 `write_file`；`skill_manage` 仍可用于仓库内技能的 patch，但不支持 `create`）

## 必需的 Frontmatter

真实来源：`tools/skill_manager_tool.py::_validate_frontmatter`。硬性要求：

- 以 `---` 作为前几个字节开始（无前导空行）。
- 以 `\n---\n` 结束，然后是正文。
- 解析为 YAML 映射。
- 存在 `name` 字段。
- 存在 `description` 字段，≤ **1024 个字符**（`MAX_DESCRIPTION_LENGTH`）。
- 结束的 `---` 之后有非空的正文。

`skills/software-development/` 下每个技能使用的同行匹配格式：

```yaml
---
name: my-skill-name               # 小写、连字符、≤64 字符（MAX_NAME_LENGTH）
description: Use when <trigger>. <one-line behavior>.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [short, descriptive, tags]
    related_skills: [other-skill, another-skill]
---
```

`version` / `author` / `license` / `metadata` 不被验证器强制执行，但每个同行都有它们——省略会让你的技能显得不完整。

## 大小限制

- 描述：≤ 1024 字符（强制执行）。
- 完整 SKILL.md：≤ 100,000 字符（作为 `MAX_SKILL_CONTENT_CHARS` 强制执行，约 36k token）。
- `software-development/` 中的同行技能在 **8-14k 字符**。目标在此范围。如果超过 20k，拆分为 `references/*.md` 并从 SKILL.md 中引用。

## 同行匹配结构

每个仓库内技能大致遵循：

```
# <标题>

## 概述
一到两段：是什么和为什么。

## 使用时机
- 项目触发条件
- "不要用于："反向触发条件

## <技能特定主题部分>
- 快速参考表很常见
- 带确切命令的代码块
- Hermes 特定配方（通过 scripts/run_tests.sh 测试、ui-tui 路径等）

## 常见陷阱
编号的错误列表及其修复。

## 验证清单
- [ ] 操作后验证的复选框列表

## 一次性配方（可选）
命名场景 → 具体命令序列。
```

并非每个部分都是必需的，但 `概述` + `使用时机` + 可操作的正文 + 常见陷阱是技能感觉像同行的最低要求。

## 目录放置

```
skills/<category>/<skill-name>/SKILL.md
```

仓库中当前的分类（用 `ls skills/` 确认）：`autonomous-ai-agents`, `creative`, `data-science`, `devops`, `dogfood`, `email`, `gaming`, `github`, `leisure`, `mcp`, `media`, `mlops/*`, `note-taking`, `productivity`, `red-teaming`, `research`, `smart-home`, `social-media`, `software-development`。

选择最接近的现有分类。不要随意创建新的顶级分类。

## 工作流

1. **调查目标分类中的同行：**
   ```
   ls skills/<category>/
   ```
   阅读 2-3 个同行 SKILL.md 文件以匹配语气和结构。
2. 如不确定，**检查** `tools/skill_manager_tool.py` **中的验证器约束**。
3. 使用 `write_file` **起草** `skills/<category>/<name>/SKILL.md`。
4. **本地验证：**
   ```python
   import yaml, re, pathlib
   content = pathlib.Path("skills/<category>/<name>/SKILL.md").read_text()
   assert content.startswith("---")
   m = re.search(r'\n---\s*\n', content[3:])
   fm = yaml.safe_load(content[3:m.start()+3])
   assert "name" in fm and "description" in fm
   assert len(fm["description"]) <= 1024
   assert len(content) <= 100_000
   ```
5. 在活动分支上 **git add + commit**。
6. **注意：** 当前会话的技能加载器有缓存——`skill_view` / `skills_list` 在新会话之前不会看到新技能。这是预期行为，不是 Bug。

## 交叉引用其他技能

`metadata.hermes.related_skills` 在加载时合并两棵树（仓库内的 `skills/` 和 `~/.hermes/skills/`）。你可以从仓库内技能引用用户本地技能，但它对于其他克隆仓库的用户无法解析。在仓库内技能中优先只引用仓库内技能。如果一个经常引用的技能只存在于 `~/.hermes/skills/` 中，考虑将其提升到仓库。

## 编辑现有仓库内技能

- **小修复（拼写错误、添加陷阱、收紧触发条件）：** `skill_manage(action='patch', name=..., old_string=..., new_string=...)` 对仓库内技能完全适用。
- **重大重写：** 使用 `write_file` 编写整个 SKILL.md。`skill_manage(action='edit')` 也可以，但需要提供完整的新内容。
- **添加支持文件：** `write_file` 到 `skills/<category>/<name>/references/<file>.md`、`templates/<file>` 或 `scripts/<file>`。`skill_manage(action='write_file')` 也可以并强制执行 references/templates/scripts/assets 子目录白名单。
- **始终提交**编辑——仓库内技能是源代码，不是运行时状态。

## 常见陷阱

1. **对仓库内技能使用 `skill_manage(action='create')`。** 它写入 `~/.hermes/skills/`，而非仓库树。使用 `write_file` 进行仓库内创建。

2. **`---` 前有前导空白。** 验证器检查 `content.startswith("---")`；任何前导空行或 BOM 都会验证失败。

3. **描述太笼统。** 同行描述以 "Use when ..." 开头，描述的是*触发类别*，而非具体任务。例如 "Use when debugging X" 要优于 "Debug X"。

4. **遗漏了 author/license/metadata 块。** 不被验证器强制执行，但每个同行都有；省略会让技能看起来半成品。

5. **编写了重复同行的技能。** 创建前，`ls skills/<category>/` 并打开 2-3 个同行。优先扩展现有技能而非创建狭窄的兄弟技能。

6. **期望当前会话看到新技能。** 它看不到。技能加载器在会话开始时初始化。在新会话中验证或使用精确路径通过 `skill_view` 验证。

7. **链接到仓库内不存在的技能。** `related_skills: [some-user-local-skill]` 对你有效但对其他克隆者无效。优先只引用仓库内技能。

## 验证清单

- [ ] 文件位于 `skills/<category>/<name>/SKILL.md`（而非 `~/.hermes/skills/`）
- [ ] Frontmatter 从字节 0 以 `---` 开始，以 `\n---\n` 结束
- [ ] `name`、`description`、`version`、`author`、`license`、`metadata.hermes.{tags, related_skills}` 都存在
- [ ] 名称 ≤ 64 字符，小写 + 连字符
- [ ] 描述 ≤ 1024 字符且以 "Use when ..." 开头
- [ ] 总文件 ≤ 100,000 字符（目标 8-15k）
- [ ] 结构：`# 标题` → `## 概述` → `## 使用时机` → 正文 → `## 常见陷阱` → `## 验证清单`
- [ ] `related_skills` 引用在仓库内可解析（或明确可以接受用户本地）
- [ ] 在目标分支上 `git add skills/<category>/<name>/ && git commit` 已完成
