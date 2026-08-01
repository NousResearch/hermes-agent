---
title: "Hermes Agent Skill 编写——在仓库中编写 SKILL"
sidebar_label: "Hermes Agent Skill 编写"
description: "在仓库中编写 SKILL.md"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent Skill 编写

编写仓库内 SKILL.md：frontmatter（前置元数据）、验证器、结构。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/hermes-agent-skill-authoring` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `skills`, `authoring`, `hermes-agent`, `conventions`, `skill-md` |
| 相关 skill | [`writing-plans`](/user-guide/skills/bundled/software-development/software-development-writing-plans), [`requesting-code-review`](/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 编写 Hermes-Agent Skills（仓库内）

## 概述

SKILL.md 可以存放在两个位置：

1. **用户本地：** `~/.hermes/skills/<maybe-category>/<name>/SKILL.md` — 个人使用，不共享。通过 `skill_manage(action='create')` 创建。
2. **仓库内（本 skill 讨论此情况）：** `/home/bb/hermes-agent/skills/<category>/<name>/SKILL.md` — 已提交，随包一起发布。使用 `write_file` + `git add`。`skill_manage(action='create')` **不**针对此目录树。

## 使用时机

- 用户要求你"在此分支 / 仓库 / 提交中"添加一个 skill
- 你正在提交一个应随 hermes-agent 一起发布的可复用工作流
- 你正在编辑 `/home/bb/hermes-agent/skills/` 下的现有 skill（小改动用 `patch`，重写用 `write_file`；`skill_manage` 对仓库内 skill 的 `patch` 仍有效，但 `create` 无效）

## 必需的 Frontmatter

真实来源：`tools/skill_manager_tool.py::_validate_frontmatter`。硬性要求：

- 以 `---` 作为首字节开头（无前导空行）。
- 在正文前以 `\n---\n` 结束。
- 可解析为 YAML 映射。
- 存在 `name` 字段。
- 存在 `description` 字段，且 ≤ **1024 个字符**（`MAX_DESCRIPTION_LENGTH`）。
- 关闭 `---` 后有非空正文。

`skills/software-development/` 下每个 skill 使用的对等匹配格式：

```yaml
---
name: my-skill-name               # 小写，连字符，≤64 个字符（MAX_NAME_LENGTH）
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

`version` / `author` / `license` / `metadata` 不受验证器强制约束，但每个同类 skill 都有这些字段——省略会使你的 skill 显得格格不入。

## 大小限制

- Description：≤ 1024 个字符（强制执行）。
- 完整 SKILL.md：≤ 100,000 个字符（强制执行为 `MAX_SKILL_CONTENT_CHARS`，约 36k token）。
- `software-development/` 中的同类 skill 大小在 **8-14k 字符**之间。以此为目标范围。若超过 20k，请拆分为 `references/*.md` 并在 SKILL.md 中引用。

## 对等匹配结构

每个仓库内 skill 大致遵循以下结构：

```
# <Title>

## Overview
One or two paragraphs: what and why.

## When to Use
- Bulleted triggers
- "Don't use for:" counter-triggers

## <Topic sections specific to the skill>
- Quick-reference tables are common
- Code blocks with exact commands
- Hermes-specific recipes (tests via scripts/run_tests.sh, ui-tui paths, etc.)

## Common Pitfalls
Numbered list of mistakes and their fixes.

## Verification Checklist
- [ ] Checkbox list of post-action verifications

## One-Shot Recipes (optional)
Named scenarios → concrete command sequences.
```

并非每个章节都是必需的，但 `Overview` + `When to Use` + 可操作正文 + 常见问题至少要有，skill 才能与同类看齐。

## 目录放置

```
skills/<category>/<skill-name>/SKILL.md
```

仓库中现有的分类（通过 `ls skills/` 确认）：`autonomous-ai-agents`、`creative`、`data-science`、`devops`、`email`、`gaming`、`github`、`leisure`、`mcp`、`media`、`mlops/*`、`note-taking`、`productivity`、`red-teaming`、`research`、`smart-home`、`social-media`、`software-development`。

选择最接近的现有分类。不要随意创建新的顶级分类。

## 工作流

1. **调查同类 skill**，位于目标分类下：
   ```
   ls skills/<category>/
   ```
   阅读 2-3 个同类 SKILL.md 文件，以匹配语气和结构。
2. **如有疑问，检查 `tools/skill_manager_tool.py` 中的验证器约束。**
3. **起草**，使用 `write_file` 写入 `skills/<category>/<name>/SKILL.md`。
4. **本地验证**：
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
5. **Git add + commit**，在当前活跃分支上。
6. **注意：** 当前会话的 skill 加载器已缓存——`skill_view` / `skills_list` 在新会话开始前不会看到新 skill。这是预期行为，不是 bug。

## 交叉引用其他 Skill

`metadata.hermes.related_skills` 在加载时会合并两个目录树（仓库内 `skills/` 和 `~/.hermes/skills/`）。你**可以**从仓库内 skill 引用用户本地 skill，但对于全新克隆仓库的其他用户，该引用无法解析。仓库内 skill 优先只引用仓库内 skill。如果某个频繁被引用的 skill 仅存在于 `~/.hermes/skills/`，请考虑将其提升到仓库中。

## 编辑现有仓库内 Skill

- **小改动（修正错别字、添加常见问题、收紧触发条件）：** `skill_manage(action='patch', name=..., old_string=..., new_string=...)` 对仓库内 skill 同样有效。
- **大规模重写：** 使用 `write_file` 写入完整 SKILL.md。`skill_manage(action='edit')` 也可以，但需要提供完整的新内容。
- **添加支持文件：** 使用 `write_file` 写入 `skills/<category>/<name>/references/<file>.md`、`templates/<file>` 或 `scripts/<file>`。`skill_manage(action='write_file')` 也可以，并会强制执行 references/templates/scripts/assets 子目录白名单。
- **始终提交**编辑——仓库内 skill 是源码，不是运行时状态。

## 常见问题

1. **对仓库内 skill 使用 `skill_manage(action='create')`。** 它会写入 `~/.hermes/skills/`，而非仓库目录树。仓库内创建请使用 `write_file`。

2. **`---` 前有前导空白。** 验证器检查 `content.startswith("---")`；任何前导空行或 BOM 都会导致验证失败。

3. **Description 过于泛泛。** 同类 skill 的 description 以"Use when ..."开头，描述的是*触发类别*，而非单一任务。"Use when debugging X" 优于 "Debug X"。

4. **忘记添加 author/license/metadata 块。** 验证器不强制要求，但每个同类 skill 都有；省略会使 skill 看起来未完成。

5. **编写了与同类重复的 skill。** 创建前先执行 `ls skills/<category>/` 并打开 2-3 个同类 skill。优先扩展现有 skill，而非创建功能狭窄的兄弟 skill。

6. **期望当前会话能看到新 skill。** 不会。skill 加载器在会话开始时初始化。请在新会话中验证，或通过 `skill_view` 使用精确路径进行验证。

7. **链接到仓库中不存在的 skill。** `related_skills: [some-user-local-skill]` 对你有效，但对其他克隆用户会失效。优先只使用仓库内链接。

## 验证清单

- [ ] 文件位于 `skills/<category>/<name>/SKILL.md`（不在 `~/.hermes/skills/` 中）
- [ ] Frontmatter 从字节 0 以 `---` 开头，以 `\n---\n` 结束
- [ ] `name`、`description`、`version`、`author`、`license`、`metadata.hermes.{tags, related_skills}` 均已填写
- [ ] Name ≤ 64 个字符，小写加连字符
- [ ] Description ≤ 1024 个字符，且以"Use when ..."开头
- [ ] 文件总大小 ≤ 100,000 个字符（目标 8-15k）
- [ ] 结构：`# Title` → `## Overview` → `## When to Use` → 正文 → `## Common Pitfalls` → `## Verification Checklist`
- [ ] `related_skills` 中的引用在仓库内可解析（或明确允许为用户本地）
- [ ] 已在目标分支上完成 `git add skills/<category>/<name>/ && git commit`