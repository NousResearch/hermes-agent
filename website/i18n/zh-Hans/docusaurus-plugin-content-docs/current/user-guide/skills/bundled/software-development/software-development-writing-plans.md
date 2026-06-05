---
title: "编写计划 — 编写实施计划：细粒度任务、路径、代码"
sidebar_label: "编写计划"
description: "编写实施计划：细粒度任务、路径、代码"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 编写计划

编写实施计划：细粒度任务、路径、代码。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/writing-plans` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `planning`, `design`, `implementation`, `workflow`, `documentation` |
| 相关 skill | [`subagent-driven-development`](/user-guide/skills/bundled/software-development/software-development-subagent-driven-development)、[`test-driven-development`](/user-guide/skills/bundled/software-development/software-development-test-driven-development)、[`requesting-code-review`](/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 编写实施计划

## 概述

编写全面的实施计划，假设实施者对代码库零上下文、品味存疑。记录他们所需的一切：需要修改哪些文件、完整代码、测试命令、需查阅的文档、如何验证。给出细粒度任务。DRY。YAGNI。TDD。频繁提交。

假设实施者是一名熟练的开发者，但对工具集或问题域几乎一无所知。假设他们对良好的测试设计了解不多。

**核心原则：** 好的计划让实施变得显而易见。如果有人需要猜测，说明计划不完整。

## 使用时机

**始终在以下情况前使用：**
- 实施多步骤功能
- 拆解复杂需求
- 通过 subagent-driven-development 委派给子 agent

**不要跳过的情况：**
- 功能看似简单（假设会导致 bug）
- 你打算自己实施（未来的你需要指引）
- 独自工作（文档很重要）

## 细粒度任务粒度

**每个任务 = 2-5 分钟的专注工作。**

每一步都是单一动作：
- "编写失败的测试" — 一步
- "运行以确认它失败" — 一步
- "编写使测试通过的最小代码" — 一步
- "运行测试并确认通过" — 一步
- "提交" — 一步

**太大：**
```markdown
### Task 1: Build authentication system
[50 lines of code across 5 files]
```

**合适大小：**
```markdown
### Task 1: Create User model with email field
[10 lines, 1 file]

### Task 2: Add password hash field to User
[8 lines, 1 file]

### Task 3: Create password hashing utility
[15 lines, 1 file]
```

## 计划文档结构

### 头部（必填）

每个计划必须以以下内容开头：

```markdown
# [Feature Name] Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---
```

### 任务结构

每个任务遵循以下格式：

````markdown
### Task N: [Descriptive Name]

**Objective:** What this task accomplishes (one sentence)

**Files:**
- Create: `exact/path/to/new_file.py`
- Modify: `exact/path/to/existing.py:45-67` (line numbers if known)
- Test: `tests/path/to/test_file.py`

**Step 1: Write failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify failure**

Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: FAIL — "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify pass**

Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## 编写流程

### 第一步：理解需求

阅读并理解：
- 功能需求
- 设计文档或用户描述
- 验收标准
- 约束条件

### 第二步：探索代码库

使用 Hermes 工具了解项目：

```python
# Understand project structure
search_files("*.py", target="files", path="src/")

# Look at similar features
search_files("similar_pattern", path="src/", file_glob="*.py")

# Check existing tests
search_files("*.py", target="files", path="tests/")

# Read key files
read_file("src/app.py")
```

### 第三步：设计方案

决定：
- 架构模式
- 文件组织
- 所需依赖
- 测试策略

### 第四步：编写任务

按顺序创建任务：
1. 搭建/基础设施
2. 核心功能（每项均采用 TDD）
3. 边界情况
4. 集成
5. 清理/文档

### 第五步：补充完整细节

每个任务包含：
- **精确的文件路径**（不是"配置文件"，而是 `src/config/settings.py`）
- **完整的代码示例**（不是"添加验证"，而是实际代码）
- **精确的命令**及预期输出
- **验证步骤**，证明任务有效

### 第六步：审查计划

检查：
- [ ] 任务顺序合理、逻辑清晰
- [ ] 每个任务粒度合适（2-5 分钟）
- [ ] 文件路径精确
- [ ] 代码示例完整（可直接复制粘贴）
- [ ] 命令精确并附有预期输出
- [ ] 无缺失上下文
- [ ] 遵循 DRY、YAGNI、TDD 原则

### 第七步：保存计划

```bash
mkdir -p docs/plans
# Save plan to docs/plans/YYYY-MM-DD-feature-name.md
git add docs/plans/
git commit -m "docs: add implementation plan for [feature]"
```

## 原则

### DRY（不要重复自己）

**差：** 在 3 处复制粘贴验证逻辑
**好：** 提取验证函数，统一使用

### YAGNI（你不会需要它）

**差：** 为未来需求添加"灵活性"
**好：** 只实现当前所需

```python
# Bad — YAGNI violation
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.preferences = {}  # Not needed yet!
        self.metadata = {}     # Not needed yet!

# Good — YAGNI
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### TDD（测试驱动开发）

每个产出代码的任务都应包含完整的 TDD 循环：
1. 编写失败的测试
2. 运行以确认失败
3. 编写最小代码
4. 运行以确认通过

详见 `test-driven-development` skill。

### 频繁提交

每个任务完成后提交：
```bash
git add [files]
git commit -m "type: description"
```

## 常见错误

### 任务描述模糊

**差：** "添加认证"
**好：** "创建包含 email 和 password_hash 字段的 User 模型"

### 代码不完整

**差：** "第一步：添加验证函数"
**好：** "第一步：添加验证函数"，后跟完整的函数代码

### 缺少验证步骤

**差：** "第三步：测试是否有效"
**好：** "第三步：运行 `pytest tests/test_auth.py -v`，预期：3 passed"

### 缺少文件路径

**差：** "创建模型文件"
**好：** "创建：`src/models/user.py`"

## 执行交接

保存计划后，提供执行方案：

**"计划已完成并保存。准备使用 subagent-driven-development 执行——我将为每个任务派发一个全新的子 agent，进行两阶段审查（规格合规性检查，然后代码质量检查）。是否继续？"**

执行时，使用 `subagent-driven-development` skill：
- 每个任务使用携带完整上下文的独立 `delegate_task`
- 每个任务完成后进行规格合规性审查
- 规格通过后进行代码质量审查
- 两项审查均通过后方可继续

## 记住

```
细粒度任务（每个 2-5 分钟）
精确的文件路径
完整代码（可直接复制粘贴）
精确命令及预期输出
验证步骤
DRY、YAGNI、TDD
频繁提交
```

**好的计划让实施变得显而易见。**