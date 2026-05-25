---
title: "子智能体驱动开发 — 通过 delegate_task 子智能体执行计划（两阶段审查）"
sidebar_label: "子智能体驱动开发"
description: "通过 delegate_task 子智能体执行计划（两阶段审查）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 子智能体驱动开发

通过 delegate_task 子智能体执行计划（两阶段审查）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/subagent-driven-development` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `delegation`, `subagent`, `implementation`, `workflow`, `parallel` |
| 相关 skill | [`writing-plans`](/user-guide/skills/bundled/software-development/software-development-writing-plans)、[`requesting-code-review`](/user-guide/skills/bundled/software-development/software-development-requesting-code-review)、[`test-driven-development`](/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是智能体在 skill 激活时所看到的指令内容。
:::

# 子智能体驱动开发

## 概述

通过为每个任务派发全新子智能体并进行系统性两阶段审查来执行实现计划。

**核心原则：** 每个任务使用全新子智能体 + 两阶段审查（规格合规性审查，然后是质量审查）= 高质量、快速迭代。

## 使用时机

在以下情况下使用此 skill：
- 你有一个实现计划（来自 writing-plans skill 或用户需求）
- 任务大体上相互独立
- 质量和规格合规性很重要
- 你希望在任务之间进行自动化审查

**与手动执行相比：**
- 每个任务拥有全新上下文（不会因累积状态而产生混乱）
- 自动化审查流程能尽早发现问题
- 对所有任务进行一致的质量检查
- 子智能体可以在开始工作前提问

## 流程

### 1. 读取并解析计划

读取计划文件。预先提取所有任务的完整文本和上下文。创建待办列表：

```python
# Read the plan
read_file("docs/plans/feature-plan.md")

# Create todo list with all tasks
todo([
    {"id": "task-1", "content": "Create User model with email field", "status": "pending"},
    {"id": "task-2", "content": "Add password hashing utility", "status": "pending"},
    {"id": "task-3", "content": "Create login endpoint", "status": "pending"},
])
```

**关键：** 只读取计划一次。提取所有内容。不要让子智能体读取计划文件——直接在上下文中提供完整的任务文本。

### 2. 每个任务的工作流

对计划中的**每个**任务执行以下步骤：

#### 步骤 1：派发实现者子智能体

使用 `delegate_task` 并提供完整上下文：

```python
delegate_task(
    goal="Implement Task 1: Create User model with email and password_hash fields",
    context="""
    TASK FROM PLAN:
    - Create: src/models/user.py
    - Add User class with email (str) and password_hash (str) fields
    - Use bcrypt for password hashing
    - Include __repr__ for debugging

    FOLLOW TDD:
    1. Write failing test in tests/models/test_user.py
    2. Run: pytest tests/models/test_user.py -v (verify FAIL)
    3. Write minimal implementation
    4. Run: pytest tests/models/test_user.py -v (verify PASS)
    5. Run: pytest tests/ -q (verify no regressions)
    6. Commit: git add -A && git commit -m "feat: add User model with password hashing"

    PROJECT CONTEXT:
    - Python 3.11, Flask app in src/app.py
    - Existing models in src/models/
    - Tests use pytest, run from project root
    - bcrypt already in requirements.txt
    """,
    toolsets=['terminal', 'file']
)
```

#### 步骤 2：派发规格合规性审查者

实现者完成后，对照原始规格进行验证：

```python
delegate_task(
    goal="Review if implementation matches the spec from the plan",
    context="""
    ORIGINAL TASK SPEC:
    - Create src/models/user.py with User class
    - Fields: email (str), password_hash (str)
    - Use bcrypt for password hashing
    - Include __repr__

    CHECK:
    - [ ] All requirements from spec implemented?
    - [ ] File paths match spec?
    - [ ] Function signatures match spec?
    - [ ] Behavior matches expected?
    - [ ] Nothing extra added (no scope creep)?

    OUTPUT: PASS or list of specific spec gaps to fix.
    """,
    toolsets=['file']
)
```

**如果发现规格问题：** 修复差距，然后重新运行规格审查。仅在规格合规后继续。

#### 步骤 3：派发代码质量审查者

规格合规性通过后：

```python
delegate_task(
    goal="Review code quality for Task 1 implementation",
    context="""
    FILES TO REVIEW:
    - src/models/user.py
    - tests/models/test_user.py

    CHECK:
    - [ ] Follows project conventions and style?
    - [ ] Proper error handling?
    - [ ] Clear variable/function names?
    - [ ] Adequate test coverage?
    - [ ] No obvious bugs or missed edge cases?
    - [ ] No security issues?

    OUTPUT FORMAT:
    - Critical Issues: [must fix before proceeding]
    - Important Issues: [should fix]
    - Minor Issues: [optional]
    - Verdict: APPROVED or REQUEST_CHANGES
    """,
    toolsets=['file']
)
```

**如果发现质量问题：** 修复问题，重新审查。仅在获得批准后继续。

#### 步骤 4：标记为完成

```python
todo([{"id": "task-1", "content": "Create User model with email field", "status": "completed"}], merge=True)
```

### 3. 最终审查

所有任务完成后，派发最终集成审查者：

```python
delegate_task(
    goal="Review the entire implementation for consistency and integration issues",
    context="""
    All tasks from the plan are complete. Review the full implementation:
    - Do all components work together?
    - Any inconsistencies between tasks?
    - All tests passing?
    - Ready for merge?
    """,
    toolsets=['terminal', 'file']
)
```

### 4. 验证并提交

```bash
# Run full test suite
pytest tests/ -q

# Review all changes
git diff --stat

# Final commit if needed
git add -A && git commit -m "feat: complete [feature name] implementation"
```

## 任务粒度

**每个任务 = 2-5 分钟的专注工作。**

**粒度过大：**
- "实现用户认证系统"

**合适的粒度：**
- "创建包含 email 和 password 字段的 User 模型"
- "添加密码哈希函数"
- "创建登录端点"
- "添加 JWT token 生成"
- "创建注册端点"

## 红线——绝对不要做这些

- 没有计划就开始实现
- 跳过审查（规格合规性审查或代码质量审查）
- 在未修复关键/重要问题的情况下继续推进
- 为涉及相同文件的任务派发多个实现子智能体
- 让子智能体读取计划文件（应在上下文中直接提供完整文本）
- 跳过场景设定上下文（子智能体需要了解任务所处的位置）
- 忽略子智能体的提问（在让其继续之前先回答）
- 在规格合规性上接受"差不多就行"
- 跳过审查循环（审查者发现问题 → 实现者修复 → 再次审查）
- 让实现者自我审查替代实际审查（两者都需要）
- **在规格合规性通过之前开始代码质量审查**（顺序错误）
- 在任一审查存在未解决问题时进入下一个任务

## 处理问题

### 如果子智能体提问

- 清晰、完整地回答
- 如有需要，提供额外上下文
- 不要催促其进入实现阶段

### 如果审查者发现问题

- 实现者子智能体（或新的子智能体）修复问题
- 审查者再次审查
- 重复直到获得批准
- 不要跳过重新审查

### 如果子智能体任务失败

- 派发新的修复子智能体，并提供关于出错原因的具体说明
- 不要在控制器会话中手动修复（会污染上下文）

## 效率说明

**为什么每个任务使用全新子智能体：**
- 防止累积状态导致的上下文污染
- 每个子智能体获得干净、专注的上下文
- 不会因先前任务的代码或推理而产生混乱

**为什么进行两阶段审查：**
- 规格审查能尽早发现构建不足或过度构建的问题
- 质量审查确保实现构建良好
- 在问题跨任务叠加之前将其捕获

**成本权衡：**
- 更多子智能体调用（每个任务：实现者 + 2 个审查者）
- 但能尽早发现问题（比后期调试叠加问题更经济）

## 与其他 Skill 的集成

### 与 writing-plans

此 skill 执行由 writing-plans skill 创建的计划：
1. 用户需求 → writing-plans → 实现计划
2. 实现计划 → subagent-driven-development → 可运行代码

### 与 test-driven-development

实现者子智能体应遵循 TDD：
1. 先编写失败的测试
2. 实现最小化代码
3. 验证测试通过
4. 提交

在每个实现者上下文中都包含 TDD 指令。

### 与 requesting-code-review

两阶段审查流程即是代码审查。对于最终集成审查，使用 requesting-code-review skill 的审查维度。

### 与 systematic-debugging

如果子智能体在实现过程中遇到 bug：
1. 遵循 systematic-debugging 流程
2. 在修复之前找到根本原因
3. 编写回归测试
4. 恢复实现

## 示例工作流

```
[Read plan: docs/plans/auth-feature.md]
[Create todo list with 5 tasks]

--- Task 1: Create User model ---
[Dispatch implementer subagent]
  Implementer: "Should email be unique?"
  You: "Yes, email must be unique"
  Implementer: Implemented, 3/3 tests passing, committed.

[Dispatch spec reviewer]
  Spec reviewer: ✅ PASS — all requirements met

[Dispatch quality reviewer]
  Quality reviewer: ✅ APPROVED — clean code, good tests

[Mark Task 1 complete]

--- Task 2: Password hashing ---
[Dispatch implementer subagent]
  Implementer: No questions, implemented, 5/5 tests passing.

[Dispatch spec reviewer]
  Spec reviewer: ❌ Missing: password strength validation (spec says "min 8 chars")

[Implementer fixes]
  Implementer: Added validation, 7/7 tests passing.

[Dispatch spec reviewer again]
  Spec reviewer: ✅ PASS

[Dispatch quality reviewer]
  Quality reviewer: Important: Magic number 8, extract to constant
  Implementer: Extracted MIN_PASSWORD_LENGTH constant
  Quality reviewer: ✅ APPROVED

[Mark Task 2 complete]

... (continue for all tasks)

[After all tasks: dispatch final integration reviewer]
[Run full test suite: all passing]
[Done!]
```

## 记住

```
Fresh subagent per task
Two-stage review every time
Spec compliance FIRST
Code quality SECOND
Never skip reviews
Catch issues early
```

**质量不是偶然的，它是系统化流程的结果。**

## 延伸阅读（按需加载）

当编排涉及大量上下文使用、较长的审查循环或复杂的验证检查点时，加载以下特定领域的参考资料：

- **`references/context-budget-discipline.md`** — 四级上下文退化模型（PEAK / GOOD / DEGRADING / POOR）、随上下文窗口大小调整的读取深度规则，以及静默退化的早期预警信号。当一次运行明显会消耗大量上下文时加载（多阶段计划、大量子智能体、大型产物）。
- **`references/gates-taxonomy.md`** — 四种规范化 gate（关卡）类型（Pre-flight、Revision、Escalation、Abort）及其行为、恢复方式和示例。在设计或审查任何包含验证检查点的工作流时加载——明确使用该词汇表，使每个 gate 都具有明确的入口、失败行为和恢复规则。

两份参考资料均改编自 gsd-build/get-shit-done（MIT © 2025 Lex Christopherson）。