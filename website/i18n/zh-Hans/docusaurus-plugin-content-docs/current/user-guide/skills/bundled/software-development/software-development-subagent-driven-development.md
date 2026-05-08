---
title: "Subagent Driven Development — 通过 delegate_task 子代理执行计划（两阶段审查）"
sidebar_label: "Subagent Driven Development"
description: "通过 delegate_task 子代理执行计划（两阶段审查）"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Subagent 驱动开发

通过 delegate_task 子代理执行计划（两阶段审查）。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/subagent-driven-development` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 标签 | `delegation`, `subagent`, `implementation`, `workflow`, `parallel` |
| 相关技能 | [`writing-plans`](/docs/user-guide/skills/bundled/software-development/software-development-writing-plans), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Subagent 驱动开发

## 概述

通过为每个任务分派独立的子代理来执行实现计划，并配合系统的两阶段审查。

**核心原则：** 每个任务使用新的子代理 + 两阶段审查（规格合规 + 代码质量） = 高质量、快速迭代。

## 使用时机

在以下情况使用此技能：
- 你有一个实现计划（来自 writing-plans 技能或用户需求）
- 任务大多独立
- 质量和规格合规很重要
- 你想要任务之间的自动化审查

**vs. 手动执行：**
- 每个任务有新的上下文（不会因累积状态而混淆）
- 自动化审查过程及早发现问题
- 所有任务间一致的质量检查
- 子代理可以在开始工作之前提问

## 流程

### 1. 阅读和解析计划

阅读计划文件。预先提取所有任务及其完整文本和上下文。创建待办列表：

```python
# 阅读计划
read_file("docs/plans/feature-plan.md")

# 创建包含所有任务的待办列表
todo([
    {"id": "task-1", "content": "Create User model with email field", "status": "pending"},
    {"id": "task-2", "content": "Add password hashing utility", "status": "pending"},
    {"id": "task-3", "content": "Create login endpoint", "status": "pending"},
])
```

**关键：** 只读取计划一次。提取所有内容。不要让子代理读取计划文件——在上下文中直接提供完整的任务文本。

### 2. 每任务工作流

对于计划中的**每个**任务：

#### 步骤 1：分发实现者子代理

使用 `delegate_task` 提供完整上下文：

```python
delegate_task(
    goal="Implement Task 1: Create User model with email and password_hash fields",
    context="""
    计划中的任务：
    - 创建: src/models/user.py
    - 添加带有 email (str) 和 password_hash (str) 字段的 User 类
    - 使用 bcrypt 进行密码哈希
    - 包含 __repr__ 用于调试

    遵循 TDD：
    1. 在 tests/models/test_user.py 中编写失败的测试
    2. 运行: pytest tests/models/test_user.py -v（验证失败）
    3. 编写最小实现
    4. 运行: pytest tests/models/test_user.py -v（验证通过）
    5. 运行: pytest tests/ -q（验证无回归）
    6. 提交: git add -A && git commit -m "feat: add User model with password hashing"

    项目上下文：
    - Python 3.11, Flask 应用在 src/app.py
    - 现有模型在 src/models/
    - 测试使用 pytest，从项目根目录运行
    - bcrypt 已在 requirements.txt 中
    """,
    toolsets=['terminal', 'file']
)
```

#### 步骤 2：分发规格合规审查者

实现者完成后，对照原始规格验证：

```python
delegate_task(
    goal="审查实现是否匹配计划中的规格",
    context="""
    原始任务规格：
    - 创建 src/models/user.py 带有 User 类
    - 字段: email (str), password_hash (str)
    - 使用 bcrypt 进行密码哈希
    - 包含 __repr__

    检查：
    - [ ] 规格中的所有要求已实现？
    - [ ] 文件路径匹配规格？
    - [ ] 函数签名匹配规格？
    - [ ] 行为符合预期？
    - [ ] 没有添加额外内容（无范围蔓延）？

    输出：PASS 或具体规格差距列表。
    """,
    toolsets=['file']
)
```

**如果发现规格问题：** 修复差距，然后重新运行规格审查。仅在规格合规时继续。

#### 步骤 3：分发代码质量审查者

规格合规通过后：

```python
delegate_task(
    goal="审查任务 1 实现的代码质量",
    context="""
    要审查的文件：
    - src/models/user.py
    - tests/models/test_user.py

    检查：
    - [ ] 遵循项目约定和风格？
    - [ ] 适当的错误处理？
    - [ ] 清晰的变量/函数命名？
    - [ ] 充足的测试覆盖？
    - [ ] 没有明显的 Bug 或遗漏的边缘情况？
    - [ ] 没有安全问题？

    输出格式：
    - 严重问题：[必须修复才能继续]
    - 重要问题：[应该修复]
    - 次要问题：[可选]
    - 结论：APPROVED 或 REQUEST_CHANGES
    """,
    toolsets=['file']
)
```

**如果发现质量问题：** 修复问题，重新审查。仅在批准时继续。

#### 步骤 4：标记完成

```python
todo([{"id": "task-1", "content": "Create User model with email field", "status": "completed"}], merge=True)
```

### 3. 最终审查

所有任务完成后，分派一个最终集成审查者：

```python
delegate_task(
    goal="审查整个实现的一致性和集成问题",
    context="""
    计划中的所有任务已完成。审查完整实现：
    - 所有组件是否协同工作？
    - 任务间是否有不一致？
    - 所有测试是否通过？
    - 是否准备好合并？
    """,
    toolsets=['terminal', 'file']
)
```

### 4. 验证和提交

```bash
# 运行完整测试套件
pytest tests/ -q

# 审查所有更改
git diff --stat

# 如需最终提交
git add -A && git commit -m "feat: complete [feature name] implementation"
```

## 任务粒度

**每个任务 = 2-5 分钟的专注工作。**

**太大：**
- "实现用户认证系统"

**合适的大小：**
- "创建带有 email 和 password 字段的 User 模型"
- "添加密码哈希函数"
- "创建登录端点"
- "添加 JWT token 生成"
- "创建注册端点"

## 红旗——绝不要做这些

- 没有计划就开始实现
- 跳过审查（规格合规或代码质量）
- 带着未修复的严重/重要问题继续
- 对触摸相同文件的任务分派多个实现子代理
- 让子代理读取计划文件（在上下文中提供完整文本）
- 跳过场景设置上下文（子代理需要理解任务的位置）
- 忽略子代理的问题（在他们继续之前回答）
- 在规格合规上接受"差不多"
- 跳过审查循环（审查者发现问题 → 实现者修复 → 再次审查）
- 让实现者的自查替代实际审查（两者都需要）
- **在规格合规通过之前开始代码质量审查**（顺序错误）
- 在任一审查有未决问题时移到下一个任务

## 处理问题

### 如果子代理提出问题

- 清晰完整地回答
- 如需要提供额外上下文
- 不要催促他们进入实现

### 如果审查者发现问题

- 实现者子代理（或新的）修复它们
- 审查者再次审查
- 重复直到批准
- 不要跳过重新审查

### 如果子代理任务失败

- 分派一个带有关于出了什么问题的具体指令的新修复子代理
- 不要在控制器会话中手动修复（上下文污染）

## 效率说明

**为什么每个任务使用新的子代理：**
- 防止累积状态导致的上下文污染
- 每个子代理获得干净的、聚焦的上下文
- 不会因之前任务的代码或推理而混淆

**为什么两阶段审查：**
- 规格审查及早捕获少建/多建
- 质量审查确保实现构建良好
- 在问题跨任务复合之前捕获

**成本权衡：**
- 更多的子代理调用（每个任务需要实现者 + 2 个审查者）
- 但及早发现问题（比后来调试复合问题更便宜）

## 与其他技能的集成

### 与 writing-plans

此技能执行由 writing-plans 技能创建的计划：
1. 用户需求 → writing-plans → 实现计划
2. 实现计划 → subagent-driven-development → 可工作的代码

### 与 test-driven-development

实现者子代理应遵循 TDD：
1. 先编写失败测试
2. 实现最小代码
3. 验证测试通过
4. 提交

在每个实现者上下文中包含 TDD 指令。

### 与 requesting-code-review

两阶段审查过程就是代码审查。对于最终集成审查，使用 requesting-code-review 技能的审查维度。

### 与 systematic-debugging

如果子代理在实现过程中遇到 Bug：
1. 遵循 systematic-debugging 过程
2. 在修复之前找到根本原因
3. 编写回归测试
4. 恢复实现

## 示例工作流

```
[阅读计划: docs/plans/auth-feature.md]
[创建包含 5 个任务的待办列表]

--- 任务 1: 创建 User 模型 ---
[分派实现者子代理]
  实现者: "email 应该唯一吗？"
  你: "是的，email 必须唯一"
  实现者: 已实现，3/3 测试通过，已提交。

[分派规格审查者]
  规格审查者: ✅ PASS — 所有要求满足

[分派质量审查者]
  质量审查者: ✅ APPROVED — 代码干净，测试良好

[标记任务 1 完成]

--- 任务 2: 密码哈希 ---
[分派实现者子代理]
  实现者: 无问题，已实现，5/5 测试通过。

[分派规格审查者]
  规格审查者: ❌ 缺少：密码强度验证（规格说"最少 8 个字符"）

[实现者修复]
  实现者: 已添加验证，7/7 测试通过。

[分派规格审查者再次审查]
  规格审查者: ✅ PASS

[分派质量审查者]
  质量审查者: 重要：魔数 8，提取为常量
  实现者: 已提取 MIN_PASSWORD_LENGTH 常量
  质量审查者: ✅ APPROVED

[标记任务 2 完成]

... (继续所有任务)

[所有任务后：分派最终集成审查者]
[运行完整测试套件：全部通过]
[完成！]
```

## 记住

```
每个任务使用新的子代理
每次都进行两阶段审查
规格合规优先
代码质量其次
永远不要跳过审查
及早发现问题
```

**质量不是偶然的。它是系统化过程的结果。**

## 进一步阅读（相关时加载）

当编排涉及大量上下文使用、长审查循环或复杂验证检查点时，加载这些参考文档以获取特定纪律：

- **`references/context-budget-discipline.md`** ——四级上下文退化模型（PEAK / GOOD / DEGRADING / POOR），随上下文窗口大小缩放的读取深度规则，以及静默退化的早期预警信号。当运行将明显消耗大量上下文（多阶段计划、许多子代理、大型制品）时加载。
- **`references/gates-taxonomy.md`** ——四种规范门类型（Pre-flight、Revision、Escalation、Abort），附有行为、恢复和示例。在设计或审查任何具有验证检查点的工作流时加载——使用明确的词汇以便每个门都有定义的进入、失败行为和恢复规则。

两个参考文档改编自 gsd-build/get-shit-done（MIT © 2025 Lex Christopherson）。
