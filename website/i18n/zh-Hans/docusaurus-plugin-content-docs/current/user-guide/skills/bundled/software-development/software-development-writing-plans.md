---
title: "编写计划 — 编写实施计划：小块任务、路径、代码"
sidebar_label: "编写计划"
description: "编写实施计划：小块任务、路径、代码"
---

{/* 此页面由网站脚本 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# 编写计划

编写实施计划：小块任务、路径、代码。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑安装（默认安装） |
| 路径 | `skills/software-development/writing-plans` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 标签 | `planning`, `design`, `implementation`, `workflow`, `documentation` |
| 相关技能 | [`subagent-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-subagent-driven-development), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes Agent 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# 编写实施计划

## 概述

编写全面的实施计划，假设实施者对代码库没有任何上下文且品味可疑。记录他们需要的一切：要修改哪些文件、完整代码、测试命令、要检查的文档、如何验证。给他们小块任务。DRY。YAGNI。TDD。频繁提交。

假设实施者是一位熟练的开发者，但对工具集或问题领域几乎一无所知。假设他们不太了解好的测试设计。

**核心原则：** 一个好的计划使实施显而易见。如果有人需要猜测，计划就不完整。

## 何时使用

**始终在以下之前使用：**
- 实施多步骤功能
- 分解复杂需求
- 通过 subagent-driven-development 分派给子代理

**不要在以下情况下跳过：**
- 功能看起来简单（假设导致 bug）
- 计划自己实施（未来的你需要指导）
- 独自工作（文档很重要）

## 小块任务粒度

**每个任务 = 2-5分钟专注工作。**

每个步骤都是一个动作：
- "编写失败的测试" — 步骤
- "运行它确保它失败" — 步骤
- "编写最少的代码让测试通过" — 步骤
- "运行测试确保它们通过" — 步骤
- "提交" — 步骤

**太大：**
```markdown
### 任务 1：构建认证系统
[跨5个文件的50行代码]
```

**正确大小：**
```markdown
### 任务 1：创建带 email 字段的 User 模型
[10行，1个文件]

### 任务 2：向 User 添加密码哈希字段
[8行，1个文件]

### 任务 3：创建密码哈希工具
[15行，1个文件]
```

## 计划文档结构

### 头部（必需）

每个计划必须以以下内容开头：

```markdown
# [功能名称] 实施计划

> **For Hermes：** 使用 subagent-driven-development 技能逐任务实施此计划。

**目标：** [一句话描述构建什么]

**架构：** [关于方法的2-3句话]

**技术栈：** [关键技术和库]

---
```

### 任务结构

每个任务遵循此格式：

````markdown
### 任务 N：[描述性名称]

**目标：** 此任务完成什么（一句话）

**文件：**
- 创建：`exact/path/to/new_file.py`
- 修改：`exact/path/to/existing.py:45-67`（如果知道行号）
- 测试：`tests/path/to/test_file.py`

**步骤 1：编写失败的测试**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**步骤 2：运行测试验证失败**

运行：`pytest tests/path/test.py::test_specific_behavior -v`
预期：失败 — "function 未定义"

**步骤 3：编写最小实现**

```python
def function(input):
    return expected
```

**步骤 4：运行测试验证通过**

运行：`pytest tests/path/test.py::test_specific_behavior -v`
预期：通过

**步骤 5：提交**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## 编写过程

### 步骤 1：理解需求

阅读并理解：
- 功能需求
- 设计文档或用户描述
- 验收标准
- 约束

### 步骤 2：探索代码库

使用 Hermes 工具理解项目：

```python
# 理解项目结构
search_files("*.py", target="files", path="src/")

# 查看类似功能
search_files("similar_pattern", path="src/", file_glob="*.py")

# 检查现有测试
search_files("*.py", target="files", path="tests/")

# 读取关键文件
read_file("src/app.py")
```

### 步骤 3：设计方法

决定：
- 架构模式
- 文件组织
- 所需依赖
- 测试策略

### 步骤 4：编写任务

按顺序创建任务：
1. 设置/基础设施
2. 核心功能（每个都用 TDD）
3. 边界情况
4. 集成
5. 清理/文档

### 步骤 5：添加完整细节

对于每个任务，包括：
- **确切的文件路径**（不是"配置文件"而是 `src/config/settings.py`）
- **完整的代码示例**（不是"添加验证"而是实际代码）
- **确切的命令** 及预期输出
- **验证步骤** 证明任务有效

### 步骤 6：审查计划

检查：
- [ ] 任务顺序合理
- [ ] 每个任务是小块的（2-5分钟）
- [ ] 文件路径确切
- [ ] 代码示例完整（可直接复制粘贴）
- [ ] 命令确切，有预期输出
- [ ] 没有缺失的上下文
- [ ] 应用了 DRY、YAGNI、TDD 原则

### 步骤 7：保存计划

```bash
mkdir -p docs/plans
# 保存计划到 docs/plans/YYYY-MM-DD-feature-name.md
git add docs/plans/
git commit -m "docs: add implementation plan for [feature]"
```

## 原则

### DRY（不要重复自己）

**坏：** 在3个地方复制粘贴验证
**好：** 提取验证函数，到处使用

### YAGNI（你不会需要它）

**坏：** 为未来需求添加"灵活性"
**好：** 只实现现在需要的

```python
# 坏 — YAGNI 违反
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.preferences = {}  # 还不需要！
        self.metadata = {}     # 还不需要！

# 好 — YAGNI
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### TDD（测试驱动开发）

每个产生代码的任务应包括完整的 TDD 循环：
1. 编写失败的测试
2. 运行验证失败
3. 编写最少代码
4. 运行验证通过

详见 `test-driven-development` 技能。

### 频繁提交

每个任务后提交：
```bash
git add [files]
git commit -m "type: description"
```

## 常见错误

### 模糊任务

**坏：** "添加认证"
**好：** "创建带 email 和 password_hash 字段的 User 模型"

### 不完整代码

**坏：** "步骤 1：添加验证函数"
**好：** "步骤 1：添加验证函数" 后跟完整函数代码

### 缺少验证

**坏：** "步骤 3：测试它有效"
**好：** "步骤 3：运行 `pytest tests/test_auth.py -v`，预期：3 passed"

### 缺少文件路径

**坏：** "创建模型文件"
**好：** "创建：`src/models/user.py`"

## 执行交接

保存计划后，提供执行方法：

**"计划完成并保存。准备使用 subagent-driven-development 执行 — 我将为每个任务分派一个新的子代理，进行两阶段审查（规范合规性然后代码质量）。是否继续？"**

执行时，使用 `subagent-driven-development` 技能：
- 每个任务一个新的 `delegate_task`，包含完整上下文
- 每个任务后进行规范合规性审查
- 规范通过后进行代码质量审查
- 仅当两个审查都批准时继续

## 记住

```
小块任务（每个2-5分钟）
确切的文件路径
完整代码（可直接复制粘贴）
确切的命令，有预期输出
验证步骤
DRY、YAGNI、TDD
频繁提交
```

**一个好的计划使实施显而易见。**
