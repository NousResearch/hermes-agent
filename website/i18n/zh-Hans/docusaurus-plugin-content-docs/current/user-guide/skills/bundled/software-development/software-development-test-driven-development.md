---
title: "测试驱动开发 — TDD：强制 RED-GREEN-REFACTOR，测试优先于代码"
sidebar_label: "测试驱动开发"
description: "TDD：强制 RED-GREEN-REFACTOR，测试优先于代码"
---

{/* 此页面由网站脚本 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# 测试驱动开发

TDD：强制 RED-GREEN-REFACTOR，测试优先于代码。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑安装（默认安装） |
| 路径 | `skills/software-development/test-driven-development` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 标签 | `testing`, `tdd`, `development`, `quality`, `red-green-refactor` |
| 相关技能 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`writing-plans`](/docs/user-guide/skills/bundled/software-development/software-development-writing-plans), [`subagent-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-subagent-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes Agent 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# 测试驱动开发（TDD）

## 概述

先写测试。看它失败。写最少的代码让它通过。

**核心原则：** 如果你没有看到测试失败，你就不知道它是否测试了正确的东西。

**违反规则的字面规定就是违反规则的精神。**

## 何时使用

**始终使用于：**
- 新功能
- Bug 修复
- 重构
- 行为变更

**例外（先询问用户）：**
- 一次性原型
- 生成的代码
- 配置文件

想"这次跳过 TDD"？停止。这是合理化。

## 铁律

```
未经失败的测试，不得编写生产代码
```

在测试之前写代码？删除它。重新开始。

**没有例外：**
- 不要保留作为"参考"
- 不要在写测试时"适配"它
- 不要看它
- 删除就是删除

从测试开始全新实现。就这样。

## 红-绿-重构循环

### 红色 — 编写失败的测试

写一个最小的测试，展示应该发生什么。

**好的测试：**
```python
def test_retries_failed_operations_3_times():
    attempts = 0
    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception('fail')
        return 'success'

    result = retry_operation(operation)

    assert result == 'success'
    assert attempts == 3
```
清晰的名称，测试真实行为，一次一件事。

**坏的测试：**
```python
def test_retry_works():
    mock = MagicMock()
    mock.side_effect = [Exception(), Exception(), 'success']
    result = retry_operation(mock)
    assert result == 'success'  # 重试次数呢？时间呢？
```
模糊的名称，测试 mock 而不是真实代码。

**要求：**
- 每个测试一个行为
- 清晰描述性的名称（名称中有"和"？拆开它）
- 真实代码，不是 mock（除非确实无法避免）
- 名称描述行为，而不是实现

### 验证红色 — 看它失败

**强制执行。永远不要跳过。**

```bash
# 使用 terminal 工具运行特定测试
pytest tests/test_feature.py::test_specific_behavior -v
```

确认：
- 测试失败（不是拼写错误导致的错误）
- 失败消息符合预期
- 因为功能缺失而失败

**测试立即通过？** 你在测试现有行为。修复测试。

**测试报错？** 修复错误，重新运行直到正确失败。

### 绿色 — 最少代码

写最简单的代码让测试通过。仅此而已。

**好的：**
```python
def add(a, b):
    return a + b  # 什么都没加
```

**坏的：**
```python
def add(a, b):
    result = a + b
    logging.info(f"Adding {a} + {b} = {result}")  # 多余的！
    return result
```

不要添加功能、重构其他代码，或超越测试进行"改进"。

**绿色阶段作弊是可以的：**
- 硬编码返回值
- 复制粘贴
- 重复代码
- 跳过边界情况

我们会在重构阶段修复。

### 验证绿色 — 看它通过

**强制执行。**

```bash
# 运行特定测试
pytest tests/test_feature.py::test_specific_behavior -v

# 然后运行所有测试检查回归
pytest tests/ -q
```

确认：
- 测试通过
- 其他测试仍然通过
- 输出干净（无错误、无警告）

**测试失败？** 修复代码，而不是测试。

**其他测试失败？** 现在修复回归。

### 重构 — 清理

仅在绿色之后：
- 移除重复
- 改进名称
- 提取辅助函数
- 简化表达式

整个过程中保持测试绿色。不要添加行为。

**如果在重构期间测试失败：** 立即撤销。采取更小的步骤。

### 重复

下一个行为的下一个失败测试。一次一个循环。

## 为什么顺序重要

**"我之后写测试来验证它有效"**

代码之后写的测试会立即通过。立即通过什么也证明不了：
- 可能测试了错误的东西
- 可能测试了实现，而不是行为
- 可能遗漏了你忘记的边界情况
- 你从未看到它捕获了 bug

测试优先强迫你看到测试失败，证明它实际上测试了某些东西。

**"我已经手动测试了所有边界情况"**

手动测试是随意的。你以为测试了一切，但：
- 没有记录你测试了什么
- 代码变更时无法重新运行
- 在压力下容易忘记情况
- "我试的时候有效" ≠ 全面

自动化测试是系统化的。它们每次都以相同方式运行。

**"删除 X 小时的工作是浪费"**

沉没成本谬误。时间已经过去了。你现在的选择：
- 删除并用 TDD 重写（高置信度）
- 保留它并在之后添加测试（低置信度，可能有 bug）

"浪费"是保留你无法信任的代码。

**"TDD 是教条的，务实意味着适应"**

TDD 就是务实的：
- 在提交前发现 bug（比之后调试更快）
- 防止回归（测试立即捕获破坏）
- 记录行为（测试展示如何使用代码）
- 支持重构（自由更改，测试捕获破坏）

"务实"的捷径 = 生产环境调试 = 更慢。

**"之后测试达到相同目标 — 是精神而不是仪式"**

不适用。之后的测试回答"这做什么？"测试优先回答"这应该做什么？"

之后的测试受实现偏见影响。你测试你构建的，而不是需要的。测试优先在实现之前强制发现边界情况。

## 常见合理化借口

| 借口 | 现实 |
|--------|---------|
| "太简单不需要测试" | 简单代码也会坏。测试只需30秒。 |
| "我之后测试" | 测试立即通过什么也证明不了。 |
| "之后测试达到相同目标" | 之后测试 = "这做什么？" 测试优先 = "这应该做什么？" |
| "已经手动测试了" | 随意 ≠ 系统化。无记录，无法重新运行。 |
| "删除 X 小时是浪费" | 沉没成本谬误。保留未验证的代码是技术债务。 |
| "保留作为参考，先写测试" | 你会适配它。那就是之后测试。删除就是删除。 |
| "需要先探索" | 可以。丢弃探索，用 TDD 开始。 |
| "测试难 = 设计不清楚" | 听从测试。难以测试 = 难以使用。 |
| "TDD 会拖慢我" | TDD 比调试快。务实 = 测试优先。 |
| "手动测试更快" | 手动不能证明边界情况。每次更改都要重新测试。 |
| "现有代码没有测试" | 你在改进它。为你接触的代码添加测试。 |

## 红旗 — 停止并重新开始

如果发现自己做以下任何一项，删除代码并用 TDD 重新开始：

- 代码在测试之前
- 测试在实现之后
- 测试首次运行立即通过
- 无法解释为什么测试失败
- "之后"添加测试
- 合理化"就这一次"
- "我已经手动测试了"
- "之后测试达到相同目的"
- "保留作为参考"或"适配现有代码"
- "已经花了 X 小时，删除是浪费"
- "TDD 是教条的，我很务实"
- "这是不同的因为..."

**以上所有都意味着：删除代码。用 TDD 重新开始。**

## 验证检查清单

在标记工作完成之前：

- [ ] 每个新函数/方法都有测试
- [ ] 在实现之前看到每个测试失败
- [ ] 每个测试因预期原因失败（功能缺失，而不是拼写错误）
- [ ] 写了最少的代码让每个测试通过
- [ ] 所有测试通过
- [ ] 输出干净（无错误、无警告）
- [ ] 测试使用真实代码（mock 仅在不可避免时使用）
- [ ] 覆盖边界情况和错误

无法勾选所有框？你跳过了 TDD。重新开始。

## 当卡住时

| 问题 | 解决方案 |
|---------|----------|
| 不知道如何测试 | 写下期望的 API。先写断言。询问用户。 |
| 测试太复杂 | 设计太复杂。简化接口。 |
| 必须 mock 一切 | 代码耦合太紧。使用依赖注入。 |
| 测试设置很大 | 提取辅助函数。还是复杂？简化设计。 |

## Hermes Agent 集成

### 运行测试

在每个步骤使用 `terminal` 工具运行测试：

```python
# 红色 — 验证失败
terminal("pytest tests/test_feature.py::test_name -v")

# 绿色 — 验证通过
terminal("pytest tests/test_feature.py::test_name -v")

# 完整套件 — 验证无回归
terminal("pytest tests/ -q")
```

### 与 delegate_task 配合

当分派子代理进行实现时，在目标中强制 TDD：

```python
delegate_task(
    goal="使用严格的 TDD 实现 [功能]",
    context="""
    遵循 test-driven-development 技能：
    1. 先写失败的测试
    2. 运行测试验证它失败
    3. 写最少的代码让它通过
    4. 运行测试验证它通过
    5. 如需要重构
    6. 提交

    项目测试命令：pytest tests/ -q
    项目结构：[描述相关文件]
    """,
    toolsets=['terminal', 'file']
)
```

### 与 systematic-debugging 配合

发现 bug？写一个复现它的失败测试。遵循 TDD 循环。测试证明修复并防止回归。

永远不要在没有测试的情况下修复 bug。

## 测试反模式

- **测试 mock 行为而不是真实行为** — mock 应该验证交互，而不是替换被测系统
- **测试实现细节** — 测试行为/结果，而不是内部方法调用
- **仅快乐路径** — 始终测试边界情况、错误和边界
- **脆弱测试** — 测试应该验证行为，而不是结构；重构不应该破坏它们

## 最终规则

```
生产代码 → 存在测试且先失败
否则 → 不是 TDD
```

未经用户明确许可，没有任何例外。
