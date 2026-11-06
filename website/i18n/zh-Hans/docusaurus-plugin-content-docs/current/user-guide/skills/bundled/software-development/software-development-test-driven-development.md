---
title: "测试驱动开发 — TDD：强制执行 RED-GREEN-REFACTOR，测试先于代码"
sidebar_label: "测试驱动开发"
description: "TDD：强制执行 RED-GREEN-REFACTOR，测试先于代码"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 测试驱动开发

TDD：强制执行 RED-GREEN-REFACTOR，测试先于代码。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/test-driven-development` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent（改编自 obra/superpowers） |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `testing`, `tdd`, `development`, `quality`, `red-green-refactor` |
| 相关 skill | [`systematic-debugging`](/user-guide/skills/bundled/software-development/software-development-systematic-debugging)、[`writing-plans`](/user-guide/skills/bundled/software-development/software-development-writing-plans)、[`subagent-driven-development`](/user-guide/skills/bundled/software-development/software-development-subagent-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 测试驱动开发（TDD）

## 概述

先写测试。看它失败。再写最少的代码使其通过。

**核心原则：** 如果你没有亲眼看到测试失败，你就不知道它是否测试了正确的东西。

**违反规则的字面意义，就是违反规则的精神。**

## 何时使用

**始终使用：**
- 新功能
- Bug 修复
- 重构
- 行为变更

**例外情况（须先询问用户）：**
- 一次性原型
- 生成的代码
- 配置文件

觉得"这次跳过 TDD 就好"？停下来。那是在自我合理化。

## 铁律

```
没有先写失败的测试，就不能写生产代码
```

在写测试之前就写了代码？删掉它。重新开始。

**没有例外：**
- 不要以"参考"为由保留它
- 不要在写测试时"改编"它
- 不要看它
- 删除就是删除

从测试出发重新实现。就这样。

## Red-Green-Refactor 循环

### RED — 编写失败的测试

编写一个最简测试，说明应该发生什么。

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
名称清晰，测试真实行为，只测一件事。

**坏的测试：**
```python
def test_retry_works():
    mock = MagicMock()
    mock.side_effect = [Exception(), Exception(), 'success']
    result = retry_operation(mock)
    assert result == 'success'  # 重试次数呢？时序呢？
```
名称模糊，测试的是 mock 而非真实代码。

**要求：**
- 每个测试只测一个行为
- 名称清晰具描述性（名称中有"and"？拆分它）
- 使用真实代码，而非 mock（除非确实不可避免）
- 名称描述行为，而非实现

### 验证 RED — 亲眼看到它失败

**强制要求。绝不跳过。**

```bash
# 使用 terminal 工具运行特定测试
pytest tests/test_feature.py::test_specific_behavior -v
```

确认：
- 测试失败（不是因为拼写错误导致的报错）
- 失败信息符合预期
- 因功能缺失而失败

**测试立即通过？** 你在测试已有的行为。修正测试。

**测试报错？** 修复错误，重新运行，直到它正确地失败。

### GREEN — 最少代码

编写最简单的代码使测试通过。不多不少。

**好的：**
```python
def add(a, b):
    return a + b  # 没有多余的东西
```

**坏的：**
```python
def add(a, b):
    result = a + b
    logging.info(f"Adding {a} + {b} = {result}")  # 多余！
    return result
```

不要添加功能、重构其他代码，或在测试范围之外"改进"。

**GREEN 阶段允许作弊：**
- 硬编码返回值
- 复制粘贴
- 重复代码
- 跳过边界情况

我们会在 REFACTOR 阶段修复它。

### 验证 GREEN — 亲眼看到它通过

**强制要求。**

```bash
# 运行特定测试
pytest tests/test_feature.py::test_specific_behavior -v

# 然后运行所有测试，检查是否有回归
pytest tests/ -q
```

确认：
- 测试通过
- 其他测试仍然通过
- 输出干净（无错误、无警告）

**测试失败？** 修复代码，而非测试。

**其他测试失败？** 立即修复回归问题。

### REFACTOR — 清理

仅在绿色之后：
- 消除重复
- 改善命名
- 提取辅助函数
- 简化表达式

全程保持测试绿色。不要添加行为。

**重构期间测试失败？** 立即撤销。步子迈小一点。

### 重复

为下一个行为编写下一个失败的测试。一次一个循环。

## 为什么顺序很重要

**"我会在之后写测试来验证它是否有效"**

在代码之后写的测试会立即通过。立即通过什么都证明不了：
- 可能测试了错误的东西
- 可能测试的是实现而非行为
- 可能遗漏了你忘记的边界情况
- 你从未看到它捕获 bug

测试先行迫使你看到测试失败，证明它确实在测试某些东西。

**"我已经手动测试了所有边界情况"**

手动测试是临时性的。你以为自己测试了所有情况，但：
- 没有记录你测试了什么
- 代码变更时无法重新运行
- 在压力下容易遗漏情况
- "我试过时它能用" ≠ 全面覆盖

自动化测试是系统性的。每次以相同方式运行。

**"删除 X 小时的工作是浪费"**

这是沉没成本谬误。时间已经过去了。你现在的选择是：
- 删除并用 TDD 重写（高置信度）
- 保留并事后添加测试（低置信度，可能有 bug）

"浪费"是保留你无法信任的代码。

**"TDD 是教条主义，务实意味着适应"**

TDD 本身就是务实的：
- 在提交前发现 bug（比事后调试更快）
- 防止回归（测试立即捕获破坏）
- 记录行为（测试展示如何使用代码）
- 支持重构（自由修改，测试捕获破坏）

"务实"的捷径 = 在生产环境调试 = 更慢。

**"事后写测试能达到相同目标——重要的是精神而非仪式"**

不对。事后写的测试回答"这做了什么？"测试先行回答"这应该做什么？"

事后写的测试受你的实现偏见影响。你测试的是你构建的东西，而非需求。测试先行迫使你在实现之前发现边界情况。

## 常见自我合理化

| 借口 | 现实 |
|--------|---------|
| "太简单了，不需要测试" | 简单的代码也会出错。写测试只需 30 秒。 |
| "我之后再测试" | 立即通过的测试什么都证明不了。 |
| "事后写测试能达到相同目标" | 事后测试 = "这做了什么？"测试先行 = "这应该做什么？" |
| "已经手动测试过了" | 临时性 ≠ 系统性。没有记录，无法重新运行。 |
| "删除 X 小时的工作是浪费" | 沉没成本谬误。保留未经验证的代码就是技术债务。 |
| "保留作参考，先写测试" | 你会改编它。那就是事后测试。删除就是删除。 |
| "需要先探索" | 没问题。丢掉探索代码，从 TDD 开始。 |
| "测试难写 = 设计不清晰" | 听测试的话。难以测试 = 难以使用。 |
| "TDD 会让我变慢" | TDD 比调试更快。务实 = 测试先行。 |
| "手动测试更快" | 手动测试无法证明边界情况。每次变更都要重新测试。 |
| "现有代码没有测试" | 你在改进它。为你接触的代码添加测试。 |

## 红色警报 — 停下来，重新开始

如果你发现自己在做以下任何一件事，删除代码并用 TDD 重新开始：

- 测试之前写了代码
- 实现之后写测试
- 测试在第一次运行时立即通过
- 无法解释测试为何失败
- 测试"稍后"添加
- 合理化"就这一次"
- "我已经手动测试过了"
- "事后写测试能达到相同目的"
- "保留作参考"或"改编现有代码"
- "已经花了 X 小时，删除是浪费"
- "TDD 是教条主义，我在务实"
- "这种情况不同，因为……"

**所有这些都意味着：删除代码。用 TDD 重新开始。**

## 验证清单

在标记工作完成之前：

- [ ] 每个新函数/方法都有测试
- [ ] 在实现之前亲眼看到每个测试失败
- [ ] 每个测试因预期原因失败（功能缺失，而非拼写错误）
- [ ] 编写了最少的代码使每个测试通过
- [ ] 所有测试通过
- [ ] 输出干净（无错误、无警告）
- [ ] 测试使用真实代码（仅在不可避免时使用 mock）
- [ ] 边界情况和错误情况已覆盖

无法勾选所有项？你跳过了 TDD。重新开始。

## 遇到困难时

| 问题 | 解决方案 |
|---------|----------|
| 不知道如何测试 | 写出期望的 API。先写断言。询问用户。 |
| 测试太复杂 | 设计太复杂。简化接口。 |
| 必须 mock 所有东西 | 代码耦合度太高。使用依赖注入。 |
| 测试 setup 很庞大 | 提取辅助函数。仍然复杂？简化设计。 |

## Hermes Agent 集成

### 运行测试

使用 `terminal` 工具在每个步骤运行测试：

```python
# RED — 验证失败
terminal("pytest tests/test_feature.py::test_name -v")

# GREEN — 验证通过
terminal("pytest tests/test_feature.py::test_name -v")

# 完整套件 — 验证无回归
terminal("pytest tests/ -q")
```

### 与 delegate_task 配合使用

向子 agent 分派实现任务时，在目标中强制执行 TDD：

```python
delegate_task(
    goal="Implement [feature] using strict TDD",
    context="""
    Follow test-driven-development skill:
    1. Write failing test FIRST
    2. Run test to verify it fails
    3. Write minimal code to pass
    4. Run test to verify it passes
    5. Refactor if needed
    6. Commit

    Project test command: pytest tests/ -q
    Project structure: [describe relevant files]
    """,
    toolsets=['terminal', 'file']
)
```

### 与 systematic-debugging 配合使用

发现 bug？编写能复现它的失败测试。遵循 TDD 循环。测试证明了修复的有效性并防止回归。

绝不在没有测试的情况下修复 bug。

## 测试反模式

- **测试 mock 行为而非真实行为** — mock 应用于验证交互，而非替代被测系统
- **测试实现细节** — 测试行为/结果，而非内部方法调用
- **只测试正常路径** — 始终测试边界情况、错误情况和边界值
- **脆弱的测试** — 测试应验证行为而非结构；重构不应导致测试失败

## 最终规则

```
生产代码 → 测试先存在且先失败
否则 → 不是 TDD
```

未经用户明确许可，没有例外。