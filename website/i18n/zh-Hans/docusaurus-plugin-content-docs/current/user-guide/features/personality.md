---
sidebar_position: 9
title: "人格与 SOUL.md"
description: "使用全局 SOUL.md、内置人格和自定义 persona 定义自定义 Hermes Agent 的人格"
---

# 人格与 SOUL.md

Hermes Agent 的人格完全可定制。`SOUL.md` 是**主要身份**——它是系统提示中的第一件事，定义 Agent 是谁。

- `SOUL.md` — 位于 `HERMES_HOME` 的持久 persona 文件，作为 Agent 的身份（系统提示槽 #1）
- 内置或自定义 `/personality` 预设 — 会话级系统提示覆盖

如果你想改变 Hermes 是谁——或者用完全不同的 Agent persona 替换它——编辑 `SOUL.md`。

## SOUL.md 现在如何工作

Hermes 现在自动在以下位置植入默认 `SOUL.md`：

```text
~/.hermes/SOUL.md
```

更准确地说，它使用当前实例的 `HERMES_HOME`，所以如果你用自定义主目录运行 Hermes，它会使用：

```text
$HERMES_HOME/SOUL.md
```

### 重要行为

- **SOUL.md 是 Agent 的主要身份。** 它占据系统提示槽 #1，替换硬编码默认身份。
- 如果尚不存在，Hermes 自动创建入门 `SOUL.md`
- 现有用户 `SOUL.md` 文件永远不会被覆盖
- Hermes 仅从 `HERMES_HOME` 加载 `SOUL.md`
- Hermes 不在当前工作目录中查找 `SOUL.md`
- 如果 `SOUL.md` 存在但为空，或无法加载，Hermes 回退到内置默认身份
- 如果 `SOUL.md` 有内容，该内容经过安全扫描和截断后逐字注入
- SOUL.md **不会**在上下文文件部分重复——它只出现一次，作为身份

这使 `SOUL.md` 成为真正的每用户或每实例身份，而不仅仅是附加层。

## 为什么这样设计

这保持人格可预测。

如果 Hermes 从你碰巧启动它的任何目录加载 `SOUL.md`，你的人格可能在项目之间意外改变。通过仅从 `HERMES_HOME` 加载，人格属于 Hermes 实例本身。

这也使教用户更容易：
- "编辑 `~/.hermes/SOUL.md` 改变 Hermes 的默认人格。"

## 在哪里编辑

对于大多数用户：

```bash
~/.hermes/SOUL.md
```

如果你使用自定义主目录：

```bash
$HERMES_HOME/SOUL.md
```

## SOUL.md 中应该放什么？

用于持久的语音和人格指导，例如：
- 语气
- 沟通风格
- 直接程度
- 默认交互风格
- 风格上要避免什么
- Hermes 如何处理不确定性、分歧或歧义

少用于：
- 一次性的项目指令
- 文件路径
- repo 约定
- 临时工作流细节

那些放在 `AGENTS.md` 中，不是 `SOUL.md`。

## 好的 SOUL.md 内容

一个好的 SOUL 文件是：
- 跨上下文稳定
- 足够广泛以适用于许多对话
- 足够具体以实质性塑造语音
- 专注于沟通和身份，不是任务特定指令

### 示例

```markdown
# Personality

You are a pragmatic senior engineer with strong taste.
You optimize for truth, clarity, and usefulness over politeness theater.

## Style
- Be direct without being cold
- Prefer substance over filler
- Push back when something is a bad idea
- Admit uncertainty plainly
- Keep explanations compact unless depth is useful

## What to avoid
- Sycophancy
- Hype language
- Repeating the user's framing if it's wrong
- Overexplaining obvious things

## Technical posture
- Prefer simple systems over clever systems
- Care about operational reality, not idealized architecture
- Treat edge cases as part of the design, not cleanup
```

## Hermes 注入到提示的内容

`SOUL.md` 内容直接进入系统提示槽 #1——agent identity 位置。没有添加包装语言。

内容经过：
- 提示注入扫描
- 如果太大则截断

如果文件为空、仅空白、或无法读取，Hermes 回退到内置默认身份（"You are Hermes Agent, an intelligent AI assistant created by Nous Research..."）。当设置 `skip_context_files` 时此回退也适用（例如在子 agent/委托上下文中）。

## 安全扫描

`SOUL.md` 像其他上下文承载文件一样在被包含之前扫描提示注入模式。

这意味着你应该保持专注于 persona/语音，而不是试图偷偷塞入奇怪的元指令。

## SOUL.md vs AGENTS.md

这是最重要的区别。

### SOUL.md
用于：
- 身份
- 语气
- 风格
- 沟通默认值
- 人格级行为

### AGENTS.md
用于：
- 项目架构
- 编码约定
- 工具偏好
- repo 特定工作流
- 命令、端口、路径、部署笔记

一个有用的规则：
- 如果应该跟随你到处，放在 `SOUL.md`
- 如果属于一个项目，放在 `AGENTS.md`

## SOUL.md vs `/personality`

`SOUL.md` 是你持久的默认人格。

`/personality` 是改变或补充当前系统提示的会话级覆盖。

所以：
- `SOUL.md` = 基线语音
- `/personality` = 临时模式切换

示例：
- 保持务实的默认 SOUL，然后在辅导对话中使用 `/personality teacher`
- 保持简洁的 SOUL，然后在头脑风暴中使用 `/personality creative`

## 内置人格

Hermes 附带你可以用 `/personality` 切换的内置人格。

| 名称 | 描述 |
|------|-------------|
| **helpful** | 友好、通用的助手 |
| **concise** | 简短、扼要的回复 |
| **technical** | 详细、准确的技术专家 |
| **creative** | 创新、打破常规的思维 |
| **teacher** | 耐心教育者，带清晰示例 |
| **kawaii** | 可爱表达，闪闪发光，热情 ★ |
| **catgirl** | Neko-chan 带猫样表达，nya~ |
| **pirate** | 船长 Hermes，技术精湛的海盗 |
| **shakespeare** | 带有戏剧气息的 Bardic 散文 |
| **surfer** | 完全 chill bro 氛围 |
| **noir** | 硬汉侦探叙述 |
| **uwu** | 最大可爱与 uwu 说话 |
| **philosopher** | 对每个查询深度沉思 |
| **hype** | 最大限度活力和热情！！！ |

## 用命令切换人格

### CLI

```text
/personality
/personality concise
/personality technical
```

### 消息平台

```text
/personality teacher
```

这些是方便的覆盖，但你的全局 `SOUL.md` 仍然给予 Hermes 其持久默认人格，除非覆盖实质性改变了它。

## 配置中的自定义人格

你也可以在 `~/.hermes/config.yaml` 的 `agent.personalities` 下定义命名自定义人格。

```yaml
agent:
  personalities:
    codereviewer: >
      You are a meticulous code reviewer. Identify bugs, security issues,
      performance concerns, and unclear design choices. Be precise and constructive.
```

然后用它切换：

```text
/personality codereviewer
```

## 推荐工作流

一个强大的默认设置是：

1. 在 `~/.hermes/SOUL.md` 保持深思熟虑的全局 `SOUL.md`
2. 在 `AGENTS.md` 放项目指令
3. 仅当想要临时模式切换时使用 `/personality`

这给你：
- 稳定的语音
- 项目特定行为在它所属的地方
- 需要时临时控制

## 人格如何与完整提示交互

高层次，提示栈包括：
1. **SOUL.md**（agent identity — 或如果 SOUL.md 不可用则内置回退）
2. 工具感知行为指导
3. 记忆/用户上下文
4. skills 指导
5. 上下文文件（`AGENTS.md`、`.cursorrules`）
6. 时间戳
7. 平台特定格式提示
8. 可选的系统提示覆盖如 `/personality`

`SOUL.md` 是基础——其他一切都在其上构建。

## 相关文档

- [上下文文件](/docs/user-guide/features/context-files)
- [配置](/docs/user-guide/configuration)
- [提示和最佳实践](/docs/guides/tips)
- [SOUL.md 指南](/docs/guides/use-soul-with-hermes)

## CLI 外观 vs 对话人格

对话人格和 CLI 外观是分开的：

- `SOUL.md`、`agent.system_prompt` 和 `/personality` 影响 Hermes 说话方式
- `display.skin` 和 `/skin` 影响 Hermes 在终端中的外观

关于终端外观，参见 [皮肤和主题](./skins.md)。
