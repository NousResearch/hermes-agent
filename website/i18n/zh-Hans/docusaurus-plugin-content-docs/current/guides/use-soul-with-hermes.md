---
sidebar_position: 7
title: "将 SOUL.md 与 Hermes 一起使用"
description: "如何使用 SOUL.md 塑造 Hermes 代理的默认声音，什么应该放在那里，以及它与 AGENTS.md 和 /personality 的区别"
---

# 将 SOUL.md 与 Hermes 一起使用

`SOUL.md` 是你的 Hermes 实例的**主要身份**。它是系统提示中的第一项——它定义代理是谁、如何说话以及应该避免什么。

如果你想让 Hermes 每次与你交谈时感觉像同一个助手——或者如果你想用自己的角色完全替换 Hermes 角色——这就是要使用的文件。

## SOUL.md 的用途

使用 `SOUL.md` 用于：
- 语气
- 个性
- 沟通风格
- Hermes 应该有多直接或温暖
- Hermes 在风格上应该避免什么
- Hermes 如何处理不确定性、分歧和歧义

简而言之：
- `SOUL.md` 是关于 Hermes 是谁以及 Hermes 如何说话

## SOUL.md 不适用于什么

不要用于：
- repo 特定的编码约定
- 文件路径
- 命令
- 服务端口
- 架构笔记
- 项目工作流指令

那些属于 `AGENTS.md`。

一个好规则：
- 如果它应该普遍适用，放在 `SOUL.md` 中
- 如果它只属于一个项目，放在 `AGENTS.md` 中

## 它的位置

Hermes 现在仅为当前实例使用全局 SOUL 文件：

```text
~/.hermes/SOUL.md
```

如果你用自定义主目录运行 Hermes，它变成：

```text
$HERMES_HOME/SOUL.md
```

## 首次运行行为

如果 `SOUL.md` 不存在，Hermes 会自动为你生成一个起始 `SOUL.md`。

这意味着大多数用户现在从一个可以立即读取和编辑的真实文件开始。

重要：
- 如果你已经有 `SOUL.md`，Hermes 不会覆盖它
- 如果文件存在但为空，Hermes 不会从中向提示添加任何内容

## Hermes 如何使用它

当 Hermes 启动会话时，它从 `HERMES_HOME` 读取 `SOUL.md`，扫描提示注入模式，如需要时截断它，并将其用作**代理身份**——系统提示中的槽位 #1。这意味着 SOUL.md 完全替换内置默认身份文本。

如果 SOUL.md 缺失、空或无法加载，Hermes 回退到内置默认身份。

文件周围不添加包装语言。内容本身很重要——以你希望代理思考和说话的方式书写。

## 一个好的首次编辑

如果你不做其他任何事情，打开文件并只更改几行，使其感觉像你。

例如：

```markdown
You are direct, calm, and technically precise.
Prefer substance over politeness theater.
Push back clearly when an idea is weak.
Keep answers compact unless deeper detail is useful.
```

仅此就可以显著改变 Hermes 的感觉。

## 示例风格

### 1. 务实的工程师

```markdown
You are a pragmatic senior engineer.
You care more about correctness and operational reality than sounding impressive.

## Style
- Be direct
- Be concise unless complexity requires depth
- Say when something is a bad idea
- Prefer practical tradeoffs over idealized abstractions

## Avoid
- Sycophancy
- Hype language
- Overexplaining obvious things
```

### 2. 研究伙伴

```markdown
You are a thoughtful research collaborator.
You are curious, honest about uncertainty, and excited by unusual ideas.

## Style
- Explore possibilities without pretending certainty
- Distinguish speculation from evidence
- Ask clarifying questions when the idea space is underspecified
- Prefer conceptual depth over shallow completeness
```

### 3. 教师/解释者

```markdown
You are a patient technical teacher.
You care about understanding, not performance.

## Style
- Explain clearly
- Use examples when they help
- Do not assume prior knowledge unless the user signals it
- Build from intuition to details
```

### 4. 严格审查者

```markdown
You are a rigorous reviewer.
You are fair, but you do not soften important criticism.

## Style
- Point out weak assumptions directly
- Prioritize correctness over harmony
- Be explicit about risks and tradeoffs
- Prefer blunt clarity to vague diplomacy
```

## 是什么造就了一个强大的 SOUL.md？

一个强大的 `SOUL.md` 是：
- 稳定的
- 广泛适用的
- 在声音上具体的
- 不过载临时指令

一个弱的 `SOUL.md` 是：
- 充满项目细节
- 矛盾的
- 试图微观管理每个响应形状
- 主要是像"有帮助"和"清晰"这样的通用填充物

Hermes 已经尝试有帮助和清晰。`SOUL.md` 应该添加真实的个性和风格，而不是重述明显的默认值。

## 建议的结构

你不需要标题，但它们有帮助。

一个简单但有效的结构：

```markdown
# Identity
Who Hermes is.

# Style
How Hermes should sound.

# Avoid
What Hermes should not do.

# Defaults
How Hermes should behave when ambiguity appears.
```

## SOUL.md vs /personality

这些是互补的。

使用 `SOUL.md` 作为你的持久基线。
使用 `/personality` 进行临时模式切换。

示例：
- 你的默认 SOUL 是务实和直接的
- 然后在一个会话中使用 `/personality teacher`
- 之后你切换回来而不更改你的基础声音文件

## SOUL.md vs AGENTS.md

这是最常见的错误。

### 放在 SOUL.md 中
- "要直接。"
- "避免炒作语言。"
- "除非深度有帮助，否则优先简短回答。"
- "当用户错了时要反驳。"

### 放在 AGENTS.md 中
- "使用 pytest，不是 unittest。"
- "前端位于 `frontend/`。"
- "永远不要直接编辑迁移文件。"
- "API 在端口 8000 上运行。"

## 如何编辑它

```bash
nano ~/.hermes/SOUL.md
```

或

```bash
vim ~/.hermes/SOUL.md
```

然后重启 Hermes 或开始新会话。

## 一个实用工作流

1. 从种子默认文件开始
2. 修剪任何不像你想要的声音的内容
3. 添加 4-8 行明确定义语气和默认值
4. 与 Hermes 交谈一会儿
5. 根据仍然感觉不对的地方进行调整

这种迭代方法比试图一次设计完美的个性更好。

## 故障排除

### 我编辑了 SOUL.md 但 Hermes 仍然听起来一样

检查：
- 你编辑了 `~/.hermes/SOUL.md` 或 `$HERMES_HOME/SOUL.md`
- 不是一些 repo 本地的 `SOUL.md`
- 文件不为空
- 编辑后会话已重启
- `/personality` 覆盖没有主导结果

### Hermes 忽略了我 SOUL.md 的部分内容

可能原因：
- 更高优先级的指令覆盖了它
- 文件包含冲突的指导
- 文件太长被截断了
- 一些文本类似于提示注入内容，可能被扫描器阻止或更改

### 我的 SOUL.md 变得太特定于项目了

将项目指令移动到 `AGENTS.md`，保持 `SOUL.md` 专注于身份和风格。

## 相关文档

- [个性 & SOUL.md](/docs/user-guide/features/personality)
- [上下文文件](/docs/user-guide/features/context-files)
- [配置](/docs/user-guide/configuration)
- [技巧与最佳实践](/docs/guides/tips)
