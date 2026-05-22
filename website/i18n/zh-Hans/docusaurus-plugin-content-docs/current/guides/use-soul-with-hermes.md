---
sidebar_position: 7
title: "使用 SOUL.md 自定义 Hermes"
description: "如何用 SOUL.md 塑造 Hermes 的默认语气、适合写入的内容，以及它与 AGENTS.md 和 /personality 的区别"
---

# 使用 SOUL.md 自定义 Hermes

`SOUL.md` 是 Hermes 实例的主要身份文件。它是系统提示词里的第一层内容，用来定义 agent 是谁、如何说话、以及要避免什么。

如果你希望 Hermes 每次对话都保持同一种风格，或者想把 Hermes 的默认人格替换成你自己的，这就是应该使用的文件。

## SOUL.md 的用途

适合写入：
- 语气
- 个性
- 表达风格
- 回答时是更直接还是更温和
- 风格上应该回避什么
- 面对不确定、争议和模糊问题时的默认态度

简而言之：
- `SOUL.md` 描述的是 Hermes 是谁，以及它怎么说话

## SOUL.md 不适合写什么

不要把这些内容放进去：
- 仓库特定的编码规范
- 文件路径
- 命令
- 服务端口
- 架构备注
- 项目工作流指令

这些内容应写入 `AGENTS.md`。

一个简单规则：
- 如果它应该在任何地方都生效，就写进 `SOUL.md`
- 如果它只属于某个项目，就写进 `AGENTS.md`

## 它放在哪里

Hermes 当前只使用实例级的全局 SOUL 文件：

```text
~/.hermes/SOUL.md
```

如果你使用自定义 home 目录，则是：

```text
$HERMES_HOME/SOUL.md
```

## 首次运行行为

如果 `SOUL.md` 不存在，Hermes 会自动为你生成一个初始版本。

也就是说，大多数用户会先得到一个可直接编辑的真实文件。

注意：
- 如果你已经有 `SOUL.md`，Hermes 不会覆盖它
- 如果文件存在但为空，Hermes 不会把它内容加入提示词

## Hermes 如何使用它

当 Hermes 启动会话时，它会读取 `HERMES_HOME` 下的 `SOUL.md`，扫描提示注入模式，必要时截断，然后把它作为系统提示词里的 **agent identity** 使用。

如果 `SOUL.md` 缺失、为空或无法加载，Hermes 会回退到内建默认身份。

文件本身就是内容，不会额外包一层 wrapper。你写进去的文本就是真正起作用的内容。

## 一个好的起步改法

如果你暂时不想设计太多，只改几行也能让 Hermes 很不一样。

例如：

```markdown
You are direct, calm, and technically precise.
Prefer substance over politeness theater.
Push back clearly when an idea is weak.
Keep answers compact unless deeper detail is useful.
```

## 示例风格

### 1. 务实工程师

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

### 3. 教学 / 解释者

```markdown
You are a patient technical teacher.
You care about understanding, not performance.

## Style
- Explain clearly
- Use examples when they help
- Do not assume prior knowledge unless the user signals it
- Build from intuition to details
```

### 4. 严格审稿人

```markdown
You are a rigorous reviewer.
You are fair, but you do not soften important criticism.

## Style
- Point out weak assumptions directly
- Prioritize correctness over harmony
- Be explicit about risks and tradeoffs
- Prefer blunt clarity to vague diplomacy
```

## 什么样的 SOUL.md 才好

一个强的 `SOUL.md` 应该：
- 稳定
- 适用面广
- 风格清晰
- 不堆临时指令

一个弱的 `SOUL.md` 通常：
- 太多项目细节
- 自相矛盾
- 想把每种响应形态都精细控制
- 大量泛泛的“要有帮助”“要清晰”

Hermes 本来就会尽量提供帮助和清晰表达。`SOUL.md` 应补充真实的人格和风格，而不是重复默认值。

## 推荐结构

你不一定要用标题，但有标题通常更好读。

一个简单有效的结构是：

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

## SOUL.md 与 /personality

两者是互补的。

`SOUL.md` 适合做长期、稳定的默认基线。
`/personality` 适合做临时模式切换。

例如：
- 默认 SOUL 是务实且直接
- 某个会话里临时切到 `/personality teacher`
- 之后再切回，不必改基础 voice 文件

## SOUL.md 与 AGENTS.md

这里最容易混淆。

### 适合写进 SOUL.md
- “直接一点。”
- “不要用夸张语气。”
- “除非需要深入，否则尽量短。”
- “用户错了时直接指出来。”

### 适合写进 AGENTS.md
- “使用 pytest，不用 unittest。”
- “前端在 `frontend/`。”
- “不要直接改 migration。”
- “API 运行在 8000 端口。”

## 如何编辑

```bash
nano ~/.hermes/SOUL.md
```

或：

```bash
vim ~/.hermes/SOUL.md
```

然后重启 Hermes，或开启一个新会话。

## 实用工作流

1. 从默认 seed 文件开始
2. 删除那些不符合你风格的部分
3. 加入 4–8 行清楚定义语气和默认行为的内容
4. 跟 Hermes 交流一段时间
5. 根据仍然不满意的地方继续调整

这种迭代方式通常比一次性设计完美人格更有效。

## 排查

### 我改了 SOUL.md，但 Hermes 还是老样子

检查：
- 你改的是 `~/.hermes/SOUL.md` 或 `$HERMES_HOME/SOUL.md`
- 不是某个仓库里的 `SOUL.md`
- 文件不是空的
- 你是在修改后重新启动了会话
- 没有被 `/personality` 覆盖

### Hermes 忽略了 SOUL.md 的一部分

可能原因：
- 更高优先级的指令覆盖了它
- 文件里有冲突性内容
- 文件太长被截断了
- 某些内容看起来像 prompt injection，被扫描器屏蔽或改写了

### 我的 SOUL.md 变得太项目化了

把项目指令移到 `AGENTS.md`，让 `SOUL.md` 只保留身份和风格。
