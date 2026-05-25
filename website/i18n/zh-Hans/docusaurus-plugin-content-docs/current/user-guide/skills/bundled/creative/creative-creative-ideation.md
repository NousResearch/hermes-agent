---
title: "创意构思 — 通过创意约束生成项目想法"
sidebar_label: "创意构思"
description: "通过创意约束生成项目想法"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 创意构思

通过创意约束生成项目想法。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/creative-ideation` |
| 版本 | `1.0.0` |
| 作者 | SHL0MS |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Creative`, `Ideation`, `Projects`, `Brainstorming`, `Inspiration` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 创意构思

## 使用时机

当用户说"我想做点什么"、"给我一个项目想法"、"我很无聊"、"我该做什么"、"给我一些灵感"，或任何类似"我有工具但没有方向"的表达时使用。适用于代码、艺术、硬件、写作、工具，以及任何可以被创造出来的事物。

通过创意约束（constraint）生成项目想法。约束 + 方向 = 创造力。

## 工作原理

1. **从下方约束库中选取一个约束** — 随机选取，或根据用户的领域/心情匹配
2. **广义解读** — 一个编程 prompt 可以变成硬件项目，一个艺术 prompt 可以变成 CLI 工具
3. **生成 3 个满足约束的具体项目想法**
4. **如果用户选定了一个，就开始构建** — 创建项目、编写代码、发布上线

## 规则

每个 prompt 都尽可能广义地解读。"这包括 X 吗？"→ 是的。prompt 提供方向和适度约束。没有这两者，就没有创造力。

## 约束库

### 面向开发者

**解决自己的痛点：**
构建你这周希望存在的工具。50 行以内。今天就发布。

**自动化那件烦人的事：**
你工作流中最繁琐的部分是什么？用脚本解决它。花两小时修复一个每天让你浪费五分钟的问题。

**那个本该存在的 CLI 工具：**
想想你希望能输入的命令。`git undo-that-thing-i-just-did`。`docker why-is-this-broken`。`npm explain-yourself`。现在把它做出来。

**除了胶水什么都不新：**
完全用现有的 API、库和数据集做点东西。唯一的原创贡献是你连接它们的方式。

**弗兰肯斯坦周：**
拿一个做 X 的东西，让它做 Y。一个能播放音乐的 git 仓库。一个能生成诗歌的 Dockerfile。一个发送赞美的 cron job。

**做减法：**
在代码库崩溃之前你能删掉多少？把一个工具精简到最小可用功能。一直删，直到只剩本质。

**高概念，低投入：**
一个深刻的想法，随意地实现。概念应该很精彩。实现应该只需要一个下午。如果花的时间更长，说明你想太多了。

### 面向创客与艺术家

**厚颜无耻地抄：**
选一个你欣赏的东西 — 一个工具、一件艺术品、一个界面。从头重新创作它。学习就在你的版本与原版之间的差距里。

**一百万个某物：**
一百万既多又不多。一百万像素是一张 1MB 的照片。一百万次 API 调用是某个普通的周二。任何东西达到一百万的规模都会变得有趣。

**做一个会死的东西：**
一个每天失去一个功能的网站。一个会遗忘的聊天机器人。一个倒计时到虚无的东西。关于腐烂、终结或放手的练习。

**做大量数学：**
生成式几何、shader golf、数学艺术、计算折纸。是时候重新学一下 arcsin 是什么了。

### 面向所有人

**文本是通用界面：**
构建一个文本是唯一界面的东西。没有按钮，没有图形，只有文字进文字出。文本几乎可以进出任何东西。

**从结语开始：**
想一个会成为有趣句子的东西。倒推着把它变成现实。"我教会了我的恒温器来煤气灯效应我" → 现在把它做出来。

**恶意 UI：**
做一个故意让人痛苦的东西。一个需要满足 47 个条件的密码框。一个每个标签都在撒谎的表单。一个评判你命令的 CLI。

**再来一次：**
回想一个旧项目。从头再做一遍。不要看原版。看看你的思维方式发生了什么变化。

更多约束请参见 `references/full-prompt-library.md`，涵盖沟通、规模、哲学、转化等 30+ 个约束。

## 将约束与用户匹配

| 用户说 | 从以下选取 |
|-----------|-----------|
| "我想做点什么"（没有方向） | 随机 — 任意约束 |
| "我在学 [语言]" | 厚颜无耻地抄、自动化那件烦人的事 |
| "我想要奇怪的东西" | 恶意 UI、弗兰肯斯坦周、从结语开始 |
| "我想要有用的东西" | 解决自己的痛点、那个本该存在的 CLI、自动化那件烦人的事 |
| "我想要美的东西" | 做大量数学、一百万个某物 |
| "我精疲力竭了" | 高概念低投入、做一个会死的东西 |
| "周末项目" | 除了胶水什么都不新、从结语开始 |
| "我想要挑战" | 一百万个某物、做减法、再来一次 |

## 输出格式

```
## 约束：[名称]
> [约束，一句话]

### 想法

1. **[一句话概括]**
   [2-3 句话：你要构建什么以及为什么有趣]
   ⏱ [周末 / 一周 / 一个月] • 🔧 [技术栈]

2. **[一句话概括]**
   [2-3 句话]
   ⏱ ... • 🔧 ...

3. **[一句话概括]**
   [2-3 句话]
   ⏱ ... • 🔧 ...
```

## 示例

```
## Constraint: The CLI tool that should exist
> Think of a command you've wished you could type. Now build it.

### Ideas

1. **`git whatsup` — show what happened while you were away**
   Compares your last active commit to HEAD and summarizes what changed,
   who committed, and what PRs merged. Like a morning standup from your repo.
   ⏱ weekend • 🔧 Python, GitPython, click

2. **`explain 503` — HTTP status codes for humans**
   Pipe any status code or error message and get a plain-English explanation
   with common causes and fixes. Pulls from a curated database, not an LLM.
   ⏱ weekend • 🔧 Rust or Go, static dataset

3. **`deps why <package>` — why is this in my dependency tree**
   Traces a transitive dependency back to the direct dependency that pulled
   it in. Answers "why do I have 47 copies of lodash" in one command.
   ⏱ weekend • 🔧 Node.js, npm/yarn lockfile parsing
```

用户选定一个后，开始构建 — 创建项目、编写代码、持续迭代。

## 致谢

约束方法灵感来源于 [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html)。已针对软件开发和通用创意构思进行改编和扩展。