---
title: "Sketch — 一次性HTML原型：2-3个设计变体用于比较"
sidebar_label: "Sketch"
description: "一次性HTML原型：2-3个设计变体用于比较"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sketch

一次性HTML原型：2-3个设计变体用于比较。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/sketch` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent (从gsd-build/get-shit-done改编） |
| 许可证 | MIT |
| 标签 | `sketch`, `mockup`, `design`, `ui`, `prototype`, `html`, `variants`, `exploration`, `wireframe`, `comparison` |
| 相关技能 | [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Sketch

当用户想要**在承诺一个设计方向之前查看它**时使用此技能——将UI/UX创意探索为一次性HTML原型。重点是生成2-3个交互式变体，以便用户可以并排比较视觉方向，而不是生成可交付的代码。

当有如下表述时加载此技能："sketch this screen"、"show me what X could look like"、"compare layout A vs B"、"give me 2-3 takes on this UI"、"let me see some variants"、"mockup this before I build"。

## 何时*不*使用此技能

- 用户想要生产组件——使用`claude-design`或正确构建它
- 用户想要打磨的一次性HTML产物（着陆页、演示文稿）——`claude-design`
- 用户想要图表—— `excalidraw`、`architecture-diagram`
- 设计已锁定——直接构建它

## 如果用户的GSD系统已完整安装

如果`gsd-sketch`作为同级技能显示（通过`npx get-shit-done-cc --hermes`安装），
  对于完整工作流**优选`gsd-sketch`**：持久化`.planning/sketches/`带有MANIFEST、前沿模式分析、跨过去sketch的一致性审计，以及与GSD其余部分的集成。此技能是轻量级独立版本——一次性sketching而无需状态机制。

## 核心方法

```
intake  →  variants  →  head-to-head  →  pick winner (or iterate)
```

### 1. Intake（如果用户提供足够信息可跳过）

在生成变体之前，获取三件事——一次一个问题，而不是一次性全部：

1. **感觉。** "这应该感觉像什么？形容词、情绪、一种氛围。" —— *"平静、编辑式、像Linear"* 告诉您的比* "极简"* 更多。
2. **参考。** "哪些应用程序、网站或产品捕捉到您想象的感受？" —— 实际参考比抽象描述更好。
3. **核心动作。** "在此屏幕上用户做的单一最重要的事情是什么？" —— 变体都应该很好地服务于它；如果它们不，它们就只是装饰。

在下一个问题之前简要反思每个答案。如果用户已经一次性提供了这三个，直接跳到变体。

### 2. Variants（2-3个，永远不要1个，很少4+）

**一次性生成2-3个变体。** 每个变体是一个完整的独立HTML文件。不要描述变体——构建它们。重点是*比较*。

每个变体应采取**不同的设计立场**，而不是不同的像素值。三个好的变体轴：

- **密度：** 紧凑/通风/超密集（选择两个对比极端）
- **强调：** 内容优先/动作优先/工具优先
- **审美：** 编辑式/实用/俏皮
- **布局：** 单列/侧边栏/分割窗格
- **基础：** 基于卡片/裸内容/文档风格

选择其中一个轴并从它拉开。两个仅在强调颜色上不同的变体是浪费精力——用户无法区分它们。

**变体命名：** 描述立场，而不是数字。

<!-- ascii-guard-ignore -->
```
sketches/
├── 001-calm-editorial/
│   ├── index.html
│   └── README.md
├── 001-utilitarian-dense/
│   ├── index.html
│   └── README.md
└── 001-playful-split/
    ├── index.html
    └── README.md
```
<!-- ascii-guard-ignore-end -->

### 3. 使它们成为真实的HTML

每个变体都是一个**单一自包含的HTML文件**：

- 内联`<style>` —— 无构建步骤，无外部CSS
- 系统字体或通过`<link>`的一个Google Font
- 通过CDN的Tailwind（`<script src="https://cdn.tailwindcss.com"></script>`）也可以
- 逼真的假内容——实际句子、实际名称，而不是"Lorem ipsum"
- **交互式**：链接可点击，悬停真实，至少一个状态转换（打开/关闭、筛选、切换）。冻结的静态图像是比拙劣的动画更糟的spik。

在浏览器中打开它。如果看起来坏了，在展示给用户之前修复它。

**可视化验证变体——使用Hermes的浏览器工具。** 不要只是编写HTML并希望它渲染；加载每个变体并查看它：

```
browser_navigate(url="file:///absolute/path/to/sketches/001-calm-editorial/index.html")
browser_vision(question="Does this layout look clean and readable? Any visible bugs (overlapping text, unstyled elements, broken images)?")
```

`browser_vision`返回页面实际内容的AI描述加上截图路径——捕获纯源检查遗漏的布局错误（例如，字体导入静默失败、flex容器折叠）。修复并重新导航直到每个变体看起来正确。

**默认CSS重置+系统字体堆栈**用于快速启动：

```html
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialised;
    color: #1a1a1a;
    background: #fafafa;
    line-height: 1.5;
  }
</style>
```

### 4. 变体README

每个变体的`README.md`回答：

```markdown
## Variant: {stance name}

### Design stance
一句话关于驱动此变体的原则。

### Key choices
- Layout: ...
- Typography: ...
- Color: ...
- Interaction: ...

### Trade-offs
- Strong at: ...
- Weak at: ...

### Best for
- 此变体实际服务的那种用户或用例
```

### 5. Head-to-head

构建所有变体后，将它们作为比较呈现。不要只是列出——**要有观点**：

```markdown
## Three takes on the home screen

| Dimension | Calm editorial | Utilitarian dense | Playful split |
|-----------|----------------|-------------------|---------------|
| Density   | Low            | High              | Medium        |
| Primary action visibility | Low | High | Medium |
| Scan-ability | High | Medium | Low |
| Feel | Calm, trusted | Sharp, tool-like | Inviting, energetic |

**My take:** Utilitarian dense for power users, calm editorial for content-forward audiences. Playful split is weakest — tries to do both and commits to neither.
```

让用户选择赢家，或将两个合并为混合体，或再请求一轮。

## 主题（当项目有视觉标识时）

如果用户有现有主题（颜色、字体、令牌），将共享令牌放在`sketches/themes/tokens.css`中并在每个变体中`@import`它们。保持令牌最小化：

```css
/* sketches/themes/tokens.css */
:root {
  --color-bg: #fafafa;
  --color-fg: #1a1a1a;
  --color-accent: #0066ff;
  --color-muted: #666;
  --radius: 8px;
  --font-display: "Inter", sans-serif;
  --font-body: -apple-system, BlinkMacSystemFont, sans-serif;
}
```

不要为一次性sketch过度令牌化——通常三种颜色和一种字体就足够了。

## 交互栏

当草图具有以下特征时，其交互性足够：

1. **点击主要动作** 并有可见的事情发生（状态改变、模态、吐司、导航假装）
2. **看到一个有意义的状态转换**（筛选列表、切换模式、打开/关闭面板）
3. **悬停可识别的启示** （按钮、行、标签页）

多于那是在过度工程化一次性草图。少于那是一个截图。

## 前沿模型（选择接下来要sketch什么）

如果草图已存在且用户说"我接下来应该sketch什么？"：

- **一致性差距** — 来自不同sketch的两个赢家做出了尚未组合在一起的正确选择
- **未sketch的屏幕** — 被引用但从未探索过
- **状态覆盖** — 快乐路径已sketch，但未sketch空/加载/错误/1000个项目
- **响应式差距** — 在一个视口验证；它在移动端/超宽屏上是否保持不变？
- **交互模式** — 静态布局存在；转换、拖动、滚动行为不

提出2-4个命名候选项。让用户选择。

## 输出

- 在仓库根目录创建`sketches/`（或如果用户对GSD约定，创建`.planning/sketches/`）
- 每个变体一个子目录：`NNN-stance-name/index.html` + `README.md`
- 告诉用户如何打开它们：`open sketches/001-calm-editorial/index.html`（macOS上），`xdg-open`（Linux上），`start`（Windows上）
- 保持变体可丢弃——您认为需要保留的sketch应提升为真实项目代码，而不是作为资产策展

**一个变体的典型工具序列：**

```
terminal("mkdir -p sketches/001-calm-editorial")
write_file("sketches/001-calm-editorial/index.html", "<!doctype html>...")
write_file("sketches/001-calm-editorial/README.md", "## Variant: Calm editorial\n...")
browser_navigate(url="file://$(pwd)/sketches/001-calm-editorial/index.html")
browser_vision(question="How does this look? Any obvious layout issues?")
```

对每个变体重复，然后呈现比较表。

## 归属

从GSD（Get Shit Done）项目的`/gsd-sketch`工作流改编——MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done))。完整的GSD系统提供持久化sketch状态、主题/变体模式参考和一致性审计工作流；使用`npx get-shit-done-cc --themes --global`安装。
