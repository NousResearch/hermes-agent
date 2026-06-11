---
title: "Sketch — 一次性 HTML 原型：2-3 个设计方案对比"
sidebar_label: "Sketch"
description: "一次性 HTML 原型：2-3 个设计方案对比"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sketch

一次性 HTML 原型：2-3 个设计方案对比。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/sketch` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent（改编自 gsd-build/get-shit-done） |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `sketch`, `mockup`, `design`, `ui`, `prototype`, `html`, `variants`, `exploration`, `wireframe`, `comparison` |
| 相关 skill | [`spike`](/user-guide/skills/bundled/software-development/software-development-spike), [`claude-design`](/user-guide/skills/bundled/creative/creative-claude-design), [`popular-web-designs`](/user-guide/skills/bundled/creative/creative-popular-web-designs), [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Sketch

当用户希望**在确定方向之前先看到设计效果**时使用此 skill——以一次性 HTML 原型的形式探索 UI/UX 想法。目的是生成 2-3 个可交互的方案，让用户并排对比视觉方向，而非产出可交付的代码。

当用户说以下内容时加载此 skill："sketch this screen"、"show me what X could look like"、"compare layout A vs B"、"give me 2-3 takes on this UI"、"let me see some variants"、"mockup this before I build"。

## 不适用场景

- 用户需要生产级组件——使用 `claude-design` 或正式构建
- 用户需要精良的一次性 HTML 产物（落地页、幻灯片）——使用 `claude-design`
- 用户需要图表——使用 `excalidraw`、`architecture-diagram`
- 设计已确定——直接构建即可

## 如果用户安装了完整的 GSD 系统

如果 `gsd-sketch` 作为同级 skill 出现（通过 `npx get-shit-done-cc --hermes` 安装），优先使用 **`gsd-sketch`** 以获得完整工作流：持久化的 `.planning/sketches/` 目录（含 MANIFEST）、前沿模式分析、跨历史草图的一致性审计，以及与 GSD 其余部分的集成。本 skill 是轻量级独立版本——无状态机制的一次性草图。

## 核心方法

```
intake  →  variants  →  head-to-head  →  pick winner (or iterate)
```

### 1. Intake（如果用户已提供足够信息则跳过）

在生成方案之前，获取三项信息——每次只问一个问题，不要一次全问：

1. **感觉。** "这个应该给人什么感觉？形容词、情绪、氛围。"——*"calm, editorial, like Linear"* 比 *"minimal"* 更有参考价值。
2. **参考。** "哪些 app、网站或产品接近你想象中的感觉？"——实际参考比抽象描述更有效。
3. **核心操作。** "用户在这个页面上最重要的单一操作是什么？"——所有方案都应服务于此；否则只是装饰。

每次回答后简短复述，再问下一个问题。如果用户已一次性提供了全部三项，直接跳到方案生成。

### 2. 方案（2-3 个，不少于 1 个，极少超过 4 个）

一次性生成 **2-3 个方案**。每个方案是一个完整的独立 HTML 文件。不要描述方案——直接构建。目的是对比。

每个方案应采取**不同的设计立场**，而非不同的像素值。三种有效的方案维度：

- **密度：** 紧凑 / 宽松 / 极密（选两个对比极端）
- **重点：** 内容优先 / 操作优先 / 工具优先
- **美学：** 编辑风格 / 实用主义 / 趣味性
- **布局：** 单列 / 侧边栏 / 分屏
- **基调：** 卡片式 / 纯内容 / 文档风格

选定一个维度并从中拉开差距。两个仅在强调色上不同的方案是无效的——用户无法区分。

**方案命名：** 描述立场，而非编号。

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

### 3. 制作真实的 HTML

每个方案是一个**单一自包含的 HTML 文件**：

- 内联 `<style>`——无需构建步骤，无外部 CSS
- 系统字体或通过 `<link>` 引入一个 Google Font
- 通过 CDN 使用 Tailwind（`<script src="https://cdn.tailwindcss.com"></script>`）可以
- 真实的虚假内容——实际句子、实际姓名，而非"Lorem ipsum"
- **可交互**：链接可点击，悬停效果真实，至少一个状态转换（展开/收起、筛选、切换）。一个冻结的静态图比一个粗糙但有动效的方案更差。

在浏览器中打开验证。如果看起来有问题，在展示给用户之前修复。

**使用 Hermes 的浏览器工具对方案进行视觉验证。** 不要只写 HTML 然后寄希望于它能正常渲染；加载每个方案并查看：

```
browser_navigate(url="file:///absolute/path/to/sketches/001-calm-editorial/index.html")
browser_vision(question="Does this layout look clean and readable? Any visible bugs (overlapping text, unstyled elements, broken images)?")
```

`browser_vision` 返回页面实际内容的 AI 描述及截图路径——能捕获纯源码检查遗漏的布局问题（例如字体导入静默失败、flex 容器塌陷）。修复后重新导航，直到每个方案看起来正确为止。

**快速启动用的默认 CSS reset + 系统字体栈：**

```html
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    color: #1a1a1a;
    background: #fafafa;
    line-height: 1.5;
  }
</style>
```

### 4. 方案 README

每个方案的 `README.md` 回答以下内容：

```markdown
## Variant: {stance name}

### Design stance
One sentence on the principle driving this variant.

### Key choices
- Layout: ...
- Typography: ...
- Color: ...
- Interaction: ...

### Trade-offs
- Strong at: ...
- Weak at: ...

### Best for
- The kind of user or use case this variant actually serves
```

### 5. 正面对比

所有方案构建完成后，以对比形式呈现。不要只是罗列——**给出观点**：

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

让用户选出胜出方案，或将两个方案合并为混合版，或要求新一轮迭代。

## 主题化（当项目有视觉标识时）

如果用户有现有主题（颜色、字体、token），将共享 token 放入 `sketches/themes/tokens.css` 并在每个方案中 `@import`。保持 token 精简：

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

不要对一次性草图过度 token 化——三种颜色加一种字体通常已足够。

## 交互基准

当用户能够完成以下操作时，草图的交互程度即为合格：

1. **点击主要操作**并看到可见的变化（状态变更、模态框、toast、导航模拟）
2. **看到一个有意义的状态转换**（筛选列表、切换模式、展开/收起面板）
3. **悬停可识别的交互元素**（按钮、行、标签页）

超过此程度是对一次性草图的过度工程化。低于此程度则只是截图。

## 前沿模式（决定下一步草图内容）

如果草图已存在且用户询问"接下来应该草图什么？"：

- **一致性缺口**——来自不同草图的两个胜出方案做出了独立选择，尚未组合在一起
- **未草图的页面**——被引用但从未探索过
- **状态覆盖**——已草图了正常路径，但未覆盖空状态 / 加载中 / 错误 / 千条数据
- **响应式缺口**——在某一视口下验证过；在移动端 / 超宽屏下是否成立？
- **交互模式**——静态布局已存在；过渡动效、拖拽、滚动行为尚未探索

提出 2-4 个命名候选项，让用户选择。

## 输出

- 在仓库根目录创建 `sketches/`（如果用户使用 GSD 约定则为 `.planning/sketches/`）
- 每个方案一个子目录：`NNN-stance-name/index.html` + `README.md`
- 告知用户如何打开：macOS 上用 `open sketches/001-calm-editorial/index.html`，Linux 上用 `xdg-open`，Windows 上用 `start`
- 保持方案的一次性特性——如果你觉得有必要保留某个草图，应将其提升为真实项目代码，而非作为资产保管

**单个方案的典型工具调用序列：**

```
terminal("mkdir -p sketches/001-calm-editorial")
write_file("sketches/001-calm-editorial/index.html", "<!doctype html>...")
write_file("sketches/001-calm-editorial/README.md", "## Variant: Calm editorial\n...")
browser_navigate(url="file://$(pwd)/sketches/001-calm-editorial/index.html")
browser_vision(question="How does this look? Any obvious layout issues?")
```

对每个方案重复上述步骤，然后呈现对比表格。

## 致谢

改编自 GSD（Get Shit Done）项目的 `/gsd-sketch` 工作流——MIT © 2025 Lex Christopherson（[gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)）。完整 GSD 系统提供持久化草图状态、主题/方案模式参考及一致性审计工作流；通过 `npx get-shit-done-cc --hermes --global` 安装。