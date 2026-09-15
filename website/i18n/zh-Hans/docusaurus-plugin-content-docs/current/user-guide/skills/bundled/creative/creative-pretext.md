---
title: "Pretext"
sidebar_label: "Pretext"
description: "适用于使用 @chenglou/pretext 构建创意浏览器演示 —— 无 DOM 文本布局，用于 ASCII 艺术、排版绕障流动、文字即几何游戏、动态排版及文字驱动的生成艺术。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pretext

适用于使用 @chenglou/pretext 构建创意浏览器演示 —— 无 DOM 文本布局，用于 ASCII 艺术、排版绕障流动、文字即几何游戏、动态排版及文字驱动的生成艺术。默认生成单文件 HTML 演示。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/pretext` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `creative-coding`, `typography`, `pretext`, `ascii-art`, `canvas`, `generative`, `text-layout`, `kinetic-typography` |
| 相关 skill | [`p5js`](/user-guide/skills/bundled/creative/creative-p5js), [`claude-design`](/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Pretext 创意演示

## 概述

[`@chenglou/pretext`](https://github.com/chenglou/pretext) 是由 Cheng Lou（React 核心团队、ReasonML、Midjourney）开发的 15KB 零依赖 TypeScript 库，用于**无 DOM 多行文本测量与布局**。它只做一件事：给定 `(text, font, width)`，返回换行位置、每行宽度、每个字形（grapheme）的坐标以及总高度 —— 全部通过 canvas 测量完成，无需触发重排（reflow）。

听起来像底层管道，但并非如此。由于它快速且几何化，它是一个**创意原语**：你可以在 60fps 下让段落绕着移动的精灵重排，构建关卡几何体由真实文字组成的游戏，将 ASCII logo 嵌入散文，利用精确的每字形起始坐标将文字炸裂成粒子，或者在不调用任何 `getBoundingClientRect` 的情况下打包紧凑的多行 UI。

此 skill 的存在是为了让 Hermes 能用它制作**酷炫演示** —— 那种人们会发到 X 上的作品。社区演示库请见 `pretext.cool` 和 `chenglou.me/pretext`。

## 使用时机

当用户要求以下内容时使用：
- "pretext 演示" / "酷炫的 pretext 作品" / "文字即 X"
- 文字绕移动形状流动（hero 区块、编辑排版、动态长文页面）
- 使用**真实文字或散文**（而非等宽字符光栅）的 ASCII 艺术效果
- 游戏场地 / 障碍物 / 砖块由文字构成的游戏（字母版俄罗斯方块、散文版打砖块）
- 带有每字形物理效果的动态排版（碎裂、散射、群集、流动）
- 排版生成艺术，尤其是非拉丁文字或混合文字
- 多行"紧缩包裹"UI（能容纳文字的最小容器宽度）
- 任何需要在渲染**前**知道换行位置的场景

不适用于：
- CSS 已能解决布局的静态 SVG/HTML 页面 —— 直接用 CSS
- 富文本编辑器、通用内联格式化引擎（pretext 有意保持功能单一）
- 图片转文字（使用 `ascii-art` / `ascii-video` skill）
- 文字不起核心作用的纯 canvas 生成艺术 —— 使用 `p5js`

## 创意标准

这是在浏览器中渲染的视觉艺术。Pretext 返回数字；**你**来绘制内容。

- **不要交付"hello world"演示。** `hello-orb-flow.html` 模板只是*起点*。每个交付的演示都必须加入有意为之的色彩、动效、构图，以及一个用户没有要求但会欣赏的视觉细节。
- **深色背景、暖色核心、精心调配的色板。** 经典的琥珀色配黑色（CRT / 终端风）可行，冷白配炭灰（编辑风）和去饱和粉彩（risograph 风）同样可行。选定一种并坚持到底。
- **比例字体才是重点。** Pretext 的核心魅力在于"非等宽" —— 充分利用这一点。使用 Iowan Old Style、Inter、JetBrains Mono、Helvetica Neue 或可变字体。绝不使用默认无衬线字体。
- **使用真实语料，而非 lorem ipsum。** 语料库应有意义。短篇宣言、诗歌、真实源代码、发现的文本、库自身的 README —— 绝不用 `lorem ipsum`。
- **首帧即精品。** 无加载状态，无空白帧。演示打开的瞬间就必须达到可发布水准。

## 技术栈

每个演示为单个自包含 HTML 文件，无需构建步骤。

| 层级 | 工具 | 用途 |
|-------|------|---------|
| 核心 | `@chenglou/pretext`（通过 `esm.sh` CDN） | 文本测量 + 行布局 |
| 渲染 | HTML5 Canvas 2D | 字形渲染、逐帧合成 |
| 分割 | `Intl.Segmenter`（内置） | emoji / CJK / 组合字符的字形拆分 |
| 交互 | 原生 DOM 事件 | 鼠标 / 触摸 / 滚轮 —— 无框架 |

```html
<script type="module">
import {
  prepare, layout,                   // use-case 1: simple height
  prepareWithSegments, layoutWithLines,  // use-case 2a: fixed-width lines
  layoutNextLineRange, materializeLineRange, // use-case 2b: streaming / variable width
  measureLineStats, walkLineRanges,  // stats without string allocation
} from "https://esm.sh/@chenglou/pretext@0.0.6";
</script>
```

锁定版本。撰写时为 `@0.0.6` —— 如演示行为异常，请在 [npm](https://www.npmjs.com/package/@chenglou/pretext) 查看最新版本。

## 两种使用场景

几乎所有需求都归结为以下两种形态之一。两种都要掌握。

### 场景 1 —— 测量，然后用 CSS/DOM 渲染

```js
const prepared = prepare(text, "16px Inter");
const { height, lineCount } = layout(prepared, 320, 20);
```

浏览器仍负责绘制文字。Pretext 只告诉你在给定宽度下文本框的高度，**无需**读取 DOM。适用于：
- 包含换行文字的虚拟列表行高计算
- 需要精确卡片高度的瀑布流布局
- "这个标签放得下吗？"的开发时检查
- 防止远程文字加载时的布局偏移

**保持 `font` 和 `letterSpacing` 与 CSS 完全同步。** canvas 的 `ctx.font` 格式（如 `"16px Inter"`、`"500 17px 'JetBrains Mono'"`）必须与渲染 CSS 一致，否则测量结果会产生偏差。

### 场景 2 —— 自行测量*并*渲染

```js
const prepared = prepareWithSegments(text, FONT);
const { lines } = layoutWithLines(prepared, 320, 26);
for (let i = 0; i < lines.length; i++) {
  ctx.fillText(lines[i].text, 0, i * 26);
}
```

创意工作就在这里。你掌控绘制，因此可以：
- 渲染到 canvas、SVG、WebGL 或任意坐标系
- 对每个字形应用变换（旋转、抖动、缩放、透明度）
- 将行元数据（宽度、字形坐标）用作几何数据

对于**每行宽度可变**的流动排版（文字绕形状流动、文字在环形带内、文字在非矩形列中）：

```js
let cursor = { segmentIndex: 0, graphemeIndex: 0 };
let y = 0;
while (true) {
  const lineWidth = widthAtY(y);  // your function: how wide is the corridor at this y?
  const range = layoutNextLineRange(prepared, cursor, lineWidth);
  if (!range) break;
  const line = materializeLineRange(prepared, range);
  ctx.fillText(line.text, leftEdgeAtY(y), y);
  cursor = range.end;
  y += lineHeight;
}
```

这是整个库中最重要的模式。它解锁了"文字绕拖拽精灵流动"的效果 —— 那个在 X 上病毒式传播的演示。

### 值得了解的辅助函数

- `measureLineStats(prepared, maxWidth)` → `{ lineCount, maxLineWidth }` —— 最宽的行，即多行紧缩包裹宽度。
- `walkLineRanges(prepared, maxWidth, callback)` —— 无字符串分配地遍历各行。在不需要字符内容时用于统计/物理计算。
- `@chenglou/pretext/rich-inline` —— 同一系统，但支持混合字体 / 标签 / 提及的段落。从子路径导入。

## 演示配方模式

社区语料库（见 `references/patterns.md`）归纳为几种强力模式。选一种进行变奏 —— 除非被要求，否则不要发明新类别。

| 模式 | 核心 API | 示例创意 |
|---|---|---|
| **绕障重排** | `layoutNextLineRange` + 逐行宽度函数 | 编辑排版段落，绕拖拽光标精灵分开 |
| **文字即几何游戏** | `layoutWithLines` + 逐行碰撞矩形 | 每块砖都是一个测量过的单词的打砖块游戏 |
| **碎裂 / 粒子** | `walkLineRanges` → 每字形 (x,y) → 物理 | 点击时句子炸裂成字母 |
| **ASCII 障碍排版** | `layoutNextLineRange` + 逐行障碍区间测量 | 位图 ASCII logo、形态变换，以及可拖拽的线框物体，使文字绕其实际几何形状展开 |
| **编辑多栏** | 每栏 `layoutNextLineRange` + 共享游标 | 带引用块的动态杂志版面 |
| **动态排版** | `layoutWithLines` + 逐行随时间变换 | 星球大战字幕滚动、波浪、弹跳、故障效果 |
| **多行紧缩包裹** | `measureLineStats` | 自动适配最紧凑容器的引用卡片 |

可参考 `templates/donut-orbit.html` 和 `templates/hello-orb-flow.html` 中可运行的单文件起始模板。

## 工作流程

1. **根据用户需求从上表选择一种模式。**
2. **从模板开始**：
   - `templates/hello-orb-flow.html` —— 文字绕移动球体重排（绕障重排模式）
   - `templates/donut-orbit.html` —— 进阶示例：测量 ASCII logo 障碍物、可拖拽线框球体/立方体、变形形状场、可选 DOM 文字及仅开发模式控件
   - 用 `write_file` 将新 `.html` 写入 `/tmp/` 或用户工作区。
3. **将语料库替换为**与需求相关的有意义内容。真实散文，10-100 句，不用 lorem。
4. **调整美学** —— 字体、色板、构图、交互。这才是核心工作，不要跳过。
5. **本地验证**：
   ```sh
   cd <dir-with-html> && python3 -m http.server 8765
   # then open http://localhost:8765/<file>.html
   ```
6. **检查控制台** —— 若 `prepareWithSegments` 传入错误的字体字符串，pretext 会抛出异常；`Intl.Segmenter` 在所有现代浏览器中均可用。
7. **向用户展示文件路径**，而非仅展示代码 —— 他们想直接打开文件。

## 性能说明

- `prepare()` / `prepareWithSegments()` 是开销较大的调用。每个文字+字体组合只调用**一次**，缓存句柄。
- 窗口大小改变时，只重新运行 `layout()` / `layoutWithLines()` —— 绝不重新 prepare。
- 对于文字内容不变但几何形状变化的逐帧动画，在紧密循环中调用 `layoutNextLineRange` 对普通长度的段落来说足够在 60fps 下每帧执行。
- 逐帧渲染 ASCII 遮罩时，维护一个单元格缓冲区（`Uint8Array` / 类型化数组），从单元格或投影几何体推导每行障碍区间，合并区间，再将这些区间传入 `layoutNextLineRange` 后绘制文字。
- 保持视觉动画与布局动画同步。若球体变形为立方体，用同一个值对渲染单元格缓冲区和障碍区间同时做补间；否则演示看起来像贴图而非物理重排。
- 淡入淡出效果优先使用图层透明度，而非改变字形强度或障碍物缩放。将瞬态 ASCII 精灵放在独立 canvas 上，用 CSS/GSAP 的 opacity 淡化该 canvas，避免几何形状看起来在缩小。
- Canvas 的 `ctx.font` 设置出人意料地慢；若字体在帧内不变，每帧只设置**一次**，而非每次 `fillText` 调用都设置。

## 常见陷阱

1. **CSS 与 canvas 字体字符串不一致。** `ctx.font = "16px Inter"` 用于测量，但 CSS 写的是 `font-family: Inter, sans-serif; font-size: 16px`。如果 Inter 加载成功则没问题。若 Inter 404，CSS 会回退到 sans-serif，测量结果偏差 5-20%。始终 `preload` 字体，或使用 web 安全字体族。

2. **在动画循环内重复 prepare。** 只有 `layout*` 是廉价的。每帧调用 `prepare` 会严重拖慢性能。将 prepared 句柄保存在模块作用域中。

3. **忘记用 `Intl.Segmenter` 拆分字形。** Emoji、组合字符、CJK —— `"é".split("")` 会给出两个字符。在采样单个可见字形时，使用 `new Intl.Segmenter(undefined, { granularity: "grapheme" })`。

4. **`break: 'never'` 标签缺少 `extraWidth`。** 在 `rich-inline` 中，若对原子标签/提及使用 `break: 'never'`，还必须提供 `extraWidth` 用于标签内边距 —— 否则标签外框会溢出容器。

5. **从 `unpkg` 使用 `@chenglou/pretext` 时遇到 TypeScript 专属入口。** 使用 `esm.sh` —— 它会自动将 TS 导出编译为浏览器可用的 ESM。`unpkg` 会 404 或返回原始 TS。

6. **等宽字体回退悄悄抹杀了整个意义。** 用户看到等宽输出，通常是因为 CSS `font-family` 回退到了 `monospace`。通过 DevTools 验证实际渲染字体。

7. **绕形状流动时跳过行而非调整宽度。** 若当前行的通道太窄无法容纳一行，应*跳过该行*（`y += lineHeight; continue;`），而非向 `layoutNextLineRange` 传入极小的 maxWidth —— pretext 会返回单字形行，看起来很破碎。

8. **交付冷启动演示。** 默认首帧看起来像教程级别。请添加：暗角、细微扫描线、空闲自动动效、一个精心选择的交互响应（拖拽、悬停、滚动、点击）。缺少这些，"酷炫 pretext 演示"就会沦为"README 复现"。

## 验证清单

- [ ] 演示是单个自包含 `.html` 文件 —— 双击或 `python3 -m http.server` 即可打开
- [ ] `@chenglou/pretext` 通过 `esm.sh` 导入并锁定版本
- [ ] 语料库为真实散文，非 lorem ipsum，且与演示概念匹配
- [ ] 传入 `prepare` 的字体字符串与 CSS 字体完全一致
- [ ] `prepare()` / `prepareWithSegments()` 只调用一次，不在每帧调用
- [ ] 深色背景 + 精心调配的色板 —— 非默认白色 canvas
- [ ] 至少一种交互响应（拖拽 / 悬停 / 滚动 / 点击）或空闲自动动效
- [ ] 已用 `python3 -m http.server` 本地测试，确认无控制台报错
- [ ] 在中端笔记本上达到 60fps（或已记录优雅降级方案）
- [ ] 一个用户未要求的"超额"细节

## 参考：社区演示

克隆以下项目获取灵感 / 模式（均为 MIT 类许可，链接来自 [pretext.cool](https://www.pretext.cool/)）：

- **Pretext Breaker** —— 单词砖块打砖块 —— `github.com/rinesh/pretext-breaker`
- **Tetris × Pretext** —— `github.com/shinichimochizuki/tetris-pretext`
- **Dragon animation** —— `github.com/qtakmalay/PreTextExperiments`
- **Somnai editorial engine** —— `github.com/somnai-dreams/pretext-demos`
- **Bad Apple!! ASCII** —— `github.com/frmlinn/bad-apple-pretext`
- **Drag-sprite reflow** —— `github.com/dokobot/pretext-demo`
- **Alarmy editorial clock** —— `github.com/SmisLee/alarmy-pretext-demo`

官方演示场：[chenglou.me/pretext](https://chenglou.me/pretext/) —— 手风琴、气泡、动态布局、编辑引擎、对齐比较、瀑布流、Markdown 聊天、富文本笔记。