---
title: "Pretext"
sidebar_label: "Pretext"
description: "使用@chenglou/pretext构建创意浏览器演示 — 无DOM文本布局用于ASCII艺术、印刷流环绕障碍、文本作为几何游戏..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pretext

使用@chenglou/pretext构建创意浏览器演示 — 无DOM文本布局用于ASCII艺术、印刷流环绕障碍、文本作为几何游戏、动态排版和文本驱动的生成艺术。默认生成单文件HTML演示。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/pretext` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `creative-coding`, `typography`, `pretext`, `ascii-art`, `canvas`, `generative`, `text-layout`, `kinetic-typography` |
| 相关技能 | [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Pretext创意演示

## 概述

[@chenglou/pretext](https://github.com/chenglou/pretext)是由Cheng Lou（React核心、ReasonML、Midjourney）开发的15KB零依赖TypeScript库，用于**无DOM多行文本测量和布局**。它做一件事：给定`(文本, 字体, 宽度)`，返回换行、每行宽度、每个字素位置以及总高度——所有这些都通过画布测量完成，无需重排。

这听起来像管道工程。它不是。因为它快速且几何，它是一个**创意原语**：您可以以60fps的速度围绕移动精灵重新排版段落、构建游戏场/障碍/砖块由真实单词组成、通过文章驱动ASCII徽标、将文本粉碎为具有每个字素起始位置的粒子，或包装收缩包装的多行UI而无需任何`getBoundingClientRect`垃圾回收。

这个技能的存在是为了让Hermes可以用它制作**酷炫演示** —— 人们发布到X的那种。参见`pretext.cool`和`chenglou.me/pretext`获取社区演示语料库。

## 使用场景

当用户请求以下内容时使用：
- 一个"pretext演示" / "酷pretext东西" / "文本作为-X"
- 围绕移动形状流动的文本（英雄区块、编辑布局、动画长格式页面）
- 使用**真实单词或文章**的ASCII艺术效果，而不是等宽光栅
- 游戏场/障碍/砖块由文本组成（字母构成的俄罗斯方块、文章构成的打砖块）
- 具有每个字形物理的动力学排版（粉碎、分散、集群、流动）
- 印刷生成艺术，特别是与非拉丁脚本或混合脚本
- 多行"收缩包装"UI（最小容器宽度仍然适合文本）
- 在渲染**之前**需要知道换行任何情况

不要用于：
- 静态SVG/HTML页面，其中CSS已经解决布局 —— 只需使用CSS
- 富文本编辑器、通用内联格式化引擎（pretext有意为狭窄）
- 图像 → 文本（使用`ascii-art` / `ascii-video`技能）
- 没有文本角色作用的纯画布生成艺术 —— 使用`p5js`

## 创意标准

这是在浏览器中渲染的视觉艺术。Pretext返回数字；**您**绘制东西。

- **不要交付"hello world"演示。** `hello-orb-flow.html`模板是*起点*。每个交付的演示必须添加有意的颜色、运动、构图，以及一个用户没有要求但会欣赏的视觉细节。
- **暗色背景、温暖核心、经过考虑的调色板。** 经典的琥珀色在黑色上（CRT / 终端）有效，冷白色在木炭上（编辑）和去饱和的粉彩（risograph）也有效。选择一个并投入。
- **比例字体是重点。** Pretext的整体氛围是"不是等宽" —— 倾斜于它。使用爱尔兰古式、Inter、JetBrains Mono、Helvetica Neue或可变字体。永远不要使用默认无衬线。
- **真实的来源/文本，而不是lorem ipsum。** 语料库应该有某种含义。短宣言、诗歌、真实源代码、发现的文本、库自己的README —— 永远不要使用`lorem ipsum`。
- **首次绘制卓越。** 没有加载状态，没有空白帧。演示必须在打开时就看起来可交付。

## 技术栈

每个演示一个自包含的HTML文件。无需构建步骤。

| 层级 | 工具 | 用途 |
|-------|------|---------|
| 核心 | `@chenglou/pretext` 通过`esm.sh` CDN | 文本测量+行布局 |
| 渲染 | HTML5 Canvas 2D | 字形渲染、逐帧构图 |
| 分词 | `Intl.Segmenter`（内置） | 字素分割用于emoji / CJK / 组合标记 |
| 交互 | 原始DOM事件 | 鼠标/触摸/滚轮 —— 无框架 |

```html
<script type="module">
import {
  prepare, layout,                   // use-case 1: 简单高度
  prepareWithSegments, layoutWithLines,  // use-case 2a: 固定宽度行
  layoutNextLineRange, materializeLineRange, // use-case 2b: 流式的/可变宽度
  measureLineStats, walkLineRanges,  // 无字符串分配统计
} from "https://esm.sh/@chenglou/pretext@0.0.6";
</script>
```

固定版本。撰写时的`@0.0.6` —— 如果演示行为不正常，请查看 [npm](https://www.npmjs.com/package/@chenglou/pretext)获取最新版本。

## 两种使用场景

几乎所有内容都可以简化为这两种形状之一。学习两者。

### 使用场景1 — 测量，然后与CSS/DOM一起渲染

```js
const prepared = prepare(text, "16px Inter");
const { height, lineCount } = layout(prepared, 320, 20);
```

您仍然让浏览器绘制文本。Pretext只是告诉您在给定的宽度下框会有多高，**无需**DOM读取。用于：
- 虚拟化列表，其中行包含 wrapping文本
- 具有精确卡片高度的砌体
- "这个标签适合吗？"开发时检查
- 防止远程文本加载时的布局偏移

**保持`font`和`letterSpacing`与您的CSS完全同步。** 画布`ctx.font`格式（例如`"16px Inter"`、`"500 17px 'JetBrains Mono'"`）必须与渲染的CSS匹配，否则测量会漂移。

### 使用场景2 — 测量*并且*自己渲染

```js
const prepared = prepareWithSegments(text, FONT);
const { lines } = layoutWithLines(prepared, 320, 26);
for (let i = 0; i < lines.length; i++) {
  ctx.fillText(lines[i].text, 0, i * 26);
}
```

这是创意工作所在的地方。您拥有绘图，所以您可以：
- 渲染到画布、SVG、WebGL或任何坐标系统
- 替换每个字形变换（旋转、抖动、缩放、不透明度）
- 使用行元数据（宽度、字素位置）作为几何

对于**每行的可变宽度**流（文本环绕形状、甜甜圈带中的文本、非矩形列中的文本）：

```js
let cursor = { segmentIndex: 0, graphemeIndex: 0 };
let y = 0;
while (true) {
  const lineWidth = widthAtY(y);  // 您的函数：此y处的走廊有多宽？
  const range = layoutNextLineRange(prepared, cursor, lineWidth);
  if (!range) break;
  const line = materializeLineRange(prepared, range);
  ctx.fillText(line.text, leftEdgeAtY(y), y);
  cursor = range.end;
  y += lineHeight;
}
```

这是整个库中*最重要的模式*。它解锁了"文本环绕拖动的精灵" —— 在X上病毒式传播的演示。

### 值得了解的辅助函数

- `measureLineStats(prepared, maxWidth)` → `{ lineCount, maxLineWidth }` —— 最宽行，即多行收缩包装宽度。
- `walkLineRanges(prepared, maxWidth, callback)` —— 迭代行而无需分配字符串。当您不需要字符时，用于字形上的统计/物理。
- `@chenglou/pretext/rich-in-line` —— 相同的系统但用于混合字体/筹码/提及的段落。从子路径导入。

## 演示食谱模式

社区语料库（参见`references/patterns.md`）聚集为少数强大模式。选择一个并即兴发挥 —— 不要发明新类别，除非被要求。

| 模式 | 关键API | 示例创意 |
|---|---|---|
| **环绕障碍回流** | `layoutNextLineRange` + 每行函数 | 编辑段落部分环绕拖动的鼠标精灵 |
| **文本作为几何游戏** | `layoutWithLines` + 每行碰撞矩形 | 打砖块，其中每个砖块是一个测量的单词 |
| **粉碎/粒子** | `walkLineRanges` → 每字素(x,y) → 物理 | 点击时句子爆炸为字母 |
| **ASCII障碍印刷** | `layoutNextLineRange` + 测量的每行障碍跨度 | 位图ASCII徽标、形状变形，以及可拖动的线对象使文本环绕其实际几何 |
| **编辑多列** | 每列的`layoutNextLineRange` + 共享光标 | 动画杂志跨页，带有拉引引用 |
| **动力学类型** | `layoutWithLines` + 随时间每行的变换 | 星球大战爬行、波浪、弹跳、故障 |
| **多行收缩包装** | `measureLineStats` | 自动调整为最紧容器大小的引用卡 |

查看`templates/donut-orbit.html`和`templates/hello-orb-flow.html`获取可用的单文件启动器。

## 工作流程

1. **从表中选择一个模式** 基于用户的简报。
2. **从模板启动**：
   - `templates/hello-orb-flow.html` — 文本环绕移动球（回流-环绕-障碍模式）
   - `templates/donut-orbit.html` — 高级示例：测量的ASCII徽标障碍、可拖动的线球体/立方体、变形形状字段、可选DOM文本以及仅开发控件
   - 使用`write_file`到新的`.html`到`/tmp/`或用户的工作空间。
3. **交换语料库** 为简报意向的内容。真实文章，10-100个句子，不要lorem。
4. **调优审美** — 字体、调色板、构图、交互。这是工作；不要跳过它。
5. **本地验证**：
   ```sh
   cd <包含html的目录> && python3 -m http.server 8765
   # 然后打开 http://localhost:8765/<文件>.html
   ```
6. **检查控制台** — 如果调用带有坏字体字符串的`prepareWithSegments`，pretext将抛出；所有现代浏览器都可用`Intl.Segmenter`。
7. **向用户显示文件路径**，而不仅仅是代码 —— 他们想要打开它。

## 性能说明

- `prepare()` / `prepareWithSegments()`是昂贵的调用。为每个文本+字体对执行**一次**。缓存句柄。
- 在调整大小时，仅重新运行`layout()` / `layoutWithLines()` —— 永远不要重新准备。
- 对于文本不改变但几何改变的逐帧动画，`layoutNextLineRange`在紧凑循环中对正常长度段落每帧执行足够便宜。
- 对于逐帧的ASCII遮罩，保留单元格缓冲区（`Uint8Array`/类型化数组），从单元格或投影几何派生测量的每行障碍跨度，合并跨度，然后在绘制文本之前将这些跨度馈送到`layoutNextLineRange`。
- 将视觉动画和布局动画耦合。如果球体变形为立方体，请用相同的值处理渲染的单元格缓冲区和障碍跨度；否则演示看起来像画上去的，而不是物理回流的。
- 对于淡入淡出，更喜欢图层不透明度而不是改变字形强度或障碍比例。将瞬态ASCII精灵放在自己的画布上，并通过CSS/GSAP不透明度淡化画布，以便几何看起来不会缩小。
- 画布`ctx.font`设置非常慢；如果字体不变化，每帧设置**一次**，而不是每次`fillText`调用。

## 常见陷阱

1. **漂移的CSS/画布字体字符串。** `ctx.font = "16px Inter"`测量，但CSS显示`font-family: Inter, sans-serif; font-size: 16px`。如果Inter加载，这很好*。*如果Inter 404，CSS回退到无衬线，测量漂移5-20%。始终`preload`字体或使用网络安全系列。
2. **在动画循环内重新准备。** 只有`layout*`是便宜的。每帧重新调用`prepare`将破坏性能。将准备好的句柄保留在模块作用域中。
3. **忘记用于字素拆分的`Intl.Segmenter`。** Emoji、组合标记、CJK —— `"é".split("")`给您两个字符。当采样单个可见字形时，使用`new Intl.Segmenter(undefined, { granularity: "grapheme" })`。
4. **`break: 'never'`芯片没有`extraWidth`。** 在`rich-in-line`中，如果您对原子芯片/提及使用`break: 'never'`，您还必须提供`extraWidth`用于 pill 填充 —— 否则芯片铬会溢出容器。
5. **从`unpkg`使用带有TypeScript唯一入口的`@chenglou/pretext`。** 使用`esm.sh` —— 它会将TS导出自动编译为浏览器就绪的ESM。`unpkg`将404或提供原始TS。
6. **等宽回退静默擦除整个重点。** 看到等宽输出外观的用户通常具有回退到`monospace`的CSS`font-family`。通过DevTools验证实际渲染的字体。
7. **环绕形状时跳过行与调整宽度。** 如果此行的走廊对于适合行来说太窄，*跳过行*`(y += lineHeight; continue;)`而不是向`layoutNextLineRange`传递微小的maxWidth —— pretext将返回单字素行，看起来像坏了一样。
8. **交付冷漠的演示。** 默认首次绘制看起来像教程级别。添加：暗角、微妙的扫描线、空闲自动运动、一个仔细选择的交互响应（拖动、悬停、滚动、点击）。没有这些，"酷pretext演示"就会变成"README的实习生复制"。

## 验证清单

- [ ] 演示是单个自包含的`.html`文件 —— 通过双击或`python3 -m http.server`打开
- [ ] `@chenglou/pretext`通过带有固定版本的`esm.sh`导入
- [ ] 语料库是真实的文章，而不是lorem ipsum，并且与演示的概念匹配
- [ ] 传递给`prepare`的字体字符串与CSS字体完全匹配
- [ ] `prepare()` / `prepareWithSegments()`调用一次，而不是每帧
- [ ] 暗色背景 + 经过考虑的调色板 —— 不是默认的白色画布
- [ ] 至少一种交互响应（拖动/悬停/滚动/点击）或空闲自动运动
- [ ] 使用`python3 -m http.server`本地测试，并确认没有控制台错误
- [ ] 在中端笔记本电脑上达到60fps（或优雅降级记录）
- [ ] 一个用户没有要求的"额外里程"细节

## 参考：社区演示

克隆这些以获取灵感/模式（全部MIT许可，从 [pretext.cool](https://www.pretext.cool/)链接）：

- **Pretext Breaker** — 带有单词砖块的打砖块 — `github.com/rinesh/pretext-breaker`
- **Tetris × Pretext** — `github.com/shinichimochizuki/tetris-pretext`
- **Dragon animation** — `github.com/qtakmalay/PreTextExperiments`
- **Somnai editorial engine** — `github.com/somnai-dreams/pretext-demos`
- **Bad Apple!! ASCII** — `github.com/frmlinn/bad-apple-pretext`
- **Drag-sprite reflow** — `github.com/dokobot/pretext-demo`
- **Alarmy editorial clock** — `github.com/SmisLee/alarmy-pretext-demo`

官方游乐场：[chenglou.me/pretext](https://chenglou.me/pretext/) — 手风琴、气泡、动态布局、编辑引擎、对齐比较、砌体、markdown-chat、富注释。
