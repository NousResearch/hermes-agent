---
title: "Hyperframes"
sidebar_label: "Hyperframes"
description: "使用 HyperFrames 创建基于 HTML 的视频合成、动画标题卡、社交叠加层、带字幕的对话视频、音频响应视觉效果和着色器转场..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hyperframes

使用 HyperFrames 创建基于 HTML 的视频合成、动画标题卡、社交叠加层、带字幕的对话视频、音频响应视觉效果和着色器转场。HTML 是视频的唯一真实来源。当用户需要从 HTML 合成渲染 MP4/WebM、在媒体上添加文字/Logo/图表动画、将字幕与音频同步、需要 TTS 旁白，或将网站转换为视频时使用本技能。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/creative/hyperframes` 安装 |
| 路径 | `optional-skills/creative/hyperframes` |
| 版本 | `1.0.0` |
| 作者 | heygen-com |
| 许可证 | Apache-2.0 |
| 平台 | linux, macos, windows |
| 标签 | `creative`, `video`, `animation`, `html`, `gsap`, `motion-graphics` |
| 相关技能 | [`manim-video`](/user-guide/skills/bundled/creative/creative-manim-video), [`meme-generation`](/user-guide/skills/optional/creative/creative-meme-generation) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发本技能时加载的完整技能定义。这是 agent 在技能激活时所看到的指令内容。
:::

# HyperFrames

HTML 是视频的唯一真实来源。合成（composition）是一个带有 `data-*` 属性用于计时、GSAP 时间轴用于动画、CSS 用于外观的 HTML 文件。HyperFrames 引擎逐帧捕获页面，并通过 FFmpeg 编码为 MP4/WebM。

**与 `manim-video` 的互补关系：** 数学/几何讲解（方程式、3B1B 风格）使用 `manim-video`。动态图形、带字幕的对话视频、产品演示、社交叠加层、着色器转场，以及任何由真实视频/音频媒体驱动的内容使用 `hyperframes`。

## 使用场景

- 用户要求从文本、脚本或网站渲染视频
- 动画标题卡、下三分之一字幕条或排版片头
- 带字幕的旁白视频（TTS + 字幕与波形同步）
- 音频响应视觉效果（节拍同步、频谱条、脉冲发光）
- 场景间转场（交叉淡入淡出、划像、着色器扭曲、闪白）
- 社交叠加层（Instagram/TikTok/YouTube 风格）
- 网站转视频流程（捕获 URL，生成宣传片）
- 任何需要确定性渲染为视频文件的 HTML/CSS/JS 动画

**不适用**本技能的场景：
- 纯数学/方程式动画（→ `manim-video`）
- 图像生成或表情包（→ `meme-generation`，图像模型）
- 实时视频会议或直播

## 快速参考

```bash
npx hyperframes init my-video               # 初始化项目脚手架
cd my-video
npx hyperframes lint                        # 预览/渲染前验证
npx hyperframes preview                     # 实时热重载浏览器预览（端口 3002）
npx hyperframes render --output final.mp4   # 渲染为 MP4
npx hyperframes doctor                      # 诊断环境问题
```

渲染参数：`--quality draft|standard|high` · `--fps 24|30|60` · `--format mp4|webm` · `--docker`（可复现）· `--strict`。

完整 CLI 参考：[references/cli.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/cli.md)。

## 初始设置（一次性）

```bash
bash "$(dirname "$(find ~/.hermes/skills -path '*/hyperframes/SKILL.md' 2>/dev/null | head -1)")/scripts/setup.sh"
```

该脚本执行以下操作：
1. 验证 Node.js >= 22 和 FFmpeg 已安装（若未安装则打印修复说明）。
2. 全局安装 `hyperframes` CLI（`npm install -g hyperframes@>=0.4.2`）。
3. 通过 Puppeteer 预缓存 `chrome-headless-shell` — **必需**，用于通过 Chrome 的 `HeadlessExperimental.beginFrame` 捕获路径实现最高质量渲染。
4. 运行 `npx hyperframes doctor` 并报告结果。

若设置失败，请参阅 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md)。

## 操作流程

### 1. 编写 HTML 前先规划

在接触代码之前，从高层次阐明：
- **内容** — 叙事弧线、关键时刻、情感节拍
- **结构** — 合成、轨道（视频/音频/叠加层）、时长
- **视觉标识** — 颜色、字体、动态风格（爆炸感 / 电影感 / 流畅 / 技术感）
- **主帧** — 每个场景中最多元素同时可见的时刻。这是你首先要构建的静态布局。

**视觉标识关卡（硬性关卡）。** 在编写任何合成 HTML 之前，必须先定义视觉标识。**不得**使用默认或通用颜色编写合成（`#333`、`#3b82f6`、`Roboto` 是跳过此步骤的明显标志）。按顺序检查：

1. **项目根目录有 `DESIGN.md`？** → 使用其中精确的颜色、字体、动态规则和"禁止事项"约束。
2. **用户指定了风格**（如"Swiss Pulse"、"暗黑科技感"、"奢侈品牌"）？ → 生成一个包含 `## Style Prompt`、`## Colors`（3-5 个带角色的十六进制色值）、`## Typography`（1-2 个字体族）、`## What NOT to Do`（3-5 个反模式）的最小 `DESIGN.md`。
3. **以上均无？** → 在编写任何 HTML 之前先提问 3 个问题：
   - 氛围？（爆炸感 / 电影感 / 流畅 / 技术感 / 混乱 / 温暖）
   - 浅色还是深色画布？
   - 是否有品牌颜色、字体或视觉参考？

   然后根据答案生成 `DESIGN.md`。每个合成的调色板和排版都必须追溯到 `DESIGN.md` 或用户的明确指示。

### 2. 初始化脚手架

```bash
npx hyperframes init my-video --non-interactive
```

模板：`blank`、`warm-grain`、`play-mode`、`swiss-grid`、`vignelli`、`decision-tree`、`kinetic-type`、`product-promo`、`nyt-graph`。传入 `--example <name>` 选择模板，`--video clip.mp4` 或 `--audio track.mp3` 以媒体文件为起点。

### 3. 先布局，后动画

先为**主帧**编写静态 HTML+CSS — 暂不添加 GSAP。`.scene-content` 容器必须填满场景（`width:100%; height:100%; padding:Npx`），使用 `display:flex` + `gap`。用 padding 将内容向内推 — 永远不要在内容容器上使用 `position: absolute; top: Npx`（内容高于剩余空间时会溢出）。

只有在主帧看起来正确之后，才添加 `gsap.from()` 入场动画（**向** CSS 位置动画）和 `gsap.to()` 退场动画（**从** CSS 位置动画）。

完整的 data 属性 schema 和合成规则见 [references/composition.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/composition.md)。

### 4. 使用 GSAP 制作动画

每个合成必须：
- 注册其时间轴：`window.__timelines["<composition-id>"] = tl`
- 初始暂停：`gsap.timeline({ paused: true })` — 播放器控制播放
- 使用有限的 `repeat` 值（禁止 `repeat: -1` — 会破坏捕获引擎）。计算方式：`repeat: Math.ceil(duration / cycleDuration) - 1`。
- 具有确定性 — 禁止 `Math.random()`、`Date.now()` 或挂钟逻辑。如需伪随机数，使用带种子的 PRNG。
- 同步构建 — 时间轴构建过程中禁止 `async`/`await`、`setTimeout` 或 Promise。

核心 GSAP API（tween、ease、stagger、timeline）见 [references/gsap.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/gsap.md)。

### 5. 场景间转场

多场景合成需要转场。规则：
1. **场景间始终使用转场** — 禁止跳切。
2. **每个场景元素始终使用入场动画**（`gsap.from(...)`）。
3. **除最后一个场景外，禁止使用退场动画** — 转场本身就是退出。
4. 最后一个场景可以淡出。

使用 `npx hyperframes add <transition-name>` 安装着色器转场（`flash-through-white`、`liquid-wipe` 等）。完整列表：`npx hyperframes add --list`。

### 6. 音频、字幕、TTS、音频响应、高亮

- **音频：** 始终使用独立的 `<audio>` 元素（视频使用 `muted playsinline`）。
- **TTS：** `npx hyperframes tts "脚本文本" --voice af_nova --output narration.wav`。使用 `--list` 列出可用音色。音色 ID 首字母编码语言（`a`/`b`=英语，`e`=西班牙语，`f`=法语，`j`=日语，`z`=普通话等）— CLI 自动推断音素化（phonemizer）语言环境；仅在需要覆盖时传入 `--lang`。非英语音素化需要系统级安装 `espeak-ng`。
- **字幕：** `npx hyperframes transcribe narration.wav` → 词级转录。根据转录内容的语气选择样式（hype / corporate / tutorial / storytelling / social — 见 `references/features.md` 中的表格）。**语言规则：** 除非确认音频为英语，否则永远不要使用 `.en` whisper 模型 — `.en` 会将非英语音频翻译而非转录。每个字幕组在其退出 tween 之后必须有一个硬性的 `tl.set(el, { opacity: 0, visibility: "hidden" }, group.end)` 清除 — 否则字幕组会泄漏到后续组中保持可见。
- **音频响应视觉效果：** 预先提取音频频段（低频 / 中频 / 高频），并在时间轴内通过 `for` 循环的 `tl.call(draw, [], f / fps)` 逐帧采样 — 单个长 tween **不会**响应音频。将低频映射到 `scale`（脉冲），高频映射到 `textShadow`/`boxShadow`（发光），整体振幅映射到 `opacity`/`y`/`backgroundColor`。避免均衡器条形图的陈词滥调 — 让内容引导视觉，让音频驱动其行为。
- **标记式高亮：** 文字强调的高亮、圆圈、爆炸、涂鸦、划除效果均为确定性 CSS+GSAP — 见 `references/features.md#marker-highlighting`。完全可寻址，无动画 SVG 滤镜。
- **场景转场：** 每个多场景合成必须使用转场（禁止跳切）。从 CSS 原语（推入滑动、模糊交叉淡入淡出、缩放穿越、交错块）或着色器转场（`flash-through-white`、`liquid-wipe`、`cross-warp-morph`、`chromatic-split` 等，通过 `npx hyperframes add` 安装）中选择。氛围和能量对照表见 `references/features.md#transitions`。同一合成中不得混用 CSS 转场和着色器转场。

### 7. Lint、验证、检查、预览、渲染

```bash
npx hyperframes lint              # 捕获缺失的 data-composition-id、重叠轨道、未注册的时间轴
npx hyperframes validate          # 在 5 个时间戳进行 WCAG 对比度审计
npx hyperframes inspect           # 视觉布局审计 — 溢出、帧外元素、被遮挡的文字
npx hyperframes preview           # 实时浏览器预览
npx hyperframes render --quality draft --output draft.mp4    # 快速迭代
npx hyperframes render --quality high --output final.mp4     # 最终交付
```

`hyperframes validate` 对每个文字元素后方的背景像素进行采样，并对对比度低于 4.5:1（大文字为 3:1）的情况发出警告。`hyperframes inspect` 是布局侧的配套工具 — 在多个时间戳运行页面，标记静态 lint 无法发现的问题（仅在 4.5s 时超出安全区域的字幕换行、标题为最长变体时溢出的卡片、被转场着色器遮挡的元素）。对于包含对话气泡、卡片、字幕或紧凑排版的合成，务必运行 `inspect`。

### 8. 网站转视频（若用户提供 URL）

使用 [references/website-to-video.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/website-to-video.md) 中的 7 步捕获转视频工作流：捕获 → DESIGN.md → SCRIPT.md → 分镜 → 合成 → 渲染 → 交付。

## 常见陷阱

- **`HeadlessExperimental.beginFrame' wasn't found`** — Chromium 147+ 移除了此协议。确保使用 `hyperframes@>=0.4.2`（自动检测并回退到截图模式）。应急方案：`export PRODUCER_FORCE_SCREENSHOT=true`。参见 [hyperframes#294](https://github.com/heygen-com/hyperframes/issues/294) 和 [references/troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md)。
- **系统 Chrome（非 `chrome-headless-shell`）** — 渲染会挂起 120 秒后超时。运行 `npx puppeteer browsers install chrome-headless-shell`（setup.sh 已处理此步骤）。`hyperframes doctor` 会报告将使用哪个二进制文件。
- **任何地方出现 `repeat: -1`** — 会破坏捕获引擎。始终计算有限的 repeat 次数。
- **在稍后入场的 clip 元素上使用 `gsap.set()`** — 页面加载时该元素不存在。改为在时间轴内使用 `tl.set(selector, vars, timePosition)`，位置在该 clip 的 `data-start` 处或之后。
- **内容文字中使用 `<br>`** — 强制换行不了解渲染字体宽度，导致自然换行 + `<br>` 双重换行。使用 `max-width` 让文字自然换行。例外：每个单词刻意独占一行的短展示标题。
- **对 `visibility` 或 `display` 进行动画** — GSAP 无法对这些属性进行 tween。使用 `autoAlpha`（同时处理 visibility 和 opacity）。
- **调用 `video.play()` 或 `audio.play()`** — 框架拥有播放控制权。永远不要自行调用这些方法。
- **异步构建时间轴** — 捕获引擎在页面加载后同步读取 `window.__timelines`。永远不要将时间轴构建包裹在 `async`、`setTimeout` 或 Promise 中。
- **独立 `index.html` 包裹在 `<template>` 中** — 会对浏览器隐藏所有内容。只有通过 `data-composition-src` 加载的**子合成**才使用 `<template>`。
- **将视频用于音频** — 始终使用静音的 `<video>` + 独立的 `<audio>`。

## 验证

渲染前后均需执行：

1. **Lint + validate + inspect 通过：** `npx hyperframes lint --strict && npx hyperframes validate && npx hyperframes inspect`（lint 捕获结构问题，validate 捕获对比度问题，inspect 捕获视觉布局/溢出问题 — 若出现警告请参阅 troubleshooting.md）。
2. **动画编排** — 对于新合成或重大动画变更，运行动画映射。`npx hyperframes init` 会将技能脚本复制到项目中，因此路径为项目本地路径：
   ```bash
   node skills/hyperframes/scripts/animation-map.mjs <composition-dir> \
     --out <composition-dir>/.hyperframes/anim-map
   ```
   输出单个 `animation-map.json`，包含每个 tween 的摘要、ASCII 甘特时间轴、stagger 检测、死区（超过 1 秒无动画）、元素生命周期和标记（`offscreen`、`collision`、`invisible`、`paced-fast` &lt;0.2s、`paced-slow` >2s）。扫描摘要和标记 — 逐一修复或说明原因。小幅编辑可跳过。
3. **文件存在且非零：** `ls -lh final.mp4`。
4. **时长与 `data-duration` 匹配：** `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 final.mp4`。
5. **视觉检查：** 提取合成中间帧：`ffmpeg -i final.mp4 -ss 00:00:05 -vframes 1 preview.png`。
6. **若预期有音频，确认音频存在：** `ffprobe -v error -show_streams -select_streams a -of default=nw=1:nk=1 final.mp4 | head -1`。

若 `hyperframes render` 失败，运行 `npx hyperframes doctor` 并在报告问题时附上其输出。

## 参考资料

- [composition.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/composition.md) — data 属性、时间轴契约、不可违反的规则、排版/资源规则
- [cli.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/cli.md) — 所有 CLI 命令（init、capture、lint、validate、inspect、preview、render、transcribe、tts、doctor、browser、info、upgrade、benchmark）
- [gsap.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/gsap.md) — HyperFrames 的 GSAP 核心 API（tween、ease、stagger、timeline、matchMedia）
- [features.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/features.md) — 字幕、TTS、音频响应、标记高亮、转场（按需加载）
- [website-to-video.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/website-to-video.md) — 7 步捕获转视频工作流
- [troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/hyperframes/references/troubleshooting.md) — OpenClaw 修复、环境变量、常见渲染错误