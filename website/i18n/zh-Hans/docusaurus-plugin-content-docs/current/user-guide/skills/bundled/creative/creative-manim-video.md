---
title: "Manim Video — Manim CE 动画：3Blue1Brown 数学/算法视频"
sidebar_label: "Manim Video"
description: "Manim CE 动画：3Blue1Brown 数学/算法视频"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Manim Video

Manim CE 动画：3Blue1Brown 数学/算法视频。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/manim-video` |
| 版本 | `1.0.0` |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在该 skill 被触发时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Manim 视频制作流水线

## 使用时机

当用户请求以下内容时使用：动画讲解、数学动画、概念可视化、算法演示、技术说明、3Blue1Brown 风格视频，或任何包含几何/数学内容的程序化动画。使用 Manim Community Edition 创建 3Blue1Brown 风格的讲解视频、算法可视化、方程推导、架构图以及数据故事。

## 创作标准

这是教育电影。每一帧都在教学。每一个动画都在揭示结构。

**在写任何一行代码之前**，先阐明叙事弧线。这个视频纠正了什么误解？"顿悟时刻"是什么？什么样的视觉故事能带领观众从困惑走向理解？用户的 prompt（提示词）只是起点——以教学抱负去诠释它。

**几何先于代数。** 先展示形状，再展示方程。视觉记忆的编码速度快于符号记忆。当观众在看到公式之前先看到几何图形，方程式就显得水到渠成。

**首次渲染即达到卓越标准，不容妥协。** 输出必须在无需修改的情况下视觉清晰、美学统一。如果某处看起来杂乱、节奏不对，或像"AI 生成的幻灯片"，那就是错的。

**透明度分层引导注意力。** 永远不要让所有元素都以全亮度显示。主要元素为 1.0，上下文元素为 0.4，结构元素（坐标轴、网格）为 0.15。大脑按视觉显著性分层处理信息。

**留白呼吸。** 每个动画之后都需要 `self.wait()`。观众需要时间消化刚刚出现的内容。永远不要从一个动画急速跳到下一个。关键揭示后的 2 秒停顿从不浪费。

**统一的视觉语言。** 所有场景共享同一色板、一致的字体大小、匹配的动画速度。一个技术上正确但每个场景随机使用不同颜色的视频，是美学上的失败。

## 前置条件

运行 `scripts/setup.sh` 验证所有依赖项。需要：Python 3.10+、Manim Community Edition v0.20+（`pip install manim`）、LaTeX（Linux 上为 `texlive-full`，macOS 上为 `mactex`）以及 ffmpeg。参考文档已针对 Manim CE v0.20.1 测试。

## 模式

| 模式 | 输入 | 输出 | 参考 |
|------|-------|--------|-----------|
| **概念讲解** | 主题/概念 | 带几何直觉的动画讲解 | `references/scene-planning.md` |
| **方程推导** | 数学表达式 | 逐步动画证明 | `references/equations.md` |
| **算法可视化** | 算法描述 | 带数据结构的逐步执行 | `references/graphs-and-data.md` |
| **数据故事** | 数据/指标 | 动画图表、对比、计数器 | `references/graphs-and-data.md` |
| **架构图** | 系统描述 | 逐步构建的组件与连接 | `references/mobjects.md` |
| **论文讲解** | 研究论文 | 关键发现与方法的动画呈现 | `references/scene-planning.md` |
| **3D 可视化** | 3D 概念 | 旋转曲面、参数曲线、空间几何 | `references/camera-and-3d.md` |

## 技术栈

每个项目使用单个 Python 脚本。无需浏览器、Node.js 或 GPU。

| 层级 | 工具 | 用途 |
|-------|------|---------|
| 核心 | Manim Community Edition | 场景渲染、动画引擎 |
| 数学 | LaTeX (texlive/MiKTeX) | 通过 `MathTex` 渲染方程 |
| 视频 I/O | ffmpeg | 场景拼接、格式转换、音频混合 |
| TTS | ElevenLabs / Qwen3-TTS（可选） | 旁白配音 |

## 流水线

```
PLAN --> CODE --> RENDER --> STITCH --> AUDIO (optional) --> REVIEW
```

1. **PLAN** — 编写 `plan.md`，包含叙事弧线、场景列表、视觉元素、色板、旁白脚本
2. **CODE** — 编写 `script.py`，每个场景一个类，每个场景可独立渲染
3. **RENDER** — 草稿用 `manim -ql script.py Scene1 Scene2 ...`，正式输出用 `-qh`
4. **STITCH** — 用 ffmpeg 将场景片段拼接为 `final.mp4`
5. **AUDIO**（可选）— 通过 ffmpeg 添加旁白和/或背景音乐。参见 `references/rendering.md`
6. **REVIEW** — 渲染预览静帧，对照计划验证，进行调整

## 项目结构

```
project-name/
  plan.md                # 叙事弧线、场景分解
  script.py              # 所有场景在一个文件中
  concat.txt             # ffmpeg 场景列表
  final.mp4              # 拼接输出
  media/                 # 由 Manim 自动生成
    videos/script/480p15/
```

## 创作方向

### 色板

| 色板 | 背景 | 主色 | 次色 | 强调色 | 使用场景 |
|---------|-----------|---------|-----------|--------|----------|
| **经典 3B1B** | `#1C1C1C` | `#58C4DD`（蓝） | `#83C167`（绿） | `#FFFF00`（黄） | 通用数学/CS |
| **暖色学术** | `#2D2B55` | `#FF6B6B` | `#FFD93D` | `#6BCB77` | 亲切风格 |
| **霓虹科技** | `#0A0A0A` | `#00F5FF` | `#FF00FF` | `#39FF14` | 系统、架构 |
| **单色** | `#1A1A2E` | `#EAEAEA` | `#888888` | `#FFFFFF` | 极简主义 |

### 动画速度

| 场景 | run_time | 之后的 self.wait() |
|---------|----------|-------------------|
| 标题/介绍出现 | 1.5s | 1.0s |
| 关键方程揭示 | 2.0s | 2.0s |
| 变换/变形 | 1.5s | 1.5s |
| 辅助标签 | 0.8s | 0.5s |
| FadeOut 清场 | 0.5s | 0.3s |
| "顿悟时刻"揭示 | 2.5s | 3.0s |

### 字体大小规范

| 角色 | 字体大小 | 用途 |
|------|-----------|-------|
| 标题 | 48 | 场景标题、开场文字 |
| 一级标题 | 36 | 场景内的章节标题 |
| 正文 | 30 | 说明文字 |
| 标签 | 24 | 注释、坐标轴标签 |
| 说明文字 | 20 | 字幕、小字注释 |

### 字体

**所有文字使用等宽字体。** Manim 的 Pango 渲染器在任何大小下使用比例字体都会产生字距错误。完整建议参见 `references/visual-design.md`。

```python
MONO = "Menlo"  # define once at top of file

Text("Fourier Series", font_size=48, font=MONO, weight=BOLD)  # titles
Text("n=1: sin(x)", font_size=20, font=MONO)                  # labels
MathTex(r"\nabla L")                                            # math (uses LaTeX)
```

最小 `font_size=18` 以保证可读性。

### 场景间差异化

永远不要对所有场景使用相同的配置。每个场景应有：
- **不同的主导色** — 来自色板
- **不同的布局** — 不要总是居中
- **不同的动画入场方式** — 在 Write、FadeIn、GrowFromCenter、Create 之间变化
- **不同的视觉密度** — 有些场景密集，有些稀疏

## 工作流程

### 第一步：规划（plan.md）

在写任何代码之前，先编写 `plan.md`。完整模板参见 `references/scene-planning.md`。

### 第二步：编码（script.py）

每个场景一个类。每个场景可独立渲染。

```python
from manim import *

BG = "#1C1C1C"
PRIMARY = "#58C4DD"
SECONDARY = "#83C167"
ACCENT = "#FFFF00"
MONO = "Menlo"

class Scene1_Introduction(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = Text("Why Does This Work?", font_size=48, color=PRIMARY, weight=BOLD, font=MONO)
        self.add_subcaption("Why does this work?", duration=2)
        self.play(Write(title), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title), run_time=0.5)
```

关键模式：
- **每个动画都添加字幕**：`self.add_subcaption("text", duration=N)` 或在 `self.play()` 中使用 `subcaption="text"`
- **共享颜色常量** 定义在文件顶部，保证跨场景一致性
- **每个场景都设置** `self.camera.background_color`
- **干净退出** — 场景结束时 FadeOut 所有 mobject：`self.play(FadeOut(Group(*self.mobjects)))`

### 第三步：渲染

```bash
manim -ql script.py Scene1_Introduction Scene2_CoreConcept  # draft
manim -qh script.py Scene1_Introduction Scene2_CoreConcept  # production
```

### 第四步：拼接

```bash
cat > concat.txt << 'EOF'
file 'media/videos/script/480p15/Scene1_Introduction.mp4'
file 'media/videos/script/480p15/Scene2_CoreConcept.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```

### 第五步：审查

```bash
manim -ql --format=png -s script.py Scene2_CoreConcept  # preview still
```

## 关键实现注意事项

### LaTeX 使用原始字符串
```python
# WRONG: MathTex("\frac{1}{2}")
# RIGHT:
MathTex(r"\frac{1}{2}")
```

### 边缘文字 buff >= 0.5
```python
label.to_edge(DOWN, buff=0.5)  # never < 0.5
```

### 替换文字前先 FadeOut
```python
self.play(ReplacementTransform(note1, note2))  # not Write(note2) on top
```

### 永远不要对未添加的 Mobject 执行动画
```python
self.play(Create(circle))  # must add first
self.play(circle.animate.set_color(RED))  # then animate
```

## 性能目标

| 质量 | 分辨率 | FPS | 速度 |
|---------|-----------|-----|-------|
| `-ql`（草稿） | 854x480 | 15 | 每场景 5-15s |
| `-qm`（中等） | 1280x720 | 30 | 每场景 15-60s |
| `-qh`（正式） | 1920x1080 | 60 | 每场景 30-120s |

始终在 `-ql` 下迭代。仅在最终输出时渲染 `-qh`。

## 参考文档

| 文件 | 内容 |
|------|----------|
| `references/animations.md` | 核心动画、速率函数、组合、`.animate` 语法、时序模式 |
| `references/mobjects.md` | 文字、形状、VGroup/Group、定位、样式、自定义 mobject |
| `references/visual-design.md` | 12 条设计原则、透明度分层、布局模板、色板 |
| `references/equations.md` | Manim 中的 LaTeX、TransformMatchingTex、推导模式 |
| `references/graphs-and-data.md` | 坐标轴、绘图、BarChart、动态数据、算法可视化 |
| `references/camera-and-3d.md` | MovingCameraScene、ThreeDScene、3D 曲面、摄像机控制 |
| `references/scene-planning.md` | 叙事弧线、布局模板、场景过渡、规划模板 |
| `references/rendering.md` | CLI 参考、质量预设、ffmpeg、旁白工作流、GIF 导出 |
| `references/troubleshooting.md` | LaTeX 错误、动画错误、常见错误、调试 |
| `references/animation-design-thinking.md` | 何时使用动画与静态展示、分解、节奏、旁白同步 |
| `references/updaters-and-trackers.md` | ValueTracker、add_updater、always_redraw、基于时间的 updater、模式 |
| `references/paper-explainer.md` | 将研究论文转化为动画——工作流、模板、领域模式 |
| `references/decorations.md` | SurroundingRectangle、Brace、箭头、DashedLine、Angle、注释生命周期 |
| `references/production-quality.md` | 编码前、渲染前、渲染后检查清单、空间布局、颜色、节奏 |

---

## 创意发散（仅在用户要求实验性/创意性/独特输出时使用）

如果用户要求创意性、实验性或非常规的讲解方式，在设计动画**之前**先选择一种策略并进行推理。

- **SCAMPER** — 当用户希望对标准讲解方式进行全新演绎时
- **假设反转** — 当用户希望挑战某个主题通常的教学方式时

### SCAMPER 变换
对标准数学/技术可视化进行变换：
- **替换（Substitute）**：替换标准视觉隐喻（数轴 → 蜿蜒路径，矩阵 → 城市网格）
- **组合（Combine）**：融合两种讲解方式（代数 + 几何同步呈现）
- **反转（Reverse）**：从结果出发反向推导——从结论解构到公理
- **修改（Modify）**：夸大某个参数以展示其重要性（学习率 ×10，样本量 ×1000）
- **消除（Eliminate）**：去掉所有符号标记——纯粹通过动画和空间关系来讲解

### 假设反转
1. 列出该主题可视化的"标准"做法（从左到右、二维、离散步骤、正式符号）
2. 选出最根本的假设
3. 将其反转（从右到左推导、将二维概念嵌入三维、用连续变形代替离散步骤、零符号标记）
4. 探索反转所揭示的、标准方式所隐藏的内容