---
title: "Manim Video — Manim CE动画：3Blue1Brown数学/算法视频"
sidebar_label: "Manim Video"
description: "Manim CE动画：3Blue1Brown数学/算法视频"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Manim Video

Manim CE动画：3Blue1Brown数学/算法视频。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/manim-video` |
| 版本 | `1.0.0` |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Manim视频制作流水线

## 使用场景

当用户请求以下内容时使用：动画解释、数学动画、概念可视化、算法演练、技术解释器、3Blue1Brown风格视频，或任何具有几何/数学内容的程序动画。使用Manim社区版创建3Blue1Brown风格的解释视频、算法可视化、方程推导、架构图和数据故事。

## 创意标准

这是教育电影。每一帧都教授。每一个动画都揭示结构。

**在写任何代码之前**，阐述叙事弧线。它纠正了哪些误解？什么是"啊哈时刻"？什么样的视觉故事将观众从困惑带到理解？用户的提示是起点 — 用教学野心来诠释它。

**几何先于代数。** 先展示形状，然后是方程。视觉记忆比符号记忆编码更快。当观众在公式之前看到几何模式时，方程感觉是应得的。

**首次渲染的卓越性是不可妥协的。** 输出必须视觉清晰且审美一致，无需修改轮次。如果看起来杂乱、时间安排不佳，或像"AI生成的幻灯片"，那就是错误的。

**不透明度分层引导注意力。** 永远不要以全亮度显示所有内容。主要元素1.0，背景元素0.4，结构元素（轴、网格）0.15。大脑分层处理视觉显著性。

**呼吸空间。** 每个动画之后都需要`self.wait()`。观众需要时间吸收刚出现的内容。永远不要从一个动画匆忙到下一个。关键揭示后的2秒停顿永远不会被浪费。

**统一的视觉语言。** 所有场景共享调色板、一致的排版大小、匹配的动画速度。技术上正确但每个场景使用随机不同颜色的视频是审美的失败。

## 先决条件

运行`scripts/setup.sh`验证所有依赖。需要：Python 3.10+、Manim社区版v0.20+（`pip install manim`）、LaTeX（Linux上的`texlive-full`，macOS上的`mactex`）和ffmpeg。参考文档针对Manim CE v0.20.1测试。

## 模式

| 模式 | 输入 | 输出 | 参考 |
|------|------|------|------|
| **概念解释器** | 主题/概念 | 带几何直觉的动画解释 | `references/scene-planning.md` |
| **方程推导** | 数学表达式 | 分步动画证明 | `references/equations.md` |
| **算法可视化** | 算法描述 | 分步执行与数据结构 | `references/graphs-and-data.md` |
| **数据故事** | 数据/指标 | 动画图表、比较、计数器 | `references/graphs-and-data.md` |
| **架构图** | 系统描述 | 组件构建与连接 | `references/mobjects.md` |
| **论文解释器** | 研究论文 | 关键发现和方法动画 | `references/scene-planning.md` |
| **3D可视化** | 3D概念 | 旋转曲面、参数曲线、空间几何 | `references/camera-and-3d.md` |

## 技术栈

每个项目一个Python脚本。无浏览器、无Node.js、无GPU要求。

| 层级 | 工具 | 用途 |
|------|------|------|
| 核心 | Manim社区版 | 场景渲染、动画引擎 |
| 数学 | LaTeX (texlive/MiKTeX) | 通过`MathTex`渲染方程 |
| 视频I/O | ffmpeg | 场景拼接、格式转换、音频混音 |
| TTS | ElevenLabs / Qwen3-TTS（可选） | 旁白配音 |

## 流水线

```
PLAN --> CODE --> RENDER --> STITCH --> AUDIO (optional) --> REVIEW
```

1. **PLAN** — 编写`plan.md`，包含叙事弧线、场景列表、视觉元素、调色板、配音脚本
2. **CODE** — 编写`script.py`，每个场景一个类，每个类可独立渲染
3. **RENDER** — `manim -ql script.py Scene1 Scene2 ...`草稿，`-qh`生产
4. **STITCH** — ffmpeg concat场景剪辑到`final.mp4`
5. **AUDIO**（可选）— 通过ffmpeg添加旁白和/或背景音乐。参见`references/rendering.md`
6. **REVIEW** — 渲染预览静止图像，对照计划验证，调整

## 项目结构

```
project-name/
  plan.md                # 叙事弧线、场景分解
  script.py              # 所有场景在一个文件
  concat.txt             # ffmpeg场景列表
  final.mp4              # 拼接输出
  media/                 # Manim自动生成
    videos/script/480p15/
```

## 创意方向

### 调色板

| 调色板 | 背景 | 主要 | 次要 | 强调 | 用途 |
|--------|------|------|------|------|------|
| **经典3B1B** | `#1C1C1C` | `#58C4DD` (蓝) | `#83C167` (绿) | `#FFFF00` (黄) | 一般数学/CS |
| **暖学术** | `#2D2B55` | `#FF6B6B` | `#FFD93D` | `#6BCB77` | 平易近人 |
| **霓虹技术** | `#0A0A0A` | `#00F5FF` | `#FF00FF` | `#39FF14` | 系统、架构 |
| **单色** | `#1A1A2E` | `#EAEAEA` | `#888888` | `#FFFFFF` | 极简 |

### 动画速度

| 上下文 | run_time | 之后self.wait() |
|--------|----------|-----------------|
| 标题/介绍出现 | 1.5s | 1.0s |
| 关键方程揭示 | 2.0s | 2.0s |
| 变换/变形 | 1.5s | 1.5s |
| 支持标签 | 0.8s | 0.5s |
| FadeOut清理 | 0.5s | 0.3s |
| "啊哈时刻"揭示 | 2.5s | 3.0s |

### 排版比例

| 角色 | 字体大小 | 用途 |
|------|----------|------|
| 标题 | 48 | 场景标题，开场文字 |
| 标题 | 36 | 场景内节标题 |
| 正文 | 30 | 解释性文字 |
| 标签 | 24 | 注释、轴标签 |
| 说明 | 20 | 字幕、附注 |

### 字体

**对所有文本使用等宽字体。** Manim的Pango渲染器在所有尺寸下都会产生断开的字距。参见`references/visual-design.md`获取完整推荐。

```python
MONO = "Menlo"  # 在文件顶部定义一次

Text("Fourier Series", font_size=48, font=MONO, weight=BOLD)  # 标题
Text("n=1: sin(x)", font_size=20, font=MONO)                  # 标签
MathTex(r"\nabla L")                                            # 数学（使用LaTeX）
```

最小`font_size=18`以确保可读性。

### 每个场景变化

永远不要对所有场景使用相同配置。对于每个场景：
- **不同的主色** 来自调色板
- **不同的布局** — 不要总是居中所有内容
- **不同的动画进入** — 在Write、FadeIn、GrowFromCenter、Create之间变化
- **不同的视觉重量** — 一些场景密集，其他稀疏

## 工作流

### 第1步：计划（plan.md）

在任何代码之前，编写`plan.md`。参见`references/scene-planning.md`获取综合模板。

### 第2步：代码（script.py）

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
- **字幕**在每个动画上：`self.add_subcaption("text", duration=N)`或在`self.play()`上`subcaption="text"`
- **文件顶部的共享颜色常量**用于跨场景一致性
- **`self.camera.background_color`**在每个场景中设置
- **干净退出** — 场景结束时FadeOut所有mobjects：`self.play(FadeOut(Group(*self.mobjects)))`

### 第3步：渲染

```bash
manim -ql script.py Scene1_Introduction Scene2_CoreConcept  # 草稿
manim -qh script.py Scene1_Introduction Scene2_CoreConcept  # 生产
```

### 第4步：拼接

```bash
cat > concat.txt << 'EOF'
file 'media/videos/script/480p15/Scene1_Introduction.mp4'
file 'media/videos/script/480p15/Scene2_CoreConcept.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```

### 第5步：审查

```bash
manim -ql --format=png -s script.py Scene2_CoreConcept  # 预览静止
```

## 关键实现注意事项

### LaTeX使用原始字符串
```python
# 错误：MathTex("\frac{1}{2}")
# 正确：
MathTex(r"\frac{1}{2}")
```

### 边缘文本buff >= 0.5
```python
label.to_edge(DOWN, buff=0.5)  # 从不< 0.5
```

### 替换文本前FadeOut
```python
self.play(ReplacementTransform(note1, note2))  # 不是Write(note2)覆盖
```

### 永远不要动画未添加的Mobjects
```python
self.play(Create(circle))  # 必须先添加
self.play(circle.animate.set_color(RED))  # 然后动画
```

## 性能目标

| 质量 | 分辨率 | FPS | 速度 |
|------|--------|-----|------|
| `-ql`（草稿） | 854x480 | 15 | 5-15s/场景 |
| `-qm`（中等） | 1280x720 | 30 | 15-60s/场景 |
| `-qh`（生产） | 1920x1080 | 60 | 30-120s/场景 |

始终在`-ql`迭代。仅对最终输出渲染`-qh`。

## 参考资料

| 文件 | 内容 |
|------|------|
| `references/animations.md` | 核心动画、速率函数、组合、.animate语法、时间模式 |
| `references/mobjects.md` | 文本、形状、VGroup/Group、定位、样式、自定义mobjects |
| `references/visual-design.md` | 12条设计原则、不透明度分层、布局模板、调色板 |
| `references/equations.md` | LaTeX在Manim中、TransformMatchingTex、推导模式 |
| `references/graphs-and-data.md` | 轴、绘图、BarChart、动画数据、算法可视化 |
| `references/camera-and-3d.md` | MovingCameraScene、ThreeDScene、3D曲面、相机控制 |
| `references/scene-planning.md` | 叙事弧线、布局模板、场景过渡、计划模板 |
| `references/rendering.md` | CLI参考、质量预设、ffmpeg、旁白工作流、GIF导出 |
| `references/troubleshooting.md` | LaTeX错误、动画错误、常见错误、调试 |
| `references/animation-design-thinking.md` | 何时动画vs显示静态、分解、节奏、旁白同步 |
| `references/updaters-and-trackers.md` | ValueTracker、add_updater、always_redraw、基于时间的更新器、模式 |
| `references/paper-explainer.md` | 将研究论文转化为动画 — 工作流、模板、领域模式 |
| `references/decorations.md` | SurroundingRectangle、Brace、箭头、DashedLine、Angle、注释生命周期 |
| `references/production-quality.md` | 编码前、渲染前、渲染后清单、空间布局、颜色、节奏 |

---

## 创意分化（仅在用户要求实验性/创意/独特输出时使用）

如果用户要求创意、实验性或非传统的解释方法，选择策略并在设计动画之前推理其步骤。

- **SCAMPER** — 当用户想要标准解释的新尝试时
- **假设反转** — 当用户想要挑战某事通常的教学方式时

### SCAMPER转换
采用标准数学/技术可视化并系统地转换它：
- ** Substitute**：替换标准视觉隐喻（数轴→蜿蜒路径、矩阵→城市网格）
- **Combine**：合并两种解释方法（代数+几何同时）
- **Reverse**：向后推导 — 从结果开始，解构到公理
- **Modify**：夸张参数以显示为什么重要（10倍学习率、1000倍样本大小）
- **Eliminate**：移除所有符号 — 仅通过动画和空间关系解释

### 假设反转
1. 列出关于如何可视化这个主题的"标准"（从左到右、2D、离散步骤、正式符号）
2. 选择最根本的假设
3. 反转它（从右到左推导、2D概念的3D嵌入、连续变形代替步骤、零符号）
4. 探索反转揭示了标准方法隐藏的内容
