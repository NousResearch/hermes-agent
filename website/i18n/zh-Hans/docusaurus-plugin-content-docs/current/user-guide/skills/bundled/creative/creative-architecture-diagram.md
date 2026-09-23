---
title: "Architecture Diagram — 深色主题 SVG 架构/云/基础设施图表（HTML 格式）"
sidebar_label: "Architecture Diagram"
description: "深色主题 SVG 架构/云/基础设施图表（HTML 格式）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Architecture Diagram

深色主题 SVG 架构/云/基础设施图表，以 HTML 格式输出。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/architecture-diagram` |
| 版本 | `1.0.0` |
| 作者 | Cocoon AI (hello@cocoon-ai.com)，由 Hermes Agent 移植 |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `architecture`, `diagrams`, `SVG`, `HTML`, `visualization`, `infrastructure`, `cloud` |
| 相关 skill | [`concept-diagrams`](/user-guide/skills/optional/creative/creative-concept-diagrams), [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Architecture Diagram Skill

生成专业的深色主题技术架构图，输出为包含内联 SVG 图形的独立 HTML 文件。无需外部工具、无需 API 密钥、无需渲染库——只需写入 HTML 文件并在浏览器中打开即可。

## 适用范围

**最适合：**
- 软件系统架构（前端/后端/数据库层）
- 云基础设施（VPC、区域、子网、托管服务）
- 微服务/服务网格拓扑
- 数据库 + API 映射、部署图
- 任何具有技术基础设施主题、适合深色网格背景风格的内容

**以下场景请优先考虑其他工具：**
- 物理、化学、数学、生物或其他科学学科
- 实物对象（车辆、硬件、解剖结构、截面图）
- 平面图、叙事流程、教育/教科书风格的视觉内容
- 手绘白板草图（建议使用 `excalidraw`）
- 动画说明（建议使用动画相关 skill）

如果有更专业的 skill 适用于该主题，请优先使用。如果没有合适的，本 skill 也可作为通用 SVG 图表的备选方案——输出内容将带有下述深色技术风格。

基于 [Cocoon AI 的 architecture-diagram-generator](https://github.com/Cocoon-AI/architecture-diagram-generator)（MIT 许可证）。

## 工作流程

1. 用户描述其系统架构（组件、连接关系、技术栈）
2. 按照下方设计规范生成 HTML 文件
3. 使用 `write_file` 保存为 `.html` 文件（例如 `~/architecture-diagram.html`）
4. 用户在任意浏览器中打开——支持离线使用，无需任何依赖

### 输出位置

将图表保存到用户指定路径，或默认保存至当前工作目录：
```
./[project-name]-architecture.html
```

### 预览

保存后，建议用户通过以下命令打开：
```bash
# macOS
open ./my-architecture.html
# Linux
xdg-open ./my-architecture.html
```

## 设计规范与视觉语言

### 颜色方案（语义映射）

使用特定的 `rgba` 填充色和十六进制描边色对组件进行分类：

| 组件类型 | 填充色（rgba） | 描边色（Hex） |
| :--- | :--- | :--- |
| **前端** | `rgba(8, 51, 68, 0.4)` | `#22d3ee`（cyan-400） |
| **后端** | `rgba(6, 78, 59, 0.4)` | `#34d399`（emerald-400） |
| **数据库** | `rgba(76, 29, 149, 0.4)` | `#a78bfa`（violet-400） |
| **AWS/云** | `rgba(120, 53, 15, 0.3)` | `#fbbf24`（amber-400） |
| **安全** | `rgba(136, 19, 55, 0.4)` | `#fb7185`（rose-400） |
| **消息总线** | `rgba(251, 146, 60, 0.3)` | `#fb923c`（orange-400） |
| **外部** | `rgba(30, 41, 59, 0.5)` | `#94a3b8`（slate-400） |

### 字体与背景
- **字体：** JetBrains Mono（等宽字体），从 Google Fonts 加载
- **字号：** 12px（名称）、9px（副标签）、8px（注释）、7px（极小标签）
- **背景：** Slate-950（`#020617`），带有细腻的 40px 网格图案

```svg
<!-- 背景网格图案 -->
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/>
</pattern>
```

## 技术实现细节

### 组件渲染
组件为圆角矩形（`rx="6"`），描边宽度 1.5px。为防止箭头透过半透明填充色显现，使用**双矩形遮罩技术**：
1. 绘制不透明背景矩形（`#0f172a`）
2. 在其上方绘制半透明样式矩形

### 连接规则
- **Z 轴顺序：** 在 SVG 早期绘制箭头（在网格之后），使其渲染在组件框的下方
- **箭头头部：** 通过 SVG marker 定义
- **安全流：** 使用 rose 色（`#fb7185`）虚线
- **边界：**
  - *安全组：* 虚线（`4,4`），rose 色
  - *区域：* 大虚线（`8,4`），amber 色，`rx="12"`

### 间距与布局规则
- **标准高度：** 60px（服务）；80–120px（大型组件）
- **垂直间距：** 组件之间最小 40px
- **消息总线：** 必须放置在服务之间的间隙中，不得与其重叠
- **图例位置：** **关键。** 必须放置在所有边界框的外部。计算所有边界的最低 Y 坐标，并将图例放置在其下方至少 20px 处。

## 文档结构

生成的 HTML 文件遵循四段式布局：
1. **页眉：** 带有脉冲点指示器的标题和副标题
2. **主 SVG：** 包含在圆角边框卡片中的图表
3. **摘要卡片：** 图表下方的三张卡片网格，用于展示高层次详情
4. **页脚：** 简洁的元数据信息

### 信息卡片模式
```html
<div class="card">
  <div class="card-header">
    <div class="card-dot cyan"></div>
    <h3>Title</h3>
  </div>
  <ul>
    <li>• Item one</li>
    <li>• Item two</li>
  </ul>
</div>
```

## 输出要求
- **单文件：** 一个自包含的 `.html` 文件
- **无外部依赖：** 所有 CSS 和 SVG 必须内联（Google Fonts 除外）
- **无 JavaScript：** 所有动画（如脉冲点）使用纯 CSS 实现
- **兼容性：** 必须在任何现代浏览器中正确渲染

## 模板参考

加载完整 HTML 模板以获取精确的结构、CSS 和 SVG 组件示例：

```
skill_view(name="architecture-diagram", file_path="templates/template.html")
```

模板包含每种组件类型（前端、后端、数据库、云、安全）、箭头样式（标准、虚线、曲线）、安全组、区域边界和图例的完整示例——生成图表时请以此作为结构参考。