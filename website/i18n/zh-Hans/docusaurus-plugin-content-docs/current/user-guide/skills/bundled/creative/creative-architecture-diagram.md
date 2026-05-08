---
title: "Architecture Diagram — 深色主题 SVG 架构/云/基础设施图表（HTML）"
sidebar_label: "Architecture Diagram"
description: "深色主题 SVG 架构/云/基础设施图表（HTML）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Architecture Diagram

深色主题 SVG 架构/云/基础设施图表（HTML）。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/architecture-diagram` |
| 版本 | `1.0.0` |
| 作者 | Cocoon AI (hello@cocoon-ai.com)，由 Hermes Agent 移植 |
| 许可证 | MIT |
| 标签 | `architecture`、`diagrams`、`SVG`、`HTML`、`visualization`、`infrastructure`、`cloud` |
| 相关技能 | [`concept-diagrams`](/docs/user-guide/skills/optional/creative/creative-concept-diagrams)、[`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# Architecture Diagram 技能

生成专业的深色主题技术架构图表为独立 HTML 文件，带内联 SVG 图形。无外部工具、无 API 密钥、无渲染库 — 只需写入 HTML 文件并在浏览器中打开。

## 范围

**最适合：**

- 软件系统架构（前端 / 后端 / 数据库层）
- 云基础设施（VPC、区域、子网、托管服务）
- 微服务 / 服务网格拓扑
- 数据库 + API 映射、部署图表
- 任何具有适合深色、网格背景美学的技术基础设施主题

**首先看其他地方：**

- 物理、化学、数学、生物或其他科学主题
- 物理对象（车辆、硬件、解剖、横截面）
- 平面图、叙事旅程、教育 / 教科书风格可视化
- 手绘白板草图（考虑 `excalidraw`）
- 动画解释（考虑动画技能）

如果有更专业的技能适用于该主题，优先使用。如果没有合适，此技能也可作为通用 SVG 图表后备 — 输出将携带下文描述的美学。

基于 [Cocoon AI 的 architecture-diagram-generator](https://github.com/Cocoon-AI/architecture-diagram-generator)（MIT）。

## 工作流

1. 用户描述他们的系统架构（组件、连接、技术）
2. 遵循下方设计系统生成 HTML 文件
3. 使用 `write_file` 保存到 `.html` 文件（如 `~/architecture-diagram.html`）
4. 用户在任意浏览器中打开 — 离线工作，无依赖

### 输出位置

将图表保存到用户指定的路径，或默认为当前工作目录：
```
./[project-name]-architecture.html
```

### 预览

保存后，建议用户打开：
```bash
# macOS
open ./my-architecture.html
# Linux
xdg-open ./my-architecture.html
```

## 设计系统和视觉语言

### 调色板（语义映射）

使用特定的 `rgba` 填充和十六进制描边对组件进行分类：

| 组件类型 | 填充 (rgba) | 描边 (Hex) |
| :--- | :--- | :--- |
| **前端** | `rgba(8, 51, 68, 0.4)` | `#22d3ee` (cyan-400) |
| **后端** | `rgba(6, 78, 59, 0.4)` | `#34d399` (emerald-400) |
| **数据库** | `rgba(76, 29, 149, 0.4)` | `#a78bfa` (violet-400) |
| **AWS/云** | `rgba(120, 53, 15, 0.3)` | `#fbbf24` (amber-400) |
| **安全** | `rgba(136, 19, 55, 0.4)` | `#fb7185` (rose-400) |
| **消息总线** | `rgba(251, 146, 60, 0.3)` | `#fb923c` (orange-400) |
| **外部** | `rgba(30, 41, 59, 0.5)` | `#94a3b8` (slate-400) |

### 排版和背景

- **字体：** JetBrains Mono（等宽），从 Google Fonts 加载
- **大小：** 12px（名称）、9px（副标签）、8px（注释）、7px（微型标签）
- **背景：** Slate-950（`#020617`）带微妙的 40px 网格图案

```svg
<!-- Background Grid Pattern -->
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/>
</pattern>
```

## 技术实现细节

### 组件渲染

组件是圆角矩形（`rx="6"`），带 1.5px 描边。为防止箭头穿透半透明填充，使用**双矩形遮罩技术**：

1. 绘制不透明背景矩形（`#0f172a`）
2. 在顶部绘制半透明样式矩形

### 连接规则

- **Z 顺序：** 在 SVG 中早绘制箭头（在网格之后），以便它们在组件框后面渲染
- **箭头：** 通过 SVG 标记定义
- **安全流：** 使用虚线，颜色为 `#fb7185`
- **边界：**
  - *安全组：* 虚线（`4,4`），玫瑰色
  - *区域：* 大虚线（`8,4`），琥珀色，`rx="12"`'

### 间距和布局逻辑

- **标准高度：** 60px（服务）；80-120px（大型组件）
- **垂直间隙：** 组件之间最小 40px
- **消息总线：** 必须放置在服务*之间*的间隙，不重叠它们
- **图例放置：** **关键。** 必须放置在所有边界框之外。计算所有边界的最低 Y 坐标，并将图例放置在其下方至少 20px。

## 文档结构

生成的 HTML 文件遵循四部分布局：
1. **标题：** 带脉动指示器和副标题的标题
2. **主 SVG：** 包含在圆角边框卡片中的图表
3. **摘要卡片：** 图表下方三张卡片的网格，用于高层详情
4. **页脚：** 最少的元数据

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
- **无 JavaScript：** 对任何动画使用纯 CSS（如脉动点）
- **兼容性：** 必须在任何现代 Web 浏览器中正确渲染

## 模板参考

加载完整 HTML 模板以获取确切结构、CSS 和 SVG 组件示例：

```
skill_view(name="architecture-diagram", file_path="templates/template.html")
```

模板包含每种组件类型（前端、后端、数据库、云、安全）、箭头样式（标准、虚线、曲线）、安全组、区域边界和图例的工作示例 — 生成图表时将其用作结构参考。
