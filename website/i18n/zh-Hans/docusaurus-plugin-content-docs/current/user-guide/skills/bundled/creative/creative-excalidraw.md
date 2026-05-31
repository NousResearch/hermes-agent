---
title: "Excalidraw — 手绘风格 Excalidraw JSON 图表（架构图、流程图、时序图）"
sidebar_label: "Excalidraw"
description: "手绘风格 Excalidraw JSON 图表（架构图、流程图、时序图）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Excalidraw

手绘风格 Excalidraw JSON 图表（架构图、流程图、时序图）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/excalidraw` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Excalidraw`, `Diagrams`, `Flowcharts`, `Architecture`, `Visualization`, `JSON` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Excalidraw 图表 Skill

通过编写标准 Excalidraw 元素 JSON 并保存为 `.excalidraw` 文件来创建图表。这些文件可以直接拖放到 [excalidraw.com](https://excalidraw.com) 进行查看和编辑。无需账号、无需 API 密钥、无需渲染库——只需 JSON。

## 使用场景

生成 `.excalidraw` 文件，用于架构图、流程图、时序图、概念图等。文件可在 excalidraw.com 打开，或上传以获取可分享链接。

## 工作流程

1. **加载此 skill**（已完成）
2. **编写元素 JSON**——一个 Excalidraw 元素对象数组
3. **保存文件**——使用 `write_file` 创建 `.excalidraw` 文件
4. **可选上传**——通过 `terminal` 运行 `scripts/upload.py` 获取可分享链接

### 保存图表

将元素数组包裹在标准 `.excalidraw` 信封中，并使用 `write_file` 保存：

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "hermes-agent",
  "elements": [ ...your elements array here... ],
  "appState": {
    "viewBackgroundColor": "#ffffff"
  }
}
```

保存到任意路径，例如 `~/diagrams/my_diagram.excalidraw`。

### 上传以获取可分享链接

通过终端运行位于此 skill 的 `scripts/` 目录中的上传脚本：

```bash
python skills/diagramming/excalidraw/scripts/upload.py ~/diagrams/my_diagram.excalidraw
```

此脚本将上传到 excalidraw.com（无需账号）并打印可分享的 URL。需要安装 `cryptography` pip 包（`pip install cryptography`）。

---

## 元素格式参考

### 必填字段（所有元素）
`type`、`id`（唯一字符串）、`x`、`y`、`width`、`height`

### 默认值（可省略——会自动应用）
- `strokeColor`: `"#1e1e1e"`
- `backgroundColor`: `"transparent"`
- `fillStyle`: `"solid"`
- `strokeWidth`: `2`
- `roughness`: `1`（手绘风格）
- `opacity`: `100`

画布背景为白色。

### 元素类型

**矩形（Rectangle）**：
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 100 }
```
- `roundness: { "type": 3 }` 表示圆角
- `backgroundColor: "#a5d8ff"`, `fillStyle: "solid"` 表示填充色

**椭圆（Ellipse）**：
```json
{ "type": "ellipse", "id": "e1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**菱形（Diamond）**：
```json
{ "type": "diamond", "id": "d1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**带标签的形状（容器绑定）**——创建一个绑定到形状的文本元素：

> **警告：** 不要在形状上使用 `"label": { "text": "..." }`。这不是有效的 Excalidraw 属性，会被静默忽略，导致形状显示为空白。必须使用下方的容器绑定方式。

形状需要在 `boundElements` 中列出文本，文本需要通过 `containerId` 反向指向形状：
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 80,
  "roundness": { "type": 3 }, "backgroundColor": "#a5d8ff", "fillStyle": "solid",
  "boundElements": [{ "id": "t_r1", "type": "text" }] },
{ "type": "text", "id": "t_r1", "x": 105, "y": 110, "width": 190, "height": 25,
  "text": "Hello", "fontSize": 20, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "r1", "originalText": "Hello", "autoResize": true }
```
- 适用于矩形、椭圆、菱形
- 设置 `containerId` 后，Excalidraw 会自动将文本居中
- 文本的 `x`/`y`/`width`/`height` 为近似值——Excalidraw 加载时会重新计算
- `originalText` 应与 `text` 保持一致
- 始终包含 `fontFamily: 1`（Virgil 手绘字体）

**带标签的箭头**——同样使用容器绑定方式：
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow",
  "boundElements": [{ "id": "t_a1", "type": "text" }] },
{ "type": "text", "id": "t_a1", "x": 370, "y": 130, "width": 60, "height": 20,
  "text": "connects", "fontSize": 16, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "a1", "originalText": "connects", "autoResize": true }
```

**独立文本**（仅用于标题和注释——无容器）：
```json
{ "type": "text", "id": "t1", "x": 150, "y": 138, "text": "Hello", "fontSize": 20,
  "fontFamily": 1, "strokeColor": "#1e1e1e", "originalText": "Hello", "autoResize": true }
```
- `x` 为左边缘。若要在位置 `cx` 处居中：`x = cx - (text.length * fontSize * 0.5) / 2`
- 不要依赖 `textAlign` 或 `width` 来定位

**箭头（Arrow）**：
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow" }
```
- `points`：相对于元素 `x`、`y` 的 `[dx, dy]` 偏移量
- `endArrowhead`：`null` | `"arrow"` | `"bar"` | `"dot"` | `"triangle"`
- `strokeStyle`：`"solid"`（默认）| `"dashed"` | `"dotted"`

### 箭头绑定（将箭头连接到形状）

```json
{
  "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 150, "height": 0,
  "points": [[0,0],[150,0]], "endArrowhead": "arrow",
  "startBinding": { "elementId": "r1", "fixedPoint": [1, 0.5] },
  "endBinding": { "elementId": "r2", "fixedPoint": [0, 0.5] }
}
```

`fixedPoint` 坐标：`top=[0.5,0]`、`bottom=[0.5,1]`、`left=[0,0.5]`、`right=[1,0.5]`

### 绘制顺序（z 轴顺序）
- 数组顺序 = z 轴顺序（第一个 = 最底层，最后一个 = 最顶层）
- 按顺序逐步输出：背景区域 → 形状 → 其绑定文本 → 其箭头 → 下一个形状
- 错误做法：所有矩形，然后所有文本，然后所有箭头
- 正确做法：bg_zone → shape1 → text_for_shape1 → arrow1 → arrow_label_text → shape2 → text_for_shape2 → ...
- 始终将绑定文本元素紧接在其容器形状之后

### 尺寸规范

**字体大小：**
- 正文文本、标签、描述的最小 `fontSize`：**16**
- 标题和大标题的最小 `fontSize`：**20**
- 次要注释的最小 `fontSize`：**14**（谨慎使用）
- 绝不使用低于 14 的 `fontSize`

**元素尺寸：**
- 带标签的矩形/椭圆最小尺寸：120x60
- 元素之间至少留 20-30px 间距
- 优先使用数量少、尺寸大的元素，而非大量细小元素

### 颜色调色板

完整颜色表见 `references/colors.md`。快速参考：

| 用途 | 填充色 | 十六进制 |
|-----|-----------|-----|
| 主要 / 输入 | 浅蓝色 | `#a5d8ff` |
| 成功 / 输出 | 浅绿色 | `#b2f2bb` |
| 警告 / 外部 | 浅橙色 | `#ffd8a8` |
| 处理 / 特殊 | 浅紫色 | `#d0bfff` |
| 错误 / 关键 | 浅红色 | `#ffc9c9` |
| 备注 / 决策 | 浅黄色 | `#fff3bf` |
| 存储 / 数据 | 浅青色 | `#c3fae8` |

### 使用技巧
- 在整个图表中保持一致的颜色调色板
- **文本对比度至关重要**——不要在白色背景上使用浅灰色。白色背景上文本颜色最低值：`#757575`
- 不要在文本中使用 emoji——Excalidraw 的字体无法渲染
- 深色模式图表，见 `references/dark-mode.md`
- 更多示例，见 `references/examples.md`