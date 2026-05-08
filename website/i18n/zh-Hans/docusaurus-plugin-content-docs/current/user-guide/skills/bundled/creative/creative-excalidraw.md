---
title: "Excalidraw — 手绘Excalidraw JSON图表（架构、流程、序列）"
sidebar_label: "Excalidraw"
description: "手绘Excalidraw JSON图表（架构、流程、序列）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Excalidraw

手绘Excalidraw JSON图表（架构、流程、序列）。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/excalidraw` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `Excalidraw`, `Diagrams`, `Flowcharts`, `Architecture`, `Visualization`, `JSON` |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Excalidraw图表技能

通过编写标准Excalidraw元素JSON并保存为`.excalidraw`文件来创建图表。这些文件可以拖放到[excalidraw.com](https://excalidraw.com)进行查看和编辑。无需账户，无需API密钥，无需渲染库 — 只有JSON。

## 使用场景

为架构图、流程图、序列图、概念图等生成`.excalidraw`文件。文件可以在excalidraw.com打开或上传以获取可分享链接。

## 工作流

1. **加载此技能**（您已经做了）
2. **编写元素JSON** — Excalidraw元素对象数组
3. **使用`write_file`保存文件**以创建`.excalidraw`文件
4. **可选上传**使用`scripts/upload.py`通过`terminal`获取可分享链接

### 保存图表

将元素数组包装在标准`.excalidraw`信封中并使用`write_file`保存：

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

保存到任何路径，例如`~/diagrams/my_diagram.excalidraw`。

### 上传以获取可分享链接

通过终端运行上传脚本（位于此技能的`scripts/`目录中）：

```bash
python skills/diagramming/excalidraw/scripts/upload.py ~/diagrams/my_diagram.excalidraw
```

这会上传到excalidraw.com（无需账户）并打印可分享URL。需要`cryptography` pip包（`pip install cryptography`）。

---

## 元素格式参考

### 所有元素的必需字段
`type`、`id`（唯一字符串）、`x`、`y`、`width`、`height`

### 默认值（跳过这些 — 自动应用）
- `strokeColor`: `"#1e1e1e"`
- `backgroundColor`: `"transparent"`
- `fillStyle`: `"solid"`
- `strokeWidth`: `2`
- `roughness`: `1`（手绘外观）
- `opacity`: `100`

画布背景是白色。

### 元素类型

**矩形：**
```json
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 100 }
```
- `roundness: { "type": 3 }` 圆角
- `backgroundColor: "#a5d8ff"`, `fillStyle: "solid"` 填充

**椭圆：**
```json
{ "type": "ellipse", "id": "e1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**菱形：**
```json
{ "type": "diamond", "id": "d1", "x": 100, "y": 100, "width": 150, "height": 150 }
```

**带标签形状（容器绑定）** — 创建绑定到形状的文本元素：

> **警告：**不要在形状上使用`"label": { "text": "..." }`。这不是有效的
> Excalidraw属性，会被静默忽略，产生空白形状。您必须
> 使用下面的容器绑定方法。

形状需要`boundElements`列出文本，文本需要`containerId`指回：
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
- 当设置`containerId`时，文本由Excalidraw自动居中
- 文本`x`/`y`/`width`/`height`是近似值 — Excalidraw加载时重新计算
- `originalText`应与`text`匹配
- 始终包含`fontFamily: 1`（Virgil/手绘字体）

**带标签箭头** — 相同的容器绑定方法：
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow",
  "boundElements": [{ "id": "t_a1", "type": "text" }] },
{ "type": "text", "id": "t_a1", "x": 370, "y": 130, "width": 60, "height": 20,
  "text": "connects", "fontSize": 16, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "a1", "originalText": "connects", "autoResize": true }
```

**独立文本**（仅标题和注释 — 无容器）：
```json
{ "type": "text", "id": "t1", "x": 150, "y": 138, "text": "Hello", "fontSize": 20,
  "fontFamily": 1, "strokeColor": "#1e1e1e", "originalText": "Hello", "autoResize": true }
```
- `x`是左边缘。要在位置`cx`居中：`x = cx - (text.length * fontSize * 0.5) / 2`
- 不要依赖`textAlign`或`width`进行定位

**箭头：**
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow" }
```
- `points`：`[dx, dy]`相对于元素`x`、`y`的偏移
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

`fixedPoint`坐标：`top=[0.5,0]`、`bottom=[0.5,1]`、`left=[0,0.5]`、`right=[1,0.5]`

### 绘制顺序（z阶）

- 数组顺序 = z阶（第一个=后面，最后=前面）
- 逐步发出：背景区域 → 形状 → 其绑定文本 → 其箭头 → 下一个形状
- 错误：所有矩形，然后所有文本，然后所有箭头
- 正确：bg_zone → shape1 → text_for_shape1 → arrow1 → arrow_label_text → shape2 → text_for_shape2 → ...
- 始终将绑定文本元素紧接在其容器形状之后

### 尺寸指南

**字体大小：**
- 最小`fontSize`：**16**用于正文文本、标签、描述
- 最小`fontSize`：**20**用于标题
- 最小`fontSize`：**14**仅用于次要注释（谨慎使用）
- 永远不要使用低于14的`fontSize`

**元素尺寸：**
- 最小形状尺寸：带标签矩形/椭圆120x60
- 元素之间至少留20-30px间隙
- 宁愿更少、更大的元素，而不是许多小元素

### 调色板

参见`references/colors.md`获取完整颜色表。快速参考：

| 用途 | 填充颜色 | 十六进制 |
|------|----------|----------|
| 主要/输入 | 浅蓝 | `#a5d8ff` |
| 成功/输出 | 浅绿 | `#b2f2bb` |
| 警告/外部 | 浅橙 | `#ffd8a8` |
| 处理/特殊 | 浅紫 | `#d0bfff` |
| 错误/关键 | 浅红 | `#ffc9c9` |
| 注释/决策 | 浅黄 | `#fff3bf` |
| 存储/数据 | 浅青 | `#c3fae8` |

### 提示

- 在整个图表中一致使用调色板
- **文本对比度至关重要** — 永远不要在白色背景上使用浅灰色。白色背景上的最小文本颜色：`#757575`
- 不要在文本中使用emoji — 它们在Excalidraw的字体中不渲染
- 对于暗模式图表，参见`references/dark-mode.md`
- 对于更大示例，参见`references/examples.md`
