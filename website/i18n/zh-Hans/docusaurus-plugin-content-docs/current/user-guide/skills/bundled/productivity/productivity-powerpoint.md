---
title: "Powerpoint — 创建、读取、编辑"
sidebar_label: "Powerpoint"
description: "创建、读取、编辑"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Powerpoint

创建、读取、编辑 .pptx 演示文稿、幻灯片、备注、模板。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/powerpoint` |
| 许可证 | 专有。LICENSE.txt 包含完整条款 |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Powerpoint 技能

## 何时使用

只要涉及 .pptx 文件 — 无论作为输入、输出还是两者，都使用此技能。包括：创建幻灯片组、推介稿或演示文稿；读取、解析或从任何 .pptx 文件中提取文本（即使提取的内容将用于其他地方，如邮件或摘要）；编辑、修改或更新现有演示文稿；合并或拆分幻灯片文件；使用模板、布局、演讲者备注或评论。当用户提到"演示文稿"、"幻灯片"、"presentation"或引用 .pptx 文件名时触发，无论他们之后计划如何处理内容。如果 .pptx 文件需要被打开、创建或操作，使用此技能。

## 快速参考

| 任务 | 指南 |
|------|-------|
| 读取/分析内容 | `python -m markitdown presentation.pptx` |
| 从模板编辑或创建 | 阅读 [editing.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/editing.md) |
| 从零创建 | 阅读 [pptxgenjs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/pptxgenjs.md) |

---

## 读取内容

```bash
# 文本提取
python -m markitdown presentation.pptx

# 视觉概览
python scripts/thumbnail.py presentation.pptx

# 原始 XML
python scripts/office/unpack.py presentation.pptx unpacked/
```

---

## 编辑工作流

**阅读 [editing.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/editing.md) 了解完整详情。**

1. 使用 `thumbnail.py` 分析模板
2. 解包 → 操作幻灯片 → 编辑内容 → 清理 → 打包

---

## 从零创建

**阅读 [pptxgenjs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/pptxgenjs.md) 了解完整详情。**

当没有模板或参考演示文稿时使用。

---

## 设计思路

**不要创建无聊的幻灯片。** 白色背景上的纯要点不会给任何人留下印象。考虑从以下列表中为每张幻灯片选择设计思路。

### 开始之前

- **选择大胆的、与内容相关的配色方案**：配色方案应该感觉是为此主题设计的。如果你的颜色放到完全不同的演示文稿中仍然"适用"，说明你的选择不够具体。
- **主次分明而非均等分配**：一种颜色应占主导（60-70% 视觉权重），1-2 种支持色调和一种鲜明的强调色。永远不要给所有颜色相同的权重。
- **深浅对比**：深色背景用于标题 + 结论幻灯片，浅色用于内容（"三明治"结构）。或全程使用深色以获得高端感。
- **坚持一个视觉主题**：选择一个独特元素并重复使用 — 圆角图片框、彩色圆圈中的图标、粗单侧边框。贯穿每张幻灯片。

### 配色方案

选择与主题匹配的颜色 — 不要默认使用通用蓝色。使用以下配色方案作为灵感：

| 主题 | 主色 | 辅色 | 强调色 |
|-------|---------|-----------|--------|
| **午夜高管** | `1E2761`（海军蓝） | `CADCFC`（冰蓝） | `FFFFFF`（白色） |
| **森林与苔藓** | `2C5F2D`（森林绿） | `97BC62`（苔藓绿） | `F5F5F5`（奶油色） |
| **珊瑚活力** | `F96167`（珊瑚红） | `F9E795`（金色） | `2F3C7E`（海军蓝） |
| **暖赤陶** | `B85042`（赤陶色） | `E7E8D1`（沙色） | `A7BEAE`（鼠尾草绿） |
| **海洋渐变** | `065A82`（深蓝） | `1C7293`（青色） | `21295C`（午夜蓝） |
| **炭灰极简** | `36454F`（炭灰色） | `F2F2F2`（灰白色） | `212121`（黑色） |
| **青绿信赖** | `028090`（青绿色） | `00A896`（海泡色） | `02C39A`（薄荷色） |
| **浆果与奶油** | `6D2E46`（浆果色） | `A26769`（玫灰色） | `ECE2D0`（奶油色） |
| **鼠尾草宁静** | `84B59F`（鼠尾草绿） | `69A297`（桉树色） | `50808E`（石板灰） |
| **樱桃大胆** | `990011`（樱桃红） | `FCF6F5`（灰白色） | `2F3C7E`（海军蓝） |

### 每张幻灯片

**每张幻灯片都需要视觉元素** — 图像、图表、图标或形状。纯文本幻灯片是容易被遗忘的。

**布局选项：**
- 双栏（左侧文本，右侧插图）
- 图标 + 文本行（彩色圆圈中的图标，粗体标题，下方描述）
- 2x2 或 2x3 网格（一侧图片，另一侧内容块网格）
- 半出血图片（左侧或右侧全幅）配内容叠加

**数据显示：**
- 大号统计数据标注（60-72pt 大号数字，下方小标签）
- 对比栏（之前/之后、优/缺、并排选项）
- 时间线或流程图（编号步骤、箭头）

**视觉润色：**
- 章节标题旁彩色小圆圈中的图标
- 关键统计或标语使用斜体强调文字

### 字体排版

**选择有趣的字体搭配** — 不要默认使用 Arial。选择有个性的标题字体并搭配简洁的正文字体。

| 标题字体 | 正文字体 |
|-------------|-----------|
| Georgia | Calibri |
| Arial Black | Arial |
| Calibri | Calibri Light |
| Cambria | Calibri |
| Trebuchet MS | Calibri |
| Impact | Arial |
| Palatino | Garamond |
| Consolas | Calibri |

| 元素 | 大小 |
|---------|------|
| 幻灯片标题 | 36-44pt 粗体 |
| 章节标题 | 20-24pt 粗体 |
| 正文 | 14-16pt |
| 说明文字 | 10-12pt 淡色 |

### 间距

- 最小 0.5" 边距
- 内容块之间 0.3-0.5"
- 留出呼吸空间 — 不要填满每一寸

### 避免（常见错误）

- **不要重复相同布局** — 在幻灯片之间变化栏、卡片和标注
- **不要居中正文** — 段落和列表左对齐；仅标题居中
- **不要吝啬大小对比** — 标题需要 36pt+ 才能从 14-16pt 正文中突出
- **不要默认使用蓝色** — 选择反映特定主题的颜色
- **不要随机混搭间距** — 选择 0.3" 或 0.5" 间距并一致使用
- **不要只设计一张幻灯片而其余保持朴素** — 要么全面设计，要么全程保持简洁
- **不要创建纯文本幻灯片** — 添加图像、图标、图表或视觉元素；避免纯标题 + 要点
- **不要忘记文本框内边距** — 将线条或形状与文本边缘对齐时，在文本框上设置 `margin: 0` 或偏移形状以考虑内边距
- **不要使用低对比度元素** — 图标和文本都需要与背景的强烈对比；避免浅色背景上的浅色文本或深色背景上的深色文本
- **绝对不要在标题下使用强调线** — 这是 AI 生成幻灯片的标志；改用空白或背景颜色

---

## QA（必需）

**假设存在问题。你的工作是找到它们。**

你的第一次渲染几乎从不正确。把 QA 当作 bug 猎杀，而不是确认步骤。如果你在首次检查时没有发现任何问题，说明你检查得不够仔细。

### 内容 QA

```bash
python -m markitdown output.pptx
```

检查缺失内容、错别字、错误顺序。

**使用模板时，检查残留的占位符文本：**

```bash
python -m markitdown output.pptx | grep -iE "xxxx|lorem|ipsum|this.*(page|slide).*layout"
```

如果 grep 返回结果，在宣布成功之前修复它们。

### 视觉 QA

**⚠️ 使用子代理** — 即使只有 2-3 张幻灯片。你一直在看代码，会看到你期望的内容，而不是实际内容。子代理有全新的视角。

将幻灯片转换为图像（参见[转换为图像](#converting-to-images)），然后使用此提示：

```
目视检查这些幻灯片。假设存在问题 — 找到它们。

查找：
- 重叠元素（文本穿过形状、线条穿过文字、堆叠元素）
- 文本溢出或在边缘/框边界处被截断
- 装饰线为单行文本定位但标题换行到两行
- 来源引用或页脚与上方内容冲突
- 元素过近（< 0.3" 间距）或卡片/部分几乎相触
- 不均匀间距（一处大量空白，另一处拥挤）
- 距幻灯片边缘边距不足（< 0.5"）
- 栏或类似元素未一致对齐
- 低对比度文本（如浅灰色文本在奶油色背景上）
- 低对比度图标（如深色图标在深色背景上无对比圆圈）
- 文本框过窄导致过度换行
- 残留的占位符内容

对于每张幻灯片，列出问题或关注区域，即使是次要的。

阅读并分析这些图像：
1. /path/to/slide-01.jpg（预期：[简要描述]）
2. /path/to/slide-02.jpg（预期：[简要描述]）

报告发现的所有问题，包括次要问题。
```

### 验证循环

1. 生成幻灯片 → 转换为图像 → 检查
2. **列出发现的问题**（如果没有发现，更严格地再检查一遍）
3. 修复问题
4. **重新验证受影响的幻灯片** — 一个修复通常会引入另一个问题
5. 重复直到完整的一轮检查不再发现新问题

**在完成至少一个修复-验证循环之前，不要宣布成功。**

---

## 转换为图像

将演示文稿转换为单独的幻灯片图像以进行目视检查：

```bash
python scripts/office/soffice.py --headless --convert-to pdf output.pptx
pdftoppm -jpeg -r 150 output.pdf slide
```

这会创建 `slide-01.jpg`、`slide-02.jpg` 等。

要在修复后重新渲染特定幻灯片：

```bash
pdftoppm -jpeg -r 150 -f N -l N output.pdf slide-fixed
```

---

## 依赖项

- `pip install "markitdown[pptx]"` - 文本提取
- `pip install Pillow` - 缩略图网格
- `npm install -g pptxgenjs` - 从零创建
- LibreOffice (`soffice`) - PDF 转换（通过 `scripts/office/soffice.py` 为沙盒环境自动配置）
- Poppler (`pdftoppm`) - PDF 转图像
