---
title: "Powerpoint — 创建、读取、编辑"
sidebar_label: "Powerpoint"
description: "创建、读取、编辑"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Powerpoint

创建、读取、编辑 .pptx 幻灯片、备注、模板。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/powerpoint` |
| 许可证 | 专有。完整条款见 LICENSE.txt |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Powerpoint Skill

## 使用时机

只要涉及 .pptx 文件——无论作为输入、输出还是两者兼有——均使用此 skill。包括：创建幻灯片、演示文稿或 pitch deck；读取、解析或提取任意 .pptx 文件中的文本（即使提取的内容将用于其他地方，如邮件或摘要）；编辑、修改或更新现有演示文稿；合并或拆分幻灯片文件；处理模板、布局、演讲者备注或注释。只要用户提到"deck"、"slides"、"presentation"或引用了 .pptx 文件名，无论之后计划如何使用内容，均触发此 skill。如果需要打开、创建或操作 .pptx 文件，请使用此 skill。

## 快速参考

| 任务 | 指南 |
|------|-------|
| 读取/分析内容 | `python -m markitdown presentation.pptx` |
| 基于模板编辑或创建 | 阅读 [editing.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/editing.md) |
| 从零创建 | 阅读 [pptxgenjs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/pptxgenjs.md) |

---

## 读取内容

```bash
# 文本提取
python -m markitdown presentation.pptx

# 可视化概览
python scripts/thumbnail.py presentation.pptx

# 原始 XML
python scripts/office/unpack.py presentation.pptx unpacked/
```

---

## 编辑工作流

**完整细节请阅读 [editing.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/editing.md)。**

1. 使用 `thumbnail.py` 分析模板
2. 解包 → 操作幻灯片 → 编辑内容 → 清理 → 打包

---

## 从零创建

**完整细节请阅读 [pptxgenjs.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/powerpoint/pptxgenjs.md)。**

在没有模板或参考演示文稿时使用。

---

## 设计建议

**不要创建无聊的幻灯片。** 白底纯文字列表不会给任何人留下深刻印象。请针对每张幻灯片参考以下建议。

### 开始之前

- **选择大胆、契合内容的配色方案**：配色应专为该主题而设计。如果把你的配色套用到完全不同的演示文稿中仍然"可用"，说明选择还不够具体。
- **主次分明，而非平均分配**：一种颜色应占主导地位（60-70% 视觉比重），搭配 1-2 种辅助色和一种鲜明的强调色。切勿让所有颜色平分秋色。
- **深浅对比**：标题页和结尾页用深色背景，内容页用浅色（"三明治"结构）。或全程使用深色背景以营造高端感。
- **坚持一种视觉母题**：选择一种独特元素并贯穿始终——圆角图片框、彩色圆圈内的图标、单侧粗边框。在每张幻灯片上保持一致。

### 配色方案

根据主题选择配色，不要默认使用通用蓝色。以下配色方案仅供参考：

| 主题 | 主色 | 辅助色 | 强调色 |
|-------|---------|-----------|--------|
| **午夜商务** | `1E2761`（深海蓝） | `CADCFC`（冰蓝） | `FFFFFF`（白） |
| **森林苔藓** | `2C5F2D`（森林绿） | `97BC62`（苔绿） | `F5F5F5`（米白） |
| **珊瑚活力** | `F96167`（珊瑚红） | `F9E795`（金黄） | `2F3C7E`（深蓝） |
| **暖陶土** | `B85042`（陶土红） | `E7E8D1`（沙色） | `A7BEAE`（鼠尾草绿） |
| **海洋渐变** | `065A82`（深蓝） | `1C7293`（青蓝） | `21295C`（午夜蓝） |
| **炭灰极简** | `36454F`（炭灰） | `F2F2F2`（近白） | `212121`（黑） |
| **青蓝信任** | `028090`（青蓝） | `00A896`（海泡绿） | `02C39A`（薄荷绿） |
| **浆果奶油** | `6D2E46`（浆果紫） | `A26769`（玫瑰灰） | `ECE2D0`（奶油） |
| **鼠尾草静谧** | `84B59F`（鼠尾草绿） | `69A297`（桉叶绿） | `50808E`（石板蓝） |
| **樱桃醒目** | `990011`（樱桃红） | `FCF6F5`（近白） | `2F3C7E`（深蓝） |

### 每张幻灯片

**每张幻灯片都需要视觉元素**——图片、图表、图标或形状。纯文字幻灯片令人印象全无。

**布局选项：**
- 双栏（左文字，右插图）
- 图标 + 文字行（彩色圆圈内图标，粗体标题，下方描述）
- 2x2 或 2x3 网格（一侧图片，另一侧内容块网格）
- 半出血图片（左侧或右侧全满）配内容叠加

**数据展示：**
- 大数字标注（60-72pt 大号数字，下方小标签）
- 对比列（前后对比、优缺点、并排选项）
- 时间线或流程图（编号步骤、箭头）

**视觉精修：**
- 章节标题旁的小彩色圆圈内放图标
- 关键数据或标语使用斜体强调文字

### 字体排版

**选择有趣的字体搭配**——不要默认使用 Arial。选择一种有个性的标题字体，搭配简洁的正文字体。

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

| 元素 | 字号 |
|---------|------|
| 幻灯片标题 | 36-44pt 粗体 |
| 章节标题 | 20-24pt 粗体 |
| 正文 | 14-16pt |
| 说明文字 | 10-12pt 弱化色 |

### 间距

- 最小 0.5" 边距
- 内容块之间 0.3-0.5"
- 留有呼吸空间——不要填满每一寸

### 避免（常见错误）

- **不要重复相同布局**——在幻灯片间变换列、卡片和标注
- **不要居中对齐正文**——段落和列表左对齐；仅标题居中
- **不要忽视字号对比**——标题需 36pt 以上才能从 14-16pt 正文中突出
- **不要默认使用蓝色**——选择能反映具体主题的颜色
- **不要随意混用间距**——选定 0.3" 或 0.5" 的间隔后保持一致
- **不要只精心设计一张幻灯片而其余保持简陋**——要么全力投入，要么全程保持简洁
- **不要创建纯文字幻灯片**——添加图片、图标、图表或视觉元素；避免纯标题 + 列表
- **不要忘记文本框内边距**——将线条或形状与文字边缘对齐时，将文本框的 `margin` 设为 `0`，或偏移形状以补偿内边距
- **不要使用低对比度元素**——图标和文字都需要与背景形成强烈对比；避免浅色背景上的浅色文字或深色背景上的深色文字
- **绝对不要在标题下方使用装饰线**——这是 AI 生成幻灯片的典型特征；改用留白或背景色

---

## QA（必须执行）

**假设存在问题。你的任务是找出它们。**

第一次渲染几乎从不正确。将 QA 视为查找 bug，而非确认步骤。如果第一次检查没有发现任何问题，说明你看得还不够仔细。

### 内容 QA

```bash
python -m markitdown output.pptx
```

检查缺失内容、错别字、顺序错误。

**使用模板时，检查是否残留占位符文本：**

```bash
python -m markitdown output.pptx | grep -iE "xxxx|lorem|ipsum|this.*(page|slide).*layout"
```

如果 grep 返回结果，在宣告完成前先修复。

### 视觉 QA

**⚠️ 使用子 agent**——即使只有 2-3 张幻灯片。你一直盯着代码，会看到你期望看到的，而非实际存在的。子 agent 有全新的视角。

将幻灯片转换为图片（见[转换为图片](#converting-to-images)），然后使用以下 prompt（提示词）：

```
Visually inspect these slides. Assume there are issues — find them.

Look for:
- Overlapping elements (text through shapes, lines through words, stacked elements)
- Text overflow or cut off at edges/box boundaries
- Decorative lines positioned for single-line text but title wrapped to two lines
- Source citations or footers colliding with content above
- Elements too close (< 0.3" gaps) or cards/sections nearly touching
- Uneven gaps (large empty area in one place, cramped in another)
- Insufficient margin from slide edges (< 0.5")
- Columns or similar elements not aligned consistently
- Low-contrast text (e.g., light gray text on cream-colored background)
- Low-contrast icons (e.g., dark icons on dark backgrounds without a contrasting circle)
- Text boxes too narrow causing excessive wrapping
- Leftover placeholder content

For each slide, list issues or areas of concern, even if minor.

Read and analyze these images:
1. /path/to/slide-01.jpg (Expected: [brief description])
2. /path/to/slide-02.jpg (Expected: [brief description])

Report ALL issues found, including minor ones.
```

### 验证循环

1. 生成幻灯片 → 转换为图片 → 检查
2. **列出发现的问题**（如果未发现任何问题，请更严格地再看一遍）
3. 修复问题
4. **重新验证受影响的幻灯片**——一处修复往往会引发另一个问题
5. 重复，直到完整检查一遍后不再出现新问题

**在完成至少一次修复并验证的循环之前，不得宣告成功。**

---

## 转换为图片

将演示文稿转换为单张幻灯片图片以供视觉检查：

```bash
python scripts/office/soffice.py --headless --convert-to pdf output.pptx
pdftoppm -jpeg -r 150 output.pdf slide
```

这将生成 `slide-01.jpg`、`slide-02.jpg` 等文件。

修复后重新渲染特定幻灯片：

```bash
pdftoppm -jpeg -r 150 -f N -l N output.pdf slide-fixed
```

---

## 依赖项

- `pip install "markitdown[pptx]"` - 文本提取
- `pip install Pillow` - 缩略图网格
- `npm install -g pptxgenjs` - 从零创建
- LibreOffice（`soffice`）- PDF 转换（通过 `scripts/office/soffice.py` 为沙箱环境自动配置）
- Poppler（`pdftoppm`）- PDF 转图片