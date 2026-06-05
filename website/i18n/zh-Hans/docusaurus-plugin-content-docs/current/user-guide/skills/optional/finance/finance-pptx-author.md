---
title: "Pptx Author — 使用 python-pptx 无头构建 PowerPoint 演示文稿"
sidebar_label: "Pptx Author"
description: "使用 python-pptx 无头构建 PowerPoint 演示文稿"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pptx Author

使用 python-pptx 无头构建 PowerPoint 演示文稿。与 excel-author 配合使用，可构建每个数字都追溯到工作簿单元格的模型驱动演示文稿。适用于融资路演材料、IC 备忘录、盈利说明。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/finance/pptx-author` 安装 |
| 路径 | `optional-skills/finance/pptx-author` |
| 版本 | `1.0.0` |
| 作者 | Anthropic（由 Nous Research 改编） |
| 许可证 | Apache-2.0 |
| 平台 | linux, macos, windows |
| 标签 | `powerpoint`, `pptx`, `python-pptx`, `presentation`, `finance` |
| 相关 skill | [`excel-author`](/user-guide/skills/optional/finance/finance-excel-author), [`powerpoint`](/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# pptx-author

使用 `python-pptx` 在磁盘上生成 .pptx 文件。当需要将演示文稿作为文件产物交付，而非驱动实时 PowerPoint 会话时使用。

改编自 Anthropic 在 [anthropics/financial-services](https://github.com/anthropics/financial-services) 中的 `pptx-author` 和 `pitch-deck` skill。原版中的 MCP / Office-JS 分支已移除 — 本 skill 假定使用无头 Python。

如需更全面的、已内置的 PowerPoint 创作 skill（幻灯片、演讲者备注、嵌入、媒体），请参阅内置的 `powerpoint` skill。本 skill 是一个更轻量的模式，专为模型驱动的演示文稿（融资路演、IC 备忘录、盈利说明）调优，要求每个数字都必须追溯到源工作簿。

## 输出约定

- 写入 `./out/<name>.pptx`。如果 `./out/` 不存在则创建。
- 在最终消息中返回相对路径。

## 安装

```bash
pip install "python-pptx>=0.6"
```

## 核心约定

### 每张幻灯片一个观点
标题陈述结论；正文支撑结论。标题为"Q3 Revenue"的幻灯片表达力弱；"Revenue growth accelerated to 14% Y/Y in Q3"则更有力。

### 每个数字都追溯到模型
如果幻灯片上的数字来自 `./out/model.xlsx`，则在脚注中注明工作表和单元格。

```
Revenue: $1,250M  (Source: model.xlsx, Inputs!C3)
```

切勿凭记忆或摘要转录数字 — 打开工作簿，读取命名区域，并在可能的情况下以编程方式将演示文稿中的值绑定到工作簿。

### 存在公司模板时使用公司模板
如果 `./templates/firm-template.pptx` 存在，则加载它，使演示文稿继承品牌颜色、字体和母版布局。

```python
from pptx import Presentation
from pathlib import Path

template = Path("./templates/firm-template.pptx")
prs = Presentation(str(template)) if template.exists() else Presentation()
```

### 图表：从模型导出 PNG 优于原生 pptx 图表
当保真度要求较高时（模型的图表样式必须与演示文稿完全匹配），从源工作簿将图表渲染为 PNG 并嵌入图片。原生 `pptx.chart` 图表较脆弱，且通常不符合公司规范。

```python
from pptx.util import Inches
slide.shapes.add_picture("./out/charts/football_field.png",
                         Inches(1), Inches(2),
                         width=Inches(8))
```

### 不对外发送
本 skill 只写入文件，不发送邮件、上传或发布。交付由编排层处理。

## 骨架代码

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pathlib import Path

template = Path("./templates/firm-template.pptx")
prs = Presentation(str(template)) if template.exists() else Presentation()

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Project Aurora — Strategic Alternatives"
slide.placeholders[1].text = "Preliminary Discussion Materials"

# Valuation summary slide (title-only layout)
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Valuation implies $38–$52 per share across methodologies"

# Add a table bound to model outputs
rows, cols = 5, 4
tbl_shape = slide.shapes.add_table(rows, cols,
                                   Inches(0.5), Inches(1.5),
                                   Inches(9), Inches(3))
tbl = tbl_shape.table
headers = ["Methodology", "Low ($)", "Mid ($)", "High ($)"]
for c, h in enumerate(headers):
    tbl.cell(0, c).text = h

# In a real deck, read these from the model workbook with openpyxl
data = [
    ("Trading comps",     "35", "41", "48"),
    ("Precedent M&A",     "39", "45", "52"),
    ("DCF (base)",        "36", "43", "51"),
    ("LBO (10% IRR)",     "33", "38", "44"),
]
for r, row in enumerate(data, start=1):
    for c, val in enumerate(row):
        tbl.cell(r, c).text = val

# Embed a chart rendered from the model
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Football field — current price $42"
slide.shapes.add_picture("./out/charts/football_field.png",
                         Inches(1), Inches(1.8), width=Inches(8))

Path("./out").mkdir(exist_ok=True)
prs.save("./out/pitch-aurora.pptx")
```

## 将演示文稿数字绑定到源工作簿

从 Excel 模型中读取命名区域或特定单元格，确保演示文稿中的数字不会偏离。

```python
from openpyxl import load_workbook

wb = load_workbook("./out/model.xlsx", data_only=True)
def nr(name):
    """Resolve a named range to its current computed value."""
    rng = wb.defined_names[name]
    sheet, coord = next(rng.destinations)
    return wb[sheet][coord].value

revenue_fy24 = nr("RevenueFY24")
implied_mid  = nr("ImpliedSharePriceBase")
```

然后使用这些值构建演示文稿内容：
```python
slide.shapes.title.text = f"Implied share price of ${implied_mid:.2f} (base case)"
```

请记住在读取工作簿之前重新计算 — openpyxl 只有在工作表已经被计算过的情况下才能看到计算值。请先运行 `excel-author` skill 中的重算辅助函数，或通过真实的 Excel 会话打开并保存。

## 融资路演幻灯片类型清单

典型的投行融资路演演示文稿遵循以下结构。不作强制要求，但可作为起始骨架参考：

1. 封面 / 标题页
2. 免责声明
3. 目录
4. 情况概述
5. 公司概况（目标公司）
6. 市场 / 行业背景
7. 估值摘要（football field）— 核心幻灯片
8. 可比交易详情
9. 先例交易详情
10. DCF 摘要
11. 示意性 LBO / 财务投资人情景
12. 流程考量
13. 附录

## 不适用本 skill 的情形

- 用户正在进行实时 PowerPoint 会话且有 Office MCP 可用 — 应直接驱动其实时文档。
- 非金融类幻灯片（季度全员会议、市场营销演示文稿）— 使用更全面的 `powerpoint` skill。
- 包含大量动画、切换效果或演讲者备注的演示文稿 — 使用更全面的 `powerpoint` skill。

## 致谢

约定改编自 Anthropic 的 Claude for Financial Services 插件套件，采用 Apache-2.0 许可证。原始来源：https://github.com/anthropics/financial-services/tree/main/plugins/agent-plugins/pitch-agent/skills/pptx-author