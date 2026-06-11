---
title: "Excel Author"
sidebar_label: "Excel Author"
description: "使用 openpyxl 无头构建可审计的 Excel 工作簿——蓝/黑/绿单元格约定、公式优先于硬编码、命名范围、余额检查、敏感性表格。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Excel Author

使用 openpyxl 无头构建可审计的 Excel 工作簿——蓝/黑/绿单元格约定、公式优先于硬编码、命名范围、余额检查、敏感性表格。适用于财务模型、审计输出、对账。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选——通过 `hermes skills install official/finance/excel-author` 安装 |
| 路径 | `optional-skills/finance/excel-author` |
| 版本 | `1.0.0` |
| 作者 | Anthropic（由 Nous Research 改编） |
| 许可证 | Apache-2.0 |
| 平台 | linux, macos, windows |
| 标签 | `excel`, `openpyxl`, `finance`, `spreadsheet`, `modeling` |
| 相关 skill | [`pptx-author`](/user-guide/skills/optional/finance/finance-pptx-author)、[`dcf-model`](/user-guide/skills/optional/finance/finance-dcf-model)、[`comps-analysis`](/user-guide/skills/optional/finance/finance-comps-analysis)、[`lbo-model`](/user-guide/skills/optional/finance/finance-lbo-model)、[`3-statement-model`](/user-guide/skills/optional/finance/finance-3-statement-model) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# excel-author

使用 `openpyxl` 在磁盘上生成 .xlsx 文件。遵循以下银行级约定，使模型可审计、灵活，并可由构建者以外的人审阅。

改编自 Anthropic 在 [anthropics/financial-services](https://github.com/anthropics/financial-services) 仓库中的 `xlsx-author` 和 `audit-xls` skill。原版中的 MCP / Office-JS / Cowork 相关分支已去除——本 skill 假设使用无头 Python。

## 输出约定

- 写入 `./out/<name>.xlsx`。如果 `./out/` 不存在则创建。
- 在最终消息中返回相对路径，以便下游工具获取。
- 每个文件对应一个逻辑模型。除非明确要求，否则不向已有工作簿追加内容。

## 安装

```bash
pip install "openpyxl>=3.0"
```

## 核心约定（不可更改）

### 蓝/黑/绿单元格颜色
- **蓝色**（`Font(color="0000FF")`）——人工输入的硬编码值。收入驱动因素、WACC 输入、终值增长率、市场数据。
- **黑色**（默认）——公式。每个派生单元格均为实时 Excel 公式。
- **绿色**（`Font(color="006100")`）——链接到另一张工作表或外部文件。

审阅者可以扫描工作表，立即区分假设值与计算值。

### 公式优先于硬编码
每个计算单元格必须是公式字符串，绝不能是在 Python 中计算后粘贴的数值。

```python
# 错误——潜在的隐性 bug
ws["D20"] = revenue_prior_year * (1 + growth)

# 正确——用户更改假设时自动联动
ws["D20"] = "=D19*(1+$B$8)"
```

唯一允许硬编码的数字：
1. 原始历史输入（实际收入、报告 EBITDA 等）
2. 用户需要调整的假设驱动因素（增长率、WACC 输入、终值 g）
3. 当前市场数据（股价、债务余额）——需在单元格注释中注明来源和日期

如果你发现自己在 Python 中计算值并写入结果，请停下来。

### 跨工作表引用使用命名范围
对从另一张工作表、演示文稿或备忘录引用的任何数值，使用命名范围。

```python
from openpyxl.workbook.defined_name import DefinedName
wb.defined_names["WACC"] = DefinedName("WACC", attr_text="Inputs!$C$8")
# 然后在其他地方：
calc["D30"] = "=D29/WACC"
```

### 余额检查标签页
包含一个 `Checks` 标签页，汇总所有内容并显示 TRUE/FALSE：
- 资产负债表平衡（资产 = 负债 + 权益）
- 现金流与资产负债表上的期间现金变动一致
- 分部加总与合并总计一致
- 计算范围内无游离硬编码

示例：
```python
checks = wb.create_sheet("Checks")
checks["A2"] = "BS balances"
checks["B2"] = "=IS!D20-IS!D21-IS!D22"
checks["C2"] = "=ABS(B2)<0.01"  # TRUE/FALSE
```

### 每个硬编码输入均添加单元格注释
在创建单元格时同步添加注释，不要事后补充。

```python
from openpyxl.comments import Comment
ws["C2"] = 1_250_000_000
ws["C2"].font = Font(color="0000FF")
ws["C2"].comment = Comment("Source: 10-K FY2024, p.47, revenue line", "analyst")
```

格式：`Source: [系统/文档], [日期], [参考], [URL（如适用）]`。

绝不推迟标注来源。绝不写 `TODO: add source`。

## 骨架：典型财务模型

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from pathlib import Path

BLUE = Font(color="0000FF")
BLACK = Font(color="000000")
GREEN = Font(color="006100")
BOLD = Font(bold=True)
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True)

wb = Workbook()

# --- Inputs 标签页 ---
inp = wb.active
inp.title = "Inputs"
inp["A1"] = "MARKET DATA & KEY INPUTS"
inp["A1"].font = HEADER_FONT
inp["A1"].fill = HEADER_FILL
inp.merge_cells("A1:C1")

inp["B3"] = "Revenue FY2024"
inp["C3"] = 1_250_000_000
inp["C3"].font = BLUE
inp["C3"].comment = Comment("Source: 10-K FY2024 p.47", "model")

inp["B4"] = "Growth Rate"
inp["C4"] = 0.12
inp["C4"].font = BLUE

# --- 计算标签页 ---
calc = wb.create_sheet("DCF")
calc["B2"] = "Projected Revenue"
calc["C2"] = "=Inputs!C3*(1+Inputs!C4)"   # 公式，黑色

# --- 检查标签页 ---
chk = wb.create_sheet("Checks")
chk["A2"] = "BS balances"
chk["B2"] = "=ABS(BS!D20-BS!D21-BS!D22)<0.01"

Path("./out").mkdir(exist_ok=True)
wb.save("./out/model.xlsx")
```

## 带合并单元格的节标题

openpyxl 特性：合并时，在左上角单元格设置值，并单独对整个范围设置样式。

```python
ws["A7"] = "CASH FLOW PROJECTION"
ws["A7"].font = HEADER_FONT
ws.merge_cells("A7:H7")
for col in range(1, 9):  # A..H
    ws.cell(row=7, column=col).fill = HEADER_FILL
```

## 敏感性表格

用循环构建，不要对每个单元格硬编码公式。规则：

- **奇数行/列数**（5×5 或 7×7）——保证存在真正的中心单元格。
- **中心单元格 = 基准情景。** 中间行/列的标题必须等于模型实际的 WACC 和终值 g，使中心输出等于基准情景隐含股价。这是合理性检验。
- **高亮中心单元格**，使用中蓝色填充（`"BDD7EE"`）并加粗。
- 每个单元格均填入完整的重新计算公式——绝不使用近似值。

```python
# 5x5 WACC（行）x 终值增长率（列）敏感性
wacc_axis = [0.08, 0.085, 0.09, 0.095, 0.10]        # 中间行 = 基准 9.0%
term_axis = [0.02, 0.025, 0.03, 0.035, 0.04]        # 中间列 = 基准 3.0%

start_row = 40
ws.cell(row=start_row, column=1).value = "Implied Share Price ($)"
ws.cell(row=start_row, column=1).font = BOLD

for j, g in enumerate(term_axis):
    ws.cell(row=start_row+1, column=2+j).value = g
    ws.cell(row=start_row+1, column=2+j).font = BLUE

for i, w in enumerate(wacc_axis):
    r = start_row + 2 + i
    ws.cell(row=r, column=1).value = w
    ws.cell(row=r, column=1).font = BLUE
    for j, g in enumerate(term_axis):
        c = 2 + j
        # 完整 DCF 重新计算公式（此处为简化示意）。
        # 在实际模型中，此处引用完整的预测区块。
        ws.cell(row=r, column=c).value = (
            f"=SUMPRODUCT(FCF_range,1/(1+{w})^year_offset) + "
            f"FCF_terminal*(1+{g})/({w}-{g})/(1+{w})^terminal_year"
        )

# 高亮中心单元格（基准情景）
center = ws.cell(row=start_row+2+len(wacc_axis)//2,
                 column=2+len(term_axis)//2)
center.fill = PatternFill("solid", fgColor="BDD7EE")
center.font = BOLD
```

## 交付前重新计算

openpyxl 写入公式字符串但不计算结果。Excel 打开时会重新计算，但下游消费者（自动检查脚本、CI）需要已计算的值。

交付前运行 LibreOffice 或专用重新计算步骤：

```bash
# LibreOffice 无头重新计算
libreoffice --headless --calc --convert-to xlsx ./out/model.xlsx --outdir ./out/
```

或使用 Python 重新计算辅助工具（参见本 skill 中的 `scripts/recalc.py`）。

## 模型布局规划

在编写任何公式之前：
1. 定义所有节的行位置
2. 写入所有标题和标签
3. 写入所有节分隔符和空行
4. 然后使用锁定的行位置编写公式

这可以避免在公式写入后插入标题行导致所有下游引用偏移的级联公式损坏问题。

## 与用户逐步验证

对于大型模型（DCF、三表模型、LBO），在继续之前停下来向用户展示中间产物。在构建下游敏感性表格之前发现错误的利润率假设，可以节省一小时。

检查点模式：
- Inputs 区块完成后→展示原始输入，确认后再进行预测
- 收入预测完成后→确认顶线收入和增长率
- FCF 构建完成后→确认完整的计划表
- WACC 完成后→确认输入
- 估值完成后→确认权益桥接
- 然后构建敏感性表格

## 不适用场景

- 用户在实时 Excel 会话中且有 Office MCP 可用——直接操作其实时工作簿。
- 纯表格数据导出且无公式——使用 `csv` 或 `pandas.to_excel` 更简单。
- 具有大量交互性的仪表板/图表——使用专业 BI 工具。

## 致谢

蓝/黑/绿约定、公式优先于硬编码、命名范围、敏感性规则等约定，改编自 Anthropic 的 Claude for Financial Services 插件套件，采用 Apache-2.0 许可证。原始地址：https://github.com/anthropics/financial-services/tree/main/plugins/vertical-plugins/financial-analysis/skills/xlsx-author