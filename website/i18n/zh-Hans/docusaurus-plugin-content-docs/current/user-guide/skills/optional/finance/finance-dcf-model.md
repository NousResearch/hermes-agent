---
title: "DCF 模型"
sidebar_label: "Dcf Model"
description: "在 Excel 中构建机构级 DCF 估值模型——收入预测、FCF 构建、WACC、终值、熊/基/牛情景、5x5 敏感性表格。与 excel-author 配合使用。适用于内在价值股权分析。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# DCF 模型

在 Excel 中构建机构级 DCF 估值模型——收入预测、FCF 构建、WACC、终值、熊/基/牛情景、5x5 敏感性表格。与 excel-author 配合使用。适用于内在价值股权分析。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选——通过 `hermes skills install official/finance/dcf-model` 安装 |
| 路径 | `optional-skills/finance/dcf-model` |
| 版本 | `1.0.0` |
| 作者 | Anthropic（由 Nous Research 改编） |
| 许可证 | Apache-2.0 |
| 平台 | linux, macos, windows |
| 标签 | `finance`, `valuation`, `dcf`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 相关 skill | [`excel-author`](/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/user-guide/skills/optional/finance/finance-pptx-author), [`comps-analysis`](/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/user-guide/skills/optional/finance/finance-lbo-model), [`3-statement-model`](/user-guide/skills/optional/finance/finance-3-statement-model) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

## 环境

本 skill 假定使用**无头 openpyxl**——你在磁盘上生成 .xlsx 文件。
遵循 `excel-author` skill 关于单元格着色、公式、命名区域和敏感性表格的约定。
交付前重新计算：`python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`。

# DCF 模型构建器

## 概述

本 skill 按照投资银行标准创建机构级 DCF 模型用于股权估值。每次分析生成一个详细的 Excel 模型（敏感性分析包含在 DCF 工作表底部）。

## 工具

- 默认使用用户提供的所有信息以及可用于数据获取的 MCP 服务器。

## 关键约束——请先阅读

以下约束适用于所有 DCF 模型构建过程。开始前请仔细阅读：

**公式优先于硬编码（不可协商）：**
- 每个预测值、利润率、折现因子、现值和敏感性单元格都必须是实时 Excel 公式——绝不能是在 Python 中计算后写入的数值
- 使用 openpyxl 时：`ws["D20"] = "=D19*(1+$B$8)"` 是正确的；`ws["D20"] = calculated_revenue` 是错误的
- 唯一允许硬编码的数字是：(1) 原始历史输入，(2) 假设驱动因子（增长率、WACC 输入、终端 g），(3) 当前市场数据（股价、债务余额）
- 如果你发现自己在 Python 中计算某个值并将结果写入——停止。模型必须在用户更改假设时能够动态调整。

**逐步与用户确认（不要端到端构建）：**
- 数据获取后→向用户展示原始输入块（收入、利润率、股份数、净债务）并在预测前确认
- 收入预测后→展示预测的顶线和增长率，在构建利润率之前确认
- FCF 构建后→展示完整的 FCF 计划，在计算 WACC 前确认逻辑
- WACC 后→展示计算过程和输入，在折现前确认
- 终值 + 现值后→展示股权桥接（EV → 股权价值 → 每股价值），在敏感性表格前确认
- 在每个阶段捕捉错误——在敏感性表格构建完成后才发现错误的利润率假设意味着需要重建所有下游内容

**敏感性表格：**
- **使用奇数行和列**（标准：5×5，有时 7×7）——这保证了一个真正的中心单元格
- **中心单元格 = 基准情景。** 构建轴值时，使中间行标题和中间列标题恰好等于模型的实际假设（例如，如果基准 WACC = 9.0%，则中间行为 9.0%；如果终端 g = 3.0%，则中间列为 3.0%）。中心单元格的输出因此必须等于模型的实际隐含每股价格——这是验证表格构建正确的合理性检验。
- **高亮中心单元格**，使用中蓝色填充（`#BDD7EE`）+ 粗体字体，使基准情景立即可见。
- 用完整的 DCF 重新计算公式填充所有单元格（通常 3 张表 × 25 个单元格 = 75 个）
- 使用 openpyxl 循环以编程方式写入公式
- 不得有占位文本、不得有线性近似、不得需要手动步骤
- 每个单元格必须针对该假设组合重新计算完整的 DCF

**单元格注释：**
- 在创建每个硬编码值时添加单元格注释
- 格式："Source: [System/Document], [Date], [Reference], [URL if applicable]"
- 每个蓝色输入在进入下一节之前必须有注释
- 不要推迟到最后或写"TODO: add source"

**模型布局规划：**
- 在写任何公式之前定义所有节的行位置
- 先写所有标题和标签
- 其次写所有节分隔符和空行
- 然后使用锁定的行位置写公式
- 创建后立即测试公式

**公式重新计算：**
- 交付前运行 `python recalc.py model.xlsx 30`
- 修复所有错误直到状态为"success"
- 要求零公式错误（#REF!、#DIV/0!、#VALUE! 等）

**情景块：**
- 为熊/基/牛情景创建独立块
- 在每个块内横向展示各预测年份的假设
- 使用 IF 公式：`=IF($B$6=1,[Bear cell],IF($B$6=2,[Base cell],[Bull cell]))`
- 验证公式引用了正确的情景块单元格

## DCF 流程工作流

### 第 1 步：数据获取与验证

从 MCP 服务器、用户提供的数据和网络获取数据。

**数据来源优先级：**
1. **MCP 服务器**（如已配置）——来自 Daloopa 等提供商的结构化财务数据
2. **用户提供的数据**——来自其研究的历史财务数据
3. **网络搜索/抓取**——需要时获取当前价格、beta、债务和现金

**验证清单：**
- 验证净债务与净现金（对估值至关重要）
- 确认稀释后流通股数（检查近期回购/发行）
- 验证历史利润率与商业模式一致
- 将收入增长率与行业基准交叉核对
- 验证税率合理（通常 21-28%）

### 第 2 步：历史分析（3-5 年）

分析并记录：
- **收入增长趋势**：计算 CAGR，识别驱动因素
- **利润率进展**：跟踪毛利率、EBIT 利润率、FCF 利润率
- **资本密集度**：D&A 和资本支出占收入的百分比
- **营运资金效率**：NWC 变化占收入增长的百分比
- **回报指标**：ROIC、ROE 趋势

创建汇总表格，显示：
```
Historical Metrics (LTM):
Revenue: $X million
Revenue growth: X% CAGR
Gross margin: X%
EBIT margin: X%
D&A % of revenue: X%
CapEx % of revenue: X%
FCF margin: X%
```

### 第 3 步：构建收入预测

**方法论：**
1. 从最新实际收入（LTM 或最近财年）开始
2. 对每个预测年份应用增长率
3. 同时显示美元金额和计算的增长百分比

**增长率框架：**
- 第 1-2 年：较高增长，反映近期可见性
- 第 3-4 年：逐步向行业平均水平收敛
- 第 5 年及以后：接近终端增长率

**公式结构：**
- 收入（第 N 年）= 收入（第 N-1 年）×（1 + 增长率）
- 增长%（第 N 年）= 收入（第 N 年）/ 收入（第 N-1 年）- 1

**三情景方法：**
```
Bear Case: Conservative growth (e.g., 8-12%)
Base Case: Most likely scenario (e.g., 12-16%)
Bull Case: Optimistic growth (e.g., 16-20%)
```

### 第 4 步：运营费用建模

**固定/可变成本分析：**

运营费用应模拟真实的运营杠杆：
- **销售与营销**：通常占收入的 15-40%，取决于商业模式
- **研究与开发**：科技公司通常占 10-30%
- **一般与行政**：通常占收入的 8-15%，随公司规模扩大显示杠杆效应

**关键原则：**
- 所有百分比基于收入，而非毛利润
- 模拟运营杠杆：随收入增长，百分比应下降
- 保持 S&M、R&D、G&A 的独立行项目
- 计算 EBIT = 毛利润 - 总运营费用

**利润率扩张框架：**
```
Current State → Target State (Year 5)
Gross Margin: X% → Y% (justify based on scale, efficiency)
EBIT Margin: X% → Y% (result of revenue growth + opex leverage)
```

### 第 5 步：自由现金流计算

**按正确顺序构建 FCF：**

```
EBIT
(-) Taxes (EBIT × Tax Rate)
= NOPAT (Net Operating Profit After Tax)
(+) D&A (non-cash expense, % of revenue)
(-) CapEx (% of revenue, typically 4-8%)
(-) Δ NWC (change in working capital)
= Unlevered Free Cash Flow
```

**营运资金建模：**
- 计算为收入变化的百分比（收入增量）
- 典型范围：收入变化的 -2% 至 +2%
- 负数 = 现金来源（营运资金释放）
- 正数 = 现金使用（营运资金积累）

**维护性与增长性资本支出：**
- 维护性资本支出：维持当前运营（约占收入 2-3%）
- 增长性资本支出：支持扩张（额外占收入 2-5%）
- 总资本支出应与公司增长战略一致

### 第 6 步：资本成本（WACC）研究

**股权成本的 CAPM 方法论：**

```
Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium

Where:
- Risk-Free Rate = Current 10-Year Treasury Yield
- Beta = 5-year monthly stock beta vs market index
- Equity Risk Premium = 5.0-6.0% (market standard)
```

**债务成本计算：**

```
After-Tax Cost of Debt = Pre-Tax Cost of Debt × (1 - Tax Rate)

Determine Pre-Tax Cost of Debt from:
- Credit rating (if available)
- Current yield on company bonds
- Interest expense / Total Debt from financials
```

**资本结构权重：**

```
Market Value Equity = Current Stock Price × Shares Outstanding
Net Debt = Total Debt - Cash & Equivalents
Enterprise Value = Market Cap + Net Debt

Equity Weight = Market Cap / Enterprise Value
Debt Weight = Net Debt / Enterprise Value

WACC = (Cost of Equity × Equity Weight) + (After-Tax Cost of Debt × Debt Weight)
```

**特殊情况：**
- **净现金头寸**：如果现金 > 债务，净债务为负
  - 债务权重可能为负
  - WACC 计算相应调整
- **无债务**：WACC = 股权成本

**典型 WACC 范围：**
- 大盘、稳定型：7-9%
- 成长型公司：9-12%
- 高增长/高风险：12-15%

### 第 7 步：折现率应用（5-10 年预测）

**年中惯例：**
- 假设现金流发生在年中
- 折现期：0.5、1.5、2.5、3.5、4.5 等
- 折现因子 = 1 / (1 + WACC)^期间

**现值计算：**
```
For each projection year:
PV of FCF = Unlevered FCF × Discount Factor

Example (Year 1):
FCF = $1,000
WACC = 10%
Period = 0.5
Discount Factor = 1 / (1.10)^0.5 = 0.9535
PV = $1,000 × 0.9535 = $954
```

**预测期选择：**
- **5 年**：大多数分析的标准
- **7-10 年**：具有较长跑道的高增长公司
- **3 年**：成熟、稳定的企业

### 第 8 步：终值计算

**永续增长法（首选）：**

```
Terminal FCF = Final Year FCF × (1 + Terminal Growth Rate)
Terminal Value = Terminal FCF / (WACC - Terminal Growth Rate)

Critical Constraint: Terminal Growth < WACC (otherwise infinite value)
```

**终端增长率选择：**
- 保守型：2.0-2.5%（GDP 增长率）
- 适中型：2.5-3.5%
- 激进型：3.5-5.0%（仅适用于市场领导者）

**不得超过**：无风险利率或长期 GDP 增长率

**退出倍数法（替代方案）：**
```
Terminal Value = Final Year EBITDA × Exit Multiple

Where Exit Multiple comes from:
- Industry comparable trading multiples
- Precedent transaction multiples
- Typical range: 8-15x EBITDA
```

**终值现值：**
```
PV of Terminal Value = Terminal Value / (1 + WACC)^Final Period

Where Final Period accounts for timing:
5-year model with mid-year convention: Period = 4.5
```

**终值合理性检验：**
- 应占企业价值的 50-70%
- 如果 >75%，模型可能过度依赖终端假设
- 如果 &lt;40%，检查终端假设是否过于保守

### 第 9 步：企业价值到股权价值桥接

**估值汇总结构：**

```
(+) Sum of PV of Projected FCFs = $X million
(+) PV of Terminal Value = $Y million
= Enterprise Value = $Z million

(-) Net Debt [or + Net Cash if negative] = $A million
= Equity Value = $B million

÷ Diluted Shares Outstanding = C million shares
= Implied Price per Share = $XX.XX

Current Stock Price = $YY.YY
Implied Return = (Implied Price / Current Price) - 1 = XX%
```

**关键调整：**
- **净债务 = 总债务 - 现金及等价物**
  - 如果为正：从 EV 中减去（降低股权价值）
  - 如果为负（净现金）：加到 EV 上（增加股权价值）
- **使用稀释股份数**：包括期权、RSU、可转换证券
- **其他调整**（如适用）：
  - 少数股东权益
  - 养老金负债
  - 经营租赁义务

**估值输出格式：**
```csv
Valuation Component,Amount ($M)
PV Explicit FCFs,X.X
PV Terminal Value,Y.Y
Enterprise Value,Z.Z
(-) Net Debt,A.A
Equity Value,B.B
,,
Shares Outstanding (M),C.C
Implied Price per Share,$XX.XX
Current Share Price,$YY.YY
Implied Upside/(Downside),+XX%
```

### 第 10 步：敏感性分析

在 DCF 工作表底部构建**三张敏感性表格**，显示估值如何随不同假设变化：

1. **WACC vs 终端增长**——显示企业价值对折现率和永续增长率的敏感性
2. **收入增长 vs EBIT 利润率**——显示顶线增长和运营杠杆的影响
3. **Beta vs 无风险利率**——显示对股权成本组成部分的敏感性

**实现方式**：这些是简单的二维网格（不是 Excel 的"数据表"功能），每个单元格中包含公式。每个单元格必须包含针对该特定假设组合的完整 DCF 重新计算。有关使用 openpyxl 以编程方式填充所有 75 个单元格的详细要求，请参阅关键约束部分。

&lt;correct_patterns>

本节包含构建 DCF 模型时应遵循的所有正确模式。

### 情景块选择模式——遵循此方法

**假设按每个情景的独立块组织：**

**关键结构——每个节标题三行：**

```csv
BEAR CASE ASSUMPTIONS (section header, merge cells across)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),12%,10%,9%,8%,7%
EBIT Margin (%),45%,44%,43%,42%,41%

BASE CASE ASSUMPTIONS (section header, merge cells across)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),16%,14%,12%,10%,9%
EBIT Margin (%),48%,49%,50%,51%,52%

BULL CASE ASSUMPTIONS (section header, merge cells across)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),20%,18%,15%,13%,11%
EBIT Margin (%),50%,51%,52%,53%,54%
```

**每个情景块必须有一个列标题行**，在节标题正下方显示预测年份（FY2025E、FY2026E 等）。没有这一行，用户无法判断哪个假设值对应哪一年。

**如何引用假设——创建合并列：**
1. 情景选择单元格（例如 B6）包含 1=熊、2=基、3=牛
2. 使用 INDEX 或 OFFSET 公式创建合并列，从正确的情景块中提取数据
3. 预测公式引用合并列（干净的单元格引用）
4. 每个情景块包含跨预测年份的完整 DCF 假设集

**推荐的合并列模式（使用 INDEX）：**
`=INDEX(B10:D10, 1, $B$6)`

**不要这样做——在整个模型中散布 IF 语句：**
`=IF($B$6=1,[Bear block cell],IF($B$6=2,[Base block cell],[Bull block cell]))`

合并列方法集中了逻辑，使模型更易于审计。

### 正确的收入预测模式

**使用 INDEX 公式创建合并列，然后在预测中引用它：**

**第 1 步——FY1 增长的合并列：**
`=INDEX([Bear FY1 growth]:[Bull FY1 growth], 1, $B$6)`

**第 2 步——收入预测引用合并列：**
`Revenue Year 1: =D29*(1+$E$10)`

其中：
- D29 = 上一年收入
- $E$10 = FY1 增长的合并列单元格（包含 INDEX 公式）
- $B$6 = 情景选择器（1=熊、2=基、3=牛）

**这种方法比在每个预测公式中嵌入 IF 语句更简洁**，并且更容易审计正在使用哪些情景假设。

### 正确的 FCF 公式模式

**使用带有 INDEX 公式的合并列，然后在 FCF 计算中引用它们：**

**合并列方法：**
```csv
Item,Formula,Reference
D&A,=E29*$E$21,$E$21 = consolidation column for D&A %
CapEx,=E29*$E$22,$E$22 = consolidation column for CapEx %
Δ NWC,=(E29-D29)*$E$23,$E$23 = consolidation column for NWC %
Unlevered FCF,=E57+E58-E60-E62,E57=NOPAT E58=D&A E60=CapEx E62=Δ NWC
```

**每个合并列单元格包含一个 INDEX 公式**，根据情景选择器从适当的情景块中提取数据。这使预测公式保持简洁且可审计。

写公式前，确认情景块行位置并设置合并列。

### 正确的单元格注释格式

**每个硬编码值需要此格式：**

"Source: [System/Document], [Date], [Reference], [URL if applicable]"

**示例：**
```csv
Item,Source Comment
Stock price,Source: Market data script 2025-10-12 Close price
Shares outstanding,Source: 10-K FY2024 Page 45 Note 12
Historical revenue,Source: 10-K FY2024 Page 32 Consolidated Statements
Beta,Source: Market data script 2025-10-12 5-year monthly beta
Consensus estimates,Source: Management guidance Q3 2024 earnings call
```

### 正确的假设表格结构

**关键：每个情景块需要三个结构元素：**

1. **节标题行**（合并单元格）：例如"BEAR CASE ASSUMPTIONS"
2. **列标题行**，显示年份——此行为必填项，不得跳过
3. **数据行**，包含假设值

**结构：**
```csv
BEAR CASE ASSUMPTIONS (section header - merge across columns A:G)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),X%,X%,X%,X%,X%
EBIT Margin (%),X%,X%,X%,X%,X%
Terminal Growth,X%,,,,
WACC,X%,,,,

BASE CASE ASSUMPTIONS (section header - merge across columns A:G)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),X%,X%,X%,X%,X%
EBIT Margin (%),X%,X%,X%,X%,X%
Terminal Growth,X%,,,,
WACC,X%,,,,

BULL CASE ASSUMPTIONS (section header - merge across columns A:G)
Assumption,FY1,FY2,FY3,FY4,FY5
Revenue Growth (%),X%,X%,X%,X%,X%
EBIT Margin (%),X%,X%,X%,X%,X%
Terminal Growth,X%,,,,
WACC,X%,,,,
```

**如果没有显示预测年份（FY2025E、FY2026E 等）的列标题行，用户无法判断哪个假设值对应哪一年。此行为必填项。**

**然后创建合并列**（通常在右侧的下一列），使用 INDEX 公式根据情景选择器从所选情景块中提取数据。这个合并列就是你的预测公式所引用的内容。

### 正确的行规划流程

**1. 首先写所有标题和标签：**
```csv
Row,Content
1,[Company Name] DCF Model
2,Ticker | Date | Year End
4,Case Selector
7,KEY ASSUMPTIONS
26,Assumption headers
27-31,Growth assumptions
...,...
```

**2. 写所有节分隔符和空行**

**3. 然后使用锁定的行位置写公式**

**4. 创建后立即测试公式**

**把它想象成建筑施工：**
- 好的做法：先浇地基，再建墙（结构稳固）
- 坏的做法：先建墙，再浇地基（墙会倒塌）

**Excel 版本：**
- 好的做法：先添加标题，再写公式（公式稳定）
- 坏的做法：先写公式，再添加标题（公式会断裂）

### 正确的敏感性表格实现

**重要**：这些不是 Excel 的"数据表"功能。这些是简单的网格，你使用 openpyxl 在其中写入常规公式。是的，这意味着总共约 75 个公式（3 张表 × 每张 25 个单元格），但这是直接且必须的。

**使用公式以编程方式填充：**

每张敏感性表格必须完全填充公式，为每种假设组合重新计算隐含每股价格。**不要使用 Excel 的数据表功能**（它需要手动干预，无法通过 openpyxl 自动化）。

**实现方法——具体示例：**

**表格结构——5×5 网格（奇数维度，基准情景居中）：**

如果模型的基准 WACC = 9.0%，基准终端增长 = 3.0%，则围绕这些值对称构建轴：

```csv
WACC vs Terminal Growth,  2.0%,  2.5%,  3.0%,  3.5%,  4.0%
              8.0%,       [fml], [fml], [fml], [fml], [fml]
              8.5%,       [fml], [fml], [fml], [fml], [fml]
              9.0%,       [fml], [fml], [★  ], [fml], [fml]   ← middle row = base WACC
              9.5%,       [fml], [fml], [fml], [fml], [fml]
             10.0%,       [fml], [fml], [fml], [fml], [fml]
                                   ↑
                          middle col = base terminal g
```

**★ = 中心单元格。** 其公式输出必须等于模型的实际隐含每股价格（来自估值汇总）。对该单元格应用中蓝色填充（`#BDD7EE`）和粗体字体，以便基准情景在视觉上有明确锚点。

**轴值规则：** `axis_values = [base - 2*step, base - step, base, base + step, base + 2*step]`——围绕基准对称，奇数个数保证有中心。

**公式模式——单元格 B88（WACC=8.0%，终端增长=2.0%）：**

B88 中的公式应使用以下内容重新计算隐含价格：
- 来自行标题的 WACC：`$A88`（8.0%）
- 来自列标题的终端增长：`B$87`（2.0%）

**推荐方法：** 引用主 DCF 计算，但替换这些值。

**示例公式结构：**
`=([SUM of PV FCFs using $A88 as discount rate] + [Terminal Value using B$87 as growth rate and $A88 as WACC] - [Net Debt]) / [Shares]`

**关键——为 5x5 网格中的每个单元格写公式（每张表 25 个单元格，共 75 个单元格）。** 使用 openpyxl 在循环中以编程方式写入这些公式。不要跳过此步骤或留下占位文本。

**Python 实现模式：**
```python
# Pseudocode for populating sensitivity table
for row_idx, wacc_value in enumerate(wacc_range):
    for col_idx, term_growth_value in enumerate(term_growth_range):
        # Build formula that uses wacc_value and term_growth_value
        formula = f"=<DCF recalc using {wacc_value} and {term_growth_value}>"
        ws.cell(row=start_row+row_idx, column=start_col+col_idx).value = formula
```

**敏感性表格在模型打开时必须立即可用，无需用户进行任何手动步骤。**

&lt;/correct_patterns>

&lt;common_mistakes>

本节包含构建 DCF 模型时应避免的所有错误模式。

### 错误：简化的敏感性表格近似或占位文本

**不要使用线性近似：**

```
// WRONG - Linear approximation
B97: =B88*(1+(0.096-0.116))    // Assumes linear relationship

// WRONG - Division shortcut
B105: =B88/(1+(E48-0.07))      // Doesn't recalculate full DCF
```

**不要留下占位文本：**
```
// WRONG - Placeholder note
"Note: Use Excel Data Table feature (Data → What-If Analysis → Data Table) to populate sensitivity tables."

// WRONG - Empty cells
[leaving cells blank because "this is complex"]
```

**不要混淆术语：**
- ❌ "敏感性表格需要 Excel 的数据表功能"（错误——那是一个我们无法使用的特定 Excel 工具）
- ✅ "敏感性表格是每个单元格中包含公式的简单网格"（正确——这就是我们构建的内容）

**这些捷径为何错误：**
- 线性近似公式实际上并不重新计算 DCF——它们只是应用简单的数学调整
- 这些关系不是线性的，因此结果将不准确
- 占位文本需要用户手动干预
- 交付时模型无法立即使用
- 不专业，不适合客户
- 空单元格 = 不完整的交付物

**应拒绝的常见合理化理由：**
"写 75+ 个公式感觉很复杂，所以我会留一个注释让用户手动完成。"

**现实：** 当你在 Python 中使用 openpyxl 循环时，写 75 个公式是直接的。每个公式遵循相同的模式——只需替换行/列值。这是交付物的必要部分。

**正确做法：** 用重新计算该特定假设组合完整 DCF 的公式填充每个敏感性单元格

### 错误：缺少单元格注释

**不要这样做：**
- 创建所有硬编码输入而不添加注释
- 认为"我稍后会添加"
- 写"TODO: add source"
- 留下没有文档的蓝色输入

**为何错误：**
- 无法验证数据来源
- 不符合 xlsx skill 要求
- 不适合审计
- 事后修复浪费时间

**正确做法：** 在创建每个硬编码值时添加单元格注释

### 错误：公式行引用偏移

**症状：**
FCF 部分引用了错误的假设行：
`D&A:  =E29*$E$34    // Should be $E$21, but referencing wrong row`
`CapEx: =E29*$E$41   // Should be $E$22, but row shifted`

**发生原因：**
1. 先写公式
2. 然后插入标题
3. 所有行引用偏移
4. 现在公式指向错误的单元格 → #REF! 错误

**正确做法：** 先锁定行布局，然后写公式

### 错误：每个情景中每个假设使用单行

**不要这样构建假设：**
```csv
Assumption,Bear,Base,Bull
Revenue Growth FY1,10%,13%,16%
Revenue Growth FY2,9%,12%,15%
```
这种垂直布局使得难以看到每个情景内各年份的进展。

**为何错误：**
- 难以看到每个情景内假设跨年份的演变
- 难以比较整个预测期内各情景的假设
- 对于审查情景逻辑不够直观

**正确做法：**
- 为每个情景（熊、基、牛）创建独立块
- 在每个块内，横向展示跨预测年份的假设
- 这使每个情景的假设作为一个整体更易于审查

### 错误：无边框

**不要交付没有边框的模型：**
- 无节分隔
- 所有单元格混在一起
- 难以阅读且不专业

**为何错误：**
- 不适合客户
- 难以导航
- 看起来业余

**正确做法：** 在所有主要节周围添加边框

### 错误：错误的字体颜色或无字体颜色区分

**不要这样做：**
- 所有文本为黑色
- 只使用填充颜色（不更改字体颜色）
- 混淆哪些单元格是蓝色还是黑色

**为何错误：**
- 无法区分输入和公式
- 审计变得不可能
- 违反 xlsx skill 要求

**正确做法：** 所有硬编码输入使用蓝色文本，所有公式使用黑色文本，工作表链接使用绿色

### 错误：运营费用基于毛利润

**不要这样做：**
`S&M: =E33*0.15    // E33 = Gross Profit (WRONG)`

**为何错误：**
- 运营费用随收入而非毛利润扩展
- 产生不切实际的利润率进展
- 不是企业实际运营方式

**正确做法：**
`S&M: =E29*0.15    // E29 = Revenue (CORRECT)`

### 前 5 大错误汇总

1. **公式行引用偏移** → 在写公式之前定义所有行位置
2. **缺少单元格注释** → 在创建单元格时添加注释，而非最后
3. **简化的敏感性表格** → 用完整 DCF 重新计算公式填充所有单元格，而非近似值
4. **情景块引用错误** → 确保 IF 公式从正确的熊/基/牛块中提取
5. **无边框** → 添加专业节边框以达到客户级外观

此外，请注意以下错误：

### WACC 计算错误
- 在资本结构中混用账面价值和市场价值
- 错误地使用股权 beta 而非资产/去杠杆 beta
- 对债务成本应用错误的税率
- 错误的无风险利率（必须使用当前 10 年期国债收益率）
- 未针对净债务与净现金头寸进行调整

### 增长假设缺陷
- 终端增长 > WACC（产生无限价值）
- 预测增长率与历史表现不一致
- 忽视行业增长约束
- 收入增长与单位经济学不一致
- 利润率扩张缺乏运营依据

### 终值错误
- 使用错误的增长方法（永续增长法 vs 退出倍数法）
- 终值 >80% 的企业价值（表明过度依赖终端假设）
- 终端利润率与稳态假设不一致
- 终值的折现期错误

### 现金流预测错误
- 运营费用基于毛利润而非收入
- D&A/资本支出百分比与商业模式不一致
- 营运资金变化计算不正确
- 各年税率不一致
- NOPAT 计算错误

**这些是最常见的错误。在开始任何 DCF 构建之前重新阅读本节。**

&lt;/common_mistakes>

## Excel 文件创建

**本 skill 使用 `xlsx` skill 进行所有电子表格操作。** xlsx skill 提供：
- 标准化公式构建规则
- 数字格式约定
- 通过 `recalc.py` 脚本自动重新计算公式
- 全面的错误检查和验证

本 skill 创建的所有 Excel 文件必须遵循 xlsx skill 要求，包括零公式错误和正确的重新计算。

## 质量评估标准

每个 DCF 模型必须在以下方面最大化：
1. **基于历史表现的真实收入和利润率假设**
2. **使用正确 CAPM 方法论的适当资本成本计算**
3. **显示估值范围的全面敏感性分析**
4. **清晰的终值计算及支持依据**
5. **支持情景分析的专业模型结构**
6. **所有关键假设的透明文档**

## 输入要求

### 最低必需输入
1. **公司标识符**：股票代码或公司名称
2. **增长假设**：预测期的收入增长率（或"使用共识预测"）
3. **可选参数**：
   - 预测期（默认：5 年）
   - 情景案例（熊/基/牛增长和利润率假设）
   - 终端增长率（默认：2.5-3.0%）
   - 如果不使用 CAPM，则提供特定的 WACC 输入

## Excel 模型结构

### 工作表架构

创建**两个工作表**：

1. **DCF** - 主估值模型，底部包含敏感性分析
2. **WACC** - 资本成本计算

**关键**：敏感性表格放在 DCF 工作表底部（不在单独的工作表上）。这将所有估值输出保持在一起。

### 公式重新计算（必须执行）

创建或修改 Excel 模型后，使用 `excel-author` skill 中的 `recalc.py` 脚本**重新计算所有公式**：

```bash
python recalc.py [path_to_excel_file] [timeout_seconds]
```

示例：
```bash
python recalc.py AAPL_DCF_Model_2025-10-12.xlsx 30
```

该脚本将：
- 使用 LibreOffice 重新计算所有工作表中的所有公式
- 扫描所有单元格中的 Excel 错误（#REF!、#DIV/0!、#VALUE!、#NAME?、#NULL!、#NUM!、#N/A）
- 返回包含错误位置和计数的详细 JSON

**预期输出格式：**
```json
{
  "status": "success",           // or "errors_found"
  "total_errors": 0,              // Total error count
  "total_formulas": 42,           // Number of formulas in file
  "error_summary": {}             // Only present if errors found
}
```

**如果发现错误**，输出将包含详细信息：
```json
{
  "status": "errors_found",
  "total_errors": 2,
  "total_formulas": 42,
  "error_summary": {
    "#REF!": {
      "count": 2,
      "locations": ["DCF!B25", "DCF!C25"]
    }
  }
}
```

**修复所有错误**并重新运行 recalc.py，直到状态为"success"，然后再交付模型。

### 格式标准

**重要**：遵循 xlsx skill 的公式构建规则和数字格式约定。DCF skill 添加了特定的视觉呈现标准。

**配色方案——两层：**

**第 1 层：字体颜色（xlsx skill 的必要要求）**
- **蓝色文本（RGB: 0,0,255）**：所有硬编码输入（股价、股份数、历史数据、假设）
- **黑色文本（RGB: 0,0,0）**：所有公式和计算
- **绿色文本（RGB: 0,128,0）**：链接到其他工作表（WACC 工作表引用）

**第 2 层：填充颜色——专业蓝/灰调色板（除非用户另有指定，否则为默认值）**
- **保持简洁**——仅使用蓝色和灰色填充。不要引入绿色、黄色、橙色或多种强调色。颜色过多的模型看起来业余。
- **默认填充调色板：**
  - **节标题**：深蓝色（RGB: 31,78,121 / `#1F4E79`）背景，白色粗体文本
  - **子标题/列标题**：浅蓝色（RGB: 217,225,242 / `#D9E1F2`）背景，黑色粗体文本
  - **输入单元格**：浅灰色（RGB: 242,242,242 / `#F2F2F2`）背景，蓝色字体——或者如果想要最大简洁性，白色背景配蓝色字体
  - **计算单元格**：白色背景，黑色字体
  - **输出/汇总行**（每股价值、EV 等）：中蓝色（RGB: 189,215,238 / `#BDD7EE`）背景，黑色粗体字体
- **就这些——3 种蓝色 + 1 种灰色 + 白色。** 抵制添加更多颜色的冲动。
- 用户提供的模板或明确的颜色偏好始终覆盖这些默认值。

**两层如何协同工作：**
- 输入单元格：蓝色字体 + 浅灰色填充 = "硬编码输入"
- 公式单元格：黑色字体 + 白色背景 = "计算值"
- 工作表链接：绿色字体 + 白色背景 = "来自另一工作表的引用"
- 关键输出：黑色粗体字体 + 中蓝色填充 = "这是答案"

**字体颜色告诉你它是什么（输入/公式/链接）。填充颜色告诉你你在哪里（标题/数据/输出）。**

### 边框标准（专业外观的必要要求）

**粗边框**（1.5pt）围绕主要节：
- 关键输入节
- 预测假设节
- 5 年现金流预测节
- 终值节
- 估值汇总节
- 每张敏感性分析表

**中等边框**（1pt）在子节之间：
- 公司详情 vs 历史表现
- 增长假设 vs EBIT 利润率 vs FCF 参数

**细边框**（0.5pt）围绕数据表：
- 情景假设表（熊 | 基 | 牛 | 已选）
- 历史 vs 预测财务矩阵

**无边框：** 表格内的单个单元格（保持简洁、可扫描）

**边框为必要要求**——没有专业边框的模型不适合客户。

**数字格式**（遵循 xlsx skill 标准）：
- **年份**：格式化为文本字符串（例如"2024"而非"2,024"）
- **百分比**：`0.0%`（一位小数）
- **货币**：百万单位用 `$#,##0`；每股用 `$#,##0.00`——始终在标题中指定单位（"Revenue ($mm)"）
- **零值**：使用数字格式将所有零显示为"-"（例如 `$#,##0;($#,##0);-`）
- **大数字**：带千位分隔符的 `#,##0`
- **负数**：用括号表示 `(#,##0)`（不用负号）

**单元格注释（所有硬编码输入的必要要求）**：

根据 xlsx skill，所有硬编码值必须有记录来源的单元格注释。格式："Source: [System/Document], [Date], [Reference], [URL if applicable]"

**关键**：在创建单元格时添加注释。不要推迟到最后。

### DCF 工作表详细结构

**第 1 节：标题**
```csv
Row,Content
1,[Company Name] DCF Model
2,Ticker: [XXX] | Date: [Date] | Year End: [FYE]
3,Blank
4,Case Selector Cell (1=Bear 2=Base 3=Bull)
5,Case Name Display (formula: =IF([Selector]=1"Bear"IF([Selector]=2"Base""Bull")))
```

**第 2 节：市场数据（不依赖情景）**
```csv
Item,Value
Current Stock Price,$XX.XX
Shares Outstanding (M),XX.X
Market Cap ($M),[Formula]
Net Debt ($M),XXX [or Net Cash if negative]
```

**第 3 节：DCF 情景假设**

为每个情景（熊、基、牛）创建独立的假设块，DCF 特定假设（收入增长%、EBIT 利润率%、税率%、D&A 占收入%、资本支出占收入%、NWC 变化占 ΔRev%、终端增长率、WACC）横向排列在各预测年份。每个块必须包含节标题、显示预测年份（FY1、FY2 等）的列标题行和数据行。有关确切布局，请参阅 `<correct_patterns>` 节中的"正确的假设表格结构"。

**第 4 节：历史与预测财务数据**

**引用合并列（例如"Selected Case"），从情景块中提取数据**，而非在每个预测行中散布 IF 公式。

```csv
Income Statement ($M),2020A,2021A,2022A,2023A,2024E,2025E,2026E
Revenue,XXX,XXX,XXX,XXX,[=E29*(1+$E$10)],[=F29*(1+$E$11)],[=G29*(1+$E$12)]
  % growth,XX%,XX%,XX%,XX%,[=E29/D29-1],[=F29/E29-1],[=G29/F29-1]
,,,,,,
Gross Profit,XXX,XXX,XXX,XXX,[=E29*E33],[=F29*F33],[=G29*G33]
  % margin,XX%,XX%,XX%,XX%,[=E33/E29],[=F33/F29],[=G33/G29]
,,,,,,
Operating Expenses:,,,,,,,
  S&M,XXX,XXX,XXX,XXX,[=E29*0.15],[=F29*0.14],[=G29*0.13]
  R&D,XXX,XXX,XXX,XXX,[=E29*0.12],[=F29*0.11],[=G29*0.10]
  G&A,XXX,XXX,XXX,XXX,[=E29*0.08],[=F29*0.07],[=G29*0.07]
  Total OpEx,XXX,XXX,XXX,XXX,[=E36+E37+E38],[=F36+F37+F38],[=G36+G37+G38]
,,,,,,
EBIT,XXX,XXX,XXX,XXX,[=E33-E39],[=F33-F39],[=G33-G39]
  % margin,XX%,XX%,XX%,XX%,[=E41/E29],[=F41/F29],[=G41/G29]
,,,,,,
Taxes,(XX),(XX),(XX),(XX),[=E41*$E$24],[=F41*$E$24],[=G41*$E$24]
  Tax rate,XX%,XX%,XX%,XX%,[=E43/E41],[=F43/F41],[=G43/G41]
,,,,,,
NOPAT,XXX,XXX,XXX,XXX,[=E41-E43],[=F41-F43],[=G41-G43]
```

**关键公式模式**：
- 收入增长：`=E29*(1+$E$10)`，其中 $E$10 是第 1 年增长的合并列
- 不要：`=E29*(1+IF($B$6=1,$B$10,IF($B$6=2,$C$10,$D$10)))`

这种方法更简洁、更易于审计，并通过集中情景逻辑防止公式错误。

**第 5 节：自由现金流构建**

**关键**：验证行引用指向正确的假设行。创建后立即测试公式。

```csv
Cash Flow ($M),2020A,2021A,2022A,2023A,2024E,2025E,2026E
NOPAT,XXX,XXX,XXX,XXX,[=E45],[=F45],[=G45]
(+) D&A,XXX,XXX,XXX,XXX,[=E29*$E$21],[=F29*$E$21],[=G29*$E$21]
    % of Rev,XX%,XX%,XX%,XX%,[=E58/E29],[=F58/F29],[=G58/G29]
(-) CapEx,(XX),(XX),(XX),(XX),[=E29*$E$22],[=F29*$E$22],[=G29*$E$22]
    % of Rev,XX%,XX%,XX%,XX%,[=E60/E29],[=F60/F29],[=G60/G29]
(-) Δ NWC,(XX),(XX),(XX),(XX),[=(E29-D29)*$E$23],[=(F29-E29)*$E$23],[=(G29-F29)*$E$23]
    % of Δ Rev,XX%,XX%,XX%,XX%,[=E62/(E29-D29)],[=F62/(F29-E29)],[=G62/(G29-F29)]
,,,,,,
Unlevered FCF,XXX,XXX,XXX,XXX,[=E57+E58-E60-E62],[=F57+F58-F60-F62],[=G57+G58-G60-G62]
```

**行引用示例**（基于布局规划）：
- $E$21 = D&A % 假设（合并列，第 21 行）
- $E$22 = 资本支出 % 假设（合并列，第 22 行）
- $E$23 = NWC % 假设（合并列，第 23 行）
- E29 = 该年收入（第 29 行）
- E45 = 该年 NOPAT（第 45 行）

**写公式前**：确认这些行号与实际布局匹配。测试一列，然后横向复制。

**第 6 节：折现与估值**
```csv
DCF Valuation,2024E,2025E,2026E,2027E,2028E,Terminal
Unlevered FCF ($M),XXX,XXX,XXX,XXX,XXX,
Period,0.5,1.5,2.5,3.5,4.5,
Discount Factor,0.XX,0.XX,0.XX,0.XX,0.XX,
PV of FCF ($M),XXX,XXX,XXX,XXX,XXX,
,,,,,,
Terminal FCF ($M),,,,,,,XXX
Terminal Value ($M),,,,,,,XXX
PV Terminal Value ($M),,,,,,,XXX
,,,,,,
Valuation Summary ($M),,,,,,
Sum of PV FCFs,XXX,,,,,
PV Terminal Value,XXX,,,,,
Enterprise Value,XXX,,,,,
(-) Net Debt,(XX),,,,,
Equity Value,XXX,,,,,
,,,,,,
Shares Outstanding (M),XX.X,,,,,
IMPLIED PRICE PER SHARE,$XX.XX,,,,,
Current Stock Price,$XX.XX,,,,,
Implied Upside/(Downside),XX%,,,,,
```

### WACC 工作表结构

```csv
COST OF EQUITY CALCULATION,,
Risk-Free Rate (10Y Treasury),X.XX%,[Yellow input]
Beta (5Y monthly),X.XX,[Yellow input]
Equity Risk Premium,X.XX%,[Yellow input]
Cost of Equity,X.XX%,[Calculated blue]
,,
COST OF DEBT CALCULATION,,
Credit Rating,AA-,[Yellow input]
Pre-Tax Cost of Debt,X.XX%,[Yellow input]
Tax Rate,XX.X%,[Link to DCF sheet]
After-Tax Cost of Debt,X.XX%,[Calculated blue]
,,
CAPITAL STRUCTURE,,
Current Stock Price,$XX.XX,[Link to DCF]
Shares Outstanding (M),XX.X,[Link to DCF]
Market Capitalization ($M),"X,XXX",[Calculated]
,,
Total Debt ($M),XXX,[Yellow input]
Cash & Equivalents ($M),XXX,[Yellow input]
Net Debt ($M),XXX,[Calculated]
,,
Enterprise Value ($M),"X,XXX",[Calculated]
,,
WACC CALCULATION,Weight,Cost,Contribution
Equity,XX.X%,X.X%,X.XX%
Debt,XX.X%,X.X%,X.XX%
,,
WEIGHTED AVERAGE COST OF CAPITAL,X.XX%,[Green output]
```

**关键 WACC 公式：**
```
Market Cap = Price × Shares
Net Debt = Total Debt - Cash
Enterprise Value = Market Cap + Net Debt
Equity Weight = Market Cap / EV
Debt Weight = Net Debt / EV
WACC = (Cost of Equity × Equity Weight) + (After-tax Cost of Debt × Debt Weight)
```

### 敏感性分析（DCF 工作表底部）

**术语提醒**："敏感性表格"= 带有行标题、列标题和每个数据单元格中公式的简单二维网格。不是 Excel 的"数据表"功能（数据 → 假设分析 → 数据表）。你将使用 openpyxl 将常规 Excel 公式写入每个单元格。

**位置**：DCF 工作表第 87 行及以后（不在单独的工作表上）

**三张敏感性表格，垂直堆叠：**

1. **WACC vs 终端增长**（第 87-100 行）——5x5 网格 = 25 个带公式的单元格
2. **收入增长 vs EBIT 利润率**（第 102-115 行）——5x5 网格 = 25 个带公式的单元格
3. **Beta vs 无风险利率**（第 117-130 行）——5x5 网格 = 25 个带公式的单元格

**需要写入的公式总数：75**（这是必要的，不是可选的）

**关键**：所有敏感性表格单元格必须使用 openpyxl 以编程方式填充公式。不要使用线性近似捷径。不要留下占位文本或关于手动步骤的注释。不要因为"太复杂"而合理化留空单元格——使用 Python 循环生成公式。

**表格设置：**
1. 创建带有行/列标题（要测试的假设值）的表格结构
2. 用公式填充每个数据单元格，该公式：
   - 使用行标题值（例如 WACC = 9.0%）
   - 使用列标题值（例如终端增长 = 3.0%）
   - 使用这些特定假设重新计算完整的 DCF
   - 返回该情景的隐含每股价格
3. 交付时所有单元格必须包含有效公式
4. 使用条件格式设置单元格：较高值用绿色刻度，较低值用红色刻度
5. 将基准情景单元格加粗
6. 表格之间留 1-2 个空行

**无需手动干预**——用户打开文件时敏感性表格必须完全可用。

## 情景选择器实现

**三情景框架：**

### 熊市情景
- 保守的收入增长（历史范围的低端）
- 利润率压缩或无扩张
- 较高的 WACC（风险溢价增加）
- 较低的终端增长率
- 较高的资本支出假设

### 基准情景
- 共识或管理层指引的收入增长
- 基于运营杠杆的适度利润率扩张
- 当前市场隐含的 WACC
- 与 GDP 一致的终端增长（2.5-3.0%）
- 标准资本支出假设

### 牛市情景
- 乐观的收入增长（预测的高端）
- 显著的利润率扩张
- 较低的 WACC（降低风险溢价）
- 较高的终端增长（3.5-5.0%）
- 降低的资本支出强度

**公式实现：**

**不要在整个模型中散布嵌套 IF 公式。** 而是创建一个合并列，使用 INDEX 或 OFFSET 公式从适当的情景块中提取数据。

**推荐模式（使用 INDEX）：**
`=INDEX(B10:D10, 1, $B$6)`，其中 `B10:D10` = 熊/基/牛值，`1` = 行偏移，`$B$6` = 情景选择器单元格（1、2 或 3）

**然后在所有预测中引用合并列：**
`Revenue Year 1: =D29*(1+$E$10)`，其中 $E$10 是第 1 年增长的合并列值。

这种方法集中了情景逻辑，使模型更易于审计和维护。

## 交付物结构

**文件命名**：`[Ticker]_DCF_Model_[Date].xlsx`

**两个工作表**：
1. **DCF** - 完整模型，包含熊/基/牛情景 + 底部三张敏感性表格（WACC vs 终端增长、收入增长 vs EBIT 利润率、Beta vs 无风险利率）
2. **WACC** - 资本成本计算

**关键功能**：情景选择器（1/2/3）、带有 INDEX/OFFSET 公式的合并列、颜色编码单元格、所有输入的单元格注释、专业边框

## 最佳实践

### 模型构建
1. **增量构建**：完成每个节后再进入下一节
2. **边构建边测试**：输入样本数字以验证公式
3. **使用一致的结构**：类似的计算遵循类似的模式
4. **注释复杂公式**：为不寻常的计算添加注释
5. **内置检查**：在适用的地方添加求和检查和平衡检查

### 文档
1. **记录所有假设**：解释关键输入背后的依据
2. **引用数据来源**：注明每个数据点的来源
3. **解释方法论**：描述任何非标准方法
4. **标记不确定性**：突出显示可见度有限的领域

### 质量控制
1. **交叉核对计算**：以多种方式验证数学
2. **压力测试假设**：运行敏感性以确保模型稳健
3. **同行评审**：让他人检查公式
4. **版本控制**：随工作进展保存版本

## 常见变体

### 高增长科技公司
- 较长的预测期（7-10 年）
- 较高的初始增长率（20-30%）
- 随时间显著的利润率扩张
- 较高的 WACC（12-15%）
- 建模单位经济学（用户、ARPU 等）

### 成熟/稳定公司
- 较短的预测期（3-5 年）
- 适度的增长率（GDP +1-3%）
- 稳定的利润率
- 较低的 WACC（7-9%）
- 关注现金生成和资本配置

### 周期性公司
- 对整个经济周期建模
- 在周期中点正常化利润率
- 考虑低谷和峰值情景
- 针对周期性调整 beta

### 多业务段公司
- 为每个业务单元建立独立的 DCF
- 按业务段设置不同的增长率和利润率
- 分部加总估值
- 考虑协同效应

## 故障排除

**如果遇到错误或不合理的结果，请阅读 [TROUBLESHOOTING.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/finance/dcf-model/TROUBLESHOOTING.md) 获取详细的调试指导。**

## 工作流集成

### DCF 构建开始时

1. **收集市场数据**：
   - 检查可用的 MCP 服务器以获取当前市场数据
   - 使用网络搜索/抓取获取股价、beta 和其他市场指标
   - 如果需要特定数据，向用户请求

2. **收集历史财务数据**：
   - 检查可用的 MCP 服务器（Daloopa 等）
   - 如果无法通过 MCP 获取，向用户请求
   - 必要时从 10-K 手动提取

3. **使用本 skill 中详述的 DCF 方法论开始模型构建**

### 模型构建期间

1. **使用 openpyxl 构建 Excel 模型**，使用公式（而非硬编码值）
2. **遵循 xlsx skill 约定**进行公式构建和格式设置
3. **仅在用户请求或提供特定品牌指南时应用填充颜色**

### 交付模型前（必须执行）

1. **验证结构**：
   - 熊/基/牛情景块，假设横向排列在各预测年份
   - 情景选择器可用，公式引用正确的情景块
   - 敏感性表格在 DCF 工作表底部（不在单独工作表上）
   - 字体颜色：蓝色输入、黑色公式、绿色工作表链接
   - 所有硬编码输入的单元格注释
   - 主要节周围的专业边框

2. **重新计算公式**：运行 `python recalc.py model.xlsx 30`

3. **检查输出**：
   - 如果 `status` 为 `"success"` → 继续第 4 步
   - 如果 `status` 为 `"errors_found"` → 检查 `error_summary` 并阅读 [TROUBLESHOOTING.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/finance/dcf-model/TROUBLESHOOTING.md) 获取调试指导

4. **修复错误并重新运行 recalc.py**，直到状态为"success"

5. **抽查公式**：
   - 测试一个 FCF 公式——它是否引用了正确的假设行？
   - 更改情景选择器——合并列是否正确更新？
   - 验证收入公式引用合并列（而非嵌套 IF 公式）

6. **交付模型**

### 可用数据来源

- **MCP 服务器**：如已配置（Daloopa 用于历史财务数据）
- **网络搜索/抓取**：用于当前股价、beta 和市场数据
- **用户提供的数据**：历史财务数据、共识预测
- **手动提取**：SEC EDGAR 文件作为备用

## 最终输出检查清单

交付 DCF 模型前：

**必要项：**
- 运行 `python recalc.py model.xlsx 30` 直到状态为"success"（零公式错误）
- 两个工作表：DCF（底部含敏感性分析）、WACC
- 字体颜色：蓝色=输入，黑色=公式，绿色=工作表链接
- 所有硬编码输入的单元格注释
- 敏感性表格完全填充公式
- 主要节周围的专业边框

**验证：**
- 运营费用基于收入（而非毛利润）
- 终值占 EV 的 50-70%
- 终端增长 &lt; WACC
- 税率 21-28%
- 文件命名：`[Ticker]_DCF_Model_[Date].xlsx`

## 数据来源——MCP 优先，网络备用

以下许多段落提到"使用 S&P Kensho MCP / Daloopa MCP / FactSet MCP"。这些是原始 Cowork 插件上下文中的商业金融数据 MCP。在 Hermes 中：

- **如果你配置了任何结构化金融数据 MCP**（Hermes 支持 MCP——参见 `native-mcp` skill），优先使用它获取时间点可比数据、先例交易和文件。
- **否则**，回退到：
  - 针对 SEC EDGAR（`https://www.sec.gov/cgi-bin/browse-edgar`）使用 `web_search` / `web_extract` 获取美国文件
  - 公司 IR 页面获取新闻稿、财报演示文稿
  - `browser_navigate` 用于交互式数据门户
  - 用户提供的数据（当上下文中没有时明确询问）
- **绝不捏造数据**。如果某个倍数、先例或文件数字无法获取来源，将该单元格标记为 `[UNSOURCED]` 并向用户说明。

## 归属

本 skill 改编自 Anthropic 的 Claude 金融服务插件套件（Apache-2.0）。Office-JS / Cowork 实时 Excel 路径已被移除；此版本通过 `excel-author` skill 的约定面向无头 openpyxl。原始来源：https://github.com/anthropics/financial-services