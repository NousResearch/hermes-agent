---
title: "并购模型 — 在 Excel 中构建增厚/摊薄（并购）模型 — 备考损益表、协同效应、融资结构、每股收益影响"
sidebar_label: "Merger Model"
description: "在 Excel 中构建增厚/摊薄（并购）模型 — 备考损益表、协同效应、融资结构、每股收益影响"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Merger Model

在 Excel 中构建增厚/摊薄（并购）模型 — 备考损益表、协同效应、融资结构、每股收益影响。与 excel-author 配合使用。适用于并购提案、董事会材料或交易评估。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/finance/merger-model` 安装 |
| 路径 | `optional-skills/finance/merger-model` |
| 版本 | `1.0.0` |
| 作者 | Anthropic（由 Nous Research 改编） |
| 许可证 | Apache-2.0 |
| 平台 | linux, macos, windows |
| 标签 | `finance`, `m-and-a`, `merger`, `accretion-dilution`, `excel`, `openpyxl`, `modeling`, `investment-banking` |
| 相关 skill | [`excel-author`](/user-guide/skills/optional/finance/finance-excel-author), [`pptx-author`](/user-guide/skills/optional/finance/finance-pptx-author), [`dcf-model`](/user-guide/skills/optional/finance/finance-dcf-model), [`3-statement-model`](/user-guide/skills/optional/finance/finance-3-statement-model) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

## 环境

本 skill 假定使用**无界面 openpyxl** — 即在磁盘上生成 .xlsx 文件。
遵循 `excel-author` skill 关于单元格着色、公式、命名区域和敏感性表格的约定。
交付前重新计算：`python /path/to/excel-author/scripts/recalc.py ./out/model.xlsx`。

# Merger Model

为并购交易构建增厚/摊薄分析。对备考每股收益影响、协同效应敏感性及购买价格分配进行建模。适用于评估潜在收购、为提案准备并购影响分析，或就交易条款提供建议。

## 工作流程

### 第一步：收集输入数据

**收购方：**
- 公司名称、当前股价、流通股数
- LTM 和 NTM 每股收益（GAAP 及调整后）
- 市盈率倍数
- 税前债务成本、税率
- 资产负债表上的现金、现有债务

**目标方：**
- 公司名称、当前股价、流通股数（如为上市公司）
- LTM 和 NTM 每股收益或净利润
- 企业价值或股权价值

**交易条款：**
- 每股要约价格（或相对当前价格的溢价）
- 对价结构：现金比例 vs. 股票比例
- 为现金部分融资而新增的债务
- 预期协同效应（收入和成本）及分阶段时间表
- 交易费用和融资成本
- 预期交割日期

### 第二步：购买价格分析

| 项目 | 金额 |
|------|-------|
| 每股要约价格 | |
| 相对当前价格的溢价 | |
| 股权价值 | |
| 加：承接净债务 | |
| 企业价值 | |
| 隐含 EV / EBITDA | |
| 隐含市盈率 | |

### 第三步：资金来源与用途

| 来源 | $ | 用途 | $ |
|---------|---|------|---|
| 新增债务 | | 股权收购价格 | |
| 自有现金 | | 偿还目标方债务 | |
| 新发行股票 | | 交易费用 | |
| | | 融资费用 | |
| **合计** | | **合计** | |

### 第四步：备考每股收益（增厚/摊薄）

逐年计算（第 1-3 年）：

| | 独立口径 | 备考口径 | 增厚/（摊薄） |
|---|-----------|-----------|---------------------|
| 收购方净利润 | | | |
| 目标方净利润 | | | |
| 协同效应（税后） | | | |
| 动用现金的利息损失（税后） | | | |
| 新增债务利息（税后） | | | |
| 无形资产摊销（税后） | | | |
| 备考净利润 | | | |
| 备考股份数 | | | |
| **备考每股收益** | | | |
| **增厚/（摊薄）%** | | | |

### 第五步：敏感性分析

**增厚/摊薄 vs. 协同效应与要约溢价：**

| | 协同效应 $0M | 协同效应 $25M | 协同效应 $50M | 协同效应 $75M | 协同效应 $100M |
|---|---------|----------|----------|----------|-----------|
| 溢价 15% | | | | | |
| 溢价 20% | | | | | |
| 溢价 25% | | | | | |
| 溢价 30% | | | | | |

**增厚/摊薄 vs. 现金/股票对价结构：**

| | 100% 现金 | 75/25 | 50/50 | 25/75 | 100% 股票 |
|---|-----------|-------|-------|-------|------------|
| 第 1 年 | | | | | |
| 第 2 年 | | | | | |

### 第六步：盈亏平衡协同效应

计算交易在第 1 年实现每股收益中性所需的最低协同效应。

### 第七步：输出

- Excel 工作簿，包含：
  - 假设条件标签页
  - 资金来源与用途
  - 备考利润表
  - 增厚/摊薄汇总
  - 敏感性表格
  - 盈亏平衡分析
- 用于提案材料的单页并购影响摘要

## 重要说明

- 在相关情况下，始终同时展示 GAAP 和调整后（现金）每股收益
- 股票交易：使用收购方当前股价计算换股比例，并注明新发行股份带来的稀释效应
- 包含购买价格分配 — 商誉和无形资产摊销对 GAAP 每股收益至关重要
- 协同效应分阶段实现至关重要 — 第 1 年通常仅为运行率协同效应的 25%-50%
- 不要遗漏动用现金的利息损失收入及新增债务的利息支出
- 协同效应和利息调整的税率应与收购方的边际税率保持一致


## 数据来源 — 优先使用 MCP，其次使用网络

以下部分内容提及"使用 S&P Kensho MCP / Daloopa MCP / FactSet MCP"。这些是原 Cowork 插件场景中的商业金融数据 MCP。在 Hermes 中：

- **如已配置任何结构化金融数据 MCP**（Hermes 支持 MCP — 参见 `native-mcp` skill），优先用于时点可比数据、前例交易及文件。
- **否则**，回退至：
  - 针对 SEC EDGAR（`https://www.sec.gov/cgi-bin/browse-edgar`）使用 `web_search` / `web_extract` 获取美国文件
  - 公司投资者关系页面获取新闻稿、财报材料
  - 使用 `browser_navigate` 访问交互式数据门户
  - 用户提供的数据（当上下文中没有时，明确向用户询问）
- **严禁捏造数据**。如果某个倍数、前例交易或文件数字无法溯源，将该单元格标记为 `[UNSOURCED]` 并告知用户。

## 归属声明

本 skill 改编自 Anthropic 的 Claude for Financial Services 插件套件（Apache-2.0）。Office-JS / Cowork 实时 Excel 路径已移除；本版本通过 `excel-author` skill 的约定，面向无界面 openpyxl。原始来源：https://github.com/anthropics/financial-services