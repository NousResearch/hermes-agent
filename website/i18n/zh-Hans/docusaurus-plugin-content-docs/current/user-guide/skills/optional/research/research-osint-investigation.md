---
title: "Osint Investigation"
sidebar_label: "Osint Investigation"
description: "公开记录 OSINT 调查框架 — SEC EDGAR 文件、USAspending 合同、参议院游说、OFAC 制裁、ICIJ 离岸泄露、纽约市房产记录（ACRIS）、OpenCorporates 注册信息、CourtListener 法院记录、Wayback Machine 存档、Wikipedia + Wikidata、GDELT 新闻监控。跨来源实体解析、交叉链接分析、时序关联、证据链。仅使用 Python 标准库。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Osint Investigation

公开记录 OSINT（开源情报）调查框架 — SEC EDGAR 文件、USAspending 合同、参议院游说、OFAC 制裁、ICIJ 离岸泄露、纽约市房产记录（ACRIS）、OpenCorporates 注册信息、CourtListener 法院记录、Wayback Machine 存档、Wikipedia + Wikidata、GDELT 新闻监控。跨来源实体解析、交叉链接分析、时序关联、证据链。仅使用 Python 标准库。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/research/osint-investigation` 安装 |
| 路径 | `optional-skills/research/osint-investigation` |
| 版本 | `0.1.0` |
| 作者 | Hermes Agent（改编自 ShinMegamiBoson/OpenPlanter，MIT 许可）|
| 平台 | linux, macos, windows |
| 标签 | `osint`, `investigation`, `public-records`, `sec`, `sanctions`, `corporate-registry`, `property`, `courts`, `due-diligence`, `journalism` |
| 相关 skill | [`domain-intel`](/user-guide/skills/optional/research/research-domain-intel), [`arxiv`](/user-guide/skills/bundled/research/research-arxiv) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# OSINT 调查 — 公开记录交叉核查

公开记录 OSINT 调查框架：政府合同、企业文件、游说、制裁、离岸泄露、房产记录、法院记录、网络存档、知识库及全球新闻。跨异构来源解析实体，以显式置信度构建交叉链接，运行统计时序检验，并生成结构化证据链。

**仅使用 Python 标准库。** 零安装。支持 Linux、macOS、Windows。大多数来源无需 API 密钥（OpenCorporates 有可选的免费 token，可提高速率限制）。

改编自 MIT 许可的 ShinMegamiBoson/OpenPlanter 项目；扩展覆盖了原项目未涉及的身份/房产/诉讼/存档/新闻来源。

## 何时使用此 skill

当用户请求以下内容时使用：

- "追踪资金流向" — 政府合同、游说 → 立法、制裁
- 企业尽职调查 — 谁控制公司 X、在哪里注册、谁担任董事会成员、提交了哪些文件
- 制裁筛查 — 实体 X 是否在 OFAC SDN 名单或 ICIJ 离岸泄露中
- 权钱交易调查 — 有离岸关联的承包商、赢得合同的游说客户
- 房产所有权 — 按姓名或地址查找已记录的契约/抵押（纽约市；其他县请用户查阅相关记录机构）
- 诉讼历史 — 查找联邦及州法院意见和 PACER 案卷
- 跨来源实体解析（命名存在差异，如 LLC 后缀、缩写）
- 以显式置信度构建证据链
- "关于 X 有哪些报道" — 国际新闻（GDELT）+ Wikipedia 叙述 + Wayback Machine 恢复失效 URL

**不适用**此 skill 的场景：

- 通用网络研究 → `web_search` / `web_extract`
- 域名/基础设施 OSINT → `domain-intel` skill
- 学术文献 → `arxiv` skill
- 社交媒体账号发现 → `sherlock` skill（可选）
- 美国**联邦**竞选财务 — FEC 在此处有意不覆盖（免费 DEMO_KEY 层级的 API 对临时贡献者姓名查询不可靠）。联邦捐款请直接引导用户访问 https://www.fec.gov/data/。

## 工作流程

Agent 通过 `terminal` 工具运行脚本。`SKILL_DIR` 是存放此 SKILL.md 的目录。

### 1. 确定适用的数据来源

阅读数据来源 wiki 条目以规划调查：

```
ls SKILL_DIR/references/sources/

# 联邦财务 / 监管
cat SKILL_DIR/references/sources/sec-edgar.md       # 企业文件
cat SKILL_DIR/references/sources/usaspending.md     # 联邦合同
cat SKILL_DIR/references/sources/senate-ld.md       # 游说
cat SKILL_DIR/references/sources/ofac-sdn.md        # 制裁
cat SKILL_DIR/references/sources/icij-offshore.md   # 离岸泄露

# 身份 / 房产 / 诉讼 / 存档 / 新闻
cat SKILL_DIR/references/sources/nyc-acris.md       # 纽约市房产记录
cat SKILL_DIR/references/sources/opencorporates.md  # 全球企业注册信息
cat SKILL_DIR/references/sources/courtlistener.md   # 法院记录（联邦 + 州）
cat SKILL_DIR/references/sources/wayback.md         # Wayback Machine 存档
cat SKILL_DIR/references/sources/wikipedia.md       # Wikipedia + Wikidata
cat SKILL_DIR/references/sources/gdelt.md           # 全球新闻监控
```

每个条目遵循 9 节模板：摘要、访问、schema、覆盖范围、交叉引用键、数据质量、获取方式、法律说明、参考资料。

**交叉引用潜力**部分列出了来源之间的关联键 — 优先阅读这部分以选择合适的配对。

### 2. 获取数据

每个来源在 `SKILL_DIR/scripts/` 中都有仅使用标准库的抓取脚本：

**联邦财务 / 监管**

```bash
# SEC EDGAR 文件（企业披露）
python3 SKILL_DIR/scripts/fetch_sec_edgar.py --cik 0000320193 \
    --types 10-K,10-Q --out data/edgar_filings.csv

# USAspending 联邦合同
python3 SKILL_DIR/scripts/fetch_usaspending.py --recipient "EXAMPLE CORP" \
    --fy 2024 --out data/contracts.csv

# 参议院 LD-1 / LD-2 游说披露
python3 SKILL_DIR/scripts/fetch_senate_ld.py --client "EXAMPLE CORP" \
    --year 2024 --out data/lobbying.csv

# OFAC SDN 制裁名单（完整快照）
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --out data/ofac_sdn.csv

# ICIJ 离岸泄露 — 首次使用时下载约 70 MB 批量 CSV，
# 之后在本地搜索。缓存 30 天，存储于
# $HERMES_OSINT_CACHE/icij/（默认：~/.cache/hermes-osint/icij/）。
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --entity "EXAMPLE CORP" \
    --out data/icij.csv
```

**身份 / 房产 / 诉讼 / 存档 / 新闻**

```bash
# 纽约市房产记录（契约、抵押、留置权）— 通过 Socrata 访问 ACRIS
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --name "SMITH, JOHN" \
    --out data/acris.csv
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --address "571 HUDSON" \
    --out data/acris_addr.csv

# OpenCorporates — 130+ 司法管辖区企业注册信息
# （需要免费 token；设置 OPENCORPORATES_API_TOKEN 或传入 --token）
python3 SKILL_DIR/scripts/fetch_opencorporates.py --query "Example Corp" \
    --jurisdiction us_ny --out data/opencorporates.csv

# CourtListener — 联邦 + 州法院意见、PACER 案卷
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Smith v. Example Corp" \
    --type opinions --out data/courts.csv

# Wayback Machine — 历史网页快照
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match host --collapse digest --out data/wayback.csv

# Wikipedia + Wikidata — 叙述性传记 + 结构化事实
# 设置 HERMES_OSINT_UA=your-app/1.0 (your@email) 以标识自身
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Bill Gates" \
    --out data/wp.csv

# GDELT — 100+ 语言全球新闻，约 2015 年至今
python3 SKILL_DIR/scripts/fetch_gdelt.py --query '"Example Corp"' \
    --timespan 1y --out data/gdelt.csv
```

所有输出均为带标题行的标准化 CSV。脚本可幂等重复运行。

当私人个人不会出现在某来源中时（例如非上市公司人员不在 SEC EDGAR 中，非联邦承包商不在 USAspending 中，非游说客户不在参议院 LDA 中），脚本返回 0 行并给出明确警告，而不是静默写入空 CSV。EDGAR 会特别标记公司名称解析器匹配到的是个人 Form 3/4/5 申报人而非企业注册人的情况。

速率限制说明见各来源的 wiki 条目。默认抓取器在分页请求之间会礼貌地休眠。**API 密钥可提高支持它们的来源的速率限制**（`SEC_USER_AGENT`、`SENATE_LDA_TOKEN`、`OPENCORPORATES_API_TOKEN`、`COURTLISTENER_TOKEN`）。所有脚本会立即将 429 响应及上游配额消息呈现给用户，以便用户知道需要降速或提供密钥。

### 3. 跨来源实体解析

规范化名称并在两个 CSV 文件之间查找匹配：

```bash
# 将游说客户（参议院 LDA）与合同受益人（USAspending）进行匹配
python3 SKILL_DIR/scripts/entity_resolution.py \
    --left  data/lobbying.csv   --left-name-col  client_name \
    --right data/contracts.csv  --right-name-col recipient_name \
    --out data/cross_links.csv
```

三个匹配层级，附带显式置信度：

| 层级 | 方法 | 置信度 |
|------|--------|------------|
| `exact` | 去除后缀/标点后规范化字符串相等 | 高 |
| `fuzzy` | 排序词元相等（词袋匹配） | 中 |
| `token_overlap` | ≥60% 词元重叠，≥2 个共享词元，词元 ≥4 个字符 | 低 |

输出 `cross_links.csv` 列：`match_type, confidence, left_name, right_name, left_normalized, right_normalized, left_row, right_row`。

### 4. 统计时序关联（可选）

检验两个时间序列是否存在可疑的时间聚集 — 例如游说文件提交时间与合同授予时间接近 — 使用置换检验（permutation test）：

```bash
python3 SKILL_DIR/scripts/timing_analysis.py \
    --donations data/lobbying.csv --donation-date-col filing_date \
        --donation-amount-col income --donation-donor-col client_name \
        --donation-recipient-col registrant_name \
    --contracts data/contracts.csv --contract-date-col award_date \
        --contract-vendor-col recipient_name \
    --cross-links data/cross_links.csv \
    --permutations 1000 \
    --out data/timing.json
```

脚本的列标志是有意设计为通用的 — 原工具是为捐款与合同授予场景编写的，但它适用于任何通过交叉链接关联的（事件，收款方）时间序列。零假设：事件时序与合同授予日期无关。单尾 p 值 = 置换中平均最近合同距离 ≤ 观测值的比例。每个（付款方，供应商）配对至少需要 3 个事件才能运行检验。

### 5. 构建调查结果 JSON（证据链）

```bash
python3 SKILL_DIR/scripts/build_findings.py \
    --cross-links data/cross_links.csv \
    --timing data/timing.json \
    --out data/findings.json
```

每条调查结果包含 `id, title, severity, confidence, summary, evidence[], sources[]`。每个证据项指向来源 CSV 中的具体行。用户（或后续 agent）可以对照来源验证每项声明。

## 置信度与证据规范

这是该 skill 的核心规则。告知用户：

- 每项声明必须可追溯至具体记录。不得有无依据的断言。
- 置信度层级随声明传递。`match_type=fuzzy` 表示"可能"，而非"已确认"。
- 实体解析产生的是候选结果，而非结论。"ACME LLC"与"Acme Holdings Group"之间的 `fuzzy` 匹配是线索，不是事实。
- 统计显著性 ≠ 违规行为。p &lt; 0.05 意味着该时序模式在零假设下不太可能出现，并不能证明腐败。
- 此处所有数据来源均为公开记录，但仍可能包含不准确信息、过时信息或已编辑内容（GDPR、封存记录）。

## 添加新数据来源

使用模板：

```bash
cp SKILL_DIR/templates/source-template.md \
    SKILL_DIR/references/sources/<your-source>.md
```

填写全部 9 个部分。在 `scripts/` 中编写仅使用标准库的 `fetch_<source>.py` 脚本，输出标准化 CSV。在上方"何时使用"部分更新来源列表。

## 工具及其限制

- `entity_resolution.py` 不使用外部模糊匹配库（无 rapidfuzz，无 jellyfish）。词袋匹配是此处的上限。如需 Levenshtein 距离、音译或音素匹配，请单独 pip 安装。
- `timing_analysis.py` 使用 Python 的 `random` 模块进行置换。如需可复现性，请传入 `--seed N`。
- `fetch_*.py` 脚本使用 `urllib.request` 并遵守 `Retry-After` 头。大量批量使用仍可能违反服务条款 — 请先阅读各来源的法律说明部分。

## 法律说明

所有第一阶段来源均为公开记录。根据各自的访问条款（FOIA、公开记录法、ICIJ 明确发布、OFAC 公开数据），允许批量获取。但是：

- 部分来源速率限制较为严格。请遵守其响应头。
- 部分来源会编辑注册人信息（WHOIS 的 GDPR 合规、封存文件）。
- 交叉引用公开记录以识别私人个人可能存在伦理影响。该 skill 生成的是证据链，而非指控。