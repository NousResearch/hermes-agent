---
title: "Stocks — 通过 Yahoo 获取股票报价、历史、搜索、比较及加密货币数据"
sidebar_label: "Stocks"
description: "通过 Yahoo 获取股票报价、历史、搜索、比较及加密货币数据"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stocks

通过 Yahoo 获取股票报价、历史、搜索、比较及加密货币数据。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/finance/stocks` 安装 |
| 路径 | `optional-skills/finance/stocks` |
| 版本 | `0.1.0` |
| 作者 | Mibay (Mibayy), Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Stocks`, `Finance`, `Market`, `Crypto`, `Investing` |
| 相关 skill | [`dcf-model`](/user-guide/skills/optional/finance/finance-dcf-model), [`comps-analysis`](/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/user-guide/skills/optional/finance/finance-lbo-model) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Stocks Skill

通过 Yahoo Finance 提供只读市场数据。五个命令：`quote`、`search`、
`history`、`compare`、`crypto`。仅使用 Python 标准库——无需 API key，无需 pip
安装。Yahoo 的接口为非官方接口，可能存在频率限制或发生变更。

## 使用场景

- 用户询问当前股票价格（AAPL、TSLA、MSFT 等）
- 用户希望通过公司名称查找股票代码
- 用户需要 OHLCV 历史数据或某日期范围内的表现
- 用户希望并排比较多个股票代码
- 用户询问加密货币价格（BTC、ETH、SOL 等）

## 前置条件

仅需 Python 3.8+ 标准库。可选：设置 `ALPHA_VANTAGE_KEY` 以在 Yahoo 的 crumb 保护字段返回 null 时补充 `market_cap`、`pe_ratio` 及 52 周高低点数据。免费 key 申请：https://www.alphavantage.co/support/#api-key

## 运行方式

通过 `terminal` 工具调用。安装完成后：

```
SCRIPT=~/.hermes/skills/finance/stocks/scripts/stocks_client.py
python3 $SCRIPT quote AAPL
```

所有输出均为 stdout 上的 JSON——如需切片处理，可通过管道传给 `jq`。

## 快速参考

```
python3 $SCRIPT quote AAPL
python3 $SCRIPT quote AAPL MSFT GOOGL TSLA
python3 $SCRIPT search "Tesla"
python3 $SCRIPT history NVDA --range 6mo
python3 $SCRIPT compare AAPL MSFT GOOGL
python3 $SCRIPT crypto BTC ETH SOL
```

## 命令

### `quote SYMBOL [SYMBOL2 ...]`

当前价格、涨跌额、涨跌幅、成交量、52 周高低点。

### `search QUERY`

通过公司名称查找股票代码。返回前 5 条结果：代码、名称、交易所、类型。

### `history SYMBOL [--range RANGE]`

每日 OHLCV 数据及统计信息（最小值、最大值、均值、总回报率 %）。时间范围：`1mo`、
`3mo`、`6mo`、`1y`、`5y`。默认：`1mo`。

### `compare SYMBOL1 SYMBOL2 [...]`

并排对比：价格、涨跌幅、52 周表现。

### `crypto SYMBOL [SYMBOL2 ...]`

加密货币价格。传入 `BTC`（脚本会自动追加 `-USD`）。

## 注意事项

- Yahoo Finance 的 API 为非官方接口。接口可能在未通知的情况下发生变更或触发频率限制——如果请求开始失败，原因即在于此。
- 当 Yahoo 的 crumb 会话未建立时，`quote` 命令中的 `market_cap` 和 `pe_ratio` 可能返回 null。设置 `ALPHA_VANTAGE_KEY` 可进行补充。
- 批量请求之间请添加适当延迟，以避免触发频率限制。
- 本 skill 为只读——不支持下单，不集成账户。

## 验证

```
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py quote AAPL
```

返回包含 `symbol: "AAPL"` 及数值型 `price` 字段的 JSON 对象。