---
title: "Hyperliquid — Hyperliquid 市场数据、账户历史、交易复盘"
sidebar_label: "Hyperliquid"
description: "Hyperliquid 市场数据、账户历史、交易复盘"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hyperliquid

Hyperliquid 市场数据、账户历史、交易复盘。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/blockchain/hyperliquid` 安装 |
| 路径 | `optional-skills/blockchain/hyperliquid` |
| 版本 | `0.1.0` |
| 作者 | Hugo Sequier (Hugo-SEQUIER), Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Hyperliquid`, `Blockchain`, `Crypto`, `Trading`, `Perpetuals`, `Spot`, `DeFi` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Hyperliquid Skill

通过公开的 `/info` 端点查询 Hyperliquid 市场和账户数据。
只读 — 无需 API key，无需签名，不支持下单。

12 个命令：`dexs`、`markets`、`spots`、`candles`、`funding`、`l2`、`state`、
`spot-balances`、`fills`、`orders`、`review`、`export`。仅使用标准库
（`urllib`、`json`、`argparse`）。

---

## 使用场景

- 用户请求 Hyperliquid 永续合约或现货市场数据、K 线、资金费率或 L2 盘口
- 用户希望查看钱包的永续仓位、现货余额、成交记录或挂单
- 用户希望结合近期成交与市场背景进行交易后复盘
- 用户希望查看 builder 部署的永续 DEX 或 HIP-3 市场
- 用户希望导出标准化的 K 线 + 资金费率 JSON 数据用于回测准备

---

## 前置条件

仅使用标准库 — 无需外部包，无需 API key。

脚本从 `~/.hermes/.env` 读取两个可选默认值：

- `HYPERLIQUID_API_URL` — 默认为 `https://api.hyperliquid.xyz`。设置为
  `https://api.hyperliquid-testnet.xyz` 可切换至测试网。
- `HYPERLIQUID_USER_ADDRESS` — `state`、`spot-balances`、`fills`、`orders` 和 `review` 的默认地址。若未设置，则将地址作为第一个位置参数传入。

当前工作目录中的项目 `.env` 文件作为开发环境的备用配置。

辅助脚本：`~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py`

---

## 运行方式

通过 `terminal` 工具调用：

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py <command> [args]
```

在任意命令后添加 `--json` 可获得机器可读输出。

---

## 快速参考

```bash
hyperliquid_client.py dexs
hyperliquid_client.py markets [--dex DEX] [--limit N] [--sort volume|oi|funding_abs|change_abs|name]
hyperliquid_client.py spots [--limit N]
hyperliquid_client.py candles <coin> [--interval 1h] [--hours 24] [--limit N]
hyperliquid_client.py funding <coin> [--hours 72] [--limit N]
hyperliquid_client.py l2 <coin> [--levels N]
hyperliquid_client.py state [address] [--dex DEX]
hyperliquid_client.py spot-balances [address] [--limit N]
hyperliquid_client.py fills [address] [--hours N] [--limit N] [--aggregate-by-time]
hyperliquid_client.py orders [address] [--limit N]
hyperliquid_client.py review [address] [--coin COIN] [--hours N] [--fills N]
hyperliquid_client.py export <coin> [--interval 1h] [--hours N] [--output PATH]
```

对于 `state`、`spot-balances`、`fills`、`orders` 和 `review`，当 `~/.hermes/.env` 中设置了 `HYPERLIQUID_USER_ADDRESS` 时，地址参数为可选。

---

## 操作流程

### 1. 发现 DEX 和市场

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py dexs

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 15 --sort volume

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spots --limit 15
```

- `--dex` 仅适用于永续合约端点；省略则使用第一个永续 DEX。
- 现货交易对可能显示为 `PURR/USDC` 或别名如 `@107`。
- HIP-3 市场的币种名称带有 DEX 前缀，例如 `mydex:BTC`。

### 2. 拉取历史市场数据

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  candles BTC --interval 1h --hours 72 --limit 48

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  funding BTC --hours 168 --limit 30
```

时间范围端点支持分页。对于较大的时间窗口，可使用更晚的 `startTime` 重复请求，或使用下方的 `export` 命令。

### 3. 查看实时盘口

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  l2 BTC --levels 10
```

当用户询问盘口深度、近期流动性或大单市场冲击时使用。

### 4. 查看账户信息

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  state 0xabc...

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spot-balances
```

`state` 返回永续仓位；`spot-balances` 返回现货持仓。
适用于"我的仓位情况如何"、"我持有什么"、"可提现金额是多少"等问题。

### 5. 查看成交记录和挂单

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  fills 0xabc... --hours 72 --limit 25

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  orders --limit 25
```

### 6. 生成交易复盘报告

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review 0xabc... --hours 72 --fills 50

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review --coin BTC --hours 168
```

报告包含已实现 PnL、手续费、盈亏次数、币种明细、每个交易永续合约的市场趋势和平均资金费率，以及启发式分析（手续费拖累、集中度、逆势亏损）。

深度交易后分析流程：先用 `review` 找出问题币种或时间段 → 拉取该时段的 `fills` 和 `orders` → 拉取每个交易币种的 `candles` 和 `funding` → 将决策质量与结果质量分开评判。

### 7. 导出可复用数据集

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 1h --hours 168 --output ./btc-1h-7d.json

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 15m --hours 72 --end-time-ms 1760000000000
```

输出 JSON 包含：schema 版本、数据源元数据、精确时间窗口、标准化 K 线行、标准化资金费率行、汇总统计。使用 `--end-time-ms` 可获得可复现的时间窗口。

---

## 注意事项

- 公开 info 端点有速率限制。大范围历史查询可能返回截断的时间窗口；请使用更晚的 `startTime` 值迭代请求。
- `fills --hours ...` 使用 `userFillsByTime`，仅暴露近期滚动窗口 — 不支持完整历史归档。
- `historicalOrders` 仅返回近期订单，不支持完整导出。
- `review` 命令基于启发式分析。仅凭成交记录无法还原交易意图、下单质量或真实滑点。
- `export` 命令输出标准化数据集，而非回测引擎。仍需自行构建滑点/成交模型。
- 现货别名如 `@107` 是有效标识符，即使 UI 显示的是更友好的名称。
- `l2` 是某一时刻的快照，不是时间序列。

---

## 验证

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 5
```

应输出按 24 小时名义成交量排名的 Hyperliquid 永续合约市场前五名。