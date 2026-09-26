---
title: "Solana"
sidebar_label: "Solana"
description: "使用 USD 定价查询 Solana 区块链数据——钱包余额、带价值的代币投资组合、交易详情、NFT、巨鲸检测及实时网络状态..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Solana

使用 USD 定价查询 Solana 区块链数据——钱包余额、带价值的代币投资组合、交易详情、NFT、巨鲸检测及实时网络状态。使用 Solana RPC + CoinGecko，无需 API 密钥。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/blockchain/solana` 安装 |
| 路径 | `optional-skills/blockchain/solana` |
| 版本 | `0.2.0` |
| 作者 | Deniz Alagoz (gizdusum)，由 Hermes Agent 增强 |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Solana`, `Blockchain`, `Crypto`, `Web3`, `RPC`, `DeFi`, `NFT` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Solana 区块链 Skill

通过 CoinGecko 查询附带 USD 定价的 Solana 链上数据。
8 个命令：钱包投资组合、代币信息、交易记录、活动记录、NFT、
巨鲸检测、网络状态及价格查询。

无需 API 密钥。仅使用 Python 标准库（urllib、json、argparse）。

---

## 使用场景

- 用户查询 Solana 钱包余额、代币持仓或投资组合价值
- 用户想通过签名查看某笔具体交易
- 用户想获取 SPL 代币元数据、价格、供应量或持仓大户
- 用户想查看某地址的近期交易历史
- 用户想查看某钱包持有的 NFT
- 用户想查找大额 SOL 转账（巨鲸检测）
- 用户想了解 Solana 网络健康状态、TPS、epoch 或 SOL 价格
- 用户询问"BONK/JUP/SOL 的价格是多少？"

---

## 前置条件

辅助脚本仅使用 Python 标准库（urllib、json、argparse），无需外部包。

价格数据来自 CoinGecko 免费 API（无需密钥，速率限制约为每分钟 10-30 次请求）。如需更快查询，请使用 `--no-prices` 标志。

---

## 快速参考

RPC 端点（默认）：https://api.mainnet-beta.solana.com
覆盖方式：export SOLANA_RPC_URL=https://your-private-rpc.com

辅助脚本路径：~/.hermes/skills/blockchain/solana/scripts/solana_client.py

```
python3 solana_client.py wallet   <address> [--limit N] [--all] [--no-prices]
python3 solana_client.py tx       <signature>
python3 solana_client.py token    <mint_address>
python3 solana_client.py activity <address> [--limit N]
python3 solana_client.py nft      <address>
python3 solana_client.py whales   [--min-sol N]
python3 solana_client.py stats
python3 solana_client.py price    <mint_or_symbol>
```

---

## 操作步骤

### 0. 环境检查

```bash
python3 --version

# 可选：设置私有 RPC 以获得更好的速率限制
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# 确认连通性
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 1. 钱包投资组合

获取 SOL 余额、带 USD 价值的 SPL 代币持仓、NFT 数量及投资组合总值。代币按价值排序，过滤粉尘（dust），已知代币按名称标注（BONK、JUP、USDC 等）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

标志说明：
- `--limit N` — 显示前 N 个代币（默认：20）
- `--all` — 显示所有代币，不过滤粉尘，不限数量
- `--no-prices` — 跳过 CoinGecko 价格查询（更快，仅 RPC）

输出内容：SOL 余额 + USD 价值、按价值排序的代币列表及价格、粉尘数量、NFT 摘要、USD 投资组合总值。

### 2. 交易详情

通过 base58 签名查看完整交易信息，显示 SOL 和 USD 的余额变化。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  tx 5j7s8K...your_signature_here
```

输出内容：slot、时间戳、手续费、状态、余额变化（SOL + USD）、程序调用。

### 3. 代币信息

获取 SPL 代币元数据、当前价格、市值、供应量、精度、铸造/冻结权限及前 5 大持仓地址。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

输出内容：名称、符号、精度、供应量、价格、市值、前 5 大持仓地址及占比。

### 4. 近期活动

列出某地址的近期交易（默认：最近 10 条，最多：25 条）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  activity 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM --limit 25
```

### 5. NFT 投资组合

列出某钱包持有的 NFT（启发式判断：amount=1 且 decimals=0 的 SPL 代币）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  nft 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

注意：此启发式方法无法检测压缩 NFT（cNFT）。

### 6. 巨鲸检测器

扫描最新区块中的大额 SOL 转账及其 USD 价值。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  whales --min-sol 500
```

注意：仅扫描最新区块——为时间点快照，非历史数据。

### 7. 网络状态

实时 Solana 网络健康状态：当前 slot、epoch、TPS、供应量、验证者版本、SOL 价格及市值。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 8. 价格查询

通过铸造地址或已知符号快速查询任意代币价格。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price BONK
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price JUP
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price SOL
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

已知符号：SOL、USDC、USDT、BONK、JUP、WETH、JTO、mSOL、stSOL、
PYTH、HNT、RNDR、WEN、W、TNSR、DRIFT、bSOL、JLP、WIF、MEW、BOME、PENGU。

---

## 注意事项

- **CoinGecko 速率限制** — 免费套餐约每分钟 10-30 次请求。价格查询每个代币消耗 1 次请求。持有大量代币的钱包可能无法获取所有代币价格。如需提速，请使用 `--no-prices`。
- **公共 RPC 速率限制** — Solana 主网公共 RPC 对请求有限制。生产环境请将 SOLANA_RPC_URL 设置为私有端点（Helius、QuickNode、Triton）。
- **NFT 检测为启发式** — amount=1 且 decimals=0。压缩 NFT（cNFT）和 Token-2022 NFT 不会出现。
- **巨鲸检测器仅扫描最新区块** — 非历史数据，结果因查询时刻而异。
- **交易历史** — 公共 RPC 保留约 2 天的数据，较旧的交易可能不可用。
- **代币名称** — 约 25 个知名代币按名称标注，其他代币显示缩写铸造地址。如需完整信息，请使用 `token` 命令。
- **429 重试** — RPC 和 CoinGecko 调用在遇到速率限制错误时均会以指数退避方式最多重试 2 次。

---

## 验证

```bash
# 应输出当前 Solana slot、TPS 及 SOL 价格
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```