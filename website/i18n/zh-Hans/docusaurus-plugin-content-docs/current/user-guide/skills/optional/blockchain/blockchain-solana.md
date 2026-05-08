---
title: "Solana"
sidebar_label: "Solana"
description: "查询 Solana 区块链数据，包含 USD 价格 — 钱包余额、带价值的代币组合、交易详情、NFT、巨鲸检测和实时网络状态。"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Solana

查询 Solana 区块链数据，包含 USD 价格 — 钱包余额、带价值的代币组合、交易详情、NFT、巨鲸检测和实时网络状态。使用 Solana RPC + CoinGecko。无需 API 密钥。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/blockchain/solana` 安装 |
| 路径 | `optional-skills/blockchain/solana` |
| 版本 | `0.2.0` |
| 作者 | Deniz Alagoz (gizdusum)，由 Hermes Agent 增强 |
| 许可证 | MIT |
| 标签 | `Solana`、`Blockchain`、`Crypto`、`Web3`、`RPC`、`DeFi`、`NFT` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Solana Blockchain Skill

使用 CoinGecko 通过 USD 价格增强查询 Solana 链上数据。
8 个命令：钱包组合、代币信息、交易、活动、NFT、
巨鲸检测、网络状态和价格查询。

无需 API 密钥。仅使用 Python 标准库（urllib、json、argparse）。

---

## 使用场景

- 用户询问 Solana 钱包余额、代币持有量或组合价值
- 用户想通过签名检查特定交易
- 用户想要 SPL 代币元数据、价格、供应量或主要持有者
- 用户想要地址的最近交易历史
- 用户想要钱包拥有的 NFT
- 用户想查找大额 SOL 转账（巨鲸检测）
- 用户想要 Solana 网络健康状况、TPS、时期或 SOL 价格
- 用户询问"BONK/JUP/SOL 的价格是多少？"

---

## 前置条件

辅助脚本仅使用 Python 标准库（urllib、json、argparse）。
无需外部包。

定价数据来自 CoinGecko 免费 API（无需密钥，受限于约 10-30 请求/分钟）。
要加快查询速度，请使用 `--no-prices` 标志。

---

## 快速参考

RPC 端点（默认）：https://api.mainnet-beta.solana.com
覆盖：export SOLANA_RPC_URL=https://your-private-rpc.com

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

## 步骤

### 0. 设置检查

```bash
python3 --version

# 可选：设置私有 RPC 以获得更好的速率限制
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# 确认连接
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 1. 钱包组合

获取 SOL 余额、带有 USD 价值的 SPL 代币持有量、NFT 数量和
组合总计。代币按价值排序，过滤小额代币，已知代币
按名称标注（BONK、JUP、USDC 等）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

标志：
- `--limit N` — 显示前 N 个代币（默认：20）
- `--all` — 显示所有代币，不过滤小额，无限制
- `--no-prices` — 跳过 CoinGecko 价格查询（更快，仅 RPC）

输出包括：SOL 余额 + USD 价值、按价值排序的带价格的代币列表、小额代币数量、NFT 摘要、以 USD 计的总组合价值。

### 2. 交易详情

通过 base58 签名检查完整交易。显示余额变化
（SOL 和 USD）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  tx 5j7s8K...your_signature_here
```

输出：插槽、时间戳、费用、状态、余额变化（SOL + USD）、
程序调用。

### 3. 代币信息

获取 SPL 代币元数据、当前价格、市值、供应量、小数位、
mint/freeze 权限和前 5 名持有者。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

输出：名称、符号、小数位、供应量、价格、市值、前 5 名
持有者及百分比。

### 4. 最近活动

列出地址的最近交易（默认：最近 10 条，最大：25 条）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  activity 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM --limit 25
```

### 5. NFT 组合

列出钱包拥有的 NFT（启发式方法：amount=1、decimals=0 的 SPL 代币）。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  nft 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

注意：此启发式方法检测不到压缩 NFT（cNFT）。

### 6. 巨鲸检测器

扫描最近区块的大额 SOL 转账及 USD 价值。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  whales --min-sol 500
```

注意：仅扫描最新区块 — 时间点快照，非历史数据。

### 7. 网络状态

实时 Solana 网络健康状况：当前插槽、时期、TPS、供应量、验证器
版本、SOL 价格和市值。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 8. 价格查询

通过 mint 地址或已知符号快速查询任何代币价格。

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price BONK
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price JUP
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price SOL
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

已知符号：SOL、USDC、USDT、BONK、JUP、WETH、JTO、mSOL、stSOL、
PYTH、HNT、RNDR、WEN、W、TNSR、DRIFT、bSOL、JLP、WIF、MEW、BOME、PENGU。

---

## 陷阱

- **CoinGecko 速率限制** — 免费层级允许约 10-30 请求/分钟。
  价格查询每个代币使用 1 个请求。拥有许多代币的钱包可能
  无法获得所有代币的价格。使用 `--no-prices` 加快速度。
- **公共 RPC 速率限制** — Solana mainnet 公共 RPC 限制请求。
  对于生产使用，将 SOLANA_RPC_URL 设置为私有端点
  （Helius、QuickNode、Triton）。
- **NFT 检测是启发式的** — amount=1 + decimals=0。压缩
  NFT（cNFT）和 Token-2022 NFT 不会显示。
- **巨鲸检测器仅扫描最新区块** — 非历史数据。结果
  因查询时刻而异。
- **交易历史** — 公共 RPC 保留约 2 天。较旧的交易
  可能不可用。
- **代币名称** — 约 25 个知名代币按名称标注。其他的
  显示缩写后的 mint 地址。使用 `token` 命令获取完整信息。
- **429 时重试** — RPC 和 CoinGecko 调用在速率限制错误时
  最多重试 2 次，使用指数退避。

---

## 验证

```bash
# 应打印当前 Solana 插槽、TPS 和 SOL 价格
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```
