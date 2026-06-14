---
title: "Evm — 只读 EVM 客户端：跨 8 条链的钱包、代币、Gas"
sidebar_label: "Evm"
description: "只读 EVM 客户端：跨 8 条链的钱包、代币、Gas"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Evm

只读 EVM 客户端：跨 8 条链的钱包、代币、Gas。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/blockchain/evm` 安装 |
| 路径 | `optional-skills/blockchain/evm` |
| 版本 | `1.0.0` |
| 作者 | Mibayy (@Mibayy), youssefea (@youssefea), ethernet8023 (@ethernet8023), Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `EVM`, `Ethereum`, `BNB`, `BSC`, `Base`, `Arbitrum`, `Polygon`, `Optimism`, `Avalanche`, `zkSync`, `Blockchain`, `Crypto`, `Web3`, `DeFi`, `NFT`, `ENS`, `Whale`, `Security` |
| 相关 skill | [`solana`](/user-guide/skills/optional/blockchain/blockchain-solana) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# EVM Blockchain Skill

跨 8 条链查询 EVM 兼容区块链数据，支持 USD 定价。
14 个命令：钱包投资组合、代币信息、交易记录、活动历史、Gas 追踪器、
网络统计、价格查询、多链扫描、巨鲸检测、ENS 解析、
授权检查器、合约检查器和交易解码器。

支持 8 条链：Ethereum、BNB Chain (BSC)、Base、Arbitrum One、Polygon、
Optimism、Avalanche (C-Chain)、zkSync Era。

无需 API 密钥。零外部依赖 — 仅使用 Python 标准库
（urllib、json、argparse、threading）。

> **取代独立的 `base` skill。** Base 专属代币（AERO、DEGEN、
> TOSHI、BRETT、WELL、cbETH、cbBTC、wstETH、rETH）以及原先位于
> `optional-skills/blockchain/base/` 下的所有 Base RPC 功能已整合
> 至本 skill。对任意命令传入 `--chain base` 即可覆盖 Base。

---

## 使用场景
- 用户查询任意 EVM 链上的钱包余额或投资组合
- 用户希望同时检查同一钱包在所有链上的情况
- 用户想通过交易哈希检查某笔交易（或解码其操作内容）
- 用户想查询 ERC-20 代币的元数据、价格、供应量或市值
- 用户想查看某地址的近期交易历史
- 用户想查询当前 Gas 价格或比较各链手续费
- 用户想在近期区块中查找大额巨鲸转账
- 用户想解析 ENS 名称（如 vitalik.eth）或反向查询地址
- 用户想检查合约是否存在危险的代币授权
- 用户想检查智能合约（是否为代理合约？ERC-20？ERC-721？字节码大小？）
- 用户想在交易前比较各链 Gas 费用

---

## 前置条件
仅需 Python 3.8+ 标准库，无需 pip 安装。
定价：CoinGecko 免费 API（有速率限制，约 10-30 次请求/分钟）。
ENS：ensideas.com 公共 API。
交易解码：4byte.directory 公共 API。

覆盖 RPC 端点：`export EVM_RPC_URL=https://your-rpc.com`

辅助脚本路径：`~/.hermes/skills/blockchain/evm/scripts/evm_client.py`

---

## 快速参考

```
SCRIPT=~/.hermes/skills/blockchain/evm/scripts/evm_client.py

# 网络与价格
python3 $SCRIPT stats                            # Ethereum 统计
python3 $SCRIPT stats --chain arbitrum           # Arbitrum 统计
python3 $SCRIPT compare                          # 全部 8 条链的 Gas + 价格

# 钱包
python3 $SCRIPT wallet 0xd8dA...96045            # 投资组合（ETH + ERC-20）
python3 $SCRIPT wallet 0xd8dA...96045 --chain bsc
python3 $SCRIPT multichain 0xd8dA...96045        # 同一钱包在所有链上的情况

# 代币与价格
python3 $SCRIPT price ETH
python3 $SCRIPT price 0xdAC1...1ec7              # 通过合约地址查询
python3 $SCRIPT token 0xdAC1...1ec7              # ERC-20 元数据 + 市值

# 交易
python3 $SCRIPT tx 0x5c50...f060                 # 交易详情
python3 $SCRIPT decode 0x5c50...f060             # 解码输入数据（4byte.directory）
python3 $SCRIPT activity 0xd8dA...96045          # 近期交易

# Gas
python3 $SCRIPT gas                              # Gas 价格 + 费用估算
python3 $SCRIPT gas --chain optimism

# 安全
python3 $SCRIPT allowance 0xd8dA...96045         # 危险的 ERC-20 授权
python3 $SCRIPT contract 0xdAC1...1ec7           # 合约检查（代理合约？标准？）

# ENS
python3 $SCRIPT ens vitalik.eth                  # 名称 -> 地址 + 个人资料
python3 $SCRIPT ens 0xd8dA...96045               # 地址 -> ENS 名称

# 巨鲸检测
python3 $SCRIPT whale                            # 大额转账（最近 20 个区块，>$10k）
python3 $SCRIPT whale --blocks 50 --min-usd 100000 --chain arbitrum
```

---

## 操作流程

### 0. 环境检查
```bash
python3 --version   # 需要 3.8+
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats
```

### 1. 钱包投资组合
原生余额 + 已知 ERC-20 代币，按 USD 价值排序。
```bash
python3 $SCRIPT wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
python3 $SCRIPT wallet 0xd8dA... --chain bsc --no-prices   # 更快
```

### 2. 多链扫描
使用多线程同时扫描同一地址在全部 8 条链上的情况。
```bash
python3 $SCRIPT multichain 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```
输出：每条链的原生余额 + 代币持仓 + USD 总计。

### 3. 比较（Gas + 价格）
并行查询全部 8 条链，显示最便宜/最贵的链。
```bash
python3 $SCRIPT compare
```

### 4. 交易详情与解码
```bash
python3 $SCRIPT tx 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060
python3 $SCRIPT decode 0x5c504ed...   # 显示人类可读的函数签名
```
解码使用 4byte.directory 将 0xa9059cbb 转换为 transfer(address,uint256)。

### 5. ENS 解析
```bash
python3 $SCRIPT ens vitalik.eth          # -> 0xd8dA... + 头像 + 社交链接
python3 $SCRIPT ens 0xd8dA...96045       # -> vitalik.eth
```

### 6. 授权检查器（安全）
检查已授予已知 DEX/跨链桥合约的 ERC-20 授权。
```bash
python3 $SCRIPT allowance 0xYourWallet
```
将无限额授权标记为高风险。

### 7. 合约检查器
```bash
python3 $SCRIPT contract 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48   # USDC（代理合约）
python3 $SCRIPT contract 0xdAC17F958D2ee523a2206206994597C13D831ec7   # USDT（ERC-20）
```
检测：代理合约（EIP-1967/EIP-1167）、ERC-20、ERC-721、ERC-165。显示字节码大小及代理合约的实现地址。

### 8. 巨鲸检测
```bash
python3 $SCRIPT whale                                    # ETH，最近 20 个区块，>$10k
python3 $SCRIPT whale --blocks 50 --min-usd 50000 --chain bsc
```

### 9. Gas 追踪器
```bash
python3 $SCRIPT gas
python3 $SCRIPT gas --chain polygon
```
显示 gwei 价格 + 以下操作的 USD 费用：转账、ERC-20 转账、授权、兑换、NFT 铸造、NFT 转账。

---

## 支持的链
| 键        | 名称           | 原生代币 | Chain ID |
|-----------|----------------|--------|----------|
| ethereum  | Ethereum       | ETH    | 1        |
| bsc       | BNB Chain      | BNB    | 56       |
| base      | Base           | ETH    | 8453     |
| arbitrum  | Arbitrum One   | ETH    | 42161    |
| polygon   | Polygon        | POL    | 137      |
| optimism  | Optimism       | ETH    | 10       |
| avalanche | Avalanche C    | AVAX   | 43114    |
| zksync    | zkSync Era     | ETH    | 324      |

---

## 注意事项
- CoinGecko 免费套餐：约 10-30 次请求/分钟。使用 `--no-prices` 可加快钱包扫描速度。
- 公共 RPC 可能限速。生产环境请将 EVM_RPC_URL 设置为私有端点。
- `wallet` 和 `allowance` 仅检查已知代币列表（每条链约 30 个代币）。如需完整代币发现，请使用区块浏览器。
- `activity` 仅扫描近期区块（最多 200 个）。如需完整历史记录，请使用 Etherscan API。
- `multichain` 运行 8 个并行线程 — 可能触发公共 RPC 的速率限制。
- ENS 解析依赖单一公共端点（ensideas.com / ens.vitalik.ca），无备用方案。若该端点不可用，`ens` 命令将失败 — 稍后重试或使用区块浏览器。
- 交易解码依赖单一公共端点（4byte.directory），无备用方案。数据库中未收录的选择器将显示为 `unknown`。
- **L2 Gas 估算仅为 L2 执行费用。** 在 Base、Arbitrum、Optimism、zkSync 等 rollup 上，实际交易费用还包含取决于 calldata 大小和当前 L1 Gas 价格的 L1 数据发布费用。`gas` 命令不估算该 L1 部分。对于 Base，请参阅网络的 L1 费用预言机（合约 `0x420000000000000000000000000000000000000F`）。
- 地址/交易哈希输入会验证 0x 前缀 + 正确长度 + 十六进制格式，但**不**强制执行 EIP-55 校验和大小写（RPC 端点接受任意大小写的十六进制）。

---

## 验证
```bash
# 应输出当前区块、Gas 价格、ETH 价格
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats

# 应将 vitalik.eth 解析为 0xd8dA...
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py ens vitalik.eth
```