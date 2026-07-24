---
title: "MPP Agent — 通过 Machine Payments Protocol (MPP) 支付 HTTP 402 API"
sidebar_label: "MPP Agent"
description: "通过 Machine Payments Protocol (MPP) 支付 HTTP 402 API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# MPP Agent

通过 Machine Payments Protocol (MPP) 支付 HTTP 402 API。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/payments/mpp-agent` 安装 |
| 路径 | `optional-skills/payments/mpp-agent` |
| 版本 | `0.1.0` |
| 作者 | Teknium (teknium1)、Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos |
| 标签 | `Payments`、`MPP`、`HTTP-402`、`Tempo`、`Stripe` |
| 相关技能 | [`stripe-link-cli`](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli)、[`stripe-projects`](/docs/user-guide/skills/optional/payments/payments-stripe-projects) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# MPP Agent 技能

包装 Machine Payments Protocol (MPP, https://mpp.dev) 客户端，使 Hermes 可以对响应 `HTTP 402 Payment Required` 的服务器进行按次请求的 API 访问支付。

三种客户端选项，均通过 npm 分发。选择解决用户需求的最轻量的。

## 何时使用

- 商户 API 返回 `HTTP 402` 和 `www-authenticate` 头 — 用户想要实际支付而非仅记录响应。
- 用户请求"按请求付费"、"设置代理钱包"、"使用 Tempo / Privy / AgentCash"。
- Stripe Link 消费产生了 Shared Payment Token (SPT)。

## 选择客户端

| 工具 | 场景 | 设置 |
|------|------|------|
| `link-cli` | 用户已设置 Stripe Link | 参见 `stripe-link-cli` 技能 |
| Tempo Wallet | MPP 服务 + 消费控制 | `tempo wallet login` |
| Privy Agent CLI | 多链钱包 | `privy-agent-wallets login` |
| AgentCash | 300+ 预定价 API | `npx agentcash onboard` |
| `mppx` | 开发+调试，最小依赖 | `npm install -g mppx` |

## 操作流程（mppx，最快路径）

### 1. 安装 + 创建账户

```
npm install -g mppx
mppx account create
```

### 2. 检查商户 402 挑战

```
curl -i <url>
```

### 3. 支付请求

```
mppx <url>
```

### 4. 验证收据

```
mppx <url> -v
```

## 陷阱

- **没有 `method="stripe"` 的 `HTTP 402` 不能用 Stripe Link 支付。**
- **零金额挑战。** 一些 MPP 端点收取 `$0.00`。
- **钱包密钥永远不进入代理上下文。**
