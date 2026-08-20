---
title: "Stripe Link CLI — 通过 Stripe Link 的代理支付——卡片、SPT、审批"
sidebar_label: "Stripe Link CLI"
description: "通过 Stripe Link 的代理支付——卡片、SPT、审批"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stripe Link CLI

通过 Stripe Link 的代理支付——卡片、SPT、审批。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/payments/stripe-link-cli` 安装 |
| 路径 | `optional-skills/payments/stripe-link-cli` |
| 版本 | `0.1.0` |
| 作者 | Teknium (teknium1)、Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos |
| 标签 | `Payments`、`Stripe`、`Link`、`Checkout`、`MPP` |
| 相关技能 | [`mpp-agent`](/docs/user-guide/skills/optional/payments/payments-mpp-agent)、[`stripe-projects`](/docs/user-guide/skills/optional/payments/payments-stripe-projects) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Stripe Link CLI 技能

包装 [@stripe/link-cli](https://github.com/stripe/link-cli)，使 Hermes 可以使用一次性虚拟卡或 Shared Payment Tokens (SPT) 代用户完成购买。每次消费都由 Link 移动/Web 应用中的应用内审批控制——Hermes 不能自行批准。

目前仅限美国（Link 账户要求）。上游 CLI 不支持 Windows——此技能限制为 `[linux, macos]`。

## 何时使用

- "buy X"、"pay for X"、"make a purchase"、"complete checkout"
- "get me a card"、"I need a payment method"
- HTTP 402 响应带 `www-authenticate: ... method="stripe"`

## 前提条件

- Node.js 20+ 在 PATH 上
- 美国地区（Link 账户要求）

## 安装

```
npm install -g @stripe/link-cli
```

## 操作流程

### 1. 检查/建立认证

```
link-cli auth status
```

如果未认证，登录：

```
link-cli auth login --client-name "Hermes" --interval 5 --timeout 300
```

**在 `auth status` 确认登录之前不要继续此步骤。**

### 2. 在创建消费请求前评估商户

| 商户类型 | `--credential-type` |
|---------|---------------------|
| 标准 Web checkout / Stripe Elements | `card`（默认） |
| 返回 HTTP 402 带 `method="stripe"` | `shared_payment_token` |
| 返回 HTTP 402 不带 `method="stripe"` | 不支持 — 停止 |

### 3. 列出支付方式 + 配送

```
link-cli payment-methods list
link-cli shipping-address list
```

### 4. 创建消费请求

在发出此命令前与用户确认最终总额。金额单位为分。

```
link-cli spend-request create \
  --payment-method-id <pm_id> \
  --merchant-name "<name>" \
  --merchant-url "<url>" \
  --context "<one sentence: what is being purchased and why>" \
  --amount <cents> \
  --line-item "name:<item>,unit_amount:<cents>,quantity:1" \
  --total "type:total,display_text:Total,amount:<cents>" \
  --request-approval
```

### 5. 检索凭据 — 安全地

**不要将卡片详情打印到 stdout。** 使用 `--output-file`：

```
link-cli spend-request retrieve <lsrq_id> \
  --include card \
  --output-file /tmp/link-card.json \
  --format json
```

### 6. 使用凭据

购买完成后立即删除卡片文件。

## 陷阱

- **仅限美国。** 美国之外 `auth login` 会失败。
- **卡片 PAN 永远不能进入代理上下文。**
- **`--request-approval` 阻塞直到用户操作。**
