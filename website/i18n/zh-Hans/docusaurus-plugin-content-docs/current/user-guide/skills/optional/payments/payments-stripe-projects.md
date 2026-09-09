---
title: "Stripe Projects — 通过 Stripe Projects 配置 SaaS 服务 + 同步凭据"
sidebar_label: "Stripe Projects"
description: "通过 Stripe Projects 配置 SaaS 服务 + 同步凭据"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stripe Projects

通过 Stripe Projects 配置 SaaS 服务 + 同步凭据。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/payments/stripe-projects` 安装 |
| 路径 | `optional-skills/payments/stripe-projects` |
| 版本 | `0.1.0` |
| 作者 | Teknium (teknium1)、Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos |
| 标签 | `Payments`、`Stripe`、`Projects`、`Provisioning`、`Infrastructure` |
| 相关技能 | [`stripe-link-cli`](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli)、[`mpp-agent`](/docs/user-guide/skills/optional/payments/payments-mpp-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Stripe Projects 技能

包装 [Stripe Projects](https://projects.dev) CLI 插件，使 Hermes 可以配置 SaaS 服务（Neon、Twilio、Vercel 等），生成凭据并同步到用户的 `.env`，并从一个地方管理跨提供商的账单。

## 何时使用

- "set up &lt;provider&gt;"、"provision &lt;Neon|Twilio|Vercel|...&gt;"、"create a database"
- "manage my stack credentials"、"rotate this key"、"upgrade my plan"

## 安装

macOS：
```
brew install stripe/stripe-cli/stripe
stripe plugin install projects
```

## 操作流程

### 1. 初始化项目

```
cd <project-root>
stripe projects init
```

### 2. 发现可用提供商

```
stripe projects catalog
```

### 3. 添加服务

```
stripe projects add <provider>/<service>
```

### 4. 验证

```
stripe projects list
```

### 5. 管理/升级/移除

```
stripe projects upgrade <provider>
stripe projects remove <provider>
stripe projects rotate <provider>
```

## 陷阱

- **`.env` 写入是真正的写入。** 始终检查 `.gitignore`。
- **每项目状态。** 在两个不同项目中配置同一服务会创建两个独立资源——以及两份账单。
- **账单发生在 Stripe 方面。** `add`/`upgrade` 期间的层级提示是真正的收费。
- **vault 中的凭据已加密但 `.env` 是明文。**
- **移除服务并不总是销毁底层资源。**
