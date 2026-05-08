---
title: "Polymarket — 查询 Polymarket：市场、价格、订单簿、历史"
sidebar_label: "Polymarket"
description: "查询 Polymarket：市场、价格、订单簿、历史"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Polymarket

查询 Polymarket：市场、价格、订单簿、历史。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/research/polymarket` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Teknium |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Polymarket — 预测市场数据

使用 Polymarket 的公共 REST API 查询预测市场数据。
所有端点都是只读的，不需要任何身份验证。

完整端点参考及 curl 示例请参阅 `references/api-endpoints.md`。

## 使用时机

- 用户询问预测市场、投注赔率或事件概率
- 用户想知道"X 发生的概率是多少？"
- 用户特别询问 Polymarket
- 用户想要市场价格、订单簿数据或价格历史
- 用户要求监控或跟踪预测市场走势

## 关键概念

- **事件** 包含一个或多个 **市场**（一对多关系）
- **市场** 是二元结果，是/否价格在 0.00 到 1.00 之间
- 价格即概率：价格 0.65 意味着市场认为 65% 可能
- `outcomePrices` 字段：JSON 编码的数组，如 `["0.80", "0.20"]`
- `clobTokenIds` 字段：JSON 编码的两个令牌 ID 数组 [是, 否]，用于价格/订单簿查询
- `conditionId` 字段：用于价格历史查询的十六进制字符串
- 交易量以 USDC（美元）为单位

## 三个公共 API

1. **Gamma API**，位于 `gamma-api.polymarket.com` — 发现、搜索、浏览
2. **CLOB API**，位于 `clob.polymarket.com` — 实时价格、订单簿、历史
3. **Data API**，位于 `data-api.polymarket.com` — 交易、持仓量

## 典型工作流程

当用户询问预测市场赔率时：

1. 使用 Gamma API 公共搜索端点 **搜索** 用户的查询
2. **解析** 响应——提取事件及其嵌套的市场
3. **呈现** 市场问题、当前价格（以百分比形式）和交易量
4. 如需 **深入** ——使用 clobTokenIds 查询订单簿，使用 conditionId 查询历史

## 结果呈现

将价格格式化为百分比以提高可读性：
- outcomePrices `["0.652", "0.348"]` 变为 "是：65.2%，否：34.8%"
- 始终显示市场问题和概率
- 如有可用交易量则包含

示例：`"X 会发生吗？" — 65.2% 是（$1.2M 交易量）`

## 解析双重编码字段

Gamma API 将 `outcomePrices`、`outcomes` 和 `clobTokenIds` 作为 JSON 字符串
返回在 JSON 响应中（双重编码）。使用 Python 处理时，用
`json.loads(market['outcomePrices'])` 解析以获取实际数组。

## 速率限制

很宽松——正常使用不太可能触及：
- Gamma：每 10 秒 4,000 个请求（一般）
- CLOB：每 10 秒 9,000 个请求（一般）
- Data：每 10 秒 1,000 个请求（一般）

## 限制

- 此技能是只读的——不支持下单交易
- 交易需要基于钱包的加密身份验证（EIP-712 签名）
- 某些新市场可能没有价格历史
- 交易有地域限制，但只读数据全球可访问
