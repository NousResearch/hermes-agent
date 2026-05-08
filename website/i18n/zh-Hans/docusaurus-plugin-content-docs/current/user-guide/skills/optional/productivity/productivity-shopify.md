---
title: "Shopify — 通过 curl 调用 Shopify Admin 和 Storefront GraphQL API"
sidebar_label: "Shopify"
description: "通过 curl 调用 Shopify Admin 和 Storefront GraphQL API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Shopify

通过 curl 调用 Shopify Admin 和 Storefront GraphQL API。支持商品、订单、客户、库存、元字段等操作。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/shopify` |
| Path | `optional-skills/productivity/shopify` |
| Version | `1.0.0` |
| Author | community |
| License | MIT |
| Tags | `Shopify`, `E-commerce`, `Commerce`, `API`, `GraphQL` |
| Related skills | [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable), [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Shopify — Admin & Storefront GraphQL API

通过 `curl` 直接与 Shopify 店铺交互：列出商品、管理库存、拉取订单、更新客户、读取元字段。无需 SDK，无需应用框架——只需 GraphQL 端点和自定义应用访问令牌。

REST Admin API 自 2024-04 起已过时，仅接收安全修复。**所有管理操作请使用 GraphQL Admin**。只读的面向客户的查询（商品、集合、购物车）请使用 **Storefront GraphQL**。

## 前置条件

1. 在 Shopify 管理后台：**Settings → Apps and sales channels → Develop apps → Create an app**。
2. 点击 **Configure Admin API scopes**，选择所需权限（示例见下），然后保存。
3. **Install app** → Admin API 访问令牌仅显示一次。请立即复制——Shopify 不会再显示它。令牌以 `shpat_` 开头。
4. 保存到 `~/.hermes/.env`：
   ```
   SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxxxxxxxxx
   SHOPIFY_STORE_DOMAIN=my-store.myshopify.com
   SHOPIFY_API_VERSION=2026-01
   ```

> **注意：** 自 2026 年 1 月 1 日起，在 Shopify 管理后台创建的"旧版自定义应用"已被移除。新配置应使用 **Dev Dashboard**（`shopify.dev/docs/apps/build/dev-dashboard`）。现有管理后台创建的应用继续正常工作。如果用户的店铺没有现有自定义应用且日期在 2026-01-01 之后，请引导他们使用 Dev Dashboard 而非管理后台流程。

按任务分类的常用权限范围：
- 商品 / 集合：`read_products`、`write_products`
- 库存：`read_inventory`、`write_inventory`、`read_locations`
- 订单：`read_orders`、`write_orders`（没有 `read_all_orders` 时仅限最近 30 笔）
- 客户：`read_customers`、`write_customers`
- 草稿订单：`read_draft_orders`、`write_draft_orders`
- 履行：`read_fulfillments`、`write_fulfillments`
- 元字段 / 元对象：由对应资源的权限范围覆盖

## API 基础

- **端点：** `https://$SHOPIFY_STORE_DOMAIN/admin/api/$SHOPIFY_API_VERSION/graphql.json`
- **认证头：** `X-Shopify-Access-Token: $SHOPIFY_ACCESS_TOKEN`（不是 `Authorization: Bearer`）
- **方法：** 始终使用 `POST`，始终使用 `Content-Type: application/json`，请求体为 `{"query": "...", "variables": {...}}`
- **HTTP 200 不代表成功。** GraphQL 在顶层 `errors` 数组和每个字段的 `userErrors` 中返回错误。务必检查两者。
- **ID 是 GID 字符串：** `gid://shopify/Product/10079467700516`、`gid://shopify/Variant/...`、`gid://shopify/Order/...`。请原样传递——不要去掉前缀。
- **速率限制：** 通过查询成本（漏桶算法）计算。每个响应的 `extensions.cost` 包含 `requestedQueryCost`、`actualQueryCost`、`throttleStatus.{currentlyAvailable, maximumAvailable, restoreRate}`。当 `currentlyAvailable` 降至下一次查询成本以下时应退避。标准店铺 = 100 点桶，50/秒恢复；Plus = 1000/100。

基础 curl 模式（可复用）：

```bash
shop_gql() {
  local query="$1"
  local variables="${2:-{}}"
  curl -sS -X POST \
    "https://${SHOPIFY_STORE_DOMAIN}/admin/api/${SHOPIFY_API_VERSION:-2026-01}/graphql.json" \
    -H "Content-Type: application/json" \
    -H "X-Shopify-Access-Token: ${SHOPIFY_ACCESS_TOKEN}" \
    --data "$(jq -nc --arg q "$query" --argjson v "$variables" '{query: $q, variables: $v}')"
}
```

通过 `jq` 管道获取可读输出。`-sS` 保留错误可见但隐藏进度条。

## 探索

### 店铺信息 + 当前 API 版本
```bash
shop_gql '{ shop { name myshopifyDomain primaryDomain { url } currencyCode plan { displayName } } }' | jq
```

### 列出所有支持的 API 版本
```bash
shop_gql '{ publicApiVersions { handle supported } }' | jq '.data.publicApiVersions[] | select(.supported)'
```

## 商品

### 搜索商品（前 20 条匹配结果）
```bash
shop_gql '
query($q: String!) {
  products(first: 20, query: $q) {
    edges { node { id title handle status totalInventory variants(first: 5) { edges { node { id sku price inventoryQuantity } } } } }
    pageInfo { hasNextPage endCursor }
  }
}' '{"q":"hoodie status:active"}' | jq
```

查询语法支持 `title:`、`sku:`、`vendor:`、`product_type:`、`status:active`、`tag:`、`created_at:>2025-01-01`。完整语法：https://shopify.dev/docs/api/usage/search-syntax

### 分页获取商品（游标）
```bash
shop_gql '
query($cursor: String) {
  products(first: 100, after: $cursor) {
    edges { cursor node { id handle } }
    pageInfo { hasNextPage endCursor }
  }
}' '{"cursor":null}'
# 后续调用：传入上一次的 endCursor
```

### 获取商品详情（含变体和元字段）
```bash
shop_gql '
query($id: ID!) {
  product(id: $id) {
    id title handle descriptionHtml tags status
    variants(first: 20) { edges { node { id sku price compareAtPrice inventoryQuantity selectedOptions { name value } } } }
    metafields(first: 20) { edges { node { namespace key type value } } }
  }
}' '{"id":"gid://shopify/Product/10079467700516"}' | jq
```

### 创建商品（含一个变体）
```bash
shop_gql '
mutation($input: ProductCreateInput!) {
  productCreate(product: $input) {
    product { id handle }
    userErrors { field message }
  }
}' '{"input":{"title":"Test Hoodie","status":"DRAFT","vendor":"Hermes","productType":"Apparel","tags":["test"]}}'
```

在较新版本中，变体有独立的 mutation：

```bash
# 创建商品后添加变体
shop_gql '
mutation($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
  productVariantsBulkCreate(productId: $productId, variants: $variants) {
    productVariants { id sku price }
    userErrors { field message }
  }
}' '{"productId":"gid://shopify/Product/...","variants":[{"optionValues":[{"optionName":"Size","name":"M"}],"price":"49.00","inventoryItem":{"sku":"HD-M","tracked":true}}]}'
```

### 更新价格 / SKU
```bash
shop_gql '
mutation($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
  productVariantsBulkUpdate(productId: $productId, variants: $variants) {
    productVariants { id sku price }
    userErrors { field message }
  }
}' '{"productId":"gid://shopify/Product/...","variants":[{"id":"gid://shopify/ProductVariant/...","price":"55.00"}]}'
```

## 订单

### 列出最近的订单（默认最近 30 笔，无 `read_all_orders` 权限时）
```bash
shop_gql '
{
  orders(first: 20, reverse: true, query: "financial_status:paid") {
    edges { node {
      id name createdAt displayFinancialStatus displayFulfillmentStatus
      totalPriceSet { shopMoney { amount currencyCode } }
      customer { id displayName email }
      lineItems(first: 10) { edges { node { title quantity sku } } }
    } }
  }
}' | jq
```

常用订单查询过滤器：`financial_status:paid|pending|refunded`、`fulfillment_status:unfulfilled|fulfilled`、`created_at:>2025-01-01`、`tag:gift`、`email:foo@example.com`。

### 获取单个订单（含收货地址）
```bash
shop_gql '
query($id: ID!) {
  order(id: $id) {
    id name email
    shippingAddress { name address1 address2 city province country zip phone }
    lineItems(first: 50) { edges { node { title quantity variant { sku } originalUnitPriceSet { shopMoney { amount currencyCode } } } } }
    transactions { id kind status amountSet { shopMoney { amount currencyCode } } }
  }
}' '{"id":"gid://shopify/Order/...."}' | jq
```

## 客户

```bash
# 搜索
shop_gql '
{
  customers(first: 10, query: "email:*@example.com") {
    edges { node { id email displayName numberOfOrders amountSpent { amount currencyCode } } }
  }
}'

# 创建
shop_gql '
mutation($input: CustomerInput!) {
  customerCreate(input: $input) {
    customer { id email }
    userErrors { field message }
  }
}' '{"input":{"email":"test@example.com","firstName":"Test","lastName":"User","tags":["api-created"]}}'
```

## 库存

库存存在于绑定到变体的**库存项目**上，数量按**地点**跟踪。

```bash
# 获取某变体在所有地点的库存
shop_gql '
query($id: ID!) {
  productVariant(id: $id) {
    id sku
    inventoryItem {
      id tracked
      inventoryLevels(first: 10) {
        edges { node { location { id name } quantities(names: ["available","on_hand","committed"]) { name quantity } } }
      }
    }
  }
}' '{"id":"gid://shopify/ProductVariant/..."}'
```

调整库存（增量）——使用 `inventoryAdjustQuantities`：

```bash
shop_gql '
mutation($input: InventoryAdjustQuantitiesInput!) {
  inventoryAdjustQuantities(input: $input) {
    inventoryAdjustmentGroup { reason changes { name delta } }
    userErrors { field message }
  }
}' '{
  "input": {
    "reason": "correction",
    "name": "available",
    "changes": [{"delta": 5, "inventoryItemId": "gid://shopify/InventoryItem/...", "locationId": "gid://shopify/Location/..."}]
  }
}'
```

设置绝对库存（非增量）——`inventorySetQuantities`：

```bash
shop_gql '
mutation($input: InventorySetQuantitiesInput!) {
  inventorySetQuantities(input: $input) {
    inventoryAdjustmentGroup { id }
    userErrors { field message }
  }
}' '{"input":{"reason":"correction","name":"available","ignoreCompareQuantity":true,"quantities":[{"inventoryItemId":"gid://shopify/InventoryItem/...","locationId":"gid://shopify/Location/...","quantity":100}]}}'
```

## 元字段和元对象

元字段将自定义数据附加到资源（商品、客户、订单、店铺）。

```bash
# 读取
shop_gql '
query($id: ID!) {
  product(id: $id) {
    metafields(first: 10, namespace: "custom") {
      edges { node { key type value } }
    }
  }
}' '{"id":"gid://shopify/Product/..."}'

# 写入（适用于所有所有者类型）
shop_gql '
mutation($metafields: [MetafieldsSetInput!]!) {
  metafieldsSet(metafields: $metafields) {
    metafields { id key namespace }
    userErrors { field message code }
  }
}' '{"metafields":[{"ownerId":"gid://shopify/Product/...","namespace":"custom","key":"care_instructions","type":"multi_line_text_field","value":"Wash cold. Tumble dry low."}]}'
```

## Storefront API（公开只读）

不同的端点、不同的令牌，用于面向客户的应用/Hydrogen 风格的无头架构。请求头不同：

- **端点：** `https://$SHOPIFY_STORE_DOMAIN/api/$SHOPIFY_API_VERSION/graphql.json`
- **认证头（公开）：** `X-Shopify-Storefront-Access-Token: <公开令牌>` ——可嵌入浏览器
- **认证头（私有）：** `Shopify-Storefront-Private-Token: <私有令牌>` ——仅限服务器端

```bash
curl -sS -X POST \
  "https://${SHOPIFY_STORE_DOMAIN}/api/${SHOPIFY_API_VERSION:-2026-01}/graphql.json" \
  -H "Content-Type: application/json" \
  -H "X-Shopify-Storefront-Access-Token: ${SHOPIFY_STOREFRONT_TOKEN}" \
  -d '{"query":"{ shop { name } products(first: 5) { edges { node { id title handle } } } }"}' | jq
```

## 批量操作

用于导出超出速率限制的数据（完整商品目录、一年内所有订单等）：

```bash
# 1. 启动批量查询
shop_gql '
mutation {
  bulkOperationRunQuery(query: """
    { products { edges { node { id title handle variants { edges { node { sku price } } } } } } }
  """) {
    bulkOperation { id status }
    userErrors { field message }
  }
}'

# 2. 轮询状态
shop_gql '{ currentBulkOperation { id status errorCode objectCount fileSize url partialDataUrl } }'

# 3. 当 status=COMPLETED 时，下载 JSONL 文件
curl -sS "$URL" > products.jsonl
```

每行 JSONL 是一个节点，嵌套连接以单独行发出并带有 `__parentId`。如有需要可在客户端重新组装。

## Webhooks

订阅事件以避免轮询：

```bash
shop_gql '
mutation($topic: WebhookSubscriptionTopic!, $sub: WebhookSubscriptionInput!) {
  webhookSubscriptionCreate(topic: $topic, webhookSubscription: $sub) {
    webhookSubscription { id topic endpoint { __typename ... on WebhookHttpEndpoint { callbackUrl } } }
    userErrors { field message }
  }
}' '{"topic":"ORDERS_CREATE","sub":{"callbackUrl":"https://example.com/webhook","format":"JSON"}}'
```

使用应用的客户端密钥（不是访问令牌）验证传入的 webhook HMAC：

```bash
echo -n "$REQUEST_BODY" | openssl dgst -sha256 -hmac "$APP_SECRET" -binary | base64
# 与 X-Shopify-Hmac-Sha256 头比较
```

## 注意事项

- **REST 端点仍然存在但已冻结。** 请勿基于 `/admin/api/.../products.json` 编写新的集成。请使用 GraphQL。
- **令牌格式检查。** Admin 令牌以 `shpat_` 开头。Storefront 公开令牌以 `shpua_` 开头。如果令牌和请求头不匹配，每个请求都会返回 401 且无有用的错误信息。
- **403 且令牌有效 = 缺少权限范围。** Shopify 返回 `{"errors":[{"message":"Access denied for ..."}]}`。请在应用中重新配置 Admin API 权限范围，然后重新安装以重新生成令牌。
- **`userErrors` 为空不代表成功。** 还需检查 `data.<mutation>.<resource>` 是否非空。某些失败两者都不填充——请检查整个响应。
- **GID 与数字 ID。** 旧版 REST 返回数字 ID；GraphQL 需要完整的 GID 字符串。转换方式：`gid://shopify/Product/<数字ID>`。
- **速率限制陷阱。** 单个 `products(first: 250)` 配合深层嵌套可能消耗 1000+ 点，在标准计划店铺上会立即触发限流。请从小范围开始，读取 `extensions.cost`，然后调整。
- **分页排序。** `products(first: N, reverse: true)` 按 `id DESC` 排序，而非 `created_at`。要按"最新优先"排列，请使用 `sortKey: CREATED_AT, reverse: true`。
- **历史数据需 `read_all_orders`。** 没有此权限时，`orders(...)` 会静默限制在 60 天窗口内。你不会收到错误，只是结果比预期的少。对于拥有大量订单的 Shopify Plus 商家，请通过应用的受保护数据设置请求此权限范围。
- **货币是字符串。** 金额以 `"49.00"` 形式返回，而非 `49.0`。如果需要保留前导零，请勿盲目使用 `jq tonumber`。
- **多币种金额字段** 包含 `shopMoney`（店铺货币）和 `presentmentMoney`（客户货币）。请统一使用其中一种。

## 安全提示

Shopify 中的 mutation 是真实操作——它们会创建商品、处理退款、取消订单、发货履行。在执行 `productDelete`、`orderCancel`、`refundCreate` 或任何批量 mutation 之前：请清楚说明将要更改什么、在哪个店铺上更改，并与用户确认。除非用户有单独的开发店铺，否则没有生产数据的暂存副本。
