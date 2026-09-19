---
title: "Shopify — curl로 Shopify Admin/Storefront GraphQL API 쿼리하기"
sidebar_label: "Shopify"
description: "curl로 Shopify Admin/Storefront GraphQL API 쿼리하기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Shopify

curl을 통해 Shopify 스토어를 직접 쿼리합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/productivity/shopify`로 설치 |
| 경로 | `optional-skills/productivity/shopify` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Shopify`, `E-commerce`, `Commerce`, `API`, `GraphQL` |
| 관련 스킬 | [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable), [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침입니다.
:::

# Shopify — Admin 및 Storefront GraphQL API

`curl`을 통해 Shopify 스토어에서 직접 작업합니다. 제품 목록 조회, 재고 관리, 주문 가져오기, 고객 업데이트, 메타필드 읽기를 수행할 수 있습니다. SDK도 앱 프레임워크도 필요 없이 GraphQL 엔드포인트와 커스텀 앱 액세스 토큰만 사용합니다.

REST Admin API는 2024-04부터 레거시 상태이며 보안 수정만 받습니다. 모든 관리자 작업에는 **GraphQL Admin**을 사용합니다. 고객에게 표시되는 읽기 전용 쿼리(제품, 컬렉션, 장바구니)에는 **Storefront GraphQL**을 사용합니다.

## 사전 요구 사항

1. Shopify 관리자에서 **Settings → Apps and sales channels → Develop apps → Create an app**으로 이동합니다.
2. **Configure Admin API scopes**를 클릭하고 필요한 권한을 선택한 다음(아래 예시 참고) 저장합니다.
3. **Install app**을 클릭하면 Admin API 액세스 토큰이 한 번만 표시됩니다. 즉시 복사하세요 — Shopify는 토큰을 다시 표시하지 않습니다. 토큰은 `shpat_`로 시작합니다.
4. `${HERMES_HOME:-~/.hermes}/.env`에 저장합니다:
   ```
   SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxxxxxxxxx
   SHOPIFY_STORE_DOMAIN=my-store.myshopify.com
   SHOPIFY_API_VERSION=2026-01
   ```

> **참고:** 2026년 1월 1일 기준으로 Shopify 관리자에서 새로 생성하는 "legacy custom apps"는 사라졌습니다. 새 설정에서는 **Dev Dashboard**(`shopify.dev/docs/apps/build/dev-dashboard`)를 사용해야 합니다. 기존에 관리자에서 생성한 앱은 계속 작동합니다. 사용자의 스토어에 기존 커스텀 앱이 없고 날짜가 2026-01-01 이후라면 관리자 흐름 대신 Dev Dashboard로 안내하세요.

작업별 일반 권한:
- 제품 / 컬렉션: `read_products`, `write_products`
- 재고: `read_inventory`, `write_inventory`, `read_locations`
- 주문: `read_orders`, `write_orders` (`read_all_orders`가 없으면 가장 최근 30개)
- 고객: `read_customers`, `write_customers`
- 임시 주문: `read_draft_orders`, `write_draft_orders`
- 주문 처리: `read_fulfillments`, `write_fulfillments`
- 메타필드 / 메타오브젝트: 일치하는 리소스 권한으로 처리

## API 기본 사항

- **엔드포인트:** `https://$SHOPIFY_STORE_DOMAIN/admin/api/$SHOPIFY_API_VERSION/graphql.json`
- **인증 헤더:** `X-Shopify-Access-Token: $SHOPIFY_ACCESS_TOKEN` (`Authorization: Bearer`가 아님)
- **메서드:** 항상 `POST`, 항상 `Content-Type: application/json`, 본문은 `{"query": "...", "variables": {...}}`
- **HTTP 200이 성공을 의미하지는 않습니다.** GraphQL은 최상위 `errors` 배열과 필드별 `userErrors`에 오류를 반환합니다. 항상 둘 다 확인하세요.
- **ID는 GID 문자열입니다:** `gid://shopify/Product/10079467700516`, `gid://shopify/Variant/...`, `gid://shopify/Order/...`. 그대로 전달하고 접두사를 제거하지 마세요.
- **속도 제한:** 쿼리 비용(누수 버킷)으로 계산됩니다. 각 응답의 `extensions.cost`에 `requestedQueryCost`, `actualQueryCost`, `throttleStatus.{currentlyAvailable, maximumAvailable, restoreRate}`가 있습니다. `currentlyAvailable`이 다음 쿼리 비용보다 낮아지면 대기하세요. 일반 스토어 = 100포인트 버킷, 초당 50포인트 복구; Plus = 1000/100.

기본 curl 패턴(재사용 가능):

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

읽기 쉬운 출력을 위해 `jq`로 파이프하세요. `-sS`는 오류를 표시하면서 진행률 표시줄은 숨깁니다.

## 탐색

### 스토어 정보 + 현재 API 버전
```bash
shop_gql '{ shop { name myshopifyDomain primaryDomain { url } currencyCode plan { displayName } } }' | jq
```

### 지원되는 모든 API 버전 목록
```bash
shop_gql '{ publicApiVersions { handle supported } }' | jq '.data.publicApiVersions[] | select(.supported)'
```

## 제품

### 제품 검색(쿼리와 일치하는 처음 20개)
```bash
shop_gql '
query($q: String!) {
  products(first: 20, query: $q) {
    edges { node { id title handle status totalInventory variants(first: 5) { edges { node { id sku price inventoryQuantity } } } } }
    pageInfo { hasNextPage endCursor }
  }
}' '{"q":"hoodie status:active"}' | jq
```

쿼리 문법은 `title:`, `sku:`, `vendor:`, `product_type:`, `status:active`, `tag:`, `created_at:>2025-01-01`을 지원합니다. 전체 문법: https://shopify.dev/docs/api/usage/search-syntax

### 제품 페이지 매김(cursor)
```bash
shop_gql '
query($cursor: String) {
  products(first: 100, after: $cursor) {
    edges { cursor node { id handle } }
    pageInfo { hasNextPage endCursor }
  }
}' '{"cursor":null}'
# subsequent calls: pass the previous endCursor
```

### 변형 상품 및 메타필드가 포함된 제품 가져오기
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

### 변형 상품 하나가 포함된 제품 생성
```bash
shop_gql '
mutation($input: ProductCreateInput!) {
  productCreate(product: $input) {
    product { id handle }
    userErrors { field message }
  }
}' '{"input":{"title":"Test Hoodie","status":"DRAFT","vendor":"Hermes","productType":"Apparel","tags":["test"]}}'
```

최근 버전에서는 변형 상품에 자체 mutation이 있습니다:

```bash
# Add variants after creating the product
shop_gql '
mutation($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
  productVariantsBulkCreate(productId: $productId, variants: $variants) {
    productVariants { id sku price }
    userErrors { field message }
  }
}' '{"productId":"gid://shopify/Product/...","variants":[{"optionValues":[{"optionName":"Size","name":"M"}],"price":"49.00","inventoryItem":{"sku":"HD-M","tracked":true}}]}'
```

### 가격 / SKU 업데이트
```bash
shop_gql '
mutation($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
  productVariantsBulkUpdate(productId: $productId, variants: $variants) {
    productVariants { id sku price }
    userErrors { field message }
  }
}' '{"productId":"gid://shopify/Product/...","variants":[{"id":"gid://shopify/ProductVariant/...","price":"55.00"}]}'
```

## 주문

### 최근 주문 목록(기본적으로 `read_all_orders` 없이 최근 30개)
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

유용한 주문 쿼리 필터: `financial_status:paid|pending|refunded`, `fulfillment_status:unfulfilled|fulfilled`, `created_at:>2025-01-01`, `tag:gift`, `email:foo@example.com`.

### 배송 주소가 포함된 단일 주문 가져오기
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

## 고객

```bash
# Search
shop_gql '
{
  customers(first: 10, query: "email:*@example.com") {
    edges { node { id email displayName numberOfOrders amountSpent { amount currencyCode } } }
  }
}'

# Create
shop_gql '
mutation($input: CustomerInput!) {
  customerCreate(input: $input) {
    customer { id email }
    userErrors { field message }
  }
}' '{"input":{"email":"test@example.com","firstName":"Test","lastName":"User","tags":["api-created"]}}'
```

## 재고

재고는 변형 상품에 연결되고 **location**별로 수량이 추적되는 **inventory item**에 저장됩니다.

```bash
# Get inventory for a variant across all locations
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

재고 조정(delta) — `inventoryAdjustQuantities`를 사용합니다:

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

절대 재고 설정(delta 아님) — `inventorySetQuantities`:

```bash
shop_gql '
mutation($input: InventorySetQuantitiesInput!) {
  inventorySetQuantities(input: $input) {
    inventoryAdjustmentGroup { id }
    userErrors { field message }
  }
}' '{"input":{"reason":"correction","name":"available","ignoreCompareQuantity":true,"quantities":[{"inventoryItemId":"gid://shopify/InventoryItem/...","locationId":"gid://shopify/Location/...","quantity":100}]}}'
```

## 메타필드 및 메타오브젝트

메타필드는 리소스(제품, 고객, 주문, 스토어)에 커스텀 데이터를 연결합니다.

```bash
# Read
shop_gql '
query($id: ID!) {
  product(id: $id) {
    metafields(first: 10, namespace: "custom") {
      edges { node { key type value } }
    }
  }
}' '{"id":"gid://shopify/Product/..."}'

# Write (works for any owner type)
shop_gql '
mutation($metafields: [MetafieldsSetInput!]!) {
  metafieldsSet(metafields: $metafields) {
    metafields { id key namespace }
    userErrors { field message code }
  }
}' '{"metafields":[{"ownerId":"gid://shopify/Product/...","namespace":"custom","key":"care_instructions","type":"multi_line_text_field","value":"Wash cold. Tumble dry low."}]}'
```

## Storefront API(공개 읽기 전용)

엔드포인트도 토큰도 다르며 고객 대상 앱과 Hydrogen 스타일의 헤드리스 설정에 사용됩니다. 헤더도 다릅니다:

- **엔드포인트:** `https://$SHOPIFY_STORE_DOMAIN/api/$SHOPIFY_API_VERSION/graphql.json`
- **인증 헤더(공개):** `X-Shopify-Storefront-Access-Token: <public token>` — 브라우저에 삽입 가능
- **인증 헤더(비공개):** `Shopify-Storefront-Private-Token: <private token>` — 서버 전용

```bash
curl -sS -X POST \
  "https://${SHOPIFY_STORE_DOMAIN}/api/${SHOPIFY_API_VERSION:-2026-01}/graphql.json" \
  -H "Content-Type: application/json" \
  -H "X-Shopify-Storefront-Access-Token: ${SHOPIFY_STOREFRONT_TOKEN}" \
  -d '{"query":"{ shop { name } products(first: 5) { edges { node { id title handle } } } }"}' | jq
```

## 대량 작업

속도 제한으로 허용되는 범위를 넘는 대규모 덤프(전체 제품 카탈로그, 1년 치 모든 주문)의 경우:

```bash
# 1. Start bulk query
shop_gql '
mutation {
  bulkOperationRunQuery(query: """
    { products { edges { node { id title handle variants { edges { node { sku price } } } } } } }
  """) {
    bulkOperation { id status }
    userErrors { field message }
  }
}'

# 2. Poll status
shop_gql '{ currentBulkOperation { id status errorCode objectCount fileSize url partialDataUrl } }'

# 3. When status=COMPLETED, download the JSONL file
curl -sS "$URL" > products.jsonl
```

각 JSONL 줄은 하나의 노드이며, 중첩된 연결은 `__parentId`가 포함된 별도 줄로 출력됩니다. 필요하면 클라이언트 측에서 재조립하세요.

## 웹훅

폴링하지 않아도 되도록 이벤트를 구독합니다:

```bash
shop_gql '
mutation($topic: WebhookSubscriptionTopic!, $sub: WebhookSubscriptionInput!) {
  webhookSubscriptionCreate(topic: $topic, webhookSubscription: $sub) {
    webhookSubscription { id topic endpoint { __typename ... on WebhookHttpEndpoint { callbackUrl } } }
    userErrors { field message }
  }
}' '{"topic":"ORDERS_CREATE","sub":{"callbackUrl":"https://example.com/webhook","format":"JSON"}}'
```

액세스 토큰이 아닌 앱의 클라이언트 시크릿으로 들어오는 웹훅 HMAC을 검증합니다:

```bash
echo -n "$REQUEST_BODY" | openssl dgst -sha256 -hmac "$APP_SECRET" -binary | base64
# Compare to X-Shopify-Hmac-Sha256 header
```

## 주의 사항

- **REST 엔드포인트는 여전히 존재하지만 동결되었습니다.** `/admin/api/.../products.json`에 새 통합을 작성하지 마세요. GraphQL을 사용하세요.
- **토큰 형식 확인.** Admin 토큰은 `shpat_`로 시작합니다. Storefront 공개 토큰은 `shpua_`로 시작합니다. 토큰을 하나 가지고 잘못된 헤더를 사용하면 모든 요청이 유용한 오류 본문 없이 401을 반환합니다.
- **유효한 토큰인데 403 = 권한 누락.** Shopify는 `{"errors":[{"message":"Access denied for ..."}]}`를 반환합니다. 앱에서 Admin API 권한을 다시 구성한 다음 재설치하여 토큰을 새로 생성하세요.
- **`userErrors`가 비어 있음 != 성공.** `data.<mutation>.<resource>`도 null이 아닌지 확인하세요. 일부 실패는 어느 쪽에도 값을 채우지 않으므로 전체 응답을 확인하세요.
- **GID와 숫자 ID.** 레거시 REST는 숫자 ID를 반환했지만 GraphQL은 전체 GID 문자열을 요구합니다. 변환 방법: `gid://shopify/Product/<numeric>`.
- **예상 밖의 속도 제한.** 깊게 중첩된 `products(first: 250)` 하나는 1000포인트 이상을 사용할 수 있어 일반 요금제 스토어에서 즉시 제한될 수 있습니다. 좁게 시작하고 `extensions.cost`를 읽어 조정하세요.
- **페이지 매김 순서.** `products(first: N, reverse: true)`는 `created_at`이 아니라 `id DESC`로 정렬합니다. "최신순"에는 `sortKey: CREATED_AT, reverse: true`를 사용하세요.
- **과거 데이터를 위한 `read_all_orders`.** 이 권한이 없으면 `orders(...)`가 60일 기간으로 조용히 제한됩니다. 오류는 발생하지 않고 예상보다 적은 결과만 반환됩니다. 주문이 많은 Shopify Plus 판매자는 앱의 보호 데이터 설정을 통해 이 권한을 요청하세요.
- **통화는 문자열입니다.** 금액은 `49.0`이 아니라 `"49.00"`으로 반환됩니다. 0 채우기를 중요하게 여긴다면 `jq tonumber`를 무조건 사용하지 마세요.
- **다중 통화 Money 필드**에는 `shopMoney`(스토어 통화)와 `presentmentMoney`(고객 통화)가 모두 있습니다. 하나를 일관되게 선택하세요.

## 안전

Shopify의 mutation은 실제 작업입니다 — 제품을 생성하고, 환불을 청구하고, 주문을 취소하고, 주문 처리를 배송합니다. `productDelete`, `orderCancel`, `refundCreate` 또는 대량 mutation을 실행하기 전에 변경 내용과 대상 스토어를 명확히 설명하고 사용자에게 확인받으세요. 별도의 개발 스토어가 없는 한 프로덕션 데이터의 스테이징 복제본은 없습니다.
