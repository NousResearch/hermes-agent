---
title: "Shopify — curl을 사용한 Shopify Admin & Storefront GraphQL API"
sidebar_label: "Shopify"
description: "curl을 사용한 Shopify Admin & Storefront GraphQL API"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Shopify

curl을 사용한 Shopify Admin & Storefront GraphQL API. 제품, 주문, 고객, 재고, 메타필드.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택적(Optional) — `hermes skills install official/productivity/shopify` 명령어로 설치 |
| 경로 | `optional-skills/productivity/shopify` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Shopify`, `E-commerce`, `Commerce`, `API`, `GraphQL` |
| 관련 스킬 | [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable), [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되어 있을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Shopify — Admin & Storefront GraphQL API

`curl`을 사용하여 Shopify 상점과 직접 작업합니다: 제품 나열, 재고 관리, 주문 가져오기, 고객 업데이트, 메타필드 읽기 등. SDK나 앱 프레임워크 없이 GraphQL 엔드포인트와 사용자 지정 앱 액세스 토큰만 사용합니다.

REST Admin API는 2024년 4월부터 레거시가 되었으며 보안 수정 사항만 제공됩니다. **모든 관리 작업에는 GraphQL Admin을 사용하세요**. 고객을 대면하는 읽기 전용 쿼리(제품, 컬렉션, 장바구니)에는 **Storefront GraphQL**을 사용하세요.

## 전제 조건

1. Shopify 관리자 화면에서: **Settings(설정) → Apps and sales channels(앱 및 판매 채널) → Develop apps(앱 개발) → Create an app(앱 만들기)**.
2. **Configure Admin API scopes(Admin API 범위 구성)**를 클릭하고, 필요한 항목(아래 예시 참조)을 선택한 다음 저장합니다.
3. **Install app(앱 설치)** → Admin API 액세스 토큰이 **한 번만** 표시됩니다. 즉시 복사하세요 — Shopify는 이를 다시 보여주지 않습니다. 토큰은 `shpat_`로 시작합니다.
4. `~/.hermes/.env` 파일에 저장합니다:
   ```
   SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxxxxxxxxx
   SHOPIFY_STORE_DOMAIN=my-store.myshopify.com
   SHOPIFY_API_VERSION=2026-01
   ```

> **주의 사항:** 2026년 1월 1일부로 Shopify 관리자 화면에서 생성되는 새로운 "레거시 맞춤형 앱"은 사라졌습니다. 새로운 설정은 **Dev Dashboard(개발 대시보드)** (`shopify.dev/docs/apps/build/dev-dashboard`)를 사용해야 합니다. 기존에 관리자 화면에서 생성된 앱은 계속 작동합니다. 만약 2026년 1월 1일 이후에 사용자의 상점에 기존 사용자 지정 앱이 없다면, 관리자 흐름 대신 Dev Dashboard로 안내하세요.

작업별 일반적인 범위(Scopes):
- 제품 / 컬렉션: `read_products`, `write_products`
- 재고: `read_inventory`, `write_inventory`, `read_locations`
- 주문: `read_orders`, `write_orders` (`read_all_orders`가 없는 경우 최근 30개)
- 고객: `read_customers`, `write_customers`
- 초안 주문: `read_draft_orders`, `write_draft_orders`
- 주문 이행: `read_fulfillments`, `write_fulfillments`
- 메타필드 / 메타오브젝트: 일치하는 리소스 범위에 포함됨

## API 기초

- **엔드포인트:** `https://$SHOPIFY_STORE_DOMAIN/admin/api/$SHOPIFY_API_VERSION/graphql.json`
- **인증 헤더:** `X-Shopify-Access-Token: $SHOPIFY_ACCESS_TOKEN` (`Authorization: Bearer`가 **아님**)
- **메서드:** 항상 `POST`, 항상 `Content-Type: application/json`, 본문은 `{"query": "...", "variables": {...}}`
- **HTTP 200이 성공을 의미하지는 않습니다.** GraphQL은 오류를 최상위 `errors` 배열과 필드별 `userErrors`로 반환합니다. 항상 두 가지를 모두 확인하세요.
- **ID는 GID 문자열입니다:** `gid://shopify/Product/10079467700516`, `gid://shopify/Variant/...`, `gid://shopify/Order/...`. 접두사를 제거하지 말고 그대로 전달하세요.
- **비율 제한(Rate limit):** 쿼리 비용(Leaky bucket 방식)을 통해 계산됩니다. 각 응답에는 `requestedQueryCost`, `actualQueryCost`, `throttleStatus.{currentlyAvailable, maximumAvailable, restoreRate}`와 함께 `extensions.cost`가 포함됩니다. `currentlyAvailable`이 다음 쿼리의 비용보다 낮아지면 대기하세요. 표준 상점 = 버킷 100포인트, 초당 50 복구; Plus = 1000/100.

기본 curl 패턴 (재사용 가능):

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

읽기 쉬운 출력을 위해 `jq`를 파이프합니다. `-sS`는 진행률 표시줄은 숨기되 오류는 계속 표시합니다.

## 디스커버리 (Discovery)

### 상점 정보 + 현재 API 버전
```bash
shop_gql '{ shop { name myshopifyDomain primaryDomain { url } currencyCode plan { displayName } } }' | jq
```

### 지원되는 모든 API 버전 목록
```bash
shop_gql '{ publicApiVersions { handle supported } }' | jq '.data.publicApiVersions[] | select(.supported)'
```

## 제품 (Products)

### 제품 검색 (쿼리와 일치하는 첫 20개)
```bash
shop_gql '
query($q: String!) {
  products(first: 20, query: $q) {
    edges { node { id title handle status totalInventory variants(first: 5) { edges { node { id sku price inventoryQuantity } } } } }
    pageInfo { hasNextPage endCursor }
  }
}' '{"q":"hoodie status:active"}' | jq
```

쿼리 구문은 `title:`, `sku:`, `vendor:`, `product_type:`, `status:active`, `tag:`, `created_at:>2025-01-01`를 지원합니다. 전체 문법: https://shopify.dev/docs/api/usage/search-syntax

### 제품 페이지네이션 (Cursor)
```bash
shop_gql '
query($cursor: String) {
  products(first: 100, after: $cursor) {
    edges { cursor node { id handle } }
    pageInfo { hasNextPage endCursor }
  }
}' '{"cursor":null}'
# 후속 호출: 이전 endCursor를 전달합니다
```

### 변형 + 메타필드와 함께 제품 가져오기
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

### 하나의 변형이 있는 제품 생성하기
```bash
shop_gql '
mutation($input: ProductCreateInput!) {
  productCreate(product: $input) {
    product { id handle }
    userErrors { field message }
  }
}' '{"input":{"title":"Test Hoodie","status":"DRAFT","vendor":"Hermes","productType":"Apparel","tags":["test"]}}'
```

최신 버전에서는 변형(Variants)이 고유한 뮤테이션을 갖습니다:

```bash
# 제품 생성 후 변형 추가
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

## 주문 (Orders)

### 최근 주문 나열 (`read_all_orders`가 없는 경우 기본적으로 마지막 30개)
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

### 배송 주소를 포함한 단일 주문 가져오기
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

## 고객 (Customers)

```bash
# 검색
shop_gql '
{
  customers(first: 10, query: "email:*@example.com") {
    edges { node { id email displayName numberOfOrders amountSpent { amount currencyCode } } }
  }
}'

# 생성
shop_gql '
mutation($input: CustomerInput!) {
  customerCreate(input: $input) {
    customer { id email }
    userErrors { field message }
  }
}' '{"input":{"email":"test@example.com","firstName":"Test","lastName":"User","tags":["api-created"]}}'
```

## 재고 (Inventory)

재고는 변형에 연결된 **인재고 항목(inventory items)**에 존재하며, 수량은 **위치(location)**별로 추적됩니다.

```bash
# 모든 위치에 걸쳐 변형에 대한 재고 가져오기
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

재고 조정 (증감/delta) — `inventoryAdjustQuantities` 사용:

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

절대 재고 설정 (증감이 아님) — `inventorySetQuantities`:

```bash
shop_gql '
mutation($input: InventorySetQuantitiesInput!) {
  inventorySetQuantities(input: $input) {
    inventoryAdjustmentGroup { id }
    userErrors { field message }
  }
}' '{"input":{"reason":"correction","name":"available","ignoreCompareQuantity":true,"quantities":[{"inventoryItemId":"gid://shopify/InventoryItem/...","locationId":"gid://shopify/Location/...","quantity":100}]}}'
```

## 메타필드 및 메타오브젝트 (Metafields & Metaobjects)

메타필드는 리소스(제품, 고객, 주문, 상점)에 사용자 지정 데이터를 첨부합니다.

```bash
# 읽기
shop_gql '
query($id: ID!) {
  product(id: $id) {
    metafields(first: 10, namespace: "custom") {
      edges { node { key type value } }
    }
  }
}' '{"id":"gid://shopify/Product/..."}'

# 쓰기 (모든 소유자 유형에서 작동함)
shop_gql '
mutation($metafields: [MetafieldsSetInput!]!) {
  metafieldsSet(metafields: $metafields) {
    metafields { id key namespace }
    userErrors { field message code }
  }
}' '{"metafields":[{"ownerId":"gid://shopify/Product/...","namespace":"custom","key":"care_instructions","type":"multi_line_text_field","value":"Wash cold. Tumble dry low."}]}'
```

## Storefront API (공개 읽기 전용)

다른 엔드포인트, 다른 토큰이며, 고객 대면 앱/Hydrogen 스타일의 헤드리스(headless) 설정에 사용됩니다. 헤더가 다릅니다:

- **엔드포인트:** `https://$SHOPIFY_STORE_DOMAIN/api/$SHOPIFY_API_VERSION/graphql.json`
- **인증 헤더 (공개):** `X-Shopify-Storefront-Access-Token: <public token>` — 브라우저에 임베드 가능
- **인증 헤더 (비공개):** `Shopify-Storefront-Private-Token: <private token>` — 서버 전용

```bash
curl -sS -X POST \
  "https://${SHOPIFY_STORE_DOMAIN}/api/${SHOPIFY_API_VERSION:-2026-01}/graphql.json" \
  -H "Content-Type: application/json" \
  -H "X-Shopify-Storefront-Access-Token: ${SHOPIFY_STOREFRONT_TOKEN}" \
  -d '{"query":"{ shop { name } products(first: 5) { edges { node { id title handle } } } }"}' | jq
```

## 일괄 작업 (Bulk Operations)

비율 제한이 허용하는 것보다 큰 덤프(전체 제품 카탈로그, 1년 치의 모든 주문)의 경우:

```bash
# 1. 일괄 쿼리 시작
shop_gql '
mutation {
  bulkOperationRunQuery(query: """
    { products { edges { node { id title handle variants { edges { node { sku price } } } } } } }
  """) {
    bulkOperation { id status }
    userErrors { field message }
  }
}'

# 2. 상태 폴링
shop_gql '{ currentBulkOperation { id status errorCode objectCount fileSize url partialDataUrl } }'

# 3. status=COMPLETED 일 때 JSONL 파일 다운로드
curl -sS "$URL" > products.jsonl
```

각 JSONL 줄은 노드이며 중첩된 연결(connections)은 `__parentId`가 있는 별도의 줄로 내보내집니다. 필요한 경우 클라이언트 측에서 다시 조합하세요.

## 웹훅 (Webhooks)

폴링하지 않도록 이벤트 구독하기:

```bash
shop_gql '
mutation($topic: WebhookSubscriptionTopic!, $sub: WebhookSubscriptionInput!) {
  webhookSubscriptionCreate(topic: $topic, webhookSubscription: $sub) {
    webhookSubscription { id topic endpoint { __typename ... on WebhookHttpEndpoint { callbackUrl } } }
    userErrors { field message }
  }
}' '{"topic":"ORDERS_CREATE","sub":{"callbackUrl":"https://example.com/webhook","format":"JSON"}}'
```

앱의 클라이언트 시크릿(액세스 토큰이 아님)을 사용하여 들어오는 웹훅의 HMAC 확인:

```bash
echo -n "$REQUEST_BODY" | openssl dgst -sha256 -hmac "$APP_SECRET" -binary | base64
# X-Shopify-Hmac-Sha256 헤더와 비교
```

## 주의 사항 (Pitfalls)

- **REST 엔드포인트는 여전히 존재하지만 동결되었습니다.** `/admin/api/.../products.json`에 대한 새 통합을 작성하지 마세요. GraphQL을 사용하세요.
- **토큰 형식 확인.** Admin 토큰은 `shpat_`로 시작합니다. Storefront 공개 토큰은 `shpua_`로 시작합니다. 올바른 토큰을 가졌는데 잘못된 헤더를 사용하면 모든 요청이 유용한 오류 본문 없이 401을 반환합니다.
- **유효한 토큰인데 403 반환 = 범위(scope) 누락.** Shopify는 `{"errors":[{"message":"Access denied for ..."}]}`를 반환합니다. 앱에서 Admin API 범위를 재구성한 후 재설치하여 토큰을 다시 생성하세요.
- **`userErrors`가 비어있다고 해서 성공이 아닙니다.** `data.<mutation>.<resource>`가 null이 아닌지도 확인하세요. 일부 실패는 어느 것도 채우지 않습니다 — 전체 응답을 검사하세요.
- **GID 대 숫자 ID.** 레거시 REST는 숫자 ID를 제공했지만, GraphQL은 완전한 GID 문자열을 요구합니다. 변환하려면: `gid://shopify/Product/<numeric>`.
- **예상치 못한 비율 제한.** 중첩이 깊은 단일 `products(first: 250)` 쿼리는 1000포인트 이상의 비용이 발생하여 표준 플랜 상점에서 즉시 제한될 수 있습니다. 좁은 범위에서 시작하여 `extensions.cost`를 읽고 조정하세요.
- **페이지네이션 순서.** `products(first: N, reverse: true)`는 `created_at`이 아니라 `id DESC` 기준으로 정렬합니다. "최신순"을 원한다면 `sortKey: CREATED_AT, reverse: true`를 사용하세요.
- **과거 데이터를 위한 `read_all_orders`.** 이것이 없으면 `orders(...)`는 자동으로 60일 기간으로 제한됩니다. 오류는 발생하지 않고 예상보다 적은 결과가 나타날 뿐입니다. 주문이 많은 Shopify Plus 판매자의 경우, 앱의 보호된 데이터(protected-data) 설정을 통해 이 범위를 요청하세요.
- **통화는 문자열입니다.** 금액은 `49.0`이 아닌 `"49.00"`으로 반환됩니다. 0으로 채워진 문자열이 유지되길 원한다면 무작정 `jq tonumber`를 하지 마세요.
- **다중 통화 머니 필드**에는 `shopMoney`(상점 통화)와 `presentmentMoney`(고객 통화)가 있습니다. 일관되게 하나를 선택하세요.

## 안전 (Safety)

Shopify에서의 뮤테이션(Mutations)은 실제입니다 — 제품을 생성하고, 환불을 청구하고, 주문을 취소하고, 주문 이행을 발송합니다. `productDelete`, `orderCancel`, `refundCreate` 또는 일괄 뮤테이션을 실행하기 전에: 변경 사항이 무엇인지, 어느 상점인지 명확히 밝히고 사용자와 확인하십시오. 사용자가 별도의 개발용 상점을 가지고 있지 않는 한 프로덕션 데이터의 스테이징 클론은 없습니다.
