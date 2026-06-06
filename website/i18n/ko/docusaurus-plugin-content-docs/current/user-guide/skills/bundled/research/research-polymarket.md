---
title: "Polymarket — Polymarket 쿼리: 시장, 가격, 오더북, 이력"
sidebar_label: "Polymarket"
description: "Polymarket 쿼리: 시장, 가격, 오더북, 이력"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Polymarket

Polymarket 쿼리: 시장, 가격, 오더북, 이력.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/research/polymarket` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Teknium |
| 플랫폼 | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Polymarket — 예측 시장 데이터 (Prediction Market Data)

Polymarket의 공개 REST API를 사용하여 예측 시장 데이터를 쿼리합니다.
모든 엔드포인트는 읽기 전용이며 인증이 전혀 필요하지 않습니다.

curl 예제가 포함된 전체 엔드포인트 참조는 `references/api-endpoints.md`를 확인하세요.

## 사용 시기

- 사용자가 예측 시장, 베팅 배당률 또는 이벤트 확률에 대해 질문할 때
- 사용자가 "X가 일어날 확률은 얼마인가요?"라고 알고 싶어할 때
- 사용자가 구체적으로 Polymarket에 대해 질문할 때
- 사용자가 시장 가격, 오더북 데이터 또는 가격 기록(이력)을 원할 때
- 사용자가 예측 시장 변동을 모니터링하거나 추적하도록 요청할 때

## 주요 개념

- **Events (이벤트)**는 하나 이상의 **Markets (시장)**을 포함합니다 (1:다 관계)
- **Markets (시장)**은 0.00에서 1.00 사이의 Yes/No 가격을 갖는 이진 결과입니다
- 가격은 곧 확률입니다: 가격 0.65는 시장이 65%의 확률로 일어날 것이라 생각함을 의미합니다
- `outcomePrices` 필드: `["0.80", "0.20"]`와 같은 JSON 인코딩된 배열
- `clobTokenIds` 필드: 가격/오더북 쿼리를 위한 두 개의 토큰 ID [Yes, No]의 JSON 인코딩된 배열
- `conditionId` 필드: 가격 기록 쿼리에 사용되는 16진수 문자열
- 거래량(Volume)은 USDC (미국 달러) 기준입니다

## 세 가지 공개 API

1. **Gamma API** (`gamma-api.polymarket.com`) — 검색(Discovery), 찾기, 탐색
2. **CLOB API** (`clob.polymarket.com`) — 실시간 가격, 오더북, 기록
3. **Data API** (`data-api.polymarket.com`) — 거래내역, 미결제약정(open interest)

## 일반적인 워크플로우

사용자가 예측 시장 확률에 대해 질문할 때:

1. **검색 (Search)** 사용자의 쿼리로 Gamma API 공개 검색 엔드포인트를 사용합니다.
2. **파싱 (Parse)** 응답 파싱 — 이벤트와 그에 중첩된 시장을 추출합니다.
3. **제시 (Present)** 시장 질문, 현재 가격을 백분율로, 그리고 거래량을 제시합니다.
4. **심층 분석 (Deep dive)** 요청 시 — 오더북에는 clobTokenIds를, 기록에는 conditionId를 사용합니다.

## 결과 제시

가독성을 위해 가격을 백분율로 형식화합니다:
- outcomePrices `["0.652", "0.348"]`는 "Yes: 65.2%, No: 34.8%"가 됩니다
- 항상 시장 질문과 확률을 표시합니다
- 가능한 경우 거래량을 포함합니다

예시: `"X가 일어날 것인가?" — 65.2% Yes ($1.2M 거래량)`

## 이중 인코딩된 필드 파싱

Gamma API는 JSON 응답 내부에서 `outcomePrices`, `outcomes`, `clobTokenIds`를 JSON 문자열로 반환합니다 (이중 인코딩). Python으로 처리할 때는 실제 배열을 얻기 위해 `json.loads(market['outcomePrices'])`로 파싱하세요.

## 속도 제한 (Rate Limits)

관대한 편이며 — 정상적인 사용 시 도달할 가능성이 낮습니다:
- Gamma: 10초당 4,000 요청 (일반)
- CLOB: 10초당 9,000 요청 (일반)
- Data: 10초당 1,000 요청 (일반)

## 제한 사항

- 이 스킬은 읽기 전용입니다 — 거래 배치는 지원하지 않습니다.
- 거래를 하려면 지갑 기반의 암호화폐 인증(EIP-712 서명)이 필요합니다.
- 일부 새로운 시장은 가격 기록이 비어있을 수 있습니다.
- 거래에는 지리적 제한이 적용되지만 읽기 전용 데이터는 전 세계에서 접근할 수 있습니다.
