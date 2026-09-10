---
title: "Polymarket — Polymarket 조회: 시장, 가격, 오더북, 이력"
sidebar_label: "Polymarket"
description: "Polymarket 조회: 시장, 가격, 오더북, 이력"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Polymarket

Polymarket 조회: 시장, 가격, 오더북, 이력.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/polymarket`로 설치 |
| 경로 | `optional-skills/finance/polymarket` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# Polymarket — 예측 시장 데이터

Polymarket의 공개 REST API를 사용해 예측 시장 데이터를 조회합니다.
모든 엔드포인트는 읽기 전용이며 인증이 전혀 필요하지 않습니다.

전체 엔드포인트 참조와 curl 예시는 `references/api-endpoints.md`를 참조하세요.

## 사용 시점

- 사용자가 예측 시장, 베팅 배당률 또는 사건 발생 확률을 물어볼 때
- "X가 일어날 확률은 얼마인가?"를 알고 싶어 할 때
- Polymarket에 대해 구체적으로 물어볼 때
- 시장 가격, 오더북 데이터 또는 가격 이력을 원할 때
- 예측 시장의 움직임을 모니터링하거나 추적해 달라고 할 때

## 핵심 개념

- **이벤트**에는 하나 이상의 **시장**이 포함됩니다 (1:다 관계).
- **시장**은 0.00에서 1.00 사이의 예/아니오 가격을 가진 이진 결과입니다.
- 가격은 확률입니다: 가격 0.65는 시장이 발생 가능성을 65%로 본다는 뜻입니다.
- `outcomePrices` 필드: `["0.80", "0.20"]`과 같은 JSON 인코딩 배열
- `clobTokenIds` 필드: 가격/오더북 조회에 사용하는 두 토큰 ID [Yes, No]의 JSON 인코딩 배열
- `conditionId` 필드: 가격 이력 조회에 사용하는 16진수 문자열
- 거래량은 USDC(미국 달러) 기준입니다.

## 세 가지 공개 API

1. **Gamma API**(`gamma-api.polymarket.com`) — 검색, 탐색, 브라우징
2. **CLOB API**(`clob.polymarket.com`) — 실시간 가격, 오더북, 이력
3. **Data API**(`data-api.polymarket.com`) — 거래, 미결제약정

## 일반적인 워크플로

사용자가 예측 시장의 확률을 물어보면:

1. Gamma API의 public-search 엔드포인트에서 사용자의 쿼리로 **검색**합니다.
2. 응답을 **분석**하여 이벤트와 그 안에 중첩된 시장을 추출합니다.
3. 시장 질문, 현재 가격(백분율), 거래량을 **제시**합니다.
4. 더 자세한 내용을 요청하면 **심층 분석**합니다 — 오더북에는 `clobTokenIds`를, 이력에는 `conditionId`를 사용합니다.

## 결과 제시

읽기 쉽도록 가격을 백분율로 표시합니다:
- `outcomePrices ["0.652", "0.348"]`은 "Yes: 65.2%, No: 34.8%"가 됩니다.
- 항상 시장 질문과 확률을 표시합니다.
- 가능한 경우 거래량을 포함합니다.

예: `"Will X happen?" — 65.2% Yes ($1.2M volume)`

## 이중 인코딩 필드 분석

Gamma API는 `outcomePrices`, `outcomes`, `clobTokenIds`를 JSON 응답 내부의 JSON 문자열로 반환합니다
(이중 인코딩). Python으로 처리할 때는 `json.loads(market['outcomePrices'])`로 분석하여 실제 배열을 얻습니다.

## 요청 제한

여유가 있어 일반적인 사용에서는 한도에 도달할 가능성이 낮습니다:
- Gamma: 10초당 4,000개 요청 (일반)
- CLOB: 10초당 9,000개 요청 (일반)
- Data: 10초당 1,000개 요청 (일반)

## 제한 사항

- 이 스킬은 읽기 전용이며 거래 실행을 지원하지 않습니다.
- 거래에는 지갑 기반 암호화폐 인증(EIP-712 서명)이 필요합니다.
- 일부 신규 시장에는 가격 이력이 비어 있을 수 있습니다.
- 거래에는 지역 제한이 적용되지만 읽기 전용 데이터는 전 세계에서 이용할 수 있습니다.
