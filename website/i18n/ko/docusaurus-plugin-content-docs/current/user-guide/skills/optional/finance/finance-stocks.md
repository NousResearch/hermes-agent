---
title: "주식 — Yahoo를 통한 주가, 이력, 검색, 비교, 암호화폐"
sidebar_label: "주식"
description: "Yahoo를 통한 주가, 이력, 검색, 비교, 암호화폐"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 주식

Yahoo를 통한 주가, 이력, 검색, 비교, 암호화폐.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/finance/stocks`로 설치 |
| 경로 | `optional-skills/finance/stocks` |
| 버전 | `0.1.0` |
| 작성자 | Mibay (Mibayy), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Stocks`, `Finance`, `Market`, `Crypto`, `Investing` |
| 관련 스킬 | [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis), [`lbo-model`](/docs/user-guide/skills/optional/finance/finance-lbo-model) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# 주식 스킬

Yahoo Finance를 통한 읽기 전용 시장 데이터입니다. 명령은 `quote`, `search`,
`history`, `compare`, `crypto`의 다섯 가지입니다. Python 표준 라이브러리만 사용하므로 API 키나 pip
설치가 필요하지 않습니다. Yahoo의 엔드포인트는 비공식이며 요청 제한이 적용되거나 변경될 수 있습니다.

## 사용 시점

- 사용자가 현재 주가를 물어볼 때 (AAPL, TSLA, MSFT, ...)
- 회사 이름으로 티커를 조회하려 할 때
- 특정 날짜 범위의 OHLCV 이력 또는 성과를 원할 때
- 여러 티커를 나란히 비교하려 할 때
- 암호화폐 가격을 물어볼 때 (BTC, ETH, SOL, ...)

## 사전 요구 사항

Python 3.8 이상, 표준 라이브러리만 사용합니다. 선택 사항: `market_cap`, `pe_ratio`, 52주 수준을 보강하려면
`ALPHA_VANTAGE_KEY`를 설정합니다. Yahoo의 crumb 보호 필드가 null로 반환될 때 유용합니다.
무료 키: https://www.alphavantage.co/support/#api-key

## 실행 방법

`terminal` 도구를 통해 호출합니다. 설치가 완료되면:

```
SCRIPT=~/.hermes/skills/finance/stocks/scripts/stocks_client.py
python3 $SCRIPT quote AAPL
```

모든 출력은 stdout의 JSON입니다 — 필요한 부분만 추출하려면 `jq`로 파이프하세요.

## 빠른 참조

```
python3 $SCRIPT quote AAPL
python3 $SCRIPT quote AAPL MSFT GOOGL TSLA
python3 $SCRIPT search "Tesla"
python3 $SCRIPT history NVDA --range 6mo
python3 $SCRIPT compare AAPL MSFT GOOGL
python3 $SCRIPT crypto BTC ETH SOL
```

## 명령

### `quote SYMBOL [SYMBOL2 ...]`

현재 가격, 변동, 변동률, 거래량, 52주 최고가/최저가를 표시합니다.

### `search QUERY`

회사 이름으로 티커를 찾습니다. 상위 5개 결과를 반환합니다: 기호, 이름, 거래소, 유형.

### `history SYMBOL [--range RANGE]`

일별 OHLCV와 통계를 표시합니다 (최솟값, 최댓값, 평균, 총수익률 %). 범위: `1mo`,
`3mo`, `6mo`, `1y`, `5y`. 기본값: `1mo`.

### `compare SYMBOL1 SYMBOL2 [...]`

나란히 비교합니다: 가격, 변동률, 52주 성과.

### `crypto SYMBOL [SYMBOL2 ...]`

암호화폐 가격입니다. `BTC`를 전달하면 스크립트가 자동으로 `-USD`를 추가합니다.

## 주의 사항

- Yahoo Finance의 API는 비공식입니다. 엔드포인트는 예고 없이 변경되거나 요청 제한이 적용될 수 있습니다 — 요청이 실패하기 시작한다면 이것이 원인입니다.
- Yahoo의 crumb 세션이 설정되지 않으면 `quote`에서 `market_cap`과 `pe_ratio`가 null로 반환될 수 있습니다. 값을 보완하려면 `ALPHA_VANTAGE_KEY`를 설정하세요.
- 요청을 대량으로 보낼 때는 요청 제한을 피하기 위해 요청 사이에 짧은 지연을 추가하세요.
- 읽기 전용입니다 — 주문 실행이나 계정 연동은 지원하지 않습니다.

## 검증

```
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py quote AAPL
```

`symbol: "AAPL"` 및 숫자형 `price` 필드를 포함하는 JSON 객체를 반환합니다.
