---
title: "Stocks"
sidebar_label: "Stocks"
description: "상장 주식 시세 데이터, 캔들 차트, 이동 평균, 배당금 등 분석"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Stocks

상장 주식에 대한 Yahoo Finance 데이터 검색 - 시세, 일일/분 단위 캔들(OHLCV), 주식 분할, 배당금, 이동 평균 분석, 비교 성능 측정 기준 등. API 키 필요 없음.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/finance/stocks`를 사용하여 설치 |
| 경로 | `optional-skills/finance/stocks` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `finance`, `stocks`, `markets`, `pricing`, `dividends`, `yahoo-finance` |
| 관련 스킬 | [`dcf-model`](/docs/user-guide/skills/optional/finance/finance-dcf-model), [`comps-analysis`](/docs/user-guide/skills/optional/finance/finance-comps-analysis) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# Stocks 스킬

yfinance(Yahoo Finance)를 통해 실시간 및 과거 시장 데이터를 조회합니다.
7개 명령어: 현재 가격 조회, 과거 OHLCV 데이터 (일/분 단위), 
주식 분할/배당 내역 파악, 여러 티커 성능 비교 및 
기술적 분석(SMA/EMA, RSI 등) 수행.

API 키가 필요하지 않습니다. 종속성: `yfinance`, `pandas`.

---

## 사용 시기

- 사용자가 티커(예: AAPL, MSFT, TSLA)의 **현재 가격이나 시가총액**을 물어볼 때.
- 사용자가 과거 성과, 캔들 데이터(시가/고가/저가/종가/거래량) 또는 시간 경과에 따른 추세를 볼 수 있는 **OHLCV 데이터**를 요청할 때.
- 사용자가 최근 주식 분할이나 배당 지급 내역 등 **기업 이벤트**를 확인하고자 할 때.
- 사용자가 **여러 주식의 실적을 비교**하고자 할 때 (예: "애플과 마이크로소프트 중 지난 1년간 누가 수익률이 좋았어?").
- 사용자가 가격 변동에 대한 **단순/지수 이동 평균(SMA/EMA)**, RSI(상대강도지수), MACD 등의 기술적 분석 지표를 원할 때.
- **재무 모델링 준비** 시 DCF 모델이나 비교 기업 분석에 최신 주가 및 거래량 데이터가 필요할 때.

---

## 전제 조건

이 스킬은 `yfinance`와 `pandas` 패키지에 의존합니다. 시작하기 전에 패키지가 설치되어 있는지 확인하세요:

```bash
uv pip install yfinance pandas
```

헬퍼 스크립트 경로: `~/.hermes/skills/finance/stocks/scripts/stocks_client.py`

---

## 빠른 참조

```bash
SCRIPT=~/.hermes/skills/finance/stocks/scripts/stocks_client.py

# 현재 가격 및 기본 정보 (종목 하나)
python3 $SCRIPT quote AAPL

# 과거 OHLCV 데이터 (일/분/주/월 단위)
python3 $SCRIPT history TSLA --period 1mo --interval 1d
python3 $SCRIPT history NVDA --start 2023-01-01 --end 2023-12-31 --interval 1wk

# 배당금 및 주식 분할 내역
python3 $SCRIPT actions MSFT --period 5y

# 종목 비교 (퍼포먼스 % 비교)
python3 $SCRIPT compare AAPL MSFT GOOG --period 1y

# 이동 평균 및 기술적 지표 (SMA, EMA, RSI 등)
python3 $SCRIPT technical AMZN --indicators sma50,sma200,rsi --period 6mo
```

---

## 절차

### 1. 설정 및 설치 확인
먼저 필요한 Python 패키지가 설치되어 있는지 확인합니다:
```bash
python3 -c "import yfinance, pandas; print('Setup complete.')"
```

### 2. 기본 시세 조회
주식의 현재 가격, 시가총액, 거래량 및 52주 최고/최저가를 가져옵니다.

```bash
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py quote AAPL
```

### 3. 과거 주가 데이터(OHLCV) 가져오기
가장 일반적인 조회: 캔들스틱 차트를 위한 시가, 고가, 저가, 종가 및 거래량 데이터입니다. 기간(`period`)이나 명시적인 시작/종료 날짜(`start`/`end`)를 사용할 수 있습니다.

```bash
# 지난 3개월 동안의 일일 데이터
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py history TSLA --period 3mo --interval 1d

# 특정 기간 동안의 주간 데이터
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py history NVDA --start 2022-01-01 --end 2023-01-01 --interval 1wk

# 지난 5일 동안의 15분 단위 데이터
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py history AMD --period 5d --interval 15m
```

### 4. 기업 이벤트 (배당금 & 분할)
해당 주식의 배당금 지급 내역 및 주식 분할 내역을 조회합니다. DCF 가치 평가나 수익률 계산에 유용합니다.

```bash
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py actions MSFT --period max
```

### 5. 다중 종목 성과 비교
동일한 기간 동안 여러 종목의 누적 수익률(%)을 비교합니다. 동종 업계 성과를 벤치마킹하는 데 유용합니다.

```bash
# 애플, 마이크로소프트, 알파벳의 1년 성과 비교
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py compare AAPL MSFT GOOG --period 1y
```

### 6. 기술적 분석 (Technical Indicators)
기본적인 기술적 분석 지표(단순이동평균(SMA), 지수이동평균(EMA), 상대강도지수(RSI))를 포함한 과거 데이터를 반환합니다.

```bash
# 50일선, 200일선 이동 평균과 RSI(14일)를 지난 6개월 데이터에 적용
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py technical AMZN --indicators sma50,sma200,rsi --period 6mo
```

---

## 지원 기간 및 간격 (yfinance 기반)

- **유효한 기간(`period`)**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- **유효한 간격(`interval`)**: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
  - *참고: 분 단위 데이터(1m ~ 90m)는 최근 60일 데이터에 대해서만 사용할 수 있습니다. 1m 데이터는 최근 7일로 제한될 수 있습니다.*

---

## 주의 사항

- `yfinance`는 Yahoo Finance의 비공식 API를 사용하므로, 스로틀링(속도 제한)이 발생할 수 있습니다. 1초에 너무 많은 요청을 보내지 마세요.
- 장 중(Intraday) 분 단위 데이터는 Yahoo Finance에서 엄격한 시간 제한(최근 60일)을 두고 제공합니다. 오래된 기간의 분 단위 데이터는 조회가 불가능합니다.
- 데이터는 일반적으로 시장 개장 시간에 맞춰 실시간/지연(15분) 상태로 제공되며, 소스 상황에 따라 차이가 있을 수 있습니다.
- 전문적인 기관 수준의 트레이딩 시스템 연동 시에는 이 무료 데이터를 참조용으로만 사용하고 보장된 데이터 피드를 확인하는 것이 좋습니다.

---

## 검증
```bash
# Apple의 현재 주가 정보를 인쇄해야 합니다
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py quote AAPL

# Tesla의 지난 달 일일 거래 데이터를 출력해야 합니다
python3 ~/.hermes/skills/finance/stocks/scripts/stocks_client.py history TSLA --period 1mo
```
