---
title: "Hyperliquid — Hyperliquid 시장 데이터, 계정 내역, 거래 검토"
sidebar_label: "Hyperliquid"
description: "Hyperliquid 시장 데이터, 계정 내역, 거래 검토"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Hyperliquid

Hyperliquid 시장 데이터, 계정 내역, 거래 검토.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/blockchain/hyperliquid`를 사용하여 설치 |
| 경로 | `optional-skills/blockchain/hyperliquid` |
| 버전 | `0.1.0` |
| 작성자 | Hugo Sequier (Hugo-SEQUIER), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Hyperliquid`, `Blockchain`, `Crypto`, `Trading`, `Perpetuals`, `Spot`, `DeFi` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# Hyperliquid 스킬

공개 `/info` 엔드포인트를 통해 Hyperliquid 시장 및 계정 데이터를 쿼리합니다.
읽기 전용 — API 키, 서명, 주문 입력이 없습니다.

12개 명령어: `dexs`, `markets`, `spots`, `candles`, `funding`, `l2`, `state`,
`spot-balances`, `fills`, `orders`, `review`, `export`. 표준 라이브러리(Stdlib)만 사용
(`urllib`, `json`, `argparse`).

---

## 사용 시기

- 사용자가 Hyperliquid 무기한(perp) 또는 현물(spot) 시장 데이터, 캔들, 펀딩 또는 L2 호가창을 요청할 때
- 사용자가 지갑의 무기한 포지션, 현물 잔고, 체결 내역 또는 주문을 검사하고자 할 때
- 사용자가 최근 체결 내역과 시장 상황을 결합한 사후 거래 검토를 원할 때
- 사용자가 빌더가 배포한 무기한 DEX 또는 HIP-3 시장을 검사하고자 할 때
- 사용자가 백테스트 준비를 위해 캔들 + 펀딩의 정규화된 JSON 내보내기를 원할 때

---

## 전제 조건

표준 라이브러리(Stdlib)만 사용 — 외부 패키지 없음, API 키 없음.

스크립트는 두 가지 선택적 기본값을 위해 `~/.hermes/.env`를 읽습니다:

- `HYPERLIQUID_API_URL` — 기본값은 `https://api.hyperliquid.xyz`입니다. 테스트넷의 경우
  `https://api.hyperliquid-testnet.xyz`로 설정하세요.
- `HYPERLIQUID_USER_ADDRESS` — `state`, `spot-balances`, `fills`, `orders`, `review`의 기본 주소입니다. 설정되지 않은 경우, 첫 번째 위치 인수로 주소를 전달하세요.

현재 작업 디렉토리의 프로젝트 `.env`는 개발용 폴백으로 존중됩니다.

헬퍼 스크립트: `~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py`

---

## 실행 방법

`terminal` 도구를 통해 호출합니다:

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py <command> [args]
```

기계가 읽을 수 있는 출력을 원하면 아무 명령어에나 `--json`을 추가하세요.

---

## 빠른 참조

```bash
hyperliquid_client.py dexs
hyperliquid_client.py markets [--dex DEX] [--limit N] [--sort volume|oi|funding_abs|change_abs|name]
hyperliquid_client.py spots [--limit N]
hyperliquid_client.py candles <coin> [--interval 1h] [--hours 24] [--limit N]
hyperliquid_client.py funding <coin> [--hours 72] [--limit N]
hyperliquid_client.py l2 <coin> [--levels N]
hyperliquid_client.py state [address] [--dex DEX]
hyperliquid_client.py spot-balances [address] [--limit N]
hyperliquid_client.py fills [address] [--hours N] [--limit N] [--aggregate-by-time]
hyperliquid_client.py orders [address] [--limit N]
hyperliquid_client.py review [address] [--coin COIN] [--hours N] [--fills N]
hyperliquid_client.py export <coin> [--interval 1h] [--hours N] [--output PATH]
```

`~/.hermes/.env`에 `HYPERLIQUID_USER_ADDRESS`가 설정되어 있는 경우, `state`, `spot-balances`, `fills`, `orders`, `review` 명령어에서 주소는 선택 사항입니다.

---

## 절차

### 1. DEX 및 시장 검색

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py dexs

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 15 --sort volume

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spots --limit 15
```

- `--dex`는 무기한 엔드포인트에만 적용됩니다. 첫 번째 무기한 DEX의 경우 생략하세요.
- 현물 페어는 `PURR/USDC` 또는 `@107`과 같은 별칭으로 표시될 수 있습니다.
- HIP-3 시장은 `mydex:BTC`와 같이 코인 앞에 DEX 접두사를 붙입니다.

### 2. 과거 시장 데이터 가져오기

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  candles BTC --interval 1h --hours 72 --limit 48

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  funding BTC --hours 168 --limit 30
```

시간 범위 엔드포인트는 페이지를 매깁니다. 더 큰 창의 경우 나중의 `startTime`으로 반복하거나 `export`(아래 참조)를 사용하세요.

### 3. 실시간 오더북 검사

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  l2 BTC --levels 10
```

호가창 깊이, 단기 유동성 또는 대규모 주문의 잠재적인 시장 영향에 대한 질문을 받았을 때 사용하세요.

### 4. 계정 검토

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  state 0xabc...

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spot-balances
```

`state`는 무기한 포지션을 반환합니다. `spot-balances`는 현물 보유량을 반환합니다.
"내 포지션은 어때?", "나는 무엇을 보유하고 있어?", "얼마나 출금할 수 있어?"와 같은 질문에 사용하세요.

### 5. 체결 및 주문 검토

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  fills 0xabc... --hours 72 --limit 25

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  orders --limit 25
```

### 6. 거래 검토 생성

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review 0xabc... --hours 72 --fills 50

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review --coin BTC --hours 168
```

실현 PnL, 수수료, 승/패 횟수, 코인별 내역, 거래된 각 무기한 계약의 시장 추세 및 평균 펀딩뿐만 아니라 휴리스틱(수수료 부담, 집중도, 역추세 손실)을 보고합니다.

더 깊은 사후 거래 분석을 위해: `review`로 시작하여 문제 코인이나 기간을 찾습니다 → 해당 기간에 대한 `fills`와 `orders`를 가져옵니다 → 거래된 각 코인에 대한 `candles`와 `funding`을 가져옵니다 → 결과 품질과 분리하여 의사 결정 품질을 판단합니다.

### 7. 재사용 가능한 데이터셋 내보내기

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 1h --hours 168 --output ./btc-1h-7d.json

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 15m --hours 72 --end-time-ms 1760000000000
```

출력 JSON 포함 항목: 스키마 버전, 소스 메타데이터, 정확한 시간 창, 정규화된 캔들 행, 정규화된 펀딩 행, 요약 통계. 재현 가능한 창을 위해 `--end-time-ms`를 사용하세요.

---

## 주의 사항

- 공개 정보 엔드포인트는 속도 제한이 있습니다. 대규모 과거 데이터 쿼리는 제한된 창을 반환할 수 있습니다. 나중의 `startTime` 값으로 반복하세요.
- `fills --hours ...`는 최근 롤링 윈도우만 노출하는 `userFillsByTime`을 사용합니다 — 전체 아카이브 내역이 아닙니다.
- `historicalOrders`는 최근 주문만 반환합니다 — 전체 내보내기가 아닙니다.
- `review` 명령어는 휴리스틱 방식입니다. 체결 내역만으로는 의도, 주문 입력 품질 또는 진정한 슬리피지를 재구성할 수 없습니다.
- `export` 명령어는 백테스트 엔진이 아닌 정규화된 데이터셋을 작성합니다. 여전히 자체 슬리피지/체결 모델이 필요합니다.
- UI에 더 친숙한 이름이 표시되더라도 `@107`과 같은 현물 별칭은 유효한 식별자입니다.
- `l2`는 시계열이 아닌 특정 시점의 스냅샷입니다.

---

## 검증

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 5
```

24시간 명목 거래량 기준 상위 Hyperliquid 무기한 시장을 인쇄해야 합니다.
