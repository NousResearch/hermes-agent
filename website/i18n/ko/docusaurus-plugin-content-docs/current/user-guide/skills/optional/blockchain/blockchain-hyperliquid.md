---
title: "Hyperliquid — Hyperliquid 시장 데이터, 계정 기록, 거래 검토"
sidebar_label: "Hyperliquid"
description: "Hyperliquid 시장 데이터, 계정 기록, 거래 검토"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Hyperliquid

Hyperliquid 시장 데이터, 계정 기록, 거래 검토.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/blockchain/hyperliquid`로 설치 |
| 경로 | `optional-skills/blockchain/hyperliquid` |
| 버전 | `0.1.0` |
| 작성자 | Hugo Sequier (Hugo-SEQUIER), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Hyperliquid`, `Blockchain`, `Crypto`, `Trading`, `Perpetuals`, `Spot`, `DeFi` |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# Hyperliquid Skill

공개 `/info` 엔드포인트를 통해 Hyperliquid 시장 및 계정 데이터를 조회합니다.
읽기 전용 — API 키, 서명, 주문 제출이 필요하지 않습니다.

12개 명령: `dexs`, `markets`, `spots`, `candles`, `funding`, `l2`, `state`,
`spot-balances`, `fills`, `orders`, `review`, `export`. 표준 라이브러리만 사용
(`urllib`, `json`, `argparse`).

---

## 사용 시점

- 사용자가 Hyperliquid 무기한 또는 현물 시장 데이터, 캔들, 펀딩 또는 L2 호가창을 요청할 때
- 사용자가 지갑의 무기한 포지션, 현물 잔고, 체결 또는 주문을 확인하려 할 때
- 사용자가 최근 체결과 시장 맥락을 결합한 거래 후 검토를 원할 때
- 사용자가 빌더가 배포한 무기한 DEX 또는 HIP-3 시장을 확인하려 할 때
- 사용자가 백테스팅 준비를 위해 캔들 + 펀딩의 정규화된 JSON 내보내기를 원할 때

---

## 사전 요구 사항

표준 라이브러리만 사용 — 외부 패키지와 API 키가 필요하지 않습니다.

스크립트는 `${HERMES_HOME:-~/.hermes}/.env`에서 다음 선택적 기본값을 읽습니다:

- `HYPERLIQUID_API_URL` — 기본값은 `https://api.hyperliquid.xyz`입니다. 테스트넷에는
  `https://api.hyperliquid-testnet.xyz`로 설정합니다.
- `HYPERLIQUID_USER_ADDRESS` — `state`, `spot-balances`,
  `fills`, `orders`, `review`의 기본 주소입니다. 설정되지 않았다면 주소를 첫 번째
  위치 인수로 전달합니다.

현재 작업 디렉터리의 프로젝트 `.env`는 개발용 대체 경로로 사용됩니다.

도우미 스크립트: `~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py`

---

## 실행 방법

`terminal` 도구를 통해 호출합니다:

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py <command> [args]
```

모든 명령에 `--json`을 추가하면 기계가 읽을 수 있는 출력이 생성됩니다.

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

`${HERMES_HOME:-~/.hermes}/.env`에 `HYPERLIQUID_USER_ADDRESS`가 설정되어 있으면
`state`, `spot-balances`, `fills`, `orders`, `review`의 주소는 선택 사항입니다.

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

- `--dex`는 무기한 엔드포인트에만 적용됩니다. 첫 번째 무기한 DEX에는 생략합니다.
- 현물 쌍은 `PURR/USDC` 또는 `@107` 같은 별칭으로 표시될 수 있습니다.
- HIP-3 시장은 코인 앞에 DEX를 붙입니다 (예: `mydex:BTC`).

### 2. 과거 시장 데이터 가져오기

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  candles BTC --interval 1h --hours 72 --limit 48

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  funding BTC --hours 168 --limit 30
```

시간 범위 엔드포인트는 페이지로 나뉩니다. 더 큰 기간에는 이후 `startTime`으로 반복하거나 아래의 `export`를 사용합니다.

### 3. 실시간 호가창 검사

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  l2 BTC --levels 10
```

호가창 깊이, 단기 유동성 또는 대규모 주문의 잠재적 시장 영향을 질문받았을 때 사용합니다.

### 4. 계정 검토

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  state 0xabc...

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spot-balances
```

`state`는 무기한 포지션을 반환하고, `spot-balances`는 현물 보유량을 반환합니다.
"내 포지션은 어때?", "무엇을 보유하고 있어?", "얼마나 출금할 수 있어?"와 같은 질문에 사용합니다.

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

거래한 각 무기한 상품에 대해 실현 손익, 수수료, 승/패 횟수, 코인별 분석, 시장 추세와 평균 펀딩을 보고하며, 수수료 부담, 집중도, 추세 역행 손실과 같은 휴리스틱도 제공합니다.

더 심층적인 거래 후 분석은 다음과 같이 진행합니다: `review`로 문제가 있는 코인이나 기간을 찾고 → 해당 기간의 `fills` 및 `orders`를 가져오고 → 거래한 각 코인의 `candles` 및 `funding`을 가져온 뒤 → 결과의 품질과 별개로 의사결정의 품질을 판단합니다.

### 7. 재사용 가능한 데이터셋 내보내기

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 1h --hours 168 --output ./btc-1h-7d.json

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 15m --hours 72 --end-time-ms 1760000000000
```

출력 JSON에는 스키마 버전, 소스 메타데이터, 정확한 시간 범위, 정규화된 캔들 행, 정규화된 펀딩 행, 요약 통계가 포함됩니다. 재현 가능한 기간에는 `--end-time-ms`를 사용합니다.

---

## 주의 사항

- 공개 info 엔드포인트에는 요청 속도 제한이 있습니다. 대규모 과거 조회는 제한된 기간을 반환할 수 있으므로 이후 `startTime` 값으로 반복합니다.
- `fills --hours ...`는 `userFillsByTime`을 사용하며, 최근의 이동 기간만 노출하고 전체 아카이브 기록은 제공하지 않습니다.
- `historicalOrders`는 최근 주문만 반환하며 전체 내보내기가 아닙니다.
- `review` 명령은 휴리스틱입니다. 체결만으로는 의도, 주문 제출 품질 또는 실제 슬리피지를 재구성할 수 없습니다.
- `export` 명령은 정규화된 데이터셋을 작성할 뿐 백테스트 엔진이 아닙니다. 자체 슬리피지/체결 모델이 여전히 필요합니다.
- UI에 더 알기 쉬운 이름이 표시되더라도 `@107` 같은 현물 별칭은 유효한 식별자입니다.
- `l2`는 시계열이 아니라 특정 시점의 스냅샷입니다.

---

## 검증

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 5
```

24시간 명목 거래량 기준으로 상위 Hyperliquid 무기한 시장을 출력해야 합니다.
