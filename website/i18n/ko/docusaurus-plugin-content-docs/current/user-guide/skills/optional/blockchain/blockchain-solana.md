---
title: "Solana — Solana 지갑, 토큰, 트랜잭션 및 NFT를 USD로 조회"
sidebar_label: "Solana"
description: "Solana 지갑, 토큰, 트랜잭션 및 NFT를 USD로 조회"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아닌 소스 SKILL.md를 편집하세요. */}

# Solana

Solana 지갑, 토큰, 트랜잭션 및 NFT를 USD로 조회합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/blockchain/solana`로 설치 |
| 경로 | `optional-skills/blockchain/solana` |
| 버전 | `0.2.0` |
| 작성자 | Deniz Alagoz (gizdusum), Hermes Agent가 개선 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Solana`, `Blockchain`, `Crypto`, `Web3`, `RPC`, `DeFi`, `NFT` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# Solana 블록체인 스킬

CoinGecko를 통해 USD 가격 정보가 보강된 Solana 온체인 데이터를 조회합니다.
지갑 포트폴리오, 토큰 정보, 트랜잭션, 활동, NFT,
고래 탐지, 네트워크 통계 및 가격 조회의 8가지 명령을 제공합니다.

API 키가 필요하지 않습니다. Python 표준 라이브러리(urllib, json, argparse)만 사용합니다.

---

## 사용 시점

- 사용자가 Solana 지갑 잔액, 토큰 보유량 또는 포트폴리오 가치를 묻는 경우
- 특정 서명으로 트랜잭션을 검사하려는 경우
- SPL 토큰 메타데이터, 가격, 공급량 또는 상위 보유자를 확인하려는 경우
- 주소의 최근 트랜잭션 기록을 확인하려는 경우
- 지갑이 보유한 NFT를 확인하려는 경우
- 대규모 SOL 이체를 찾으려는 경우(고래 탐지)
- Solana 네트워크 상태, TPS, 에포크 또는 SOL 가격을 묻는 경우
- "BONK/JUP/SOL의 가격이 얼마인가요?"라고 묻는 경우

---

## 사전 요구 사항

도우미 스크립트는 Python 표준 라이브러리(urllib, json, argparse)만 사용합니다.
외부 패키지는 필요하지 않습니다.

가격 데이터는 CoinGecko의 무료 API에서 가져옵니다(키가 필요하지 않으며 분당 약
10~30회로 제한됨). 더 빠르게 조회하려면 `--no-prices` 플래그를 사용하세요.

---

## 빠른 참조

RPC 엔드포인트(기본값): https://api.mainnet-beta.solana.com
재정의: export SOLANA_RPC_URL=https://your-private-rpc.com

도우미 스크립트 경로: ~/.hermes/skills/blockchain/solana/scripts/solana_client.py

```
python3 solana_client.py wallet   <address> [--limit N] [--all] [--no-prices]
python3 solana_client.py tx       <signature>
python3 solana_client.py token    <mint_address>
python3 solana_client.py activity <address> [--limit N]
python3 solana_client.py nft      <address>
python3 solana_client.py whales   [--min-sol N]
python3 solana_client.py stats
python3 solana_client.py price    <mint_or_symbol>
```

---

## 절차

### 0. 설정 확인

```bash
python3 --version

# Optional: set a private RPC for better rate limits
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# Confirm connectivity
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 1. 지갑 포트폴리오

SOL 잔액, USD 가치가 표시된 SPL 토큰 보유량, NFT 수 및
포트폴리오 총액을 가져옵니다. 토큰은 가치순으로 정렬되고, 잔돈은 필터링되며, 알려진 토큰은
이름(BONK, JUP, USDC 등)으로 표시됩니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

플래그:
- `--limit N` — 상위 N개 토큰 표시(기본값: 20)
- `--all` — 잔돈 필터와 제한 없이 모든 토큰 표시
- `--no-prices` — CoinGecko 가격 조회 생략(더 빠르며 RPC만 사용)

출력에는 SOL 잔액 및 USD 가치, 가치순으로 정렬된 가격 포함 토큰 목록,
잔돈 개수, NFT 요약 및 USD 기준 전체 포트폴리오 가치가 포함됩니다.

### 2. 트랜잭션 세부 정보

base58 서명으로 전체 트랜잭션을 검사합니다. SOL 및 USD 단위의 잔액 변동을 보여 줍니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  tx 5j7s8K...your_signature_here
```

출력: 슬롯, 타임스탬프, 수수료, 상태, 잔액 변동(SOL + USD),
프로그램 호출.

### 3. 토큰 정보

SPL 토큰 메타데이터, 현재 가격, 시가총액, 공급량, 소수점 자릿수,
민트/동결 권한 및 상위 5명의 보유자를 가져옵니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

출력: 이름, 심볼, 소수점 자릿수, 공급량, 가격, 시가총액, 백분율이 포함된 상위 5명의
보유자.

### 4. 최근 활동

주소의 최근 트랜잭션을 나열합니다(기본값: 최근 10개, 최대: 25개).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  activity 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM --limit 25
```

### 5. NFT 포트폴리오

지갑이 보유한 NFT를 나열합니다(휴리스틱: amount=1, decimals=0인 SPL 토큰).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  nft 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

참고: 이 휴리스틱으로는 압축 NFT(cNFT)를 탐지하지 못합니다.

### 6. 고래 탐지기

가장 최근 블록에서 USD 가치가 큰 SOL 이체를 검색합니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  whales --min-sol 500
```

참고: 최신 블록만 검색합니다 — 과거 데이터가 아닌 특정 시점의 스냅샷입니다.

### 7. 네트워크 통계

현재 슬롯, 에포크, TPS, 공급량, 검증자 버전, SOL 가격 및
시가총액을 포함한 실시간 Solana 네트워크 상태입니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 8. 가격 조회

민트 주소 또는 알려진 심볼로 모든 토큰의 가격을 빠르게 확인합니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price BONK
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price JUP
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price SOL
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

알려진 심볼: SOL, USDC, USDT, BONK, JUP, WETH, JTO, mSOL, stSOL,
PYTH, HNT, RNDR, WEN, W, TNSR, DRIFT, bSOL, JLP, WIF, MEW, BOME, PENGU.

---

## 주의 사항

- **CoinGecko 요청 제한** — 무료 등급은 분당 약 10~30회의 요청을 허용합니다.
  가격 조회는 토큰당 1회의 요청을 사용합니다. 토큰이 많은 지갑은
  모든 토큰의 가격을 가져오지 못할 수 있습니다. 속도를 높이려면 `--no-prices`를 사용하세요.
- **공개 RPC 요청 제한** — Solana 메인넷의 공개 RPC는 요청을 제한합니다.
  운영 환경에서는 SOLANA_RPC_URL을 비공개 엔드포인트로 설정하세요
  (Helius, QuickNode, Triton).
- **NFT 탐지는 휴리스틱입니다** — amount=1 + decimals=0입니다. 압축
  NFT(cNFT)와 Token-2022 NFT는 표시되지 않습니다.
- **고래 탐지기는 최신 블록만 검색합니다** — 과거 데이터가 아닙니다. 결과는 조회 시점에 따라 달라집니다.
- **트랜잭션 기록** — 공개 RPC는 약 2일간 보관합니다. 더 오래된 트랜잭션은
  제공되지 않을 수 있습니다.
- **토큰 이름** — 잘 알려진 토큰 약 25개는 이름으로 표시됩니다. 나머지는
  축약된 민트 주소로 표시됩니다. 전체 정보는 `token` 명령을 사용하세요.
- **429 재시도** — RPC와 CoinGecko 호출 모두 요청 제한 오류 발생 시 지수 백오프로
  최대 2회 재시도합니다.

---

## 검증

```bash
# Should print current Solana slot, TPS, and SOL price
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```
