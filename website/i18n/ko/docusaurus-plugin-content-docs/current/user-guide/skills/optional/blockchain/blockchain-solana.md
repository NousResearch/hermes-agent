---
title: "Solana"
sidebar_label: "Solana"
description: "USD 가격 책정과 함께 Solana 블록체인 데이터 쿼리 — 지갑 잔고, 가격이 포함된 토큰 포트폴리오, 트랜잭션 세부 정보, NFT, 고래 감지, 실시간 네트워크 통계..."
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Solana

USD 가격 책정과 함께 Solana 블록체인 데이터 쿼리 — 지갑 잔고, 가치가 포함된 토큰 포트폴리오, 트랜잭션 세부 정보, NFT, 고래 감지 및 실시간 네트워크 통계. Solana RPC + CoinGecko를 사용합니다. API 키가 필요하지 않습니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/blockchain/solana`를 사용하여 설치 |
| 경로 | `optional-skills/blockchain/solana` |
| 버전 | `0.2.0` |
| 작성자 | Deniz Alagoz (gizdusum), enhanced by Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Solana`, `Blockchain`, `Crypto`, `Web3`, `RPC`, `DeFi`, `NFT` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# Solana 블록체인 스킬

CoinGecko를 통해 USD 가격으로 강화된 Solana 온체인 데이터를 쿼리합니다.
8개 명령어: 지갑 포트폴리오, 토큰 정보, 트랜잭션, 활동, NFT,
고래 감지, 네트워크 통계 및 가격 조회.

API 키가 필요하지 않습니다. Python 표준 라이브러리(urllib, json, argparse)만 사용합니다.

---

## 사용 시기

- 사용자가 Solana 지갑 잔고, 토큰 보유량 또는 포트폴리오 가치를 요청할 때
- 사용자가 서명으로 특정 트랜잭션을 검사하고자 할 때
- 사용자가 SPL 토큰 메타데이터, 가격, 공급량 또는 상위 보유자를 원할 때
- 사용자가 주소에 대한 최근 트랜잭션 내역을 원할 때
- 사용자가 지갑이 소유한 NFT를 원할 때
- 사용자가 대규모 SOL 전송을 찾고자 할 때 (고래 감지)
- 사용자가 Solana 네트워크 상태, TPS, 에폭(epoch) 또는 SOL 가격을 원할 때
- 사용자가 "BONK/JUP/SOL의 가격이 얼마야?"라고 물을 때

---

## 전제 조건

헬퍼 스크립트는 Python 표준 라이브러리(urllib, json, argparse)만 사용합니다.
외부 패키지가 필요하지 않습니다.

가격 데이터는 CoinGecko의 무료 API에서 가져옵니다 (키 필요 없음,
~10-30 요청/분으로 속도 제한). 더 빠른 조회를 위해 `--no-prices` 플래그를 사용하세요.

---

## 빠른 참조

RPC 엔드포인트 (기본값): https://api.mainnet-beta.solana.com
재정의: export SOLANA_RPC_URL=https://your-private-rpc.com

헬퍼 스크립트 경로: ~/.hermes/skills/blockchain/solana/scripts/solana_client.py

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

# 선택 사항: 더 나은 속도 제한을 위해 프라이빗 RPC 설정
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# 연결 확인
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 1. 지갑 포트폴리오

SOL 잔고, USD 가치가 포함된 SPL 토큰 보유량, NFT 개수 및 포트폴리오 총액을 가져옵니다.
토큰은 가치별로 정렬되고 더스트(dust)는 필터링되며 알려진 토큰은 이름(BONK, JUP, USDC 등)으로 레이블이 지정됩니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

플래그:
- `--limit N` — 상위 N개 토큰 표시 (기본값: 20)
- `--all` — 더스트 필터링 없이 제한 없이 모든 토큰 표시
- `--no-prices` — CoinGecko 가격 조회 건너뛰기 (더 빠름, RPC 전용)

출력 포함 항목: SOL 잔고 + USD 가치, 가치별로 정렬된 가격이 포함된 토큰 목록, 더스트 개수, NFT 요약, USD 기준 총 포트폴리오 가치.

### 2. 트랜잭션 세부 정보

base58 서명으로 전체 트랜잭션을 검사합니다. SOL 및 USD로 잔고 변화를 보여줍니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  tx 5j7s8K...your_signature_here
```

출력: 슬롯, 타임스탬프, 수수료, 상태, 잔고 변화(SOL + USD), 프로그램 호출.

### 3. 토큰 정보

SPL 토큰 메타데이터, 현재 가격, 시가총액, 공급량, 소수점 자리수(decimals),
발행/동결 권한(mint/freeze authorities) 및 상위 5명 보유자를 가져옵니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

출력: 이름, 기호, 소수점, 공급량, 가격, 시가총액, 백분율이 포함된 상위 5명 보유자.

### 4. 최근 활동

주소에 대한 최근 트랜잭션을 나열합니다 (기본값: 최근 10개, 최대: 25개).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  activity 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM --limit 25
```

### 5. NFT 포트폴리오

지갑이 소유한 NFT를 나열합니다 (휴리스틱: 양=1, 소수점=0인 SPL 토큰).

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  nft 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
```

참고: 압축된 NFT(cNFT)는 이 휴리스틱에 의해 감지되지 않습니다.

### 6. 고래 감지기

가장 최근 블록에서 USD 가치가 포함된 대규모 SOL 전송을 스캔합니다.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py \
  whales --min-sol 500
```

참고: 가장 최근 블록만 스캔합니다 — 과거 데이터가 아닌 특정 시점의 스냅샷입니다.

### 7. 네트워크 통계

실시간 Solana 네트워크 상태: 현재 슬롯, 에폭, TPS, 공급량, 검증자 버전, SOL 가격 및 시가총액.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```

### 8. 가격 조회

발행 주소(mint address) 또는 알려진 기호로 모든 토큰에 대한 빠른 가격 확인.

```bash
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price BONK
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price JUP
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price SOL
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py price DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

알려진 기호: SOL, USDC, USDT, BONK, JUP, WETH, JTO, mSOL, stSOL,
PYTH, HNT, RNDR, WEN, W, TNSR, DRIFT, bSOL, JLP, WIF, MEW, BOME, PENGU.

---

## 주의 사항

- **CoinGecko 속도 제한** — 무료 티어는 ~10-30 요청/분을 허용합니다.
  가격 조회는 토큰당 1번의 요청을 사용합니다. 많은 토큰이 있는 지갑은
  모든 토큰의 가격을 가져오지 못할 수 있습니다. 속도를 위해 `--no-prices`를 사용하세요.
- **공개 RPC 속도 제한** — Solana 메인넷 공개 RPC는 요청을 제한합니다.
  프로덕션 용도의 경우 SOLANA_RPC_URL을 프라이빗 엔드포인트(Helius, QuickNode, Triton)로 설정하세요.
- **NFT 감지는 휴리스틱 방식입니다** — 양=1 + 소수점=0. 압축된
  NFT(cNFT) 및 Token-2022 NFT는 표시되지 않습니다.
- **고래 감지기는 최신 블록만 스캔합니다** — 과거 데이터가 아닙니다. 결과는
  쿼리하는 순간에 따라 다릅니다.
- **트랜잭션 내역** — 공개 RPC는 약 2일간 유지합니다. 오래된 트랜잭션은
  사용할 수 없을 수 있습니다.
- **토큰 이름** — 약 25개의 잘 알려진 토큰이 이름으로 레이블 지정됩니다. 기타 토큰은
  축약된 발행 주소(mint addresses)를 표시합니다. 전체 정보를 보려면 `token` 명령어를 사용하세요.
- **429 오류 시 재시도** — RPC 및 CoinGecko 호출 모두 속도 제한 오류 시
  지수 백오프와 함께 최대 2번 재시도합니다.

---

## 검증

```bash
# 현재 Solana 슬롯, TPS 및 SOL 가격을 인쇄해야 합니다
python3 ~/.hermes/skills/blockchain/solana/scripts/solana_client.py stats
```
