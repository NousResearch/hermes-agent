---
title: "Evm — 읽기 전용 EVM 클라이언트: 8개 체인에 걸친 지갑, 토큰, 가스"
sidebar_label: "Evm"
description: "읽기 전용 EVM 클라이언트: 8개 체인에 걸친 지갑, 토큰, 가스"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Evm

읽기 전용 EVM 클라이언트: 8개 체인에 걸친 지갑, 토큰, 가스.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/blockchain/evm`을 사용하여 설치 |
| 경로 | `optional-skills/blockchain/evm` |
| 버전 | `1.0.0` |
| 작성자 | Mibayy (@Mibayy), youssefea (@youssefea), ethernet8023 (@ethernet8023), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `EVM`, `Ethereum`, `BNB`, `BSC`, `Base`, `Arbitrum`, `Polygon`, `Optimism`, `Avalanche`, `zkSync`, `Blockchain`, `Crypto`, `Web3`, `DeFi`, `NFT`, `ENS`, `Whale`, `Security` |
| 관련 스킬 | [`solana`](/docs/user-guide/skills/optional/blockchain/blockchain-solana) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# EVM 블록체인 스킬

USD 가격 책정과 함께 8개 체인에 걸쳐 EVM 호환 블록체인 데이터를 쿼리합니다.
14개 명령어: 지갑 포트폴리오, 토큰 정보, 트랜잭션, 활동, 가스 트래커,
네트워크 통계, 가격 조회, 다중 체인 스캔, 고래 감지, ENS 확인,
허용량 검사기, 컨트랙트 검사기, 트랜잭션 디코더.

8개 체인 지원: Ethereum, BNB Chain (BSC), Base, Arbitrum One, Polygon,
Optimism, Avalanche (C-Chain), zkSync Era.

API 키가 필요하지 않습니다. 외부 종속성 없음 — Python 표준 라이브러리만 사용
(urllib, json, argparse, threading).

> **독립 실행형 `base` 스킬을 대체합니다.** Base 특정 토큰(AERO, DEGEN,
> TOSHI, BRETT, WELL, cbETH, cbBTC, wstETH, rETH) 및 이전에는 `optional-skills/blockchain/base/`
> 아래에 있던 모든 Base RPC 기능이 이 스킬로 통합되었습니다. Base 커버리지를 위해
> 모든 명령어에 `--chain base`를 전달하세요.

---

## 사용 시기
- 사용자가 모든 EVM 체인의 지갑 잔고 또는 포트폴리오를 요청할 때
- 사용자가 한 번에 모든 체인에서 동일한 지갑을 확인하고자 할 때
- 사용자가 해시로 트랜잭션을 검사(또는 무슨 일을 했는지 디코드)하고자 할 때
- 사용자가 ERC-20 토큰 메타데이터, 가격, 공급량 또는 시가총액을 원할 때
- 사용자가 주소에 대한 최근 트랜잭션 내역을 원할 때
- 사용자가 현재 가스 가격을 원하거나 체인 간 수수료를 비교하고자 할 때
- 사용자가 최근 블록에서 대규모 고래 전송을 찾고자 할 때
- 사용자가 ENS 이름(vitalik.eth)을 확인하거나 주소를 역조회하고자 할 때
- 사용자가 컨트랙트에 위험한 토큰 승인이 있는지 확인하고자 할 때
- 사용자가 스마트 컨트랙트(프록시? ERC-20? ERC-721? 바이트코드 크기?)를 검사하고자 할 때
- 사용자가 트랜잭션 전 체인 간 가스 비용을 비교하고자 할 때

---

## 전제 조건
Python 3.8+ 표준 라이브러리만 사용합니다. pip 설치가 필요하지 않습니다.
가격 책정: CoinGecko 무료 API (속도 제한, ~10-30 요청/분).
ENS: ensideas.com 공개 API.
Tx 디코딩: 4byte.directory 공개 API.

RPC 엔드포인트 재정의: `export EVM_RPC_URL=https://your-rpc.com`

헬퍼 스크립트 경로: `~/.hermes/skills/blockchain/evm/scripts/evm_client.py`

---

## 빠른 참조

```bash
SCRIPT=~/.hermes/skills/blockchain/evm/scripts/evm_client.py

# Network & prices
python3 $SCRIPT stats                            # Ethereum stats
python3 $SCRIPT stats --chain arbitrum           # Arbitrum stats
python3 $SCRIPT compare                          # Gas + prices ALL 8 chains

# Wallet
python3 $SCRIPT wallet 0xd8dA...96045            # Portfolio (ETH + ERC-20)
python3 $SCRIPT wallet 0xd8dA...96045 --chain bsc
python3 $SCRIPT multichain 0xd8dA...96045        # Same wallet on ALL chains

# Tokens & prices
python3 $SCRIPT price ETH
python3 $SCRIPT price 0xdAC1...1ec7              # By contract address
python3 $SCRIPT token 0xdAC1...1ec7              # ERC-20 metadata + market cap

# Transactions
python3 $SCRIPT tx 0x5c50...f060                 # Transaction details
python3 $SCRIPT decode 0x5c50...f060             # Decode input data (4byte.directory)
python3 $SCRIPT activity 0xd8dA...96045          # Recent transactions

# Gas
python3 $SCRIPT gas                              # Gas prices + cost estimates
python3 $SCRIPT gas --chain optimism

# Security
python3 $SCRIPT allowance 0xd8dA...96045         # Dangerous ERC-20 approvals
python3 $SCRIPT contract 0xdAC1...1ec7           # Contract inspection (proxy? standards?)

# ENS
python3 $SCRIPT ens vitalik.eth                  # Name -> address + profile
python3 $SCRIPT ens 0xd8dA...96045               # Address -> ENS name

# Whale detection
python3 $SCRIPT whale                            # Large transfers (last 20 blocks, >$10k)
python3 $SCRIPT whale --blocks 50 --min-usd 100000 --chain arbitrum
```

---

## 절차

### 0. 설정 확인
```bash
python3 --version   # 3.8+ required
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats
```

### 1. 지갑 포트폴리오
기본 잔고 + 알려진 ERC-20 토큰, USD 값으로 정렬.
```bash
python3 $SCRIPT wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
python3 $SCRIPT wallet 0xd8dA... --chain bsc --no-prices   # faster
```

### 2. 다중 체인 스캔
스레드를 사용하여 동일한 주소에 대해 8개 체인을 동시에 스캔합니다.
```bash
python3 $SCRIPT multichain 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```
출력: 체인별 기본 잔고 + 토큰 보유량 + 총 USD 합계.

### 3. 비교 (가스 + 가격)
8개 체인 모두 병렬로 쿼리. 가장 저렴한/가장 비싼 체인을 보여줍니다.
```bash
python3 $SCRIPT compare
```

### 4. 트랜잭션 세부 정보 및 디코드
```bash
python3 $SCRIPT tx 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060
python3 $SCRIPT decode 0x5c504ed...   # Shows human-readable function signature
```
디코드는 4byte.directory를 사용하여 `0xa9059cbb`를 `transfer(address,uint256)`로 변환합니다.

### 5. ENS 확인
```bash
python3 $SCRIPT ens vitalik.eth          # -> 0xd8dA... + avatar + social links
python3 $SCRIPT ens 0xd8dA...96045       # -> vitalik.eth
```

### 6. 허용량 검사기 (보안)
알려진 DEX/브릿지 컨트랙트에 부여된 ERC-20 승인을 확인합니다.
```bash
python3 $SCRIPT allowance 0xYourWallet
```
무제한(UNLIMITED) 승인을 고위험(HIGH risk)으로 플래그합니다.

### 7. 컨트랙트 검사기
```bash
python3 $SCRIPT contract 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48   # USDC (proxy)
python3 $SCRIPT contract 0xdAC17F958D2ee523a2206206994597C13D831ec7   # USDT (ERC-20)
```
프록시(EIP-1967/EIP-1167), ERC-20, ERC-721, ERC-165를 감지합니다. 프록시의 경우 바이트코드 크기와 구현 주소를 보여줍니다.

### 8. 고래 감지
```bash
python3 $SCRIPT whale                                    # ETH, last 20 blocks, >$10k
python3 $SCRIPT whale --blocks 50 --min-usd 50000 --chain bsc
```

### 9. 가스 트래커
```bash
python3 $SCRIPT gas
python3 $SCRIPT gas --chain polygon
```
전송, ERC-20 전송, 승인, 스왑, NFT 민팅, NFT 전송에 대한 Gwei 가격 + USD 비용을 보여줍니다.

---

## 지원 체인
| Key       | Name           | Native | Chain ID |
|-----------|----------------|--------|----------|
| ethereum  | Ethereum       | ETH    | 1        |
| bsc       | BNB Chain      | BNB    | 56       |
| base      | Base           | ETH    | 8453     |
| arbitrum  | Arbitrum One   | ETH    | 42161    |
| polygon   | Polygon        | POL    | 137      |
| optimism  | Optimism       | ETH    | 10       |
| avalanche | Avalanche C    | AVAX   | 43114    |
| zksync    | zkSync Era     | ETH    | 324      |

---

## 주의 사항
- CoinGecko 무료 티어: ~10-30 요청/분. 더 빠른 지갑 스캔을 위해 `--no-prices`를 사용하세요.
- 공개 RPC는 스로틀링될 수 있습니다. 프로덕션의 경우 `EVM_RPC_URL`을 프라이빗 엔드포인트로 설정하세요.
- `wallet` 및 `allowance`는 알려진 토큰 목록(체인당 ~30개 토큰)만 확인합니다. 완전한 토큰 발견을 위해서는 블록 익스플로러를 사용하세요.
- `activity`는 최근 블록만 스캔합니다(최대 200개). 전체 내역의 경우 Etherscan API를 사용하세요.
- `multichain`은 8개의 병렬 스레드를 실행하므로 공개 RPC에서 속도 제한을 유발할 수 있습니다.
- ENS 확인은 대체가 없는 단일 공개 엔드포인트(ensideas.com / ens.vitalik.ca)에 의존합니다. 해당 엔드포인트가 다운된 경우 `ens`는 실패합니다 — 나중에 다시 실행하거나 블록 익스플로러를 사용하세요.
- Tx 디코딩은 대체가 없는 단일 공개 엔드포인트(4byte.directory)에 의존합니다. 해당 데이터베이스에 없는 선택기는 `unknown`으로 표시됩니다.
- **L2 가스 추정치는 L2 실행 전용입니다.** Base, Arbitrum, Optimism, zkSync와 같은 롤업의 경우 실제 트랜잭션 비용에는 콜데이터 크기 및 현재 L1 가스 가격에 따라 달라지는 L1 데이터 게시 수수료도 포함됩니다. `gas` 명령어는 해당 L1 컴포넌트를 추정하지 않습니다. 특히 Base의 경우 네트워크의 L1 수수료 오라클(컨트랙트 `0x420000000000000000000000000000000000000F`)을 참조하세요.
- 주소 / tx-해시 입력은 0x-접두사 + 올바른 길이 + 16진수로 검증되지만, EIP-55 체크섬 대소문자는 적용되지 **않습니다** (RPC 엔드포인트는 모든 대소문자의 16진수를 허용합니다).

---

## 검증
```bash
# 현재 블록, 가스 가격, ETH 가격을 인쇄해야 합니다
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats

# vitalik.eth를 0xd8dA...로 확인해야 합니다
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py ens vitalik.eth
```
