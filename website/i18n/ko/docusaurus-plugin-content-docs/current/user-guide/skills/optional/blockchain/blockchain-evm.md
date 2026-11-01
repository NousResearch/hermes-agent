---
title: "Evm — 읽기 전용 EVM 클라이언트: 8개 체인의 지갑, 토큰, 가스"
sidebar_label: "Evm"
description: "읽기 전용 EVM 클라이언트: 8개 체인의 지갑, 토큰, 가스"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Evm

8개 체인의 지갑, 토큰, 가스 정보를 조회하는 읽기 전용 EVM 클라이언트입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/blockchain/evm`으로 설치 |
| 경로 | `optional-skills/blockchain/evm` |
| 버전 | `1.0.0` |
| 작성자 | Mibayy (@Mibayy), youssefea (@youssefea), ethernet8023 (@ethernet8023), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `EVM`, `Ethereum`, `BNB`, `BSC`, `Base`, `Arbitrum`, `Polygon`, `Optimism`, `Avalanche`, `zkSync`, `Blockchain`, `Crypto`, `Web3`, `DeFi`, `NFT`, `ENS`, `Whale`, `Security` |
| 관련 스킬 | [`solana`](/docs/user-guide/skills/optional/blockchain/blockchain-solana) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# EVM 블록체인 스킬

USD 가격 정보와 함께 8개 체인의 EVM 호환 블록체인 데이터를 조회합니다.
14개 명령을 제공합니다: 지갑 포트폴리오, 토큰 정보, 트랜잭션, 활동, 가스 추적기,
네트워크 통계, 가격 조회, 멀티체인 스캔, 고래 탐지, ENS 확인, 허용량 검사,
컨트랙트 검사, 트랜잭션 디코딩입니다.

지원하는 체인은 8개입니다: Ethereum, BNB Chain (BSC), Base, Arbitrum One, Polygon,
Optimism, Avalanche (C-Chain), zkSync Era.

API 키가 필요하지 않습니다. 외부 의존성도 없습니다 — Python 표준 라이브러리만 사용합니다
(urllib, json, argparse, threading).

> 독립 실행형 `base` 스킬을 대체합니다. 이전에 `optional-skills/blockchain/base/`에 있던 Base 전용 토큰(AERO, DEGEN,
> TOSHI, BRETT, WELL, cbETH, cbBTC, wstETH, rETH)과 모든 Base RPC 기능이 이 스킬에 통합되었습니다.
> Base 데이터를 조회하려면 모든 명령에 `--chain base`를 전달하세요.

---

## 사용 시점
- 사용자가 EVM 체인의 지갑 잔액이나 포트폴리오를 요청할 때
- 사용자가 동일한 지갑을 모든 체인에서 한 번에 확인하려 할 때
- 사용자가 해시로 트랜잭션을 검사하거나 트랜잭션이 수행한 작업을 디코딩하려 할 때
- 사용자가 ERC-20 토큰 메타데이터, 가격, 공급량 또는 시가총액을 요청할 때
- 사용자가 주소의 최근 트랜잭션 내역을 요청할 때
- 사용자가 현재 가스 가격을 확인하거나 체인별 수수료를 비교하려 할 때
- 사용자가 최근 블록에서 대규모 고래 전송을 찾으려 할 때
- 사용자가 ENS 이름(vitalik.eth)을 확인하거나 주소를 역조회하려 할 때
- 사용자가 컨트랙트에 위험한 토큰 승인이 있는지 확인하려 할 때
- 사용자가 스마트 컨트랙트를 검사하려 할 때(프록시? ERC-20? ERC-721? 바이트코드 크기?)
- 사용자가 트랜잭션 전에 체인별 가스 비용을 비교하려 할 때

---

## 사전 요구 사항
Python 3.8 이상의 표준 라이브러리만 필요합니다. pip 설치는 필요하지 않습니다.
가격 정보: CoinGecko 무료 API(요청 제한, 분당 약 10~30회).
ENS: ensideas.com 공개 API.
트랜잭션 디코딩: 4byte.directory 공개 API.

RPC 엔드포인트 재정의: `export EVM_RPC_URL=https://your-rpc.com`

도우미 스크립트 경로: `~/.hermes/skills/blockchain/evm/scripts/evm_client.py`

---

## 빠른 참조

```
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
네이티브 잔액과 알려진 ERC-20 토큰을 USD 가치순으로 정렬합니다.
```bash
python3 $SCRIPT wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
python3 $SCRIPT wallet 0xd8dA... --chain bsc --no-prices   # faster
```

### 2. 멀티체인 스캔
스레드를 사용해 동일한 주소를 8개 체인에서 동시에 스캔합니다.
```bash
python3 $SCRIPT multichain 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```
출력: 체인별 네이티브 잔액 + 토큰 보유량 + USD 총합.

### 3. 비교(가스 + 가격)
8개 체인을 모두 병렬로 조회합니다. 가장 저렴한 체인과 가장 비싼 체인을 표시합니다.
```bash
python3 $SCRIPT compare
```

### 4. 트랜잭션 세부 정보 및 디코딩
```bash
python3 $SCRIPT tx 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060
python3 $SCRIPT decode 0x5c504ed...   # Shows human-readable function signature
```
디코딩은 4byte.directory를 사용해 0xa9059cbb를 transfer(address,uint256)로 변환합니다.

### 5. ENS 확인
```bash
python3 $SCRIPT ens vitalik.eth          # -> 0xd8dA... + avatar + social links
python3 $SCRIPT ens 0xd8dA...96045       # -> vitalik.eth
```

### 6. 허용량 검사기(보안)
알려진 DEX/브리지 컨트랙트에 부여된 ERC-20 승인을 확인합니다.
```bash
python3 $SCRIPT allowance 0xYourWallet
```
무제한 승인을 높은 위험으로 표시합니다.

### 7. 컨트랙트 검사기
```bash
python3 $SCRIPT contract 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48   # USDC (proxy)
python3 $SCRIPT contract 0xdAC17F958D2ee523a2206206994597C13D831ec7   # USDT (ERC-20)
```
프록시(EIP-1967/EIP-1167), ERC-20, ERC-721, ERC-165를 탐지합니다. 프록시의 경우 바이트코드 크기와 구현 주소를 표시합니다.

### 8. 고래 탐지
```bash
python3 $SCRIPT whale                                    # ETH, last 20 blocks, >$10k
python3 $SCRIPT whale --blocks 50 --min-usd 50000 --chain bsc
```

### 9. 가스 추적기
```bash
python3 $SCRIPT gas
python3 $SCRIPT gas --chain polygon
```
전송, ERC-20 전송, approve, 스왑, NFT 발행, NFT 전송에 대한 gwei 가격과 USD 비용을 표시합니다.

---

## 지원 체인
| 키       | 이름           | 네이티브 | 체인 ID |
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
- CoinGecko 무료 등급은 분당 약 10~30회로 제한됩니다. 더 빠른 지갑 스캔에는 `--no-prices`를 사용하세요.
- 공개 RPC는 요청을 제한할 수 있습니다. 운영 환경에서는 비공개 엔드포인트에 EVM_RPC_URL을 설정하세요.
- `wallet`과 `allowance`는 체인당 알려진 토큰 목록(약 30개)만 확인합니다. 전체 토큰을 찾으려면 블록 탐색기를 사용하세요.
- `activity`는 최근 블록만 스캔합니다(최대 200개). 전체 내역에는 Etherscan API를 사용하세요.
- `multichain`은 8개의 병렬 스레드를 실행하므로 공개 RPC의 요청 제한이 발생할 수 있습니다.
- ENS 확인은 단일 공개 엔드포인트(ensideas.com / ens.vitalik.ca)에 의존하며 대체 경로가 없습니다. 해당 엔드포인트가 중단되면 `ens`가 실패하므로 나중에 다시 실행하거나 블록 탐색기를 사용하세요.
- 트랜잭션 디코딩은 단일 공개 엔드포인트(4byte.directory)에 의존하며 대체 경로가 없습니다. 데이터베이스에 없는 셀렉터는 `unknown`으로 표시됩니다.
- **L2 가스 추정치는 L2 실행 비용만 포함합니다.** Base, Arbitrum, Optimism, zkSync와 같은 롤업에서는 실제 트랜잭션 비용에 calldata 크기와 현재 L1 가스 가격에 따라 달라지는 L1 데이터 게시 수수료도 포함됩니다. `gas` 명령은 이 L1 구성 요소를 추정하지 않습니다. Base의 경우 네트워크 L1 수수료 오라클(컨트랙트 `0x420000000000000000000000000000000000000F`)을 참조하세요.
- 주소와 트랜잭션 해시 입력은 0x 접두사, 올바른 길이, 16진수 여부를 검증하지만 EIP-55 체크섬 대소문자는 강제하지 않습니다(RPC 엔드포인트는 대소문자가 섞인 16진수를 허용합니다).

---

## 검증
```bash
# Should print current block, gas price, ETH price
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py stats

# Should resolve vitalik.eth to 0xd8dA...
python3 ~/.hermes/skills/blockchain/evm/scripts/evm_client.py ens vitalik.eth
```
