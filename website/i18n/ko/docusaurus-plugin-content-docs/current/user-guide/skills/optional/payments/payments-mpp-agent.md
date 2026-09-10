---
title: "Mpp Agent — Machine Payments Protocol(MPP)을 통해 HTTP 402 API 결제"
sidebar_label: "Mpp Agent"
description: "Machine Payments Protocol(MPP)을 통해 HTTP 402 API 결제"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Mpp Agent

Machine Payments Protocol(MPP)을 통해 HTTP 402 API를 결제합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/payments/mpp-agent`로 설치 |
| 경로 | `optional-skills/payments/mpp-agent` |
| 버전 | `0.1.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `Payments`, `MPP`, `HTTP-402`, `Tempo`, `Stripe` |
| 관련 스킬 | [`stripe-link-cli`](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli), [`stripe-projects`](/docs/user-guide/skills/optional/payments/payments-stripe-projects) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보는 내용입니다.
:::

# MPP Agent 스킬

Machine Payments Protocol(MPP, https://mpp.dev) 클라이언트를 래핑하여 `HTTP 402 Payment Required`로 응답하는 서버에 요청별 API 액세스 비용을 Hermes가 결제할 수 있도록 합니다.

세 가지 클라이언트 옵션은 모두 npm을 통해 배포됩니다. 사용자의 필요를 해결하는 가장 가벼운 옵션을 선택하세요. 더 폭넓은 결제 도구가 Windows에서 성숙할 때까지 `[linux, macos]`로 제한됩니다.

## 사용 시점

- 판매자 API가 `www-authenticate` 헤더와 함께 `HTTP 402`를 반환하고, 사용자가 응답을 단순히 기록하는 것이 아니라 실제로 결제하려는 경우
- 사용자가 “요청별 결제”, “에이전트 지갑 설정”, “Tempo / Privy / AgentCash 사용”을 요청하거나 MPP 가격이 적용된 서비스를 탐색하려는 경우
- Stripe Link 결제로 Shared Payment Token(SPT)이 생성되었고 에이전트가 이를 402 챌린지에 첨부해야 하는 경우. 이 흐름에서는 `link-cli mpp pay`를 우선 사용합니다(`stripe-link-cli` 스킬 참조).

## 클라이언트 선택

| 도구 | 시점 | 설정 |
|---|---|---|
| `link-cli` | 사용자가 이미 Stripe Link를 설정했거나 402 챌린지가 `method="stripe"`를 광고하는 경우 | `stripe-link-cli` 스킬 참조 |
| Tempo Wallet | 지출 제어 및 서비스 탐색 기능이 있는 MPP 서비스 | `tempo wallet login` |
| Privy Agent CLI | 멀티체인 지갑, 브라우저 기반 자금 충전 | `privy-agent-wallets login` |
| AgentCash | 하나의 USDC.e 잔액으로 300개 이상의 사전 가격 책정 API 사용 | `npx agentcash onboard` |
| `mppx` | 개발 및 디버깅, 가장 작은 의존성 범위 | `npm install -g mppx` 후 `mppx account create` |

기본값: 사용자가 이미 Stripe Link를 구성했거나 402 챌린지가 `method="stripe"`를 지정하면 `link-cli mpp pay`(`stripe-link-cli` 스킬)를 사용합니다. 그 외에는 일회성 결제 호출과 디버깅에는 `mppx`를, 사용자가 지속적인 지출 제어를 원하면 Tempo Wallet을 사용합니다.

## 사전 요구 사항

- `PATH`에 Node.js 20 이상
- 자금이 충전된 지갑(Tempo / Privy / AgentCash) 또는 `mppx` 계정
- Tempo / Privy / AgentCash의 경우 각 온보딩 스킬을 따릅니다.
  - `https://tempo.xyz/SKILL.md`
  - `https://agents.privy.io/skill.md`
  - `https://agentcash.dev/skill.md`

사용자가 하나를 선택하면 `web_extract`를 사용해 해당 SKILL.md를 가져옵니다.

## 절차(mppx, 가장 빠른 경로)

모든 명령은 `terminal` 도구를 통해 실행합니다.

### 1. 설치 및 계정 생성

```
npm install -g mppx
mppx account create
```

결과로 생성된 계정 인증 정보는 CLI가 안내하는 위치에 저장합니다(CLI가 자체 구성 아래에 기록하므로 에이전트 트랜스크립트에 붙여 넣지 마세요).

### 2. 판매자의 402 챌린지 검사

사용자가 URL을 제공하면 먼저 해당 URL을 확인하여 실제로 MPP를 사용하는지 검증합니다.

```
curl -i <url>
```

실제 MPP 402는 다음과 같은 형태입니다.

```
HTTP/1.1 402 Payment Required
www-authenticate: tempo amount=0.1 currency=...
```

### 3. 요청 결제

```
mppx <url>
```

GET이 아닌 메서드나 요청 본문을 사용하는 경우:

```
mppx <url> --method POST --data '<json>'
```

`mppx`는 402 챌린지/인증 정보 교환을 자동으로 처리하고, 성공하면 판매자의 실제 응답을 출력합니다.

### 4. 영수증 검증

`mppx`는 영수증 헤더를 자동으로 첨부합니다. 검사하려면:

```
mppx <url> -v
```

## 절차(Tempo Wallet)

Tempo Wallet 스킬(https://tempo.xyz/SKILL.md)이 표준 참고 자료이므로 `web_extract`로 가져와 따릅니다. 핵심 명령은 다음과 같습니다.

```
tempo wallet login
tempo wallet pay <url>
```

지출 제어 및 서비스 탐색은 https://wallet.tempo.xyz의 지갑 UI에서 제공합니다.

## 주의할 점

- **`method="stripe"`가 없는 `HTTP 402`는 Stripe Link로 결제할 수 없습니다.** 챌린지가 Tempo나 다른 방법만 광고하면 `mppx`(또는 일치하는 지갑)를 사용합니다. 반대로 `method="stripe"`를 광고하면 사용자가 승인한 카드로 지출이 처리되도록 `stripe-link-cli` 스킬을 통한 Link를 우선 사용합니다.
- **하나의 헤더에 여러 챌린지가 있을 수 있습니다.** `www-authenticate`에 여러 방법(예: `tempo, stripe`)이 나열될 수 있습니다. Link CLI의 `mpp decode`는 Stripe 항목을 선택하고, `mppx`는 Tempo를 선택합니다. 단 하나의 “올바른” 클라이언트가 있는 것은 아니며, 사용자가 자금을 충전한 지갑에 따라 선택합니다.
- **금액이 0인 챌린지.** 일부 MPP 엔드포인트는 `$0.00`을 청구하고 증명 인증 정보만 요구합니다. 자금이 충전된 지갑 없이도 작동합니다. 이를 “고장 난” 것으로 간주하여 거부하지 마세요.
- **지갑 키는 에이전트 컨텍스트에 들어가지 않습니다.** 네 클라이언트 모두 자체 구성 디렉터리에 키를 저장합니다(또는 Privy의 경우 세션별 임시 키 쌍을 생성합니다). 키를 `cat`/`read_file`하지 마세요.
- **서버 측 MPP는 별도의 스킬입니다.** 사용자가 자신의 API에 402를 추가하려는 경우 이 스킬은 적합하지 않습니다. https://mpp.dev/quickstart/server 및 `mppx/nextjs` / `mppx/hono` / `mppx/express` / `mppx/elysia` 미들웨어를 안내하세요. 전용 `mpp-server` 스킬은 추후 추가될 수 있습니다.

## 검증

```
mppx --version && mppx account list
```

종료 코드 0은 설치되어 있고 계정이 존재한다는 의미입니다.
