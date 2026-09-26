---
title: "Stripe Link Cli — Stripe Link을 통한 에이전트 결제 — 카드, SPT, 승인"
sidebar_label: "Stripe Link Cli"
description: "Stripe Link을 통한 에이전트 결제 — 카드, SPT, 승인"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Stripe Link Cli

Stripe Link을 통한 에이전트 결제 — 카드, SPT, 승인.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/payments/stripe-link-cli`로 설치 |
| 경로 | `optional-skills/payments/stripe-link-cli` |
| 버전 | `0.1.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `Payments`, `Stripe`, `Link`, `Checkout`, `MPP` |
| 관련 skill | [`mpp-agent`](/docs/user-guide/skills/optional/payments/payments-mpp-agent), [`stripe-projects`](/docs/user-guide/skills/optional/payments/payments-stripe-projects) |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# Stripe Link CLI Skill

[@stripe/link-cli](https://github.com/stripe/link-cli)를 래핑하여 Hermes가 일회용 가상 카드 또는 Shared Payment Token (SPT)을 사용해 사용자를 대신하여 구매를 완료할 수 있게 합니다. 모든 지출은 Link 모바일/웹 앱에서 인앱 승인으로 제한되며 — Hermes는 스스로 승인할 수 없습니다.

현재 미국에서만 사용할 수 있습니다 (Link 계정 필요). Windows는 업스트림 CLI에서 지원되지 않으므로 이 skill은 `[linux, macos]`로 제한됩니다.

## 사용 시점

트리거 문구:

- "X를 사 줘", "X를 결제해 줘", "구매해 줘", "결제를 완료해 줘"
- "카드를 받아 줘", "결제 수단이 필요해"
- "Link에 로그인해 줘", "내 Link 지갑을 연결해 줘"
- `www-authenticate: ... method="stripe"`가 포함된 상인의 HTTP 402 응답

사용자가 유료 API 호출을 원하고 (HTTP 402, 결제 양식 없음), 결제 챌린지에 `method="stripe"`가 포함되어 있으면 `card` 경로가 아니라 이 skill을 통한 SPT를 사용하거나 `mpp-agent` skill로 넘깁니다.

## 사전 요구 사항

- `PATH`에서 Node.js 20+ 사용 가능 (`node --version`)
- 미국 기반 (Link 계정 필요)

Hermes가 결제를 시도하기 전에 Link 계정, 결제 수단, 지출 승인 앱을 설정할 필요는 없습니다 — CLI가 첫 실행 시 다음 과정을 안내합니다:

- https://app.link.com의 Link 계정 — 최초 `link-cli` 인증 중 생성/연결
- 하나 이상의 결제 수단 — 최초 실행 중 https://app.link.com/wallet에서 추가
- Link 모바일/웹 앱 — 최초 지출 요청이 생성되면 이를 열어 승인

환경 변수는 필요하지 않습니다 — 인증 상태는 CLI가 자체 설정 디렉터리에 로컬로 저장합니다.

## 설치

한 번 전역으로 설치합니다:

```
npm install -g @stripe/link-cli
```

또는 `npx @stripe/link-cli`로 임시 실행합니다. 아래 skill에서는 설치된 `link-cli` 형식을 사용합니다.

## 실행 방법

모든 명령은 `terminal` 도구를 통해 실행합니다. CLI는 TTY가 아닌 호출자를 자동으로 감지하고 기본적으로 간결한 `toon` 출력을 생성하므로 모델에서 사용하기에 적합합니다. 단계에 구조화된 필드가 필요하면 `--format json`을 전달합니다.

명령 검색: `link-cli --llms-full`.
호출하기 전에 명령의 스키마 확인: `link-cli <command> --schema`.

## 절차

### 1. 인증 확인 / 설정

```
link-cli auth status
```

인증되지 않았다면 명확한 클라이언트 이름으로 로그인합니다 (이 레이블은 사용자의 Link 앱에 표시됩니다):

```
link-cli auth login --client-name "Hermes" --interval 5 --timeout 300
```

`--interval`/`--timeout` 형식은 인라인으로 폴링하므로 에이전트가 `_next` 단계를 관리할 필요가 없습니다. 확인 URL과 문구를 사용자에게 출력하고 CLI가 반환할 때까지 기다립니다.

**`auth status`에서 로그인이 확인될 때까지 이 단계를 지나 진행하지 마세요.**

### 2. 지출 요청 생성 전 상인 평가

자격 증명 유형을 결정합니다:

| 상인 화면 | `--credential-type` |
|---|---|
| 일반 웹 결제 양식 / Stripe Elements | `card` (기본값) |
| `method="stripe"`가 포함된 `www-authenticate`와 함께 HTTP 402 반환 | `shared_payment_token` |
| `method="stripe"`가 없는 HTTP 402 반환 | 지원되지 않음 — 중지 |

402 응답의 경우 챌린지를 직접 디코딩하지 마세요. 원본 헤더를 전달합니다:

```
link-cli mpp decode --challenge '<full WWW-Authenticate header>'
```

이 명령은 챌린지를 검증하고 네트워크 ID와 디코딩된 요청 본문을 추출합니다.

### 3. 결제 수단 + 배송지 나열

```
link-cli payment-methods list
link-cli shipping-address list
```

사용자가 지정하지 않는 한 첫 번째 항목을 사용합니다. `payment-methods list`의 `id`가 다음 단계에서 사용할 `--payment-method-id`입니다.

### 4. 지출 요청 생성

이 명령을 실행하기 전에 사용자에게 최종 합계를 확인합니다. 금액은 센트 단위입니다.

```
link-cli spend-request create \
  --payment-method-id <pm_id> \
  --merchant-name "<name>" \
  --merchant-url "<url>" \
  --context "<one sentence: what is being purchased and why>" \
  --amount <cents> \
  --line-item "name:<item>,unit_amount:<cents>,quantity:1" \
  --total "type:total,display_text:Total,amount:<cents>" \
  --request-approval
```

MPP 상인의 경우 `--credential-type shared_payment_token`을 추가합니다.

`--request-approval`은 사용자의 Link 앱에 알림을 보내고 사용자가 승인하거나 거부할 때까지 폴링합니다. 거부/시간 초과 시 CLI는 0이 아닌 종료 코드를 반환합니다.

### 5. 자격 증명 가져오기 — 안전하게

**카드 정보를 stdout에 출력하지 마세요.** `--output-file`을 사용해 PAN이 에이전트의 대화 기록이나 로그에 들어가지 않도록 합니다:

```
link-cli spend-request retrieve <lsrq_id> \
  --include card \
  --output-file /tmp/link-card.json \
  --format json
```

파일은 권한 `0600`으로 작성됩니다. stdout에는 마스킹된 필드 (브랜드, 마지막 4자리, 만료일)와 `card_output_file` 경로만 표시됩니다.

### 6. 자격 증명 사용

- 웹 결제의 경우 파일 경로를 사용자에게 전달하거나, 디스크에서 직접 양식을 채우는 브라우저 구동 도구에 전달합니다. 카드 파일을 에이전트의 추론 컨텍스트에 넣기 위해 `read_file` 또는 `cat`을 사용하지 마세요.
- MPP 상인의 경우:

  ```
  link-cli mpp pay <merchant-url> \
    --spend-request-id <lsrq_id> \
    --method POST \
    --data '<json body>'
  ```

### 7. 정리

구매가 끝나는 즉시 카드 파일을 삭제합니다:

```
rm -f /tmp/link-card.json
```

## 선택 사항: 대신 MCP 서버로 실행

`@stripe/link-cli --mcp`는 동일한 명령을 stdio를 통한 MCP 도구로 노출합니다. Hermes의 기본 MCP에 등록하려면 다음을 실행합니다:

```
hermes mcp add stripe-link --command "npx" --args "@stripe/link-cli --mcp"
```

그런 다음 `hermes mcp list`에 `stripe-link`이 표시되어야 합니다. 동일한 승인 규칙이 적용됩니다 — MCP도 Link 앱 승인 단계를 우회하지 않습니다.

## 주의 사항

- **미국 전용입니다.** 미국 외 지역에서는 `auth login`이 실패합니다. 사용자에게 알리고 계속 재시도하지 마세요.
- **카드 PAN이 에이전트 컨텍스트에 들어가서는 안 됩니다.** 매번 `--output-file`을 사용하세요. 이미 이 옵션 없이 가져왔다면 즉시 `link-cli auth logout`을 실행하는 것만으로는 충분하지 않습니다 — 카드는 일회용이지만 보안 위생을 위해 교체가 중요합니다.
- **`--request-approval`은 사용자가 조치할 때까지 차단됩니다.** 사용자가 잠들어 있으면 CLI가 시간 초과에 도달합니다. 미리 기대 시간을 안내하세요.
- **다단계 `_next` 명령.** 일부 명령은 계속 진행하려면 실행해야 하는 `_next.command`를 반환합니다. 확실하지 않다면 인라인 폴링 플래그 (`--interval`/`--timeout`)를 우선 사용합니다.
- **비 TTY 모드의 기본 출력 형식은 `toon`입니다.** 일반적인 문장에는 적합하지만, 다음 단계에서 특정 필드를 파싱해야 한다면 `--format json`을 전달합니다.
- **`card`를 기본으로 선택하지 마세요.** 상인 평가 단계 (섹션 2)가 존재하는 이유는 잘못된 자격 증명 유형을 선택하면 구매가 조용히 실패하거나 필요한 것보다 많은 데이터가 유출될 수 있기 때문입니다.

## 검증

```
link-cli --version && link-cli auth status
```

종료 코드 0은 설치 및 로그인 완료를 의미합니다.
