---
title: "Stripe Projects — SaaS 서비스 프로비저닝 + Stripe Projects를 통한 자격 증명 동기화"
sidebar_label: "Stripe Projects"
description: "SaaS 서비스 프로비저닝 + Stripe Projects를 통한 자격 증명 동기화"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stripe Projects

SaaS 서비스를 프로비저닝하고 Stripe Projects를 통해 자격 증명을 동기화합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/payments/stripe-projects`로 설치 |
| 경로 | `optional-skills/payments/stripe-projects` |
| 버전 | `0.1.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `Payments`, `Stripe`, `Projects`, `Provisioning`, `Infrastructure` |
| 관련 스킬 | [`stripe-link-cli`](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli), [`mpp-agent`](/docs/user-guide/skills/optional/payments/payments-mpp-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# Stripe Projects 스킬

[Stripe Projects](https://projects.dev) CLI 플러그인을 래핑하여 Hermes가 SaaS 서비스(Neon, Twilio, Vercel 등)를 프로비저닝하고, 자격 증명을 생성하여 사용자의 `.env`에 동기화하며, 여러 공급자의 결제를 한 곳에서 관리할 수 있게 합니다.

Windows에서 결제 클러스터가 성숙하는 동안에는 `[linux, macos]`로 제한됩니다. Stripe CLI 자체는 크로스 플랫폼이지만 이 제한은 클러스터의 운영 방침이며 엄격한 한계는 아닙니다.

## 사용 시점

트리거 문구:

- "&lt;provider> 설정해줘", "&lt;Neon|Twilio|Vercel|...> 프로비저닝해줘", "데이터베이스를 만들어줘"
- "이 프로젝트에 &lt;Postgres|Redis|Twilio number|...>를 제공해줘"
- "내 스택 자격 증명을 관리해줘", "이 키를 교체해줘", "내 플랜을 업그레이드해줘"
- "추가할 수 있는 공급자는 무엇인가요?"

사용자에게 이미 공급자 계정이 있는 경우에도 `stripe projects link <provider>`로 연결할 수 있습니다. 사용자가 기존 데이터베이스나 Vercel 프로젝트처럼 기존 공급자 리소스를 사용하려는 경우에는 먼저 공급자 지원 여부를 확인하세요. 현재 많은 공급자가 새 리소스 프로비저닝은 지원하지만 기존 리소스 가져오기는 지원하지 않습니다.

## 사전 요구 사항

- Stripe CLI 설치(macOS에서는 Homebrew, Linux에서는 패키지 관리자 또는 https://docs.stripe.com/stripe-cli/install에서 다운로드)
- Stripe Projects 플러그인 설치
- Stripe 계정. 아직 계정이 없다면 CLI가 설정 중 브라우저에서 로그인 또는 계정 생성을 안내할 수 있습니다.

## 설치

macOS:

```
brew install stripe/stripe-cli/stripe
stripe plugin install projects
```

Linux: https://docs.stripe.com/stripe-cli/install에서 플랫폼별 설치 방법을 따른 다음 실행합니다.

```
stripe plugin install projects
```

## 실행 방법

모든 명령은 사용자의 프로젝트 디렉터리 안에서 `terminal` 도구를 통해 실행합니다(CLI는 CWD에 `.env`와 `.projects/vault/vault.json`을 작성합니다).

## 절차

### 1. 프로젝트 초기화

```
cd <project-root>
stripe projects init
```

이 명령은 `.projects/vault/vault.json`(암호화된 자격 증명 저장소)을 생성하고 프로젝트가 공급자를 받을 수 있도록 준비합니다.

### 2. 사용 가능한 공급자 검색

```
stripe projects catalog
```

Stripe Projects가 지원하는 모든 공급자(데이터베이스, 호스팅, 인증, AI, 분석, 메시징 등)를 나열합니다.

### 3. 서비스 추가

```
stripe projects add <provider>/<service>
```

예시:

- `stripe projects add neon/postgres`
- `stripe projects add twilio/sms`
- `stripe projects add runloop/sandbox`

CLI는 공급자와 사용자의 계정에 서비스를 프로비저닝하고, 자격 증명을 생성하여 `.env`에 동기화하며, 저장소에 리소스를 기록합니다. 사용자는 등급 선택 또는 가격 확인 메시지에 응답해야 할 수 있습니다.

### 4. 확인

```
stripe projects list
```

새로 추가된 공급자와 해당 `.env` 키가 표시되어야 합니다.

### 5. 관리 / 업그레이드 / 제거

```
stripe projects upgrade <provider>     # tier change
stripe projects remove <provider>      # deprovision
stripe projects rotate <provider>      # rotate credentials
```

## 주의할 점

- **`.env` 쓰기는 실제 쓰기 작업입니다.** CLI는 프로젝트 루트의 `.env`에 내용을 추가합니다. 사용자의 `.env`가 (일반적인 경우처럼) gitignore에 포함되어 있으면 키가 안전하게 저장되지만 그렇지 않으면 이 스킬이 자격 증명 유출 경로가 될 수 있습니다. 항상 먼저 `.gitignore`를 확인하세요.
- **프로젝트별 상태.** `.projects/vault/vault.json`은 프로젝트별 파일입니다. 서로 다른 두 프로젝트에서 같은 서비스를 프로비저닝하면 서로 다른 리소스 두 개와 청구서 두 개가 생성됩니다.
- **결제는 Stripe 측에서 발생합니다.** `add`/`upgrade` 중 표시되는 등급 확인 메시지는 실제 청구로 이어집니다. 확인하기 전에 사용자에게 이를 알려야 합니다.
- **공급자 사용 가능 여부는 바뀝니다.** 카탈로그는 확장됩니다. 사용자가 지정한 공급자가 목록에 없다면 `add` 호출에 실패하기 전에 `stripe projects catalog | grep <name>`을 먼저 실행하세요.
- **저장소의 자격 증명은 암호화되지만 `.env`는 평문입니다.** 일반적인 `.env` 위생 규칙을 적용하세요 — 절대로 커밋하지 마세요.
- **서비스를 제거해도 기반 리소스가 항상 삭제되는 것은 아닙니다.** 일부 공급자는 일시 중지되거나 휴면 상태인 리소스를 남깁니다. 비용이 높은 서비스(특히 관리형 데이터베이스)는 `remove` 후 공급자 자체 대시보드에서 확인하세요.

## 검증

```
stripe projects --version && stripe projects list
```

초기화된 프로젝트에서 종료 코드 0이 반환되면 플러그인이 정상적으로 작동하는 것입니다.
