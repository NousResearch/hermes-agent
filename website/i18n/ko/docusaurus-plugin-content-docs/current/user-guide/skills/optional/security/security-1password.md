---
title: "1Password — 1Password CLI (op) 설정 및 사용"
sidebar_label: "1Password"
description: "1Password CLI (op) 설정 및 사용"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 1Password

1Password CLI (op)를 설정하고 사용합니다. CLI를 설치하고, 데스크톱 앱 통합을 활성화하고, 로그인하며, 명령어를 위해 비밀정보를 읽고 주입할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/security/1password`로 설치 |
| 경로 | `optional-skills/security/1password` |
| 버전 | `1.0.0` |
| 작성자 | arceus77-7, Hermes Agent에 의해 향상됨 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `security`, `secrets`, `1password`, `op`, `cli` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# 1Password CLI

사용자가 평문 환경 변수나 파일 대신 1Password를 통해 비밀정보를 관리하고자 할 때 이 스킬을 사용합니다.

## 요구 사항

- 1Password 계정
- 1Password CLI (`op`) 설치됨
- 다음 중 하나 필요: 데스크톱 앱 연동, 서비스 계정 토큰 (`OP_SERVICE_ACCOUNT_TOKEN`), 또는 Connect 서버
- 데스크톱 앱 흐름의 경우, Hermes 터미널 호출 시 안정적인 인증 세션을 유지하기 위해 `tmux` 필요

## 언제 사용하나요

- 1Password CLI를 설치하거나 구성할 때
- `op signin`으로 로그인할 때
- `op://Vault/Item/field` 와 같은 비밀정보 참조를 읽을 때
- `op inject`를 사용하여 구성(config)/템플릿에 비밀정보를 주입할 때
- `op run`을 통해 비밀정보 환경 변수가 필요한 명령어를 실행할 때

## 인증 방법

### 서비스 계정 (Hermes에 권장됨)

`~/.hermes/.env` 에 `OP_SERVICE_ACCOUNT_TOKEN`을 설정합니다 (스킬을 처음 로드할 때 메시지를 표시하여 입력하도록 안내합니다).
데스크톱 앱이 필요하지 않습니다. `op read`, `op inject`, `op run`을 지원합니다.

```bash
export OP_SERVICE_ACCOUNT_TOKEN="your-token-here"
op whoami  # 검증 — Type: SERVICE_ACCOUNT 가 표시되어야 합니다
```

### 데스크톱 앱 통합 (대화형)

1. 1Password 데스크톱 앱에서 활성화: 설정(Settings) → 개발자(Developer) → 1Password CLI 연동(Integrate with 1Password CLI)
2. 앱이 잠금 해제되어 있는지 확인
3. `op signin`을 실행하고 생체 인식 프롬프트 승인

### Connect 서버 (자체 호스팅)

```bash
export OP_CONNECT_HOST="http://localhost:8080"
export OP_CONNECT_TOKEN="your-connect-token"
```

## 설정

1. CLI 설치:

```bash
# macOS
brew install 1password-cli

# Linux (공식 패키지/설치 문서 참조)
# 배포판별 링크는 references/get-started.md를 확인하세요.

# Windows (winget)
winget install AgileBits.1Password.CLI
```

2. 검증:

```bash
op --version
```

3. 위의 인증 방법 중 하나를 선택하고 구성하십시오.

## Hermes 실행 패턴 (데스크톱 앱 흐름)

Hermes 터미널 명령은 기본적으로 비대화형(non-interactive)이며 호출 사이에 인증 컨텍스트를 잃을 수 있습니다.
데스크톱 앱 연동으로 안정적인 `op` 사용을 원한다면 전용 tmux 세션 내에서 로그인 및 비밀정보 관련 작업을 실행하십시오.

참고: `OP_SERVICE_ACCOUNT_TOKEN`을 사용하는 경우에는 필요하지 않습니다 — 터미널 호출 간에 토큰이 자동으로 유지됩니다.

```bash
SOCKET_DIR="${TMPDIR:-/tmp}/hermes-tmux-sockets"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/hermes-op.sock"
SESSION="op-auth-$(date +%Y%m%d-%H%M%S)"

tmux -S "$SOCKET" new -d -s "$SESSION" -n shell

# 로그인 (안내 시 데스크톱 앱에서 승인)
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "eval \"\$(op signin --account my.1password.com)\"" Enter

# 인증 검증
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op whoami" Enter

# 읽기 예시
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op read 'op://Private/Npmjs/one-time password?attribute=otp'" Enter

# 필요할 때 출력 캡처
tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200

# 정리(Cleanup)
tmux -S "$SOCKET" kill-session -t "$SESSION"
```

## 주요 작업 (Common Operations)

### 비밀정보 읽기

```bash
op read "op://app-prod/db/password"
```

### OTP 가져오기

```bash
op read "op://app-prod/npm/one-time password?attribute=otp"
```

### 템플릿에 주입하기

```bash
echo "db_password: {{ op://app-prod/db/password }}" | op inject
```

### 비밀정보 환경변수를 사용하여 명령 실행

```bash
export DB_PASSWORD="op://app-prod/db/password"
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set" || echo "DB_PASSWORD missing"'
```

## 가드레일 (Guardrails)

- 사용자가 명시적으로 값을 요구하지 않는 한 생(raw) 비밀정보를 다시 출력하지 마십시오.
- 비밀정보를 파일에 작성하는 것보다 `op run` / `op inject` 사용을 선호하십시오.
- 명령어가 "account is not signed in" 오류로 실패하는 경우, 동일한 tmux 세션 내에서 다시 `op signin`을 실행하십시오.
- 데스크톱 앱 통합을 사용할 수 없는 환경(헤드리스/CI)이라면 서비스 계정 토큰 방식의 흐름을 이용하십시오.

## CI / 헤드리스(Headless) 환경에 대한 참고사항

비대화형(non-interactive) 사용을 위해서는 대화형 `op signin`을 피하고 `OP_SERVICE_ACCOUNT_TOKEN`으로 인증하십시오.
서비스 계정은 CLI v2.18.0 이상이 필요합니다.

## 참조 문서

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/
