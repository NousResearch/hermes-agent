---
title: "1Password — op CLI 설정, 로그인, 시크릿 읽기 또는 주입"
sidebar_label: "1Password"
description: "op CLI를 설정하고 로그인한 후 시크릿을 읽거나 주입"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 1Password CLI

1Password CLI를 설정하고 로그인한 후 시크릿을 읽거나 주입합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/security/1password`로 설치 |
| 경로 | `optional-skills/security/1password` |
| 버전 | `1.0.0` |
| 작성자 | arceus77-7, Hermes Agent가 개선 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `security`, `secrets`, `1password`, `op`, `cli` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 내용입니다.
:::

# 1Password CLI

사용자가 평문 환경 변수나 파일 대신 1Password를 통해 시크릿을 관리하려는 경우 이 스킬을 사용합니다.

## 요구 사항

- 1Password 계정
- 1Password CLI(`op`) 설치
- 다음 중 하나: 데스크톱 앱 통합, 서비스 계정 토큰(`OP_SERVICE_ACCOUNT_TOKEN`) 또는 Connect 서버
- Hermes 터미널 호출 중 안정적인 인증 세션을 위해 `tmux` 사용 가능(데스크톱 앱 방식만 해당)

## 사용 시점

- 1Password CLI 설치 또는 구성
- `op signin`으로 로그인
- `op://Vault/Item/field`와 같은 시크릿 참조 읽기
- `op inject`를 사용해 구성/템플릿에 시크릿 주입
- `op run`을 사용해 시크릿 환경 변수로 명령 실행

## 인증 방법

### 서비스 계정(Hermes에 권장)

`${HERMES_HOME:-~/.hermes}/.env`에 `OP_SERVICE_ACCOUNT_TOKEN`을 설정합니다(처음 로드할 때 이 스킬이 이를 요청합니다).
데스크톱 앱이 필요하지 않습니다. `op read`, `op inject`, `op run`을 지원합니다.

```bash
export OP_SERVICE_ACCOUNT_TOKEN="your-token-here"
op whoami  # verify — should show Type: SERVICE_ACCOUNT
```

### 데스크톱 앱 통합(대화형)

1. 1Password 데스크톱 앱에서 활성화: Settings → Developer → Integrate with 1Password CLI
2. 앱이 잠금 해제되어 있는지 확인
3. `op signin`을 실행하고 생체 인증 프롬프트 승인

### Connect 서버(자체 호스팅)

```bash
export OP_CONNECT_HOST="http://localhost:8080"
export OP_CONNECT_TOKEN="your-connect-token"
```

## 설정

1. CLI 설치:

```bash
# macOS
brew install 1password-cli

# Linux (official package/install docs)
# See references/get-started.md for distro-specific links.

# Windows (winget)
winget install AgileBits.1Password.CLI
```

2. 확인:

```bash
op --version
```

3. 위의 인증 방법 중 하나를 선택하고 구성합니다.

## Hermes 실행 패턴(데스크톱 앱 방식)

Hermes 터미널 명령은 기본적으로 비대화형이며 호출 간 인증 컨텍스트가 사라질 수 있습니다.
데스크톱 앱 통합을 사용해 `op`를 안정적으로 사용하려면 전용 tmux 세션 안에서 로그인과 시크릿 작업을 실행합니다.

참고: `OP_SERVICE_ACCOUNT_TOKEN`을 사용할 때는 이 작업이 필요하지 않습니다 — 토큰이 터미널 호출 간 자동으로 유지됩니다.

```bash
SOCKET_DIR="${TMPDIR:-/tmp}/hermes-tmux-sockets"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/hermes-op.sock"
SESSION="op-auth-$(date +%Y%m%d-%H%M%S)"

tmux -S "$SOCKET" new -d -s "$SESSION" -n shell

# Sign in (approve in desktop app when prompted)
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "eval \"\$(op signin --account my.1password.com)\"" Enter

# Verify auth
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op whoami" Enter

# Example read
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op read 'op://Private/Npmjs/one-time password?attribute=otp'" Enter

# Capture output when needed
tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200

# Cleanup
tmux -S "$SOCKET" kill-session -t "$SESSION"
```

## 일반 작업

### 시크릿 읽기

```bash
op read "op://app-prod/db/password"
```

### OTP 가져오기

```bash
op read "op://app-prod/npm/one-time password?attribute=otp"
```

### 템플릿에 주입

```bash
echo "db_password: {{ op://app-prod/db/password }}" | op inject
```

### 시크릿 환경 변수로 명령 실행

```bash
export DB_PASSWORD="op://app-prod/db/password"
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set" || echo "DB_PASSWORD missing"'
```

## 보호 장치

- 사용자가 값을 명시적으로 요청하지 않는 한 원시 시크릿을 사용자에게 다시 출력하지 마세요.
- 시크릿을 파일에 기록하는 대신 `op run` / `op inject`를 우선 사용하세요.
- 명령이 "account is not signed in"으로 실패하면 같은 tmux 세션에서 `op signin`을 다시 실행하세요.
- 데스크톱 앱 통합을 사용할 수 없는 경우(헤드리스/CI) 서비스 계정 토큰 방식을 사용하세요.

## CI / 헤드리스 참고

비대화형 사용에서는 `OP_SERVICE_ACCOUNT_TOKEN`으로 인증하고 대화형 `op signin`을 피하세요.
서비스 계정에는 CLI v2.18.0 이상이 필요합니다.

## 참고 자료

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/
