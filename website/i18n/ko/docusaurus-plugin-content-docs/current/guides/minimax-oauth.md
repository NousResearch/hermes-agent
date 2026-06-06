---
sidebar_position: 15
title: "MiniMax OAuth"
description: "브라우저 OAuth를 통해 MiniMax에 로그인하고 Hermes Agent에서 MiniMax-M2.7 모델 사용하기 — API 키 불필요"
---

# MiniMax OAuth

Hermes Agent는 [MiniMax 포털](https://www.minimax.io)과 동일한 자격 증명을 사용하는 브라우저 기반 OAuth 로그인 흐름을 통해 **MiniMax**를 지원합니다. API 키나 신용카드가 필요하지 않습니다 — 한 번 로그인하면 Hermes가 세션을 자동으로 갱신합니다.

트랜스포트는 `anthropic_messages` 어댑터를 재사용하므로(MiniMax는 `/anthropic`에서 Anthropic Messages와 호환되는 엔드포인트를 노출합니다), 어댑터 변경 없이 모든 기존 도구 호출(tool-calling), 스트리밍 및 컨텍스트 기능이 작동합니다.

## 개요

| 항목 | 값 |
|------|-------|
| 제공자 ID | `minimax-oauth` |
| 표시 이름 | MiniMax (OAuth) |
| 인증 방식 | Browser OAuth (PKCE 리디렉션 흐름) |
| 트랜스포트 | Anthropic Messages 호환 (`anthropic_messages`) |
| 모델 | `MiniMax-M2.7`, `MiniMax-M2.7-highspeed` |
| 글로벌 엔드포인트 | `https://api.minimax.io/anthropic` |
| 중국 엔드포인트 | `https://api.minimaxi.com/anthropic` |
| 환경 변수 필요 여부 | 아니오 (이 제공자에서는 `MINIMAX_API_KEY`를 사용하지 **않음**) |

## 사전 요구 사항

- Python 3.9+
- Hermes Agent 설치됨
- [minimax.io](https://www.minimax.io) (글로벌) 또는 [minimaxi.com](https://www.minimaxi.com) (중국)의 MiniMax 계정
- 로컬 컴퓨터에 브라우저 사용 가능 (또는 원격 세션의 경우 `--no-browser` 사용)

## 빠른 시작

```bash
# 제공자 및 모델 선택기 실행
hermes model
# → 제공자 목록에서 "MiniMax (OAuth)" 선택
# → Hermes가 브라우저를 열어 MiniMax 권한 부여 페이지로 이동
# → 브라우저에서 액세스 승인
# → 모델 선택 (MiniMax-M2.7 또는 MiniMax-M2.7-highspeed)
# → 채팅 시작

hermes
```

첫 번째 로그인 후, 자격 증명은 `~/.hermes/auth.json`에 저장되며 매 세션 전에 자동으로 갱신됩니다.

## 수동으로 로그인하기

모델 선택기를 거치지 않고 로그인을 트리거할 수 있습니다:

```bash
hermes auth add minimax-oauth
```

### 중국 리전 (China region)

계정이 중국 플랫폼(`minimaxi.com`)에 있는 경우, API 키 기반의 `minimax-cn` 제공자를 대신 사용하세요 — `minimax-cn`은 `auth_type="api_key"`로만 등록되어 있습니다(OAuth 흐름 없음). `MINIMAX_CN_API_KEY`(선택적으로 `MINIMAX_CN_BASE_URL`도)를 직접 구성하세요:

```bash
echo 'MINIMAX_CN_API_KEY=your-key' >> ~/.hermes/.env
```

### 원격 / 헤드리스 세션

브라우저를 사용할 수 없는 서버나 컨테이너의 경우:

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes가 인증 URL과 사용자 코드를 출력합니다 — 임의의 기기에서 URL을 열고 프롬프트가 표시될 때 코드를 입력하세요.

## OAuth 흐름

Hermes는 MiniMax OAuth 엔드포인트에 대해 PKCE 브라우저 OAuth 흐름을 구현합니다:

1. Hermes는 PKCE verifier/challenge 쌍과 임의의 상태(state) 값을 생성합니다.
2. `{base_url}/oauth/code`에 challenge와 함께 POST 요청을 보내고 `user_code`와 `verification_uri`를 받습니다.
3. 브라우저가 `verification_uri`를 엽니다. 메시지가 나타나면 `user_code`를 입력합니다.
4. Hermes는 토큰이 도착할 때까지(또는 기한이 지날 때까지) `{base_url}/oauth/token`을 폴링합니다.
5. 토큰(`access_token`, `refresh_token`, expiry)은 `minimax-oauth` 키 아래의 `~/.hermes/auth.json`에 저장됩니다.

토큰 갱신(표준 OAuth `refresh_token` 부여)은 액세스 토큰의 만료가 60초 이내로 남았을 때 각 세션이 시작될 때 자동으로 실행됩니다.

## 로그인 상태 확인

```bash
hermes doctor
```

`◆ Auth Providers` 섹션에 다음과 같이 표시됩니다:

```
✓ MiniMax OAuth  (logged in, region=global)
```

또는 로그인하지 않은 경우:

```
⚠ MiniMax OAuth  (not logged in)
```

## 모델 전환하기

```bash
hermes model
# → "MiniMax (OAuth)" 선택
# → 모델 목록에서 선택
```

또는 모델을 직접 설정하세요:

```bash
hermes config set model.default MiniMax-M2.7
hermes config set model.provider minimax-oauth
```

## 구성 레퍼런스

로그인 후 `~/.hermes/config.yaml`에는 다음과 유사한 항목이 포함됩니다:

```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```

### 리전 엔드포인트

| 제공자 id | 포털 | 추론 엔드포인트 |
|-------------|--------|-------------------|
| `minimax-oauth` (글로벌) | `https://api.minimax.io` | `https://api.minimax.io/anthropic` |
| `minimax-cn` (중국) | `https://api.minimaxi.com` | `https://api.minimaxi.com/anthropic` |

### 제공자 별칭 (Provider aliases)

다음은 모두 `minimax-oauth`로 확인됩니다:

```bash
hermes --provider minimax-oauth    # 정식 명칭
hermes --provider minimax-portal   # 별칭
hermes --provider minimax-global   # 별칭
hermes --provider minimax_oauth    # 별칭 (언더스코어 형태)
```

## 환경 변수

`minimax-oauth` 제공자는 `MINIMAX_API_KEY`나 `MINIMAX_BASE_URL`을 사용하지 **않습니다**. 이러한 변수는 API 키 기반의 `minimax` 및 `minimax-cn` 제공자 전용입니다.

| 변수 | 효과 |
|----------|--------|
| `MINIMAX_API_KEY` | `minimax` 제공자에서만 사용됨 — `minimax-oauth`에서는 무시됨 |
| `MINIMAX_CN_API_KEY` | `minimax-cn` 제공자에서만 사용됨 — `minimax-oauth`에서는 무시됨 |

`minimax-oauth`를 활성 제공자로 사용하려면 `config.yaml`에서 `model.provider: minimax-oauth`를 설정하거나(안내되는 흐름을 위해 `hermes setup` 사용), 단일 호출 시 `--provider minimax-oauth`를 전달하세요:

```bash
hermes --provider minimax-oauth
```

## 모델

| 모델 | 권장 용도 |
|-------|----------|
| `MiniMax-M2.7` | 긴 컨텍스트 추론, 복잡한 도구 호출 |
| `MiniMax-M2.7-highspeed` | 더 낮은 지연 시간, 가벼운 작업, 보조 호출 |

두 모델 모두 최대 200,000 토큰의 컨텍스트를 지원합니다.

`minimax-oauth`가 기본 제공자일 때 `MiniMax-M2.7-highspeed`는 비전 및 위임 작업을 위한 보조 모델로도 자동으로 사용됩니다.

## 문제 해결

### 토큰 만료됨 — 자동으로 재로그인되지 않음

Hermes는 액세스 토큰의 만료가 60초 이내인 경우 매 세션 시작 시 토큰을 갱신합니다. 액세스 토큰이 이미 만료된 경우(예: 오프라인 상태가 길었던 경우), 다음 요청 시 자동으로 갱신됩니다. `refresh_token_reused` 또는 `invalid_grant`로 갱신에 실패하면 Hermes는 해당 세션을 재로그인이 필요한 상태로 표시합니다.

갱신 실패가 치명적인 경우(HTTP 4xx, `invalid_grant`, 권한 취소 등), Hermes는 갱신 토큰을 사용할 수 없는 것으로 표시하고 로컬에서 격리하여 실패할 교환을 계속 재시도하지 않도록 합니다. 에이전트는 "재인증 필요" 메시지를 한 번 표시하고 다시 로그인할 때까지 대기합니다.

**해결 방법:** 다시 로그인 과정을 시작하려면 `hermes auth add minimax-oauth`를 다시 실행하세요. 다음번 교환에 성공하면 격리 상태가 해제됩니다.

### 승인 시간 초과 (Authorization timed out)

디바이스 코드 흐름에는 제한된 만료 시간이 있습니다. 제시간에 로그인을 승인하지 않으면 Hermes가 시간 초과 오류를 발생시킵니다.

**해결 방법:** `hermes auth add minimax-oauth` (또는 `hermes model`)를 다시 실행하세요. 흐름이 새로 시작됩니다.

### 상태 불일치 (CSRF 가능성)

인증 서버에서 반환한 `state` 값이 Hermes가 보낸 값과 일치하지 않음을 감지했습니다.

**해결 방법:** 로그인을 다시 실행하세요. 문제가 지속되면 OAuth 응답을 수정하는 프록시나 리디렉션이 있는지 확인하세요.

### 원격 서버에서 로그인하기

`hermes`가 브라우저 창을 열 수 없는 경우 `--no-browser`를 사용하세요:

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes가 URL과 코드를 출력합니다. 어느 기기에서든 URL을 열고 거기서 흐름을 완료하세요.

### 런타임 시 "Not logged into MiniMax OAuth" 오류 발생

인증 저장소에 `minimax-oauth`에 대한 자격 증명이 없습니다. 아직 로그인하지 않았거나 자격 증명 파일이 삭제되었습니다.

**해결 방법:** `hermes model`을 실행하고 MiniMax (OAuth)를 선택하거나 `hermes auth add minimax-oauth`를 실행하세요.

## 로그아웃

저장된 MiniMax OAuth 자격 증명을 제거하려면:

```bash
hermes auth remove minimax-oauth
```

## 더 보기

- [AI Providers 레퍼런스](../integrations/providers.md)
- [환경 변수](../reference/environment-variables.md)
- [구성 (Configuration)](../user-guide/configuration.md)
- [hermes doctor](../reference/cli-commands.md)
