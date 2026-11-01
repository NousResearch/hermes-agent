---
sidebar_position: 15
title: "MiniMax OAuth"
description: "브라우저 OAuth로 MiniMax에 로그인하고 API 키 없이 Hermes Agent에서 MiniMax-M2.7 모델 사용"
---

# MiniMax OAuth

Hermes Agent는 [MiniMax 포털](https://www.minimax.io)과 동일한 자격 증명을 사용하는 브라우저 기반 OAuth 로그인 흐름을 통해 **MiniMax**를 지원합니다. API 키나 신용카드가 필요하지 않으며, 한 번 로그인하면 Hermes가 세션을 자동으로 갱신합니다.

전송 방식은 `anthropic_messages` 어댑터를 재사용합니다(MiniMax는 `/anthropic`에서 Anthropic Messages 호환 엔드포인트를 제공함). 따라서 기존의 도구 호출, 스트리밍, 컨텍스트 기능을 어댑터 변경 없이 사용할 수 있습니다.

## 개요

| 항목 | 값 |
|------|-------|
| 제공자 ID | `minimax-oauth` |
| 표시 이름 | MiniMax (OAuth) |
| 인증 유형 | 브라우저 OAuth (PKCE 리디렉션 흐름) |
| 전송 방식 | Anthropic Messages 호환 (`anthropic_messages`) |
| 모델 | `MiniMax-M2.7`, `MiniMax-M2.7-highspeed` |
| 글로벌 엔드포인트 | `https://api.minimax.io/anthropic` |
| 중국 엔드포인트 | `https://api.minimaxi.com/anthropic` |
| 환경 변수 필요 여부 | 아니요 (이 제공자는 `MINIMAX_API_KEY`를 사용하지 않음) |

## 사전 요구 사항

- Python 3.9 이상
- Hermes Agent 설치
- [minimax.io](https://www.minimax.io)의 MiniMax 계정(글로벌) 또는 [minimaxi.com](https://www.minimaxi.com)의 계정(중국)
- 로컬 컴퓨터에서 사용할 수 있는 브라우저(또는 원격 세션에서는 `--no-browser` 사용)

## 빠른 시작

```bash
# Launch the provider and model picker
hermes model
# → Select "MiniMax (OAuth)" from the provider list
# → Hermes opens your browser to the MiniMax authorization page
# → Approve access in the browser
# → Select a model (MiniMax-M2.7 or MiniMax-M2.7-highspeed)
# → Start chatting

hermes
```

첫 로그인 후 자격 증명은 `~/.hermes/auth.json`에 저장되며 각 세션 전에 자동으로 갱신됩니다.

## 수동 로그인

모델 선택기를 거치지 않고 직접 로그인을 시작할 수 있습니다.

```bash
hermes auth add minimax-oauth
```

### 중국 지역

계정이 중국 플랫폼(`minimaxi.com`)에 있다면 API 키 기반 `minimax-cn` 제공자를 대신 사용하세요. `minimax-cn`은 `auth_type="api_key"`만 등록되어 있으며 OAuth 흐름은 없습니다. `MINIMAX_CN_API_KEY`(선택적으로 `MINIMAX_CN_BASE_URL`)를 직접 구성하세요.

```bash
echo 'MINIMAX_CN_API_KEY=your-key' >> ~/.hermes/.env
```

### 원격/헤드리스 세션

브라우저를 사용할 수 없는 서버나 컨테이너에서는 다음을 실행합니다.

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes가 인증 URL과 사용자 코드를 출력합니다. 어떤 기기에서든 URL을 열고 메시지가 표시되면 코드를 입력하세요.

## OAuth 흐름

Hermes는 MiniMax OAuth 엔드포인트에 대해 PKCE 브라우저 OAuth 흐름을 구현합니다.

1. Hermes가 PKCE 검증자/챌린지 쌍과 무작위 상태 값을 생성합니다.
2. 챌린지와 함께 `{base_url}/oauth/code`로 POST하고 `user_code`와 `verification_uri`를 받습니다.
3. 브라우저에서 `verification_uri`를 엽니다. 메시지가 표시되면 `user_code`를 입력합니다.
4. Hermes가 토큰이 도착할 때까지(또는 기한이 지날 때까지) `{base_url}/oauth/token`을 폴링합니다.
5. 토큰(`access_token`, `refresh_token`, 만료 시간)이 `minimax-oauth` 키 아래 `~/.hermes/auth.json`에 저장됩니다.

표준 OAuth `refresh_token` 부여를 사용하는 토큰 갱신은 액세스 토큰이 만료 60초 이내에 들어오면 각 세션 시작 시 자동으로 실행됩니다.

## 로그인 상태 확인

```bash
hermes doctor
```

`◆ Auth Providers` 섹션에 다음과 같이 표시됩니다.

```
✓ MiniMax OAuth  (logged in, region=global)
```

또는 로그인하지 않은 경우 다음과 같이 표시됩니다.

```
⚠ MiniMax OAuth  (not logged in)
```

## 모델 전환

```bash
hermes model
# → Select "MiniMax (OAuth)"
# → Pick from the model list
```

또는 모델을 직접 설정합니다.

```bash
hermes config set model.default MiniMax-M2.7
hermes config set model.provider minimax-oauth
```

## 설정 참고

로그인 후 `~/.hermes/config.yaml`에 다음과 비슷한 항목이 포함됩니다.

```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```

### 지역별 엔드포인트

| 제공자 ID | 포털 | 추론 엔드포인트 |
|-------------|--------|-------------------|
| `minimax-oauth` (글로벌) | `https://api.minimax.io` | `https://api.minimax.io/anthropic` |
| `minimax-cn` (중국) | `https://api.minimaxi.com` | `https://api.minimaxi.com/anthropic` |

### 제공자 별칭

다음 항목은 모두 `minimax-oauth`로 확인됩니다.

```bash
hermes --provider minimax-oauth    # canonical
hermes --provider minimax-portal   # alias
hermes --provider minimax-global   # alias
hermes --provider minimax_oauth    # alias (underscore form)
```

## 환경 변수

`minimax-oauth` 제공자는 `MINIMAX_API_KEY` 또는 `MINIMAX_BASE_URL`을 사용하지 않습니다. 해당 변수는 API 키 기반 `minimax` 및 `minimax-cn` 제공자 전용입니다.

| 변수 | 효과 |
|----------|--------|
| `MINIMAX_API_KEY` | `minimax` 제공자에서만 사용 — `minimax-oauth`에서는 무시됨 |
| `MINIMAX_CN_API_KEY` | `minimax-cn` 제공자에서만 사용 — `minimax-oauth`에서는 무시됨 |

`minimax-oauth`를 활성 제공자로 사용하려면 `config.yaml`에서 `model.provider: minimax-oauth`를 설정하세요(`hermes setup`을 사용하면 안내에 따라 설정할 수 있음). 또는 한 번만 실행할 때 `--provider minimax-oauth`를 전달합니다.

```bash
hermes --provider minimax-oauth
```

## 모델

| 모델 | 적합한 용도 |
|-------|----------|
| `MiniMax-M2.7` | 긴 컨텍스트 추론, 복잡한 도구 호출 |
| `MiniMax-M2.7-highspeed` | 더 낮은 지연 시간, 가벼운 작업, 보조 호출 |

두 모델 모두 최대 200,000토큰의 컨텍스트를 지원합니다.

`MiniMax-M2.7`은 `minimax-oauth`가 기본 제공자일 때 비전 및 위임 작업의 보조 모델로도 자동 사용됩니다.

## 문제 해결

### 토큰 만료 — 자동으로 다시 로그인하지 않음

Hermes는 각 세션 시작 시 액세스 토큰이 만료 60초 이내에 있으면 토큰을 갱신합니다. 액세스 토큰이 이미 만료된 경우(예: 장기간 오프라인 상태 이후) 다음 요청에서 자동으로 갱신됩니다. 갱신이 `refresh_token_reused` 또는 `invalid_grant`와 함께 실패하면 Hermes는 세션에 다시 로그인이 필요하다고 표시합니다.

갱신 실패가 최종적인 경우(HTTP 4xx, `invalid_grant`, 취소된 부여 등), Hermes는 갱신 토큰을 더 이상 사용할 수 없는 것으로 표시하고 로컬에서 격리하여 실패할 교환을 계속 재생하지 않도록 합니다. 에이전트는 "re-authentication required"라는 메시지를 한 번 표시하고 사용자가 다시 로그인할 때까지 더 이상 개입하지 않습니다.

**해결:** `hermes auth add minimax-oauth`를 다시 실행해 새 로그인을 시작하세요. 다음 교환이 성공하면 격리가 해제됩니다.

### 인증 시간이 초과됨

디바이스 코드 흐름에는 유한한 만료 시간이 있습니다. 제때 로그인을 승인하지 않으면 Hermes가 시간 초과 오류를 발생시킵니다.

**해결:** `hermes auth add minimax-oauth`(또는 `hermes model`)를 다시 실행하세요. 흐름이 새로 시작됩니다.

### 상태 불일치(가능한 CSRF)

Hermes가 인증 서버에서 반환한 `state` 값이 자신이 보낸 값과 일치하지 않음을 감지했습니다.

**해결:** 로그인을 다시 실행하세요. 문제가 지속되면 OAuth 응답을 수정하는 프록시 또는 리디렉션이 있는지 확인하세요.

### 원격 서버에서 로그인

`hermes`가 브라우저 창을 열 수 없다면 `--no-browser`를 사용하세요.

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes가 URL과 코드를 출력합니다. 어떤 기기에서든 URL을 열고 그곳에서 흐름을 완료하세요.

### 런타임에서 "Not logged into MiniMax OAuth" 오류

인증 저장소에 `minimax-oauth` 자격 증명이 없습니다. 아직 로그인하지 않았거나 자격 증명 파일이 삭제되었습니다.

**해결:** `hermes model`을 실행하고 MiniMax (OAuth)를 선택하거나 `hermes auth add minimax-oauth`를 실행하세요.

## 로그아웃

저장된 MiniMax OAuth 자격 증명을 삭제하려면 다음을 실행합니다.

```bash
hermes auth logout minimax-oauth
```

## 관련 문서

- [AI 제공자 참고](../integrations/providers.md)
- [환경 변수](../reference/environment-variables.md)
- [설정](../user-guide/configuration.md)
- [hermes doctor](../reference/cli-commands.md)
