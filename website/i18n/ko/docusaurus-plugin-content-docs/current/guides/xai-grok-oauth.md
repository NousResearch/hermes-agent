---
sidebar_position: 16
title: "xAI Grok OAuth (SuperGrok / X Premium+)"
description: "SuperGrok 또는 X Premium+ 구독으로 로그인해 Hermes Agent에서 Grok 모델을 사용하세요 — API 키가 필요하지 않습니다"
---

# xAI Grok OAuth (SuperGrok / X Premium+)

Hermes Agent는 [accounts.x.ai](https://accounts.x.ai)의 브라우저 기반 OAuth 디바이스 코드 로그인 흐름을 통해 xAI Grok을 지원합니다. **SuperGrok 구독**([grok.com](https://x.ai/grok)) 또는 **X Premium+ 구독**(연결된 X 계정)을 사용할 수 있습니다. `XAI_API_KEY`는 필요하지 않습니다. 한 번 로그인하면 Hermes가 백그라운드에서 세션을 자동으로 갱신합니다.

Premium+가 적용된 X 계정으로 로그인하면 xAI가 구독 상태를 xAI 세션에 자동으로 연결하므로, OAuth 흐름은 직접 SuperGrok을 구독한 경우와 동일하게 작동합니다.

전송 계층은 `codex_responses` 어댑터를 재사용합니다(xAI가 Responses 스타일 엔드포인트를 노출함). 따라서 어댑터를 변경하지 않아도 추론, 도구 호출, 스트리밍, 프롬프트 캐싱이 작동합니다.

동일한 OAuth bearer 토큰은 Hermes의 xAI 직접 연결 기능인 TTS, 이미지 생성, 동영상 생성, 전사에서도 재사용됩니다. 따라서 한 번의 로그인으로 네 가지 기능을 모두 사용할 수 있습니다.

## 개요

| 항목 | 값 |
|------|-------|
| Provider ID | `xai-oauth` |
| 표시 이름 | xAI Grok OAuth (SuperGrok / X Premium+) |
| 인증 유형 | 브라우저 OAuth 2.0 디바이스 코드 |
| 전송 계층 | xAI Responses API (`codex_responses`) |
| 기본 모델 | `grok-4.6` |
| 엔드포인트 | `https://api.x.ai/v1` |
| 인증 서버 | `https://accounts.x.ai` |
| 환경 변수 필요 여부 | 아니요 (이 provider에서는 `XAI_API_KEY`를 **사용하지 않음**) |
| 구독 | [SuperGrok](https://x.ai/grok) 또는 [X Premium+](https://x.com/i/premium_sign_up) — 아래 참고 사항 참조 |

## 사전 요구 사항

- Python 3.9+
- Hermes Agent 설치
- xAI 계정의 활성 **SuperGrok** 구독, **또는** 로그인하는 X 계정의 **X Premium+** 구독 (xAI가 구독을 자동으로 연결함)
- 표시된 인증 URL을 열 수 있는 곳에서 사용할 수 있는 브라우저

:::warning xAI는 등급에 따라 OAuth API 액세스를 제한할 수 있습니다
xAI의 백엔드는 OAuth API 표면에 자체 허용 목록을 적용합니다. 앱 내 구독이 활성 상태인데도 일반 SuperGrok 구독자를 `HTTP 403`으로 거부하는 사례가 확인되었습니다(이슈 [#26847](https://github.com/NousResearch/hermes-agent/issues/26847) 참조). 브라우저에서 OAuth 로그인이 성공했지만 추론이 403을 반환한다면 `XAI_API_KEY`를 설정하고 API 키 경로(`provider: xai`)로 전환하세요. 현재 이 표면에는 동일한 제한이 적용되지 않습니다.
:::

## 빠른 시작

```bash
# Launch the provider and model picker
hermes model
# → Select "xAI Grok OAuth (SuperGrok / X Premium+)" from the provider list
# → Hermes opens or prints an accounts.x.ai verification URL
# → Enter the displayed code if prompted, then approve access in the browser
# → Pick a model (grok-4.6 is at the top)
# → Start chatting

hermes
```

첫 로그인 후 자격 증명은 `~/.hermes/auth.json`에 저장되며 만료 전에 자동으로 갱신됩니다.

## 수동 로그인

모델 선택기를 거치지 않고 로그인을 실행할 수 있습니다.

```bash
hermes auth add xai-oauth
```

### 원격 / 헤드리스 세션

서버, 컨테이너, 브라우저 전용 콘솔(Cloud Shell, Codespaces, EC2 Instance Connect) 또는 Hermes가 로컬에서 브라우저를 열 수 없는 SSH 세션에서는 Hermes가 xAI 인증 URL과 사용자 코드를 출력합니다. 노트북이나 클라우드 콘솔의 브라우저에서 URL을 열고, 메시지가 표시되면 코드를 입력하세요. 그러면 Hermes가 xAI의 로그인 승인을 받을 때까지 폴링을 계속합니다. SSH 터널이나 로컬 콜백 리스너는 필요하지 않습니다.

```bash
hermes auth add xai-oauth --no-browser
# Open the printed verification URL in your browser.
```

웹 대시보드나 데스크톱 앱에서 로그인할 때도 동일한 디바이스 코드 흐름이 적용됩니다. Hermes가 인증 URL과 사용자 코드를 표시한 다음, 액세스를 승인할 때까지 백그라운드에서 폴링합니다.

## 로그인 작동 방식

1. Hermes가 `auth.x.ai`에서 디바이스 코드를 요청합니다.
2. 인증 URL을 열고 로그인한 뒤, 메시지가 표시되면 코드를 입력하고 액세스를 승인합니다.
3. Hermes가 승인될 때까지 xAI를 폴링한 다음 토큰을 `~/.hermes/auth.json`에 저장합니다.
4. 이후 Hermes는 백그라운드에서 액세스 토큰을 갱신합니다. `hermes auth logout xai-oauth`를 실행하거나 xAI 계정 설정에서 액세스를 취소할 때까지 로그인 상태가 유지됩니다.

## 로그인 상태 확인

```bash
hermes doctor
```

`◆ Auth Providers` 섹션에서 `xai-oauth`를 포함한 모든 provider의 현재 상태를 확인할 수 있습니다.

## 모델 전환

```bash
hermes model
# → Select "xAI Grok OAuth (SuperGrok / X Premium+)"
# → Pick from the model list (grok-4.6 is pinned to the top)
```

또는 모델을 직접 설정합니다.

```bash
hermes config set model.default grok-4.6
hermes config set model.provider xai-oauth
```

## 설정 참고

로그인 후 `~/.hermes/config.yaml`에는 다음 내용이 포함됩니다.

```yaml
model:
  default: grok-4.6
  provider: xai-oauth
  base_url: https://api.x.ai/v1
```

### Provider 별칭

다음 이름은 모두 `xai-oauth`로 확인됩니다.

```bash
hermes --provider xai-oauth        # canonical
hermes --provider grok-oauth       # alias
hermes --provider x-ai-oauth       # alias
hermes --provider xai-grok-oauth   # alias
```

## xAI 직접 연결 도구 (TTS / 이미지 / 동영상 / 전사 / X 검색)

OAuth로 로그인하면 모든 xAI 직접 연결 도구가 동일한 bearer 토큰을 자동으로 재사용합니다. API 키를 사용하려는 경우가 아니라면 별도의 설정은 **필요하지 않습니다**.

각 도구의 백엔드를 선택하려면 다음을 실행합니다.

```bash
hermes tools
# → Text-to-Speech       → "xAI TTS"
# → Image Generation     → "xAI Grok Imagine (image)"
# → Video Generation     → "xAI Grok Imagine"
# → X (Twitter) Search   → "xAI Grok OAuth (SuperGrok / X Premium+)"
```

OAuth 토큰이 이미 저장되어 있으면 선택기가 이를 확인하고 자격 증명 입력을 건너뜁니다. OAuth와 `XAI_API_KEY`가 모두 설정되지 않은 경우 선택기는 OAuth 로그인, API 키 붙여넣기, 건너뛰기 중 하나를 고르는 3가지 메뉴를 제공합니다.

:::note 동영상 생성은 기본적으로 꺼져 있습니다
`video_gen` toolset은 기본적으로 비활성화되어 있습니다. 에이전트가 `video_generate`를 호출할 수 있도록 `hermes tools` → `🎬 Video Generation`(스페이스 키 누르기)에서 활성화하세요. 그렇지 않으면 에이전트가 동영상 생성용으로 태그된 번들 ComfyUI skill로 대체할 수 있습니다.
:::

:::note xAI 자격 증명이 있으면 X 검색이 자동으로 활성화됩니다
`x_search` toolset은 xAI 자격 증명(SuperGrok / X Premium+ OAuth 토큰 또는 `XAI_API_KEY`)이 구성되어 있으면 항상 자동으로 활성화됩니다. 원하지 않는 경우 `hermes tools` → `🐦 X (Twitter) Search`(스페이스 키 누르기)에서 명시적으로 비활성화하세요. 이 도구는 xAI의 내장 `x_search` Responses API를 통해 라우팅됩니다. **SuperGrok / X Premium+ OAuth 로그인 또는 유료 `XAI_API_KEY` 중 어느 쪽으로도** 작동하며, 둘 다 구성된 경우 OAuth를 우선 사용합니다(API 비용 대신 구독 할당량 사용). xAI 자격 증명이 구성되지 않은 경우 toolset 활성화 여부와 관계없이 도구 스키마가 모델에 표시되지 않습니다.
:::

### 모델

| 도구 | 모델 | 참고 |
|------|-------|-------|
| 채팅 | `grok-4.6` | 기본값; OAuth 선택기 맨 위에 고정됨 |
| 채팅 | `grok-build-0.1` | 코딩 중심 Grok Build 모델 |
| 채팅 | `grok-4.3` | 이전 세대 |
| 채팅 | `grok-4.20-0309-reasoning` | 추론 변형 |
| 채팅 | `grok-4.20-0309-non-reasoning` | 비추론 변형 |
| 채팅 | `grok-4.20-multi-agent-0309` | 멀티 에이전트 변형 |
| 이미지 | `grok-imagine-image` | 기본값; 약 5–10초 |
| 이미지 | `grok-imagine-image-quality` | 더 높은 충실도; 약 10–20초 |
| 동영상 | `grok-imagine-video` | 텍스트-동영상 |
| 동영상 | `grok-imagine-video-1.5-preview` | 이미지-동영상; 날짜가 포함된 별칭 `grok-imagine-video-1.5-2026-05-30` |
| TTS | (기본 음성) | xAI `/v1/tts` 엔드포인트 |

채팅 카탈로그는 디스크에 저장된 `models.dev` 캐시에서 실시간으로 파생됩니다. 새 xAI 릴리스는 해당 캐시가 갱신되면 자동으로 표시됩니다. `grok-4.6`은 항상 목록 맨 위에 고정됩니다.

## 환경 변수

| 변수 | 효과 |
|----------|--------|
| `XAI_BASE_URL` | 기본 `https://api.x.ai/v1` 엔드포인트 재정의 (대부분 필요하지 않음). |

xAI를 활성 provider로 선택하려면 `config.yaml`에서 `model.provider: xai-oauth`를 설정하세요(`hermes setup`으로 안내 흐름 사용). 또는 한 번의 실행에만 `--provider xai-oauth`를 전달할 수 있습니다.

## 문제 해결

### 토큰 만료 — 자동으로 다시 로그인하지 않음

Hermes는 각 세션 전에 토큰을 갱신하고 401이 발생하면 다시 반응적으로 갱신합니다. 갱신이 `invalid_grant`로 실패하면(갱신 토큰이 취소되었거나 계정이 교체된 경우) Hermes는 충돌하는 대신 형식화된 재인증 메시지를 표시합니다.

갱신 실패가 최종적인 경우(HTTP 4xx, `invalid_grant`, 취소된 grant 등) Hermes는 갱신 토큰을 사용 불가로 표시하고 로컬에서 격리합니다. 이후 호출에서는 같은 401을 반복해서 재현하는 대신 실패가 예정된 갱신 시도를 건너뜁니다. 에이전트는 단 한 번의 "재인증 필요" 메시지를 표시하고, 다시 로그인할 때까지 개입하지 않습니다.

**해결:** `hermes auth add xai-oauth`를 다시 실행해 새 로그인을 시작하세요. 다음 교환이 성공하면 격리가 해제됩니다.

### 인증 시간 초과

디바이스 코드 승인에는 유효 기간이 제한되어 있습니다(xAI는 디바이스 코드 응답에 `expires_in`을 설정하며, 일반적으로 수십 분 정도입니다). 제시간에 로그인을 승인하지 않으면 Hermes가 시간 초과 오류를 발생시킵니다.

**해결:** `hermes auth add xai-oauth`(또는 `hermes model`)를 다시 실행하세요. 흐름이 새로 시작됩니다.

### 원격 서버에서 로그인

SSH 또는 컨테이너 세션에서 Hermes는 브라우저를 여는 대신 인증 URL과 사용자 코드를 출력합니다. 노트북이나 클라우드 콘솔의 브라우저에서 해당 URL을 여세요. xAI Grok OAuth에는 SSH 포트 포워딩이 필요하지 않습니다.

```bash
hermes auth add xai-oauth --no-browser
```

루프백 리디렉션 provider(Spotify, MCP 서버)는 [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md)를 참조하세요.

### 성공적으로 로그인한 후 HTTP 403 (등급 / 권한)

브라우저에서 OAuth가 완료되고 토큰이 저장되었지만, 추론 또는 토큰 갱신에서 *"The caller does not have permission to execute the specified operation"*과 비슷한 메시지와 함께 `HTTP 403`을 반환합니다.

이는 **오래된 토큰 문제**가 아닙니다. `hermes model`을 다시 실행해도 달라지지 않습니다. 앱 내 구독이 활성 상태인데도 xAI의 백엔드가 특정 SuperGrok 등급으로 OAuth API 액세스를 제한하는 사례가 확인되었습니다(이슈 [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)).

**해결:** `XAI_API_KEY`를 설정하고 API 키 경로로 전환하세요.

```bash
export XAI_API_KEY=xai-...
hermes config set model.provider xai
```

또는 OAuth 경로가 필요하다면 [x.ai/grok](https://x.ai/grok)에서 구독을 업그레이드하세요.

### 런타임의 "No xAI credentials found" 오류

인증 저장소에 `xai-oauth` 항목이 없고 `XAI_API_KEY`도 설정되지 않았습니다. 아직 로그인하지 않았거나 자격 증명 파일이 삭제된 것입니다.

**해결:** `hermes model`을 실행하고 xAI Grok OAuth provider를 선택하거나 `hermes auth add xai-oauth`를 실행하세요.

## 로그아웃

저장된 모든 xAI Grok OAuth 자격 증명을 제거하려면 다음을 실행합니다.

```bash
hermes auth logout xai-oauth
```

이 명령은 `auth.json`의 singleton OAuth 항목과 `xai-oauth`의 모든 credential-pool 행을 모두 삭제합니다. 단일 pool 항목만 삭제하려면 `hermes auth remove xai-oauth <index|id|label>`을 사용하세요(`hermes auth list xai-oauth`를 실행하면 항목을 확인할 수 있습니다).

## 함께 보기

- [OAuth over SSH / Remote Hosts](./oauth-over-ssh.md) — 루프백 리디렉션 provider(Spotify, MCP)를 위한 SSH 터널; xAI는 디바이스 코드를 사용하므로 터널이 필요하지 않음
- [AI 제공업체 참고 자료](../integrations/providers.md)
- [환경 변수](../reference/environment-variables.md)
- [구성](../user-guide/configuration.md)
- [음성 및 TTS](../user-guide/features/tts.md)
