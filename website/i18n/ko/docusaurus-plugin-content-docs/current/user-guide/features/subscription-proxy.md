---
sidebar_position: 15
title: "구독 프록시 (Subscription Proxy)"
description: "Nous Portal 구독(또는 기타 OAuth 공급자)을 외부 앱을 위한 OpenAI 호환 엔드포인트로 사용하세요"
---

# 구독 프록시 (Subscription Proxy)

구독 프록시는 외부 앱들 — OpenViking, Karakeep, Open WebUI 등 OpenAI 호환 채팅 완성을 지원하는 모든 앱 — 이 Hermes가 관리하는 공급자 구독을 그들의 LLM 엔드포인트로 사용할 수 있게 해주는 로컬 HTTP 서버입니다. 프록시가 올바른 자격 증명을 자동으로 첨부하고 갱신하므로, 앱 측에서는 고정된 API 키가 필요하지 않습니다.

이것은 [API 서버](./api-server.md)와 다릅니다:

| | API 서버 | 구독 프록시 |
|---|---|---|
| 제공하는 것 | 내 에이전트 (전체 도구 세트, 메모리, 스킬) | 순수 모델 추론(Raw model inference) |
| 사용 사례 | "Hermes를 채팅 백엔드로 사용하기" | "다른 앱에서 내 Portal 구독 사용하기" |
| 인증 (Auth) | 나의 `API_SERVER_KEY` | 아무 베어러 토큰 (프록시가 진짜를 첨부함) |
| 도구 호출 | 예 — 에이전트가 도구를 실행함 | 아니요 — 단순 통과(passthrough)만 수행 |

**에이전트**를 백엔드로 사용하려면 API 서버를 사용하세요. 구독을 통해 **모델** 기능만을 원한다면 프록시를 사용하세요.

## 빠른 시작

### 1. 공급자 로그인 (최초 1회)

```bash
hermes portal
```

이 명령어는 Nous Portal OAuth 흐름을 위해 브라우저를 엽니다. Hermes는 리프레시 토큰(refresh token)을 `~/.hermes/auth.json`에 저장합니다 — 이곳은 모든 Hermes 공급자 로그인 정보가 저장되는 장소입니다.

### 2. 프록시 시작하기

```bash
hermes proxy start
```

```
Starting Hermes proxy for Nous Portal
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your subscription)
  Use any bearer token in the client — the proxy attaches your real credential.
```

이 프로세스를 포그라운드에 실행 상태로 두세요. 로그아웃 후에도 유지하고 싶다면 `tmux`, `nohup` 또는 systemd 유닛을 사용하세요.

### 3. 앱에서 프록시 가리키기

모든 OpenAI 호환 앱 설정은 다음 세 가지 값을 취합니다:

```
Base URL:   http://127.0.0.1:8645/v1
API key:    아무거나 (예: "sk-unused")
Model:      Hermes-4-70B    # 또는 Hermes-4.3-36B, Hermes-4-405B
```

프록시는 앱이 보내는 `Authorization` 헤더를 무시하고, 업스트림 요청에 진짜 Portal 자격 증명을 첨부합니다. 베어러 토큰의 만료가 다가오면 자동으로 갱신(refresh)이 일어납니다.

## 사용 가능한 공급자

```bash
hermes proxy providers
```

현재 지원되는 공급자: `nous` (Nous Portal) 및 `xai` (xAI / Grok). 더 많은 OAuth 공급자는 `hermes_cli/proxy/adapters/` 디렉터리에 `UpstreamAdapter` 인터페이스를 구현하여 추가할 수 있습니다.

## 상태 확인

```bash
hermes proxy status
```

```
Hermes proxy upstream adapters

  [nous    ] Nous Portal — ready (bearer expires 2026-05-15T06:43:21Z)
```

만약 `not logged in` 메시지가 보인다면 `hermes portal`을 실행하세요. 만약 `credentials need attention` 메시지가 보인다면, 이는 귀하의 리프레시 토큰이 취소되었음을 의미합니다 (드문 경우로 Portal 웹 UI에서 로그아웃한 경우 발생) — 이럴 때는 그냥 `hermes portal`을 다시 실행하시면 됩니다.

## 허용된 경로 (Allowed paths)

프록시는 업스트림이 실제로 제공하는 경로만 포워딩(forwarding)합니다. Nous Portal의 경우:

| 경로 | 목적 |
|------|---------|
| `/v1/chat/completions` | 채팅 완성 (스트리밍 + 비-스트리밍) |
| `/v1/completions` | 레거시 텍스트 완성 |
| `/v1/embeddings` | 임베딩 (Embeddings) |
| `/v1/models` | 모델 목록 |

그 외의 경로들(`/v1/images/generations`, `/v1/audio/speech` 등)은 404를 반환하며 허용된 경로를 명확하게 가리키는 에러 메시지를 표시합니다. 이를 통해 길 잃은(stray) 클라이언트가 업스트림에 이상한 요청을 흘려보내는 것을 방지합니다.

## OpenViking에서 Portal을 사용하도록 설정하기

[OpenViking](https://github.com/volcengine/OpenViking)은 VLM(메모리를 추출하는 데 사용되는 비전/언어 모델) 및 임베딩 모델을 위한 LLM 공급자가 필요한 컨텍스트 데이터베이스입니다. 프록시를 사용하면 해당 앱의 `vlm.api_base`가 로컬 프록시를 가리키게 할 수 있습니다:

`~/.openviking/ov.conf` 편집:

```json
{
  "vlm": {
    "provider": "openai",
    "model": "Hermes-4-70B",
    "api_base": "http://127.0.0.1:8645/v1",
    "api_key": "unused-proxy-attaches-real-creds"
  }
}
```

그런 다음 `openviking-server`와 함께 터미널에서 프록시를 시작하세요:

```bash
# 터미널 1
hermes proxy start

# 터미널 2
openviking-server
```

이제 OpenViking의 VLM 호출은 당신의 Portal 구독을 통해 흐릅니다. 임베딩 모델 측은 여전히 자체 공급자가 필요합니다 — Portal은 `/v1/embeddings`를 제공하긴 하지만, 모델 선택은 귀하의 구독 티어가 무엇을 지원하는지에 따라 다릅니다; `portal.nousresearch.com/models`를 확인하세요.

## Karakeep (또는 모든 북마크/요약 앱) 구성하기

[Karakeep](https://karakeep.app/)은 북마크 요약을 위해 OpenAI 호환 API를 사용합니다. 해당 앱의 구성(config)에서:

```bash
# Karakeep .env
OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
OPENAI_API_KEY=아무-빈-문자열이-아닌-값
INFERENCE_TEXT_MODEL=Hermes-4-70B
```

이 패턴은 Open WebUI, LobeChat, NextChat 또는 기타 OpenAI 호환 클라이언트에서도 동일하게 작동합니다.

## LAN 네트워크에 노출하기

기본적으로 프록시는 `127.0.0.1` (localhost 전용)에 바인딩됩니다. 네트워크상의 다른 기기들이 이 프록시를 사용하게 하려면:

```bash
hermes proxy start --host 0.0.0.0 --port 8645
```

⚠ **주의:** 이제 네트워크상의 누구든 당신의 Portal 구독을 사용할 수 있습니다. 프록시 자체에는 인증 기능이 없습니다 — 어떤 베어러(bearer) 토큰이든 수락합니다. 신뢰할 수 있는 네트워크 범위를 넘어서 노출하는 경우 방화벽, VPN, 또는 적절한 인증이 포함된 리버스 프록시(reverse proxy)를 사용하세요.

## 속도 제한 (Rate limits)

해당 Portal 티어의 RPM/TPM 제한이 프록시 전체에 적용됩니다. 프록시는 분산시키거나(fan out) 풀링(pooling)하지 않습니다 — 이것은 단일 베어러이며 당신의 전체 구독 할당량을 사용합니다. 사용량 모니터링은 [portal.nousresearch.com](https://portal.nousresearch.com)에서 할 수 있습니다.

## 아키텍처 (Architecture)

프록시는 의도적으로 최소한의 기능만 갖추도록 설계되었습니다. 각 요청에 대해:

1. 앱에서 `POST /v1/chat/completions`를 받습니다.
2. 어댑터의 현재 자격 증명을 조회합니다 (만료 임박 시 갱신).
3. 요청 본문(body)을 그대로 전달하며, 헤더에 `Authorization: Bearer <minted-key>`를 첨부합니다.
4. 응답을 변경 없이 그대로 스트리밍하여 되돌려 보냅니다 (SSE 유지).

어떤 변환(transformation)도 하지 않습니다. 요청 본문을 기록(logging)하지 않습니다. 에이전트 루프도 없습니다. 프록시는 자격 증명만 덧붙여서 통과시키는 역할(pass-through)만 합니다.

## 미래 계획: 더 많은 OAuth 공급자 추가

어댑터 시스템은 플러그 앤 플레이 방식입니다. 새로운 공급자 (예: HuggingFace, GitHub Copilot 채팅 엔드포인트, OAuth를 통한 Anthropic)를 추가하려면 `hermes_cli/proxy/adapters/<provider>.py`에 `UpstreamAdapter`를 구현하고 이를 `adapters/__init__.py`에 등록해야 합니다. 프로토콜 수준에서 OpenAI 호환이 되지 않는 공급자 (예를 들어, Anthropic Messages API)의 경우 변환 계층이 필요하며, 이는 현재 형태에서는 범위를 벗어납니다.
