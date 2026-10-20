---
sidebar_position: 15
title: "구독 프록시"
description: "Nous Portal 구독(또는 다른 OAuth 제공자)을 외부 앱용 OpenAI 호환 엔드포인트로 사용"
---

# 구독 프록시

구독 프록시는 외부 앱 — OpenViking, Karakeep, Open WebUI 등 OpenAI 호환 채팅 완성을 지원하는 모든 앱 — 이 Hermes가 관리하는 제공자 구독을 LLM 엔드포인트로 사용할 수 있게 하는 로컬 HTTP 서버입니다. 프록시는 올바른 자격 증명을 자동으로 갱신해 연결하므로 앱에는 정적 API 키가 필요하지 않습니다.

[API 서버](./api-server.md)와는 다릅니다.

| | API 서버 | 구독 프록시 |
|---|---|---|
| 제공하는 것 | 에이전트 (전체 도구 모음, 메모리, 스킬) | 원시 모델 추론 |
| 사용 사례 | "Hermes를 채팅 백엔드로 사용" | "다른 앱에서 내 Portal 구독 사용" |
| 인증 | `API_SERVER_KEY` | 모든 bearer (프록시가 실제 키를 연결) |
| 도구 호출 | 예 — 에이전트가 도구를 실행 | 아니요 — 전달만 수행 |

**에이전트**를 백엔드로 사용하려면 API 서버를 사용하세요. 구독을 통해 **모델**만 사용하려면 프록시를 사용하세요.

## 빠른 시작

### 1. 제공자에 로그인 (최초 한 번)

```bash
hermes portal
```

브라우저에서 Nous Portal OAuth 흐름이 열립니다. Hermes는 갱신 토큰을 `~/.hermes/auth.json`에 저장합니다 — 모든 Hermes 제공자 로그인이 저장되는 곳과 같습니다.

### 2. 프록시 시작

```bash
hermes proxy start
```

```
Starting Hermes proxy for Nous Portal
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your subscription)
  Use any bearer token in the client — the proxy attaches your real credential.
```

이 프로세스를 포그라운드에서 실행하세요. 로그아웃 후에도 계속 실행하려면 `tmux`, `nohup` 또는 systemd 유닛을 사용하세요.

### 3. 앱 연결

모든 OpenAI 호환 앱 구성에는 다음 세 항목이 동일하게 필요합니다.

```
Base URL:   http://127.0.0.1:8645/v1
API key:    anything (e.g. "sk-unused")
Model:      Hermes-4-70B    # or Hermes-4.3-36B, Hermes-4-405B
```

프록시는 앱에서 보낸 `Authorization` 헤더를 무시하고 실제 Portal 자격 증명을 업스트림 요청에 연결합니다. bearer가 만료에 가까워지면 자동으로 갱신됩니다.

## 사용 가능한 제공자

```bash
hermes proxy providers
```

현재 제공되는 것은 `nous` (Nous Portal)와 `xai` (xAI / Grok)입니다. `hermes_cli/proxy/adapters/`의 `UpstreamAdapter` 인터페이스를 구현하면 OAuth 제공자를 더 추가할 수 있습니다.

## 상태 확인

```bash
hermes proxy status
```

```
Hermes proxy upstream adapters

  [nous    ] Nous Portal — ready (bearer expires 2026-05-15T06:43:21Z)
```

`not logged in`이 표시되면 `hermes portal`을 실행하세요. `credentials need attention`이 표시되면 갱신 토큰이 폐기된 것입니다(드물지만 Portal 웹 UI에서 로그아웃하면 발생함). `hermes portal`을 다시 실행하면 됩니다.

## 허용되는 경로

프록시는 업스트림이 실제로 제공하는 경로만 전달합니다. Nous Portal의 경우:

| 경로 | 용도 |
|------|---------|
| `/v1/chat/completions` | 채팅 완성 (스트리밍 + 비스트리밍) |
| `/v1/completions` | 레거시 텍스트 완성 |
| `/v1/embeddings` | 임베딩 |
| `/v1/models` | 모델 목록 |

그 밖의 경로(`/v1/images/generations`, `/v1/audio/speech` 등)는 허용된 경로를 명확히 안내하는 오류와 함께 404를 반환합니다. 이를 통해 부주의한 클라이언트가 업스트림으로 이상한 요청을 유출하지 않도록 합니다.

## OpenViking에서 Portal 사용 구성

[OpenViking](https://github.com/volcengine/OpenViking)은 VLM(메모리 추출에 사용하는 비전/언어 모델)과 임베딩 모델에 사용할 LLM 제공자가 필요한 컨텍스트 데이터베이스입니다. 프록시를 사용하면 `vlm.api_base`가 로컬 프록시를 가리키도록 설정할 수 있습니다.

`~/.openviking/ov.conf`를 수정하세요.

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

그런 다음 `openviking-server`와 함께 터미널에서 프록시를 시작하세요.

```bash
# Terminal 1
hermes proxy start

# Terminal 2
openviking-server
```

이제 OpenViking의 VLM 호출이 Portal 구독을 통해 전달됩니다. 임베딩 모델 측에는 여전히 별도의 제공자가 필요합니다 — Portal은 `/v1/embeddings`를 제공하지만 모델 선택은 사용 중인 요금제가 지원하는 항목에 따라 달라집니다. `portal.nousresearch.com/models`를 확인하세요.

## Karakeep(또는 모든 북마크/요약 앱) 구성

[Karakeep](https://karakeep.app/)은 북마크 요약을 위해 OpenAI 호환 API를 사용합니다. 구성에서 다음을 설정하세요.

```bash
# Karakeep .env
OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
OPENAI_API_KEY=any-non-empty-string
INFERENCE_TEXT_MODEL=Hermes-4-70B
```

Open WebUI, LobeChat, NextChat 또는 기타 OpenAI 호환 클라이언트에도 같은 패턴을 사용할 수 있습니다.

## LAN에 노출

기본적으로 프록시는 `127.0.0.1`(localhost 전용)에 바인딩됩니다. 네트워크의 다른 컴퓨터에서 사용하도록 하려면 다음을 실행하세요.

```bash
hermes proxy start --host 0.0.0.0 --port 8645
```

⚠ **주의:** 이제 네트워크의 누구나 Portal 구독을 사용할 수 있습니다. 프록시 자체에는 인증이 없으며 모든 bearer를 허용합니다. 신뢰하는 네트워크 외부로 노출한다면 방화벽, VPN 또는 적절한 인증이 있는 리버스 프록시를 사용하세요.

## 요청 제한

Portal 요금제의 RPM/TPM 제한은 프록시 전체에 적용됩니다. 프록시는 여러 경로로 분산하거나 풀링하지 않습니다 — 전체 구독 할당량을 사용하는 하나의 bearer입니다. [portal.nousresearch.com](https://portal.nousresearch.com)에서 사용량을 모니터링하세요.

## 아키텍처

프록시는 의도적으로 최소한으로 구성되어 있습니다. 요청마다 다음을 수행합니다.

1. 앱에서 `POST /v1/chat/completions` 수신
2. 어댑터의 현재 자격 증명 조회(만료 예정이면 갱신)
3. `Authorization: Bearer <minted-key>`와 함께 요청 본문을 그대로 전달
4. 응답을 변경하지 않고 스트리밍으로 반환(SSE 유지)

변환하지 않습니다. 요청 본문을 기록하지 않습니다. 에이전트 루프도 없습니다. 프록시는 자격 증명을 연결하는 전달 계층입니다.

## 향후 계획: 더 많은 OAuth 제공자

어댑터 시스템은 플러그 방식입니다. 새 제공자(예: HuggingFace, GitHub Copilot의 채팅 엔드포인트, OAuth를 통한 Anthropic)를 추가하려면 `hermes_cli/proxy/adapters/<provider>.py`에서 `UpstreamAdapter`를 구현하고 `adapters/__init__.py`에 등록해야 합니다. 프로토콜 수준에서 OpenAI 호환이 아닌 제공자(예를 들어 Anthropic Messages API)는 변환 계층이 필요하며, 현재 구조에서는 범위에 포함되지 않습니다.
