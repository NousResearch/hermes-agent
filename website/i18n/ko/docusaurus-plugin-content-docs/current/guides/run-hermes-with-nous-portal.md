---
sidebar_position: 1
title: "Nous Portal로 Hermes Agent 실행하기"
description: "처음부터 끝까지 따라 하는 안내: 구독, 설정, 모델 전환, 게이트웨이 도구 활성화 및 라우팅 확인"
---

# Nous Portal로 Hermes Agent 실행하기

이 가이드는 [Nous Portal](https://portal.nousresearch.com) 구독으로 Hermes Agent를 처음부터 끝까지 실행하는 방법을 안내합니다. 가입부터 모든 도구가 올바르게 라우팅되는지 확인하는 과정까지 다룹니다. Portal이 무엇이고 구독에 무엇이 포함되는지 개괄적으로만 알고 싶다면 [Nous Portal 통합 페이지](/integrations/nous-portal)를 참고하세요. 이 페이지는 실제 작업 순서에 따른 안내서입니다.

## 사전 준비

- Hermes Agent 설치 ([빠른 시작](/getting-started/quickstart))
- 설정할 컴퓨터의 웹 브라우저 (또는 SSH 포트 포워딩 — [SSH를 통한 OAuth](/guides/oauth-over-ssh) 참고)
- 약 5분

다음은 **필요하지 않습니다**: OpenAI 키, Anthropic 키, Firecrawl 계정, FAL 계정, Browser Use 계정 또는 그 밖의 벤더별 자격 증명. 이것이 바로 이 서비스의 핵심입니다.

## 1. 구독하기

[portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)을 열고 가입한 다음 요금제를 선택하세요.

이미 구독 중인가요? 2단계로 건너뛰세요.

## 2. 한 번에 설정 실행하기

```bash
hermes setup --portal
```

이 단일 명령은 다음 다섯 가지를 수행합니다.

1. OAuth 로그인을 위해 브라우저에서 portal.nousresearch.com을 엽니다
2. 갱신 토큰을 `~/.hermes/auth.json`에 저장합니다
3. `~/.hermes/config.yaml`에 `model.provider: nous`를 설정합니다
4. 기본 에이전트 모델(`anthropic/claude-sonnet-4.6` 또는 유사 모델)을 선택합니다
5. 웹 검색, 이미지 생성, TTS 및 브라우저 자동화를 위한 Tool Gateway를 켭니다

완료되면 터미널로 돌아와 바로 대화할 수 있습니다.

### SSH로 서버에 접속해 있다면?

OAuth에는 브라우저가 필요하지만, 루프백 콜백은 Hermes가 실행 중인 컴퓨터에서 처리됩니다. 두 가지 방법이 있습니다.

```bash
# Option A: SSH port forwarding (preferred)
ssh -N -L 8642:127.0.0.1:8642 user@remote-host    # in a local terminal
hermes setup --portal                              # on the remote, open the printed URL in your local browser

# Option B: device-code login (works from Cloud Shell, Codespaces, EC2 Instance Connect)
hermes auth add nous --type oauth
# Then re-run `hermes setup --portal` to wire the provider + gateway
```

ProxyJump 체인, mosh/tmux 및 ControlMaster 관련 주의 사항을 포함한 전체 안내는 [SSH / 원격 호스트를 통한 OAuth](/guides/oauth-over-ssh)를 참고하세요.

## 3. 제대로 작동하는지 확인하기

```bash
hermes portal info
```

다음과 같은 내용이 표시되어야 합니다.

```
  Nous Portal
  ───────────
  Auth:    ✓ logged in
  Portal:  https://portal.nousresearch.com
  Model:   ✓ using Nous as inference provider

  Tool Gateway
  ────────────
  Web search & extract  via Nous Portal
  Image generation      via Nous Portal
  Text-to-speech        via Nous Portal
  Browser automation    via Nous Portal
```

어느 줄이든 `via Nous Portal` 이외의 내용이 표시되거나 인증 줄에 `not logged in`이 표시되면 아래의 [문제 해결](#troubleshooting)로 이동하세요.

## 4. 첫 대화 실행하기

```bash
hermes chat
```

모델과 Tool Gateway를 모두 사용해 보는 요청을 시도하세요.

```
Hey, search the web for "Hermes Agent release notes" and summarize the top 3 hits.
```

Hermes가 `web_search`(게이트웨이를 통해 Firecrawl을 사용하는 도구)를 호출한 뒤 요약을 응답하는 것을 볼 수 있습니다. 검색이 실행되고 응답이 타당하다면 완료된 것입니다 — Portal이 처음부터 끝까지 연결되었습니다.

## 5. 실제로 원하는 모델 선택하기

`hermes setup --portal`을 사용하면 설정 중에 모델을 선택할 수 있지만, 이 구독의 핵심은 전체 카탈로그에 접근할 수 있다는 점입니다 — 세션 중 언제든 `/model`로 전환하세요.

```bash
/model anthropic/claude-sonnet-4.6     # best general-purpose agentic
/model openai/gpt-5.4                  # strong reasoning + tool calling
/model google/gemini-2.5-pro           # huge context window
/model deepseek/deepseek-v3.2          # cost-effective coder
/model anthropic/claude-opus-4.6       # heavyweight for hard problems
```

또는 선택기를 열어 둘러볼 수 있습니다.

```bash
/model
```

영구적으로 다른 기본값을 선택하려면 다음을 실행하세요.

```bash
# in your terminal, outside any session
hermes config set model.default anthropic/claude-sonnet-4.6
```

### 에이전트 작업에는 Hermes-4를 선택하지 마세요

Hermes-4-70B와 Hermes-4-405B는 Portal에서 큰 할인으로 이용할 수 있지만 **도구 호출에 맞게 튜닝된 모델이 아니라 채팅/추론 모델**입니다. 다단계 에이전트 루프에서 어려움을 겪을 수 있습니다. 에이전트가 아닌 도구에서 [구독 프록시](/user-guide/features/subscription-proxy)를 통해 대화/조사 작업에 사용하세요. Hermes Agent 자체에서는 위의 최첨단 에이전트 모델을 사용하세요.

Portal의 공식 [정보 페이지](https://portal.nousresearch.com/info)에도 이 경고가 있습니다 — 이는 Hermes 측의 단순한 의견이 아니라 Nous의 공식 안내입니다.

## 6. (선택 사항) Tool Gateway 라우팅 사용자 지정하기

게이트웨이는 전체를 한꺼번에 켜거나 끄는 방식이 아니라 도구별로 선택할 수 있습니다. 이미 Browserbase 계정이 있고 웹 검색과 이미지 생성만 Nous를 통해 라우팅하고 싶다면 이 구성이 지원됩니다.

```bash
hermes tools
# → Web search       → "Nous Subscription"     (recommended)
# → Image generation → "Nous Subscription"     (recommended)
# → Browser          → "Browserbase"           (your existing key)
# → TTS              → "Nous Subscription"     (recommended)
```

Nous Portal에 로그인하기 전에도 `hermes tools`에 이 항목들이 표시됩니다 — 활성 세션 없이 "Nous Subscription"을 선택하면 Hermes가 추론 공급자나 다른 도구를 변경하지 않고 Portal 로그인을 그 자리에서 실행합니다.

다음 명령으로 도구 조합을 확인하세요.

```bash
hermes portal tools
```

도구별 라우팅을 확인할 수 있습니다 — 구독을 통해 라우팅되는 도구에는 `via Nous Portal`이, 자체 키를 사용하는 도구에는 파트너 이름(`browserbase`, `firecrawl` 등)이 표시됩니다.

## 7. (선택 사항) 음성 모드 활성화하기

Tool Gateway에 OpenAI TTS가 포함되어 있으므로 [음성 모드](/user-guide/features/voice-mode)를 별도의 OpenAI 키 없이 사용할 수 있습니다.

```bash
hermes setup tts
# → pick "Nous Subscription" for TTS
# → pick a speech-to-text backend (local faster-whisper is free, no setup)
```

그런 다음 모든 메시징 플랫폼 세션(Telegram, Discord, Signal 등)에서 음성 메시지를 보내면 Hermes가 Portal 구독만으로 음성을 전사하고, 응답하고, 합성 음성으로 답장합니다.

## 8. (선택 사항) Cron + 상시 실행 워크플로

Portal 구독은 대화형 채팅과 동일한 방식으로 [cron 작업](/user-guide/features/cron)과 [배치 처리](/user-guide/features/batch-processing)에 사용할 수 있습니다 — OAuth 갱신 토큰이 자동으로 재사용됩니다. 추가 설정은 필요하지 않습니다. cron 작업을 예약하면 구독 사용량으로 청구됩니다.

```bash
hermes cron create "0 9 * * *" \
  "Search the web for top AI news and summarize the 5 most important stories" \
  --name "Daily AI news"
```

cron 작업은 사용자가 지켜보지 않아도 실행되며, 모델 호출과 웹 검색 및 요약을 모두 Portal 구독을 통해 수행합니다.

## 프로필 및 다중 사용자 설정

[Hermes 프로필](/user-guide/profiles)(예: 프로젝트별 별도 설정)을 사용하는 경우 Portal 갱신 토큰은 공유 토큰 저장소를 통해 모든 프로필에서 자동으로 공유됩니다. 어느 프로필에서든 한 번 로그인하면 나머지 프로필이 자동으로 토큰을 사용합니다.

여러 사람이 한 컴퓨터를 함께 사용하는 팀 환경에서는 각자 Portal 계정을 사용하고, 각자의 홈 디렉터리에 자체 `~/.hermes/auth.json`이 저장되므로 사용자 간에 토큰이 공유되지 않습니다. 이것이 올바른 경계입니다.

## 문제 해결

### `hermes setup --portal` 후 `hermes portal info`에 "not logged in"이 표시됨

OAuth 흐름이 완료되지 않았습니다. 다시 실행하세요.

```bash
hermes portal
```

브라우저가 열리지 않거나 콜백이 실패한다면 원격 또는 헤드리스 호스트를 사용 중일 가능성이 큽니다 — 포트 포워딩 해결 방법은 [SSH를 통한 OAuth](/guides/oauth-over-ssh)를 참고하세요.

### "using Nous as inference provider" 대신 "Model: currently openrouter" (또는 다른 공급자)가 표시됨

로컬 설정이 변경되었습니다. OAuth는 성공했지만 `model.provider`가 여전히 다른 공급자를 가리키고 있습니다. 다음으로 수정하세요.

```bash
hermes config set model.provider nous
```

또는 대화형으로 실행하세요.

```bash
hermes model
# pick Nous Portal
```

`hermes portal info`로 다시 확인하세요.

### Tool Gateway 도구에 "via Nous Portal" 대신 파트너 이름이 표시됨

도구별 설정이 게이트웨이를 덮어쓰고 있습니다. 다음을 실행하세요.

```bash
hermes tools
# pick "Nous Subscription" for any tool you want gateway-routed
```

일부 사용자는 의도적으로 조합해 사용합니다 — 예를 들어 웹은 Nous를 통해 라우팅하면서 브라우저에는 자신의 Browserbase 키를 사용할 수 있습니다. 의도한 구성이라면 그대로 두세요. 아니라면 이 명령으로 수정할 수 있습니다.

### 세션 중 "Re-authentication required"가 표시됨

Portal 갱신 토큰이 무효화되었습니다(비밀번호 변경, 수동 폐기 또는 세션 만료). Hermes가 토큰을 계속 재사용하지 않도록 토큰이 로컬에서 격리되었습니다. 다시 로그인하기만 하면 됩니다.

```bash
hermes auth add nous
```

다시 로그인에 성공하면 격리가 자동으로 해제됩니다.

### 원하는 모델이 `/model` 선택기에 없음

Portal 카탈로그는 OpenRouter의 모델 목록(300개 이상)과 독점 또는 보조 공급자를 통해 제공되는 모델을 함께 사용합니다. 모델이 보이지 않으면 OpenRouter 형식의 슬러그를 직접 입력해 보세요.

```bash
/model anthropic/claude-opus-4.6
/model openai/o1-2025-12-17
```

모델을 실제로 사용할 수 없다면 [이슈를 열어 주세요](https://github.com/NousResearch/hermes-agent/issues) — 대부분은 업데이트할 수 있는 라우팅 설정 누락입니다.

### Portal 계정에 청구 내역이 표시되지 않음

`hermes portal info`를 실행하면 실제로 Portal을 통해 라우팅되는지, 아니면 다른 공급자를 사용하는지 알려 줍니다. 흔한 원인은 다음과 같습니다.

- `model.provider`가 `nous`가 아닌 `openrouter`/`anthropic`/등으로 설정됨
- OAuth 갱신 실패로 다른 설정된 공급자로 대체됨
- 여러 Hermes 프로필 중 잘못된 프로필을 사용 중임(`hermes profile list`로 확인)

### 권한을 취소하고 처음부터 다시 시작하고 싶음

```bash
hermes auth logout nous       # wipes the local refresh token
# Then re-run setup or remove the subscription from the Portal web UI
```

## 숫자로 간단히 알아보는 효과

| Portal 없이 | Portal 사용 |
|----------------|-------------|
| `.env`에 OpenRouter / Anthropic / OpenAI 키 1개 | `.env` 키 없이 OAuth 갱신 토큰 1개 |
| 웹용 Firecrawl 키 1개 | 게이트웨이를 통해 웹 라우팅 |
| 이미지 생성용 FAL 키 1개 | 게이트웨이를 통해 이미지 생성 라우팅 |
| 브라우저용 Browser Use / Browserbase 키 1개 | 게이트웨이를 통해 브라우저 라우팅 |
| TTS / 음성 모드용 OpenAI 키 1개 | 게이트웨이를 통해 TTS 라우팅 |
| 별도 대시보드, 충전, 청구서 5개 | 구독 1개, 청구서 1개 |
| 컴퓨터 간: 키 5개 모두 복제 | 컴퓨터 간: OAuth 한 번만 다시 인증 |

이것이 전부입니다. 어차피 위 백엔드 중 두 개 이상을 사용한다면 구독료가 스스로 충당됩니다.

## 함께 보기

- **[Nous Portal 통합 페이지](/integrations/nous-portal)** — 구독에 포함된 항목 개요
- **[Tool Gateway](/user-guide/features/tool-gateway)** — 게이트웨이를 통해 라우팅되는 모든 도구의 상세 정보
- **[구독 프록시](/user-guide/features/subscription-proxy)** — Hermes가 아닌 도구에서 Portal 구독 사용하기
- **[음성 모드](/user-guide/features/voice-mode)** — Portal 구독으로 음성 대화 설정하기
- **[SSH를 통한 OAuth](/guides/oauth-over-ssh)** — 원격 / 헤드리스 로그인 패턴
- **[프로필](/user-guide/profiles)** — 여러 Hermes 설정에서 하나의 Portal 로그인 공유하기
