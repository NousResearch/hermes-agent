---
sidebar_position: 1
title: "Nous Portal로 Hermes Agent 실행"
description: "구독, 설정, 모델 전환, 게이트웨이 도구 활성화, 라우팅 확인 등 처음부터 끝까지 살펴보는 연습 과정입니다."
---

# Nous Portal로 Hermes Agent 실행

이 가이드는 [Nous Portal](https://portal.nousresearch.com) 구독으로 Hermes Agent를 실행하는 과정을 가입부터 모든 도구가 올바르게 라우팅되는지 확인하는 것까지 안내합니다. Portal이 무엇이고 구독에 어떤 내용이 포함되어 있는지에 대한 개요만 원한다면, [Nous Portal 통합 페이지](/integrations/nous-portal)를 참조하세요. 이 페이지는 작업 스크립트입니다.

## 사전 요구 사항

- Hermes Agent 설치됨 ([빠른 시작](/getting-started/quickstart))
- 설정 중인 머신의 웹 브라우저 (또는 SSH 포트 포워딩 — [SSH를 통한 OAuth](/guides/oauth-over-ssh) 참조)
- 약 5분의 시간

OpenAI 키, Anthropic 키, Firecrawl 계정, FAL 계정, Browser Use 계정 또는 기타 제공업체별 자격 증명은 **필요하지 않습니다**. 그것이 이 설정의 핵심입니다.

## 1. 구독 얻기

[portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)을 열어 가입하고 플랜을 선택합니다.

이미 구독하셨나요? 2단계로 건너뛰세요.

## 2. 원샷(One-shot) 설정 실행

```bash
hermes setup --portal
```

이 단일 명령어는 다섯 가지 작업을 수행합니다:

1. OAuth 로그인을 위해 브라우저를 열어 portal.nousresearch.com으로 이동합니다.
2. 새로 고침 토큰을 `~/.hermes/auth.json`에 저장합니다.
3. `~/.hermes/config.yaml`에 `model.provider: nous`를 설정합니다.
4. 기본 에이전트 모델(`anthropic/claude-sonnet-4.6` 등)을 선택합니다.
5. 웹 검색, 이미지 생성, TTS, 브라우저 자동화를 위한 도구 게이트웨이를 켭니다.

작업이 끝나면 터미널로 돌아가서 채팅할 준비가 완료됩니다.

### 서버에 SSH로 접속한 경우라면?

OAuth에는 브라우저가 필요하지만, 루프백 콜백은 Hermes가 실행되는 머신에서 작동합니다. 두 가지 옵션이 있습니다:

```bash
# 옵션 A: SSH 포트 포워딩 (권장)
ssh -N -L 8642:127.0.0.1:8642 user@remote-host    # 로컬 터미널에서
hermes setup --portal                              # 원격 머신에서 실행 후, 출력된 URL을 로컬 브라우저에서 열기

# 옵션 B: 수동 붙여넣기 (Cloud Shell, Codespaces, EC2 Instance Connect용)
hermes auth add nous --type oauth --manual-paste
# 그런 다음 `hermes setup --portal`을 다시 실행하여 제공자 및 게이트웨이 연결
```

ProxyJump 체인, mosh/tmux, ControlMaster 주의 사항을 포함한 전체 과정은 [SSH / 원격 호스트를 통한 OAuth](/guides/oauth-over-ssh)를 참조하세요.

## 3. 작동 확인

```bash
hermes portal info
```

다음과 같이 보여야 합니다:

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

어떤 줄이라도 "via Nous Portal"이 아닌 다른 내용이 표시되거나 인증 줄에 "not logged in"이라고 표시되면 아래의 [문제 해결](#문제-해결) 섹션으로 이동하세요.

## 4. 첫 번째 대화 실행

```bash
hermes chat
```

모델과 도구 게이트웨이를 모두 사용하는 작업을 시도해 보세요:

```
Hey, search the web for "Hermes Agent release notes" and summarize the top 3 hits.
```

Hermes가 게이트웨이를 통해 Firecrawl이 지원하는 `web_search`를 호출하고 요약으로 응답하는 것을 볼 수 있습니다. 검색이 실행되고 응답이 이치에 맞는다면 설정이 완료된 것입니다 — Portal이 처음부터 끝까지 연결되었습니다.

## 5. 원하는 모델 선택

`hermes setup --portal`은 설정 중에 모델을 선택하게 하지만, 구독의 핵심은 전체 카탈로그에 대한 액세스입니다. 세션 도중에 `/model`을 사용하여 언제든지 전환할 수 있습니다:

```bash
/model anthropic/claude-sonnet-4.6     # 최고 다목적 에이전트 모델
/model openai/gpt-5.4                  # 강력한 추론 + 도구 호출
/model google/gemini-2.5-pro           # 거대한 컨텍스트 창
/model deepseek/deepseek-v3.2          # 비용 효율적인 코더
/model anthropic/claude-opus-4.6       # 어려운 문제를 위한 강력한 모델
```

또는 선택기(picker)를 열어 둘러볼 수 있습니다:

```bash
/model
```

기본값을 영구적으로 다른 것으로 선택:

```bash
# 세션 외부의 터미널에서
hermes config set model.default anthropic/claude-sonnet-4.6
```

### 에이전트 작업에 Hermes-4를 선택하지 마세요

Hermes-4-70B와 Hermes-4-405B는 Portal에서 큰 할인율로 제공되지만, 이들은 도구 호출에 최적화된 것이 아니라 **채팅/추론 모델**입니다. 이들은 다단계 에이전트 루프에서 어려움을 겪을 것입니다. 대화/조사 작업에는 [Nous Chat](https://chat.nousresearch.com)을 사용하거나, 비 에이전트 도구에서 [구독 프록시](/user-guide/features/subscription-proxy)를 통해 사용하세요. Hermes Agent 자체의 경우 위의 최신 에이전트 모델을 고수하세요.

Portal 자체의 [정보 페이지](https://portal.nousresearch.com/info)에도 이 경고가 있습니다 — 이는 단순한 Hermes 측의 의견이 아니라 Nous의 공식 지침입니다.

## 6. (선택 사항) 도구 게이트웨이 라우팅 사용자 정의

게이트웨이는 전부 아니면 전무(all-or-nothing)가 아니라 도구별 옵트인(opt-in) 방식입니다. 이미 Browserbase 계정이 있고 이를 계속 사용하면서 웹 검색과 이미지 생성은 Nous를 통해 라우팅하고 싶다면 지원됩니다:

```bash
hermes tools
# → Web search       → "Nous Subscription"     (권장)
# → Image generation → "Nous Subscription"     (권장)
# → Browser          → "Browserbase"           (기존 키)
# → TTS              → "Nous Subscription"     (권장)
```

이러한 행은 Nous Portal에 로그인하기 전에도 `hermes tools`에 표시됩니다 — 활성 세션 없이 "Nous Subscription"을 선택하면 Hermes가 인라인으로 Portal 로그인을 실행합니다 (추론 제공자나 다른 도구는 변경하지 않음).

다음과 같이 구성을 확인하세요:

```bash
hermes portal tools
```

도구별 라우팅을 볼 수 있습니다 — 구독을 통해 라우팅되는 도구는 `via Nous Portal`, 자체 키를 사용하는 도구는 파트너 이름(`browserbase`, `firecrawl` 등)이 표시됩니다.

## 7. (선택 사항) 음성 모드 활성화

도구 게이트웨이에 OpenAI TTS가 포함되어 있으므로, 별도의 OpenAI 키 없이도 [음성 모드](/user-guide/features/voice-mode)가 작동합니다:

```bash
hermes setup voice
# → TTS에 "Nous Subscription" 선택
# → STT(speech-to-text) 백엔드 선택 (local faster-whisper는 무료이며 설정 필요 없음)
```

그런 다음 모든 메시징 플랫폼 세션(Telegram, Discord, Signal 등)에서 음성 메시지를 보내면 Hermes가 이를 전사하고, 응답하고, 합성된 음성으로 회신합니다 — 이 모든 것이 Portal 구독을 기반으로 합니다.

## 8. (선택 사항) 크론 + 상시 작업(always-on workflows)

Portal 구독은 대화형 채팅과 동일한 방식으로 [크론 작업(cron jobs)](/user-guide/features/cron) 및 [일괄 처리(batch processing)](/user-guide/features/batch-processing)에 작동합니다 — OAuth 새로 고침 토큰이 자동으로 재사용됩니다. 추가 설정 없이 크론 작업을 예약하기만 하면 구독에 비용이 청구됩니다.

```bash
hermes cron create "every day at 9am" \
  "Search the web for top AI news and summarize the 5 most important stories" \
  --name "Daily AI news"
```

크론 작업은 자동으로 실행되며, Portal 구독을 통해 모델 + 웹 검색 + 요약을 모두 호출합니다.

## 프로필 및 다중 사용자 설정

[Hermes 프로필](/user-guide/profiles)(예: 프로젝트별 별도 구성)을 사용하는 경우 Portal 새로 고침 토큰은 공유 토큰 저장소를 통해 모든 프로필에 자동으로 공유됩니다. 어느 프로필에서든 한 번 로그인하면 나머지 프로필에서도 자동으로 가져옵니다.

여러 사람이 머신을 공유하는 팀 설정의 경우, 각 사람마다 자신의 Portal 계정이 있고 → 각 홈 디렉터리마다 고유한 `~/.hermes/auth.json`이 있으며 → 사용자 간에 토큰이 공유되지 않습니다. 이것이 올바른 경계입니다.

## 문제 해결

### `hermes setup --portal` 이후 `hermes portal info`에 "not logged in"으로 표시됨

OAuth 흐름이 완료되지 않았습니다. 다시 실행하세요:

```bash
hermes portal
```

브라우저가 열리지 않거나 콜백이 실패하면, 원격/헤드리스 호스트에 있을 가능성이 높습니다. 포트 포워딩 및 수동 붙여넣기 해결 방법은 [SSH를 통한 OAuth](/guides/oauth-over-ssh)를 참조하세요.

### "using Nous as inference provider" 대신 "Model: currently openrouter" (또는 다른 제공자)로 표시됨

로컬 구성이 변경되었습니다. OAuth는 작동했지만 `model.provider`가 여전히 다른 제공자를 가리키고 있습니다. 수정하려면:

```bash
hermes config set model.provider nous
```

또는 대화형으로:

```bash
hermes model
# Nous Portal 선택
```

`hermes portal info`로 다시 확인하세요.

### 도구 게이트웨이 도구가 "via Nous Portal" 대신 파트너 이름을 표시함

도구별 구성이 게이트웨이를 무시하고 있습니다. 다음을 실행하세요:

```bash
hermes tools
# 게이트웨이 라우팅을 원하는 도구에 대해 "Nous Subscription"을 선택하세요
```

일부 사용자는 의도적으로 혼합합니다 — 예를 들어 웹은 Nous를 통해 라우팅하지만 브라우저에는 자체 Browserbase 키를 사용하는 것입니다. 의도적인 것이라면 그대로 두세요. 그렇지 않다면 이 명령어가 해결해 줍니다.

### 세션 중간에 "Re-authentication required"가 표시됨

Portal 새로 고침 토큰이 무효화되었습니다(비밀번호 변경, 수동 취소, 세션 만료). Hermes가 토큰을 끝없이 재사용하지 않도록 토큰이 로컬에 격리되었습니다. 다시 로그인하기만 하면 됩니다:

```bash
hermes auth add nous
```

성공적으로 다시 로그인하면 격리가 자동으로 해제됩니다.

### 원하는 모델이 `/model` 선택기에 없음

Portal 카탈로그는 OpenRouter의 모델 목록(300개 이상)을 미러링합니다. 모델이 누락된 경우 OpenRouter 스타일의 슬러그(slug)를 직접 입력해 보세요:

```bash
/model anthropic/claude-opus-4.6
/model openai/o1-2025-12-17
```

모델을 정말로 사용할 수 없다면 [이슈를 여세요](https://github.com/NousResearch/hermes-agent/issues) — 대부분의 간극은 우리가 업데이트할 수 있는 라우팅 구성 문제입니다.

### Portal 계정에 결제 정보가 나타나지 않음

`hermes portal info`는 실제로 Portal을 통해 라우팅하고 있는지 아니면 다른 제공자를 통해 라우팅하고 있는지 알려줍니다. 일반적인 원인:

- `model.provider`가 `nous` 대신 `openrouter`/`anthropic` 등으로 설정됨
- OAuth 새로 고침 실패로 인해 구성된 다른 제공자로 대체됨
- 잘못된 Hermes 프로필을 사용 중인 여러 프로필 설정 (`hermes profile current` 확인)

### 토큰을 취소하고 깨끗하게 다시 시작하고 싶음

```bash
hermes auth remove nous       # 로컬 새로 고침 토큰 지우기
# 그런 다음 설정을 다시 실행하거나 Portal 웹 UI에서 구독 제거
```

## 이 설정을 통해 얻는 이점 (숫자로 보기)

| Portal 없음 | Portal 사용 |
|----------------|-------------|
| `.env`에 1× OpenRouter / Anthropic / OpenAI 키 | 1× OAuth 새로 고침 토큰, `.env` 키 없음 |
| 웹 검색용 1× Firecrawl 키 | 웹 검색이 게이트웨이를 통해 라우팅됨 |
| 이미지 생성용 1× FAL 키 | 이미지 생성이 게이트웨이를 통해 라우팅됨 |
| 브라우저용 1× Browser Use / Browserbase 키 | 브라우저가 게이트웨이를 통해 라우팅됨 |
| TTS / 음성 모드용 1× OpenAI 키 | TTS가 게이트웨이를 통해 라우팅됨 |
| 5개의 별도 대시보드, 충전, 인보이스 | 1개의 구독, 1개의 인보이스 |
| 교차 머신: 5개의 키 모두 복제 | 교차 머신: 한 번만 재인증 (re-OAuth) |

이것이 전부입니다. 어차피 이 백엔드 중 두 개 이상을 사용하고 있다면, 구독은 그 값을 충분히 합니다.

## 참고 항목

- **[Nous Portal 통합 페이지](/integrations/nous-portal)** — 구독에 포함된 항목 개요
- **[도구 게이트웨이(Tool Gateway)](/user-guide/features/tool-gateway)** — 게이트웨이 라우팅 도구에 대한 전체 세부 정보
- **[구독 프록시(Subscription proxy)](/user-guide/features/subscription-proxy)** — Hermes가 아닌 다른 도구에서 Portal 구독 사용
- **[음성 모드(Voice mode)](/user-guide/features/voice-mode)** — Portal 구독에서 음성 대화 설정
- **[SSH를 통한 OAuth](/guides/oauth-over-ssh)** — 원격 / 헤드리스 로그인 패턴
- **[프로필(Profiles)](/user-guide/profiles)** — 여러 Hermes 구성에서 단일 Portal 로그인 공유
