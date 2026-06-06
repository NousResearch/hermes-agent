---
title: "Telephony — 핵심 도구 변경 없이 Hermes에게 전화 기능 부여하기"
sidebar_label: "Telephony"
description: "핵심 도구 변경 없이 Hermes에게 전화 기능 부여하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Telephony

핵심 도구 변경 없이 Hermes에게 전화 기능을 부여합니다. Twilio 번호를 프로비저닝하여 유지하고, SMS/MMS를 송수신하며, 직접 전화를 걸거나, Bland.ai 또는 Vapi를 통해 AI 기반의 발신 전화를 걸 수 있습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택적(Optional) — `hermes skills install official/productivity/telephony` 명령어로 설치 |
| 경로 | `optional-skills/productivity/telephony` |
| 버전 | `1.0.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `telephony`, `phone`, `sms`, `mms`, `voice`, `twilio`, `bland.ai`, `vapi`, `calling`, `texting` |
| 관련 스킬 | [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps), [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`agentmail`](/docs/user-guide/skills/optional/email/email-agentmail) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되어 있을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Telephony — 핵심 도구 변경 없는 번호, 전화, 문자 기능

이 선택적 스킬은 전화 관련 기능을 핵심 도구 목록에 추가하지 않고도 Hermes에게 실용적인 전화 기능을 제공합니다.

이 스킬에는 다음과 같은 작업을 수행할 수 있는 헬퍼 스크립트 `scripts/telephony.py`가 포함되어 있습니다:
- 공급자 자격 증명을 `~/.hermes/.env`에 저장
- Twilio 전화번호 검색 및 구매
- 이후 세션을 위해 소유한 번호 기억
- 소유한 번호로 SMS / MMS 전송
- 웹훅 서버 없이 해당 번호의 수신 SMS 폴링(polling)
- TwiML `<Say>` 또는 `<Play>`를 사용해 직접 Twilio 전화 걸기
- 소유한 Twilio 번호를 Vapi로 가져오기
- Bland.ai 또는 Vapi를 통해 AI 아웃바운드 전화 걸기

## 해결하는 문제

이 스킬은 사용자가 실제로 원하는 다음과 같은 실용적인 전화 작업을 처리하기 위한 것입니다:
- 아웃바운드 전화 (발신)
- 문자 전송
- 재사용 가능한 에이전트 번호 소유
- 나중에 해당 번호로 수신된 메시지 확인
- 세션 간 번호 및 관련 ID 유지
- 수신 SMS 폴링 및 기타 자동화를 위한 향후 확장이 용이한 전화 ID 유지

이 스킬은 Hermes를 실시간 인바운드 전화 게이트웨이로 만들지 **않습니다**. 수신 SMS는 Twilio REST API를 폴링하여 처리합니다. 이것은 핵심 웹훅 인프라를 추가하지 않고도 알림이나 일회성 코드 수신 등을 포함한 많은 워크플로우에 충분합니다.

## 안전 규칙 — 필수 사항

1. 전화를 걸거나 문자를 보내기 전에 항상 확인하세요.
2. 응급 번호로는 절대 전화를 걸지 마세요.
3. 괴롭힘, 스팸, 사칭 또는 불법적인 목적에 전화를 절대 사용하지 마세요.
4. 서드파티(제3자) 전화번호를 민감한 운영 데이터로 취급하세요:
   - Hermes의 메모리에 저장하지 마세요.
   - 사용자가 명시적으로 원하지 않는 한, 스킬 문서, 요약 또는 후속 노트에 포함하지 마세요.
5. **에이전트가 소유한 Twilio 번호**는 사용자의 설정 일부이므로 유지(persist)하는 것이 좋습니다.
6. VoIP 번호는 모든 서드파티 2FA(이중 인증) 흐름에서 **작동한다고 보장되지 않습니다.** 주의해서 사용하고 사용자에게 기대치를 명확히 설정하세요.

## 결정 트리 (Decision Tree) — 어떤 서비스를 사용할까?

하드코딩된 제공자 라우팅 대신 다음 논리를 사용하세요:

### 1) "Hermes가 진짜 전화번호를 소유했으면 좋겠어"
**Twilio**를 사용하세요.

이유:
- 번호를 구매하고 유지하는 가장 쉬운 방법
- 최고의 SMS / MMS 지원
- 가장 간단한 수신 SMS 폴링 방식
- 향후 인바운드 웹훅이나 통화 처리를 위한 깔끔한 방법

사용 사례:
- 나중에 문자 수신하기
- 배포 알림 / 크론(cron) 알림 보내기
- 에이전트를 위한 재사용 가능한 전화 ID 유지하기
- 추후 전화 기반 인증 흐름 실험하기

### 2) "지금 당장 가장 쉬운 아웃바운드 AI 전화 걸기 기능만 필요해"
**Bland.ai**를 사용하세요.

이유:
- 가장 빠른 설정
- 단일 API 키
- 먼저 번호를 구매하거나 가져올 필요 없음

단점:
- 덜 유연함
- 음성 품질이 괜찮은 편이지만 최고는 아님

### 3) "최고 품질의 대화형 AI 음성이 필요해"
**Twilio + Vapi**를 사용하세요.

이유:
- Twilio는 번호 소유권을 제공합니다.
- Vapi는 더 나은 대화형 AI 통화 품질과 더 많은 음성/모델의 유연성을 제공합니다.

권장 흐름:
1. Twilio 번호 구매/저장
2. 번호를 Vapi로 가져오기
3. 반환된 `VAPI_PHONE_NUMBER_ID` 저장
4. `ai-call --provider vapi` 사용

### 4) "사전 녹음된 내 음성 메시지로 전화를 걸고 싶어"
공개 오디오 URL을 활용한 **Twilio 직접 통화(direct call)**를 사용하세요.

이유:
- 사용자 지정 MP3를 재생하는 가장 쉬운 방법
- 공개 파일 호스트 또는 터널과 결합하여 Hermes의 `text_to_speech` 기능과 잘 맞음

## 파일 및 영구 상태 (Persistent state)

이 스킬은 두 곳에 전화 상태를 유지합니다:

### `~/.hermes/.env`
수명이 긴 공급자 자격 증명 및 소유한 번호 ID에 사용됩니다. 예:
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_PHONE_NUMBER_SID`
- `BLAND_API_KEY`
- `VAPI_API_KEY`
- `VAPI_PHONE_NUMBER_ID`
- `PHONE_PROVIDER` (AI 통화 공급자: bland 또는 vapi)

### `~/.hermes/telephony_state.json`
세션 간에 유지되어야 하는 스킬 전용 상태에 사용됩니다. 예:
- 기억된 기본 Twilio 번호 / SID
- 기억된 Vapi 전화번호 ID
- 수신함 폴링 체크포인트를 위한 마지막 수신 메시지 SID/날짜

의미:
- 스킬이 다시 로드될 때, `diagnose`를 통해 이미 어떤 번호가 구성되어 있는지 알 수 있습니다.
- `twilio-inbox --since-last --mark-seen`이 이전 체크포인트부터 계속 실행될 수 있습니다.

## 헬퍼 스크립트 찾기

이 스킬을 설치한 후, 다음과 같이 스크립트를 찾으세요:

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/telephony/scripts/telephony.py' -print -quit)"
```

`SCRIPT`가 비어 있다면 스킬이 아직 설치되지 않은 것입니다.

## 설치

이것은 공식 선택 스킬이므로 Skills Hub에서 설치하세요:

```bash
hermes skills search telephony
hermes skills install official/productivity/telephony
```

## 공급자 설정

### Twilio — 소유 번호, SMS/MMS, 직접 통화, 수신 SMS 폴링

가입:
- https://www.twilio.com/try-twilio

그런 다음 Hermes에 자격 증명을 저장합니다:

```bash
python3 "$SCRIPT" save-twilio ACXXXXXXXXXXXXXXXXXXXXXXXXXXXX your_auth_token_here
```

사용 가능한 번호 검색:

```bash
python3 "$SCRIPT" twilio-search --country US --area-code 702 --limit 5
```

번호 구매 및 기억:

```bash
python3 "$SCRIPT" twilio-buy "+17025551234" --save-env
```

소유한 번호 나열:

```bash
python3 "$SCRIPT" twilio-owned
```

나중에 그 중 하나를 기본값으로 설정:

```bash
python3 "$SCRIPT" twilio-set-default "+17025551234" --save-env
# 또는
python3 "$SCRIPT" twilio-set-default PNXXXXXXXXXXXXXXXXXXXXXXXXXXXX --save-env
```

### Bland.ai — 가장 쉬운 아웃바운드 AI 통화

가입:
- https://app.bland.ai

설정 저장:

```bash
python3 "$SCRIPT" save-bland your_bland_api_key --voice mason
```

### Vapi — 더 나은 대화형 음성 품질

가입:
- https://dashboard.vapi.ai

먼저 API 키를 저장합니다:

```bash
python3 "$SCRIPT" save-vapi your_vapi_api_key
```

소유한 Twilio 번호를 Vapi로 가져오고 반환된 전화번호 ID를 저장합니다:

```bash
python3 "$SCRIPT" vapi-import-twilio --save-env
```

Vapi 전화번호 ID를 이미 알고 있다면 직접 저장하세요:

```bash
python3 "$SCRIPT" save-vapi your_vapi_api_key --phone-number-id vapi_phone_number_id_here
```

## 현재 상태 진단 (Diagnose)

언제든지 스킬이 이미 알고 있는 정보를 확인하세요:

```bash
python3 "$SCRIPT" diagnose
```

나중에 세션에서 작업을 재개할 때 먼저 이 명령을 사용하세요.

## 일반적인 워크플로우

### A. 에이전트 번호를 구매하고 나중에 계속 사용하기

1. Twilio 자격 증명 저장:
```bash
python3 "$SCRIPT" save-twilio AC... auth_token_here
```

2. 번호 검색:
```bash
python3 "$SCRIPT" twilio-search --country US --area-code 702 --limit 10
```

3. 번호를 구매하고 `~/.hermes/.env` + 상태(state)에 저장:
```bash
python3 "$SCRIPT" twilio-buy "+17025551234" --save-env
```

4. 다음 세션에서 다음을 실행:
```bash
python3 "$SCRIPT" diagnose
```
기억된 기본 번호와 수신함 체크포인트 상태를 보여줍니다.

### B. 에이전트 번호에서 문자 보내기

```bash
python3 "$SCRIPT" twilio-send-sms "+15551230000" "Your deployment completed successfully."
```

미디어 포함:

```bash
python3 "$SCRIPT" twilio-send-sms "+15551230000" "Here is the chart." --media-url "https://example.com/chart.png"
```

### C. 웹훅 서버 없이 나중에 수신된 문자 확인하기

기본 Twilio 번호의 수신함을 폴링합니다:

```bash
python3 "$SCRIPT" twilio-inbox --limit 20
```

마지막 체크포인트 이후에 도착한 메시지만 표시하고, 읽기를 마친 후 체크포인트를 진행(advance)합니다:

```bash
python3 "$SCRIPT" twilio-inbox --since-last --mark-seen
```

이것이 "다음에 스킬을 로드할 때 이 번호가 받은 메시지에 어떻게 접근하나요?"에 대한 주요 답변입니다.

### D. 내장 TTS를 사용한 직접 Twilio 통화

```bash
python3 "$SCRIPT" twilio-call "+15551230000" --message "Hello! This is Hermes calling with your status update." --voice Polly.Joanna
```

### E. 사전 녹음된 / 사용자 지정 음성 메시지로 전화하기

Hermes의 기존 `text_to_speech` 지원을 재사용하기 위한 주요 방법입니다.

다음과 같은 경우에 사용하세요:
- Twilio `<Say>` 기능 대신 Hermes에 설정된 TTS 음성으로 전화를 걸고자 할 때
- 단방향 음성 전달이 필요할 때 (브리핑, 알림, 농담, 리마인더, 상태 업데이트 등)
- 실시간 대화형 전화가 **필요하지 않을 때**

오디오를 별도로 생성하거나 호스팅한 다음:

```bash
python3 "$SCRIPT" twilio-call "+155****0000" --audio-url "https://example.com/briefing.mp3"
```

권장되는 Hermes TTS -> Twilio Play 워크플로우:

1. Hermes `text_to_speech`를 사용하여 오디오를 생성합니다.
2. 생성된 MP3 파일을 외부에서 공개적으로 접근할 수 있도록 만듭니다.
3. `--audio-url` 파라미터와 함께 Twilio 전화를 겁니다.

에이전트 처리 예시:
- Hermes에게 `text_to_speech`를 사용해 메시지 오디오를 생성하도록 요청
- 필요한 경우 임시 정적 호스트 / 터널 / 오브젝트 스토리지 URL을 통해 파일 노출
- `twilio-call --audio-url ...`을 사용해 전화로 음성 전달

MP3에 적합한 호스팅 옵션:
- 임시 공개 오브젝트/스토리지 URL
- 로컬 정적 파일 서버로 연결되는 수명이 짧은 터널
- 전화 서비스 제공자가 직접 가져올 수 있는 모든 기존 HTTPS URL

중요 참고 사항:
- Hermes TTS는 사전에 녹음된 아웃바운드 메시지에 매우 적합합니다.
- Bland/Vapi는 실시간 텔레포니 오디오 스택을 자체적으로 처리하므로 **실시간 대화형 AI 통화**에 더 적합합니다.
- 여기서 Hermes STT/TTS 단독으로는 완전한 양방향 전화 대화 엔진으로 사용되지 않습니다. 이를 위해서는 이 스킬이 도입하려는 것보다 훨씬 무거운 스트리밍/웹훅 통합이 필요합니다.

### F. Twilio 직접 전화를 사용한 폰 트리(Phone Tree) / IVR 내비게이션 탐색

전화가 연결된 후 숫자를 눌러야 한다면 `--send-digits`를 사용하세요.
Twilio는 `w`를 짧은 대기로 해석합니다.

```bash
python3 "$SCRIPT" twilio-call "+18005551234" --message "Connecting to billing now." --send-digits "ww1w2w3"
```

이 기능은 사람에게 연결하거나 짧은 상태 메시지를 전달하기 전에 특정 메뉴 분기에 도달하는 데 유용합니다.

### G. Bland.ai를 통한 아웃바운드 AI 통화

```bash
python3 "$SCRIPT" ai-call "+15551230000" "Call the dental office, ask for a cleaning appointment on Tuesday afternoon, and if they do not have Tuesday availability, ask for Wednesday or Thursday instead." --provider bland --voice mason --max-duration 3
```

상태 확인:

```bash
python3 "$SCRIPT" ai-status <call_id> --provider bland
```

통화가 끝난 후 Bland 분석 질문하기:

```bash
python3 "$SCRIPT" ai-status <call_id> --provider bland --analyze "Was the appointment confirmed?,What date and time?,Any special instructions?"
```

### H. 내가 소유한 번호로 Vapi를 통한 아웃바운드 AI 통화

1. Twilio 번호를 Vapi로 가져오기:
```bash
python3 "$SCRIPT" vapi-import-twilio --save-env
```

2. 전화 걸기:
```bash
python3 "$SCRIPT" ai-call "+15551230000" "You are calling to make a dinner reservation for two at 7:30 PM. If that is unavailable, ask for the nearest time between 6:30 and 8:30 PM." --provider vapi --max-duration 4
```

3. 결과 확인:
```bash
python3 "$SCRIPT" ai-status <call_id> --provider vapi
```

## 제안하는 에이전트 절차

사용자가 전화나 문자를 요청할 때:

1. 결정 트리(Decision tree)를 통해 요청에 맞는 경로를 결정합니다.
2. 구성 상태가 불확실한 경우 `diagnose`를 실행합니다.
3. 전체 작업 세부 정보를 수집합니다.
4. 전화를 걸거나 문자를 보내기 전에 사용자와 확인합니다.
5. 올바른 명령어를 사용합니다.
6. 필요한 경우 결과를 폴링(poll)합니다.
7. 서드파티 번호를 Hermes 메모리에 저장하지 않고 결과를 요약합니다.

## 이 스킬이 지원하지 않는 기능

- 실시간 인바운드 통화 응답
- 웹훅 기반의 에이전트 루프 내 실시간 SMS 푸시 수신
- 임의의 서드파티 2FA 제공자에 대한 완벽한 호환성 보장

이러한 기능들은 순수 선택적 스킬보다 더 많은 인프라를 필요로 합니다.

## 주의 사항 (Pitfalls)

- Twilio 평가판 계정 및 지역 규칙에 따라 전화를 걸거나 문자를 보낼 수 있는 대상이 제한될 수 있습니다.
- 일부 서비스는 2FA에 대해 VoIP 번호를 거부합니다.
- `twilio-inbox`는 REST API를 폴링하며 즉각적인 푸시 전송이 아닙니다.
- Vapi 아웃바운드 통화는 가져온 유효한 번호가 있는지 여부에 의존합니다.
- Bland가 설정은 가장 쉽지만, 항상 최상의 음질을 보장하지는 않습니다.
- 임의의 서드파티 전화번호를 Hermes 메모리에 저장하지 마세요.

## 검증 체크리스트

설치 후 이 스킬만으로 다음 모든 작업을 수행할 수 있어야 합니다:

1. `diagnose`를 통해 제공자 준비 상태와 기억된 상태를 표시할 수 있습니다.
2. Twilio 번호를 검색하고 구매할 수 있습니다.
3. 해당 번호를 `~/.hermes/.env`에 유지(persist)할 수 있습니다.
4. 소유한 번호에서 SMS를 전송할 수 있습니다.
5. 나중에 소유한 번호의 수신 문자를 폴링할 수 있습니다.
6. 직접 Twilio 전화를 걸 수 있습니다.
7. Bland 또는 Vapi를 통해 AI 통화를 걸 수 있습니다.

## 참조 링크

- Twilio 전화번호: https://www.twilio.com/docs/phone-numbers/api
- Twilio 메시징: https://www.twilio.com/docs/messaging/api/message-resource
- Twilio 음성 통화: https://www.twilio.com/docs/voice/api/call-resource
- Vapi 문서: https://docs.vapi.ai/
- Bland.ai: https://app.bland.ai/
