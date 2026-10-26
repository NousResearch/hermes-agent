---
title: "Telephony — Twilio 번호, SMS/MMS 및 AI 발신 전화 프로비저닝"
sidebar_label: "Telephony"
description: "Twilio 번호, SMS/MMS 및 AI 발신 전화 프로비저닝"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Telephony

Twilio 번호, SMS/MMS 및 AI 발신 전화를 프로비저닝합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/productivity/telephony`로 설치 |
| 경로 | `optional-skills/productivity/telephony` |
| 버전 | `1.0.0` |
| 작성자 | Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `telephony`, `phone`, `sms`, `mms`, `voice`, `twilio`, `bland.ai`, `vapi`, `calling`, `texting` |
| 관련 스킬 | [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps), [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`agentmail`](/docs/user-guide/skills/optional/email/email-agentmail) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 활성화될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Telephony — 코어 도구를 변경하지 않고 번호, 전화, 문자 사용하기

이 선택적 스킬은 전화 기능을 코어 도구 목록에 추가하지 않고도 Hermes에서 실용적인 전화 기능을 사용할 수 있게 합니다.

다음 기능을 수행할 수 있는 도우미 스크립트 `scripts/telephony.py`가 함께 제공됩니다.
- `${HERMES_HOME:-~/.hermes}/.env`에 공급자 자격 증명 저장
- Twilio 전화번호 검색 및 구매
- 소유한 번호를 기억하여 이후 세션에서 사용
- 소유한 번호로 SMS/MMS 전송
- 웹훅 서버 없이 해당 번호로 수신된 SMS 폴링
- TwiML `<Say>` 또는 `<Play>`를 사용한 직접 Twilio 전화
- 소유한 Twilio 번호를 Vapi로 가져오기
- Bland.ai 또는 Vapi를 통한 AI 발신 전화

## 해결하는 문제

이 스킬은 사용자가 실제로 원하는 다음과 같은 전화 작업을 다루기 위한 것입니다.
- 발신 전화
- 문자 보내기
- 재사용 가능한 에이전트 번호 소유
- 나중에 해당 번호로 도착한 메시지 확인
- 세션 간에 해당 번호와 관련 ID 보존
- 수신 SMS 폴링 및 기타 자동화를 위한 미래 지향적인 전화 식별자

이 스킬은 Hermes를 실시간 수신 전화 게이트웨이로 바꾸지 **않습니다**. 수신 SMS는 Twilio REST API를 폴링하여 처리합니다. 코어 웹훅 인프라를 추가하지 않고도 알림이나 일부 일회용 코드 검색을 비롯한 여러 작업에 충분합니다.

## 안전 규칙 — 필수

1. 전화를 걸거나 문자를 보내기 전에 항상 확인을 받으세요.
2. 긴급 전화번호로 절대 전화하지 마세요.
3. 괴롭힘, 스팸, 사칭 또는 불법적인 목적에 전화를 사용하지 마세요.
4. 타인의 전화번호를 민감한 운영 데이터로 취급하세요.
   - Hermes 메모리에 저장하지 마세요.
   - 사용자가 명시적으로 원하지 않는 한 스킬 문서, 요약 또는 후속 메모에 포함하지 마세요.
5. **에이전트가 소유한 Twilio 번호**는 사용자의 구성 일부이므로 저장해도 괜찮습니다.
6. VoIP 번호가 모든 타사 2FA 흐름에서 작동한다고 보장할 수 없습니다. 주의해서 사용하고 사용자에게 기대치를 명확히 설명하세요.

## 의사 결정 트리 — 어떤 서비스를 사용할까요?

하드코딩된 공급자 라우팅 대신 다음 로직을 사용하세요.

### 1) "Hermes가 실제 전화번호를 소유하게 하고 싶다"

**Twilio**를 사용하세요.

이유:
- 번호를 구매하고 유지하는 가장 쉬운 경로
- 최고의 SMS/MMS 지원
- 가장 간단한 수신 SMS 폴링 방식
- 향후 수신 웹훅 또는 전화 처리를 위한 가장 깔끔한 경로

사용 사례:
- 나중에 문자 받기
- 배포 알림/cron 알림 보내기
- 에이전트가 재사용할 전화 식별자 유지
- 향후 전화 기반 인증 흐름 실험

### 2) "지금 당장 가장 쉬운 발신 AI 전화를 원한다"

**Bland.ai**를 사용하세요.

이유:
- 가장 빠른 설정
- API 키 하나
- 먼저 번호를 구매하거나 직접 가져올 필요 없음

절충점:
- 유연성이 낮음
- 음성 품질은 괜찮지만 최고 수준은 아님

### 3) "최고의 대화형 AI 음성 품질을 원한다"

**Twilio + Vapi**를 사용하세요.

이유:
- Twilio가 소유 번호 제공
- Vapi가 더 나은 대화형 AI 통화 품질과 더 많은 음성/모델 유연성 제공

권장 흐름:
1. Twilio 번호 구매/저장
2. Vapi로 가져오기
3. 반환된 `VAPI_PHONE_NUMBER_ID` 저장
4. `ai-call --provider vapi` 사용

### 4) "사용자 지정 사전 녹음 음성 메시지로 전화하고 싶다"

공개 오디오 URL을 사용하는 **직접 Twilio 전화**를 사용하세요.

이유:
- 사용자 지정 MP3를 재생하는 가장 쉬운 방법
- Hermes `text_to_speech`와 공개 파일 호스트 또는 터널을 함께 사용하기 좋음

## 파일 및 영구 상태

이 스킬은 두 곳에 전화 상태를 저장합니다.

### `${HERMES_HOME:-~/.hermes}/.env`

다음과 같은 장기 공급자 자격 증명과 소유 번호 ID에 사용됩니다.
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_PHONE_NUMBER_SID`
- `BLAND_API_KEY`
- `VAPI_API_KEY`
- `VAPI_PHONE_NUMBER_ID`
- `PHONE_PROVIDER` (AI 통화 공급자: bland 또는 vapi)

### `~/.hermes/telephony_state.json`

세션 간 유지해야 하는 스킬 전용 상태에 사용됩니다. 예를 들면 다음과 같습니다.
- 기억된 기본 Twilio 번호/ SID
- 기억된 Vapi 전화번호 ID
- 받은 편지함 폴링 체크포인트를 위한 마지막 수신 메시지 SID/날짜

따라서 다음이 가능합니다.
- 다음에 스킬을 로드할 때 `diagnose`로 이미 구성된 번호를 확인할 수 있습니다.
- `twilio-inbox --since-last --mark-seen`으로 이전 체크포인트부터 계속 확인할 수 있습니다.

## 도우미 스크립트 찾기

이 스킬을 설치한 후 다음과 같이 스크립트를 찾으세요.

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/telephony/scripts/telephony.py' -print -quit)"
```

`SCRIPT`가 비어 있으면 아직 스킬이 설치되지 않은 것입니다.

## 설치

공식 선택적 스킬이므로 Skills Hub에서 설치하세요.

```bash
hermes skills search telephony
hermes skills install official/productivity/telephony
```

## 공급자 설정

### Twilio — 소유 번호, SMS/MMS, 직접 통화, 수신 SMS 폴링

다음에서 가입하세요.
- https://www.twilio.com/try-twilio

자격 증명을 Hermes에 저장합니다.

```bash
python3 "$SCRIPT" save-twilio ACXXXXXXXXXXXXXXXXXXXXXXXXXXXX your_auth_token_here
```

사용 가능한 번호를 검색합니다.

```bash
python3 "$SCRIPT" twilio-search --country US --area-code 702 --limit 5
```

번호를 구매하고 기억합니다.

```bash
python3 "$SCRIPT" twilio-buy "+17025551234" --save-env
```

소유한 번호를 나열합니다.

```bash
python3 "$SCRIPT" twilio-owned
```

나중에 그중 하나를 기본값으로 설정합니다.

```bash
python3 "$SCRIPT" twilio-set-default "+17025551234" --save-env
# or
python3 "$SCRIPT" twilio-set-default PNXXXXXXXXXXXXXXXXXXXXXXXXXXXX --save-env
```

### Bland.ai — 가장 쉬운 발신 AI 전화

다음에서 가입하세요.
- https://app.bland.ai

구성을 저장합니다.

```bash
python3 "$SCRIPT" save-bland your_bland_api_key --voice mason
```

### Vapi — 더 나은 대화형 음성 품질

다음에서 가입하세요.
- https://dashboard.vapi.ai

먼저 API 키를 저장합니다.

```bash
python3 "$SCRIPT" save-vapi your_vapi_api_key
```

소유한 Twilio 번호를 Vapi로 가져오고 반환된 전화번호 ID를 저장합니다.

```bash
python3 "$SCRIPT" vapi-import-twilio --save-env
```

Vapi 전화번호 ID를 이미 알고 있다면 직접 저장합니다.

```bash
python3 "$SCRIPT" save-vapi your_vapi_api_key --phone-number-id vapi_phone_number_id_here
```

## 현재 상태 진단

언제든 스킬이 이미 알고 있는 내용을 확인할 수 있습니다.

```bash
python3 "$SCRIPT" diagnose
```

나중 세션에서 작업을 재개할 때 먼저 사용하세요.

## 일반적인 작업 흐름

### A. 에이전트 번호를 구매하고 나중에도 계속 사용하기

1. Twilio 자격 증명을 저장합니다.
```bash
python3 "$SCRIPT" save-twilio AC... auth_token_here
```

2. 번호를 검색합니다.
```bash
python3 "$SCRIPT" twilio-search --country US --area-code 702 --limit 10
```

3. 번호를 구매하고 `${HERMES_HOME:-~/.hermes}/.env` 및 상태에 저장합니다.
```bash
python3 "$SCRIPT" twilio-buy "+17025551234" --save-env
```

4. 다음 세션에서 실행합니다.
```bash
python3 "$SCRIPT" diagnose
```
기억된 기본 번호와 받은 편지함 체크포인트 상태가 표시됩니다.

### B. 에이전트 번호로 문자 보내기

```bash
python3 "$SCRIPT" twilio-send-sms "+15551230000" "Your deployment completed successfully."
```

미디어 포함:

```bash
python3 "$SCRIPT" twilio-send-sms "+15551230000" "Here is the chart." --media-url "https://example.com/chart.png"
```

### C. 웹훅 서버 없이 나중에 수신 문자 확인하기

기본 Twilio 번호의 받은 편지함을 폴링합니다.

```bash
python3 "$SCRIPT" twilio-inbox --limit 20
```

마지막 체크포인트 이후 도착한 메시지만 표시하고, 읽기를 마치면 체크포인트를 갱신합니다.

```bash
python3 "$SCRIPT" twilio-inbox --since-last --mark-seen
```

이는 "다음에 스킬이 로드될 때 번호가 받는 메시지에 어떻게 접근하나요?"에 대한 핵심 답입니다.

### D. 내장 TTS로 직접 Twilio 전화 걸기

```bash
python3 "$SCRIPT" twilio-call "+15551230000" --message "Hello! This is Hermes calling with your status update." --voice Polly.Joanna
```

### E. 사전 녹음/사용자 지정 음성 메시지로 전화 걸기

이것은 Hermes의 기존 `text_to_speech` 지원을 재사용하는 주요 경로입니다.

다음과 같은 경우 사용하세요.
- Twilio `<Say>` 대신 Hermes가 구성한 TTS 음성으로 통화하고 싶은 경우
- 단방향 음성 전달(브리핑, 알림, 농담, 리마인더, 상태 업데이트)이 필요한 경우
- 실시간 대화형 전화가 필요하지 않은 경우

오디오를 별도로 생성하거나 호스팅한 다음:

```bash
python3 "$SCRIPT" twilio-call "+155****0000" --audio-url "https://example.com/briefing.mp3"
```

권장 Hermes TTS -> Twilio Play 흐름:

1. Hermes `text_to_speech`로 오디오를 생성합니다.
2. 생성된 MP3를 공개적으로 접근할 수 있게 만듭니다.
3. `--audio-url`로 Twilio 전화를 겁니다.

에이전트 흐름 예시:
- Hermes에 `text_to_speech`로 메시지 오디오를 만들도록 요청
- 필요한 경우 임시 정적 호스트/터널 또는 객체 스토리지 URL로 파일 공개
- `twilio-call --audio-url ...`을 사용해 전화로 전달

MP3 호스팅에 적합한 방법:
- 임시 공개 객체/스토리지 URL
- 로컬 정적 파일 서버로 연결되는 단기 터널
- 전화 공급자가 직접 가져올 수 있는 기존 HTTPS URL

중요 참고:
- Hermes TTS는 사전 녹음 발신 메시지에 적합합니다.
- Bland/Vapi는 실시간 전화 오디오 스택을 자체적으로 처리하므로 **실시간 대화형 AI 전화**에 더 적합합니다.
- 여기서는 Hermes STT/TTS만으로 양방향 전화 대화 엔진을 만들지 않습니다. 이 스킬이 도입하려는 범위를 훨씬 넘어서는 스트리밍/웹훅 통합이 필요하기 때문입니다.

### F. Twilio 직접 통화로 전화 트리/IVR 탐색하기

통화가 연결된 후 숫자를 눌러야 한다면 `--send-digits`를 사용하세요.
Twilio는 `w`를 짧은 대기 시간으로 해석합니다.

```bash
python3 "$SCRIPT" twilio-call "+18005551234" --message "Connecting to billing now." --send-digits "ww1w2w3"
```

이는 사람에게 연결하거나 짧은 상태 메시지를 전달하기 전에 특정 메뉴 분기로 이동할 때 유용합니다.

### G. Bland.ai로 발신 AI 전화 걸기

```bash
python3 "$SCRIPT" ai-call "+15551230000" "Call the dental office, ask for a cleaning appointment on Tuesday afternoon, and if they do not have Tuesday availability, ask for Wednesday or Thursday instead." --provider bland --voice mason --max-duration 3
```

상태 확인:

```bash
python3 "$SCRIPT" ai-status <call_id> --provider bland
```

완료 후 Bland 분석 질문을 합니다.

```bash
python3 "$SCRIPT" ai-status <call_id> --provider bland --analyze "Was the appointment confirmed?,What date and time?,Any special instructions?"
```

### H. 소유한 번호의 Vapi로 발신 AI 전화 걸기

1. Twilio 번호를 Vapi로 가져옵니다.
```bash
python3 "$SCRIPT" vapi-import-twilio --save-env
```

2. 전화를 겁니다.
```bash
python3 "$SCRIPT" ai-call "+15551230000" "You are calling to make a dinner reservation for two at 7:30 PM. If that is unavailable, ask for the nearest time between 6:30 and 8:30 PM." --provider vapi --max-duration 4
```

3. 결과를 확인합니다.
```bash
python3 "$SCRIPT" ai-status <call_id> --provider vapi
```

## 권장 에이전트 절차

사용자가 전화나 문자를 요청하면:

1. 의사 결정 트리를 통해 요청에 맞는 경로를 결정합니다.
2. 구성 상태가 불분명하면 `diagnose`를 실행합니다.
3. 작업에 필요한 전체 세부 정보를 수집합니다.
4. 전화를 걸거나 문자를 보내기 전에 사용자에게 확인을 받습니다.
5. 올바른 명령을 사용합니다.
6. 필요한 경우 결과를 폴링합니다.
7. 타인의 전화번호를 Hermes 메모리에 저장하지 않고 결과를 요약합니다.

## 이 스킬이 여전히 지원하지 않는 것

- 실시간 수신 전화 응답
- 에이전트 루프로 웹훅 기반 실시간 SMS 푸시
- 임의의 타사 2FA 공급자에 대한 보장된 지원

이러한 기능에는 순수 선택적 스킬보다 더 많은 인프라가 필요합니다.

## 주의 사항

- Twilio 체험 계정과 지역별 규정에 따라 전화/문자를 보낼 수 있는 대상이 제한될 수 있습니다.
- 일부 서비스는 2FA에 VoIP 번호를 거부합니다.
- `twilio-inbox`는 REST API를 폴링하며 즉시 푸시로 전달하지 않습니다.
- Vapi 발신 전화도 유효한 가져온 번호가 있어야 합니다.
- Bland가 가장 쉽지만 항상 가장 자연스러운 음성을 제공하는 것은 아닙니다.
- 임의의 타사 전화번호를 Hermes 메모리에 저장하지 마세요.

## 검증 체크리스트

설정이 끝나면 이 스킬만으로 다음을 모두 수행할 수 있어야 합니다.

1. `diagnose`가 공급자 준비 상태와 기억된 상태를 표시
2. Twilio 번호 검색 및 구매
3. 해당 번호를 `${HERMES_HOME:-~/.hermes}/.env`에 저장
4. 소유한 번호로 SMS 전송
5. 나중에 소유한 번호의 수신 문자 폴링
6. 직접 Twilio 전화 걸기
7. Bland 또는 Vapi를 통한 AI 전화 걸기

## 참조

- Twilio 전화번호: https://www.twilio.com/docs/phone-numbers/api
- Twilio 메시징: https://www.twilio.com/docs/messaging/api/message-resource
- Twilio 음성: https://www.twilio.com/docs/voice/api/call-resource
- Vapi 문서: https://docs.vapi.ai/
- Bland.ai: https://app.bland.ai/
