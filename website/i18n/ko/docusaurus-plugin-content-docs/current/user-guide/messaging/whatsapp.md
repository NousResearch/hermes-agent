---
sidebar_position: 5
title: "WhatsApp"
description: "내장 Baileys 브리지를 통해 Hermes Agent를 WhatsApp 봇으로 설정하기"
---

# WhatsApp 설정

Hermes는 **Baileys** 기반의 내장 브리지를 통해 WhatsApp에 연결합니다. 이 방식은 공식 WhatsApp Business API가 **아닌**, WhatsApp Web 세션을 에뮬레이션하는 방식입니다. Meta 개발자 계정이나 비즈니스 인증은 필요하지 않습니다.

> 안내에 따라 설정하려면 `hermes gateway setup`을 실행하고 **WhatsApp**을 선택하세요.

:::tip 두 가지 WhatsApp 통합
이 페이지에서는 **Baileys 브리지**를 다룹니다. 빠르게 설정할 수 있고 개인 계정을 사용하며 공개 URL이 필요하지 않지만, 계정 정지 위험이 있습니다.

실제 비즈니스 봇을 운영하며 안정성이 필요하다면 **[WhatsApp Business Cloud API 가이드](./whatsapp-cloud.md)**를 참조하세요. 공식 Meta 지원 경로이므로 계정 정지 위험은 없지만, Meta 비즈니스 계정과 공개 웹훅 URL이 필요합니다.

두 어댑터는 필요에 따라 서로 다른 전화번호에서 동시에 실행할 수도 있습니다.
:::

:::warning 비공식 API — 계정 정지 위험
WhatsApp은 Business API 외부에서의 서드파티 봇을 **공식적으로 지원하지 않습니다**. 서드파티 브리지를 사용하면 계정이 제한될 위험이 조금 있습니다. 위험을 최소화하려면 다음을 지키세요.
- 봇에는 **전용 전화번호를 사용하세요**(개인 번호는 사용하지 마세요).
- **대량 또는 스팸 메시지를 보내지 마세요** — 대화형으로 사용하세요.
- 먼저 메시지를 보내지 않은 사람에게 **아웃바운드 메시지를 자동으로 보내지 마세요**.
:::

:::warning WhatsApp Web 프로토콜 업데이트
WhatsApp은 Web 프로토콜을 주기적으로 업데이트하며, 이로 인해 서드파티 브리지와의 호환성이 일시적으로 깨질 수 있습니다.
이 경우 Hermes가 브리지 의존성을 업데이트합니다. WhatsApp 업데이트 후 봇이 작동하지 않으면 최신 Hermes 버전을 가져온 뒤 다시 페어링하세요.
:::

## 두 가지 모드

| 모드 | 작동 방식 | 적합한 경우 |
|------|-------------|----------|
| **별도 봇 번호**(권장) | 봇 전용 전화번호를 마련합니다. 사람들이 해당 번호로 직접 메시지를 보냅니다. | 깔끔한 UX, 여러 사용자, 낮은 계정 정지 위험 |
| **개인 셀프 채팅** | 자신의 WhatsApp을 사용합니다. 에이전트와 대화하려면 자기 자신에게 메시지를 보냅니다. | 빠른 설정, 단일 사용자, 테스트 |

---

## 사전 요구 사항

- **Node.js v18+** 및 **npm** — WhatsApp 브리지는 Node.js 프로세스로 실행됩니다.
- WhatsApp이 설치된 **휴대전화**(QR 코드 스캔용)

이전의 브라우저 기반 브리지와 달리, 현재 Baileys 기반 브리지는 로컬 Chromium 또는 Puppeteer 의존성 스택이 필요하지 않습니다.

---

## 1단계: 설정 마법사 실행

```bash
hermes whatsapp
```

마법사는 다음 작업을 수행합니다.

1. 원하는 모드(**봇** 또는 **셀프 채팅**)를 묻습니다.
2. 필요한 경우 브리지 의존성을 설치합니다.
3. 터미널에 **QR 코드**를 표시합니다.
4. QR 코드를 스캔할 때까지 기다립니다.

**QR 코드를 스캔하려면:**

1. 휴대전화에서 WhatsApp을 엽니다.
2. **설정 → 연결된 기기**로 이동합니다.
3. **기기 연결**을 탭합니다.
4. 카메라를 터미널의 QR 코드에 갖다 댑니다.

페어링이 완료되면 마법사가 연결을 확인하고 종료합니다. 세션은 자동으로 저장됩니다.

:::tip
QR 코드가 깨져 보이면 터미널 너비가 최소 60열이고 유니코드를 지원하는지 확인하세요.
다른 터미널 에뮬레이터를 사용해 볼 수도 있습니다.
:::

---

## 2단계: 두 번째 전화번호 받기(봇 모드)

봇 모드에서는 아직 WhatsApp에 등록되지 않은 전화번호가 필요합니다. 세 가지 방법이 있습니다.

| 방법 | 비용 | 참고 |
|--------|------|-------|
| **Google Voice** | 무료 | 미국에서만 사용 가능. [voice.google.com](https://voice.google.com)에서 번호를 받습니다. Google Voice 앱을 통해 SMS로 WhatsApp을 인증합니다. |
| **선불 SIM** | 1회 $5–15 | 모든 통신사에서 가능. 활성화하고 WhatsApp을 인증한 다음 SIM을 서랍에 보관해도 됩니다. 번호는 활성 상태를 유지해야 합니다(90일마다 통화). |
| **VoIP 서비스** | 무료–월 $5 | TextNow, TextFree 또는 유사 서비스. 일부 VoIP 번호는 WhatsApp에서 차단됩니다 — 처음에 작동하지 않으면 몇 가지를 시도해 보세요. |

번호를 받은 후:

1. 휴대전화에 WhatsApp을 설치합니다(또는 듀얼 SIM에서 WhatsApp Business 앱을 사용합니다).
2. 새 번호를 WhatsApp에 등록합니다.
3. `hermes whatsapp`을 실행하고 해당 WhatsApp 계정에서 QR 코드를 스캔합니다.

---

## 3단계: Hermes 설정

다음 내용을 `~/.hermes/.env` 파일에 추가합니다.

```bash
# Required
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" or "self-chat"

# Access control — pick ONE of these options:
WHATSAPP_ALLOWED_USERS=15551234567         # Comma-separated phone numbers (with country code, no +)
# WHATSAPP_ALLOWED_USERS=*                 # OR use * to allow everyone
# WHATSAPP_ALLOW_ALL_USERS=true            # OR set this flag instead (same effect as *)
```

:::tip 모두 허용 축약형
`WHATSAPP_ALLOWED_USERS=*`로 설정하면 **모든 발신자**를 허용합니다(`WHATSAPP_ALLOW_ALL_USERS=true`와 동일).
이는 [Signal 그룹 허용 목록](/reference/environment-variables)과 일관된 방식입니다.
페어링 흐름을 사용하려면 두 변수를 모두 제거하고
[DM 페어링 시스템](/user-guide/security#dm-pairing-system)을 사용하세요.
:::

`~/.hermes/config.yaml`에서 동작 설정을 선택적으로 지정할 수 있습니다.

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `unauthorized_dm_behavior: pair`는 전역 기본값입니다. 알 수 없는 DM 발신자에게 페어링 코드가 전송됩니다.
- `whatsapp.unauthorized_dm_behavior: ignore`로 설정하면 WhatsApp이 권한 없는 DM에 응답하지 않습니다. 비공개 번호에는 대개 이 설정이 더 적합합니다.

그런 다음 게이트웨이를 시작합니다.

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

게이트웨이는 저장된 세션을 사용하여 WhatsApp 브리지를 자동으로 시작합니다.

---

## 세션 유지

Baileys 브리지는 세션을 `~/.hermes/platforms/whatsapp/session`에 저장합니다. 따라서:

- **재시작 후에도 세션이 유지됩니다** — 매번 QR 코드를 다시 스캔할 필요가 없습니다.
- 세션 데이터에는 암호화 키와 기기 자격 증명이 포함됩니다.
- **이 세션 디렉터리를 공유하거나 커밋하지 마세요** — WhatsApp 계정에 대한 전체 접근 권한이 부여됩니다.

---

## 다시 페어링

세션이 끊기면(휴대전화 초기화, WhatsApp 업데이트, 수동 연결 해제 등) 게이트웨이 로그에 연결 오류가 표시됩니다. 해결하려면:

```bash
hermes whatsapp
```

새 QR 코드가 생성됩니다. 다시 스캔하면 세션이 재설정됩니다. 게이트웨이는 네트워크 순간 장애나 휴대전화의 짧은 오프라인 상태 같은 **일시적인** 연결 해제를 재연결 로직으로 자동 처리합니다.

---

## 음성 메시지

Hermes는 WhatsApp에서 음성을 지원합니다.

- **수신:** 음성 메시지(`.ogg` opus)는 설정된 STT 제공자를 사용하여 자동으로 전사됩니다. 사용 가능한 제공자는 로컬 `faster-whisper`, Groq Whisper(`GROQ_API_KEY`), OpenAI Whisper(`VOICE_TOOLS_OPENAI_KEY`)입니다.
- **발신:** TTS 응답은 MP3 오디오 파일 첨부로 전송됩니다.
- 에이전트 응답에는 기본적으로 "⚕ **Hermes Agent**"가 접두사로 붙습니다. `config.yaml`에서 이를 사용자 지정하거나 비활성화할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
whatsapp:
  reply_prefix: ""                          # Empty string disables the header
  # reply_prefix: "🤖 *My Bot*\n──────\n"  # Custom prefix (supports \n for newlines)
  send_read_receipts: false                 # Mark accepted inbound messages as read (blue ticks)
```

`send_read_receipts`가 `true`이면 어댑터는 DM/그룹/멘션 필터링을 통과한 정책상 허용되는 수신 메시지를 읽음으로 표시합니다. (허용 목록에 없는 발신자 등) 거부된 메시지는 읽음으로 표시되지 않습니다. 개인정보 보호를 위해 기본적으로 비활성화되어 있습니다. 이 설정을 변경하면 다음 연결 시 브리지 서브프로세스가 자동으로 재시작됩니다.

---

## 메시지 형식 및 전달

WhatsApp은 **스트리밍(점진적) 응답**을 지원합니다 — 봇이 AI가 텍스트를 생성하는 동안 실시간으로 메시지를 수정하며, Discord와 Telegram과 동일합니다. 내부적으로 WhatsApp은 전달 기능 기준 TIER_MEDIUM 플랫폼으로 분류됩니다.

### 청크 분할

긴 응답은 청크당 **4,096자**로 자동 분할됩니다(WhatsApp의 실용적인 표시 한도). 별도로 설정할 필요가 없습니다 — 게이트웨이가 분할하고 청크를 순서대로 전송합니다.

### WhatsApp 호환 Markdown

AI 응답의 표준 Markdown은 WhatsApp의 기본 서식으로 자동 변환됩니다.

| Markdown | WhatsApp | 표시 결과 |
|----------|----------|------------|
| `**bold**` | `*bold*` | **굵은 글씨** |
| `~~strikethrough~~` | `~strikethrough~` | ~~취소선~~ |
| `# Heading` | `*Heading*` | 굵은 텍스트(기본 제목 기능 없음) |
| `[link text](url)` | `link text (url)` | 인라인 URL |

WhatsApp이 트리플 백틱 형식을 기본 지원하므로 코드 블록과 인라인 코드는 그대로 유지됩니다.

### 도구 진행 상황

에이전트가 도구(웹 검색, 파일 작업 등)를 호출하면 WhatsApp은 어떤 도구가 실행 중인지 보여 주는 실시간 진행 표시기를 표시합니다. 기본적으로 활성화되어 있으므로 별도의 설정이 필요하지 않습니다.

### 기본 투표, 투표로 표시되는 명확화 질문 및 위치

Baileys 브리지 어댑터(봇 모드)는 다음과 같은 WhatsApp 기본 메시지 유형을 지원합니다.

- **투표** — 에이전트는 브리지의 `/send-poll` 엔드포인트를 통해 기본 WhatsApp 투표(질문 + 선택지)를 보낼 수 있습니다. 투표 결과는 대화로 다시 전달됩니다.
- **투표로 표시되는 명확화 질문** — 에이전트가 객관식 명확화 질문을 하면 기본 단일 선택 투표로 렌더링됩니다. 선택지를 탭하면 질문에 답합니다. 투표 전송에 실패하면 어댑터가 일반 텍스트 질문으로 대체합니다. 승인 요청은 **절대** 투표로 매핑되지 않습니다 — 투표는 실제 객관식 명확화 질문에만 사용됩니다.
- **위치 핀** — 에이전트는 `/send-location`을 통해 기본 위치 핀(위도/경도, 선택적 이름/주소)을 보낼 수 있으며, 수신된 공유 위치(실시간 위치 포함)는 위치 메시지로 에이전트에 전달됩니다.

이 기능은 봇(Baileys) 모드에서 별도 설정 없이 바로 작동합니다.

### 메시지 일괄 처리(디바운스)

WhatsApp은 각 메시지를 개별적으로 전달하므로, 빠르게 연속된 메시지(전달된 메시지 묶음, 붙여넣기로 분할된 내용, 여러 줄 텍스트)는 그렇지 않으면 조각마다 별도의 에이전트 호출을 발생시킵니다 — 토큰을 낭비하고 서로 이어지지 않는 답변을 여러 개 만들게 됩니다. 어댑터는 같은 채팅에서 연속으로 들어온 텍스트 메시지를 버퍼링하고 짧은 유휴 시간이 지난 후 하나의 결합된 요청으로 전달합니다(기본 **5초**, 매우 긴 조각에는 **10초**로 연장). `config.yaml`에서 조정할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    whatsapp:
      extra:
        text_batch_delay_seconds: 5.0         # quiet period before flushing a batch
        text_batch_split_delay_seconds: 10.0  # extended delay near the split threshold
```

`text_batch_delay_seconds: 0`으로 설정하면 각 메시지를 즉시 전달합니다(일괄 처리를 비활성화).

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| **QR 코드가 스캔되지 않음** | 터미널 너비가 충분한지(60열 이상) 확인합니다. 다른 터미널을 사용해 봅니다. 올바른 WhatsApp 계정(개인 계정이 아닌 봇 번호)에서 스캔하고 있는지 확인합니다. |
| **QR 코드가 만료됨** | QR 코드는 약 20초마다 갱신됩니다. 시간 초과되면 `hermes whatsapp`을 다시 시작합니다. |
| **세션이 유지되지 않음** | `~/.hermes/platforms/whatsapp/session`이 존재하고 쓰기 가능한지 확인합니다. 컨테이너 환경이라면 영구 볼륨으로 마운트합니다. |
| **예기치 않게 로그아웃됨** | WhatsApp은 장기간 비활성 상태인 기기의 연결을 해제합니다. 휴대전화를 켜고 네트워크에 연결한 다음, 필요한 경우 `hermes whatsapp`으로 다시 페어링합니다. |
| **브리지가 충돌하거나 재연결을 반복함** | 게이트웨이를 다시 시작하고 Hermes를 업데이트합니다. WhatsApp 프로토콜 변경으로 세션이 무효화되었다면 다시 페어링합니다. |
| **WhatsApp 업데이트 후 봇이 작동하지 않음** | 최신 브리지 버전을 받도록 Hermes를 업데이트한 다음 다시 페어링합니다. |
| **macOS: 터미널에서는 node가 작동하지만 "Node.js not installed"가 표시됨** | launchd 서비스는 셸의 PATH를 상속하지 않습니다. `hermes gateway install`을 실행하여 현재 PATH를 plist에 다시 저장한 다음 `hermes gateway start`를 실행합니다. 자세한 내용은 [게이트웨이 서비스 문서](./index.md#macos-launchd)를 참조하세요. |
| **메시지가 수신되지 않음** | `WHATSAPP_ALLOWED_USERS`에 발신자 번호(국가 코드 포함, `+` 또는 공백 제외)가 포함되어 있는지 확인하거나 `*`로 설정하여 모두 허용합니다. `.env`에서 `WHATSAPP_DEBUG=true`를 설정하고 게이트웨이를 다시 시작하면 `bridge.log`에서 원시 메시지 이벤트를 확인할 수 있습니다. |
| **봇이 낯선 사람에게 페어링 코드를 답장함** | 권한 없는 DM을 조용히 무시하려면 `~/.hermes/config.yaml`에서 `whatsapp.unauthorized_dm_behavior: ignore`를 설정합니다. |

---

## 보안

:::warning
실행 전에 **접근 제어를 설정하세요**. 특정 전화번호(국가 코드 포함, `+` 제외)를 사용하여 `WHATSAPP_ALLOWED_USERS`를 설정하거나, `*`를 사용해 모두 허용하거나, `WHATSAPP_ALLOW_ALL_USERS=true`를 설정합니다. 이 중 어느 것도 설정하지 않으면 안전 조치로 게이트웨이가 **모든 수신 메시지를 거부합니다**.
:::

기본적으로 권한 없는 DM에는 여전히 페어링 코드가 답장으로 전송됩니다. 비공개 WhatsApp 번호가 낯선 사람에게 완전히 응답하지 않도록 하려면 다음을 설정하세요.

```yaml
whatsapp:
  unauthorized_dm_behavior: ignore
```

- `~/.hermes/platforms/whatsapp/session` 디렉터리에는 전체 세션 자격 증명이 들어 있으므로 비밀번호처럼 보호하세요.
- 파일 권한을 설정합니다: `chmod 700 ~/.hermes/platforms/whatsapp/session`
- 개인 계정과 위험을 분리하려면 봇에 **전용 전화번호**를 사용하세요.
- 계정이 침해되었다고 의심되면 WhatsApp → 설정 → 연결된 기기에서 기기 연결을 해제하세요.
- 로그의 전화번호는 일부가 마스킹되지만, 로그 보존 정책을 검토하세요.
