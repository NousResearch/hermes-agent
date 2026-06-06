---
sidebar_position: 5
title: "WhatsApp"
description: "내장된 Baileys 브리지를 통해 Hermes Agent를 WhatsApp 봇으로 설정하기"
---

# WhatsApp 설정

Hermes는 **Baileys** 기반의 내장 브리지를 통해 WhatsApp에 연결됩니다. 이는 공식 WhatsApp Business API를 통하지 **않고** WhatsApp Web 세션을 에뮬레이션하여 작동합니다. Meta 개발자 계정이나 비즈니스 인증이 필요하지 않습니다.

> `hermes gateway setup`을 실행하고 안내에 따른 설정을 위해 **WhatsApp**을 선택하세요.

:::warning 비공식 API — 밴(Ban) 위험
WhatsApp은 Business API 외부의 서드파티 봇을 공식적으로 지원하지 **않습니다**. 서드파티 브리지를 사용하면 계정 제한의 약간의 위험이 따릅니다. 위험을 최소화하려면:
- 봇을 위해 **전용 전화번호를 사용**하세요 (개인 번호 사용 금지)
- **대량/스팸 메시지를 보내지 마세요** — 대화형으로만 사용하세요
- 먼저 메시지를 보내지 않은 사람에게 **아웃바운드 메시징을 자동화하지 마세요**
:::

:::warning WhatsApp Web 프로토콜 업데이트
WhatsApp은 주기적으로 Web 프로토콜을 업데이트하며, 이는 서드파티 브리지와의 호환성을 일시적으로 깨뜨릴 수 있습니다. 이런 경우 Hermes는 브리지 의존성을 업데이트할 것입니다. WhatsApp 업데이트 후 봇 작동이 멈추면 최신 Hermes 버전을 가져와 다시 페어링하세요.
:::

## 두 가지 모드

| 모드 | 작동 방식 | 추천 대상 |
|------|-------------|----------|
| **별도 봇 번호** (권장) | 봇 전용 전화번호를 할당합니다. 사람들이 그 번호로 직접 메시지를 보냅니다. | 깔끔한 UX, 다수 사용자, 낮은 밴 위험 |
| **개인 나와의 채팅** | 본인의 WhatsApp을 사용합니다. 에이전트와 대화하기 위해 자신에게 메시지를 보냅니다. | 빠른 설정, 단일 사용자, 테스트용 |

---

## 전제 조건

- **Node.js v18+** 및 **npm** — WhatsApp 브리지는 Node.js 프로세스로 실행됩니다.
- (QR 코드를 스캔하기 위해) **WhatsApp이 설치된 휴대폰**

오래된 브라우저 구동 브리지와 달리, 현재의 Baileys 기반 브리지는 로컬 Chromium이나 Puppeteer 의존성 스택을 요구하지 **않습니다**.

---

## 1단계: 설정 마법사 실행

```bash
hermes whatsapp
```

마법사는 다음을 수행합니다:

1. 원하는 모드(**bot** 또는 **self-chat**) 질문
2. 필요한 경우 브리지 의존성 설치
3. 터미널에 **QR 코드** 표시
4. 사용자가 스캔할 때까지 대기

**QR 코드를 스캔하려면:**

1. 휴대폰에서 WhatsApp 열기
2. **설정(Settings) → 연결된 기기(Linked Devices)**로 이동
3. **기기 연결(Link a Device)** 탭하기
4. 카메라로 터미널의 QR 코드 가리키기

페어링되면 마법사가 연결을 확인하고 종료됩니다. 세션은 자동으로 저장됩니다.

:::tip
QR 코드가 깨져 보인다면 터미널 너비가 60열 이상이고 유니코드를 지원하는지 확인하세요. 다른 터미널 에뮬레이터를 사용해 볼 수도 있습니다.
:::

---

## 2단계: 두 번째 전화번호 구하기 (봇 모드)

봇 모드의 경우 WhatsApp에 등록되지 않은 전화번호가 필요합니다. 세 가지 옵션이 있습니다:

| 옵션 | 비용 | 참고 |
|--------|------|-------|
| **Google Voice** | 무료 | 미국 전용. [voice.google.com](https://voice.google.com)에서 번호를 받으세요. Google Voice 앱을 통해 SMS로 WhatsApp을 인증하세요. |
| **선불 SIM** | 1회 $5–15 | 아무 통신사나 무관합니다. 개통하고 WhatsApp을 인증한 다음 SIM은 서랍에 넣어두셔도 됩니다. 번호는 활성 상태를 유지해야 합니다 (90일마다 통화 발생 필요). |
| **VoIP 서비스** | 무료–월 $5 | TextNow, TextFree 등. 일부 VoIP 번호는 WhatsApp에서 차단됩니다 — 첫 번째 번호가 안 되면 몇 가지 더 시도해 보세요. |

번호를 얻은 후:

1. 휴대폰에 WhatsApp을 설치합니다 (또는 듀얼 SIM 기기에서 WhatsApp Business 앱 사용).
2. 새 번호를 WhatsApp에 등록합니다.
3. `hermes whatsapp`을 실행하고 해당 WhatsApp 계정에서 QR 코드를 스캔합니다.

---

## 3단계: Hermes 구성

`~/.hermes/.env` 파일에 다음을 추가하세요:

```bash
# 필수
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" 또는 "self-chat"

# 접근 제어 — 다음 옵션 중 하나만 선택:
WHATSAPP_ALLOWED_USERS=15551234567         # 쉼표로 구분된 전화번호 (국가 코드 포함, + 제외)
# WHATSAPP_ALLOWED_USERS=*                 # 또는 모두를 허용하려면 * 사용
# WHATSAPP_ALLOW_ALL_USERS=true            # 또는 이 플래그 설정 (* 와 동일한 효과)
```

:::tip 전체 허용(Allow-all) 단축표현
`WHATSAPP_ALLOWED_USERS=*`를 설정하면 **모든** 발신자를 허용합니다 (`WHATSAPP_ALLOW_ALL_USERS=true`와 동일). 이는 [Signal 그룹 허용 목록](/reference/environment-variables) 설정과 일관성을 가집니다. 페어링 흐름을 사용하려면 두 변수를 모두 제거하고 [DM 페어링 시스템](/user-guide/security#dm-pairing-system)에 의존하세요.
:::

`~/.hermes/config.yaml`의 선택적 동작 설정:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `unauthorized_dm_behavior: pair`는 전역 기본값입니다. 알 수 없는 DM 발신자는 페어링 코드를 받습니다.
- `whatsapp.unauthorized_dm_behavior: ignore`는 승인되지 않은 DM에 대해 WhatsApp이 침묵을 유지하도록 하며, 이는 비공개 번호의 경우 대체로 더 나은 선택입니다.

그 후 게이트웨이를 시작합니다:

```bash
hermes gateway              # 포그라운드
hermes gateway install      # 사용자 서비스로 설치
sudo hermes gateway install --system   # Linux 전용: 부팅 시 시스템 서비스
```

게이트웨이는 저장된 세션을 사용하여 자동으로 WhatsApp 브리지를 시작합니다.

---

## 세션 영속성

Baileys 브리지는 `~/.hermes/platforms/whatsapp/session` 아래에 세션을 저장합니다. 이는 다음을 의미합니다:

- **세션이 재시작 후에도 유지됩니다** — 매번 QR 코드를 다시 스캔할 필요가 없습니다.
- 세션 데이터에는 암호화 키와 기기 자격 증명이 포함됩니다.
- **이 세션 디렉터리를 공유하거나 커밋하지 마세요** — WhatsApp 계정에 대한 전체 액세스 권한을 부여합니다.

---

## 다시 페어링하기

세션이 끊어지면 (전화기 초기화, WhatsApp 업데이트, 수동 연결 해제 등) 게이트웨이 로그에 연결 오류가 표시됩니다. 수정하려면:

```bash
hermes whatsapp
```

그러면 새 QR 코드가 생성됩니다. 다시 스캔하면 세션이 재설정됩니다. 게이트웨이는 **일시적인** 연결 끊김 (네트워크 오류, 짧은 오프라인 전환)을 재연결 로직을 통해 자동으로 처리합니다.

---

## 음성 메시지

Hermes는 WhatsApp에서 음성을 지원합니다:

- **수신:** 음성 메시지(`.ogg` opus)는 구성된 STT 제공자를 사용하여 자동으로 텍스트로 변환됩니다: 로컬 `faster-whisper`, Groq Whisper (`GROQ_API_KEY`), 또는 OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`).
- **발신:** TTS 응답은 MP3 오디오 파일 첨부로 전송됩니다.
- 에이전트 응답 앞에는 기본적으로 "⚕ **Hermes Agent**" 접두사가 붙습니다. `config.yaml`에서 이를 사용자 지정하거나 비활성화할 수 있습니다:

```yaml
# ~/.hermes/config.yaml
whatsapp:
  reply_prefix: ""                          # 빈 문자열은 헤더를 비활성화합니다.
  # reply_prefix: "🤖 *My Bot*\n──────\n"  # 사용자 지정 접두사 (\n으로 줄바꿈 지원)
```

---

## 메시지 포매팅 및 전송

WhatsApp은 **스트리밍(점진적) 응답**을 지원합니다 — Discord나 Telegram처럼 AI가 텍스트를 생성함에 따라 봇이 실시간으로 메시지를 편집합니다. 내부적으로 WhatsApp은 전달 기능 면에서 TIER_MEDIUM 플랫폼으로 분류됩니다.

### 청킹 (Chunking)

긴 응답은 청크당 **4,096자**(WhatsApp의 실질적 표시 한도)씩 여러 메시지로 자동 분할됩니다. 설정할 필요가 없습니다 — 게이트웨이가 분할을 처리하고 순차적으로 청크를 전송합니다.

### WhatsApp 호환 마크다운

AI 응답의 표준 마크다운은 WhatsApp의 네이티브 포맷으로 자동 변환됩니다:

| 마크다운 | WhatsApp | 렌더링 형태 |
|----------|----------|------------|
| `**bold**` | `*bold*` | **bold** |
| `~~strikethrough~~` | `~strikethrough~` | ~~strikethrough~~ |
| `# Heading` | `*Heading*` | 굵은 텍스트 (네이티브 제목 없음) |
| `[link text](url)` | `link text (url)` | 인라인 URL |

WhatsApp은 삼중 백틱(triple-backtick) 형식을 자체 지원하므로 코드 블록과 인라인 코드는 그대로 유지됩니다.

### 도구 진행 상황

에이전트가 도구(웹 검색, 파일 작업 등)를 호출할 때, WhatsApp은 실행 중인 도구를 보여주는 실시간 진행 표시기를 표시합니다. 이는 기본적으로 활성화되어 있으며 구성이 필요하지 않습니다.

### 메시지 일괄 처리 (Debounce)

WhatsApp은 각 메시지를 개별적으로 전달하므로 메시지가 쏟아지면 (전달된 메시지 묶음, 붙여넣기로 인한 분할, 여러 줄의 텍스트) 각 조각마다 별도의 에이전트 호출을 유발하게 됩니다 — 이는 토큰을 낭비하고 조각난 여러 개의 답변을 생성합니다. 어댑터는 동일한 채팅에서 연속되는 텍스트 메시지들을 버퍼링한 후, 짧은 대기 시간(기본 **5초**, 매우 긴 메시지 조각의 경우 **10초**로 연장됨) 이후에 하나로 합쳐서 전송합니다. `config.yaml`에서 조정하세요:

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    whatsapp:
      extra:
        text_batch_delay_seconds: 5.0         # 일괄 처리를 비우기 전 대기 시간
        text_batch_split_delay_seconds: 10.0  # 분할 임계치 근처에서의 연장된 대기 시간
```

각 메시지를 즉시 디스패치하려면 `text_batch_delay_seconds: 0`으로 설정하세요 (일괄 처리 비활성화).

---

## 문제 해결

| 문제 | 해결책 |
|---------|----------|
| **QR 코드가 스캔되지 않음** | 터미널이 충분히 넓은지 확인하세요 (60열 이상). 다른 터미널을 시도해 보세요. 올바른 WhatsApp 계정 (개인 번호가 아닌 봇 번호)에서 스캔하고 있는지 확인하세요. |
| **QR 코드 만료됨** | QR 코드는 약 20초마다 새로 고쳐집니다. 시간 초과된 경우 `hermes whatsapp`을 다시 시작하세요. |
| **세션이 유지되지 않음** | `~/.hermes/platforms/whatsapp/session`이 존재하고 쓰기 가능한지 확인하세요. 컨테이너 환경인 경우 영구 볼륨으로 마운트하세요. |
| **예기치 않게 로그아웃됨** | WhatsApp은 장시간 활동이 없으면 기기 연결을 해제합니다. 휴대폰을 켜고 네트워크에 연결된 상태로 유지한 다음 필요한 경우 `hermes whatsapp`으로 다시 페어링하세요. |
| **브리지 충돌 또는 재연결 반복** | 게이트웨이를 다시 시작하고, Hermes를 업데이트하고, WhatsApp 프로토콜 변경으로 인해 세션이 무효화된 경우 다시 페어링하세요. |
| **WhatsApp 업데이트 후 봇 작동 멈춤** | 최신 브리지 버전을 얻기 위해 Hermes를 업데이트한 다음 다시 페어링하세요. |
| **macOS: "Node.js not installed" 그러나 터미널에서 node 작동함** | launchd 서비스는 쉘 PATH를 상속하지 않습니다. `hermes gateway install`을 실행하여 현재 PATH를 plist로 다시 스냅샷한 다음 `hermes gateway start`를 실행하세요. 자세한 내용은 [게이트웨이 서비스 문서](./index.md#macos-launchd)를 참조하세요. |
| **메시지가 수신되지 않음** | `WHATSAPP_ALLOWED_USERS`에 발신자의 번호(국가 코드 포함, `+`나 공백 제외)가 포함되어 있는지 확인하거나, 모두를 허용하려면 `*`로 설정하세요. `.env`에 `WHATSAPP_DEBUG=true`를 설정하고 게이트웨이를 다시 시작하여 `bridge.log`에서 원시 메시지 이벤트를 확인하세요. |
| **봇이 낯선 사람에게 페어링 코드로 응답함** | 승인되지 않은 DM이 조용히 무시되기를 원한다면 `~/.hermes/config.yaml`에 `whatsapp.unauthorized_dm_behavior: ignore`를 설정하세요. |

---

## 보안

:::warning
라이브로 전환하기 전에 **접근 제어를 구성**하세요. 특정 전화번호(국가 코드 포함, `+` 제외)로 `WHATSAPP_ALLOWED_USERS`를 설정하거나, 모두를 허용하려면 `*`를 사용하거나, `WHATSAPP_ALLOW_ALL_USERS=true`를 설정하세요. 이 중 아무것도 없으면 게이트웨이는 안전 조치로 **들어오는 모든 메시지를 거부**합니다.
:::

기본적으로 승인되지 않은 DM은 여전히 페어링 코드 응답을 받습니다. 개인 WhatsApp 번호가 낯선 사람에게 완전히 침묵을 유지하도록 하려면 다음과 같이 설정하세요:

```yaml
whatsapp:
  unauthorized_dm_behavior: ignore
```

- `~/.hermes/platforms/whatsapp/session` 디렉터리에는 전체 세션 자격 증명이 포함되어 있습니다 — 비밀번호처럼 보호하세요.
- 파일 권한 설정: `chmod 700 ~/.hermes/platforms/whatsapp/session`
- 개인 계정으로부터의 위험을 분리하기 위해 봇 전용 **전용 전화번호**를 사용하세요.
- 침해가 의심되는 경우 WhatsApp → 설정 → 연결된 기기에서 기기 연결을 해제하세요.
- 로그의 전화번호는 부분적으로 교정되지만, 로그 보존 정책을 검토하세요.
