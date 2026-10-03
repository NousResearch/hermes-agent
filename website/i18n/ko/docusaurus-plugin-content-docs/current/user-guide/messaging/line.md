---
sidebar_position: 17
title: "LINE"
description: "Hermes Agent를 LINE Messaging API 봇으로 설정"
---

# LINE 설정

공식 [LINE](https://line.me/) Messaging API를 통해 Hermes Agent를 LINE 봇으로 실행할 수 있습니다. 어댑터는 `plugins/platforms/line/` 아래에 번들된 플랫폼 플러그인으로 제공되므로 코어를 수정할 필요 없이 다른 플랫폼과 마찬가지로 활성화하면 됩니다.

LINE은 일본, 대만, 태국에서 가장 널리 사용되는 메시징 앱입니다. 사용자가 해당 지역에 있다면 이 방식으로 봇에 접근할 수 있습니다.

> `hermes gateway setup`을 실행하고 안내에 따라 **LINE**을 선택하세요.

## 봇의 응답 방식

| 컨텍스트 | 동작 |
|---------|----------|
| **1:1 채팅** (`U` IDs) | 모든 메시지에 응답 |
| **그룹 채팅** (`C` IDs) | 그룹이 허용 목록에 있을 때 응답 |
| **다중 사용자 방** (`R` IDs) | 방이 허용 목록에 있을 때 응답 |

수신 텍스트, 이미지, 오디오, 동영상, 파일, 스티커, 위치를 모두 처리합니다. 발신 텍스트는 먼저 **무료 답장 토큰**(한 번만 사용 가능, 약 60초 유효)을 사용하고, 토큰이 만료되면 과금되는 Push API로 대체합니다.

---

## 1단계: LINE Messaging API 채널 만들기

1. [LINE Developers Console](https://developers.line.biz/console/)로 이동합니다.
2. Provider를 만든 다음 그 아래에 **Messaging API** 채널을 만듭니다.
3. 채널의 **Basic settings** 탭에서 **Channel secret**을 복사합니다.
4. **Messaging API** 탭에서 **Channel access token (long-lived)**까지 스크롤한 다음 **Issue**를 클릭합니다. 토큰을 복사합니다.
5. **Messaging API** 탭에서 **Auto-reply messages**와 **Greeting messages**도 비활성화하여 봇의 응답과 충돌하지 않게 합니다.

---

## 2단계: 웹훅 포트 공개

LINE은 공개 HTTPS를 통해 웹훅을 전달합니다. 기본 포트는 `8646`이며, 필요하면 `LINE_PORT`로 재정의할 수 있습니다.

```bash
# Cloudflare Tunnel (recommended for production — fixed hostname)
cloudflared tunnel --url http://localhost:8646

# ngrok (good for dev)
ngrok http 8646

# devtunnel
devtunnel create hermes-line --allow-anonymous
devtunnel port create hermes-line -p 8646 --protocol https
devtunnel host hermes-line
```

`https://...` URL을 복사합니다. 아래에서 웹훅 URL로 설정합니다. **테스트하는 동안 터널을 실행한 상태로 유지하세요.** 프로덕션에서는 고정 Cloudflare named tunnel을 설정하여 재시작해도 웹훅 URL이 바뀌지 않게 하세요.

---

## 3단계: Hermes 설정

`~/.hermes/.env`에 다음을 추가합니다.

```env
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LONG_LIVED_TOKEN
LINE_CHANNEL_SECRET=YOUR_CHANNEL_SECRET

# Allowlist — at least one of these (or LINE_ALLOW_ALL_USERS=true for dev)
LINE_ALLOWED_USERS=U1234567890abcdef...           # comma-separated U-prefixed IDs
LINE_ALLOWED_GROUPS=C1234567890abcdef...          # optional group IDs
LINE_ALLOWED_ROOMS=R1234567890abcdef...           # optional room IDs

# Required for image / audio / video sends — the public HTTPS base URL
# the tunnel resolves to.  Without it, send_image/voice/video will refuse.
LINE_PUBLIC_URL=https://my-tunnel.example.com
```

그런 다음 `~/.hermes/config.yaml`에 다음을 추가합니다.

```yaml
gateway:
  platforms:
    line:
      enabled: true
```

이것으로 충분합니다. `gateway/config.py`의 번들 플러그인 검색이 `plugins/platforms/line/`을 자동으로 찾습니다. `Platform.LINE` enum을 수정하거나 `_create_adapter`를 등록할 필요가 없습니다.

---

## 4단계: 웹훅 URL 설정

LINE 콘솔로 돌아갑니다.

1. 채널 → **Messaging API** 탭을 엽니다.
2. **Webhook settings** → **Webhook URL**에서 `https://<your-tunnel>/line/webhook`을 붙여 넣습니다. `/line/webhook` 경로에 주의하세요. 어댑터가 이 경로에서 수신합니다.
3. **Verify**를 클릭합니다. LINE이 URL을 호출하며 200 응답이 표시되어야 합니다.
4. **Use webhook**을 **On**으로 전환합니다.

---

## 5단계: 게이트웨이 실행

```bash
hermes gateway
```

에이전트 로그에 다음이 표시됩니다.

```
LINE: webhook listening on * (all interfaces, IPv4+IPv6):8646/line/webhook (public: https://my-tunnel.example.com)
```

LINE 앱에서 봇을 친구로 추가합니다(채널의 **Messaging API** 탭에서 QR을 스캔). 그런 다음 봇에 메시지를 보냅니다.

---

## LLM 응답이 느린 경우

LINE의 답장 토큰은 한 번만 사용할 수 있으며 수신 이벤트 후 약 60초가 지나면 만료됩니다. 느린 LLM은 제때 답장하지 못하므로 일반적으로 유료 Push API 호출이 필요합니다.

LLM이 `LINE_SLOW_RESPONSE_THRESHOLD`초(기본값 `45`)를 넘겨 계속 실행 중이면 어댑터는 원래 답장 토큰을 사용하여 **Template Buttons** 버블을 보냅니다.

> 🤔 아직 생각 중입니다. 준비되면 아래를 눌러 답변을 가져오세요.
>
> [ 답변 가져오기 ]

사용자는 편한 때 **답변 가져오기**를 누를 수 있습니다. 이 postback은 *새로운* 답장 토큰을 전달하며, 어댑터는 이를 사용해 캐시된 답변을 보냅니다(여전히 무료).

상태 머신은 `PENDING → READY → DELIVERED`이며, 취소된 실행에는 `ERROR`가 추가됩니다. `/stop` 이후 고아 PENDING 상태는 "완료 전에 실행이 중단되었습니다."로 해결되므로 영구 버튼이 반복 표시되지 않습니다.

postback 버튼을 비활성화하고 항상 Push로 대체하려면 다음을 설정합니다.

```env
LINE_SLOW_RESPONSE_THRESHOLD=0
```

postback 흐름이 안정적으로 작동하려면 임계값에 도달하기 전에 답장 토큰을 소비하는 대화를 억제합니다.

```yaml
# ~/.hermes/config.yaml
display:
  interim_assistant_messages: false
  platforms:
    line:
      tool_progress: off
```

---

## Cron / 알림 전달

```env
LINE_HOME_CHANNEL=Uxxxxxxxxxxxxxxxxxxxx     # default delivery target
```

`deliver: line`인 Cron 작업은 `LINE_HOME_CHANNEL`로 라우팅됩니다. 어댑터에는 독립 실행형 Push 전용 발신기가 포함되어 있으므로 Cron이 게이트웨이와 별도 프로세스에서 실행되어도 Cron 작업이 작동합니다.

---

## 환경 변수 참조

| 변수 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | 예 | — | 장기 채널 액세스 토큰 |
| `LINE_CHANNEL_SECRET` | 예 | — | 채널 시크릿(HMAC-SHA256 웹훅 검증) |
| `LINE_HOST` | 아니요 | 설정되지 않음(듀얼 스택: 모든 인터페이스, IPv4+IPv6) | 웹훅 바인딩 호스트 |
| `LINE_PORT` | 아니요 | `8646` | 웹훅 바인딩 포트 |
| `LINE_PUBLIC_URL` | 미디어 전송 시 | — | 공개 HTTPS 기본 URL. 이미지/음성/동영상 전송에 필요 |
| `LINE_ALLOWED_USERS` | 다음 중 하나 | — | 쉼표로 구분한 사용자 ID(`U` 접두사) |
| `LINE_ALLOWED_GROUPS` | 다음 중 하나 | — | 쉼표로 구분한 그룹 ID(`C` 접두사) |
| `LINE_ALLOWED_ROOMS` | 다음 중 하나 | — | 쉼표로 구분한 방 ID(`R` 접두사) |
| `LINE_ALLOW_ALL_USERS` | 개발 전용 | `false` | 허용 목록을 완전히 건너뜀 |
| `LINE_HOME_CHANNEL` | 아니요 | — | 기본 Cron / 알림 전달 대상 |
| `LINE_SLOW_RESPONSE_THRESHOLD` | 아니요 | `45` | postback 버튼이 표시되기 전의 초(`0` = 비활성화) |
| `LINE_PENDING_TEXT` | 아니요 | "🤔 Still thinking…" | postback 버튼과 함께 표시되는 버블 텍스트 |
| `LINE_BUTTON_LABEL` | 아니요 | "Get answer" | 버튼 레이블 |
| `LINE_DELIVERED_TEXT` | 아니요 | "Already replied ✅" | 이미 전달된 버튼을 다시 눌렀을 때의 답장 |
| `LINE_INTERRUPTED_TEXT` | 아니요 | "Run was interrupted before completion." | `/stop`으로 고아 버튼을 눌렀을 때의 답장 |

---

## 문제 해결

**웹훅 검증에서 "invalid signature"가 표시됩니다.** `Channel secret`을 잘못 복사했거나 터널이 요청 본문을 다시 작성한 것입니다. 먼저 `curl -i https://<tunnel>/line/webhook/health`로 확인하세요. `{"status":"ok","platform":"line"}`이 반환되어야 합니다.

**그룹에서 봇이 아무것도 받지 못합니다.** `LINE_ALLOWED_GROUPS`에 `C...` 그룹 ID가 포함되어 있는지 확인합니다. 그룹 ID를 찾으려면 테스트 메시지를 보내고 `~/.hermes/logs/gateway.log`에서 `LINE: rejecting unauthorized source`를 grep합니다. 거부된 source dict에 ID가 들어 있습니다.

**`send_image`가 "LINE_PUBLIC_URL must be set"와 함께 실패합니다.** LINE의 Messaging API는 바이너리 업로드를 허용하지 않습니다. 이미지, 오디오, 동영상은 HTTPS URL에서 접근 가능해야 합니다. `LINE_PUBLIC_URL`을 터널의 공개 호스트 이름으로 설정하면 어댑터가 `/line/media/<token>/<filename>`에서 파일을 자동으로 제공합니다.

**postback 버튼이 표시되지 않습니다.** LLM이 `LINE_SLOW_RESPONSE_THRESHOLD`보다 빠르게 응답했거나 다른 버블(tool-progress, streaming)이 먼저 답장 토큰을 소비한 것입니다. "느린 LLM 응답"의 억제 설정 블록을 참고하세요.

**"already in use by another profile"가 표시됩니다.** 같은 채널 액세스 토큰이 실행 중인 다른 Hermes 프로필에 연결되어 있습니다. 다른 게이트웨이를 중지하거나 별도의 채널을 사용하세요.

---

## 제한 사항

* **버블 및 길이 제한.** 각 LINE 텍스트 버블은 5000자로 제한됩니다. 더 긴 응답은 Reply/Push 호출 하나당 최대 5개의 버블에 약 4500자씩 자연스러운 경계를 우선하여 지능적으로 분할됩니다.
* **기본 메시지 편집 불가.** LINE에는 메시지 편집 API가 없습니다. 스트리밍 응답은 이전 메시지를 편집하지 않고 항상 새 버블을 보냅니다.
* **Markdown 렌더링 없음.** 굵게(`**`), 기울임(`*`), 코드 펜스, 제목은 리터럴 문자로 표시됩니다. 어댑터가 이를 제거한 뒤 전송하며 URL은 보존됩니다(`[label](url)`은 `label (url)`이 됨).
* **로딩 표시기는 DM 전용.** LINE은 그룹과 방에서 chat/loading API를 거부하므로 입력 중 표시기는 1:1 채팅에서만 나타납니다.
