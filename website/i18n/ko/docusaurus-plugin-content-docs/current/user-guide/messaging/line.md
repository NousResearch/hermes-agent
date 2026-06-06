---
sidebar_position: 17
title: "LINE"
description: "Hermes Agent를 LINE Messaging API 봇으로 설정하기"
---

# LINE 설정

공식 LINE Messaging API를 통해 Hermes Agent를 [LINE](https://line.me/) 봇으로 실행하세요. 어댑터는 번들 플랫폼 플러그인으로 `plugins/platforms/line/` 아래에 위치합니다. 코어 수정 없이 다른 플랫폼처럼 활성화하기만 하면 됩니다.

LINE은 일본, 대만, 태국에서 주로 사용되는 주요 메시징 앱입니다. 사용자가 해당 지역에 있다면, 이것이 사용자에게 도달하는 방법입니다.

> `hermes gateway setup`을 실행하고 **LINE**을 선택하면 안내에 따라 설정할 수 있습니다.

## 봇 응답 방식

| 컨텍스트 | 동작 |
|---------|----------|
| **1:1 채팅** (`U` ID) | 모든 메시지에 응답 |
| **그룹 채팅** (`C` ID) | 그룹이 허용 목록(allowlist)에 있을 때 응답 |
| **멀티 유저 룸** (`R` ID) | 룸이 허용 목록(allowlist)에 있을 때 응답 |

수신되는 텍스트, 이미지, 오디오, 비디오, 파일, 스티커 및 위치 정보가 모두 처리됩니다. 발신 텍스트는 먼저 **무료 응답 토큰**(1회용, 약 60초의 시간 제한)을 사용하고, 토큰이 만료되면 종량제 Push API로 대체(fallback)됩니다.

---

## 1단계: LINE Messaging API 채널 생성

1. [LINE Developers Console](https://developers.line.biz/console/)로 이동합니다.
2. Provider를 생성한 다음, 그 아래에 **Messaging API** 채널을 생성합니다.
3. 채널의 **Basic settings(기본 설정)** 탭에서 **Channel secret(채널 시크릿)**을 복사합니다.
4. **Messaging API** 탭에서 **Channel access token (long-lived)(채널 액세스 토큰 (장기))**까지 스크롤한 다음 **Issue(발급)**를 클릭합니다. 토큰을 복사합니다.
5. **Messaging API** 탭에서 봇의 응답과 충돌하지 않도록 **Auto-reply messages(자동 응답 메시지)**와 **Greeting messages(인사 메시지)**를 비활성화합니다.

---

## 2단계: 웹훅 포트 노출

LINE은 퍼블릭 HTTPS를 통해 웹훅을 전달합니다. 기본 포트는 `8646`이며, 필요한 경우 `LINE_PORT`로 재정의할 수 있습니다.

```bash
# Cloudflare Tunnel (프로덕션 환경 권장 — 고정 호스트 이름)
cloudflared tunnel --url http://localhost:8646

# ngrok (개발 환경에 적합)
ngrok http 8646

# devtunnel
devtunnel create hermes-line --allow-anonymous
devtunnel port create hermes-line -p 8646 --protocol https
devtunnel host hermes-line
```

`https://...` URL을 복사하세요 — 아래에서 웹훅 URL로 설정하게 됩니다. 테스트하는 동안 **터널을 계속 실행**해 두세요. 프로덕션 환경의 경우, 재시작 시 웹훅 URL이 변경되지 않도록 Cloudflare named tunnel을 고정으로 설정하세요.

---

## 3단계: Hermes 구성

`~/.hermes/.env` 파일에 다음을 추가합니다:

```env
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LONG_LIVED_TOKEN
LINE_CHANNEL_SECRET=YOUR_CHANNEL_SECRET

# 허용 목록 — 다음 중 최소 하나 이상 (개발 환경의 경우 LINE_ALLOW_ALL_USERS=true)
LINE_ALLOWED_USERS=U1234567890abcdef...           # 쉼표로 구분된 U 접두사 ID
LINE_ALLOWED_GROUPS=C1234567890abcdef...          # 선택적 그룹 ID
LINE_ALLOWED_ROOMS=R1234567890abcdef...           # 선택적 룸 ID

# 이미지 / 오디오 / 비디오 전송에 필요 — 터널이 확인되는 퍼블릭 HTTPS 기본 URL입니다. 
# 이 값이 없으면 send_image/voice/video가 거부됩니다.
LINE_PUBLIC_URL=https://my-tunnel.example.com
```

그런 다음 `~/.hermes/config.yaml` 파일에 다음을 설정합니다:

```yaml
gateway:
  platforms:
    line:
      enabled: true
```

이것으로 충분합니다. `gateway/config.py`의 번들 플러그인 스캔 기능이 `plugins/platforms/line/`을 자동으로 감지합니다. `Platform.LINE` 열거형 수정이나 `_create_adapter` 등록이 필요 없습니다.

---

## 4단계: 웹훅 URL 설정

다시 LINE 콘솔로 돌아갑니다:

1. 채널 열기 → **Messaging API** 탭으로 이동합니다.
2. **Webhook settings(웹훅 설정)** → **Webhook URL(웹훅 URL)**에 `https://<your-tunnel>/line/webhook`을 붙여넣습니다 (어댑터가 이 경로를 수신 대기하므로 `/line/webhook` 경로에 유의하세요).
3. **Verify(확인)**를 클릭합니다. LINE이 URL에 핑을 보내고, 200이 표시되어야 합니다.
4. **Use webhook(웹훅 사용)** 토글을 **On(켜기)**으로 변경합니다.

---

## 5단계: 게이트웨이 실행

```bash
hermes gateway
```

에이전트 로그에 다음과 같이 표시됩니다:

```
LINE: webhook listening on 0.0.0.0:8646/line/webhook (public: https://my-tunnel.example.com)
```

LINE 앱에서 봇을 친구로 추가하고(채널의 **Messaging API** 탭에 있는 QR 스캔) 메시지를 보내보세요.

---

## 느린 LLM 응답

LINE의 응답 토큰은 1회용이며 인바운드 이벤트 후 대략 60초 후에 만료됩니다. 느린 LLM은 제시간에 응답할 수 없으며, 이는 일반적으로 유료 Push API 호출을 강제합니다.

LLM이 `LINE_SLOW_RESPONSE_THRESHOLD` 초(기본값 `45`)를 초과하여 계속 실행 중일 때, 어댑터는 원래의 응답 토큰을 소비하여 **Template Buttons(템플릿 버튼)** 버블을 전송합니다:

> 🤔 생각 중입니다. 준비가 완료되면 아래 버튼을 탭하여 답변을 확인하세요.
>
> [ 답변 확인 ]

사용자가 편리할 때 **답변 확인(Get answer)**을 탭하면 — 해당 포스트백(postback)이 *새로운* 응답 토큰을 전달하고, 어댑터는 이를 사용하여 캐시된 답변을 전송합니다 (여전히 무료).

상태 머신: `PENDING → READY → DELIVERED`, 취소된 실행을 위한 `ERROR` 포함 (`/stop` 이후 남겨진 PENDING 상태는 "실행이 완료되기 전에 중단되었습니다."로 해결되어 지속적인 버튼 탭이 무한 반복되지 않도록 합니다).

포스트백 버튼을 비활성화하고 항상 Push 대체(fallback)를 사용하려면:

```env
LINE_SLOW_RESPONSE_THRESHOLD=0
```

포스트백 흐름이 안정적으로 작동하려면, 임계값 이전에 응답 토큰을 소비할 수 있는 메시지 전송을 차단하세요:

```yaml
# ~/.hermes/config.yaml
display:
  interim_assistant_messages: false
  platforms:
    line:
      tool_progress: off
```

---

## 크론(Cron) / 알림 전송

```env
LINE_HOME_CHANNEL=Uxxxxxxxxxxxxxxxxxxxx     # 기본 전송 대상
```

`deliver: line`이 있는 크론 작업은 `LINE_HOME_CHANNEL`로 라우팅됩니다. 어댑터는 Push 전용 단독 송신기를 제공하므로, 게이트웨이와 별도의 프로세스에서 크론이 실행되더라도 크론 작업이 정상 작동합니다.

---

## 환경 변수 참조

| 변수 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|
| `LINE_CHANNEL_ACCESS_TOKEN` | 예 | — | 장기 채널 액세스 토큰 |
| `LINE_CHANNEL_SECRET` | 예 | — | 채널 시크릿 (HMAC-SHA256 웹훅 검증) |
| `LINE_HOST` | 아니요 | `0.0.0.0` | 웹훅 바인드 호스트 |
| `LINE_PORT` | 아니요 | `8646` | 웹훅 바인드 포트 |
| `LINE_PUBLIC_URL` | 미디어의 경우 | — | 퍼블릭 HTTPS 기본 URL; 이미지/음성/비디오 전송에 필수 |
| `LINE_ALLOWED_USERS` | 다음 중 하나 | — | 쉼표로 구분된 사용자 ID (U 접두사) |
| `LINE_ALLOWED_GROUPS` | 다음 중 하나 | — | 쉼표로 구분된 그룹 ID (C 접두사) |
| `LINE_ALLOWED_ROOMS` | 다음 중 하나 | — | 쉼표로 구분된 룸 ID (R 접두사) |
| `LINE_ALLOW_ALL_USERS` | 개발 전용 | `false` | 허용 목록을 완전히 건너뜀 |
| `LINE_HOME_CHANNEL` | 아니요 | — | 기본 크론 / 알림 전송 대상 |
| `LINE_SLOW_RESPONSE_THRESHOLD` | 아니요 | `45` | 포스트백 버튼이 작동하기 전의 시간(초) (`0` = 비활성화) |
| `LINE_PENDING_TEXT` | 아니요 | "🤔 Still thinking…" | 포스트백 버튼과 함께 표시되는 버블 텍스트 |
| `LINE_BUTTON_LABEL` | 아니요 | "Get answer" | 버튼 라벨 |
| `LINE_DELIVERED_TEXT` | 아니요 | "Already replied ✅" | 이미 답변된 버튼을 다시 탭했을 때의 응답 |
| `LINE_INTERRUPTED_TEXT` | 아니요 | "Run was interrupted before completion." | `/stop`으로 고립된 버튼을 탭했을 때의 응답 |

---

## 문제 해결

**웹훅 확인 시 "invalid signature".** `Channel secret(채널 시크릿)`이 잘못 복사되었거나, 터널이 요청 본문을 다시 작성했습니다. 먼저 `curl -i https://<tunnel>/line/webhook/health`를 사용하여 확인하세요 — `{"status":"ok","platform":"line"}`이 반환되어야 합니다.

**그룹에서 봇이 아무것도 수신하지 못함.** `LINE_ALLOWED_GROUPS`에 `C...` 그룹 ID가 포함되어 있는지 확인하세요. 그룹 ID를 찾으려면 테스트 메시지를 보내고 `~/.hermes/logs/gateway.log`에서 `LINE: rejecting unauthorized source`를 grep 검색하세요 — 거부된 소스 딕셔너리에 ID가 있습니다.

**`send_image`가 "LINE_PUBLIC_URL must be set" 오류와 함께 실패함.** LINE의 Messaging API는 바이너리 업로드를 허용하지 않습니다 — 이미지, 오디오, 비디오는 접근 가능한 HTTPS URL이어야 합니다. `LINE_PUBLIC_URL`을 터널의 퍼블릭 호스트 이름으로 설정하면, 어댑터가 자동으로 `/line/media/<token>/<filename>` 경로를 통해 파일을 제공합니다.

**포스트백 버튼이 표시되지 않음.** LLM이 `LINE_SLOW_RESPONSE_THRESHOLD`보다 빠르게 응답했거나, 다른 버블(도구 진행 상황, 스트리밍)이 응답 토큰을 먼저 소비했습니다. "느린 LLM 응답" 아래의 차단 설정을 참조하세요.

**"already in use by another profile".** 동일한 채널 액세스 토큰이 실행 중인 다른 Hermes 프로필에 바인딩되어 있습니다. 다른 게이트웨이를 중지하거나 별도의 채널을 사용하세요.

---

## 제한 사항

* **청크 당 단일 버블.** 각 LINE 텍스트 버블은 최대 5000자로 제한되며, Reply/Push 호출 당 최대 5개의 버블이 전송됩니다. 긴 응답은 줄임표로 잘립니다.
* **네이티브 메시지 편집 불가.** LINE에는 메시지 편집 API가 없습니다 — 스트리밍 응답은 항상 이전 버블을 편집하지 않고 새로운 버블을 전송합니다.
* **마크다운 렌더링 불가.** 굵게(`**`), 기울임꼴(`*`), 코드 펜스, 제목은 문자 그대로 렌더링됩니다. 어댑터는 전송 전에 이를 제거합니다. URL은 보존됩니다(`[label](url)`은 `label (url)`이 됩니다).
* **로딩 표시기는 DM 전용.** LINE은 그룹과 룸에서 채팅/로딩 API를 거부하므로, 입력 표시기는 1:1 채팅에서만 나타납니다.
