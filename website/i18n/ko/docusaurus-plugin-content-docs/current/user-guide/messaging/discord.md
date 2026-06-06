# Discord

Hermes Agent는 Discord와 원활하게 통합되어 텍스트 채널, 스레드, 다이렉트 메시지 및 음성 채널에서 사용할 수 있습니다.

## 전제 조건

- Discord 계정
- (선택) 봇을 추가할 권한이 있는 자신만의 Discord 서버(Guild)

## 1단계: Discord 애플리케이션 생성

1. [Discord Developer Portal](https://discord.com/developers/applications)로 이동합니다.
2. 우측 상단의 **New Application**을 클릭합니다.
3. 봇의 이름을 입력하고 약관에 동의한 후 **Create**를 클릭합니다.
4. (선택) 앱 설정에서 아이콘과 설명을 추가합니다.

## 2단계: 봇 생성 및 토큰 받기

1. 왼쪽 사이드바에서 **Bot**으로 이동합니다.
2. 봇 이름과 아이콘을 설정합니다 (앱 정보와 기본적으로 동기화됨).
3. **Reset Token**을 클릭하여 봇 토큰을 생성합니다. 
4. **토큰을 복사하여 안전한 곳에 저장하세요.** 이 토큰은 다시 볼 수 없습니다.

## 3단계: Privileged Gateway Intents 활성화

Hermes가 정상적으로 작동하려면 특정 권한(인텐트)이 필요합니다:

1. **Bot** 페이지에서 아래로 스크롤하여 **Privileged Gateway Intents** 섹션을 찾습니다.
2. 다음 인텐트를 모두 활성화합니다:
   - **Presence Intent**
   - **Server Members Intent** 
   - **Message Content Intent** (메시지 내용을 읽고 응답하는 데 필수)
3. **Save Changes**를 클릭합니다.

## 4단계: 봇 초대(Invite) 링크 생성

1. 왼쪽 사이드바에서 **OAuth2** -> **URL Generator**로 이동합니다.
2. **Scopes**에서 `bot`을 선택합니다.
3. **Bot Permissions**에서 필요한 권한을 선택합니다. 권장되는 기본 권한은 다음과 같습니다:
   - Read Messages/View Channels
   - Send Messages
   - Send Messages in Threads
   - Embed Links
   - Attach Files
   - Read Message History
   - Connect (음성 채널용)
   - Speak (음성 채널용)
   - Use Voice Activity
4. 하단에 생성된 URL을 복사합니다.

## 5단계: 서버에 봇 추가

1. 복사한 URL을 웹 브라우저에 붙여넣습니다.
2. 봇을 추가할 서버를 선택합니다 (해당 서버에 '서버 관리' 권한이 있어야 함).
3. 권한을 확인하고 **승인(Authorize)**을 클릭합니다.

## 6단계: 환경 변수 구성

`~/.hermes/.env` 파일에 봇 토큰과 자신의 Discord User ID를 설정하세요:

```bash
# 2단계에서 복사한 봇 토큰
DISCORD_BOT_TOKEN=your-bot-token

# 당신의 Discord User ID (봇 사용을 허가할 사용자)
DISCORD_ALLOWED_USERS=your-discord-user-id
```

### Discord User ID 찾는 방법:
1. Discord에서 **사용자 설정(User Settings)** -> **고급(Advanced)**으로 이동합니다.
2. **개발자 모드(Developer Mode)**를 켭니다.
3. 채팅창이나 우측 목록에서 자신의 프로필을 우클릭하고 **Copy User ID**를 선택합니다.

## 7단계: 게이트웨이 시작

```bash
hermes gateway
```

게이트웨이가 성공적으로 시작되면 터미널에 `[Discord] Connected as YourBotName#1234`와 같은 메시지가 표시됩니다.

---

## 봇 사용 방법

Hermes Discord 봇은 다양한 상호작용 방식을 지원합니다:

### 1. 다이렉트 메시지 (DM)
- 봇에게 직접 다이렉트 메시지를 보냅니다.
- 다른 사람의 방해 없이 에이전트와 개인적으로 대화할 수 있습니다.

### 2. 채널 내 멘션 (Mentions)
- 아무 텍스트 채널에서 `@봇이름`을 멘션하여 대화합니다.
- 멘션된 메시지에만 응답합니다 (공개 채널의 다른 메시지에는 반응하지 않음).

### 3. 스레드 (Threads)
- Discord 스레드 기능을 지원합니다.
- 봇이 보낸 메시지에 "답장(Reply)" 하거나, 채널 내 스레드에서 봇과 대화하면 해당 스레드의 문맥을 기억합니다.

### 4. 파일 및 이미지 처리
- 채팅에 텍스트, 이미지, 문서, PDF 등의 파일을 업로드하면 봇이 해당 파일을 읽고 쿼리 컨텍스트로 사용할 수 있습니다.

## 음성 채널 및 Voice Mode

Hermes는 Discord 음성 채널에 연결하여 오디오를 스트리밍할 수 있습니다. 음성 모드 설정 및 작동 방식에 대해서는 다음 문서를 참조하세요:

- [Voice Mode](/user-guide/features/voice-mode)
- [Use Voice Mode with Hermes](/guides/use-voice-mode-with-hermes)

### Voice Channel Audio Effects (주변음 및 구두 응답)

봇이 음성 채널에 있을 때 좀 더 대화하는 듯한 느낌을 줄 수 있습니다: 도구를 실행하기 전에 짧은 구두 응답("제가 확인해 볼게요" 등)을 하고, 도구가 실행되는 동안 백그라운드에서 "생각하는" 듯한 주변음(ambient)이 재생됩니다. (Grok 보이스 모드와 유사)

이 기능은 **기본적으로 꺼져 있습니다**. `config.yaml`에서 활성화할 수 있습니다:

```yaml
discord:
  voice_fx:
    enabled: true          # 메인 스위치
    ambient_enabled: true  # 도구 실행 중 "생각하는" 대기음
    ambient_path: ""       # 커스텀 반복 파일 경로; "" = 내장 합성 패드 사용
    ambient_gain: 0.18     # 대기음 볼륨 (0.0–1.0)
    duck_gain: 0.06        # 봇이 말할 때 대기음 볼륨 감소율
    speech_gain: 1.0       # TTS / 구두 응답 볼륨
    ack_enabled: true      # 턴의 첫 번째 도구 호출 전 짧은 문구 말하기
    ack_phrases:           # 무작위로 선택됨; 비활성화하려면 []로 설정
      - "Let me look into that."
      - "One moment."
      - "Checking on that now."
```

## 포럼 채널 (Forum Channels)

Discord 포럼 채널(유형 15)은 다이렉트 메시지를 허용하지 않습니다. ㅡ 포럼의 모든 게시물은 스레드여야 합니다. Hermes는 포럼 채널을 자동 감지하고 전송해야 할 때마다 새 스레드 게시물을 생성하여 `send_message`, TTS, 이미지, 음성 메시지 및 파일 첨부가 모두 에이전트의 특별한 처리 없이 작동합니다.

## 문제 해결 (Troubleshooting)

### 봇이 온라인이지만 메시지에 응답하지 않음
**원인**: Message Content Intent가 비활성화되어 있습니다.
**해결**: Developer Portal → 앱 → Bot → Privileged Gateway Intents → **Message Content Intent** 활성화 → Save Changes 클릭 후 게이트웨이 재시작.

### 시작 시 "Disallowed Intents" 오류
**원인**: Developer Portal에서 활성화되지 않은 인텐트를 코드에서 요청했습니다.
**해결**: Bot 설정에서 3개의 Privileged Gateway Intents(Presence, Server Members, Message Content)를 모두 활성화한 후 재시작하세요.

### 봇이 특정 채널의 메시지를 볼 수 없음
**원인**: 봇 역할에 해당 채널을 볼 수 있는 권한이 없습니다.
**해결**: 채널 설정 → 권한(Permissions) → 봇의 역할 추가 및 **채널 보기(View Channel)**, **메시지 기록 보기(Read Message History)** 권한 활성화.

### 403 Forbidden 오류
**원인**: 봇에 필요한 권한이 부족합니다.
**해결**: 4단계의 URL을 사용하여 올바른 권한으로 봇을 다시 초대하거나, 서버 설정 → 역할에서 봇의 역할 권한을 수동으로 조정하세요.

### 봇이 오프라인 상태임
**원인**: Hermes 게이트웨이가 실행 중이 아니거나 토큰이 잘못되었습니다.
**해결**: `hermes gateway`가 실행 중인지 확인하세요. `.env` 파일의 `DISCORD_BOT_TOKEN`을 확인하고 재설정했다면 업데이트하세요.

### "User not allowed" / 봇이 무시함
**원인**: 사용자 ID가 `DISCORD_ALLOWED_USERS`에 없습니다.
**해결**: `~/.hermes/.env`의 `DISCORD_ALLOWED_USERS`에 본인의 User ID를 추가하고 게이트웨이를 재시작하세요.

### 같은 채널의 사람들이 컨텍스트를 예기치 않게 공유함
**원인**: `group_sessions_per_user`가 비활성화되었거나 플랫폼이 해당 컨텍스트의 메시지에 대한 사용자 ID를 제공할 수 없습니다.
**해결**: `~/.hermes/config.yaml`에 다음을 설정하고 게이트웨이를 재시작하세요:
```yaml
group_sessions_per_user: true
```

## 보안 (Security)

:::warning
봇과 상호 작용할 수 있는 사람을 제한하려면 항상 `DISCORD_ALLOWED_USERS` (또는 `DISCORD_ALLOWED_ROLES`)를 설정하세요. 둘 다 설정되지 않은 경우, 게이트웨이는 안전 조치로 기본적으로 모든 사용자를 거부합니다. 신뢰하는 사람만 승인하세요 — 승인된 사용자는 도구 사용 및 시스템 액세스를 포함하여 에이전트 기능에 대한 전체 액세스 권한을 갖습니다.
:::

### 역할 기반 접근 제어 (Role-Based Access Control)
개별 사용자 목록 대신 역할(roles)로 접근을 관리하는 서버의 경우, 역할 ID의 쉼표 구분 목록인 `DISCORD_ALLOWED_ROLES`를 사용하세요.

```bash
# ~/.hermes/.env — DISCORD_ALLOWED_USERS와 함께 사용하거나 대체 가능
DISCORD_ALLOWED_ROLES=987654321098765432,876543210987654321
```

### 멘션 제어 (Mention Control)
기본적으로 Hermes는 봇의 응답에 `@everyone`, `@here` 및 역할 멘션이 포함되어 있더라도 봇이 핑(ping)하는 것을 차단합니다. 이는 봇이 의도치 않게 서버 전체에 스팸을 보내는 것을 방지합니다.

`config.yaml`이나 환경 변수를 통해 이 기본값을 완화할 수 있습니다:
```yaml
# ~/.hermes/config.yaml
discord:
  allow_mentions:
    everyone: false      # 봇이 @everyone / @here를 핑하도록 허용
    roles: false         # 봇이 @role 멘션을 핑하도록 허용
    users: true          # 봇이 개별 @user를 핑하도록 허용
    replied_user: true   # 메시지에 응답할 때 작성자를 핑
```
