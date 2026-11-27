---
sidebar_position: 9
title: "Matrix"
description: "Hermes Agent를 Matrix 봇으로 설정하기"
---

# Matrix 설정

Hermes Agent는 개방형 연합 메시징 프로토콜인 Matrix와 통합됩니다. Matrix를 사용하면 직접 홈서버를 운영하거나 matrix.org 같은 공개 서버를 사용할 수 있으며, 어느 경우든 통신을 계속 직접 관리할 수 있습니다. 봇은 `mautrix` Python SDK를 통해 연결되고, 도구 사용, 메모리, 추론을 포함한 Hermes Agent 파이프라인으로 메시지를 처리한 뒤 실시간으로 응답합니다. 텍스트, 파일 첨부, 이미지, 오디오, 비디오 및 선택적 종단 간 암호화(E2EE)를 지원합니다.

Hermes는 Synapse, Conduit, Dendrite 또는 matrix.org 등 모든 Matrix 홈서버에서 작동합니다.

설정에 들어가기 전에, 대부분의 사용자가 가장 궁금해하는 부분부터 살펴보겠습니다. 연결된 뒤 Hermes가 어떻게 동작하는지에 대한 내용입니다.

## Hermes의 동작 방식

| 컨텍스트 | 동작 |
|---------|----------|
| **DM** | Hermes는 모든 메시지에 응답합니다. `@멘션`이 필요하지 않습니다. 각 DM에는 별도의 세션이 있습니다. DM에서 봇이 `@멘션`될 때 스레드를 시작하려면 `MATRIX_DM_MENTION_THREADS=true`로 설정합니다. |
| **방** | 기본적으로 Hermes는 응답하려면 `@멘션`을 요구합니다. 자유 응답 방을 사용하려면 `MATRIX_REQUIRE_MENTION=false`로 설정하거나 방 ID를 `MATRIX_FREE_RESPONSE_ROOMS`에 추가합니다. 방 초대는 자동으로 수락됩니다. |
| **스레드** | Hermes는 Matrix 스레드(MSC3440)를 지원합니다. 스레드에서 답장하면 Hermes는 스레드 컨텍스트를 기본 방 타임라인과 분리하여 유지합니다. 봇이 이미 참여한 스레드에는 멘션이 필요하지 않습니다. |
| **자동 스레드 생성** | 기본적으로 Hermes는 방에서 응답하는 각 메시지에 대해 자동으로 스레드를 생성합니다. 이를 통해 대화를 분리할 수 있습니다. 비활성화하려면 `MATRIX_AUTO_THREAD=false`로 설정합니다. DM 메시지에도 자동으로 스레드를 생성하려면 `MATRIX_DM_AUTO_THREAD=true`(기본값 false)로 설정합니다. 이는 DM에서 봇이 `@멘션`될 때만 스레드를 시작하는 `MATRIX_DM_MENTION_THREADS`와는 별개입니다. |
| **명령어** | Matrix 클라이언트가 일반 `/commands`를 전송하면 Hermes가 이를 받습니다. 클라이언트가 `/`를 로컬 명령어용으로 예약한다면 대신 `!commands`를 사용합니다. Hermes는 알려진 `!command` 별칭을 `/command`로 정규화합니다. |
| **대화형 컨트롤** | 위험한 명령어 승인과 `/model` 선택에 Matrix 리액션을 사용할 수 있습니다. 승인 리액션은 작업을 요청한 사용자로 제한할 수 있습니다. |
| **사고 및 도구 활동** | 게이트웨이 진행률 표시가 활성화되면 Matrix는 스레드가 있는 편집 가능한 사고/도구 활동 패널을 사용하므로 업데이트가 기본 방 타임라인을 도배하지 않습니다. |
| **여러 사용자가 있는 공유 방** | 기본적으로 Hermes는 방 안에서 사용자별로 세션 기록을 분리합니다. 명시적으로 비활성화하지 않는 한 같은 방에서 대화하는 두 사람은 하나의 기록을 공유하지 않습니다. |

:::tip
봇은 초대받으면 자동으로 방에 참여합니다. 봇의 Matrix 사용자를 어느 방에든 초대하면 봇이 참여하여 응답을 시작합니다.
:::

## 기능 매트릭스

이 표는 Matrix 어댑터의 기능 선언과 Matrix 테스트 커버리지를 기반으로 합니다. E2EE는 배포 환경에서 암호화된 방을 비활성화할지, 기회적으로 사용할지, 필수로 사용할지를 선택하므로 모드 기반으로 표시됩니다.

| 기능 | Matrix |
|------------|--------|
| 텍스트 | yes |
| 스레드 | yes |
| 리액션 | yes |
| 승인 | yes |
| 모델 선택기 | yes |
| 사고 패널 | yes |
| 이미지 | yes |
| 여러 이미지 | yes |
| 파일 | yes |
| 음성/오디오 | yes |
| 비디오 | yes |
| E2EE | off / optional / required |
| 진단 | yes |

### Matrix의 세션 모델

기본값은 다음과 같습니다.

- 각 DM에는 자체 세션이 있습니다.
- 각 스레드에는 자체 세션 네임스페이스가 있습니다.
- 공유 방의 각 사용자는 해당 방 안에서 자체 세션을 가집니다.

이는 `config.yaml`로 제어합니다.

```yaml
group_sessions_per_user: true
```

방 전체에 하나의 공유 대화를 사용하려는 경우에만 `false`로 설정합니다.

```yaml
group_sessions_per_user: false
```

공유 세션은 협업 방에서 유용할 수 있지만 다음과 같은 의미도 있습니다.

- 사용자가 컨텍스트 증가량과 토큰 비용을 공유합니다.
- 한 사용자의 도구를 많이 사용하는 긴 작업이 다른 모든 사용자의 컨텍스트를 부풀릴 수 있습니다.
- 한 사용자의 실행 중인 작업이 같은 방에서 다른 사용자의 후속 요청을 방해할 수 있습니다.

### 멘션 및 스레드 설정

환경 변수 또는 `config.yaml`을 통해 멘션 및 자동 스레드 생성 동작을 설정할 수 있습니다.

```yaml
matrix:
  require_mention: true           # Require @mention in rooms (default: true)
  allowed_users:                  # Matrix users allowed to trigger agent turns
    - "@alice:matrix.org"
  allowed_rooms:                  # Matrix rooms allowed to trigger agent turns
    - "!abc123:matrix.org"
  free_response_rooms:            # Rooms exempt from mention requirement
    - "!abc123:matrix.org"
  ignore_user_patterns:           # Bridge/appservice ghost users to ignore
    - "^@telegram_"
    - "^@whatsapp_"
  process_notices: false          # Ignore m.notice by default
  session_scope: room             # auto|room|thread; room is recommended for project rooms
  auto_thread: true               # Auto-create threads for responses (default: true)
  dm_mention_threads: false       # Create thread when @mentioned in DM (default: false)
  max_message_length: 16000       # Outbound chunk size in chars (default: 16000, max: 65535)
```

또는 환경 변수를 사용합니다.

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_ALLOWED_USERS=@alice:matrix.org
MATRIX_ALLOWED_ROOMS=!abc123:matrix.org
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@whatsapp_'
MATRIX_PROCESS_NOTICES=false
MATRIX_SESSION_SCOPE=room       # recommended for stable project-room context
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_REACTIONS=true          # default: true — emoji reactions during processing
MATRIX_ALLOW_ROOM_MENTIONS=false
```

:::tip 리액션 비활성화
`MATRIX_REACTIONS=false`는 봇이 수신 메시지에 게시하는 처리 수명 주기 이모지 리액션(👀/✅/❌)을 끕니다. 리액션 이벤트가 시끄럽거나 참여 중인 모든 클라이언트에서 지원되지 않는 방에 유용합니다.
:::

:::tip 방 전체 멘션
Hermes는 `@alice:example.org` 같은 명시적인 Matrix ID에 대해 구조화된 Matrix 사용자 멘션을 보냅니다. 방 전체 `@room` 알림은 기본적으로 비활성화되어 있습니다. 봇이 모든 사용자에게 알림을 보낼 수 있는 방에서만 `MATRIX_ALLOW_ROOM_MENTIONS=true`로 설정합니다.
:::

:::note
`MATRIX_REQUIRE_MENTION`이 없던 버전에서 업그레이드하는 경우, 봇은 이전에 방의 모든 메시지에 응답했습니다. 이 동작을 유지하려면 `MATRIX_REQUIRE_MENTION=false`로 설정합니다.
:::

### 프로젝트 방 격리

같은 Matrix 봇을 여러 프로젝트 방에서 사용하는 경우 안정적인 방 범위 세션을 설정합니다.

```bash
MATRIX_SESSION_SCOPE=room
MATRIX_AUTO_THREAD=false
```

`MATRIX_SESSION_SCOPE`는 다음 값을 허용합니다.

| 범위 | 동작 |
|-------|----------|
| `auto` | 이전 버전과 호환되는 기본값입니다. 기존 `MATRIX_AUTO_THREAD` 동작이 가상 스레드를 제어합니다. |
| `room` | 스레드가 없는 방 메시지는 하나의 안정적인 방 세션에 유지됩니다. 실제 Matrix 스레드는 여전히 해당 스레드의 루트를 사용합니다. |
| `thread` | 스레드가 없는 방 메시지는 트리거 이벤트 ID로부터 스레드/세션을 합성합니다. |

이제 Hermes는 에이전트 프롬프트에 현재 Matrix 방 이름, 방 ID, 주제, 메시지 ID 및 Matrix 방 경계 참고 사항을 포함합니다. `/status`에는 현재 Matrix 방/세션 범위도 표시되며, `/resume`은 명시적으로 `/resume --cross-room <session name>`을 사용하지 않는 한 다른 Matrix 방의 이름 있는 세션을 조용히 재개하지 않습니다.

`MATRIX_SESSION_SCOPE=room`은 방/스레드 레인을 제어합니다. 기존 `group_sessions_per_user` 설정은 해당 방 안의 사용자가 레인을 공유할지 여부를 계속 제어합니다. `group_sessions_per_user: true`(기본값)인 경우 Alice와 Bob은 별도의 Project B 세션을 얻습니다. `group_sessions_per_user: false`인 경우 방에는 하나의 공유 Project B 기록이 있습니다.

이 가이드는 봇 계정 생성부터 첫 메시지 전송까지 전체 설정 과정을 안내합니다.

## 1단계: 봇 계정 생성

봇용 Matrix 사용자 계정이 필요합니다. 계정을 만드는 방법은 여러 가지입니다.

### 옵션 A: 홈서버에 등록(권장)

직접 홈서버(Synapse, Conduit, Dendrite)를 운영하는 경우:

1. 관리자 API 또는 등록 도구를 사용하여 새 사용자를 생성합니다.

```bash
# Synapse example
register_new_matrix_user -c /etc/synapse/homeserver.yaml http://localhost:8008
```

2. `hermes` 같은 사용자 이름을 선택합니다. 전체 사용자 ID는 `@hermes:your-server.org`가 됩니다.

### 옵션 B: matrix.org 또는 다른 공개 홈서버 사용

1. [Element Web](https://app.element.io)으로 이동하여 새 계정을 만듭니다.
2. 봇의 사용자 이름(예: `hermes-bot`)을 선택합니다.

### 옵션 C: 자신의 계정 사용

Hermes를 자신의 사용자로 실행할 수도 있습니다. 이 경우 봇이 사용자를 대신하여 게시하므로 개인 비서에 유용합니다.

## 2단계: 액세스 토큰 받기

Hermes는 홈서버에 인증하기 위해 액세스 토큰이 필요합니다. 두 가지 방법이 있습니다.

### 옵션 A: 액세스 토큰(권장)

토큰을 받는 가장 안정적인 방법입니다.

**Element 사용:**
1. 봇 계정으로 [Element](https://app.element.io)에 로그인합니다.
2. **설정** → **도움말 및 정보**로 이동합니다.
3. 아래로 스크롤하여 **고급**을 펼치면 액세스 토큰이 표시됩니다.
4. **즉시 복사합니다.**

**API 사용:**

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "user": "@hermes:your-server.org",
    "password": "your-password"
  }'
```

응답에 `access_token` 필드가 포함됩니다. 이를 복사합니다.

:::warning[액세스 토큰을 안전하게 보관하세요]
액세스 토큰은 봇의 Matrix 계정에 대한 전체 액세스 권한을 제공합니다. 절대로 공개적으로 공유하거나 Git에 커밋하지 마세요. 유출된 경우 해당 사용자의 모든 세션에서 로그아웃하여 취소합니다.
:::

### 옵션 B: 비밀번호 로그인

액세스 토큰을 제공하는 대신 봇의 사용자 ID와 비밀번호를 Hermes에 제공할 수 있습니다. Hermes가 시작할 때 자동으로 로그인합니다. 더 간단하지만 비밀번호가 `.env` 파일에 저장된다는 의미입니다.

```bash
MATRIX_USER_ID=@hermes:your-server.org
MATRIX_PASSWORD=your-password
```

## 3단계: Matrix 사용자 ID 찾기

Hermes Agent는 Matrix 사용자 ID를 사용하여 봇과 상호작용할 수 있는 사람을 제어합니다. Matrix 사용자 ID 형식은 `@username:server`입니다.

다음과 같이 찾습니다.

1. [Element](https://app.element.io)(또는 선호하는 Matrix 클라이언트)을 엽니다.
2. 아바타 클릭 → **설정**으로 이동합니다.
3. 프로필 상단에 사용자 ID가 표시됩니다(예: `@alice:matrix.org`).

:::tip
Matrix 사용자 ID는 항상 `@`로 시작하고 서버 이름 앞에 `:`가 포함됩니다. 예: `@alice:matrix.org`, `@bob:your-server.com`.
:::

## 4단계: Hermes Agent 설정

### 옵션 A: 대화형 설정(권장)

안내형 설정 명령어를 실행합니다.

```bash
hermes gateway setup
```

메시지가 표시되면 **Matrix**를 선택하고 홈서버 URL, 액세스 토큰(또는 사용자 ID + 비밀번호), 허용된 사용자 ID를 입력합니다.

### 옵션 B: 수동 설정

`~/.hermes/.env` 파일에 다음을 추가합니다.

**액세스 토큰 사용:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Optional: restrict which rooms can trigger the bot
MATRIX_ALLOWED_ROOMS=!abc123:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**비밀번호 로그인 사용:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# Security
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

## 비공개 배포 강화

비공개 Matrix 배포에서는 사용자 및 방 허용 목록을 모두 설정합니다. `MATRIX_ALLOWED_USERS`가 설정되지 않으면 참여한 방에서 봇에 도달할 수 있는 모든 발신자가 에이전트 턴을 트리거할 수 있습니다. `MATRIX_ALLOWED_ROOMS`가 설정되지 않으면 봇이 참여하는 모든 방이 에이전트 턴을 트리거할 수 있습니다. 잠금 상태의 배포에서는 다음 두 항목을 모두 설정해야 합니다.

```bash
MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
MATRIX_ALLOWED_ROOMS=!ops:matrix.example.org,!dmroom:matrix.example.org
```

브리지 및 appservice 배포에는 추가적인 루프 보호가 필요합니다. Hermes는 항상 자체 이벤트, 로컬 파트가 `_`로 시작하는 Matrix appservice 스타일 사용자, 중복 이벤트 ID, 오래된 시작 이벤트, 편집 대체 이벤트 및 기본적으로 `m.notice` 이벤트를 무시합니다. 브리지에서 다른 이름 규칙을 사용하는 경우 배포 환경에 맞는 브리지 고스트 패턴을 추가합니다.

```bash
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@slack_,^@whatsapp_'
```

신뢰할 수 있는 사람의 워크플로에서 실제로 `m.notice`를 보내는 경우에만 알림을 활성화합니다.

```bash
MATRIX_PROCESS_NOTICES=true
```

발신 방 전체 알림은 기본적으로 비활성화되어 있습니다. 봇이 `@room`으로 방 전체를 깨울 수 있도록 명시적으로 허용한 경우가 아니라면 `MATRIX_ALLOW_ROOM_MENTIONS=false`로 유지합니다.

진단 및 디버그 페이로드에서는 Matrix 액세스 토큰, 복구 키, 장치 식별자 및 메시지 본문을 마스킹합니다. 미디어 다운로드는 Matrix `mxc://` 콘텐츠 URI로 제한되며 `MATRIX_MAX_MEDIA_BYTES`를 초과하면 거부됩니다. 연합 방과 신뢰할 수 없는 홈서버의 입력은 신뢰할 수 없는 것으로 취급합니다. 방 허용 목록을 엄격하게 유지하고, 도구를 많이 사용하는 작업에는 DM 또는 비공개 방을 우선 사용하며, 브리지 고스트 또는 appservice 퍼펫을 허용 사용자로 승인하지 마세요.

`~/.hermes/config.yaml`의 선택적 동작 설정:

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true`는 공유 방 안에서 각 참여자의 컨텍스트를 격리합니다.

### 게이트웨이 시작

설정이 끝나면 Matrix 게이트웨이를 시작합니다.

```bash
hermes gateway
```

봇이 홈서버에 연결되고 몇 초 안에 동기화를 시작해야 합니다. DM 또는 봇이 참여한 방에서 메시지를 보내 테스트합니다.

:::tip
지속적으로 실행하려면 `hermes gateway`를 백그라운드에서 실행하거나 systemd 서비스로 실행할 수 있습니다. 자세한 내용은 배포 문서를 참조하세요.
:::

## 종단 간 암호화(E2EE)

Hermes는 Matrix 종단 간 암호화를 지원하므로 암호화된 방에서 봇과 대화할 수 있습니다.

### 요구 사항

E2EE에는 암호화 추가 기능이 포함된 `mautrix` 라이브러리와 `libolm` C 라이브러리가 필요합니다.

```bash
# Install mautrix with E2EE support
pip install 'mautrix[encryption]'

# Or install with hermes extras
cd ~/.hermes/hermes-agent && uv pip install -e ".[matrix]"
```

시스템에 `libolm`도 설치해야 합니다.

```bash
# Debian/Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm

# Fedora
sudo dnf install libolm-devel
```

### E2EE 활성화

`~/.hermes/.env`에 다음을 추가합니다.

```bash
MATRIX_E2EE_MODE=required
```

`MATRIX_E2EE_MODE`는 다음 값을 허용합니다.

| 모드 | 동작 |
|------|----------|
| `off` | Matrix E2EE를 초기화하지 않습니다. |
| `optional` | 종속성을 사용할 수 있으면 E2EE를 시도하지만, 암호화를 초기화할 수 없어도 암호화되지 않은 방은 계속 작동합니다. |
| `required` | E2EE 종속성 또는 암호화 설정을 사용할 수 없으면 폐쇄적으로 실패합니다. |

암호화 설정을 사용할 수 없으면 선택 모드가 비 E2EE 동작으로 대체될 수 있습니다. 필수 모드는 조용히 수준을 낮추는 대신 폐쇄적으로 실패합니다.

이전 버전과의 호환성을 위해 `MATRIX_ENCRYPTION=true`도 여전히 필수 E2EE 동작을 활성화합니다.

E2EE가 활성화되면 Hermes는 다음을 수행합니다.

- 암호화 키를 `~/.hermes/platforms/matrix/store/`에 저장합니다(레거시 설치: `~/.hermes/matrix/store/`).
- 처음 연결할 때 장치 키를 업로드합니다.
- 수신 메시지를 복호화하고 발신 메시지를 자동으로 암호화합니다.
- 초대받으면 암호화된 방에 자동으로 참여합니다.

### Matrix 도구 및 컨트롤

Matrix 대화에서 Hermes는 에이전트에 Matrix 전용 도구를 노출합니다.

- `matrix_send_reaction`
- `matrix_redact_message`
- `matrix_create_room`
- `matrix_invite_user`
- `matrix_fetch_history`
- `matrix_set_presence`

이 도구는 Matrix 컨텍스트에 한정되며 Matrix가 아닌 도구 모음에서는 사용할 수 없습니다. 관리자 스타일 도구는 기본적으로 비활성화되어 있습니다. 삭제에는 `MATRIX_TOOLS_ALLOW_REDACTION=true`, 초대에는 `MATRIX_TOOLS_ALLOW_INVITES=true`, 방 생성에는 `MATRIX_TOOLS_ALLOW_ROOM_CREATE=true`가 필요합니다. 공개 방 생성에는 `MATRIX_ALLOW_PUBLIC_ROOMS=true`도 필요합니다. `MATRIX_ALLOWED_ROOMS`가 설정된 경우 Matrix 도구는 해당 방만 대상으로 삼을 수 있습니다.

리액션 컨트롤은 다음을 사용합니다.

- ✅ 한 번 승인
- ♾️ 항상 승인
- ❌ 거부
- `/model` 선택을 위한 숫자 리액션

승인/모델 선택 프롬프트를 방 안의 인증된 어떤 Matrix 사용자든 조작하도록 의도적으로 허용하려면 `MATRIX_APPROVAL_REQUIRE_SENDER=false`로 설정합니다. Hermes가 작업을 요청한 사람을 알고 있는 경우 기본값은 요청자에게 바인딩됩니다.

### 미디어 제한

Hermes는 Matrix 미디어 API를 통해 Matrix 이미지, 파일, 오디오 및 비디오를 업로드하고 다운로드합니다. 생성된 여러 이미지는 하나의 순서가 있는 논리적 배치로 전송되며, 배치 전체에서 캡션과 스레드 컨텍스트가 유지됩니다.

기본적으로 100MB를 초과하는 Matrix 미디어는 업로드/다운로드 전에 거부됩니다. 다음으로 재정의합니다.

```bash
MATRIX_MAX_MEDIA_BYTES=104857600
```

수신 미디어는 Matrix `mxc://` 콘텐츠 URI를 사용해야 합니다. 연합 방이 제한 없는 다운로더로 변하는 것을 방지하기 위해 Matrix 이벤트의 임의 HTTP(S) 미디어 URL은 거부됩니다.

### 교차 서명 검증(권장)

Matrix 계정에서 교차 서명이 활성화되어 있다면(Element의 기본값) 복구 키를 설정하여 봇이 시작할 때 자체적으로 장치에 서명할 수 있도록 합니다. 이렇게 하지 않으면 장치 키가 교체된 뒤 다른 Matrix 클라이언트가 봇과 암호화 세션을 공유하지 않을 수 있습니다.

```bash
MATRIX_RECOVERY_KEY=EsT... your recovery key here
```

**찾는 위치:** Element에서 **설정** → **보안 및 개인정보 보호** → **암호화** → 복구 키(또는 "보안 키")로 이동합니다. 이는 교차 서명을 처음 설정할 때 저장하라는 안내를 받은 키입니다.

시작할 때 `MATRIX_RECOVERY_KEY`가 설정되어 있으면 Hermes는 홈서버의 보안 비밀 저장소에서 교차 서명 키를 가져와 현재 장치에 서명합니다. 이 작업은 멱등적이며 영구적으로 활성화해 두어도 안전합니다.

Hermes가 새 Matrix 복구 키를 부트스트랩하는 경우 원시 키를 절대 로그에 남기지 않습니다. 시작하기 전에 `MATRIX_RECOVERY_KEY_OUTPUT_FILE=/secure/path/matrix-recovery-key.txt`를 설정하면 생성된 키를 파일 모드 `0600`으로 한 번 기록합니다. 파일이 이미 존재하면 덮어쓰지 않습니다.

:::warning[암호화 저장소 삭제]
`~/.hermes/platforms/matrix/store/crypto.db`를 삭제하면 봇은 암호화 ID를 잃습니다. 같은 장치 ID로 단순히 재시작해도 완전히 복구되지 않습니다. 홈서버에는 이전 ID 키로 서명된 일회용 키가 여전히 있고 피어가 새 Olm 세션을 설정할 수 없기 때문입니다.

Hermes는 시작 시 이 상태를 감지하고 E2EE 활성화를 거부하며 다음을 기록합니다: `device XXXX has stale one-time keys on the server signed with a previous identity key`.

**가장 쉬운 복구 방법: 새 액세스 토큰 생성**(이전 키 기록이 없는 새 장치 ID를 얻음). 아래의 "이전 버전에서 E2EE를 사용하여 업그레이드" 절을 참조하세요. 이 방법이 가장 안정적이며 홈서버 데이터베이스를 건드리지 않아도 됩니다.

**수동 복구**(고급 — 동일한 장치 ID 유지):

1. Synapse를 중지하고 데이터베이스에서 이전 장치를 삭제합니다.
   ```bash
   sudo systemctl stop matrix-synapse
   sudo sqlite3 /var/lib/matrix-synapse/homeserver.db "
     DELETE FROM e2e_device_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_one_time_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_fallback_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM devices WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
   "
   sudo systemctl start matrix-synapse
   ```
   또는 Synapse 관리자 API를 사용합니다(사용자 ID는 URL 인코딩해야 함).
   ```bash
   curl -X DELETE -H "Authorization: Bearer ADMIN_TOKEN" \
     'https://your-server/_synapse/admin/v2/users/%40hermes%3Ayour-server/devices/DEVICE_ID'
   ```
   참고: 관리자 API로 장치를 삭제하면 연결된 액세스 토큰도 무효화될 수 있습니다. 이후 새 토큰을 생성해야 할 수 있습니다.

2. 로컬 암호화 저장소를 삭제하고 Hermes를 다시 시작합니다.
   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db*
   # restart hermes
   ```

다른 Matrix 클라이언트(Element, matrix-commander)는 이전 장치 키를 캐시할 수 있습니다. 복구 후 Element에서 `/discardsession`을 입력하여 봇과 새 암호화 세션을 강제로 설정합니다.
:::

:::info
`mautrix[encryption]`이 설치되지 않았거나 `libolm`이 없으면 봇은 자동으로 일반(암호화되지 않은) 클라이언트로 대체됩니다. 로그에 경고가 표시됩니다.
:::

## 홈 방

봇이 주기적 작업 출력, 리마인더 및 알림 같은 능동적 메시지를 보내는 "홈 방"을 지정할 수 있습니다. 설정 방법은 두 가지입니다.

### 슬래시 명령어 사용

봇이 있는 Matrix 방에서 `/sethome`을 입력합니다. 해당 방이 홈 방이 됩니다. Matrix 클라이언트가 슬래시 명령어를 가로채는 경우 대신 `!sethome`을 입력합니다.

### 수동 설정

`~/.hermes/.env`에 다음을 추가합니다.

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

## 방 허용 목록(`allowed_rooms`)

봇을 고정된 Matrix 방 집합으로 제한합니다. 설정하면 봇은 목록에 ID가 있는 방에서만 응답하며, 다른 방의 메시지는 봇이 멘션된 경우에도 조용히 무시됩니다.

**DM(직접 대화 방)은 이 필터에서 제외**되므로 인증된 사용자는 항상 일대일로 봇에 접근할 수 있습니다.

```yaml
matrix:
  allowed_rooms:
    - "!abc123def456:matrix.example.org"
    - "!opsroom789:matrix.example.org"
```

또는 환경 변수(쉼표로 구분)를 사용합니다.

```bash
MATRIX_ALLOWED_ROOMS="!abc123def456:matrix.example.org,!opsroom789:matrix.example.org"
```

동작:

- 비어 있음/설정되지 않음 → 제한 없음(기본값).
- 비어 있지 않음 → 방 ID가 목록에 있어야 합니다. 이 검사는 다른 모든 게이트(멘션 요구, 발신자 허용 목록 등)보다 먼저 실행됩니다.
- 별칭(`#room:server`)이 아니라 방의 **내부 ID**(`!abc...:server`)를 사용합니다. Element에서 방 → 설정 → 고급으로 이동하면 방의 내부 ID를 찾을 수 있습니다.

참조: [관리자/사용자 슬래시 명령어 분리](../../reference/slash-commands.md#permissions-and-adminuser-split).

:::tip
방 ID를 찾으려면 Element에서 방 → **설정** → **고급**으로 이동합니다. **내부 방 ID**가 표시되며 `!`로 시작합니다.
:::

## Matrix의 명령어

Hermes는 다른 메시징 플랫폼에서 지원하는 것과 동일한 게이트웨이 명령어를 Matrix에서도 지원합니다. 여기에는 `/commands`, `/model`, `/stop`, `/queue`, `/steer`, `/goal`, `/subgoal`, `/background`, `/bg`, `/btw`, `/tasks`, `/yolo`가 포함됩니다.

일부 Matrix 클라이언트는 앞에 오는 `/`를 로컬 클라이언트 명령어용으로 예약하여 알 수 없는 슬래시 명령어를 방에 보내지 않을 수 있습니다. 이 경우 Matrix에서 안전하게 사용할 수 있는 별칭으로 `!`를 사용합니다.

```text
!commands
!model
!model gpt-5.5 --provider openrouter
!queue continue with the next task
!stop
```

Hermes는 게이트웨이에 알려진 명령어, 등록된 플러그인 명령어 또는 설치된 스킬 명령어일 때만 `!command`를 정규화합니다. `!important` 같은 일반적인 느낌표 표현은 일반 채팅 메시지로 유지됩니다.

## 문제 해결

### 봇이 메시지에 응답하지 않음

**원인**: 봇이 방에 참여하지 않았거나, `MATRIX_ALLOWED_USERS`에 사용자 ID가 없거나, `MATRIX_ALLOWED_ROOMS`에 방이 없거나, 방 메시지에서 봇이 멘션되지 않았습니다.

**해결 방법**: 봇을 방에 초대합니다. 초대하면 자동으로 참여합니다. 사용자 ID가 `MATRIX_ALLOWED_USERS`에 있는지(전체 `@user:server` 형식 사용), 허용 목록이 설정된 경우 방 ID가 `MATRIX_ALLOWED_ROOMS`에 있는지 확인합니다. 방에서 봇을 멘션하거나 방을 `MATRIX_FREE_RESPONSE_ROOMS`에 추가합니다. 게이트웨이를 다시 시작합니다.

### 봇은 방에 참여하지만 모든 메시지를 조용히 삭제함(시계 불일치)

**원인**: 호스트의 시스템 시계가 실제 시간보다 앞서 있습니다. Matrix 어댑터는 시작 시 동기화 과정에서 재생된 이벤트를 무시하기 위해 5초의 시작 유예 필터(`event_ts < startup_ts - 5`)를 적용합니다. 벽시계가 앞서 있으면 모든 수신 이벤트가 "시작보다 오래된" 것으로 보여 메시지 핸들러에 도달하기 전에 삭제됩니다. 봇은 연결된 것처럼 보이지만 응답하지 않습니다. [#12614](https://github.com/NousResearch/hermes-agent/issues/12614)를 참조하세요.

**증상**: 게이트웨이 로그에 `Matrix: dropped N live events as 'too old' more than 30s after startup`가 표시됩니다.

**해결 방법**: 호스트 시계를 NTP와 동기화하고 봇을 다시 시작합니다.

```bash
# Debian/Ubuntu
sudo timedatectl set-ntp true
timedatectl status   # confirm "System clock synchronized: yes"

# macOS
sudo sntp -sS time.apple.com
```

### 시작 시 "Failed to authenticate" / "whoami failed"

**원인**: 액세스 토큰 또는 홈서버 URL이 올바르지 않습니다.

**해결 방법**: `MATRIX_HOMESERVER`가 홈서버를 가리키는지 확인합니다(`https://` 포함, 끝의 슬래시 제외). 액세스 토큰이 유효한지 확인합니다. curl로 시도해 보세요.

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/_matrix/client/v3/account/whoami
```

사용자 정보가 반환되면 토큰이 유효합니다. 오류가 반환되면 새 토큰을 생성합니다.

### "mautrix not installed" 오류

**원인**: `mautrix` Python 패키지가 설치되지 않았습니다.

**해결 방법**: 설치합니다.

```bash
pip install 'mautrix[encryption]'
```

또는 Hermes 추가 기능을 사용합니다.

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[matrix]"
```

### 암호화 오류 / "could not decrypt event"

**원인**: 암호화 키가 없거나, `libolm`이 설치되지 않았거나, 봇의 장치를 신뢰하지 않습니다.

**해결 방법**:
1. 시스템에 `libolm`이 설치되어 있는지 확인합니다(위 E2EE 절 참조).
2. `.env`에 `MATRIX_ENCRYPTION=true`가 설정되어 있는지 확인합니다.
3. Matrix 클라이언트(Element)에서 봇의 프로필로 이동 → 세션 → 봇의 장치를 확인/신뢰합니다.
4. 봇이 방금 암호화된 방에 참여했다면 참여 **이후**에 전송된 메시지만 복호화할 수 있습니다. 이전 메시지에는 접근할 수 없습니다.

### 이전 버전에서 E2EE를 사용하여 업그레이드

:::tip
`crypto.db`도 수동으로 삭제했다면 위 E2EE 절의 "암호화 저장소 삭제" 경고를 확인하세요. 홈서버에서 오래된 일회용 키를 제거하는 추가 단계가 있습니다.
:::

이전에 `MATRIX_ENCRYPTION=true`로 Hermes를 사용했고 새로운 SQLite 기반 암호화 저장소를 사용하는 버전으로 업그레이드하는 경우 봇의 암호화 ID가 변경되었습니다. Matrix 클라이언트(Element)가 이전 장치 키를 캐시하여 암호화 세션을 봇과 공유하지 않을 수 있습니다.

**증상**: 봇이 연결되고 로그에 "E2EE enabled"가 표시되지만 모든 메시지에 "could not decrypt event"가 표시되고 봇이 응답하지 않습니다.

**발생 원인**: 이전 `matrix-nio` 또는 직렬화 기반 `mautrix` 백엔드의 오래된 암호화 상태가 새 SQLite 암호화 저장소와 호환되지 않습니다. 봇은 새 암호화 ID를 생성하지만 Matrix 클라이언트에는 이전 키가 캐시되어 있어 키가 변경된 장치와 방의 암호화 세션을 공유하지 않습니다. 이는 Matrix 보안 기능입니다. 클라이언트는 같은 장치에서 변경된 ID 키를 의심스러운 것으로 취급합니다.

**해결 방법**(일회성 마이그레이션):

1. 새 장치 ID를 얻기 위해 새 액세스 토큰을 생성합니다. 가장 간단한 방법은 다음과 같습니다.

   ```bash
   curl -X POST https://your-server/_matrix/client/v3/login \
     -H "Content-Type: application/json" \
     -d '{
       "type": "m.login.password",
       "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
       "password": "***",
       "initial_device_display_name": "Hermes Agent"
     }'
   ```

   새 `access_token`을 복사하고 `~/.hermes/.env`의 `MATRIX_ACCESS_TOKEN`을 업데이트합니다.

2. 오래된 암호화 상태를 삭제합니다.

   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db
   rm -f ~/.hermes/platforms/matrix/store/crypto_store.*
   ```

3. 교차 서명을 사용하는 경우(대부분의 Element 사용자) 복구 키를 설정합니다. `~/.hermes/.env`에 추가합니다.

   ```bash
   MATRIX_RECOVERY_KEY=EsT... your recovery key here
   ```

   이렇게 하면 봇이 시작할 때 교차 서명 키로 자체 서명하므로 Element가 새 장치를 즉시 신뢰합니다. 그렇지 않으면 Element가 새 장치를 인증되지 않은 것으로 보고 암호화 세션 공유를 거부할 수 있습니다. Element의 **설정** → **보안 및 개인정보 보호** → **암호화**에서 복구 키를 찾습니다.

4. Matrix 클라이언트가 암호화 세션을 교체하도록 합니다. Element에서 봇과의 DM 방을 열고 `/discardsession`을 입력합니다. Element가 새 암호화 세션을 만들고 봇의 새 장치와 공유하도록 합니다.

5. 게이트웨이를 다시 시작합니다.

   ```bash
   hermes gateway run
   ```

   `MATRIX_RECOVERY_KEY`가 설정되어 있으면 로그에 `Matrix: cross-signing verified via recovery key`가 표시되어야 합니다.

6. 새 메시지를 보냅니다. 봇이 정상적으로 복호화하고 응답해야 합니다.

:::note
마이그레이션 전에 전송한 메시지는 이전 암호화 키가 사라졌으므로 복호화할 수 없습니다. 이는 전환 과정에만 영향을 주며 새 메시지는 정상적으로 작동합니다.
:::

:::tip
**새로 설치한 경우에는 영향을 받지 않습니다.** 이 마이그레이션은 이전 버전의 Hermes에서 작동하는 E2EE 설정을 사용하다가 업그레이드하는 경우에만 필요합니다.

**왜 새 액세스 토큰인가요?** 각 Matrix 액세스 토큰은 특정 장치 ID에 연결됩니다. 새 암호화 키와 같은 장치 ID를 재사용하면 다른 Matrix 클라이언트가 장치를 신뢰하지 않게 됩니다(변경된 ID 키를 잠재적인 보안 침해로 보기 때문입니다). 새 액세스 토큰을 사용하면 오래된 키 기록이 없는 새 장치 ID를 얻으므로 다른 클라이언트가 즉시 신뢰합니다.
:::

## 프록시 모드(macOS에서 E2EE)

Matrix E2EE에는 `libolm`이 필요하지만 macOS ARM64(Apple Silicon)에서는 컴파일되지 않습니다. `hermes-agent[matrix]` 추가 기능은 Linux에서만 사용할 수 있습니다. macOS를 사용하는 경우 프록시 모드를 사용하면 Linux VM의 Docker 컨테이너에서 E2EE를 실행하면서 실제 에이전트는 macOS에서 로컬 파일, 메모리 및 스킬에 완전히 접근하여 네이티브로 실행할 수 있습니다.

### 작동 방식

```
macOS (Host):
  └─ hermes gateway
       ├─ api_server adapter ← listens on 0.0.0.0:8642
       ├─ AIAgent ← single source of truth
       ├─ Sessions, memory, skills
       └─ Local file access (Obsidian, projects, etc.)

Linux VM (Docker):
  └─ hermes gateway (proxy mode)
       ├─ Matrix adapter ← E2EE decryption/encryption
       └─ HTTP forward → macOS:8642/v1/chat/completions
           (no LLM API keys, no agent, no inference)
```

Docker 컨테이너는 Matrix 프로토콜과 E2EE만 처리합니다. 메시지가 도착하면 메시지를 복호화하고 표준 HTTP 요청을 통해 텍스트를 호스트로 전달합니다. 호스트는 에이전트를 실행하고 도구를 호출하며 응답을 생성하고 스트리밍하여 돌려보냅니다. 컨테이너는 응답을 암호화하여 Matrix로 전송합니다. 모든 세션이 통합되므로 CLI, Matrix, Telegram 및 다른 플랫폼이 동일한 메모리와 대화 기록을 공유합니다.

### 1단계: 호스트(macOS) 설정

Docker 컨테이너에서 들어오는 요청을 호스트가 수락하도록 API 서버를 활성화합니다.

`~/.hermes/.env`에 다음을 추가합니다.

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=your-secret-key-here
API_SERVER_HOST=0.0.0.0
```

- `API_SERVER_HOST=0.0.0.0`은 모든 인터페이스에 바인딩하여 Docker 컨테이너가 접근할 수 있도록 합니다.
- 루프백이 아닌 바인딩에는 `API_SERVER_KEY`가 필요합니다. 강력한 무작위 문자열을 선택합니다.
- API 서버는 기본적으로 포트 8642에서 실행됩니다(필요하면 `API_SERVER_PORT`로 변경).

게이트웨이를 시작합니다.

```bash
hermes gateway
```

다른 설정된 플랫폼과 함께 API 서버가 시작되는 것을 확인할 수 있습니다. VM에서 접근 가능한지 확인합니다.

```bash
# From the Linux VM
curl http://<mac-ip>:8642/health
```

### 2단계: Docker 컨테이너(Linux VM) 설정

컨테이너에는 Matrix 자격 증명과 프록시 URL이 필요합니다. LLM API 키는 필요하지 않습니다.

**`docker-compose.yml`:**

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

**`Dockerfile`:**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libolm-dev && rm -rf /var/lib/apt/lists/*
RUN cd ~/.hermes/hermes-agent && uv pip install -e ".[matrix]"

CMD ["hermes", "gateway"]
```

이것이 컨테이너의 전부입니다. OpenRouter, Anthropic 또는 다른 추론 제공자의 API 키는 필요하지 않습니다.

### 3단계: 양쪽 시작

1. 먼저 호스트 게이트웨이를 시작합니다.
   ```bash
   hermes gateway
   ```

2. Docker 컨테이너를 시작합니다.
   ```bash
   docker compose up -d
   ```

3. 암호화된 Matrix 방에서 메시지를 보냅니다. 컨테이너가 메시지를 복호화하고 호스트로 전달하며 응답을 다시 스트리밍합니다.

### 설정 참조

프록시 모드는 **컨테이너 측**(얇은 게이트웨이)에서 설정합니다.

| 설정 | 설명 |
|---------|-------------|
| `GATEWAY_PROXY_URL` | 원격 Hermes API 서버의 URL(예: `http://192.168.1.100:8642`) |
| `GATEWAY_PROXY_KEY` | 인증용 Bearer 토큰(호스트의 `API_SERVER_KEY`와 일치해야 함) |
| `gateway.proxy_url` | `GATEWAY_PROXY_URL`과 동일하지만 `config.yaml`에서 설정 |

호스트 측에는 다음이 필요합니다.

| 설정 | 설명 |
|---------|-------------|
| `API_SERVER_ENABLED` | `true`로 설정 |
| `API_SERVER_KEY` | Bearer 토큰(컨테이너와 공유) |
| `API_SERVER_HOST` | 네트워크 접근을 위해 `0.0.0.0`으로 설정 |
| `API_SERVER_PORT` | 포트 번호(기본값: `8642`) |

### 모든 플랫폼에서 작동

프록시 모드는 Matrix에만 한정되지 않습니다. 모든 플랫폼 어댑터가 사용할 수 있습니다. 모든 게이트웨이 인스턴스에서 `GATEWAY_PROXY_URL`을 설정하면 로컬에서 에이전트를 실행하는 대신 원격 에이전트로 전달합니다. 플랫폼 어댑터를 에이전트와 다른 환경에서 실행해야 하는 배포(네트워크 격리, E2EE 요구 사항, 리소스 제약 등)에 유용합니다.

:::tip
세션 연속성은 `X-Hermes-Session-Id` 헤더로 유지됩니다. 호스트의 API 서버는 이 ID로 세션을 추적하므로 대화가 로컬 에이전트를 사용할 때처럼 메시지 간에 유지됩니다.
:::

:::note
**제한 사항(v1):** 원격 에이전트의 도구 진행률 메시지는 전달되지 않으므로 사용자는 개별 도구 호출이 아니라 스트리밍되는 최종 응답만 봅니다. 위험한 명령어 승인 프롬프트는 Matrix 사용자에게 전달되지 않고 호스트 측에서 처리됩니다. 향후 업데이트에서 해결할 수 있습니다.
:::

### 봇은 연결되어 메시지를 보내지만 수신 메시지를 무시함

**원인**: 동기화 페이로드가 mautrix의 `handle_sync()` 메커니즘을 통해 디스패치될 때만 Matrix 이벤트 핸들러가 실행됩니다. `handle_sync()`를 호출하지 않는 원시 `client.sync()` 폴링은 어댑터를 연결된 상태로 두면서(전송은 작동) 수신 메시지가 `_on_room_message`에 도달하지 못하게 할 수 있습니다.

**해결 방법**: Hermes는 초기 동기화와 모든 증분 동기화 응답에서 `client.handle_sync()`를 호출하는 명시적 동기화 루프를 사용합니다. 이는 업스트림 이슈 #7914와 종료된 PR #37807의 진단과 일치하지만, 전체 수명 주기를 `client.start()`에 위임하는 대신 Hermes 자체의 백그라운드 유지 관리 작업(참여 방 추적, 초대 처리, E2EE 키 공유)을 유지합니다. 게이트웨이를 다시 시작한 후에도 수신 메시지가 실패하면 첫 동기화 전에 핸들러가 등록되었는지 확인하고 `sync event dispatch error` 로그를 확인합니다.

### 동기화 문제 / 봇이 뒤처짐

**원인**: 장시간 실행되는 도구 작업이 동기화 루프를 지연시키거나 홈서버가 느립니다.

**해결 방법**: 동기화 루프는 오류 발생 시 5초마다 자동으로 재시도합니다. Hermes 로그에서 동기화 관련 경고를 확인합니다. 봇이 계속 뒤처지면 홈서버에 충분한 리소스가 있는지 확인합니다.

### 봇이 오프라인임

**원인**: Hermes 게이트웨이가 실행 중이 아니거나 연결에 실패했습니다.

**해결 방법**: `hermes gateway`가 실행 중인지 확인합니다. 터미널 출력에서 오류 메시지를 확인합니다. 일반적인 문제는 잘못된 홈서버 URL, 만료된 액세스 토큰, 홈서버에 접근할 수 없는 경우입니다.

### "User not allowed" / 봇이 나를 무시함

**원인**: 사용자 ID가 `MATRIX_ALLOWED_USERS`에 없습니다.

**해결 방법**: `~/.hermes/.env`의 `MATRIX_ALLOWED_USERS`에 사용자 ID를 추가하고 게이트웨이를 다시 시작합니다. 전체 `@user:server` 형식을 사용합니다.

### 봇이 방 전체를 무시함

**원인**: `MATRIX_ALLOWED_ROOMS`가 설정되어 있는데 현재 방 ID가 목록에 없거나, 방에서 멘션이 필요하지만 메시지에 봇 멘션이 없습니다.

**해결 방법**: 방 ID를 `MATRIX_ALLOWED_ROOMS`에 추가하거나 개인 배포라면 방 허용 목록을 제거합니다. Element에서 방 ID를 찾으려면 방 설정을 열고 **고급**을 확인합니다.

### 브리지 메시지가 반복되거나 에코됨

**원인**: 브리지/appservice 퍼펫이 봇 출력을 새 사용자 메시지로 다시 전달하거나, 브리지가 비표준 고스트 사용자 ID를 사용합니다.

**해결 방법**: 브리지 고스트를 `MATRIX_ALLOWED_USERS`에서 제외하고, 일치하는 `MATRIX_IGNORE_USER_PATTERNS` 항목을 추가하며, 알림이 신뢰할 수 있는 워크플로의 일부가 아닌 한 `MATRIX_PROCESS_NOTICES=false`로 유지합니다.

## 보안

:::warning
항상 `MATRIX_ALLOWED_USERS`를 설정하고, 공유/비공개 배포에서는 `MATRIX_ALLOWED_ROOMS`도 설정하세요. 이 값이 없으면 참여한 방에서 봇에 메시지를 보낼 수 있는 누구나 에이전트를 트리거할 수 있습니다. 신뢰하는 사람과 방만 승인하세요. 승인된 사용자는 도구 사용과 시스템 접근을 포함하여 에이전트의 모든 기능에 접근할 수 있습니다.
:::

Hermes Agent 배포 보안에 대한 자세한 내용은 [보안 가이드](../security.md)를 참조하세요.

## 참고

- **모든 홈서버**: Synapse, Conduit, Dendrite, matrix.org 또는 사양을 준수하는 모든 Matrix 홈서버에서 작동합니다. 특정 홈서버 소프트웨어가 필요하지 않습니다.
- **연합**: 연합 홈서버를 사용하는 경우 다른 서버의 사용자와 통신할 수 있습니다. 전체 `@user:server` ID를 `MATRIX_ALLOWED_USERS`에 추가하기만 하면 됩니다.
- **자동 참여**: 봇은 방 초대를 자동으로 수락하고 참여합니다. 참여한 직후 응답을 시작합니다.
- **미디어 지원**: Hermes는 이미지, 오디오, 비디오 및 파일 첨부를 보내고 받을 수 있습니다. Matrix 콘텐츠 저장소 API를 사용하여 홈서버에 미디어를 업로드합니다.
- **네이티브 음성 메시지(MSC3245)**: Matrix 어댑터는 발신 음성 메시지에 `org.matrix.msc3245.voice` 플래그를 자동으로 표시합니다. 따라서 TTS 응답과 음성 오디오는 일반 오디오 파일 첨부가 아니라 Element 및 MSC3245를 지원하는 다른 클라이언트에서 **네이티브 음성 말풍선**으로 렌더링됩니다. MSC3245 플래그가 있는 수신 음성 메시지도 올바르게 식별되어 음성-텍스트 변환으로 전달됩니다. 설정이 필요하지 않으며 자동으로 작동합니다.
