---
sidebar_position: 9
title: "Matrix"
description: "Hermes Agent를 Matrix 봇으로 설정하기"
---

# Matrix 설정

Hermes Agent는 개방형 탈중앙화 연합(federated) 메시징 프로토콜인 Matrix와 통합됩니다. Matrix를 사용하면 자체 홈서버를 운영하거나 matrix.org와 같은 공개 서버를 사용할 수 있으며 — 어느 쪽이든 커뮤니케이션에 대한 통제권을 유지할 수 있습니다. 봇은 `mautrix` Python SDK를 통해 연결하고, Hermes Agent 파이프라인(도구 사용, 기억력, 추론 기능 포함)을 통해 메시지를 처리하며, 실시간으로 응답합니다. 텍스트, 파일 첨부, 이미지, 오디오, 비디오를 지원하며 종단간 암호화(E2EE)를 선택적으로 사용할 수 있습니다.

Hermes는 Synapse, Conduit, Dendrite 또는 matrix.org 등 어떠한 Matrix 홈서버와도 작동합니다.

설정하기 전에 대부분의 사람들이 가장 알고 싶어 하는 부분인 Hermes가 연결된 후 어떻게 작동하는지에 대해 알아봅시다.

## Hermes의 동작 방식

| 컨텍스트 | 동작 방식 |
|---------|----------|
| **DM** | Hermes는 모든 메시지에 응답합니다. `@mention`이 필요하지 않습니다. 각 DM은 고유한 세션을 갖습니다. 봇이 DM에서 `@mention`될 때 스레드를 시작하려면 `MATRIX_DM_MENTION_THREADS=true`로 설정하세요. |
| **방 (Rooms)** | 기본적으로 Hermes는 응답하기 위해 `@mention`을 요구합니다. 멘션 없이 응답하는 방으로 만들려면 `MATRIX_REQUIRE_MENTION=false`를 설정하거나 `MATRIX_FREE_RESPONSE_ROOMS`에 방 ID를 추가하세요. 방 초대는 자동으로 수락됩니다. |
| **스레드 (Threads)** | Hermes는 Matrix 스레드(MSC3440)를 지원합니다. 스레드에서 답장하면 Hermes는 메인 방 타임라인과 스레드 컨텍스트를 분리하여 유지합니다. 봇이 이미 참여한 스레드에서는 멘션이 필요하지 않습니다. |
| **자동 스레딩** | 기본적으로 Hermes는 방에서 응답하는 각 메시지에 대해 자동으로 스레드를 생성합니다. 이렇게 하면 대화가 분리된 상태로 유지됩니다. 비활성화하려면 `MATRIX_AUTO_THREAD=false`로 설정하세요. |
| **명령어** | Hermes는 사용자의 Matrix 클라이언트가 전송할 때 일반적인 `/commands`를 수용합니다. 귀하의 클라이언트가 로컬 명령어를 위해 `/`를 예약해 둔 경우 `!commands`를 대신 사용하십시오. Hermes는 알려진 `!command` 별칭(alias)들을 `/command`로 정규화합니다. |
| **다수 사용자가 있는 공유 방** | 기본적으로 Hermes는 방 내부의 각 사용자 단위로 세션 내역을 분리합니다. 명시적으로 비활성화하지 않는 한, 같은 방에서 대화하는 두 사람은 하나의 대화 내용을 공유하지 않습니다. |

:::tip
봇은 초대받으면 자동으로 방에 참여합니다. 그냥 봇의 Matrix 유저를 아무 방에나 초대하면 참여해서 응답하기 시작할 것입니다.
:::

### Matrix에서의 세션 모델

기본적으로:

- 각 DM은 고유한 세션을 가집니다.
- 각 스레드는 고유한 세션 네임스페이스를 가집니다.
- 공유된 방의 각 사용자는 그 방 안에서 그들만의 세션을 가집니다.

이는 `config.yaml`에 의해 제어됩니다:

```yaml
group_sessions_per_user: true
```

방 전체를 위한 하나의 공유된 대화를 명시적으로 원할 때만 `false`로 설정하십시오:

```yaml
group_sessions_per_user: false
```

공유 세션은 협업을 위한 방에서는 유용할 수 있지만, 다음과 같은 의미이기도 합니다:

- 사용자들은 컨텍스트 길이와 토큰 비용을 공유하게 됩니다.
- 한 사람이 도구를 무겁게 사용하는 긴 작업을 시키면 다른 사람들의 컨텍스트까지 비대하게 만듭니다.
- 한 사람의 실행 중인 작업이 같은 방에 있는 다른 사람의 후속 메시지로 인해 중단될 수 있습니다.

### 멘션 및 스레딩 구성

환경 변수나 `config.yaml`을 통해 멘션 및 자동 스레딩 동작을 구성할 수 있습니다:

```yaml
matrix:
  require_mention: true           # 방에서 @mention 요구 (기본값: true)
  free_response_rooms:            # 멘션 요구에서 면제되는 방
    - "!abc123:matrix.org"
  auto_thread: true               # 응답을 위해 자동으로 스레드 생성 (기본값: true)
  dm_mention_threads: false       # DM에서 @mention 될 때 스레드 생성 (기본값: false)
```

또는 환경 변수를 통해:

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_REACTIONS=true          # 기본값: true — 처리 과정 중 이모지 리액션 표시
```

:::tip 리액션 끄기
`MATRIX_REACTIONS=false`는 봇이 인바운드 메시지에 게시하는 처리 주기 이모지 리액션(👀/✅/❌)을 끕니다. 리액션 이벤트가 시끄럽거나 모든 참여 클라이언트에서 지원하지 않는 방에 유용합니다.
:::

:::note
`MATRIX_REQUIRE_MENTION` 설정이 없던 이전 버전에서 업그레이드하는 경우, 과거에 봇은 방에 있는 모든 메시지에 응답했습니다. 그 동작을 유지하려면 `MATRIX_REQUIRE_MENTION=false`로 설정하십시오.
:::

이 가이드는 봇 계정 생성부터 첫 메시지 전송까지의 전체 설정 과정을 안내합니다.

## 1단계: 봇 계정 생성

봇을 위한 Matrix 사용자 계정이 필요합니다. 몇 가지 방법이 있습니다:

### 옵션 A: 자체 홈서버에 등록 (권장)

자체 홈서버(Synapse, Conduit, Dendrite)를 운영하는 경우:

1. 관리자 API나 등록 도구를 사용하여 새 사용자를 생성합니다:

```bash
# Synapse 예시
register_new_matrix_user -c /etc/synapse/homeserver.yaml http://localhost:8008
```

2. `hermes`와 같은 사용자 이름을 선택합니다 — 전체 사용자 ID는 `@hermes:your-server.org`가 됩니다.

### 옵션 B: matrix.org나 다른 공개 홈서버 사용

1. [Element Web](https://app.element.io)으로 이동하여 새 계정을 생성합니다.
2. 봇의 사용자 이름을 선택합니다 (예: `hermes-bot`).

### 옵션 C: 본인의 계정 사용

자신의 사용자로 Hermes를 실행할 수도 있습니다. 즉, 봇이 여러분으로서 게시물을 올리게 됩니다 — 개인 비서 용도로 유용합니다.

## 2단계: 액세스 토큰 얻기

Hermes는 홈서버와 인증할 액세스 토큰이 필요합니다. 두 가지 옵션이 있습니다:

### 옵션 A: 액세스 토큰 (권장)

토큰을 얻는 가장 확실한 방법:

**Element를 통해:**
1. 봇 계정으로 [Element](https://app.element.io)에 로그인합니다.
2. **Settings(설정)** → **Help & About(도움말 및 정보)**로 이동합니다.
3. 아래로 스크롤하여 **Advanced(고급)**를 확장합니다 — 거기에 액세스 토큰이 표시됩니다.
4. **즉시 복사하세요.**

**API를 통해:**

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "user": "@hermes:your-server.org",
    "password": "your-password"
  }'
```

응답에 `access_token` 필드가 포함됩니다 — 이를 복사하세요.

:::warning[액세스 토큰 보관 주의]
액세스 토큰은 봇의 Matrix 계정에 대한 완전한 접근 권한을 제공합니다. 절대 공개적으로 공유하거나 Git에 커밋하지 마세요. 유출된 경우 해당 사용자의 모든 세션을 로그아웃시켜 토큰을 폐기(revoke)하십시오.
:::

### 옵션 B: 비밀번호 로그인

액세스 토큰을 제공하는 대신, Hermes에게 봇의 사용자 ID와 비밀번호를 부여할 수 있습니다. Hermes는 시작 시 자동으로 로그인할 것입니다. 이는 더 간단하지만 `.env` 파일에 비밀번호가 저장됨을 의미합니다.

```bash
MATRIX_USER_ID=@hermes:your-server.org
MATRIX_PASSWORD=your-password
```

## 3단계: 본인의 Matrix 사용자 ID 찾기

Hermes Agent는 여러분의 Matrix User ID를 사용하여 누가 봇과 상호작용할 수 있는지 제어합니다. Matrix User ID는 `@username:server` 형식을 따릅니다.

본인의 ID를 찾으려면:

1. [Element](https://app.element.io) (또는 선호하는 Matrix 클라이언트)를 엽니다.
2. 본인의 아바타 → **Settings(설정)**을 클릭합니다.
3. 사용자 ID는 프로필 상단에 표시됩니다 (예: `@alice:matrix.org`).

:::tip
Matrix 사용자 ID는 항상 `@`로 시작하며 서버 이름 앞에 `:`가 포함되어 있습니다. 예: `@alice:matrix.org`, `@bob:your-server.com`.
:::

## 4단계: Hermes Agent 구성

### 옵션 A: 대화형 설정 (권장)

안내형 설정 명령을 실행합니다:

```bash
hermes gateway setup
```

프롬프트가 나타나면 **Matrix**를 선택한 후 홈서버 URL, 액세스 토큰(또는 사용자 ID + 비밀번호), 그리고 허용할 사용자 ID들을 입력하십시오.

### 옵션 B: 수동 구성

`~/.hermes/.env` 파일에 다음을 추가합니다:

**액세스 토큰 사용 시:**

```bash
# 필수
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# 선택 사항: 사용자 ID (생략 시 토큰에서 자동 감지됨)
# MATRIX_USER_ID=@hermes:matrix.example.org

# 보안: 봇과 상호작용할 수 있는 사람 제한
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# 다수의 허용된 사용자 (쉼표로 구분)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**비밀번호 로그인 사용 시:**

```bash
# 필수
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# 보안
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

`~/.hermes/config.yaml`의 선택적 동작 설정:

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true`는 공유 방 내부에서 각 참가자의 컨텍스트를 분리된 상태로 유지합니다.

### 게이트웨이 시작

구성이 완료되면 Matrix 게이트웨이를 시작합니다:

```bash
hermes gateway
```

봇은 홈서버에 연결하고 몇 초 내로 동기화를 시작해야 합니다. 테스트하려면 봇에게 DM을 보내거나 봇이 참여한 방에서 메시지를 보내보세요.

:::tip
지속적인 운영을 위해 `hermes gateway`를 백그라운드나 systemd 서비스로 실행할 수 있습니다. 자세한 내용은 배포 문서를 참조하세요.
:::

## 종단간 암호화 (E2EE)

Hermes는 Matrix의 종단간 암호화를 지원하므로, 암호화된 방에서 봇과 채팅할 수 있습니다.

### 요구 사항

E2EE를 위해서는 암호화 부가 기능이 포함된 `mautrix` 라이브러리와 `libolm` C 라이브러리가 필요합니다:

```bash
# E2EE 지원을 포함하여 mautrix 설치
pip install 'mautrix[encryption]'

# 또는 hermes 엑스트라와 함께 설치
pip install 'hermes-agent[matrix]'
```

시스템에 `libolm`도 설치되어 있어야 합니다:

```bash
# Debian/Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm

# Fedora
sudo dnf install libolm-devel
```

### E2EE 활성화

`~/.hermes/.env`에 추가하세요:

```bash
MATRIX_ENCRYPTION=true
```

E2EE가 활성화되면 Hermes는 다음을 수행합니다:

- 암호화 키를 `~/.hermes/platforms/matrix/store/`에 저장 (이전 설치판은 `~/.hermes/matrix/store/`)
- 첫 연결 시 기기 키(device keys)를 업로드
- 수신 메시지는 자동으로 복호화, 발신 메시지는 자동으로 암호화
- 암호화된 방에 초대받으면 자동 참여

### 교차 서명 검증 (권장)

Matrix 계정에 교차 서명(cross-signing)이 활성화되어 있다면(Element에서는 기본값), 복구 키를 설정하여 봇이 시작 시 자체적으로 자기 기기에 서명할 수 있도록 하십시오. 이것이 없으면 다른 Matrix 클라이언트들은 기기 키가 순환(rotation)된 후 봇과 암호화 세션을 공유하는 것을 거부할 수 있습니다.

```bash
MATRIX_RECOVERY_KEY=EsT... 여기에 복구 키 입력
```

**어디서 찾나요:** Element의 **Settings(설정)** → **Security & Privacy(보안 및 프라이버시)** → **Encryption(암호화)** → 복구 키(또는 "보안 키"라고도 함)로 이동합니다. 이는 교차 서명을 처음 설정할 때 저장하라고 안내받았던 그 키입니다.

매 시작 시 `MATRIX_RECOVERY_KEY`가 설정되어 있으면, Hermes는 홈서버의 안전한 비밀 저장소(secure secret storage)에서 교차 서명 키를 가져와 현재 기기에 서명합니다. 이 작업은 멱등성(idempotent)을 가지므로 영구적으로 활성화해 두어도 안전합니다.

:::warning[암호화 저장소 삭제 주의]
만약 `~/.hermes/platforms/matrix/store/crypto.db`를 삭제한다면, 봇은 자신의 암호화 신원(identity)을 잃게 됩니다. 이전과 동일한 기기 ID(device ID)로 그냥 재시작하는 것만으로는 완전하게 복구되지 **않습니다** — 홈서버는 여전히 과거 신원 키(identity key)로 서명된 일회용 키들을 붙들고 있으며, 피어 클라이언트들은 새로운 Olm 세션을 맺지 못하게 됩니다.

Hermes는 시작 시 이 상태를 감지하면 로그에 `device XXXX has stale one-time keys on the server signed with a previous identity key`라는 메시지를 남기고 E2EE 활성화를 거부합니다.

**가장 쉬운 복구 방법: 새 액세스 토큰 생성** (이전 키 내역이 없는 완전히 새로운 기기 ID를 발급받습니다). 아래 "E2EE가 설정된 이전 버전에서 업그레이드하기" 섹션을 참고하십시오. 홈서버 데이터베이스를 건드리지 않는 가장 확실한 방법입니다.

**수동 복구 방법** (고급 사용자용 — 동일한 기기 ID를 유지합니다):

1. Synapse를 정지하고 데이터베이스에서 과거 기기 정보를 삭제합니다:
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
   또는 Synapse 관리자 API를 통해 (사용자 ID가 URL 인코딩되었음에 유의):
   ```bash
   curl -X DELETE -H "Authorization: Bearer ADMIN_TOKEN" \
     'https://your-server/_synapse/admin/v2/users/%40hermes%3Ayour-server/devices/DEVICE_ID'
   ```
   참고: 관리자 API로 기기를 삭제하면 연관된 액세스 토큰도 무효화될 수 있습니다. 이후 새 토큰을 발급받아야 할 수 있습니다.

2. 로컬 암호화 저장소를 지우고 Hermes를 다시 시작합니다:
   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db*
   # restart hermes
   ```

다른 Matrix 클라이언트(Element, matrix-commander 등)가 과거 기기 키를 캐시하고 있을 수 있습니다. 복구가 끝난 후 Element 채팅창에서 `/discardsession`을 쳐서 봇과의 새 암호화 세션을 강제로 맺어주십시오.
:::

:::info
만약 `mautrix[encryption]`이 설치되지 않았거나 `libolm`이 없는 경우, 봇은 자동으로 평문(비암호화) 클라이언트로 전환(fallback)됩니다. 로그에 경고 메시지가 나타날 것입니다.
:::

## 홈 룸 (Home Room)

봇이 능동적인 메시지(크론 작업 결과물, 리마인더, 알림 등)를 전송할 "홈 룸(home room)"을 지정할 수 있습니다. 두 가지 방법이 있습니다:

### 슬래시 명령어 사용

봇이 참여 중인 Matrix 방에서 `/sethome`을 입력하세요. 그 방이 홈 룸이 됩니다.
만약 여러분의 Matrix 클라이언트가 슬래시 명령어를 가로챈다면 `!sethome`을 대신 사용하세요.

### 수동 구성

`~/.hermes/.env`에 다음을 추가하세요:

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

## 방 허용 목록 (`allowed_rooms`)

봇을 정해진 Matrix 방들에만 상주하도록 제한합니다. 이 값이 설정되면 봇은 목록에 있는 ID의 방에서**만** 답변합니다 — 봇이 호출(mention)되더라도 다른 모든 방의 메시지는 조용히 무시합니다.

**DM(일대일 채팅방)은 이 필터에서 제외되므로**, 승인된 사용자는 언제든 봇에게 일대일로 연락할 수 있습니다.

```yaml
matrix:
  allowed_rooms:
    - "!abc123def456:matrix.example.org"
    - "!opsroom789:matrix.example.org"
```

또는 환경 변수(쉼표 구분)를 통해:

```bash
MATRIX_ALLOWED_ROOMS="!abc123def456:matrix.example.org,!opsroom789:matrix.example.org"
```

동작 원리:

- 값이 없거나 설정하지 않음 → 제한 없음 (기본값).
- 값이 있음 → 방 ID가 반드시 목록에 있어야 합니다. 이 검사는 다른 모든 관문(멘션 요구, 발신자 허용 목록 등) **이전에** 수행됩니다.
- 방의 별칭(alias, `#room:server`)이 아닌 **내부 ID(internal ID)** (`!abc...:server`)를 사용해야 합니다. Element의 경우 방 → Settings(설정) → Advanced(고급)에서 방의 내부 ID를 확인할 수 있습니다.

같이 보기: [관리자/사용자 슬래시 권한 분할](../../reference/slash-commands.md#permissions-and-adminuser-split).

:::tip
방 ID(Room ID) 찾는 법: Element에서 방으로 이동 → **Settings(설정)** → **Advanced(고급)** → 거기에 **Internal room ID**(내부 방 ID)가 표시되어 있습니다 (`!`로 시작함).
:::

## Matrix 내 명령어

Hermes는 다른 메시징 플랫폼에서 지원하는 것과 동일한 게이트웨이 명령어를 Matrix에서도 지원합니다. 여기에는 `/commands`, `/model`, `/stop`, `/queue`, `/steer`, `/goal`, `/subgoal`, `/background`, `/bg`, `/btw`, `/tasks`, `/yolo` 등이 포함됩니다.

일부 Matrix 클라이언트는 앞에 오는 `/`를 로컬 클라이언트 명령어용으로 예약하여 알 수 없는 슬래시 명령어를 방으로 전송하지 않을 수 있습니다. 이 경우 Matrix에서 안전한 별칭인 `!`를 사용하세요:

```text
!commands
!model
!model gpt-5.5 --provider openrouter
!queue 다음 작업으로 계속 진행
!stop
```

Hermes는 해당 명령어가 게이트웨이에 알려진 명령어, 등록된 플러그인 명령어 또는 설치된 스킬 명령어일 때만 `!command`를 정규화합니다. `!important`와 같은 일반적인 느낌표 표현은 평범한 채팅 메시지로 남습니다.

## 문제 해결

### 봇이 메시지에 응답하지 않음

**원인**: 봇이 방에 참여하지 않았거나, `MATRIX_ALLOWED_USERS`에 귀하의 User ID가 포함되지 않았습니다.

**해결 방법**: 방에 봇을 초대하세요 — 초대받으면 자동으로 참여합니다. 귀하의 User ID가 `MATRIX_ALLOWED_USERS`에 있는지 확인하세요 (전체 `@user:server` 형식 사용). 게이트웨이를 다시 시작하세요.

### 봇이 방에는 들어오지만 모든 메시지를 조용히 무시함 (시계 오차)

**원인**: 호스트 시스템의 시계가 실제 시간보다 미래로 설정되어 있습니다. Matrix 어댑터는 초기 동기화(initial sync) 때 딸려오는 과거 이벤트들을 쳐내기 위해 5초의 시작-유예 필터(`event_ts < startup_ts - 5`)를 적용합니다. 만일 벽시계(wall clock)가 미래로 당겨져 있다면, 새로 들어오는 모든 이벤트조차 "시작 시점보다 오래된" 것으로 보여 메시지 핸들러에 닿기도 전에 버려집니다 — 봇은 잘 켜져 있는데도 절대로 대답하지 않습니다. [#12614](https://github.com/NousResearch/hermes-agent/issues/12614) 이슈를 참고하세요.

**증상**: 게이트웨이 로그에 `Matrix: dropped N live events as 'too old' more than 30s after startup` 라는 메시지가 뜹니다.

**해결 방법**: 호스트 시계를 NTP와 동기화하고 봇을 재시작하세요:

```bash
# Debian/Ubuntu
sudo timedatectl set-ntp true
timedatectl status   # "System clock synchronized: yes" 확인

# macOS
sudo sntp -sS time.apple.com
```

### 시작 시 "Failed to authenticate" / "whoami failed" 오류 발생

**원인**: 액세스 토큰이나 홈서버 URL이 올바르지 않습니다.

**해결 방법**: `MATRIX_HOMESERVER`가 홈서버를 가리키는지 확인하세요 (`https://` 포함, 후행 슬래시 제외). `MATRIX_ACCESS_TOKEN`이 유효한지 확인하세요 — curl로 시도해 보세요:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/_matrix/client/v3/account/whoami
```

이것이 귀하의 사용자 정보를 반환한다면 토큰은 유효합니다. 오류를 반환한다면 새 토큰을 생성하세요.

### "mautrix not installed" 오류

**원인**: `mautrix` Python 패키지가 설치되지 않았습니다.

**해결 방법**: 설치하세요:

```bash
pip install 'mautrix[encryption]'
```

또는 Hermes 엑스트라와 함께:

```bash
pip install 'hermes-agent[matrix]'
```

### 암호화 오류 / "could not decrypt event"

**원인**: 암호화 키 누락, `libolm` 미설치, 또는 봇의 기기가 신뢰되지 않았습니다.

**해결 방법**:
1. 시스템에 `libolm`이 설치되어 있는지 검증하세요 (위의 E2EE 섹션 참조).
2. `.env`에 `MATRIX_ENCRYPTION=true`가 설정되어 있는지 확인하세요.
3. Matrix 클라이언트(Element)에서 봇의 프로필 -> Sessions(세션)으로 이동하여 봇의 기기를 검증/신뢰(verify/trust)하세요.
4. 봇이 암호화된 방에 방금 참여했다면, 참여한 *이후*에 보낸 메시지만 복호화할 수 있습니다. 이전 메시지에는 접근할 수 없습니다.

### E2EE가 설정된 이전 버전에서 업그레이드하기

:::tip
만약 `crypto.db`마저 수동으로 삭제하셨다면, 위 E2EE 섹션에 있는 "암호화 저장소 삭제 주의" 경고문을 꼭 참조하십시오 — 홈서버에서 낡은 일회용 키들을 치워야 하는 추가 과정이 필요합니다.
:::

만약 이전 버전의 Hermes에서 `MATRIX_ENCRYPTION=true`를 사용하다가 새로운 SQLite 기반의 암호화 저장소(crypto store)를 쓰는 최신 버전으로 업그레이드하셨다면, 봇의 암호화 신원(identity)이 변경되었을 것입니다. 여러분의 Matrix 클라이언트(Element 등)는 낡은 기기 키(device keys)를 기억하고 있다가 봇과 암호화 세션 맺기를 거부할 수 있습니다.

**증상**: 봇은 잘 연결되고 로그에도 "E2EE enabled"라고 뜨지만, 모든 채팅창 메시지가 "could not decrypt event"라고 표기되며 봇이 절대 답변하지 않습니다.

**원인**: 예전 암호화 상태(과거의 `matrix-nio` 나 직렬화(serialization) 기반의 `mautrix` 백엔드)가 새로운 SQLite 암호화 저장소와 호환되지 않습니다. 봇은 깨끗한 새 암호화 신원을 발급받았지만, 여러분의 Matrix 클라이언트는 여전히 과거의 키들을 캐시해둔 상태이므로, "키가 바뀌어버린" 기기와는 방의 암호화 세션을 공유하지 않으려 듭니다. 이는 잠재적 보안 사고(해킹 등)를 막기 위한 Matrix의 정상적인 보안 방어 기제입니다.

**해결 방법** (1회성 마이그레이션):

1. **새로운 기기 ID를 얻기 위해 새 액세스 토큰을 발급받습니다**. 가장 간단한 방법은 이렇습니다:

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

   새 `access_token`을 복사하여 `~/.hermes/.env`의 `MATRIX_ACCESS_TOKEN`을 교체합니다.

2. **오래된 암호화 상태 삭제**:

   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db
   rm -f ~/.hermes/platforms/matrix/store/crypto_store.*
   ```

3. **복구 키(recovery key) 설정** (교차 서명을 켜 두었다면 — 대부분의 Element 유저가 켜져 있습니다). `~/.hermes/.env` 에 추가:

   ```bash
   MATRIX_RECOVERY_KEY=EsT... 여기에 복구 키 입력
   ```

   이는 봇이 기동될 때 교차 서명 키(cross-signing keys)로 셀프 서명을 하게 만들어, Element가 이 새로운 기기를 즉각 신뢰하게 만들어줍니다. 이것이 없으면 Element는 새 기기를 "확인되지 않은 기기"로 간주하여 암호화 세션 공유를 거부할 수 있습니다. Element의 **Settings(설정)** → **Security & Privacy(보안 및 프라이버시)** → **Encryption(암호화)** 메뉴에서 복구 키를 찾을 수 있습니다.

4. **Matrix 클라이언트에게 강제로 암호화 세션 갱신시키기**. Element에서 봇과의 1:1 대화방을 열고 `/discardsession` 을 입력하십시오. Element가 강제로 새 암호화 세션을 만들어 봇의 새 기기와 공유하게 됩니다.

5. **게이트웨이 재시작**:

   ```bash
   hermes gateway run
   ```

   `MATRIX_RECOVERY_KEY`가 제대로 설정되었다면, 로그에서 `Matrix: cross-signing verified via recovery key`를 보실 수 있습니다.

6. **새 메시지 전송**. 이제 봇이 정상적으로 메시지를 복호화하고 대답할 것입니다.

:::note
마이그레이션이 끝난 뒤라도 업그레이드 *이전에* 전송되었던 메시지들은 여전히 복호화되지 않습니다 — 과거의 키가 이미 사라졌기 때문입니다. 이 현상은 오직 과도기적인 것이며, 새로운 메시지는 모두 정상 작동합니다.
:::

:::tip
**신규 설치자는 해당 사항이 없습니다.** 이 마이그레이션은 오직 과거 버전의 Hermes에서 이미 E2EE를 잘 쓰고 있었던 분들의 업그레이드를 위한 것입니다.

**왜 새 액세스 토큰이 필요한가요?** 각각의 Matrix 액세스 토큰은 특정 기기 ID 하나에 결속됩니다. 암호화 키는 새로 바뀌었는데 기기 ID를 그대로 재사용하려 들면, 다른 Matrix 클라이언트들은 그 기기를 불신하게 됩니다 (이를 보안 침해 시도로 봅니다). 반면 새 액세스 토큰을 발급받으면 과거 키 이력이 전혀 없는 뽀송뽀송한 새 기기 ID를 얻게 되므로, 다른 클라이언트들이 즉각 신뢰하게 됩니다.
:::

## 프록시 모드 (macOS에서의 E2EE)

Matrix E2EE는 `libolm`을 요구하는데, 이것은 macOS ARM64 (Apple Silicon)에서 컴파일되지 않습니다. 따라서 `hermes-agent[matrix]` 엑스트라 패키지는 오직 Linux 전용으로 막혀있습니다. 만약 당신이 macOS 사용자라면, 프록시 모드를 활용하여 Linux VM(혹은 Docker 컨테이너) 안에서 E2EE 통신을 처리하고, 실제 에이전트는 당신의 파일, 메모리, 스킬을 다 쓸 수 있도록 macOS 네이티브로 돌릴 수 있습니다.

### 작동 원리

```
macOS (호스트):
  └─ hermes gateway
       ├─ api_server 어댑터 ← 0.0.0.0:8642 에서 리스닝
       ├─ AIAgent ← 단일 진실 공급원 (single source of truth)
       ├─ 세션, 메모리, 스킬들
       └─ 로컬 파일 접근 (Obsidian, 각종 프로젝트 등)

Linux VM (Docker):
  └─ hermes gateway (프록시 모드)
       ├─ Matrix 어댑터 ← E2EE 복호화/암호화
       └─ HTTP 포워드 → macOS:8642/v1/chat/completions
           (LLM API 키 없음, 에이전트 없음, 추론 작업 없음)
```

Docker 컨테이너는 오직 Matrix 프로토콜과 E2EE만 처리합니다. 메시지가 오면 복호화한 뒤 그 텍스트를 표준 HTTP 요청에 담아 호스트로 넘깁니다(forward). 호스트가 에이전트를 돌리고 도구를 호출하며, 답변을 만들어 스트리밍으로 돌려줍니다. 그러면 컨테이너가 그것을 암호화해 Matrix로 보냅니다. 모든 세션은 통합됩니다 — CLI, Matrix, Telegram 그리고 다른 어떤 플랫폼이라도 같은 기억력과 대화 히스토리를 공유합니다.

### 1단계: 호스트 설정 (macOS)

호스트가 Docker 컨테이너로부터 들어오는 요청을 받아들이도록 API 서버를 켭니다.

`~/.hermes/.env` 에 추가:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=여기에_당신의_비밀_키
API_SERVER_HOST=0.0.0.0
```

- `API_SERVER_HOST=0.0.0.0`은 Docker 컨테이너가 접근할 수 있도록 모든 인터페이스를 개방합니다.
- 로컬 호스트(loopback) 이외의 바인딩을 위해서는 `API_SERVER_KEY`가 필수입니다. 강력한 임의의 문자열을 선택하세요.
- API 서버는 기본적으로 포트 8642로 돕니다 (필요시 `API_SERVER_PORT`로 바꿀 수 있습니다).

게이트웨이 시작:

```bash
hermes gateway
```

당신이 세팅해둔 다른 플랫폼들과 함께 API 서버가 구동되는 걸 볼 수 있을 것입니다. VM에서 접근이 가능한지 확인하십시오:

```bash
# Linux VM 터미널에서
curl http://<mac-ip>:8642/health
```

### 2단계: Docker 컨테이너 설정 (Linux VM)

컨테이너는 Matrix 자격 증명과 프록시 URL을 필요로 합니다. LLM API 키는 **필요 없습니다**.

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

      # 프록시 모드 — 호스트의 에이전트로 포워드
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "여기에_당신의_비밀_키"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

**`Dockerfile`:**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libolm-dev && rm -rf /var/lib/apt/lists/*
RUN pip install 'hermes-agent[matrix]'

CMD ["hermes", "gateway"]
```

컨테이너 세팅은 이게 끝입니다. OpenRouter, Anthropic 혹은 다른 어떤 추론 엔진을 위한 API 키도 들어가지 않습니다.

### 3단계: 양쪽 구동

1. 호스트 게이트웨이를 먼저 구동합니다:
   ```bash
   hermes gateway
   ```

2. Docker 컨테이너를 구동합니다:
   ```bash
   docker compose up -d
   ```

3. 암호화된 Matrix 방에서 메시지를 보내보십시오. 컨테이너가 복호화하고, 호스트로 포워드하며, 답변을 스트리밍으로 되돌려 받습니다.

### 설정값 요약 (Configuration Reference)

프록시 모드 설정은 **컨테이너 측**(가벼운 게이트웨이)에 합니다:

| 설정값 | 설명 |
|---------|-------------|
| `GATEWAY_PROXY_URL` | 원격 Hermes API 서버의 URL (예: `http://192.168.1.100:8642`) |
| `GATEWAY_PROXY_KEY` | 인증을 위한 Bearer 토큰 (호스트 측 `API_SERVER_KEY`와 동일해야 함) |
| `gateway.proxy_url` | `config.yaml` 상에서의 `GATEWAY_PROXY_URL` |

호스트 측에 필요한 설정:

| 설정값 | 설명 |
|---------|-------------|
| `API_SERVER_ENABLED` | `true` 로 설정 |
| `API_SERVER_KEY` | Bearer 토큰 (컨테이너와 맞춤) |
| `API_SERVER_HOST` | 네트워크 접근을 위해 `0.0.0.0` 으로 지정 |
| `API_SERVER_PORT` | 포트 넘버 (기본값: `8642`) |

### 다른 플랫폼에서도 사용 가능

프록시 모드는 Matrix 전용이 아닙니다. 어떤 플랫폼의 어댑터라도 사용할 수 있습니다 — 임의의 게이트웨이에서 `GATEWAY_PROXY_URL`만 잡아주면 로컬에서 에이전트를 돌리는 대신 그 원격 에이전트로 작업을 던집니다. 이는 플랫폼 어댑터가 실제 에이전트와 물리적으로/환경적으로 격리되어야 할 때(망 분리, E2EE 제약, 리소스 부족 등) 매우 유용합니다.

:::tip
세션 연속성은 `X-Hermes-Session-Id` 헤더를 통해 유지됩니다. 호스트의 API 서버는 이 ID를 바탕으로 세션을 추적하므로, 로컬 에이전트를 쓸 때와 완전히 동일하게 메시지들 간의 대화 흐름(context)이 이어집니다.
:::

:::note
**제한 사항 (v1):** 원격 에이전트가 무슨 도구를 쓰고 있는지에 대한 "진행 상태 메시지"는 아직 중계(relay)되지 않습니다 — 사용자는 개별적인 툴 호출이 아니라 완성된 스트리밍 최종 답변만 보게 됩니다. 또한 위험한 명령어에 대한 "승인 프롬프트(approval prompt)" 처리 역시 Matrix 채팅창으로 넘어오지 않고 호스트 쪽에서만 돕니다. 향후 업데이트에서 개선될 예정입니다.
:::

### 동기화(Sync) 문제 / 봇이 반응을 못 따라옴

**원인**: 도구 실행 시간이 길어지면 동기화 루프가 지연될 수 있거나 홈서버가 느린 경우입니다.

**해결 방법**: 동기화 루프는 에러 시 5초마다 자동 재시도합니다. Hermes 로그에서 동기화 관련 경고가 있는지 확인하세요. 지속적으로 봇이 뒤처지면 홈서버의 리소스가 충분한지 확인하세요.

### 봇이 오프라인임

**원인**: Hermes 게이트웨이가 실행 중이 아니거나 연결에 실패했습니다.

**해결 방법**: `hermes gateway`가 실행 중인지 확인하세요. 터미널 출력에서 오류 메시지를 확인하세요. 흔한 문제들: 잘못된 홈서버 URL, 만료된 액세스 토큰, 홈서버 접속 불가.

### "User not allowed" / 봇이 나를 무시함

**원인**: 당신의 사용자 ID가 `MATRIX_ALLOWED_USERS`에 없습니다.

**해결 방법**: 당신의 사용자 ID를 `~/.hermes/.env`의 `MATRIX_ALLOWED_USERS`에 추가하고 게이트웨이를 재시작하세요. 전체 `@user:server` 포맷을 써야 합니다.

## 보안

:::warning
봇과 상호작용할 수 있는 사람을 제한하기 위해 항상 `MATRIX_ALLOWED_USERS`를 설정하세요. 이를 설정하지 않으면 게이트웨이는 안전 조치로 기본적으로 모든 사용자의 접근을 차단합니다. 봇을 제어할 권한이 주어지면 봇이 다룰 수 있는 도구 및 시스템에 전체 접근이 가능해지므로, 전적으로 신뢰할 수 있는 사용자 ID만 등록하십시오.
:::

Hermes Agent 배포 보안에 관한 더 자세한 정보는 [보안 가이드](../security.md)를 참고하세요.

## 참고 사항

- **어떤 홈서버든 가능**: Synapse, Conduit, Dendrite, matrix.org 등 사양(spec)을 준수하는 모든 Matrix 홈서버에서 작동합니다. 특정 홈서버 소프트웨어가 필요하지 않습니다.
- **연합 (Federation)**: 연합된 홈서버에 있는 경우 봇이 다른 서버의 사용자들과 소통할 수 있습니다 — 그들의 전체 `@user:server` ID를 `MATRIX_ALLOWED_USERS`에 추가하기만 하면 됩니다.
- **자동 참여 (Auto-join)**: 봇은 방 초대를 수락하고 자동으로 참여합니다. 참여한 후 즉각적으로 응답을 시작합니다.
- **미디어 지원**: Hermes는 이미지, 오디오, 비디오, 그리고 파일 첨부 등 미디어를 수신하고 송신할 수 있습니다. 미디어 파일은 Matrix의 컨텐츠 저장소 API(content repository API)를 통해 홈서버로 업로드됩니다.
- **네이티브 음성 메시지 (MSC3245)**: Matrix 어댑터는 발신 음성 메시지에 자동으로 `org.matrix.msc3245.voice` 플래그를 태그합니다. 즉, TTS 응답이나 음성 데이터가 일반적인 오디오 파일 첨부가 아닌, Element나 MSC3245를 지원하는 다른 클라이언트에서 **네이티브 음성 버블(voice bubbles)** 형태로 렌더링됩니다. 또한 MSC3245 플래그가 붙어 들어온 인바운드 음성 메시지도 올바르게 감지되어 STT(Speech-to-Text) 전사기로 바로 라우팅됩니다. 어떠한 설정도 필요 없으며 자동 적용됩니다.
