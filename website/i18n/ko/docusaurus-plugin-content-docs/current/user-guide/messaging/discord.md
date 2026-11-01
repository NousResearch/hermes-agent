---
sidebar_position: 3
title: "Discord"
description: "Hermes Agent를 Discord 봇으로 설정하기"
---

# Discord 설정

Hermes Agent는 Discord와 봇으로 통합되므로, 다이렉트 메시지나 서버 채널을 통해 AI 어시스턴트와 대화할 수 있습니다. 봇은 메시지를 받아 도구 사용, 메모리, 추론을 포함한 Hermes Agent 파이프라인으로 처리한 뒤 실시간으로 응답합니다. 텍스트, 음성 메시지, 파일 첨부, 슬래시 명령을 지원합니다.

설정을 시작하기 전에, 대부분의 사람들이 가장 궁금해하는 부분부터 살펴보겠습니다. 봇이 서버에 들어온 뒤 어떻게 동작하는지 알아보세요.

## Hermes의 동작 방식

| 컨텍스트 | 동작 |
|---------|----------|
| **DM** | Hermes는 모든 메시지에 응답합니다. `@멘션`이 필요하지 않습니다. 각 DM은 별도의 세션을 가집니다. |
| **서버 채널** | 기본적으로 Hermes는 `@멘션`된 경우에만 응답합니다. 멘션 없이 채널에 게시하면 Hermes는 해당 메시지를 무시합니다. |
| **자유 응답 채널** | `DISCORD_FREE_RESPONSE_CHANNELS`로 특정 채널에서 멘션을 생략하도록 하거나, `DISCORD_REQUIRE_MENTION=false`로 전역 멘션 요구를 끌 수 있습니다. 이러한 채널의 메시지는 인라인으로 답변되며, 채널을 가벼운 채팅 공간으로 유지하기 위해 자동 스레드 생성은 건너뜁니다. |
| **스레드** | Hermes는 같은 스레드에서 답장합니다. 해당 스레드나 부모 채널이 자유 응답으로 설정되지 않은 한 멘션 규칙은 계속 적용됩니다. 세션 기록 측면에서 스레드는 부모 채널과 격리됩니다. |
| **여러 사용자가 있는 공유 채널** | 기본적으로 Hermes는 안전성과 명확성을 위해 채널 안에서 사용자별로 세션 기록을 격리합니다. 명시적으로 비활성화하지 않는 한, 같은 채널에서 대화하는 두 사람은 하나의 기록을 공유하지 않습니다. |
| **다른 사용자를 멘션하는 메시지** | `DISCORD_IGNORE_NO_MENTION`이 `true`(기본값)인 경우, 메시지가 다른 사용자를 `@멘션`하지만 봇은 멘션하지 않으면 Hermes는 조용히 있습니다. 이를 통해 다른 사람을 대상으로 한 대화에 봇이 끼어드는 것을 막습니다. 멘션된 사람과 관계없이 모든 메시지에 봇이 응답하게 하려면 `false`로 설정하세요. 이는 DM이 아닌 서버 채널에만 적용됩니다. |

:::tip
사람들이 매번 태그하지 않고 Hermes와 대화할 수 있는 일반적인 봇 도움말 채널을 원한다면 해당 채널을 `DISCORD_FREE_RESPONSE_CHANNELS`에 추가하세요.
:::

### Discord 게이트웨이 모델

Discord의 Hermes는 상태 없이 응답하는 웹훅이 아닙니다. 전체 메시징 게이트웨이를 통해 실행되므로, 수신되는 각 메시지는 다음 과정을 거칩니다.

1. 권한 부여(`DISCORD_ALLOWED_USERS`)
2. 멘션 / 자유 응답 검사
3. 세션 조회
4. 세션 기록 로드
5. 도구, 메모리, 슬래시 명령을 포함한 일반 Hermes 에이전트 실행
6. Discord로 응답 전달

따라서 바쁜 서버에서의 동작은 Discord 라우팅과 Hermes 세션 정책 모두에 좌우됩니다.

### Discord의 세션 모델

기본값은 다음과 같습니다.

- 각 DM은 자체 세션을 가집니다.
- 각 서버 스레드는 자체 세션 네임스페이스를 가집니다.
- 공유 채널의 각 사용자는 해당 채널 안에서 자체 세션을 가집니다.

따라서 Alice와 Bob이 모두 `#research`에서 Hermes와 대화하더라도, 눈에 보이는 Discord 채널은 같지만 Hermes는 기본적으로 이를 별도의 대화로 취급합니다.

이는 `config.yaml`로 제어합니다.

```yaml
group_sessions_per_user: true
```

방 전체에서 하나의 공유 대화를 명시적으로 원할 때만 `false`로 설정하세요.

```yaml
group_sessions_per_user: false
```

공유 세션은 협업용 방에 유용할 수 있지만 다음과 같은 의미도 있습니다.

- 사용자가 컨텍스트 증가와 토큰 비용을 공유합니다.
- 한 사람의 도구 사용이 많은 긴 작업이 다른 모든 사람의 컨텍스트를 부풀릴 수 있습니다.
- 한 사람의 진행 중인 실행이 같은 방에서 다른 사람의 후속 메시지를 중단할 수 있습니다.

### 중단과 동시성

Hermes는 세션 키로 실행 중인 에이전트를 추적합니다.

기본값인 `group_sessions_per_user: true`에서는 다음과 같습니다.

- Alice가 자신의 진행 중인 요청을 중단해도 해당 채널에 있는 Alice의 세션에만 영향을 줍니다.
- Bob은 Alice의 기록을 이어받거나 Alice의 실행을 중단하지 않고 같은 채널에서 계속 대화할 수 있습니다.

`group_sessions_per_user: false`에서는 다음과 같습니다.

- 방 전체가 해당 채널/스레드에서 하나의 실행 중 에이전트 슬롯을 공유합니다.
- 서로 다른 사람의 후속 메시지가 서로를 중단하거나 서로의 뒤에서 대기할 수 있습니다.

이 가이드는 Discord Developer Portal에서 봇을 만드는 단계부터 첫 메시지를 보내는 단계까지 전체 설정 과정을 안내합니다.

### 게이트웨이 WebSocket 상태

Discord REST와 Gateway WebSocket은 별도의 전송 방식입니다. REST 응답이 성공했다는 사실(예: `fetch_user()`가 HTTP 200을 반환한 경우)만으로는 봇이 여전히 Gateway 이벤트를 수신할 수 있다는 뜻이 아닙니다. 따라서 Hermes는 준비 상태, 클라이언트/소켓 종료 상태, 소켓 개방 여부, heartbeat ACK 경과 시간, 유한한 heartbeat 지연 시간을 함께 확인합니다.

설정된 연속 비정상 샘플 수에 도달하면 어댑터는 재시도 가능한 치명적 이벤트를 한 번 발생시킵니다. 기존 게이트웨이 재연결 감시자가 새 어댑터를 만들며, Discord 어댑터는 두 번째 무제한 재연결 루프를 시작하지 않습니다.

비밀이 아닌 임계값은 `config.yaml`에서 설정하세요.

```yaml
discord:
  websocket_liveness_interval_seconds: 15
  websocket_liveness_failure_threshold: 2
  websocket_heartbeat_ack_max_age_seconds: 60
  websocket_max_latency_seconds: 30
```

이전의 `liveness_interval_seconds` 및 `liveness_failure_threshold` 이름은 호환성을 위한 별칭으로만 남아 있으며, 더 이상 REST 프로빙을 의미하지 않습니다.

## 1단계: Discord 애플리케이션 만들기

1. [Discord Developer Portal](https://discord.com/developers/applications)로 이동해 Discord 계정으로 로그인합니다.
2. 오른쪽 상단에서 **New Application**을 클릭합니다.
3. 애플리케이션 이름(예: "Hermes Agent")을 입력하고 Developer Terms of Service에 동의합니다.
4. **Create**를 클릭합니다.

**General Information** 페이지가 열립니다. 나중에 초대 URL을 만들 때 필요하므로 **Application ID**를 기록해 두세요.

## 2단계: 봇 만들기

1. 왼쪽 사이드바에서 **Bot**을 클릭합니다.
2. Discord가 애플리케이션에 맞는 봇 사용자를 자동으로 만듭니다. 봇의 사용자 이름이 표시되며, 이를 사용자 지정할 수 있습니다.
3. **Authorization Flow**에서 다음을 설정합니다.
   - **Public Bot**을 **ON**으로 설정합니다. Discord가 제공하는 초대 링크를 사용하려면 필요하며(권장), Installation 탭에서 기본 인증 URL을 생성할 수 있습니다.
   - **Require OAuth2 Code Grant**는 **OFF**로 둡니다.

:::tip
이 페이지에서 봇의 사용자 지정 아바타와 배너를 설정할 수 있습니다. 사용자가 Discord에서 보게 되는 이미지입니다.
:::

:::info[비공개 봇 대안]
봇을 비공개로 유지하려면(Public Bot = OFF) Installation 탭 대신 5단계의 **Manual URL** 방식을 **반드시** 사용해야 합니다. Discord가 제공하는 링크는 Public Bot이 활성화되어 있어야 합니다.
:::

## 3단계: 권한이 있는 Gateway Intents 활성화

이것은 전체 설정에서 가장 중요한 단계입니다. 올바른 intent를 활성화하지 않으면 봇은 Discord에 연결되지만 **메시지 내용을 읽을 수 없습니다**.

**Bot** 페이지에서 **Privileged Gateway Intents**까지 아래로 스크롤합니다. 세 개의 토글이 표시됩니다.

| Intent | 목적 | 필수 여부 |
|--------|---------|-----------| 
| **Presence Intent** | 사용자의 온라인/오프라인 상태 확인 | 선택 사항 |
| **Server Members Intent** | 멤버 목록에 접근하고 사용자 이름 확인 | **필수** |
| **Message Content Intent** | 메시지의 텍스트 내용 읽기 | **필수** |

**Server Members Intent와 Message Content Intent를 모두 토글하여 **ON**으로 활성화하세요.**

- **Message Content Intent**가 없으면 봇은 메시지 이벤트를 받지만 메시지 텍스트가 비어 있습니다. 즉, 입력한 내용을 봇이 전혀 볼 수 없습니다.
- **Server Members Intent**가 없으면 봇은 허용된 사용자 목록의 사용자 이름을 확인할 수 없고, 누가 메시지를 보냈는지 식별하지 못할 수 있습니다.

:::warning[Discord 봇이 작동하지 않는 가장 큰 원인]
봇이 온라인인데 메시지에 전혀 응답하지 않는다면 **Message Content Intent**가 비활성화되어 있을 가능성이 매우 높습니다. [Developer Portal](https://discord.com/developers/applications)로 돌아가 애플리케이션 → Bot → Privileged Gateway Intents를 선택하고 **Message Content Intent**가 ON으로 설정되어 있는지 확인하세요. **Save Changes**를 클릭합니다.
:::

**서버 수에 관하여:**
- 봇이 **100개 미만의 서버**에 있다면 intent를 자유롭게 켜고 끌 수 있습니다.
- 봇이 **100개 이상의 서버**에 있다면 Discord는 권한이 있는 intent를 사용하기 위해 인증 신청을 요구합니다. 개인 용도라면 걱정할 필요가 없습니다.

페이지 하단에서 **Save Changes**를 클릭합니다.

## 4단계: 봇 토큰 받기

봇 토큰은 Hermes Agent가 봇으로 로그인할 때 사용하는 자격 증명입니다. 계속해서 **Bot** 페이지에서 다음을 수행합니다.

1. **Token** 섹션에서 **Reset Token**을 클릭합니다.
2. Discord 계정에서 2단계 인증을 활성화했다면 2FA 코드를 입력합니다.
3. Discord가 새 토큰을 표시합니다. **즉시 복사하세요.**

:::warning[토큰은 한 번만 표시됨]
토큰은 한 번만 표시됩니다. 잃어버리면 토큰을 재설정하고 새 토큰을 생성해야 합니다. 토큰을 공개적으로 공유하거나 Git에 커밋하지 마세요. 이 토큰을 가진 사람은 누구나 봇을 완전히 제어할 수 있습니다.
:::

토큰은 안전한 곳(예: 비밀번호 관리자)에 보관하세요. 8단계에서 필요합니다.

## 5단계: 초대 URL 생성

봇을 서버에 초대하려면 OAuth2 URL이 필요합니다. 두 가지 방법이 있습니다.

### 옵션 A: Installation 탭 사용(권장)

:::note[Public Bot 필요]
이 방법을 사용하려면 2단계에서 **Public Bot**이 **ON**으로 설정되어 있어야 합니다. Public Bot을 OFF로 설정했다면 아래의 Manual URL 방식을 사용하세요.
:::

1. 왼쪽 사이드바에서 **Installation**을 클릭합니다.
2. **Installation Contexts**에서 **Guild Install**을 활성화합니다.
3. **Install Link**에서 **Discord Provided Link**를 선택합니다.
4. Guild Install의 **Default Install Settings**에서 다음을 설정합니다.
   - **Scopes**: `bot`과 `applications.commands`를 선택합니다.
   - **Permissions**: 아래에 나열된 권한을 선택합니다.

### 옵션 B: Manual URL

다음 형식을 사용해 초대 URL을 직접 구성할 수 있습니다.

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274878286912
```

`YOUR_APP_ID`를 1단계의 Application ID로 바꿉니다.

### 필수 권한

봇에 필요한 최소 권한은 다음과 같습니다.

- **View Channels** — 봇이 접근할 수 있는 채널 보기
- **Send Messages** — 메시지에 응답
- **Embed Links** — 서식이 있는 응답 형식 지정
- **Attach Files** — 이미지, 오디오, 파일 결과 보내기
- **Read Message History** — 대화 컨텍스트 유지

### 권장 추가 권한

- **Send Messages in Threads** — 스레드 대화에 응답
- **Add Reactions** — 확인을 위해 메시지에 반응 추가

### 권한 정수

| 수준 | 권한 정수 | 포함 항목 |
|-------|-------------------|-----------------|
| 최소 | `117760` | View Channels, Send Messages, Read Message History, Attach Files |
| 권장 | `274878286912` | 위 항목과 Embed Links, Send Messages in Threads, Add Reactions |

## 6단계: 서버에 초대

1. 브라우저에서 초대 URL을 엽니다(Installation 탭에서 가져오거나 직접 만든 URL).
2. **Add to Server** 드롭다운에서 서버를 선택합니다.
3. **Continue**를 클릭한 다음 **Authorize**를 클릭합니다.
4. 메시지가 표시되면 CAPTCHA를 완료합니다.

:::info
봇을 초대하려면 Discord 서버에 대한 **Manage Server** 권한이 필요합니다. 드롭다운에 서버가 표시되지 않으면 서버 관리자에게 대신 초대 링크를 사용해 달라고 요청하세요.
:::

인증이 끝나면 봇이 서버의 멤버 목록에 나타납니다(Hermes 게이트웨이를 시작할 때까지 오프라인으로 표시됩니다).

## 7단계: Discord 사용자 ID 찾기

Hermes Agent는 Discord 사용자 ID를 사용해 봇과 상호작용할 수 있는 사람을 제어합니다. 다음과 같이 찾을 수 있습니다.

1. Discord(데스크톱 또는 웹 앱)를 엽니다.
2. **Settings** → **Advanced**로 이동해 **Developer Mode**를 **ON**으로 설정합니다.
3. 설정을 닫습니다.
4. 자신의 사용자 이름(메시지, 멤버 목록 또는 프로필에서)을 마우스 오른쪽 버튼으로 클릭한 뒤 **Copy User ID**를 선택합니다.

사용자 ID는 `284102345871466496`과 같은 긴 숫자입니다.

:::tip
Developer Mode를 사용하면 같은 방식으로 **Channel ID**와 **Server ID**도 복사할 수 있습니다. 채널을 수동으로 홈 채널로 설정하려면 Channel ID가 필요합니다.
:::

## 8단계: Hermes Agent 설정

### 옵션 A: 대화형 설정(권장)

안내형 설정 명령을 실행합니다.

```bash
hermes gateway setup
```

메시지가 표시되면 **Discord**를 선택하고 봇 토큰과 사용자 ID를 붙여 넣습니다.

### 옵션 B: 수동 설정

다음 내용을 `~/.hermes/.env` 파일에 추가합니다.

```bash
# Required
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# Multiple allowed users (comma-separated)
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

그런 다음 게이트웨이를 시작합니다.

```bash
hermes gateway
```

몇 초 안에 Discord에서 봇이 온라인 상태가 됩니다. DM이나 봇이 볼 수 있는 채널에서 메시지를 보내 테스트하세요.

:::tip
지속적으로 실행하려면 `hermes gateway`를 백그라운드나 systemd 서비스로 실행할 수 있습니다. 자세한 내용은 배포 문서를 참고하세요.
:::

## 설정 참고

Discord 동작은 두 파일로 제어합니다. 자격 증명과 환경 수준 토글은 **`~/.hermes/.env`**, 구조화된 설정은 **`~/.hermes/config.yaml`**에 둡니다. 두 곳에 모두 값이 설정되어 있으면 환경 변수가 항상 config.yaml 값보다 우선합니다.

### 환경 변수(`.env`)

| 변수 | 필수 여부 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | **예** | — | [Discord Developer Portal](https://discord.com/developers/applications)의 봇 토큰. |
| `DISCORD_ALLOWED_USERS` | 조건부 | — | 봇과 상호작용할 수 있는 Discord 사용자 ID의 쉼표로 구분된 목록. 이 값이나 `DISCORD_ALLOWED_ROLES`가 없으면 `DISCORD_ALLOW_ALL_USERS=true`, `GATEWAY_ALLOW_ALL_USERS=true`, 또는 `DISCORD_ALLOWED_CHANNELS`가 길드 접근 범위를 명시적으로 제한하지 않는 한 게이트웨이는 모든 사용자를 거부합니다. |
| `DISCORD_ALLOWED_ROLES` | 아니요 | — | 쉼표로 구분된 Discord 역할 ID. 이 역할 중 하나를 가진 멤버는 인증됩니다. `DISCORD_ALLOWED_USERS`와는 OR 의미입니다. 연결 시 **Server Members Intent**를 자동으로 활성화합니다. 운영 팀이 자주 바뀌는 경우 유용합니다. 새 운영자는 역할이 부여되는 즉시 접근할 수 있으므로 설정을 배포할 필요가 없습니다. |
| `DISCORD_ALLOW_ALL_USERS` | 아니요 | `false` | 봇에 접근할 수 있는 모든 Discord 사용자를 허용하기 위한 명시적 옵트인. Discord에만 이전 0.18의 개방형 동작을 복원합니다. 신뢰할 수 있는/비공개 길드 또는 개발 환경에서만 사용하세요. |
| `GATEWAY_ALLOW_ALL_USERS` | 아니요 | `false` | 모든 게이트웨이 플랫폼에 적용되는 전역 전체 허용 옵트인. 연결된 모든 플랫폼을 의도적으로 개방하려는 경우가 아니라면 플랫폼별 `DISCORD_ALLOW_ALL_USERS`를 우선 사용하세요. |
| `DISCORD_HOME_CHANNEL` | 아니요 | — | 봇이 주도적으로 메시지(크론 출력, 리마인더, 알림)를 보내는 채널 ID. |
| `DISCORD_HOME_CHANNEL_NAME` | 아니요 | `"Home"` | 로그와 상태 출력에 표시할 홈 채널 이름. |
| `DISCORD_COMMAND_SYNC_POLICY` | 아니요 | `"safe"` | 네이티브 슬래시 명령의 시작 시 동기화를 제어합니다. `"safe"`는 기존 전역 명령을 비교하고 변경된 항목만 업데이트하며, Discord 메타데이터 변경을 패치로 적용할 수 없으면 명령을 다시 만듭니다. `"bulk"`는 이전의 `tree.sync()` 동작을 유지합니다. `"off"`는 시작 동기화를 완전히 건너뜁니다. |
| `DISCORD_REQUIRE_MENTION` | 아니요 | `true` | `true`이면 봇은 서버 채널에서 `@멘션`되었을 때만 응답합니다. 모든 채널의 모든 메시지에 응답하게 하려면 `false`로 설정하세요. |
| `DISCORD_THREAD_REQUIRE_MENTION` | 아니요 | `false` | `true`이면 스레드 안의 멘션 단축 동작이 비활성화됩니다. 봇이 이미 참여한 뒤에도 채널과 동일하게 스레드에서 `@멘션`을 요구합니다. 여러 봇이 하나의 스레드를 공유하고 각 봇이 명시적 `@멘션`에만 실행되기를 원할 때 사용하세요. |
| `DISCORD_FREE_RESPONSE_CHANNELS` | 아니요 | — | `DISCORD_REQUIRE_MENTION`이 `true`여도 `@멘션` 없이 봇이 응답하는 채널 ID의 쉼표로 구분된 목록. |
| `DISCORD_IGNORE_NO_MENTION` | 아니요 | `true` | `true`이면 메시지가 다른 사용자를 `@멘션`하지만 봇은 멘션하지 않을 때 봇이 조용히 있습니다. 다른 사람을 대상으로 한 대화에 봇이 끼어드는 것을 막습니다. 서버 채널에만 적용되며 DM에는 적용되지 않습니다. |
| `DISCORD_AUTO_THREAD` | 아니요 | `true` | `true`이면 텍스트 채널에서 모든 `@멘션`에 대해 새 스레드를 자동으로 만들어 각 대화를 격리합니다(Slack과 유사). 이미 스레드 안에 있는 메시지나 DM에는 영향을 주지 않습니다. |
| `DISCORD_ALLOW_BOTS` | 아니요 | `"none"` | 다른 Discord 봇의 메시지를 처리하는 방식을 제어합니다. `"none"` — 다른 모든 봇을 무시합니다. `"mentions"` — Hermes를 `@멘션`한 봇 메시지만 수락합니다. `"all"` — 모든 봇 메시지를 수락합니다. |
| `DISCORD_REACTIONS` | 아니요 | `true` | `true`이면 처리 중 메시지에 이모지 반응을 추가합니다(시작 시 👀, 성공 시 ✅, 오류 시 ❌). 반응을 완전히 끄려면 `false`로 설정하세요. |
| `DISCORD_IGNORED_CHANNELS` | 아니요 | — | 봇이 `@멘션`을 받아도 **절대** 응답하지 않는 채널 ID의 쉼표로 구분된 목록. |
| `DISCORD_ALLOWED_CHANNELS` | 아니요 | — | 쉼표로 구분된 채널 ID. 설정하면 봇은 이 채널에서만 응답합니다(허용된 경우 DM은 추가). `config.yaml`의 `discord.allowed_channels`보다 우선합니다. `DISCORD_IGNORED_CHANNELS`와 결합해 허용/거부 규칙을 표현하세요. |
| `DISCORD_NO_THREAD_CHANNELS` | 아니요 | — | 봇이 스레드를 만들지 않고 채널에 직접 응답하는 채널 ID의 쉼표로 구분된 목록. `DISCORD_AUTO_THREAD`가 `true`일 때만 의미가 있습니다. |
| `DISCORD_HISTORY_BACKFILL` | 아니요 | `true` | `true`이면 봇이 멘션될 때 최근 채널 스크롤백(봇의 마지막 응답 이후)을 사용자 메시지 앞에 추가합니다. `require_mention`으로 인해 봇이 놓칠 수 있는 컨텍스트를 복구합니다. DM과 자유 응답 채널에서는 건너뜁니다. 비활성화하려면 `false`로 설정하세요. |
| `DISCORD_HISTORY_BACKFILL_LIMIT` | 아니요 | `50` | 백필 블록을 구성할 때 뒤로 탐색할 최대 메시지 수. 실제로는 보통 더 일찍 중단되며, 일반적으로 채널에 있는 봇 자신의 마지막 메시지에서 멈춥니다. |
| `DISCORD_REPLY_TO_MODE` | 아니요 | `"first"` | 답장 참조 동작을 제어합니다. `"off"` — 원본 메시지에 절대 답장하지 않습니다. `"first"` — 첫 메시지 청크에만 답장을 참조합니다(기본값). `"all"` — 모든 청크에 답장을 참조합니다. |
| `DISCORD_ALLOW_MENTION_EVERYONE` | 아니요 | `false` | `false`(기본값)이면 응답에 해당 토큰이 포함되어 있어도 봇은 `@everyone` 또는 `@here`를 호출할 수 없습니다. 다시 허용하려면 `true`로 설정하세요. 아래의 [멘션 제어](#mention-control)를 참고하세요. |
| `DISCORD_ALLOW_MENTION_ROLES` | 아니요 | `false` | `false`(기본값)이면 봇은 `@role` 멘션을 호출할 수 없습니다. 허용하려면 `true`로 설정하세요. |
| `DISCORD_ALLOW_MENTION_USERS` | 아니요 | `true` | `true`(기본값)이면 봇이 ID로 개별 사용자를 호출할 수 있습니다. |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | 아니요 | `true` | `true`(기본값)이면 메시지에 답장할 때 원 작성자를 호출합니다. |
| `DISCORD_PROXY` | 아니요 | — | Discord 연결(HTTP, WebSocket, REST)에 사용할 프록시 URL. `HTTPS_PROXY`/`ALL_PROXY`보다 우선합니다. `http://`, `https://`, `socks5://` 스킴을 지원합니다. |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | 아니요 | `false` | `true`이면 모든 파일 형식의 첨부 파일을 수락합니다(기본 제공 PDF/텍스트/zip/office 허용 목록에 한정되지 않음). 알 수 없는 형식은 디스크에 캐시되고 `application/octet-stream` MIME과 함께 로컬 경로로 에이전트에 제공되므로 `terminal` / `read_file` / `ffprobe` 등을 사용해 검사할 수 있습니다. |
| `DISCORD_MAX_ATTACHMENT_BYTES` | 아니요 | `33554432` | 게이트웨이가 다운로드하고 캐시할 첨부 파일 하나당 최대 바이트 수. 기본값은 32 MiB입니다. 제한을 없애려면 `0`으로 설정하세요(첨부 파일은 기록 중 메모리에 유지되므로 무제한 설정은 실제 메모리 비용이 발생합니다). |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | 아니요 | `0.6` | 대기 중인 텍스트 청크를 플러시하기 전에 어댑터가 기다리는 유예 시간. 스트리밍 출력을 매끄럽게 만드는 데 유용합니다. |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 아니요 | `2.0` | 단일 메시지가 Discord 길이 제한을 초과해 나뉜 경우 분할된 청크 사이의 지연 시간. |

:::warning 봇 간 대화는 지원되지 않음
`DISCORD_ALLOW_BOTS`는 특정 신뢰할 수 있는 봇(예: 릴레이 또는 웹훅 봇)의 입력을 받기 위한 것이며, 두 Hermes 프로필이 서로 대화하도록 하기 위한 것이 아닙니다. 기본값인 `"none"`은 다른 모든 봇을 무시하며 안전한 설정입니다.

여러 Hermes 프로필이 공유 채널에서 서로 응답하도록 연결하는 것은(`"mentions"` 또는 `"all"`을 여러 프로필에 설정) 지원되지 않는 토폴로지입니다. Discord는 모든 답장에서 답장을 받은 작성자를 자동으로 `@멘션`하므로, `"mentions"`를 사용하면 두 봇이 서로의 멘션 조건을 계속 만족해 확인 루프가 무한히 발생합니다. 지원되는 설정은 `DISCORD_ALLOW_BOTS`를 `"none"`으로 유지하는 것이므로 이에 대한 회로 차단기는 없습니다. 특정 봇을 반드시 수락해야 한다면 범위를 좁게 지정하고 다른 자동 응답 에이전트는 대상으로 삼지 마세요.
:::

### 설정 파일(`config.yaml`)

`~/.hermes/config.yaml`의 `discord` 섹션은 위의 환경 변수를 반영합니다. config.yaml 설정은 기본값으로 적용되며, 동일한 환경 변수가 이미 설정되어 있으면 환경 변수가 우선합니다.

```yaml
# Discord-specific settings
discord:
  require_mention: true           # Require @mention in server channels
  thread_require_mention: false   # If true, require @mention in threads too (multi-bot threads)
  free_response_channels: ""      # Comma-separated channel IDs (or YAML list)
  auto_thread: true               # Auto-create threads on @mention
  reactions: true                 # Add emoji reactions during processing
  ignored_channels: []            # Channel IDs where bot never responds
  no_thread_channels: []          # Channel IDs where bot responds without threading
  history_backfill: true          # Prepend recent channel scrollback on mention (default: true)
  history_backfill_limit: 50      # Max messages to scan backwards (default: 50)
  missed_message_backfill:        # Replay messages missed while disconnected (opt-in)
    enabled: false
    channels: []                  # Empty uses free_response_channels
    window_seconds: 21600         # Look back at most 6 hours
    limit: 100                    # Global scan cap per reconnect
    max_dispatches: 10            # Recovery dispatch cap per reconnect
  channel_prompts: {}             # Per-channel ephemeral system prompts
  voice_channel_inactivity_timeout_seconds: 300  # Set 0 to stay in VC until explicit /voice leave
  voice_playback_timeout_seconds: 120             # Minimum playback watchdog; long clips get duration+padding
  allow_mentions:                 # What the bot is allowed to ping (safe defaults)
    everyone: false               # @everyone / @here pings (default: false)
    roles: false                  # @role pings (default: false)
    users: true                   # @user pings (default: true)
    replied_user: true            # reply-reference pings the author (default: true)

# Session isolation (applies to all gateway platforms, not just Discord)
group_sessions_per_user: true     # Isolate sessions per user in shared channels
```

#### `discord.require_mention`

**유형:** boolean — **기본값:** `true`

활성화하면 봇은 서버 채널에서 직접 `@멘션`된 경우에만 응답합니다. 이 설정과 관계없이 DM에는 항상 응답합니다.

#### `discord.thread_require_mention`

**유형:** boolean — **기본값:** `false`

기본적으로 봇이 스레드에 참여하면( `@멘션`으로 자동 생성했거나 한 번 답장한 경우) 이후 스레드의 모든 메시지에 다시 `@멘션`하지 않아도 계속 응답합니다. 일대일 대화에는 적절한 기본값입니다.

사용자가 차례로 봇 하나씩을 부르는 **다중 봇 스레드**에서는 이 기본값이 문제가 됩니다. 스레드의 다른 모든 봇도 모든 메시지에 실행되어 비용을 소모하고 채널에 스팸을 보냅니다. `thread_require_mention: true`로 설정하면 스레드 안의 단축 동작이 비활성화되고, 채널과 같은 방식으로 스레드에도 멘션 조건이 적용됩니다. 명시적 `@멘션`은 이전과 동일하게 작동합니다.

```yaml
discord:
  require_mention: true
  thread_require_mention: true    # multi-bot setup
```

#### `discord.free_response_channels`

**유형:** string 또는 list — **기본값:** `""`

`@멘션` 없이 모든 메시지에 응답하는 채널 ID입니다. 쉼표로 구분한 문자열이나 YAML 목록 중 하나를 사용할 수 있습니다.

```yaml
# String format
discord:
  free_response_channels: "1234567890,9876543210"

# List format
discord:
  free_response_channels:
    - 1234567890
    - 9876543210
```

스레드의 부모 채널이 이 목록에 있으면 해당 스레드도 멘션이 필요 없는 상태가 됩니다.

자유 응답 채널에서는 **자동 스레드 생성도 건너뜁니다**. 봇은 메시지마다 새 스레드를 만드는 대신 인라인으로 답장합니다. 이를 통해 채널을 가벼운 채팅 공간으로 사용할 수 있습니다. 스레드 동작이 필요하다면 채널을 자유 응답으로 등록하지 말고 일반적인 `@멘션` 흐름을 사용하세요.

#### `discord.auto_thread`

**유형:** boolean — **기본값:** `true`

활성화하면 일반 텍스트 채널의 모든 `@멘션`이 대화를 위한 새 스레드를 자동으로 만듭니다. 이를 통해 메인 채널을 깔끔하게 유지하고 각 대화에 격리된 세션 기록을 제공할 수 있습니다. 스레드가 생성된 뒤에는 해당 스레드의 후속 메시지에 `@멘션`이 필요하지 않습니다. 봇이 이미 참여하고 있다는 것을 알고 있기 때문입니다. 다중 봇 설정에서 이 스레드 내 단축 동작을 비활성화하려면 [`thread_require_mention`](#discordthread_require_mention)을 `true`로 설정하세요.

기존 스레드나 DM으로 보낸 메시지는 이 설정의 영향을 받지 않습니다. `discord.free_response_channels` 또는 `discord.no_thread_channels`에 등록된 채널도 자동 스레드 생성을 우회하고 인라인으로 응답합니다.

#### `discord.reactions`

**유형:** boolean — **기본값:** `true`

봇이 시각적 피드백으로 메시지에 이모지 반응을 추가할지 제어합니다.
- 봇이 메시지 처리를 시작하면 👀 추가
- 응답이 성공적으로 전달되면 ✅ 추가
- 처리 중 오류가 발생하면 ❌ 추가

반응이 산만하거나 봇 역할에 **Add Reactions** 권한이 없다면 비활성화하세요.

#### `discord.ignored_channels`

**유형:** string 또는 list — **기본값:** `[]`

직접 `@멘션`되어도 봇이 **절대** 응답하지 않는 채널 ID입니다. 가장 높은 우선순위를 가지므로 채널이 이 목록에 있으면 `require_mention`, `free_response_channels` 또는 다른 설정과 관계없이 봇은 해당 채널의 모든 메시지를 조용히 무시합니다.

```yaml
# String format
discord:
  ignored_channels: "1234567890,9876543210"

# List format
discord:
  ignored_channels:
    - 1234567890
    - 9876543210
```

스레드의 부모 채널이 이 목록에 있으면 해당 스레드의 메시지도 무시됩니다.

#### `discord.no_thread_channels`

**유형:** string 또는 list — **기본값:** `[]`

봇이 자동으로 스레드를 만들지 않고 채널에 직접 응답하는 채널 ID입니다. `auto_thread`가 `true`일 때(기본값)만 효과가 있습니다. 이 채널에서 봇은 새 스레드를 만드는 대신 일반 메시지처럼 인라인으로 응답합니다.

```yaml
discord:
  no_thread_channels:
    - 1234567890  # Bot responds inline here
```

스레드가 불필요한 소음을 더할 수 있는 봇 상호작용 전용 채널에 유용합니다.

#### `discord.channel_prompts`

**유형:** mapping — **기본값:** `{}`

일치하는 Discord 채널이나 스레드에서 매 턴마다 주입되지만 기록에는 저장되지 않는 채널별 임시 시스템 프롬프트입니다.

```yaml
discord:
  channel_prompts:
    "1234567890": |
      This channel is for research tasks. Prefer deep comparisons,
      citations, and concise synthesis.
    "9876543210": |
      This forum is for therapy-style support. Be warm, grounded,
      and non-judgmental.
```

동작은 다음과 같습니다.
- 정확히 일치하는 스레드/채널 ID가 우선합니다.
- 스레드나 포럼 게시물 안에서 메시지가 도착했고 해당 스레드에 명시적 항목이 없으면 Hermes는 부모 채널/포럼 ID로 대체합니다.
- 프롬프트는 실행 중 임시로 적용되므로 프롬프트를 변경하면 과거 세션 기록을 다시 작성하지 않고도 이후 턴에 즉시 반영됩니다.

#### `discord.history_backfill`

**유형:** boolean — **기본값:** `true`

활성화하면 봇은 각 `@멘션`에서 놓친 채널 메시지를 복구합니다. `require_mention: true`이면 봇을 직접 태그한 메시지만 처리하며, 채널의 나머지 모든 내용은 세션 기록에 보이지 않습니다. 백필은 트리거되었을 때 최근 채널 기록을 뒤로 검색해 봇의 마지막 응답과 현재 멘션 사이의 메시지를 수집하고 컨텍스트로 포함합니다.

표면별 동작은 다음과 같습니다.

- **서버 채널**(`require_mention: true`): 봇의 마지막 응답 이후 채널을 백필합니다. 다른 참여자가 봇을 부르지 않고 게시한 내용을 반영할 때 유용합니다.
- **스레드**: 스레드만 백필합니다. 스레드에 대한 Discord의 `channel.history()`는 부모 채널이 아니라 해당 스레드의 메시지만 반환합니다. 스레드는 보통 독립적인 대화이므로 이것이 올바른 범위입니다.
- **DM**: 건너뜁니다. 모든 DM 메시지가 봇을 트리거하므로 세션 기록은 이미 완전하며, 채울 멘션 공백이 없습니다.
- **자유 응답 채널** 및 **봇이 자동으로 만든 자체 스레드**: 같은 이유로 건너뜁니다. 멘션 조건이 없어 공백이 없습니다.

사용자별 세션(`group_sessions_per_user: true`, 기본값)에도 도움이 됩니다. 사용자의 세션에는 다른 채널 참여자가 게시한 컨텍스트와 사용자가 봇을 태그하기 전에 보낸 자신의 메시지가 빠져 있기 때문입니다. 백필이 두 공백을 모두 채웁니다.

```yaml
discord:
  history_backfill: true   # default
```

끄려면 다음과 같이 설정합니다.

```yaml
discord:
  history_backfill: false
```

> **참고:** 봇이 처리 중인 동안(트리거와 응답 사이)에 도착한 메시지는 캡처되지 않습니다. 이는 허용된 단순화이며, 사용자는 메시지를 다시 보내거나 다시 태그할 수 있습니다.

#### `discord.history_backfill_limit`

**유형:** integer — **기본값:** `50`

채널 컨텍스트를 복구할 때 뒤로 검색할 최대 메시지 수입니다. 실제로는 보통 봇 자신의 마지막 메시지에서 훨씬 일찍 중단되며, 이는 턴 사이의 자연스러운 경계입니다. 이 제한은 최근 기록에 이전 봇 메시지가 없는 콜드 스타트와 긴 공백을 위한 안전 상한입니다.

```yaml
discord:
  history_backfill: true
  history_backfill_limit: 50
```

#### `discord.missed_message_backfill`

**유형:** object — **기본값:** 비활성화

재시작이나 네트워크 장애 중에 Discord의 WebSocket 재개 창이 만료될 수 있습니다. 그 공백 동안 전송된 메시지는 실시간 게이트웨이 이벤트로 전달되지 않습니다. 이 옵션을 활성화하면 Hermes는 Discord에 다시 연결한 뒤 설정된 채널과 스레드 기록의 제한된 범위를 검색하고, 아직 처리되지 않은 메시지를 실시간 이벤트와 동일한 권한 부여, 멘션, 채널, 중복 제거, 디스패치 경로로 보냅니다.

```yaml
discord:
  missed_message_backfill:
    enabled: true
    channels: ["123456789012345678"]
    window_seconds: 3600
    limit: 100
    max_dispatches: 10
```

`channels`가 비어 있으면 Hermes는 `discord.free_response_channels`를 사용합니다. 봇이 접근 가능한 모든 서버 텍스트 채널을 검사해야 할 때만 `"*"`로 설정하세요. 복구 원장은 프로필별로 `gateway/discord_message_recovery.db`에 저장되므로, 성공적으로 답변한 메시지가 이후 재시작 뒤 다시 재생되지 않습니다.

#### `group_sessions_per_user`

**유형:** boolean — **기본값:** `true`

이는 Discord에만 해당하지 않는 전역 게이트웨이 설정으로, 같은 채널의 사용자들이 세션 기록을 격리할지 제어합니다.

`true`인 경우: `#research`에서 대화하는 Alice와 Bob은 각각 Hermes와 별도의 대화를 가집니다. `false`인 경우: 전체 채널이 하나의 대화 기록과 하나의 실행 중 에이전트 슬롯을 공유합니다.

```yaml
group_sessions_per_user: true
```

각 모드의 전체 영향은 위의 [세션 모델](#session-model-in-discord) 섹션을 참고하세요.

#### `display.tool_progress`

**유형:** string — **기본값:** `"all"` — **값:** `off`, `new`, `all`, `verbose`

처리 중 봇이 채팅에 진행 메시지(예: "파일 읽는 중...", "터미널 명령 실행 중...")를 보낼지 제어합니다. 모든 플랫폼에 적용되는 전역 게이트웨이 설정입니다.

```yaml
display:
  tool_progress: "all"    # off | new | all | verbose
```

- `off` — 진행 메시지를 보내지 않음
- `new` — 턴마다 첫 번째 도구 호출만 표시
- `all` — 모든 도구 호출 표시(게이트웨이 메시지에서는 40자로 잘림)
- `verbose` — 전체 도구 호출 세부 정보 표시(긴 메시지가 생성될 수 있음)

#### `display.tool_progress_command`

**유형:** boolean — **기본값:** `false`

활성화하면 게이트웨이에서 `/verbose` 슬래시 명령을 사용할 수 있습니다. config.yaml을 편집하지 않고 도구 진행 모드(`off → new → all → verbose → off`)를 순환할 수 있습니다.

```yaml
display:
  tool_progress_command: true
```

#### `display.reasoning_style`

**유형:** string — **기본값(Discord):** `"subtext"` — **값:** `code`, `blockquote`, `subtext`

추론 표시가 활성화되었을 때 모델의 추론 블록을 렌더링하는 방식을 제어합니다. Discord의 기본값은 `subtext`이며, Discord의 네이티브 `-# ` 작은 회색 메타데이터 텍스트를 사용해 추론을 답변보다 시각적으로 부차적인 위치에 둡니다. `blockquote`는 `>` 인용으로 렌더링하고, `code`(다른 플랫폼의 기본값)는 펜스 코드 블록을 사용합니다. 긴 추론은 처음 15줄로 접힙니다.

```yaml
display:
  platforms:
    discord:
      reasoning_style: subtext   # code | blockquote | subtext
```

## 슬래시 명령 접근 제어

기본적으로 허용된 모든 사용자는 모든 슬래시 명령을 실행할 수 있습니다. 허용 목록을 **관리자**(슬래시 명령 전체 접근)와 **일반 사용자**(명시적으로 활성화한 명령만 접근)로 나누려면 Discord 플랫폼의 `extra` 블록에 `allow_admin_from`과 `user_allowed_commands`를 추가하세요.

```yaml
gateway:
  platforms:
    discord:
      extra:
        # Existing user allowlist (unchanged)
        allow_from:
          - "123456789012345678"  # admin user ID
          - "999888777666555444"  # regular user ID

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789012345678"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin / command lists for server channels
        group_allow_admin_from:
          - "123456789012345678"
        group_user_allowed_commands:
          - status
```

**동작:**

- 특정 범위(DM 또는 서버 채널)의 `allow_admin_from`에 포함된 사용자는 실시간 명령 레지스트리를 통해 등록된 **모든** 슬래시 명령(내장 및 플러그인 등록)을 실행할 수 있습니다.
- `allow_admin_from`에 포함되지 않은 사용자는 `user_allowed_commands`에 나열된 명령과 항상 허용되는 `/help`, `/whoami`만 실행할 수 있습니다.
- 일반 채팅(슬래시가 아닌 메시지)에는 영향을 주지 않습니다. 관리자가 아닌 사용자도 평소처럼 에이전트와 대화할 수 있으며, 임의의 명령을 실행할 수 없을 뿐입니다.
- **이전 버전 호환:** 범위에 `allow_admin_from`이 설정되지 않으면 해당 범위에서는 슬래시 명령 제한이 비활성화됩니다. 기존 설치는 변경 없이 계속 작동합니다.
- DM 관리자 상태가 서버 채널 관리자 상태를 의미하지는 않습니다. 각 범위에는 자체 관리자 목록이 있습니다.

`/whoami`를 사용하면 활성 범위, 자신의 등급(관리자 / 사용자 / 제한 없음), 실행할 수 있는 슬래시 명령을 확인할 수 있습니다.

## 대화형 모델 선택기

Discord 채널에서 인수 없이 `/model`을 보내 드롭다운 기반 모델 선택기를 엽니다.

1. **Provider selection** — 사용 가능한 프로바이더를 표시하는 Select 드롭다운(최대 25개)
2. **Model selection** — 선택한 프로바이더의 모델을 표시하는 두 번째 드롭다운(최대 25개)

선택기는 120초 후 시간 초과됩니다. `DISCORD_ALLOWED_USERS`에 포함된 승인된 사용자만 상호작용할 수 있습니다. 모델 이름을 알고 있다면 `/model <name>`을 직접 입력하세요.

## 스킬용 네이티브 슬래시 명령

Hermes는 설치된 스킬을 **네이티브 Discord Application Commands**로 자동 등록합니다. 따라서 스킬은 내장 명령과 함께 Discord의 자동 완성 `/` 메뉴에 표시됩니다.

- 각 스킬은 Discord 슬래시 명령이 됩니다(예: `/code-review`, `/ascii-art`).
- 스킬은 선택적 `args` 문자열 매개변수를 받습니다.
- Discord는 봇 하나당 애플리케이션 명령을 100개로 제한합니다. 사용 가능한 슬롯보다 스킬이 많으면 초과 스킬은 로그에 경고를 남기고 건너뜁니다.
- 스킬은 `/model`, `/reset`, `/background` 같은 내장 명령과 함께 봇 시작 시 등록됩니다.

추가 설정은 필요하지 않습니다. `hermes skills install`로 설치한 스킬은 다음 게이트웨이 재시작 시 Discord 슬래시 명령으로 자동 등록됩니다.

### 슬래시 명령 등록 비활성화

같은 Discord 애플리케이션에 여러 Hermes 게이트웨이를 실행하는 경우(예: 스테이징 + 프로덕션), 전역 슬래시 명령 등록을 담당하는 게이트웨이는 하나만 있어야 합니다. 그렇지 않으면 마지막으로 시작한 게이트웨이가 승자가 되어 등록이 계속 바뀝니다. "follower" 게이트웨이에서는 슬래시 명령 등록을 끄세요.

```yaml
gateway:
  platforms:
    discord:
      extra:
        slash_commands: false   # default: true
```

"primary" 게이트웨이에서는 `true`로 두면 정상 동작이 유지되며, 내장 명령과 설치된 스킬에 대한 전역 `/` 메뉴 명령이 제공됩니다.

## 미디어 보내기(인라인 `MEDIA:` 태그)

Discord 어댑터는 에이전트 응답에 인라인 `MEDIA:/path/to/file` 태그를 사용해 모든 일반적인 미디어 형식의 네이티브 파일 업로드를 지원합니다. 어댑터가 태그를 제거하고 파일을 자동으로 업로드합니다.

| 유형 | 전달 방식 |
|---|---|
| 이미지(PNG/JPG/WebP) | 인라인 미리보기가 있는 네이티브 Discord 이미지 첨부 |
| 애니메이션 GIF | `send_animation`이 `animation.gif`로 업로드하므로 Discord가 정적 썸네일이 아니라 인라인으로 재생 |
| 동영상(MP4/MOV) | `send_video` — 네이티브 동영상 플레이어 |
| 오디오 / 음성 | `send_voice` — 가능한 경우 네이티브 음성 메시지, 그렇지 않으면 파일 첨부 |
| 문서(PDF/ZIP/docx 등) | `send_document` — 다운로드 버튼이 있는 네이티브 첨부 |

Discord의 업로드당 크기 제한은 서버의 부스트 등급에 따라 다릅니다(무료 서버 25MB, 최대 500MB). Hermes가 HTTP 413을 받으면 어댑터는 조용히 실패하는 대신 로컬 캐시 경로를 가리키는 링크로 대체합니다.

## 임의의 파일 형식 받기

사용자가 업로드하는 모든 파일 형식이 허용됩니다. 에이전트에 메시지를 보낼 권한이 관문이며 파일 확장자는 관문이 아닙니다. 모든 업로드는 `~/.hermes/cache/documents/` 아래에 다운로드되고 캐시되며, 에이전트가 `terminal`(`ffprobe`, `unzip`, `file`, `strings` 등)이나 `read_file`로 파일을 검사할 수 있도록 `DOCUMENT` 유형 메시지 이벤트로 제공됩니다.

- 알려진 형식(PDF, docx/xlsx/pptx, zip, 이미지/오디오/동영상 등)은 정확한 MIME을 유지합니다.
- 알 수 없는 형식은 업로드가 보고한 콘텐츠 유형으로 대체되며, 콘텐츠 유형이 없으면 `application/octet-stream`을 사용합니다.
- 작은 UTF-8 디코딩 가능 파일(텍스트, 코드, 설정, HTML, CSS, JSON, YAML 등)은 최대 100KiB까지 내용이 프롬프트에 자동 주입됩니다. 디코딩할 수 없는 바이너리 파일은 경로를 가리키는 컨텍스트 메모로만 제공됩니다(`to_agent_visible_cache_path`를 통해 Docker/Modal 샌드박스 터미널에서는 자동 변환됨). 따라서 컨텍스트 창을 과도하게 키우지 않습니다.

수신 측 제한은 파일당 크기 상한 하나뿐입니다(기본값 32MiB).

```yaml
discord:
  # Optional — raise/disable the per-file size cap. Default is 32 MiB.
  # The whole file is held in memory while being cached, so unlimited
  # uploads carry a real memory cost.
  max_attachment_bytes: 33554432   # bytes; 0 = unlimited
```

동등한 환경 변수: `DISCORD_MAX_ATTACHMENT_BYTES=33554432`(또는 제한 없음은 `0`).

기존 `discord.allow_any_attachment` 플래그는 이제 아무 동작도 하지 않습니다. 모든 파일 형식이 항상 허용되며, 기존 설정이 오류를 일으키지 않도록 이 플래그만 유지됩니다.

:::warning 무제한 설정의 메모리 비용
크기 상한을 비활성화하면(`max_attachment_bytes: 0`) 사용자가 멀티 GB 파일을 봇에 놓을 수 있고, 게이트웨이는 파일을 디스크에 캐시하는 동안 메모리에 성실하게 버퍼링합니다. 신뢰할 수 있는 단일 사용자 설치에서만 설정하세요. 공유 봇에서는 기본값인 32MiB를 유지하거나 신중하게 높이세요.
:::

## 대화형 프롬프트(clarify)

에이전트가 `clarify` 도구를 호출해 선호하는 접근 방식을 묻거나, 작업 후 피드백을 받거나, 중요하지 않은 결정을 내리기 전에 확인할 때 Discord는 선택지마다 **버튼 하나씩** 질문을 렌더링합니다.

> 대시보드에 어떤 프레임워크를 사용해야 하나요?
>
> [1. Next.js] [2. Remix] [3. Astro] [기타(답변 입력)]

번호가 있는 버튼을 클릭해 답하거나 **기타**를 클릭해 자유 형식의 응답을 입력하세요(다음에 해당 채널에서 보내는 메시지가 답변이 됩니다). 열린 형태의 `clarify` 호출(미리 정해진 선택지 없음)은 버튼을 생략하고 다음 메시지만 수집합니다.

선택이 이루어지면 버튼이 비활성화되므로 중복 클릭으로 프롬프트가 두 번 해결되지 않습니다. `~/.hermes/config.yaml`의 `agent.clarify_timeout`으로 응답 제한 시간을 설정하세요(기본값 `600`초). 제한 시간 내에 응답하지 않으면 에이전트는 센티널 메시지와 함께 차단이 해제되고 멈추는 대신 상황에 맞게 조정합니다.

## 홈 채널

봇이 주도적인 메시지(예: 크론 작업 출력, 리마인더, 알림)를 보내는 "홈 채널"을 지정할 수 있습니다. 두 가지 방법이 있습니다.

### 슬래시 명령 사용

봇이 있는 Discord 채널에서 `/sethome`을 입력합니다. 해당 채널이 홈 채널이 됩니다.

### 수동 설정

다음 내용을 `~/.hermes/.env`에 추가합니다.

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

ID를 실제 채널 ID로 바꿉니다(Developer Mode가 켜진 상태에서 마우스 오른쪽 버튼 클릭 → Copy Channel ID).

## 음성 메시지

Hermes Agent는 Discord 음성 메시지를 지원합니다.

- **수신 음성 메시지**는 설정된 STT 프로바이더를 사용해 자동으로 전사됩니다. 로컬 `faster-whisper`(키 불필요), Groq Whisper(`GROQ_API_KEY`) 또는 OpenAI Whisper(`VOICE_TOOLS_OPENAI_KEY`)를 사용할 수 있습니다.
- **텍스트 음성 변환**: `/voice tts`를 사용하면 봇이 텍스트 답장과 함께 음성 응답을 보냅니다.
- **Discord 음성 채널**: Hermes가 음성 채널에 참여해 사용자의 말을 듣고 채널에서 응답하도록 할 수도 있습니다.

전체 설정 및 운영 가이드는 다음을 참고하세요.
- [음성 모드](/user-guide/features/voice-mode)
- [Hermes에서 음성 모드 사용하기](/guides/use-voice-mode-with-hermes)

### 음성 채널 오디오 효과(분위기음 + 음성 확인)

봇이 음성 채널에 있을 때 더 대화하는 듯한 느낌을 줄 수 있습니다. 작업을 시작하기 전에 짧은 음성 확인("확인해 볼게요")을 재생하고, 도구가 실행되는 동안 은은한 주변 "생각 중" 배경음을 재생할 수 있습니다. 음성은 주변 소리를 낮추고 완료되면 다시 키우며, Grok 음성 모드와 비슷합니다.

discord.py는 연결당 하나의 오디오 스트림만 재생하므로 Hermes는 나가는 스트림에 소프트웨어 믹서를 설치합니다. 이 믹서는 주변음, 확인 음성, TTS 답변을 하나의 스트림으로 합치므로 서로 끊지 않고 겹쳐 재생됩니다.

기본적으로 **꺼져 있습니다**. `config.yaml`에서 활성화하세요.

```yaml
discord:
  voice_fx:
    enabled: true          # master switch
    ambient_enabled: true  # idle "thinking" bed while tools run
    ambient_path: ""       # custom loop file (any audio format); "" = built-in synthesised pad
    ambient_gain: 0.18     # idle bed loudness (0.0–1.0)
    duck_gain: 0.06        # ambient loudness while the bot is speaking
    speech_gain: 1.0       # TTS / acknowledgement loudness
    ack_enabled: true      # speak a short phrase before the first tool call of a turn
    ack_phrases:           # picked at random; set to [] to disable the spoken ack
      - "Let me look into that."
      - "One moment."
      - "Checking on that now."
```

참고:
- 봇이 명시적인 `/voice leave` 또는 수동 연결 해제 전까지 음성 채널에 남아 있기를 원한다면 `voice_channel_inactivity_timeout_seconds: 0`으로 설정하세요. 기본값은 과거의 300초 유휴 자동 퇴장을 유지합니다.
- `voice_playback_timeout_seconds`는 긴 TTS의 하드 상한이 아니라 하한입니다. Hermes는 생성된 오디오 길이를 확인하고, 설정된 하한보다 길면 `길이 + 30초`를 기다립니다.
- 확인 음성은 턴마다 최대 한 번, 봇이 음성 채널에 있고 믹서가 활성화된 경우에만 재생됩니다. 설정된 TTS 프로바이더를 사용합니다.
- `ambient_path`는 `ffmpeg`가 디코딩할 수 있는 모든 파일을 받으며 끊김 없이 반복됩니다. 비워 두면 내장 합성 패드가 사용되므로 별도 에셋이 필요하지 않습니다.
- 모든 설정은 `config.yaml`에 둡니다(`.env`가 아님). 이는 비밀이 아니라 동작 설정이기 때문입니다.
- `voice_fx.enabled`가 `false`이면 음성 재생은 원래의 단일 재생 경로를 사용하며 아무것도 변경되지 않습니다.

## 포럼 채널

Discord 포럼 채널(유형 15)은 직접 메시지를 받지 않으며, 포럼의 모든 게시물은 스레드여야 합니다. Hermes는 포럼 채널을 자동으로 감지하고 해당 채널에 보내야 할 때마다 새 스레드 게시물을 만들므로, 텍스트 답변, TTS, 이미지, 파일 첨부가 모두 에이전트의 별도 처리 없이 작동합니다.

- **스레드 이름**은 메시지의 첫 줄에서 가져옵니다(마크다운 제목 접두사는 제거하며 100자로 제한). 첨부 파일만 있는 경우에는 파일 이름을 대체 스레드 이름으로 사용합니다.
- **첨부 파일**은 새 스레드의 시작 메시지에 함께 포함됩니다. 별도의 업로드 단계나 부분 전송이 없습니다.
- **한 번 호출하면 한 스레드**: 포럼에 보내는 각 전송은 새 스레드를 만듭니다. 따라서 같은 포럼에 연속으로 보내면 별도의 스레드가 생성됩니다.
- **감지는 세 계층으로 이루어집니다**: 먼저 채널 디렉터리 캐시, 다음으로 프로세스 로컬 프로브 캐시, 마지막으로 실시간 `GET /channels/{id}` 프로브를 사용합니다. 마지막 결과는 프로세스 수명 동안 메모이즈됩니다.

디렉터리를 새로 고치면(해당 기능을 제공하는 플랫폼에서 `/channels refresh` 사용 또는 게이트웨이 재시작) 봇이 시작된 뒤 생성된 포럼 채널이 캐시에 추가됩니다.

## 문제 해결

### 봇이 온라인인데 메시지에 응답하지 않음

**원인**: Message Content Intent가 비활성화되어 있거나, 접근 정책이 설정되지 않아 Discord 인증이 기본적으로 거부되고 있습니다.

**해결 방법**:

1. [Developer Portal](https://discord.com/developers/applications) → 앱 → Bot → Privileged Gateway Intents에서 **Message Content Intent**를 활성화한 뒤 **Save Changes**를 클릭합니다.
2. Discord 접근 정책이 하나 이상 설정되어 있는지 확인합니다.

   ```bash
   # recommended: allow specific users
   DISCORD_ALLOWED_USERS=284102345871466496

   # or allow a trusted guild/dev bot to behave like pre-0.18 Discord
   DISCORD_ALLOW_ALL_USERS=true
   ```

3. 게이트웨이를 재시작합니다.

   ```bash
   hermes gateway restart
   ```

게이트웨이 로그에 Discord가 연결되었고 REST API 검사가 작동한다고 나오는데 모든 수신 메시지가 조용하다면 `~/.hermes/logs/gateway.log`에서 다음 경고를 확인하세요.

```text
No Discord access policy configured; inbound Discord messages will be denied by default.
```

Hermes 0.18은 외부에서 접근 가능한 어댑터에 의도적으로 기본 거부 방식을 적용합니다. `DISCORD_ALLOWED_USERS`, `DISCORD_ALLOWED_ROLES`, `DISCORD_ALLOWED_CHANNELS`가 없고 명시적인 전체 허용 플래그도 없는 Discord 봇은 성공적으로 연결되지만 일반 메시지 처리 전에 수신 사용자를 거부합니다.

### 시작 시 "Privileged intents" / `PrivilegedIntentsRequired` 오류

**원인**: Hermes가 Developer Portal에서 봇에 활성화되지 않은 권한 있는 Gateway Intent를 요청하고 있습니다. 그러면 Discord가 WebSocket 연결을 거부합니다. Hermes는 항상 **Message Content Intent**를 요청합니다. 허용 목록에서 사용자 이름(숫자 ID가 아님)을 사용하거나 `DISCORD_ALLOWED_ROLES`를 설정한 경우에는 **Server Members Intent**도 요청합니다. Presence Intent는 필요하지 않습니다.

**해결 방법**:

1. [Developer Portal](https://discord.com/developers/applications) → 앱 → Bot → Privileged Gateway Intents로 이동합니다.
2. **Message Content Intent**(필수)를 활성화합니다. 사용자 이름이나 역할 허용 목록을 사용한다면 **Server Members Intent**도 활성화합니다.
3. **Save Changes**를 클릭한 뒤 게이트웨이를 재시작합니다(`hermes gateway restart`).

게이트웨이 로그에는 Hermes가 요청한 정확한 intent가 표시됩니다. 이를 활성화하기 전까지 Discord는 연결을 계속 거부합니다. 이는 불안정한 네트워크 문제가 아니라 포털 설정 오류입니다.

### 특정 채널의 메시지를 봇이 볼 수 없음

**원인**: 봇의 역할에 해당 채널을 볼 권한이 없습니다.

**해결 방법**: Discord에서 채널 설정 → Permissions로 이동해 봇의 역할을 추가하고 **View Channel**과 **Read Message History**를 활성화합니다.

### 403 Forbidden 오류

**원인**: 봇에 필요한 권한이 없습니다.

**해결 방법**: 5단계의 URL을 사용해 올바른 권한으로 봇을 다시 초대하거나, Server Settings → Roles에서 봇 역할의 권한을 수동으로 조정합니다.

### 봇이 오프라인임

**원인**: Hermes 게이트웨이가 실행 중이 아니거나 토큰이 올바르지 않습니다.

**해결 방법**: `hermes gateway`가 실행 중인지 확인합니다. `.env` 파일의 `DISCORD_BOT_TOKEN`을 확인하세요. 최근 토큰을 재설정했다면 토큰을 업데이트합니다.

### "User not allowed" / 봇이 나를 무시함

**원인**: 사용자의 User ID가 `DISCORD_ALLOWED_USERS`에 없습니다.

**해결 방법**: `~/.hermes/.env`의 `DISCORD_ALLOWED_USERS`에 사용자 ID를 추가하고 게이트웨이를 재시작합니다.

### 같은 채널의 사람들이 예기치 않게 컨텍스트를 공유함

**원인**: `group_sessions_per_user`가 비활성화되어 있거나, 해당 컨텍스트의 메시지에 대해 플랫폼이 사용자 ID를 제공할 수 없습니다.

**해결 방법**: `~/.hermes/config.yaml`에 다음을 설정하고 게이트웨이를 재시작합니다.

```yaml
group_sessions_per_user: true
```

공유 방 대화를 의도적으로 원한다면 끈 상태로 두세요. 다만 대화 기록과 중단 동작을 공유하게 됩니다.

## 보안

:::warning
봇과 상호작용할 수 있는 사람을 제한하려면 항상 `DISCORD_ALLOWED_USERS`(또는 `DISCORD_ALLOWED_ROLES`)를 설정하세요. 둘 다 없으면 안전 조치로 게이트웨이가 기본적으로 모든 사용자를 거부합니다. 인증된 사용자는 도구 사용과 시스템 접근을 포함한 에이전트 기능에 전부 접근할 수 있으므로, 신뢰하는 사람만 승인하세요.
:::

### 역할 기반 접근 제어

개별 사용자 목록 대신 역할로 접근을 관리하는 서버(운영 팀, 지원 담당자, 내부 도구 등)에서는 `DISCORD_ALLOWED_ROLES`를 사용하세요. 역할 ID를 쉼표로 구분해 지정합니다. 해당 역할 중 하나를 가진 모든 멤버가 승인됩니다.

```bash
# ~/.hermes/.env — works alongside or instead of DISCORD_ALLOWED_USERS
DISCORD_ALLOWED_ROLES=987654321098765432,876543210987654321
```

의미는 다음과 같습니다.

- **사용자 허용 목록과 OR 관계.** 사용자의 ID가 `DISCORD_ALLOWED_USERS`에 있거나 `DISCORD_ALLOWED_ROLES`의 역할을 하나라도 가지고 있으면 승인됩니다.
- **Server Members Intent 자동 활성화.** `DISCORD_ALLOWED_ROLES`가 설정되면 봇은 연결 시 Members intent를 활성화합니다. Discord가 멤버 정보와 함께 역할 정보를 보내려면 필요합니다.
- **이름이 아닌 역할 ID.** Discord에서 가져오세요: **User Settings → Advanced → Developer Mode ON**, 그다음 아무 역할이나 마우스 오른쪽 버튼으로 클릭하고 **Copy Role ID**를 선택합니다.
- **DM 대체 동작.** DM에서는 역할 검사가 공유 서버를 검색합니다. 공유 서버 중 하나에서 허용된 역할을 가진 사용자는 DM에서도 승인됩니다.

이는 운영 팀이 자주 바뀌는 경우에 권장되는 패턴입니다. 새 운영자는 역할이 부여되는 순간 접근할 수 있으며, `.env`를 수정하거나 게이트웨이를 재시작할 필요가 없습니다.

### 멘션 제어

기본적으로 Hermes는 응답에 해당 토큰이 포함되어 있어도 봇이 `@everyone`, `@here`, 역할 멘션을 호출하지 못하게 합니다. 이를 통해 잘못 작성된 프롬프트나 반사된 사용자 콘텐츠가 서버 전체에 스팸을 보내는 것을 막습니다. 개별 `@user` 멘션과 답장 참조 멘션(작은 "답장 중…" 칩)은 일반적인 대화가 계속 작동하도록 활성화된 상태로 유지됩니다.

다음 환경 변수나 `config.yaml`을 사용해 기본값을 완화할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
discord:
  allow_mentions:
    everyone: false      # allow the bot to ping @everyone / @here
    roles: false         # allow the bot to ping @role mentions
    users: true          # allow the bot to ping individual @users
    replied_user: true   # ping the author when replying to their message
```

```bash
# ~/.hermes/.env — env vars win over config.yaml
DISCORD_ALLOW_MENTION_EVERYONE=false
DISCORD_ALLOW_MENTION_ROLES=false
DISCORD_ALLOW_MENTION_USERS=true
DISCORD_ALLOW_MENTION_REPLIED_USER=true
```

:::tip
정확한 이유를 알고 있는 경우가 아니라면 `everyone`과 `roles`는 `false`로 유지하세요. LLM이 일반적인 응답 안에 `@everyone` 문자열을 생성하기는 매우 쉽습니다. 이 보호 장치가 없으면 서버의 모든 멤버에게 알림이 전송됩니다.
:::

Hermes Agent 배포 보안에 대한 자세한 내용은 [보안 가이드](../security.md)를 참고하세요.

