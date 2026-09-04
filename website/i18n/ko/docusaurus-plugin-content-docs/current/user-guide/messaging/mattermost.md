---
sidebar_position: 8
title: "Mattermost"
description: "Hermes Agent를 Mattermost 봇으로 설정하기"
---

# Mattermost 설정

Hermes Agent는 Mattermost와 봇으로 통합되므로 DM이나 팀 채널을 통해 AI 어시스턴트와 대화할 수 있습니다. Mattermost는 자체 인프라에서 실행해 데이터를 완전히 통제할 수 있는 셀프 호스팅 오픈 소스 Slack 대안입니다. 봇은 Mattermost REST API(v4)와 실시간 이벤트를 위한 WebSocket을 통해 연결되고, 도구 사용, 메모리, 추론을 포함한 Hermes Agent 파이프라인으로 메시지를 처리한 뒤 실시간으로 응답합니다. 텍스트, 파일 첨부, 이미지, 슬래시 명령을 지원합니다.

별도의 Mattermost 라이브러리는 필요하지 않습니다 — 이 어댑터는 이미 Hermes의 의존성에 포함된 `aiohttp`를 사용합니다.

설정에 앞서, 대부분의 사용자가 가장 궁금해하는 부분부터 알아보겠습니다. Mattermost 인스턴스에 Hermes를 추가하면 어떻게 동작하는지에 관한 내용입니다.

## Hermes의 동작 방식

| 컨텍스트 | 동작 |
|---------|----------|
| **DM** | Hermes는 모든 메시지에 응답합니다. `@mention`이 필요하지 않습니다. 각 DM은 자체 세션을 가집니다. |
| **공개/비공개 채널** | `@mention`하면 Hermes가 응답합니다. 멘션이 없으면 Hermes는 메시지를 무시합니다. |
| **스레드** | `MATTERMOST_REPLY_MODE=thread`이면 Hermes가 메시지 아래 스레드로 답장합니다. 스레드 컨텍스트는 상위 채널과 격리됩니다. |
| **여러 사용자가 있는 공유 채널** | 기본적으로 Hermes는 채널 안에서 사용자별로 세션 기록을 격리합니다. 명시적으로 비활성화하지 않는 한 같은 채널에서 대화하는 두 사람은 하나의 대화 기록을 공유하지 않습니다. |

:::tip
Hermes가 원래 메시지 아래에 중첩되는 스레드 대화 방식으로 답장하게 하려면 `MATTERMOST_REPLY_MODE=thread`를 설정하세요. 기본값은 `off`이며, 채널에 일반 메시지를 보냅니다.
:::

### Mattermost의 세션 모델

기본값은 다음과 같습니다.

- 각 DM은 자체 세션을 가집니다
- 각 스레드는 자체 세션 네임스페이스를 가집니다
- 공유 채널의 각 사용자는 해당 채널 안에서 자신의 세션을 가집니다

이는 `config.yaml`에서 제어합니다.

```yaml
group_sessions_per_user: true
```

채널 전체에서 하나의 공유 대화를 원하는 경우에만 `false`로 설정하세요.

```yaml
group_sessions_per_user: false
```

공유 세션은 협업 채널에 유용할 수 있지만 다음과 같은 의미도 있습니다.

- 사용자가 컨텍스트 증가와 토큰 비용을 공유합니다
- 한 사람의 도구를 많이 사용하는 작업이 다른 모든 사람의 컨텍스트를 크게 늘릴 수 있습니다
- 한 사람의 진행 중인 실행이 같은 채널에서 다른 사람의 후속 요청을 방해할 수 있습니다

이 가이드는 Mattermost에서 봇을 만드는 과정부터 첫 메시지를 보내는 것까지 전체 설정 과정을 안내합니다.

## 1단계: 봇 계정 활성화

봇을 만들기 전에 Mattermost 서버에서 봇 계정을 활성화해야 합니다.

1. **시스템 관리자**로 Mattermost에 로그인합니다.
2. **System Console** → **Integrations** → **Bot Accounts**로 이동합니다.
3. **Enable Bot Account Creation**을 **true**로 설정합니다.
4. **Save**를 클릭합니다.

:::info
시스템 관리자 액세스 권한이 없다면 Mattermost 관리자에게 봇 계정을 활성화하고 생성해 달라고 요청하세요.
:::

## 2단계: 봇 계정 만들기

1. Mattermost에서 **☰** 메뉴(왼쪽 위) → **Integrations** → **Bot Accounts**를 클릭합니다.
2. **Add Bot Account**를 클릭합니다.
3. 세부 정보를 입력합니다.
   - **Username**: 예: `hermes`
   - **Display Name**: 예: `Hermes Agent`
   - **Description**: 선택 사항
   - **Role**: `Member`면 충분합니다
4. **Create Bot Account**를 클릭합니다.
5. Mattermost가 **봇 토큰**을 표시합니다. **즉시 복사하세요.**

:::warning[토큰은 한 번만 표시됨]
봇 토큰은 봇 계정을 만들 때 한 번만 표시됩니다. 잃어버리면 봇 계정 설정에서 다시 생성해야 합니다. 토큰을 공개적으로 공유하거나 Git에 커밋하지 마세요 — 이 토큰을 가진 사람은 누구나 봇을 완전히 제어할 수 있습니다.
:::

토큰은 안전한 곳(예: 비밀번호 관리자)에 보관하세요. 5단계에서 필요합니다.

:::tip
봇 계정 대신 **개인 액세스 토큰**을 사용할 수도 있습니다. **Profile** → **Security** → **Personal Access Tokens** → **Create Token**으로 이동하세요. 별도의 봇 사용자가 아니라 자신의 사용자로 Hermes가 게시하도록 하려는 경우 유용합니다.
:::

## 3단계: 채널에 봇 추가

봇이 응답하기를 원하는 모든 채널의 구성원으로 봇을 추가해야 합니다.

1. 봇을 사용할 채널을 엽니다.
2. 채널 이름을 클릭하고 **Add Members**를 선택합니다.
3. 봇 사용자 이름(예: `hermes`)을 검색해 추가합니다.

DM의 경우 봇과의 쪽지를 열기만 하면 즉시 응답을 받을 수 있습니다.

## 4단계: Mattermost 사용자 ID 찾기

Hermes Agent는 Mattermost 사용자 ID를 사용해 봇과 상호작용할 수 있는 사람을 제어합니다. 다음과 같이 찾을 수 있습니다.

1. 왼쪽 위의 **아바타**를 클릭하고 **Profile**을 선택합니다.
2. 프로필 대화 상자에 사용자 ID가 표시됩니다 — 클릭해 복사합니다.

사용자 ID는 `3uo8dkh1p7g1mfk49ear5fzs5c`와 같은 26자리 영숫자 문자열입니다.

:::warning
사용자 ID는 **사용자 이름이 아닙니다**. 사용자 이름은 `@` 뒤에 표시되는 이름입니다(예: `@alice`). 사용자 ID는 Mattermost가 내부적으로 사용하는 긴 영숫자 식별자입니다.
:::

**대안**: API를 통해서도 사용자 ID를 가져올 수 있습니다.

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-mattermost-server/api/v4/users/me | jq .id
```

:::tip
**채널 ID**를 가져오려면 채널 이름을 클릭하고 **View Info**를 선택하세요. 정보 패널에 채널 ID가 표시됩니다. 홈 채널을 수동으로 설정하려면 이 ID가 필요합니다.
:::

## 5단계: Hermes Agent 구성

### 옵션 A: 대화형 설정 (권장)

안내에 따라 설정하는 명령을 실행합니다.

```bash
hermes gateway setup
```

메시지가 표시되면 **Mattermost**를 선택한 다음 서버 URL, 봇 토큰, 사용자 ID를 붙여 넣습니다.

### 옵션 B: 수동 구성

`~/.hermes/.env` 파일에 다음을 추가합니다.

```bash
# Required
MATTERMOST_URL=https://mm.example.com
MATTERMOST_TOKEN=***
MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c

# Multiple allowed users (comma-separated)
# MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c,8fk2jd9s0a7bncm1xqw4tp6r3e

# Optional: reply mode (thread or off, default: off)
# MATTERMOST_REPLY_MODE=thread

# Optional: respond without @mention (default: true = require mention)
# MATTERMOST_REQUIRE_MENTION=false

# Optional: channels where bot responds without @mention (comma-separated channel IDs)
# MATTERMOST_FREE_RESPONSE_CHANNELS=channel_id_1,channel_id_2
```

선택적 동작 설정은 `~/.hermes/config.yaml`에 추가합니다.

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true`는 공유 채널과 스레드에서 각 참여자의 컨텍스트를 격리합니다

### 게이트웨이 시작

구성이 완료되면 Mattermost 게이트웨이를 시작합니다.

```bash
hermes gateway
```

봇이 몇 초 안에 Mattermost 서버에 연결되어야 합니다. DM이나 봇이 추가된 채널에서 메시지를 보내 테스트하세요.

:::tip
지속적으로 실행하려면 `hermes gateway`를 백그라운드나 systemd 서비스로 실행할 수 있습니다. 자세한 내용은 배포 문서를 참조하세요.
:::

## 홈 채널

봇이 사전 메시지(예: cron 작업 출력, 리마인더, 알림)를 보내는 "홈 채널"을 지정할 수 있습니다. 설정 방법은 두 가지입니다.

### 슬래시 명령 사용

봇이 참여한 Mattermost 채널에서 `/sethome`을 입력합니다. 해당 채널이 홈 채널이 됩니다.

### 수동 구성

`~/.hermes/.env`에 다음을 추가합니다.

```bash
MATTERMOST_HOME_CHANNEL=abc123def456ghi789jkl012mn
```

ID를 실제 채널 ID로 바꿉니다(채널 이름 클릭 → View Info → ID 복사).

## 답장 모드

`MATTERMOST_REPLY_MODE` 설정은 Hermes가 응답을 게시하는 방식을 제어합니다.

| 모드 | 동작 |
|------|----------|
| `off` (기본값) | Hermes가 일반 사용자처럼 채널에 일반 메시지를 게시합니다. |
| `thread` | Hermes가 원래 메시지 아래 스레드로 답장합니다. 대화가 많이 오갈 때 채널을 깔끔하게 유지합니다. |

`~/.hermes/.env`에서 설정합니다.

```bash
MATTERMOST_REPLY_MODE=thread
```

## 멘션 동작

기본적으로 봇은 채널에서 `@mention`되었을 때만 응답합니다. 이 동작은 변경할 수 있습니다.

| 변수 | 기본값 | 설명 |
|----------|---------|-------------|
| `MATTERMOST_REQUIRE_MENTION` | `true` | 채널의 모든 메시지에 응답하려면 `false`로 설정합니다 (DM은 항상 작동). |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | _(없음)_ | `@mention` 없이도 봇이 응답할 채널 ID를 쉼표로 구분한 목록입니다. require_mention이 true여도 적용됩니다. |

Mattermost에서 채널 ID를 찾으려면 채널을 열고 채널 이름 헤더를 클릭한 다음 URL이나 채널 세부 정보에서 ID를 찾습니다.

봇이 `@mentioned`되면 처리 전에 멘션이 메시지에서 자동으로 제거됩니다.

## 채널 허용 목록(`allowed_channels`)

봇을 특정 Mattermost 채널 집합으로 제한합니다. 설정하면 봇은 목록에 ID가 있는 채널에서만 응답합니다 — 봇이 `@mentioned`되었더라도 다른 채널의 메시지는 조용히 무시됩니다.

**DM은 이 필터에서 제외**되므로 승인된 사용자는 언제나 DM으로 봇에 접근할 수 있습니다.

```yaml
mattermost:
  allowed_channels:
    - "abc123def456ghi789jkl012mno"   # #ops
    - "xyz987uvw654rst321opq098nml"   # #incident-response
```

또는 환경 변수(쉼표로 구분)를 사용할 수 있습니다.

```bash
MATTERMOST_ALLOWED_CHANNELS="abc123def456ghi789jkl012mno,xyz987uvw654rst321opq098nml"
```

동작:

- 비어 있거나 설정되지 않음 → 제한 없음(완전한 하위 호환)
- 비어 있지 않음 → 다른 게이팅(멘션 요구, `MATTERMOST_FREE_RESPONSE_CHANNELS` 등)이 실행되기 전에 채널 ID가 목록에 있어야 하며, 그렇지 않으면 메시지가 삭제됩니다
- 채널 ID는 Mattermost UI → 채널 헤더 → "View Info"에서 찾거나 채널 URL에서 확인할 수 있습니다

참고: [관리자/사용자 슬래시 명령 분리](../../reference/slash-commands.md#permissions-and-adminuser-split).

## 문제 해결

### 봇이 메시지에 응답하지 않음

**원인**: 봇이 채널의 구성원이 아니거나 `MATTERMOST_ALLOWED_USERS`에 사용자 ID가 포함되어 있지 않습니다.

**해결 방법**: 채널에 봇을 추가하고(채널 이름 → Add Members → 봇 검색) 사용자 ID가 `MATTERMOST_ALLOWED_USERS`에 있는지 확인합니다. 게이트웨이를 다시 시작합니다.

### 403 Forbidden 오류

**원인**: 봇 토큰이 유효하지 않거나 봇에게 채널에 게시할 권한이 없습니다.

**해결 방법**: `.env` 파일의 `MATTERMOST_TOKEN`이 올바른지 확인합니다. 봇 계정이 비활성화되지 않았는지 확인합니다. 봇이 채널에 추가되었는지 확인합니다. 개인 액세스 토큰을 사용한다면 계정에 필요한 권한이 있는지 확인합니다.

### WebSocket 연결 해제 / 재연결 반복

**원인**: 네트워크 불안정, Mattermost 서버 재시작, WebSocket 연결 관련 방화벽/프록시 문제입니다.

**해결 방법**: 어댑터는 지수 백오프(2초 → 60초)로 자동 재연결합니다. 서버의 WebSocket 구성을 확인하세요 — 역방향 프록시(nginx, Apache)는 WebSocket 업그레이드 헤더를 구성해야 합니다. Mattermost 서버에서 WebSocket 연결을 차단하는 방화벽이 없는지 확인합니다.

nginx의 경우 구성에 다음이 포함되어 있는지 확인합니다.

```nginx
location /api/v4/websocket {
    proxy_pass http://mattermost-backend;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 600s;
}
```

### 시작 시 "Failed to authenticate"

**원인**: 토큰 또는 서버 URL이 올바르지 않습니다.

**해결 방법**: `MATTERMOST_URL`이 Mattermost 서버를 가리키는지 확인합니다(`https://` 포함, 끝에 슬래시 없음). `MATTERMOST_TOKEN`이 유효한지 확인하세요 — curl로 테스트할 수 있습니다.

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/api/v4/users/me
```

봇의 사용자 정보가 반환되면 토큰이 유효합니다. 오류가 반환되면 토큰을 다시 생성하세요.

### 봇이 오프라인 상태

**원인**: Hermes 게이트웨이가 실행 중이 아니거나 연결에 실패했습니다.

**해결 방법**: `hermes gateway`가 실행 중인지 확인합니다. 터미널 출력에서 오류 메시지를 확인하세요. 일반적인 문제는 잘못된 URL, 만료된 토큰, Mattermost 서버에 연결할 수 없는 경우입니다.

### "User not allowed" / 봇이 응답하지 않음

**원인**: 사용자 ID가 `MATTERMOST_ALLOWED_USERS`에 없습니다.

**해결 방법**: `~/.hermes/.env`의 `MATTERMOST_ALLOWED_USERS`에 사용자 ID를 추가하고 게이트웨이를 다시 시작합니다. 사용자 ID는 `@username`이 아니라 26자리 영숫자 문자열이라는 점을 기억하세요.

## 채널별 프롬프트

특정 Mattermost 채널에 임시 시스템 프롬프트를 할당합니다. 프롬프트는 실행 시 각 턴에 삽입되며 대화 기록에는 저장되지 않으므로 변경 사항이 즉시 적용됩니다.

```yaml
mattermost:
  channel_prompts:
    "channel_id_abc123": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "channel_id_def456": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

키는 Mattermost 채널 ID입니다(채널 URL이나 API에서 찾을 수 있음). 일치하는 채널의 모든 메시지에 임시 시스템 지침으로 프롬프트가 삽입됩니다.

## 보안

:::warning
봇과 상호작용할 수 있는 사용자를 제한하려면 항상 `MATTERMOST_ALLOWED_USERS`를 설정하세요. 이 값이 없으면 안전 조치로 게이트웨이가 기본적으로 모든 사용자를 거부합니다. 도구 사용과 시스템 액세스를 포함한 에이전트 기능에 대한 전체 액세스 권한이 있으므로 신뢰하는 사람의 사용자 ID만 추가하세요.
:::

Hermes Agent 배포 보안에 관한 자세한 내용은 [보안 가이드](../security.md)를 참조하세요.

## 참고

- **셀프 호스팅 친화적**: 모든 셀프 호스팅 Mattermost 인스턴스에서 작동합니다. Mattermost Cloud 계정이나 구독이 필요하지 않습니다.
- **추가 의존성 없음**: 어댑터는 HTTP와 WebSocket에 `aiohttp`를 사용하며, 이미 Hermes Agent에 포함되어 있습니다.
- **Team Edition 호환**: Mattermost Team Edition(무료)과 Enterprise Edition 모두에서 작동합니다.
