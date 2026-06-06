# Mattermost

[Mattermost](https://mattermost.com/)는 개발자 협업을 위한 오픈소스 대안입니다. Mattermost 어댑터는 실시간 이벤트, 파일 업로드/다운로드, 마크다운 렌더링 및 에이전트와 대화하는 다양한 방법을 지원하는 풍부한 기능의 통합을 제공합니다.

## 전제 조건

1. 관리 권한이 있는 Mattermost 서버 (자체 호스팅 또는 클라우드).
2. Hermes가 이벤트를 폴링하고 메시지를 보낼 수 있는 개인 액세스 토큰(Personal Access Token)을 가진 "Bot" 계정.

### 봇 계정 생성

1. Mattermost에서 **Integrations → Bot Accounts**로 이동합니다. (보이지 않는다면 관리자가 시스템 콘솔에서 봇 생성을 활성화해야 합니다).
2. **Add Bot Account**를 클릭합니다.
3. 세부 정보 입력:
   - **Username**: `hermes` (또는 원하는 이름)
   - **Display Name**: Hermes Agent
   - **Description**: AI Assistant
   - **Role**: 기본 봇은 사용자가 초대하는 위치에서 작동합니다. `System Admin` 역할을 부여하면 봇이 모든 공개 채널에 자동으로 참여하고, 서버 전체의 이벤트를 볼 수 있으며, 초대 없이 사용자에게 직접 메시지를 보낼 수 있습니다.
4. 봇을 생성하고 제공된 **액세스 토큰(Access Token)**을 복사합니다.

## 환경 변수 구성

`~/.hermes/.env`에 자격 증명을 설정하세요:

```bash
# Mattermost 서버 URL (반드시 http:// 또는 https:// 포함)
MATTERMOST_URL=https://mattermost.example.com

# 봇용 액세스 토큰
MATTERMOST_TOKEN=your-bot-access-token
```

### 고급 구성

추가 환경 변수 옵션:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MATTERMOST_URL` | — | Mattermost 서버 기본 URL |
| `MATTERMOST_TOKEN` | — | 봇 개인 액세스 토큰 |
| `MATTERMOST_ALLOWED_USERS` | (모두) | 상호작용이 허용된 사용자 이름의 쉼표 구분 목록 |
| `MATTERMOST_TEAM` | — | 설정 시 봇이 부팅 시 이 팀의 모든 공개 채널에 참여합니다 |
| `MATTERMOST_INSECURE` | `false` | 인증서 오류를 무시하려면 `true`로 설정 (로컬 테스트용) |

또한, `~/.hermes/config.yaml`의 `extra` 설정을 통해 동작을 제어할 수 있습니다:

```yaml
platforms:
  mattermost:
    enabled: true
    extra:
      # 메시지 스레드를 별도의 에이전트 세션으로 처리
      # (false인 경우 채널 내의 모든 메시지가 하나의 세션을 공유함)
      thread_isolation: true
```

## 에이전트와 대화하기

어댑터는 폴링을 사용하여 WebSocket에 의존하지 않고 Mattermost API에서 실시간으로 이벤트를 가져옵니다.

Hermes와 다음과 같이 상호작용할 수 있습니다:

1. **다이렉트 메시지(DM)**: 봇에게 보내는 모든 메시지가 처리됩니다.
2. **채널 멘션**: 채널에서 `@hermes`를 멘션합니다. 봇은 채널에서 응답합니다.
3. **스레드 답장**: 봇 메시지가 포함된 스레드에서 대화하는 경우, 후속 메시지에서 멘션할 필요가 없습니다.

## 스레드 및 세션 격리 (Thread and Session Isolation)

기본적으로 (`thread_isolation: true`), Mattermost의 스레드는 별도의 에이전트 대화 세션으로 처리됩니다.

- 봇에게 보내는 DM은 단일 루트 대화 공간입니다.
- 채널에서 봇을 멘션하면 해당 채널에 바인딩된 세션이 시작됩니다.
- 봇 메시지에 대한 "답장(Reply)"을 통해 스레드를 시작하면, 해당 스레드는 루트 채널과 별개의 고유한 메모리 컨텍스트를 가진 격리된 에이전트 세션이 됩니다. 이렇게 하면 봇과 여러 가지 다른 주제에 대해 혼동 없이 동시에 채팅할 수 있습니다.

이 기능을 비활성화하려면 구성을 참조하세요.

## 파일 처리

어댑터는 Mattermost의 파일 첨부 기능을 완벽하게 지원합니다:

- **입력(Input)**: 메시지와 함께 파일을 업로드하면 Hermes가 이를 다운로드하여 읽고 쿼리에 컨텍스트로 사용합니다. 이미지, 문서, 코드 등을 지원합니다.
- **출력(Output)**: Hermes는 응답에 생성되거나 검색된 파일을 첨부할 수 있습니다.

## 슬래시 명령어(Slash Commands) 및 도구

Mattermost 통합은 Hermes의 핵심 슬래시 명령어와 완벽하게 작동합니다:

- `/help` - 사용 가능한 명령어 보기
- `/model` - 제공자/모델 변경
- `/clear` - 세션 기록 지우기
- `/prompt` - 시스템 프롬프트 업데이트

명령어는 봇에게 보내는 DM에서만 지원되며, 공개 채널에서는 명령어가 일반 메시지로 해석되는 것을 방지하기 위해 지원되지 않습니다.

## 문제 해결

**봇이 채널 메시지에 응답하지 않습니다.**
봇이 채널에 초대되었는지 확인하세요. 사용자가 `@bot-name`을 멘션하면 봇이 메시지를 볼 수 있어야 합니다. 봇에게 `System Admin` 역할을 부여하면 이 과정이 단순화되지만, 민감한 환경에서는 권장되지 않습니다.

**`mattermost: Connection refused`**
`MATTERMOST_URL`이 올바르고 프로토콜(`http://` 또는 `https://`)을 포함하고 있는지, 그리고 Hermes를 실행하는 호스트에서 해당 URL에 접근할 수 있는지 확인하세요. 자체 서명된 인증서를 사용하는 테스트 환경의 경우 `MATTERMOST_INSECURE=true`를 설정하세요.

**파일이 작동하지 않습니다.**
Mattermost 서버 구성이 봇의 파일 업로드 및 다운로드를 허용하는지 확인하세요. 이는 Mattermost 시스템 콘솔 설정에 의해 제한될 수 있습니다.
