---
sidebar_position: 11
title: "ACP 호스트 통합"
description: "ACP 호환 편집기와 협업 플랫폼에서 Hermes Agent 사용하기"
---

# ACP 호스트 통합

Hermes Agent는 ACP 서버로 실행할 수 있으므로 ACP 호환 호스트가 stdio를
통해 Hermes와 통신할 수 있습니다. 편집기는 다음을 렌더링할 수 있습니다:

- 채팅 메시지
- 도구 활동
- 파일 diff
- 터미널 명령
- 승인 요청
- 스트리밍되는 사고 / 응답 청크

다른 호스트도 동일한 프로토콜을 사용해 협업 이벤트를 Hermes로 라우팅할 수
있습니다. 다른 애플리케이션이 대화 전송을 담당하는 동안 Hermes가 기존
정체성, 공급자 설정, 메모리, 스킬, 도구를 유지하길 원한다면 ACP가 잘
맞습니다.

## ACP 모드에서 Hermes가 제공하는 기능

Hermes는 편집기 워크플로를 위해 선별된 `hermes-acp` 도구 세트로
실행됩니다. 여기에는 다음이 포함됩니다:

- 파일 도구: `read_file`, `write_file`, `patch`, `search_files`
- 터미널 도구: `terminal`, `process`
- 웹/브라우저 도구
- 메모리, 할 일, 세션 검색
- 스킬
- execute_code 및 delegate_task
- 비전

메시지 전송이나 cronjob 관리처럼 일반적인 편집기 UX에 맞지 않는 기능은
의도적으로 제외됩니다.

## 설치

Hermes를 일반적인 방법으로 설치한 다음, 설치 checkout에서 ACP extra를
추가합니다:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

이렇게 하면 `agent-client-protocol` 의존성이 설치되고 다음 기능이
활성화됩니다:

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

## ACP 서버 시작

다음 명령 중 하나를 실행하면 Hermes가 ACP 모드로 시작됩니다:

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

Hermes는 stderr에 로그를 기록하므로 stdout은 ACP JSON-RPC 트래픽 전용으로
남습니다.

대화형이 아닌 점검에는 다음을 사용합니다:

```bash
hermes acp --version
hermes acp --check
```

### 브라우저 도구 (선택 사항)

브라우저 도구(`browser_navigate`, `browser_click` 등)는 `agent-browser`
npm 패키지와 Chromium에 의존하며, Python wheel에는 포함되지 않습니다.
다음 명령으로 설치합니다:

```bash
hermes acp --setup-browser           # interactive (prompts before ~400 MB download)
hermes acp --setup-browser --yes     # accept the download non-interactively
```

이 명령은 독립 실행 명령입니다. 터미널 인증 흐름(`hermes acp --setup`)도
모델 선택 후 추가 질문으로 브라우저 부트스트랩을 제공합니다. 따라서
대부분의 사용자는 `--setup-browser`를 직접 실행할 필요가 없습니다.

수행하는 작업은 다음과 같습니다:

- 없으면 Node.js 26을 `~/.hermes/node/`에 설치
- 해당 prefix에 `npm install -g agent-browser @askjo/camofox-browser` 설치 (sudo 불필요 — `npm`의 `--prefix`가 사용자가 쓸 수 있는 Hermes 관리 Node를 가리킴)
- Playwright Chromium을 설치하거나, 사용 가능한 시스템 Chrome/Chromium이 감지되면 이를 사용

부트스트랩은 멱등적이므로 다시 실행해도 빠르게 완료되며 이미 끝난 작업은
건너뜁니다.

## 호스트 설정

### Buzz 채널 (릴레이 브리지)

[Buzz](https://github.com/block/buzz)는 사람과 에이전트를 위한 Nostr 기반
협업 플랫폼입니다. `buzz-acp` 하네스는 Buzz 채널을 stdio를 통해 모든 ACP
에이전트에 연결합니다:

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

이는 전송 통합이며, Hermes를 두 번째로 설치하는 것이 아닙니다.
`buzz-acp`가 시작하는 하위 프로세스는 해당 호스트에서 실행되는 `hermes`와
동일한 Hermes 설정, 자격 증명, 메모리, 스킬, 상태를 사용합니다.

(이는 [Buzz Desktop의 관리형 런타임](#buzz-desktop)과 다릅니다. 관리형
런타임은 Hermes를 사전 설정된 하네스로 로컬에서 시작합니다. 릴레이 브리지는
일반적으로 서버에서 에이전트 정체성으로 Buzz *채널*에 참여하기 위한
것입니다.)

필수 조건:

- 위의 ACP 설치와 `hermes acp --check`를 완료합니다.
- [Buzz 저장소](https://github.com/block/buzz)에서 `buzz-acp`와 `buzz` CLI를
  빌드합니다 (`cargo build --release -p buzz-acp`).
- Hermes 전용 Nostr 키 쌍을 발급하고(`buzz-admin generate-key`), 릴레이
  멤버로 등록합니다(`buzz-admin add-member`). 모든 에이전트는 고유한
  정체성을 가져야 하므로 사람의 키 쌍을 재사용하지 마세요.
- 해당 정체성을 사용할 Buzz 채널에 추가합니다.

다음 명령으로 브리지를 시작합니다:

```bash
export BUZZ_RELAY_URL="wss://community.example.com"
export BUZZ_PRIVATE_KEY="..."
export BUZZ_API_TOKEN="..."
export BUZZ_ACP_AGENT_COMMAND="hermes"
export BUZZ_ACP_AGENT_ARGS="acp"

buzz-acp
```

`BUZZ_API_TOKEN`은 릴레이가 토큰 인증을 적용할 때만 필요합니다. 개인 키나
API 토큰을 커밋하거나 붙여 넣지 마세요.

지속적인 서버 배포에서는 의도한 Hermes 홈 디렉터리를 소유한 동일한 운영
체제 사용자로 서비스 관리자를 통해 `buzz-acp`를 실행합니다. 설정, 키 생성,
채널 검색, 에이전트별 옵션은 [buzz-acp README](https://github.com/block/buzz/tree/main/crates/buzz-acp)에
문서화되어 있습니다.

브리지는 Hermes 정체성이 멤버인 모든 Buzz 채널을 검색하고, 다른 채널에
추가되면 자동으로 구독합니다. 따라서 Buzz 채널 멤버십이 계속해서 접근
경계를 담당하며, Hermes 자체 설정에 별도의 채널 목록이 필요하지 않습니다.

소유자의 Buzz Desktop에 Hermes ACP 활동을 표시하려면 다음을 추가합니다:

```bash
export BUZZ_ACP_RELAY_OBSERVER="true"
```

이렇게 하면 에이전트 소유자에게 지정된 암호화된 kind `24200` 관찰자
프레임(Buzz의 NIP-AO)이 게시됩니다. Desktop은 에이전트의 **활동 로그**에
실시간 수명 주기, 도구, 응답, 사용량 스트림을 렌더링합니다. 릴레이는 이
프레임을 임시 데이터로 처리하므로, 턴이 시작되기 전에 Desktop이 온라인
상태여야 합니다. 로컬 관찰자 아카이브가 소유자 측의 영구 기록입니다.

헤드리스 브리지는 승인 대화 상자를 표시할 편집기가 없으므로 ACP 권한
요청에 직접 응답합니다 — [Buzz 에이전트를 소유자 전용으로 유지](#keep-buzz-agents-owner-only)를
참조하세요. 브리지를 권한 있는 자동화로 취급하세요. 전용 운영 체제
계정을 사용하고, Buzz 사용자가 에이전트에 프롬프트할 수 있는 범위를
제한하며(`buzz-acp`는 `BUZZ_ACP_AGENT_OWNER`를 통한 소유자 전용 응답
게이트를 지원), Hermes가 작동해야 하는 채널에만 멤버십을 부여하세요.

### VS Code

[ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client) 확장을 설치합니다.

연결하려면:

1. Activity Bar에서 ACP Client 패널을 엽니다.
2. 기본 제공 에이전트 목록에서 **Hermes Agent**를 선택합니다.
3. 연결하고 채팅을 시작합니다.

Hermes를 수동으로 정의하려면 VS Code 설정의 `acp.agents` 아래에 추가합니다:

```json
{
  "acp.agents": {
    "Hermes Agent": {
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### Zed

Zed 설정에서 Hermes를 사용자 지정 에이전트 서버로 구성합니다:

1. Agent Panel을 엽니다.
2. 다음 설정으로 사용자 지정 에이전트 서버를 추가합니다:

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

3. 새 Hermes 외부 에이전트 스레드를 시작합니다.

필수 조건:

- 먼저 `hermes model`로 Hermes 공급자 자격 증명을 구성하거나,
  `~/.hermes/.env` / `~/.hermes/config.yaml`에 설정합니다.

### JetBrains

ACP 호환 플러그인을 사용하고 `hermes acp` 또는 `hermes-acp`를 지정합니다.

### Buzz Desktop

[Buzz](https://github.com/block/buzz)는 Hermes Agent를 사전 설정된 런타임으로
제공합니다. Hermes를 일반적인 방법으로 설치하면 Buzz가 자동으로 검색합니다
— **Settings → Runtimes**를 열면 런타임 목록에 Hermes가 나타납니다.

검색에 실패하면(이전 설치) ACP 런처가 로그인 셸의 PATH에서 확인되는지
확인합니다:

```bash
command -v hermes-acp || command -v hermes
```

최근 설치에서는 `~/.local/bin`에 `hermes`와 `hermes-acp` 런처를 모두
기록합니다. `hermes update`를 실행하면 이전 설치에 `hermes-acp` 런처가
추가됩니다. 수동 대안으로 Buzz의 에이전트 명령을 `hermes`로, 인수를
`["acp"]`로 구성합니다.

#### 모델 선택기

Buzz Desktop(v0.5.1 이상)은 에이전트의 런타임 설정에 Hermes의 전체 모델
메뉴를 표시합니다. 목록은 ACP를 통해 Hermes 자체에서 가져옵니다. 즉,
Hermes에서 인증한 공급자의 모든 모델이 표시됩니다(`hermes model`과
`/model` 명령에서 사용하는 동일한 목록). 따라서 메뉴에 없는 모델은
Hermes 측에서 해당 공급자의 자격 증명이 구성되지 않았다는 뜻입니다.

항목 ID는 `provider:model` 형식(예: `openrouter:z-ai/glm-5.1`) 또는
`config.yaml`에 정의된 사용자 지정 OpenAI 호환 엔드포인트의 경우
`custom:<name>:<model>` 형식입니다. 모델을 선택하면 해당 에이전트의
세션에 적용되며 Hermes 전체의 기본값은 변경하지 않습니다 — 기본값을
변경하려면 `hermes model`을 사용합니다.

#### Buzz 에이전트를 소유자 전용으로 유지

Buzz는 모든 에이전트의 **이 에이전트와 대화할 수 있는 사람**을
`Owner only`로 설정해 생성합니다. 런타임이 Hermes인 경우 이 설정을
그대로 유지하세요.

두 가지 동작이 이 경로에서 결합됩니다. `hermes-acp` 도구 세트에는
`terminal`과 `execute_code`가 포함되고, Buzz의 ACP 브리지는 Hermes의 권한
요청에 직접 응답하여 이를 표시하는 대신 `allow_once`로 처리합니다. 따라서
Buzz의 Hermes 에이전트는 묻지 않고 호스트에서 셸 명령을 실행합니다. 한
에이전트에게 스크래치 디렉터리에 `rm -rf`를 실행하도록 요청했더니 어디에도
확인 요청이 나타나지 않은 채 디렉터리가 삭제되었습니다.

`Anyone`을 선택하면 채널에 접근할 수 있는 모든 작성자에게 동일한 셸
접근 권한을 넘깁니다. Buzz는 이를 선택해도 경고하지 않습니다.

현재는 다음과 같은 명백한 완화책도 어느 것도 작동하지 않습니다:

- `approvals.mode: manual`은 Hermes가 권한 요청을 발생시키도록 하지만,
  Buzz가 이를 자동 승인하므로 명령은 여전히 실행됩니다.
- `platform_toolsets.acp`는 ACP 도구 세트를 좁히지 않으므로 `terminal`을
  제거하는 데 사용할 수 없습니다.

소유자가 보내는 `!shutdown`은 어떤 모드에서든 에이전트를 중지하지만, Buzz는
다른 사람이 보낸 해당 명령을 무시합니다.

## 설정 및 자격 증명

ACP 모드는 CLI와 동일한 Hermes 설정을 사용합니다:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

공급자 확인은 Hermes의 일반 런타임 확인기를 사용하므로 ACP는 현재 구성된
공급자와 자격 증명을 상속합니다. Hermes는 첫 실행 ACP 클라이언트를 위해
터미널 인증 방법(`--setup`)도 제공합니다. 이를 실행하면 Hermes의 대화형
모델/공급자 설정이 열립니다.

## 호스트 통합

다음 변수는 **ACP 호스트 프로세스**(편집기 또는 다른 에이전트 하네스)가
시작한 Hermes 하위 프로세스에 설정합니다. 사용자 설정이 아니므로 `.env`나
`config.yaml`에서 직접 설정하지 마세요.

| 변수 | 값 | 효과 |
|----------|-------|--------|
| `HERMES_ACP_SKIP_CONFIGURED_MCP` | `1` | ACP JSON-RPC 루프가 시작되기 전에 `config.yaml`에서 **전역으로 구성된** MCP 서버의 시작을 건너뜁니다. |

Hermes는 일반적으로 ACP JSON-RPC 루프에 진입하기 전에 `config.yaml`에
구성된 모든 MCP 서버를 시작합니다. 세션의 서버를 `session/new`를 통해
명시적으로 전달하여 MCP를 직접 관리하는 호스트는 이러한 전역 시작이
필요하지 않습니다. 그렇지 않으면 관련 없는 느린 MCP 서버나 대화형 MCP
서버가 `initialize`를 지연시킬 수 있습니다. 마커를 정확히 `1`로 설정하면
이러한 호스트가 해당 시작을 건너뛸 수 있습니다.

건너뛰는 대상은 전역 `config.yaml` 검색뿐입니다. **ACP 세션이 `session/new`를
통해 제공하는 MCP 서버는 계속 등록**되므로 호스트가 요청한 기능은 손실되지
않습니다. 다른 값(설정되지 않음, 빈 값, `0`, `false`)은 기본 동작을 유지하므로
관련 없는 참처럼 보이는 문자열이 MCP를 조용히 비활성화할 수 없습니다.

## 세션 동작

ACP 세션은 서버가 실행되는 동안 ACP 어댑터의 메모리 내 세션 관리자가
추적합니다.

각 세션에는 다음이 저장됩니다:

- 세션 ID
- 작업 디렉터리
- 선택한 모델
- 현재 대화 기록
- 취소 이벤트

기반 `AIAgent`는 여전히 Hermes의 일반 영속성/로깅 경로를 사용하지만,
ACP의 `list/load/resume/fork`는 현재 실행 중인 ACP 서버 프로세스로 범위가
제한됩니다.

## 작업 디렉터리 동작

ACP 세션은 편집기의 cwd를 Hermes 작업 ID에 연결하므로 파일 및 터미널
도구가 서버 프로세스의 cwd가 아니라 편집기 작업 공간을 기준으로
실행됩니다.

## 승인

위험한 터미널 명령은 승인 요청으로 편집기에 다시 전달할 수 있습니다.
ACP 승인 옵션은 CLI 흐름보다 단순합니다:

- 한 번 허용
- 항상 허용
- 거부

실제로 승인 요청을 볼 수 있는지는 호스트에 달려 있습니다. 호스트는
요청을 표시하지 않고 프로그래밍 방식으로 응답할 수 있으며, 이 경우 이
옵션은 전선상에만 존재하고 사람에게 전달되지 않습니다. Buzz Desktop이
이렇게 동작하므로, `approvals` 설정과 관계없이 이 경로를 무인 실행으로
취급하세요.

시간 초과 또는 오류가 발생하면 승인 브리지는 요청을 거부합니다.

### 세션 범위 편집 자동 승인

ACP는 *한 번 허용*과 *항상 허용* 사이에 세 번째 단계인 *세션에 대해 허용*을
제공합니다. 편집기의 권한 요청에서 이를 선택하면 현재 ACP 세션 내부에만
승인이 기록됩니다. 해당 세션에서는 이후 일치하는 모든 명령이 요청 없이
통과하지만, 새 ACP 세션을 시작하거나 편집기를 다시 시작하면 초기화되어
처음에는 다시 요청합니다.

| 옵션 | 편집기 레이블 | 범위 | 재시작 후 유지 |
|---|---|---|---|
| `allow_once` | 한 번 허용 | 이 도구 호출 한 번 | 아니요 |
| `allow_session` | 세션에 대해 허용 | 이 ACP 세션의 일치하는 모든 호출 | 아니요 — 세션 종료 시 삭제 |
| `allow_always` | 항상 허용 | 앞으로의 모든 세션 | 예 (Hermes 영구 허용 목록에 기록) |
| `deny` | 거부 | 이 도구 호출 한 번 | 아니요 |

작업 기간 동안 에이전트를 신뢰하지만 오래 유지되는 허용 목록 항목을
부여하고 싶지 않은 편집기 워크플로에서는 `allow_session`이 적절한
기본값입니다. 안전성 측면의 상충 관계는 분명합니다. 범위가 넓을수록
편집기의 개입은 줄어들지만, 오작동하는 에이전트(또는 프롬프트 인젝션)가
사용자가 알아차리기 전에 일으킬 수 있는 피해는 커집니다. 익숙하지 않은
명령에는 `allow_once`로 시작하고, 같은 패턴을 에이전트가 몇 차례 올바르게
실행한 뒤 `allow_session`으로 승격하며, 영원히 신뢰할 수 있는 진정한
멱등 명령(예: `git status`)에만 `allow_always`를 사용하세요.

ACP 브리지는 이 옵션을 Hermes의 내부 승인 의미로 매핑합니다.
`allow_always`는 CLI와 동일한 방식으로 영구 허용 목록 항목을 기록하는
반면, `allow_session`은 현재 ACP 세션의 프로세스 내 승인 캐시에만
영향을 줍니다.

## 문제 해결

### ACP 에이전트가 편집기에 나타나지 않음

다음을 확인합니다:

- 수동/로컬 개발 환경에서는 호스트 명령이 `hermes acp`를 가리키는지
  확인합니다.
- Hermes가 설치되어 있고 PATH에 있는지 확인합니다.
- ACP extra가 설치되어 있는지 확인합니다 (`cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'`).

### ACP가 시작되지만 즉시 오류 발생

다음 점검을 시도합니다:

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### 자격 증명 누락

ACP 모드는 Hermes의 기존 공급자 설정을 사용합니다. 다음으로 자격 증명을
구성합니다:

```bash
hermes model
```

또는 `~/.hermes/.env`를 편집합니다. 터미널 인증 흐름(`hermes acp --setup`)도
대화형 공급자/모델 설정을 시작할 수 있습니다.

## 함께 보기

- [Buzz ACP 하네스](https://github.com/block/buzz/tree/main/crates/buzz-acp)
- [ACP 내부 구조](../../developer-guide/acp-internals.md)
- [공급자 런타임 확인](../../developer-guide/provider-runtime.md)
- [도구 런타임](../../developer-guide/tools-runtime.md)
