---
sidebar_position: 11
title: "ACP 편집기 통합"
description: "VS Code, Zed, JetBrains 등 ACP 호환 편집기 내부에서 Hermes Agent 사용하기"
---

# ACP 편집기 통합 (ACP Editor Integration)

Hermes Agent는 ACP 서버로 실행될 수 있어, ACP 호환 편집기가 stdio를 통해 Hermes와 통신하고 다음을 렌더링할 수 있게 해줍니다:

- 채팅 메시지 (chat messages)
- 도구 활동 (tool activity)
- 파일 변경 사항 (file diffs)
- 터미널 명령 (terminal commands)
- 승인 프롬프트 (approval prompts)
- 스트리밍되는 생각 / 응답 청크 (streamed thinking / response chunks)

ACP는 독립형 CLI나 메시징 봇이 아닌 편집기 기본 코딩 에이전트처럼 Hermes가 동작하기를 원할 때 적합합니다.

## Hermes가 ACP 모드에서 노출하는 것

Hermes는 편집기 작업 흐름을 위해 특별히 설계된 `hermes-acp` 도구 세트를 사용하여 실행됩니다. 여기에는 다음이 포함됩니다:

- 파일 도구: `read_file`, `write_file`, `patch`, `search_files`
- 터미널 도구: `terminal`, `process`
- 웹/브라우저 도구
- 메모리, 할 일(todo), 세션 검색
- 스킬 (skills)
- execute_code 및 delegate_task
- 비전 (vision)

메시지 전달 및 크론 작업(cronjob) 관리와 같이 일반적인 편집기 사용자 경험(UX)에 맞지 않는 기능은 의도적으로 제외합니다.

## 설치 (Installation)

Hermes를 정상적으로 설치한 후, ACP 추가 기능을 설치합니다:

```bash
pip install -e '.[acp]'
```

이 명령은 `agent-client-protocol` 종속성을 설치하고 다음을 활성화합니다:

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

Zed 레지스트리 설치의 경우, Zed는 공식 ACP 레지스트리 항목을 통해 Hermes를 실행합니다. 해당 항목은 다음을 실행하는 `uvx` 배포판을 사용합니다:

```bash
uvx --from 'hermes-agent[acp]==<version>' hermes-acp
```

레지스트리 설치 경로를 사용하기 전에 `PATH`에서 `uv`를 사용할 수 있는지 확인하세요.

## ACP 서버 실행

다음 중 하나를 사용하여 ACP 모드에서 Hermes를 시작할 수 있습니다:

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

Hermes는 stderr에 로그를 기록하므로, stdout은 ACP JSON-RPC 트래픽을 위해 예약된 상태로 유지됩니다.

비대화형 확인을 위해 다음을 사용할 수 있습니다:

```bash
hermes acp --version
hermes acp --check
```

### 브라우저 도구 (선택 사항)

브라우저 도구(`browser_navigate`, `browser_click` 등)는 `agent-browser` npm 패키지 및 Chromium에 의존하며, 이는 Python 휠(wheel)의 일부가 아닙니다. 다음 명령으로 설치하세요:

```bash
hermes acp --setup-browser           # 대화형 (약 400MB 다운로드 전 프롬프트 표시)
hermes acp --setup-browser --yes     # 비대화형으로 다운로드 수락
```

이것은 독립형 명령입니다. Zed 레지스트리의 터미널 인증 흐름(`hermes acp --setup`)은 모델 선택 후 후속 질문으로 브라우저 부트스트랩을 제공하므로, 대부분의 사용자는 `--setup-browser`를 직접 실행할 필요가 없습니다.

작동 방식:

- 누락된 경우 `~/.hermes/node/`에 Node.js 22 LTS 설치
- 해당 접두사에 `npm install -g agent-browser @askjo/camofox-browser` 실행 (sudo 불필요 — `npm`의 `--prefix`가 사용자가 쓰기 가능한 Hermes 관리형 Node를 가리킴)
- Playwright Chromium 설치 또는 사용 가능한 경우 감지된 시스템 Chrome/Chromium 사용

이 부트스트랩은 멱등성(idempotent)을 가지므로 다시 실행해도 빠르며 이미 완료된 작업은 건너뜁니다.

## 편집기 설정

### VS Code

[ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client) 확장 프로그램을 설치합니다.

연결 방법:

1. 활동 막대(Activity Bar)에서 ACP Client 패널을 엽니다.
2. 내장된 에이전트 목록에서 **Hermes Agent**를 선택합니다.
3. 연결하고 채팅을 시작합니다.

Hermes를 수동으로 정의하려면 VS Code 설정의 `acp.agents` 아래에 추가하세요:

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

Zed v0.221.x 이상에서는 공식 ACP 레지스트리를 통해 외부 에이전트를 설치합니다.

1. 에이전트 패널을 엽니다.
2. **Add Agent**를 클릭하거나 `zed: acp registry` 명령을 실행합니다.
3. **Hermes Agent**를 검색합니다.
4. 이를 설치하고 새로운 Hermes 외부 에이전트 스레드를 시작합니다.

필수 조건:

- 먼저 `hermes model`을 사용하거나 `~/.hermes/.env` / `~/.hermes/config.yaml`에서 Hermes 제공자 자격 증명을 구성합니다.
- 레지스트리 런처가 `uvx --from 'hermes-agent[acp]==<version>' hermes-acp`를 실행할 수 있도록 `uv`를 설치합니다.

레지스트리 항목을 사용하기 전의 로컬 개발의 경우 Zed 설정에서 사용자 지정 에이전트 서버를 사용하세요:

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

### JetBrains

ACP 호환 플러그인을 사용하고 다음을 가리키도록 설정하세요:

```text
/path/to/hermes-agent/acp_registry
```

## 레지스트리 매니페스트 (Registry manifest)

Hermes의 공식 ACP 레지스트리 메타데이터의 소스 복사본은 다음에 위치합니다:

```text
acp_registry/agent.json
acp_registry/icon.svg
```

업스트림 레지스트리 PR은 해당 파일들을 `agentclientprotocol/registry` 내부의 최상위 `hermes-agent/` 디렉토리에 복사합니다.

레지스트리 항목은 `hermes-agent` PyPI 릴리스를 직접 가리키는 `uvx` 배포판을 사용합니다:

```text
uvx --from 'hermes-agent[acp]==<version>' hermes-acp
```

레지스트리 CI는 고정된 버전이 PyPI에 존재하는지 확인하므로, 매니페스트의 `version`과 uvx `package` 고정은 항상 `pyproject.toml`과 일치해야 합니다. `scripts/release.py`는 이들을 자동으로 동기화된 상태로 유지합니다.

## 구성 및 자격 증명 (Configuration and credentials)

ACP 모드는 CLI와 동일한 Hermes 구성을 사용합니다:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

제공자 분석(Provider resolution)은 Hermes의 일반 런타임 리졸버를 사용하므로, ACP는 현재 구성된 제공자와 자격 증명을 상속합니다. Hermes는 또한 첫 실행 레지스트리 클라이언트를 위한 터미널 인증 방법(`--setup`)을 광고합니다. 이렇게 하면 Hermes의 대화형 모델/제공자 설정이 열립니다.

## 세션 동작 (Session behavior)

ACP 세션은 서버가 실행되는 동안 ACP 어댑터의 인메모리 세션 관리자에 의해 추적됩니다.

각 세션은 다음을 저장합니다:

- 세션 ID (session ID)
- 작업 디렉토리 (working directory)
- 선택된 모델 (selected model)
- 현재 대화 기록 (current conversation history)
- 취소 이벤트 (cancel event)

기본 `AIAgent`는 여전히 Hermes의 일반적인 지속성/로깅 경로를 사용하지만, ACP의 `list/load/resume/fork`는 현재 실행 중인 ACP 서버 프로세스의 범위 내에서 관리됩니다.

## 작업 디렉토리 동작 (Working directory behavior)

ACP 세션은 편집기의 cwd(현재 작업 디렉토리)를 Hermes 작업 ID에 바인딩하므로, 파일 및 터미널 도구는 서버 프로세스의 cwd가 아닌 편집기 작업 공간을 기준으로 실행됩니다.

## 승인 (Approvals)

위험한 터미널 명령은 편집기로 승인 프롬프트로 라우팅될 수 있습니다. ACP 승인 옵션은 CLI 흐름보다 더 간단합니다:

- 한 번 허용 (allow once)
- 항상 허용 (allow always)
- 거부 (deny)

시간 초과나 오류 발생 시, 승인 브리지는 요청을 거부합니다.

### 세션 범위 내 편집 자동 승인 (Session-scoped edit auto-approval)

ACP는 *한 번 허용*과 *항상 허용* 사이의 세 번째 단계를 노출합니다: **세션 동안 허용 (Allow for session)**. 편집기의 권한 프롬프트에서 이것을 선택하면 승인이 현재 ACP 세션 내부에만 기록됩니다. 해당 세션의 후속 일치 명령은 확인 없이 실행되지만, 새로운 ACP 세션을 시작하거나 편집기를 다시 시작하면 초기화되어 처음 발생 시 다시 프롬프트가 표시됩니다.

| 옵션 | 편집기 라벨 | 범위 | 재시작 시 유지 여부 |
|---|---|---|---|
| `allow_once` | Allow once (한 번 허용) | 현재 하나의 도구 호출 | 아니오 |
| `allow_session` | Allow for session (세션 동안 허용) | 이 ACP 세션에서 일치하는 모든 호출 | 아니오 — 세션이 끝나면 지워짐 |
| `allow_always` | Allow always (항상 허용) | 향후 모든 세션 | 예 (Hermes의 영구적인 허용 목록에 기록됨) |
| `deny` | Deny (거부) | 현재 하나의 도구 호출 | 아니오 |

`allow_session`은 작업이 진행되는 동안 에이전트를 신뢰하지만 수명이 긴 허용 목록 항목을 부여하고 싶지 않은 편집기 작업 흐름에 적합한 기본값입니다. 안전에 관한 장단점은 간단합니다: 범위가 넓을수록 편집기가 덜 방해하지만, 에이전트 오작동(또는 프롬프트 인젝션)이 눈치채기 전에 더 많은 피해를 줄 수 있습니다. 익숙하지 않은 명령에는 `allow_once`로 시작하고, 에이전트가 동일한 패턴을 여러 번 올바르게 실행하는 것을 확인한 후 `allow_session`으로 승격시키며, 항상 신뢰할 수 있는 진정한 멱등성 명령(예: `git status`)을 위해 `allow_always`를 보류해 두십시오.

ACP 브리지는 이러한 옵션들을 Hermes의 내부 승인 체계에 매핑합니다. `allow_always`는 CLI와 같은 방식으로 영구 허용 목록 항목을 기록하는 반면, `allow_session`은 현재 ACP 세션에 대한 프로세스 내 승인 캐시에만 영향을 미칩니다.

## 문제 해결 (Troubleshooting)

### 편집기에서 ACP 에이전트가 나타나지 않음

확인 사항:

- Zed에서 `zed: acp registry`로 ACP 레지스트리를 열고 **Hermes Agent**를 검색합니다.
- 수동/로컬 개발의 경우 사용자 지정 `agent_servers` 명령이 `hermes acp`를 가리키는지 확인합니다.
- Hermes가 설치되어 있고 PATH에 있는지 확인합니다.
- ACP 추가 기능이 설치되어 있는지 확인합니다(`pip install -e '.[acp]'`).
- 공식 Zed 레지스트리 항목에서 시작하는 경우 `uv`가 설치되어 있는지 확인합니다.

### ACP가 시작되지만 즉시 오류 발생

다음 검사를 시도해 보세요:

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### 누락된 자격 증명 (Missing credentials)

ACP 모드는 Hermes의 기존 제공자 설정을 사용합니다. 다음과 같이 자격 증명을 구성하세요:

```bash
hermes model
```

또는 `~/.hermes/.env`를 편집하여 구성할 수 있습니다. 레지스트리 클라이언트는 Hermes의 터미널 인증 흐름을 트리거할 수도 있으며, 이는 동일한 대화형 제공자/모델 설정을 실행합니다.

### Zed 레지스트리 런처가 uv를 찾지 못함

공식 uv 설치 문서에서 `uv`를 설치한 다음, Zed에서 Hermes Agent 스레드를 다시 시도하세요.

## 참고 항목

- [ACP Internals (ACP 내부)](../../developer-guide/acp-internals.md)
- [Provider Runtime Resolution (제공자 런타임 분석)](../../developer-guide/provider-runtime.md)
- [Tools Runtime (도구 런타임)](../../developer-guide/tools-runtime.md)
