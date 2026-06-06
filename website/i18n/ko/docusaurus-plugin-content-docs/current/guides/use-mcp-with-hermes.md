---
sidebar_position: 6
title: "Hermes와 함께 MCP 사용하기 (Use MCP with Hermes)"
description: "Hermes Agent에 MCP 서버를 연결하고, 도구를 필터링하며, 실제 워크플로우에서 안전하게 사용하기 위한 실용 가이드"
---

# Hermes와 함께 MCP 사용하기 (Use MCP with Hermes)

이 가이드는 실제 일상적인 워크플로우에서 Hermes Agent와 함께 MCP를 사용하는 방법을 보여줍니다.

기능 페이지가 MCP가 무엇인지 설명한다면, 이 가이드는 MCP를 빠르고 안전하게 활용하여 가치를 얻는 방법에 초점을 맞춥니다.

## 언제 MCP를 사용해야 할까요?

다음과 같은 경우 MCP를 사용하세요:
- MCP 형태로 이미 도구가 존재하여 네이티브 Hermes 도구를 직접 만들고 싶지 않을 때
- 깔끔한 RPC 계층을 통해 Hermes가 로컬 또는 원격 시스템에 대해 작동하도록 하고 싶을 때
- 각 서버에 노출되는 도구를 세밀하게 제어하고 싶을 때
- Hermes의 코어 코드를 수정하지 않고 내부 API, 데이터베이스 또는 회사 시스템에 Hermes를 연결하고 싶을 때

다음과 같은 경우에는 MCP를 사용하지 마세요:
- 기본 내장된 Hermes 도구로 이미 작업을 잘 해결할 수 있을 때
- 서버가 수많은 위험한 도구 영역을 노출하는데 여러분이 이를 필터링할 준비가 되지 않았을 때
- 매우 제한적인 연동 하나만 필요한 상황이라 네이티브 도구를 작성하는 편이 더 간단하고 안전할 때

## 멘탈 모델 (Mental model)

MCP를 일종의 어댑터 계층(adapter layer)이라고 생각하세요:

- Hermes는 여전히 에이전트 역할을 유지합니다.
- MCP 서버는 도구를 제공합니다.
- Hermes는 시작 시 또는 다시 로드될 때 이러한 도구들을 검색(discover)합니다.
- 모델은 이 도구들을 일반적인 도구처럼 사용할 수 있습니다.
- 여러분은 각 서버가 얼마나 많은 도구를 노출할지 통제할 수 있습니다.

마지막 부분이 매우 중요합니다. MCP를 잘 사용한다는 것은 단순히 "모든 것을 연결하는 것"이 아닙니다. "가장 작고 유용한 영역만 남기고, 적절한 것을 연결하는 것"을 의미합니다.

## 1단계: MCP 지원 패키지 설치

표준 설치 스크립트로 Hermes를 설치했다면 MCP 지원 기능이 이미 포함되어 있습니다 (설치 프로그램은 `uv pip install -e ".[all]"`을 실행합니다).

extras 옵션 없이 설치한 뒤에 별도로 MCP를 추가해야 하는 경우:

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

npm 기반 서버의 경우 Node.js 및 `npx`를 사용할 수 있는지 확인하세요.

대부분의 Python MCP 서버에서는 `uvx`를 사용하는 것이 좋습니다.

## 2단계: 먼저 하나의 서버 추가해 보기

안전한 하나의 서버부터 시작해 보세요.

예시: 오직 하나의 프로젝트 디렉토리에 대해서만 파일 시스템 접근 권한 부여하기.

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

그런 다음 Hermes를 시작합니다:

```bash
hermes chat
```

이제 구체적인 질문을 던져보세요:

```text
이 프로젝트를 검사하고 저장소 구조를 요약해 줘.
```

## 3단계: MCP 로드 확인

몇 가지 방법으로 MCP를 확인할 수 있습니다:

- 설정이 완료되면 Hermes 배너/상태에서 MCP 통합 정보가 표시되어야 합니다.
- 사용 가능한 도구가 무엇인지 Hermes에게 질문하세요.
- 설정 변경 후 `/reload-mcp`를 사용하세요.
- 서버가 연결에 실패한 경우 로그를 확인하세요.

실용적인 테스트 프롬프트:

```text
지금 사용 가능한 MCP 지원 도구가 어떤 것들이 있는지 알려줘.
```

## 4단계: 즉시 필터링 시작하기

서버가 많은 도구를 노출하는 경우 나중으로 미루지 말고 바로 필터링하세요.

### 예시: 원하는 것만 허용 목록(Whitelist)에 넣기

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
```

이것은 민감한 시스템에 대해 일반적으로 가장 권장되는 기본 설정입니다.

## WSL2: WSL의 Hermes를 Windows Chrome과 브릿징하기

이 설정은 다음과 같은 경우에 실용적입니다:

- Hermes가 WSL2 내부에서 실행됩니다.
- 제어하고 싶은 브라우저가 Windows에서 로그인하여 평소 사용하는 Chrome입니다.
- `/browser connect` 명령어가 WSL에서 다루기 번거롭거나 불안정할 때입니다.

이 설정에서 Hermes는 Chrome에 직접 연결되지 **않습니다**. 대신:

- Hermes는 WSL에서 실행됩니다.
- Hermes는 로컬 stdio MCP 서버를 시작합니다.
- 이 MCP 서버는 Windows 상호운용성 기능(`cmd.exe` 또는 `powershell.exe`)을 통해 시작됩니다.
- MCP 서버는 현재 사용 중인 라이브 Windows Chrome 세션에 연결됩니다.

멘탈 모델:

```text
Hermes (WSL) -> MCP stdio 브릿지 -> Windows Chrome
```

### 이 모드가 유용한 이유

- 실제 Windows 브라우저의 프로필, 쿠키 및 로그인을 그대로 유지할 수 있습니다.
- Hermes는 기본적으로 지원되는 Unix 환경(WSL2)에 그대로 머물 수 있습니다.
- 브라우저 제어가 Hermes 핵심 브라우저 트랜스포트에 의존하는 대신 MCP 도구로 노출됩니다.

### 권장되는 서버

`chrome-devtools-mcp`를 사용하세요.

Windows Chrome에서 `chrome://inspect/#remote-debugging`을 통해 라이브 원격 디버깅이 이미 활성화되어 있는 경우, WSL에서 다음과 같이 추가하세요:

```bash
hermes mcp add chrome-devtools-win --command cmd.exe --args /c npx -y chrome-devtools-mcp@latest --autoConnect --no-usage-statistics
```

서버를 저장한 후:

```bash
hermes mcp test chrome-devtools-win
```

그런 다음 새로운 Hermes 세션을 시작하거나 다음 명령어를 실행하세요:

```text
/reload-mcp
```

### 전형적인 프롬프트

로드되고 나면 Hermes는 MCP 접두사가 붙은 브라우저 도구를 직접 사용할 수 있습니다. 예시:

```text
MCP 도구 mcp_chrome_devtools_win_list_pages를 호출해서 현재 브라우저 탭들을 나열해 줘.
```

### `/browser connect`가 적합하지 않은 상황

Hermes가 WSL에서 실행되고 Chrome이 Windows에서 실행되는 경우, Chrome이 열려 있고 디버그 가능한 상태라도 `/browser connect`가 실패할 수 있습니다.

일반적인 이유:

- Windows 도구에 대해 Chrome이 노출하는 로컬호스트 엔드포인트에 WSL이 동일하게 접근할 수 없습니다.
- 최신 Chrome의 라이브 디버깅 흐름은 고전적인 `ws://localhost:9222` 방식과 동일하지 않습니다.
- 브라우저는 `chrome-devtools-mcp` 같은 Windows 측 헬퍼 프로그램에서 연결하기가 더 쉽습니다.

이러한 경우, `/browser connect`는 같은 환경(same-environment) 설정에 남겨두고, WSL에서 Windows 브라우저를 제어할 때(브릿징)는 MCP를 사용하세요.

### 주의 사항 (Known pitfalls)

- MCP를 통해 Windows stdio 실행 파일을 사용할 때는 `/mnt/c/Users/<you>`나 `/mnt/c/workspace/...`와 같이 Windows에 마운트된 경로에서 Hermes를 시작하세요.
- `/root`나 `/home/...`에서 Hermes를 시작할 경우, MCP 서버가 시작되기 전에 Windows에서 `UNC` 현재 디렉터리 경로 경고를 낼 수 있습니다.
- `chrome-devtools-mcp --autoConnect`가 페이지를 나열하는 동안 시간 초과가 발생하면 Chrome에서 백그라운드나 동결된 탭을 줄이고 다시 시도하세요.

### 예시: 위험한 행동 차단(Blacklist)하기

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### 예시: 유틸리티 래퍼 비활성화하기

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

## 필터링은 실제 어떤 영향을 주나요?

Hermes에서 MCP를 통해 노출되는 기능에는 두 가지 범주가 있습니다:

1. 서버 네이티브 MCP 도구
- 필터링에 사용하는 옵션:
  - `tools.include`
  - `tools.exclude`

2. Hermes가 추가하는 유틸리티 래퍼(wrappers)
- 필터링에 사용하는 옵션:
  - `tools.resources`
  - `tools.prompts`

### 볼 수 있는 유틸리티 래퍼

Resources(자원):
- `list_resources`
- `read_resource`

Prompts(프롬프트):
- `list_prompts`
- `get_prompt`

이러한 래퍼는 다음 두 조건을 모두 만족할 때만 표시됩니다:
- 설정 파일에서 허용됨
- MCP 서버 세션이 실제로 해당 기능(capability)들을 지원함

따라서 Hermes는 서버가 프롬프트나 리소스를 가지고 있지 않은데도 가진 척하지 않습니다.

## 일반적인 패턴 (Common patterns)

### 패턴 1: 로컬 프로젝트 어시스턴트

Hermes가 제한된 워크스페이스 내에서 추론하도록 만들고 싶을 때, 저장소 전용(repo-local) 파일 시스템이나 git 서버용으로 MCP를 사용하세요.

```yaml
mcp_servers:
  fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/project"]
```

유용한 프롬프트:

```text
프로젝트 구조를 검토하고 설정 파일이 어디에 있는지 파악해 줘.
```

```text
로컬 git 상태를 확인하고 최근에 변경된 내용을 요약해 줘.
```

### 패턴 2: GitHub 심사(triage) 어시스턴트

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false
```

유용한 프롬프트:

```text
MCP에 대한 열려있는 이슈를 나열하고 주제별로 그룹화한 뒤, 가장 흔한 버그에 대해 양질의 이슈를 초안으로 작성해 줘.
```

```text
코드베이스에서 _discover_and_register_server가 사용된 곳을 검색하고 MCP 도구가 등록되는 방식을 설명해 줘.
```

### 패턴 3: 내부 API 어시스턴트

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      include: [list_customers, get_customer, list_invoices]
      resources: false
      prompts: false
```

유용한 프롬프트:

```text
ACME Corp 고객을 찾아보고 최근 인보이스 활동을 요약해 줘.
```

이런 경우에서는 제외 목록(exclude list)보다 엄격한 허용 목록(whitelist)을 사용하는 것이 훨씬 낫습니다.

### 패턴 4: 문서화 / 지식 서버

어떤 MCP 서버들은 직접적인 동작(action)이라기보다 공유된 지식 자산에 가까운 프롬프트나 리소스를 노출하기도 합니다.

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: true
      resources: true
```

유용한 프롬프트:

```text
문서 서버에서 사용 가능한 MCP 리소스를 나열한 다음 온보딩 가이드를 읽고 요약해 줘.
```

```text
문서 서버가 노출하는 프롬프트를 나열하고 장애 대응(incident response)에 도움이 될 만한 것이 무엇인지 알려줘.
```

## 튜토리얼: 필터링이 포함된 End-to-end 설정

실제 구축 과정입니다.

### 1단계: 좁은 허용 목록으로 GitHub MCP 추가하기

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
      prompts: false
      resources: false
```

Hermes를 시작하고 물어보세요:

```text
코드베이스에서 MCP 참조를 검색하고 주요 연동 지점을 요약해 줘.
```

### 2단계: 필요할 때만 도구 확장하기

나중에 이슈 업데이트 기능도 필요해졌다면:

```yaml
tools:
  include: [list_issues, create_issue, update_issue, search_code]
```

그리고 리로드합니다:

```text
/reload-mcp
```

### 3단계: 다른 정책을 가진 두 번째 서버 추가하기

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false

  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]
```

이제 Hermes는 두 서버를 조합해서 활용할 수 있습니다:

```text
로컬 프로젝트 파일들을 검사하고 발견한 버그를 요약하는 GitHub 이슈를 생성해 줘.
```

이것이 MCP가 강력해지는 지점입니다. Hermes 코어를 변경하지 않고도 다양한 시스템 간 워크플로우를 구성할 수 있습니다.

## 안전한 사용을 위한 권장 사항

### 위험한 시스템은 허용 목록(Allowlists)을 우선할 것

재무 관련, 고객 관련, 파괴적인 성격이 있는 모든 것:
- `tools.include` 사용
- 가능한 최소한의 세트로 시작할 것

### 사용하지 않는 유틸리티 비활성화

모델이 서버에서 제공하는 리소스/프롬프트를 탐색하지 않게 하려면 이를 끄세요:

```yaml
tools:
  resources: false
  prompts: false
```

### 서버 범위는 좁게 유지할 것

예시:
- 전체 홈 디렉토리가 아닌 하나의 프로젝트 디렉토리에만 맞춘 파일 시스템 서버
- 하나의 저장소를 바라보는 git 서버
- 기본적으로 읽기 전용 작업으로 많이 노출시킨 내부 API 서버

### 설정 변경 후 다시 로드하기

```text
/reload-mcp
```

다음 사항들을 변경한 후 이 명령어를 실행하세요:
- include/exclude 목록
- enabled 플래그
- resources/prompts 토글 스위치
- 인증(Auth) 헤더 / env 설정값

## 증상별 문제 해결 (Troubleshooting by symptom)

### "서버가 연결되었지만 기대한 도구가 보이지 않습니다"

가능한 원인:
- `tools.include`로 필터링됨
- `tools.exclude`에서 제외됨
- `resources: false` 또는 `prompts: false`를 통해 유틸리티 래퍼 비활성화됨
- 서버가 리소스/프롬프트를 실제로 지원하지 않음

### "서버가 설정되어 있는데 아무것도 로드되지 않습니다"

확인할 사항:
- `enabled: false`가 설정 파일에 남아있지 않은지
- 명령어/런타임이 존재하는지 (`npx`, `uvx` 등)
- HTTP 엔드포인트에 접속할 수 있는지
- auth 환경변수 또는 헤더가 올바른지

### "MCP 서버가 광고하는 것보다 적은 수의 도구가 표시되는 이유는 무엇인가요?"

Hermes는 서버 단위의 정책과 기능을 인지하는 등록 과정을 존중하기 때문입니다. 이것은 예상된 작동이며 대개 바람직한 결과입니다.

### "설정을 삭제하지 않고 MCP 서버를 제거하려면 어떻게 해야 하나요?"

다음을 사용하세요:

```yaml
enabled: false
```

설정 내용은 유지되지만 서버의 연결과 등록은 방지합니다.

## 권장하는 첫 MCP 설정

대부분의 사용자에게 좋은 첫 서버:
- 파일 시스템 (filesystem)
- git
- GitHub
- 페치(fetch) / 문서 MCP 서버
- 범위가 좁은 내부 API 하나

별로 좋지 않은 첫 서버:
- 필터링 없이 수많은 파괴적인 작업을 포함한 거대한 비즈니스 시스템
- 제약 조건을 걸 만큼 당신이 제대로 이해하지 못하는 시스템

## 관련 문서 (Related docs)

- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [자주 묻는 질문 (FAQ)](/reference/faq)
- [슬래시 명령어 (Slash Commands)](/reference/slash-commands)
