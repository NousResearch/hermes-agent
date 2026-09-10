---
sidebar_position: 6
title: "Hermes에서 MCP 사용하기"
description: "MCP 서버를 Hermes Agent에 연결하고, 해당 도구를 필터링하며, 실제 워크플로에서 안전하게 사용하는 실용적인 가이드"
---

# Hermes에서 MCP 사용하기

이 가이드는 일상적인 워크플로에서 MCP를 실제로 사용하고 빠르고 안전하게 가치를 얻는 방법을 설명합니다.

기능 페이지에서 MCP가 무엇인지 설명한다면, 이 가이드는 MCP를 빠르게 활용하는 방법을 다룹니다.

## MCP는 언제 사용해야 하나요?

다음과 같은 경우 MCP를 사용하세요.
- 도구가 이미 MCP 형태로 존재하며 네이티브 Hermes 도구를 만들고 싶지 않을 때
- 깔끔한 RPC 계층을 통해 Hermes가 로컬 또는 원격 시스템을 대상으로 작업하게 하고 싶을 때
- 서버별로 세밀한 노출 제어를 원할 때
- Hermes 코어를 수정하지 않고 Hermes를 내부 API, 데이터베이스 또는 회사 시스템에 연결하고 싶을 때

다음과 같은 경우에는 MCP를 사용하지 마세요.
- 내장 Hermes 도구가 이미 작업을 잘 해결할 때
- 서버가 매우 크고 위험한 도구 표면을 노출하며 이를 필터링할 준비가 되어 있지 않을 때
- 매우 좁은 통합 하나만 필요하고 네이티브 도구가 더 간단하고 안전할 때

## 개념 모델

MCP를 어댑터 계층이라고 생각하세요.

- Hermes는 에이전트로 남습니다.
- MCP 서버가 도구를 제공합니다.
- Hermes는 시작 또는 다시 로드 시 해당 도구를 검색합니다.
- 모델은 일반 도구처럼 이를 사용할 수 있습니다.
- 각 서버에서 얼마나 많은 부분을 표시할지 직접 제어합니다.

마지막 항목이 중요합니다. 좋은 MCP 사용은 단순히 "모두 연결하기"가 아닙니다. "올바른 것을, 유용한 최소 표면으로 연결하기"입니다.

## 1단계: MCP 지원 설치

표준 설치 스크립트로 Hermes를 설치했다면 MCP 지원이 이미 포함되어 있습니다(설치 프로그램이 `uv pip install -e ".[all]"`을 실행합니다).

추가 기능 없이 설치했고 MCP를 별도로 추가해야 한다면 다음을 실행하세요.

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

npm 기반 서버를 사용하는 경우 Node.js와 `npx`를 사용할 수 있는지 확인하세요.

많은 Python MCP 서버에서는 `uvx`가 좋은 기본값입니다.

## 2단계: 먼저 서버 하나 추가

안전한 서버 하나로 시작하세요.

예시: 하나의 프로젝트 디렉터리에만 파일 시스템 접근을 허용합니다.

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

그런 다음 Hermes를 시작합니다.

```bash
hermes chat
```

이제 구체적인 질문을 하세요.

```text
Inspect this project and summarize the repo layout.
```

## 3단계: MCP가 로드되었는지 확인

다음과 같은 몇 가지 방법으로 MCP를 확인할 수 있습니다.

- 설정된 경우 Hermes 배너/상태에 MCP 통합이 표시되어야 합니다.
- Hermes에 현재 사용할 수 있는 도구를 물어보세요.
- 설정을 변경한 후 `/reload-mcp`를 사용하세요.
- 서버 연결에 실패했다면 로그를 확인하세요.

실용적인 테스트 프롬프트:

```text
Tell me which MCP-backed tools are available right now.
```

## 4단계: 즉시 필터링 시작

서버가 많은 도구를 노출한다면 나중까지 기다리지 마세요.

### 예시: 원하는 항목만 허용

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

민감한 시스템에는 일반적으로 이것이 가장 좋은 기본값입니다.

## WSL2: WSL의 Hermes를 Windows Chrome에 연결하기

다음과 같은 상황에서 유용한 실용적 설정입니다.

- Hermes는 WSL2 안에서 실행됩니다.
- 제어하려는 브라우저는 Windows의 평소 로그인된 Chrome입니다.
- WSL에서 `/browser connect`를 사용하는 것이 번거롭거나 불안정합니다.

이 설정에서 Hermes는 Chrome에 직접 연결하지 않습니다. 대신:

- Hermes는 WSL에서 실행됩니다.
- Hermes는 로컬 stdio MCP 서버를 시작합니다.
- 해당 MCP 서버는 Windows 인터롭(`cmd.exe` 또는 `powershell.exe`)을 통해 실행됩니다.
- MCP 서버가 실행 중인 Windows Chrome 세션에 연결합니다.

개념 모델:

```text
Hermes (WSL) -> MCP stdio bridge -> Windows Chrome
```

### 이 모드가 유용한 이유

- 실제 Windows 브라우저 프로필, 쿠키 및 로그인을 유지할 수 있습니다.
- Hermes는 지원되는 Unix 환경(WSL2) 안에서 계속 실행됩니다.
- 브라우저 제어가 Hermes 코어 브라우저 전송에 의존하지 않고 MCP 도구로 노출됩니다.

### 권장 서버

`chrome-devtools-mcp`를 사용하세요.

Windows Chrome에 `chrome://inspect/#remote-debugging`에서 이미 실시간 원격 디버깅이 활성화되어 있다면 WSL에서 다음과 같이 추가하세요.

```bash
hermes mcp add chrome-devtools-win --command cmd.exe --args /c npx -y chrome-devtools-mcp@latest --autoConnect --no-usage-statistics
```

서버를 저장한 후 다음을 실행합니다.

```bash
hermes mcp test chrome-devtools-win
```

그런 다음 새 Hermes 세션을 시작하거나 다음을 실행하세요.

```text
/reload-mcp
```

### 일반적인 프롬프트

로드되면 Hermes가 MCP 접두사가 붙은 브라우저 도구를 직접 사용할 수 있습니다. 예를 들면:

```text
调用 MCP 工具 mcp_chrome_devtools_win_list_pages，列出当前浏览器标签页。
```

### `/browser connect`가 올바른 도구가 아닌 경우

Hermes가 WSL에서 실행되고 Chrome이 Windows에서 실행되는 경우 Chrome이 열려 있고 디버깅 가능한 상태여도 `/browser connect`가 실패할 수 있습니다.

일반적인 이유는 다음과 같습니다.

- WSL이 Chrome이 Windows 도구에 노출하는 동일한 호스트 로컬 엔드포인트에 접근할 수 없습니다.
- 최신 Chrome 실시간 디버깅 흐름은 기존의 `ws://localhost:9222`와 동일하지 않습니다.
- `chrome-devtools-mcp`와 같은 Windows 측 헬퍼에서 브라우저에 연결하는 편이 더 쉽습니다.

이 경우 동일한 환경의 설정에는 `/browser connect`를 사용하고, WSL에서 Windows로 브라우저를 연결할 때는 MCP를 사용하세요.

### 알려진 문제

- MCP를 통해 Windows stdio 실행 파일을 사용할 때는 `/mnt/c/Users/<you>` 또는 `/mnt/c/workspace/...`와 같은 Windows 마운트 경로에서 Hermes를 시작하세요.
- `/root` 또는 `/home/...`에서 Hermes를 시작하면 MCP 서버가 시작되기 전에 Windows가 `UNC` 현재 디렉터리 경고를 출력할 수 있습니다.
- 페이지를 열거하는 동안 `chrome-devtools-mcp --autoConnect`가 시간 초과되면 Chrome에서 백그라운드/동결 탭을 줄이고 다시 시도하세요.

### 예시: 위험한 작업을 블랙리스트에 추가

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### 예시: 유틸리티 래퍼도 비활성화

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

## 필터링은 실제로 무엇에 영향을 주나요?

Hermes에서 MCP가 노출하는 기능은 두 가지 범주로 나뉩니다.

1. 서버 네이티브 MCP 도구
- 다음으로 필터링합니다.
  - `tools.include`
  - `tools.exclude`

2. Hermes가 추가하는 유틸리티 래퍼
- 다음으로 필터링합니다.
  - `tools.resources`
  - `tools.prompts`

### 표시될 수 있는 유틸리티 래퍼

리소스:
- `list_resources`
- `read_resource`

프롬프트:
- `list_prompts`
- `get_prompt`

이러한 래퍼는 다음 조건을 모두 충족할 때만 표시됩니다.
- 설정에서 허용함
- MCP 서버 세션이 실제로 해당 기능을 지원함

따라서 Hermes는 서버에 리소스/프롬프트가 없는데도 있는 것처럼 가장하지 않습니다.

## 일반적인 패턴

### 패턴 1: 로컬 프로젝트 도우미

제한된 워크스페이스를 대상으로 Hermes가 추론하도록 하려면 저장소 로컬 파일 시스템 또는 git 서버에 MCP를 사용하세요.

```yaml
mcp_servers:
  fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/project"]
```

좋은 프롬프트:

```text
Review the project structure and identify where configuration lives.
```

```text
Check the local git state and summarize what changed recently.
```

### 패턴 2: Open Scaffold를 사용한 저장소 네이티브 작업 기록

[Open Scaffold](https://github.com/graphanov/open-scaffold)는 Hermes가 저장소의 영속적인 AI 작업 기록(미션, 계획, 근거 노트, 핸드오프 패킷, 검토/게이트 결과)을 읽도록 하고 싶을 때 사용하세요. Hermes는 에이전트로 남고, Open Scaffold는 저장소 로컬 기록으로 남습니다.

하나의 Scaffold 저장소에 서버를 추가하세요.

```bash
hermes mcp add open_scaffold --command npx --args -y open-scaffold@latest mcp serve --repo /absolute/path/to/repo
hermes mcp test open_scaffold
```

그런 다음 노출 표면을 읽기 중심으로 유지하세요. `hermes mcp add` 프롬프트에서 `select`를 선택하거나 이후 `config.yaml`을 편집합니다.

```yaml
mcp_servers:
  open_scaffold:
    command: "npx"
    args: ["-y", "open-scaffold@latest", "mcp", "serve", "--repo", "/absolute/path/to/repo"]
    tools:
      include:
        - list_plans
        - get_plan
        - get_mission
        - list_evidence
        - get_evidence
        - get_status
        - search_plans
        - list_amendments
        - get_handoff
        - analyze_loop
        - gate_loop
      prompts: false
```

좋은 프롬프트:

```text
Use the Open Scaffold MCP tools to compile the current handoff packet and tell me the next legal action.
```

```text
Inspect the active plans and evidence notes, then say whether this repo is ready for human review or needs another attempt.
```

경계 참고 사항:

- Open Scaffold MCP는 로컬 우선이며 기본적으로 읽기 전용입니다.
- 쓰기 도구를 사용하려면 서버를 `--allow-write`로 시작해야 합니다. Hermes가 `.osc` 파일을 변경하도록 명시적으로 원할 때까지 활성화하지 마세요.
- Open Scaffold는 작업을 기록하고 게이트를 적용하지만 Hermes가 병합, 게시, 배포 또는 런타임 생성을 수행하도록 승인하지는 않습니다.
- 재현 가능한 도구 스키마가 필요하다면 `open-scaffold@<version>`을 `@latest` 대신 고정하세요.

### 패턴 3: GitHub 트리아지 도우미

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

좋은 프롬프트:

```text
List open issues about MCP, cluster them by theme, and draft a high-quality issue for the most common bug.
```

```text
Search the repo for uses of _discover_and_register_server and explain how MCP tools are registered.
```

### 패턴 4: 내부 API 도우미

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

좋은 프롬프트:

```text
Look up customer ACME Corp and summarize recent invoice activity.
```

이런 경우에는 제외 목록보다 엄격한 허용 목록이 훨씬 낫습니다.

### 패턴 4: 문서/지식 서버

일부 MCP 서버는 직접 작업보다는 공유 지식 자산에 가까운 프롬프트 또는 리소스를 노출합니다.

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: true
      resources: true
```

좋은 프롬프트:

```text
List available MCP resources from the docs server, then read the onboarding guide and summarize it.
```

```text
List prompts exposed by the docs server and tell me which ones would help with incident response.
```

## 튜토리얼: 필터링을 포함한 엔드투엔드 설정

실용적인 진행 순서는 다음과 같습니다.

### 1단계: 엄격한 허용 목록으로 GitHub MCP 추가

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

Hermes를 시작하고 다음과 같이 요청하세요.

```text
Search the codebase for references to MCP and summarize the main integration points.
```

### 2단계: 필요할 때만 확장

나중에 이슈 업데이트도 필요하다면 다음과 같이 합니다.

```yaml
tools:
  include: [list_issues, create_issue, update_issue, search_code]
```

그런 다음 다시 로드합니다.

```text
/reload-mcp
```

### 3단계: 다른 정책으로 두 번째 서버 추가

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

이제 Hermes가 두 서버를 결합할 수 있습니다.

```text
Inspect the local project files, then create a GitHub issue summarizing the bug you find.
```

이것이 MCP가 강력해지는 지점입니다. Hermes 코어를 변경하지 않고 여러 시스템의 워크플로를 사용할 수 있습니다.

## 안전한 사용 권장 사항

### 위험한 시스템에는 허용 목록을 우선 사용

금융, 고객 대상 또는 파괴적인 작업에는 다음을 적용하세요.
- `tools.include`를 사용합니다.
- 가능한 가장 작은 집합으로 시작합니다.

### 사용하지 않는 유틸리티 비활성화

모델이 서버가 제공하는 리소스/프롬프트를 탐색하지 않기를 원한다면 다음을 끄세요.

```yaml
tools:
  resources: false
  prompts: false
```

### 서버 범위를 좁게 유지

예시:
- 파일 시스템 서버는 홈 디렉터리 전체가 아니라 하나의 프로젝트 디렉터리를 루트로 지정합니다.
- git 서버는 하나의 저장소를 가리킵니다.
- 내부 API 서버는 기본적으로 읽기 중심의 도구를 노출합니다.

### 설정 변경 후 다시 로드

```text
/reload-mcp
```

다음 항목을 변경한 후 실행하세요.
- 포함/제외 목록
- 활성화 플래그
- 리소스/프롬프트 토글
- 인증 헤더 / 환경 변수

## 증상별 문제 해결

### "서버는 연결되지만 예상한 도구가 보이지 않습니다"

가능한 원인:
- `tools.include`로 필터링됨
- `tools.exclude`로 제외됨
- `resources: false` 또는 `prompts: false`로 유틸리티 래퍼가 비활성화됨
- 서버가 실제로 리소스/프롬프트를 지원하지 않음

### "서버가 설정되었지만 아무것도 로드되지 않습니다"

다음을 확인하세요.
- 설정에 `enabled: false`가 남아 있지 않음
- 명령/런타임이 존재함(`npx`, `uvx` 등)
- HTTP 엔드포인트에 연결할 수 있음
- 인증 환경 변수 또는 헤더가 올바름

### "MCP 서버가 광고하는 도구보다 적은 수의 도구가 보이는 이유는 무엇인가요?"

Hermes가 이제 서버별 정책과 기능 인식 등록을 적용하기 때문입니다. 이는 예상된 동작이며 일반적으로 바람직합니다.

### "설정을 삭제하지 않고 MCP 서버를 제거하려면 어떻게 하나요?"

다음과 같이 사용하세요.

```yaml
enabled: false
```

이렇게 하면 설정은 유지되지만 연결 및 등록은 방지됩니다.

## 처음 사용하기에 권장되는 MCP 설정

대부분의 사용자에게 좋은 첫 서버:
- 파일 시스템
- git
- GitHub
- fetch / 문서 MCP 서버
- 범위가 좁은 내부 API 하나

처음 사용하기에 좋지 않은 서버:
- 필터링 없이 파괴적인 작업이 많고 규모가 큰 비즈니스 시스템
- 제한할 수 있을 만큼 충분히 이해하지 못한 모든 것

## 관련 문서

- [MCP(Model Context Protocol)](/user-guide/features/mcp)
- [FAQ](/reference/faq)
- [슬래시 명령](/reference/slash-commands)
