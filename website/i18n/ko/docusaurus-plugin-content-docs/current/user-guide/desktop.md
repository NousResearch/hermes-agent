---
sidebar_position: 3
title: "데스크톱 앱"
description: "스트리밍 도구 출력, 나란히 보기 미리보기, 파일 브라우저, 음성, cron, 프로필, 스킬, 설정을 제공하는 세련된 Hermes 데스크톱 네이티브 앱입니다. macOS, Windows, Linux를 지원합니다."
---

# 데스크톱 앱

Hermes 데스크톱 앱은 CLI와 게이트웨이에서 사용하는 것과 **동일한** 에이전트를 기반으로 만들어진 네이티브 앱입니다. 동일한 설정, API 키, 세션, 스킬, 메모리를 사용합니다. 별도의 제품이나 가벼운 복제본이 아니라, 동일한 Hermes Agent 코어와 설정을 사용하고 현대적이며 세심하게 설계된 UI를 통해 구동합니다. 터미널에서 `hermes`를 사용해 본 적이 있다면 그곳에서 설정한 모든 것이 이미 여기에 있으며, 여기서 수행한 작업도 그곳에 나타납니다.

**macOS, Windows, Linux에서 실행됩니다.**

:::tip 인터페이스별 차이는 무엇인가요?
Hermes에는 모두 동일한 에이전트와 통신하는 여러 프런트엔드가 있습니다.

- **데스크톱 앱**(이 페이지) — 채팅, 구성, 관리를 위한 전용 UI를 갖춘 네이티브 애플리케이션입니다.
- **CLI**(`hermes`) 및 **[TUI](./tui.md)**(`hermes --tui`) — 터미널 인터페이스입니다.
- **[웹 대시보드](./features/web-dashboard.md)**(`hermes dashboard`) — 브라우저 관리 패널이며, 선택적 **Chat** 탭은 의사 터미널을 통해 TUI를 삽입합니다.

상황에 맞는 것을 선택하세요. 상태를 공유하므로 한 인터페이스에서 세션을 시작하고 다른 인터페이스에서 이어갈 수 있습니다.
:::

## 설치

[Hermes Desktop 설치 안내](../getting-started/installation.md)를 따르세요.

Hermes가 이미 설치되어 있다면 다음을 실행하기만 하면 됩니다.

```bash
hermes desktop
```

현재 설정, 키, 세션, 스킬을 사용합니다.

## 앱에 포함된 기능

데스크톱 앱은 탐색을 위한 왼쪽 사이드바가 있는 채팅 중심 창으로 구성되어 있습니다. 여러 에이전트 대화를 동시에 관리하고, 메시징 제공자를 구성하고, 아티팩트를 만들고, 프로젝트 폴더 구조를 탐색하며, 여러 프로젝트를 동시에 작업할 수 있도록 설계되었습니다.

### 채팅

앱의 중앙 영역입니다. 다음 기능을 사용할 수 있습니다.

- 에이전트가 작업하는 동안 실시간 도구 활동과 구조화된 도구 호출 요약을 보여 주는 **스트리밍 응답**.
- 다른 모든 Hermes 표면과 **동일한 대화 기록** — 여기서 시작한 세션을 CLI/TUI에서 이어가거나 그 반대로 할 수 있습니다.
- 채팅 영역 어디든 **파일을 드래그 앤 드롭**하여 다음 메시지에 첨부.
- 채팅을 계속하면서 웹 페이지, 파일, 도구 출력을 나란히 렌더링하는 **오른쪽 미리보기 레일**.
- **작성기 기록 및 대기열 편집** — 빈 작성기에서 위/아래 화살표 키를 눌러 이전 프롬프트를 불러와 재사용하고, 전송 전에 대기열에 넣은 메시지를 편집할 수 있습니다. 턴이 대기열에 있는 동안 Stop(또는 Esc)을 누르면 대기열이 일시 중지되고 작성기 위로 펼쳐집니다. 그곳에서 재개하거나 개별 항목을 전송, 편집, 삭제할 수 있습니다.
- **대화 타임라인 레일** — 긴 채팅에는 대화 기록 가장자리를 따라 프롬프트마다 하나씩 가느다란 마커가 표시됩니다. 마커 위에 마우스를 올리면 프롬프트 목록이 열리고, 하나를 클릭하면 대화의 해당 지점으로 바로 이동합니다. (채팅에 여러 턴이 쌓인 후 나타납니다.)
- **페이지에서 찾기** — **Cmd/Ctrl+F**를 눌러 렌더링된 채팅 기록을 검색하는 찾기 바를 엽니다. Enter / Shift+Enter(또는 바가 열린 상태에서 Cmd/Ctrl+G / Cmd/Ctrl+Shift+G)로 일치 항목을 이동하고, Esc로 닫습니다.

#### 상태 표시줄

채팅 하단의 표시줄에는 실시간 세션 상태가 표시되고, 설정을 열지 않아도 빠른 제어 기능을 사용할 수 있습니다.

- **세션별 YOLO 토글** — 이 세션에서만 YOLO를 켜거나 끕니다(TUI와 동일). YOLO는 위험한 명령 승인 프롬프트를 우회하므로, 무엇을 끄는지 알고 사용하세요 — [보안 → YOLO 모드](./security.md#yolo-mode)를 참고하세요.
- **컨텍스트 사용량 미터** — 세션 컨텍스트 창이 얼마나 "가득 찼는지"를 실시간 백분율로 보여 줍니다. 클릭하면 시스템 프롬프트, 도구 정의, 스킬, 메모리, 규칙, MCP, 서브에이전트 정의, 대화 자체 등 카테고리별 토큰 분석을 제공하는 **Context Usage** 팝오버가 열려 압축이 시작되기 전에 창을 무엇이 차지하고 있는지 정확히 확인할 수 있습니다.
- **사용자 지정 항목** — 상태 표시줄을 마우스 오른쪽 버튼으로 클릭(**Show in status bar**)하여 표시할 항목을 선택할 수 있습니다. 컨텍스트 미터, 작업 공간, 모델, 승인, 턴/세션 타이머, 터미널, Command Center, 백엔드 버전 등이 포함되며, 표시줄 전체를 숨길 수도 있습니다(**Cmd/Ctrl+Shift+S**로 전환).

번들로 제공되는 로컬 백엔드 대신 다른 컴퓨터의 Hermes 인스턴스에 연결해 채팅하고 있나요? 아래 [원격 백엔드에 연결](#connecting-to-a-remote-backend)을 참고하세요. 원격 호스팅 대시보드 연결의 전체 흐름(인증 게이트, `/api/ws` 채팅 소켓, WebSocket 종료 코드 분류)은 [웹 대시보드 → Hermes Desktop을 원격 백엔드에 연결](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)을 참고하세요.

#### 저장소 검색

Hermes Desktop은 홈 디렉터리를 제한된 깊이까지 검색하여 Projects 사이드바에 표시할 로컬 Git 저장소를 찾습니다. **Settings → Workspace**에서 프로필별로 변경하거나 `config.yaml`에서 설정할 수 있습니다.

```yaml
desktop:
  repo_scan_enabled: true
  repo_scan_roots: []
  repo_scan_exclude_paths: []
```

- `repo_scan_enabled: false`로 설정하면 파일 시스템 검색이 완전히 중지됩니다. 해당 프로필의 기존 디스크 검색 캐시 행은 삭제되지만, 의도적인 Hermes 세션에서 확인된 명시적 프로젝트와 저장소는 계속 사용할 수 있습니다.
- `repo_scan_roots`를 폴더 목록으로 설정하여 검색 범위를 제한합니다. 빈 목록이면 기본 홈 디렉터리 검색을 유지합니다.
- `repo_scan_exclude_paths`를 설정하면 해당 폴더의 전체 하위 트리를 건너뜁니다.

이 값 중 하나를 변경하면 해당 프로필의 디스크 검색 캐시만 무효화되고 정책을 준수하는 새로 고침이 시작됩니다. **Hide from sidebar**는 별도의 항목별 큐레이션 작업입니다.

#### 모델 선택

모델 선택기는 마이크 바로 왼쪽의 **composer**에 있습니다. 하나의 드롭다운에서 모델, 추론 수준, fast mode를 변경할 수 있습니다.

- **작성기 선택기는 고정되는 UI 상태이며 기본값에는 영향을 주지 않습니다.** 기기별로 로컬에 기억되며 새 채팅과 재시작에서도 **유지**되어 기본값으로 되돌아가지 않습니다. 모델을 한 번 선택하면 다음 `Cmd/Ctrl+N`이 해당 모델로 열립니다. 활성 채팅에서 모델을 변경하면 변경 사항은 **현재 채팅**에 적용됩니다. 어느 경우든 세션을 만들거나 전환할 때 선택이 따라가며 프로필 기본값에는 **절대** 기록되지 않습니다. ([프로필](#sessions--profiles)을 전환하면 해당 프로필의 자체 기본값으로 다시 설정됩니다.)
- **Settings → Model에서 기본값을 설정합니다.** 이 "main" 모델은 **프로필별 전역 기본값**입니다. 새 채팅, cron, 서브에이전트, 보조 작업이 이 모델로 시작하며, 이 값을 기록하는 유일한 위치입니다. 각 [프로필](#sessions--profiles)은 자체 기본값을 유지합니다.
- **모델별 effort/fast 프리셋.** 각 모델은 데스크톱 앱에서 자체 추론 수준과 fast-mode 선택을 기억하며, 해당 모델을 선택할 때마다 세션에 다시 적용합니다. 이 프리셋은 데스크톱의 편의 기능이며 cron이나 서브에이전트를 변경하지 않습니다.
- **채팅 중간에 모델을 전환하면 프롬프트 캐시가 초기화됩니다.** 활성 채팅에서 모델을 전환하면 다음 메시지가 전체 대화를 입력 비용 전액으로 다시 읽습니다(제공자 프롬프트 캐시는 모델별로 구분됩니다). 가끔 전환하는 것은 괜찮지만, 긴 채팅에서는 모델을 계속 오가는 것보다 새 모델로 새 채팅을 시작하는 편이 더 저렴한 경우가 많습니다.

### 파일 브라우저

앱을 벗어나지 않고 작업 디렉터리를 탐색하고 미리 볼 수 있습니다. 에이전트가 파일을 읽고 쓰고 편집하는 과정을 따라갈 때 유용합니다. `hermes desktop --cwd <path>`(또는 `HERMES_DESKTOP_CWD` 환경 변수)로 초기 프로젝트 디렉터리를 설정합니다.

### 아티팩트

**Artifacts** 보기에서는 세션이 생성한 **이미지, 파일, 링크**를 검색하고 탐색할 수 있는 하나의 갤러리로 모읍니다. 사이드바, 명령 팔레트(**Artifacts — Browse generated outputs**) 또는 직접 지정한 `nav.artifacts` 단축키에서 엽니다. 최근 세션 출력을 자동으로 색인하고, 모든 아티팩트에는 생성한 세션과 해당 채팅으로 돌아가는 바로 가기가 표시됩니다. 이미지와 파일은 미리보기에서 다운로드 / 브라우저에서 열기 / 복사 작업을 수행할 수 있습니다.

### 창, 탭 및 창 분할

앱은 여러 작업을 동시에 진행하도록 설계되었습니다.

- **탭** — **Cmd/Ctrl+T**로 새 세션 탭을 열고, **Ctrl+Tab** / **Ctrl+Shift+Tab**으로 세션을 순환하며, **Ctrl+1…9**로 위치에 따라 최근 세션으로 이동합니다. **Cmd/Ctrl+W**로 포커스된 탭을 닫고 **Cmd/Ctrl+Shift+T**로 마지막에 닫은 탭을 다시 엽니다.
- **여러 창** — **Cmd/Ctrl+Shift+N**으로 새 창을 열 수 있으며, 세션의 컨텍스트 메뉴(**New window**) 또는 명령 팔레트에서 세션을 분리할 수 있습니다. 분리된 창은 전역 사이드바 없이 해당 채팅 하나만 표시하므로, 다른 모니터에 장시간 실행 중인 세션을 두기에 편리합니다. 세션을 표시하는 모든 창으로 실시간 에이전트 출력이 스트리밍됩니다.
- **창 분할** — **Cmd/Ctrl+B**로 왼쪽 사이드바를, **Cmd/Ctrl+J**로 오른쪽 사이드바를 전환하고, **Cmd/Ctrl+\**로 사이드바가 놓인 쪽을 바꿉니다.

### 터미널

파일 브라우저 옆 오른쪽 사이드바에 실제 터미널이 있습니다.

- **Ctrl+`**로 터미널을 표시합니다(없으면 하나를 엽니다). **Ctrl+Shift+`**로 터미널을 추가합니다. 여러 터미널은 탭 레일에 쌓이며 **Ctrl+Shift+↓/↑**로 터미널 사이를 이동하고 **Ctrl+Shift+W**로 활성 터미널을 닫습니다.
- **숨겨져도 셸은 유지됩니다.** 패널을 닫거나 숨겨도 셸이 종료되지 않습니다. 명시적으로 닫을 때까지 열려 있는 모든 터미널은 스크롤백과 실행 중인 프로세스를 유지한 채 마운트되어 있습니다.
- **채팅에 추가** — 터미널 출력을 선택하고 다음 메시지의 컨텍스트로 작성기에 보냅니다.

### Git 검토 및 worktree

Git 저장소 안에서 실행되는 세션에는 내장 소스 제어 화면이 제공됩니다.

- **검토 창** — **Cmd/Ctrl+G**로 작업 트리 검토 창을 전환합니다. 브랜치 및 ahead/behind 상태, 변경된 파일(목록 또는 트리 보기), **Uncommitted**, **Branch**, **Last turn**(에이전트가 가장 최근 턴에서 변경한 내용만) 범위의 diff를 표시합니다. 파일을 스테이징/언스테이징하고, 커밋 메시지를 작성하거나(**Generate commit message**), **Commit** 또는 **Commit & Push**를 실행한 다음 GitHub CLI(`gh`)를 통해 **Create PR**을 실행할 수 있습니다. 또는 **Ask Hermes to open PR**로 전체 작업을 에이전트에게 맡길 수도 있습니다. 여기서 브랜치를 만들고 전환할 수도 있습니다.
- **Worktree** — **Cmd/Ctrl+Shift+B**(또는 사이드바 프로젝트의 **New worktree**)로 새 브랜치에 Git worktree를 만들어 에이전트가 체크아웃을 건드리지 않고 저장소의 병렬 복사본에서 작업하도록 합니다. Worktree는 프로젝트 아래 별도의 레인으로 표시됩니다. 하나를 제거하면 worktree 디렉터리를 삭제하거나(브랜치는 유지) 레인을 숨기고 디스크에 남겨 둘 수 있으며, 커밋되지 않은 변경 사항이 있을 때 강제 옵션도 제공됩니다.

### 메모리 그래프

**Memory Graph**(명령 팔레트 → *Memory Graph* 또는 상태 표시줄 항목)는 Hermes가 학습한 내용을 보여 주는 대화형 지도입니다. 스킬과 메모리가 확대/축소 가능한 노드 그래프로 배치되며, **All / Used / Learned**로 필터링할 수 있습니다. 공유 컨트롤은 지도 레이아웃을 다른 사람에게 붙여 넣을 수 있는 압축 코드로 내보냅니다(레이아웃만 포함하며 메모리나 스킬 텍스트는 포함하지 않음). 같은 방식으로 코드를 가져올 수도 있습니다.

### 빠른 입력

Quick Entry는 **시스템 어디서나 전역 단축키로** 호출하는 작은 상시 사용 가능 작성기입니다. 주 창으로 전환하거나 주 창을 열지 않고도 프롬프트를 보낼 수 있습니다. **Settings → Advanced → Quick Entry**에서 활성화하세요. 기본 단축키는 **Ctrl/Cmd+Shift+Space**이며 직접 지정할 수 있습니다(하나 이상의 수정 키가 필요합니다). 다른 앱이 이미 해당 키 조합을 사용 중이면 설정 행에 표시되므로 다른 조합을 선택할 수 있습니다.

### 음성

Hermes와 대화하고 응답을 들을 수 있습니다. 다른 곳에서도 사용할 수 있는 동일한 [음성 모드](./features/voice-mode.md)입니다. macOS에서는 마이크 접근 권한을 한 번 요청합니다.

### HUD 모드

**⌘/Ctrl+Shift+H**(또는 제목 표시줄 버튼)는 채팅을 크롬 없는 항상 위에 표시되는 부동 바로 분리하여 현재 작업 중인 항목 위에 띄웁니다. 앱 창은 옆으로 물러나고 HUD에는 실시간 대화와 작성기가 유지됩니다. 어디에 배치하는지는 컨텍스트가 됩니다. 바의 위치가 Hermes에 어떤 앱과 화면에 대해 묻는지 알려 주므로 "this", "here", "that page"가 바 아래에 있는 항목을 가리키게 됩니다.

- **바 이동** — 작성기의 아무 곳이나 **잠시 누르고 있다가** 드래그합니다. 짧게 누르면 입력하고, 누른 채 있으면 창을 잡습니다. 이것이 HUD를 이동하는 유일한 방법이며 드래그할 제목 표시줄은 없습니다.
- **크기 조정** — 바의 오른쪽 아래 모서리를 드래그합니다.
- **포인터 위치로 맞추기** — **⌘/Ctrl+Shift+G**(어떤 앱에서도 작동하는 전역 단축키)로 HUD를 커서가 있는 위치로 이동합니다.
- **종료** — 바의 종료 버튼을 클릭하거나 **⌘/Ctrl+Shift+H**를 다시 누릅니다. 세션을 유지한 채 앱 창이 돌아옵니다.

### 설정 및 온보딩

YAML을 편집하는 대신 실제 UI에서 제공자, 모델, 도구, 자격 증명을 관리할 수 있습니다. 최초 실행 온보딩을 통해 몇 초 안에 첫 메시지까지 도달합니다. 설정 창에서는 제공자/키, 모델 선택, 도구 세트 구성, MCP 서버, 게이트웨이, 세션 관리를 다룹니다.

- **Providers 설정 창** — 추론 제공자를 관리하는 전용 공간으로, 제공자별 자격 증명을 로그인하고 저장할 수 있는 Accounts / API-keys UX를 제공합니다.
- **메뉴에 모든 제공자와 모델 표시** — GUI에는 전체 제공자 목록과 `hermes model`이 알고 있는 모든 모델이 표시됩니다. 따라서 엄선된 하위 목록이 아니라 CLI와 동일한 카탈로그에서 선택할 수 있습니다.
- **xAI Grok OAuth** — Grok은 런처에서 일급 OAuth 제공자입니다. 다른 OAuth 제공자와 마찬가지로 브라우저 흐름을 통해 로그인합니다.
- **GUI에서 도구 백엔드 설치** — 터미널로 전환하지 않고 앱에서 도구 백엔드의 사후 설정 설치 단계를 직접 실행합니다.
- **터미널 글꼴 선택기** — **Settings → Appearance**에서 설치된 글꼴을 선택합니다. `MesloLGS NF`와 같은 Nerd Fonts는 대화형 터미널과 에이전트 터미널 모두에서 Powerlevel10k 구분자와 아이콘을 렌더링하며, 설정은 프로필별로 저장됩니다.
- **보조 모델 경고** — 보조 작업(제목 지정, 요약 및 유사한 도우미)이 여전히 다른 제공자에 고정된 상태에서 주 모델을 새 제공자로 바꾸면 앱이 경고하여 자신도 모르게 두 제공자에 작업을 분산하지 않도록 합니다.
- **VS Code Marketplace 테마** — 기본 제공 테마 프리셋 외에도 모양 설정에 실시간 VS Code Marketplace 검색 기능이 있습니다. 색상 테마를 선택하면 앱이 이를 다운로드, 변환, 설치하여 데스크톱 테마로 만듭니다. 명령 팔레트(*Install theme*)에서도 동일한 가져오기를 사용할 수 있으며, 가져온 테마는 모양 설정에서 다시 제거할 수 있습니다.
- **컴퓨터 절전 방지** — **Settings → Advanced → Keep computer awake**는 컴퓨터가 절전 모드로 전환되지 않게 하여 장시간 또는 밤새 실행되는 에이전트 작업이 계속되도록 합니다(디스플레이는 계속 어두워질 수 있음). 컴퓨터별 설정입니다.

최초 실행 온보딩은 통합 오버레이 디자인 시스템으로 재설계되었으며, **Choose provider later**를 선택하여 제공자 설정을 건너뛰고 먼저 앱에 들어갈 수 있습니다.

### 관리 창

앱은 터미널로 전환하지 않아도 되도록 더 넓은 Hermes 관리 화면도 제공합니다.

- **Skills** — [스킬](./features/skills.md)을 탐색하고 설치하고 관리합니다.
- **Memory graph (Star Map)** — 채팅에서 `/journey`(`/learning`, `/memory-graph` 별칭)를 입력하면 시간에 따른 학습 스킬과 메모리의 대화형 별자리와 재생 스크러버를 엽니다. 패널에서 바로 노드를 편집하거나 삭제할 수 있습니다(스킬은 보관되고 메모리는 제거됨). [Learning Journey](./features/memory.md#learning-journey-journey)를 참고하세요.
- **Cron** — [예약된 작업](../reference/cli-commands.md#hermes-cron)을 확인하고 관리합니다.
- **Profiles** — [Hermes 프로필](./profiles.md)(격리된 설정/스킬/세션) 사이를 전환합니다.
- **Messaging** — 게이트웨이 채널을 설정합니다.
- **Agents** 및 **Command Center** — 다중 에이전트 작업을 위한 오케스트레이션 화면입니다.

### 키보드 및 탐색

- **명령 팔레트** — **Cmd+K** 또는 **Cmd+P**(Windows/Linux에서는 Ctrl+K / Ctrl+P)를 눌러 키보드로 작업과 앱 탐색을 시작합니다. 모든 페이지나 설정 섹션 열기, 제목 또는 id로 세션 이동, 모델/테마/색상 모드 전환, 터미널 생성, 게이트웨이 재시작, Hermes 업데이트 등을 수행할 수 있습니다.
- **단축키 재지정** — **Settings → Keyboard Shortcuts**(또는 **Cmd/Ctrl+/**)에서 단축키 패널을 열고 거의 모든 바인딩을 다시 매핑할 수 있습니다. 프로필 전환, 세션 탐색, 보기 전환 및 데스크톱 플러그인이 추가한 단축키도 포함됩니다. 중복 할당은 충돌로 표시됩니다. 알아 둘 만한 기본값은 **Cmd/Ctrl+N** 새 세션, **Cmd/Ctrl+.** Command Center, **Cmd/Ctrl+,** 설정, **Cmd/Ctrl+Shift+F** 세션 검색, **Cmd/Ctrl+1–9** 프로필 전환, **Shift+X** 라이트/다크 전환입니다.
- **사용자 지정 확대/축소 단축키** — 반 단계씩 인터페이스를 확대하여 텍스트 크기를 더 세밀하게 조절합니다.
- **UI 언어 전환기** — 간체 중국어(zh-Hans)를 포함해 앱 인터페이스 언어를 앱 안에서 변경합니다.

### 세션 및 프로필

- **세션 목록 개편** — 보관 및 일반적인 세션 정리 기능이 있는 새 세션 목록으로, 목록이 커져도 관리하기 쉽습니다.
- **id로 세션 검색** — id로 특정 세션을 직접 찾습니다.
- **동시 다중 프로필 세션** — 여러 [프로필](./profiles.md)에서 동시에 세션을 실행하고, 프로필 간 `@session` 링크로 다른 프로필의 세션을 참조합니다.
- **프로필 내보내기 / 가져오기** — 전체 설정을 하나의 파일로 공유합니다. **⌘K → Export profile…**(또는 레일에서 프로필 사각형을 마우스 오른쪽 버튼으로 클릭)는 스킬, 메모리, 페르소나, cron, 플러그인, 설정이 포함된 `.tar.gz`를 작성하며 API 키는 제거합니다. 데스크톱에서 내보내면 모양과 인터페이스(스킨, 라이트/다크 모드, 사용자 지정 테마, 프로필 레일 색상, 창 레이아웃)도 함께 묶이므로 가져온 프로필은 보낸 사람이 사용하던 모습으로 도착합니다. **⌘K → Import profile…** 또는 레일의 **+** 옆 버튼으로 가져옵니다. 오버레이를 적용하고 새 프로필로 이동합니다. 동일한 아카이브는 채팅의 `/export` / `/import`와 셸의 `hermes profile export` / `import`에서도 사용할 수 있습니다. [프로필 파일 내보내기 및 가져오기](./profile-distributions.md#export-and-import-a-profile-file)를 참고하세요.

## 업데이트

앱은 백그라운드에서 업데이트를 확인하고 준비되면 한 번의 클릭으로 업데이트할 수 있도록 제공합니다.

[수동 업데이트 절차](https://hermes-agent.nousresearch.com/docs/getting-started/updating)도 GUI에서 사용할 수 있습니다.

## 제거

**Settings → About → Danger zone**을 열고 얼마나 제거할지 선택합니다.

- **Uninstall Chat GUI only** — 데스크톱 앱과 데이터를 제거하지만 Hermes 에이전트, 설정, 채팅은 유지합니다. (`hermes uninstall --gui`와 동일합니다.)
- **Uninstall GUI + agent, keep my data** — 앱과 에이전트를 제거하지만 이후 재설치를 위해 설정, 채팅, 비밀 정보를 유지합니다. (`hermes uninstall`과 동일합니다.)
- **Uninstall everything** — 앱, 에이전트, 모든 사용자 데이터를 제거합니다. (`hermes uninstall --full`과 동일합니다.)

작업을 마치기 위해 앱이 종료됩니다(실행 중인 앱 번들과 자체 venv를 제거할 수 있도록 종료 후 정리 작업이 실행됩니다). 로컬 에이전트가 설치되어 있지 않으면(예: 원격 백엔드에 연결된 GUI 전용 "lite" 클라이언트) 에이전트를 제거하는 옵션이 자동으로 숨겨집니다.

터미널에서도 동일하게 수행할 수 있습니다. GUI만 제거하려면 `hermes uninstall --gui`를, 에이전트도 제거하려면 `hermes uninstall` / `hermes uninstall --full`을 사용하세요.

:::note
**소스 체크아웃**(`hermes desktop` 개발 빌드)에서 `hermes uninstall --gui`를 실행하면 작업 공간의 `node_modules`와 `apps/desktop/{dist,release}` 빌드 출력도 제거됩니다. 이는 GUI 빌드 아티팩트이기 때문입니다. `hermes desktop`(또는 `npm install` 후 다시 빌드)으로 복구할 수 있지만, 데스크톱 앱을 직접 개발 중이라면 이후 의존성을 다시 설치해야 합니다.
:::

## CLI 참조: `hermes desktop`

CLI로 실행하려면 `hermes desktop`을 실행하기만 하면 됩니다. 기본적으로 작업 공간 Node 의존성을 설치하고, 현재 OS의 압축 해제된 Electron 앱을 빌드한 다음, 패키징된 아티팩트를 실행합니다.

| 플래그 | 설명 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build` | npm 설치/패키징을 건너뛰고 `apps/desktop/release`에 있는 기존 압축 해제 앱을 실행합니다. |
| `--force-build` | 콘텐츠 스탬프가 일치하더라도 전체 재빌드를 강제합니다. |
| `--build-only` | 데스크톱 앱을 빌드하지만 실행하지 않습니다(`hermes update`에서 사용). |
| `--source` | 패키징된 앱 대신 `apps/desktop/dist`에 대해 `electron .`으로 실행합니다. |
| `--cwd PATH` | 데스크톱 채팅 세션의 초기 프로젝트 디렉터리(`HERMES_DESKTOP_CWD` 설정). |
| `--hermes-root PATH` | 앱이 사용하는 Hermes 소스 루트를 재정의합니다(`HERMES_DESKTOP_HERMES_ROOT` 설정). |
| `--ignore-existing` | 백엔드 확인 중 PATH에 있는 기존 `hermes` CLI를 앱이 무시하도록 강제합니다. |
| `--fake-boot` | 시작 UI 검증을 위한 결정론적 부팅 지연을 활성화합니다. |

## 작동 방식

패키징된 앱에는 Electron 셸과 네이티브 React 채팅 화면이 포함됩니다. 최초 실행 시 `HERMES_HOME`(`~/.hermes`, Windows에서는 `%LOCALAPPDATA%\\hermes`)에 Hermes Agent 런타임을 설치할 수 있습니다. **CLI 설치와 동일한 레이아웃**이므로 서로 호환됩니다. 백엔드 확인은 먼저 `HERMES_DESKTOP_HERMES_ROOT`, 다음으로 완료된 관리형 설치, 그 다음 PATH에서 확인한 `hermes`(단, `--ignore-existing` / `HERMES_DESKTOP_IGNORE_EXISTING=1`이 설정된 경우 제외), 마지막으로 Nix와 같은 패키저를 위한 명시적인 `HERMES_DESKTOP_HERMES` 명령 재정의를 순서대로 사용합니다. React 렌더러는 앱이 실행하는 헤드리스 백엔드, 즉 `tui_gateway` JSON-RPC/WebSocket API를 제공하는 `hermes serve` 프로세스와 통신하며 `hermes --tui`를 삽입하는 대신 에이전트 런타임을 재사용합니다. 데스크톱 앱은 **독립 실행형**입니다. 자체 `hermes serve` 백엔드를 실행하며 [웹 대시보드](./features/web-dashboard.md)를 열거나 필요로 하지 않습니다. (`serve` 명령보다 오래된 런타임은 자동으로 헤드리스 `dashboard --no-open`으로 대체되므로 앱 업데이트가 백엔드보다 앞서가지 않습니다.) 설치, 백엔드 확인, 자체 업데이트 로직은 Electron 메인 프로세스에 있습니다.

## 원격 백엔드에 연결

기본적으로 앱은 자체 **로컬** 백엔드를 시작하고 관리합니다. 대신 다른 컴퓨터( VPS, 홈 서버, Tailscale 뒤의 Mini 등)에서 실행 중인 Hermes 백엔드를 지정할 수 있습니다.

**Settings → Gateway → Connection mode**에서 로컬 게이트웨이 대신 다음을 선택할 수 있습니다.

- **Remote gateway** — 직접 실행하는 `hermes serve` 백엔드의 URL을 입력하고 로그인합니다. 이 섹션의 나머지 부분에서 설명하는 모드입니다.
- **Hermes Cloud** — Hermes Cloud에 한 번 로그인하고 계정의 에이전트 중에서 선택합니다. 붙여 넣을 URL은 필요하지 않습니다. 앱은 에이전트를 검색하고(계정이 여러 조직에 속해 있으면 조직 선택기 표시), 에이전트에 연결하면 세션을 자동으로 전환합니다. 연결이 활성화되면 상태 표시줄에 클라우드 연결이 표시됩니다.

연결 모드는 **프로필별로** 구성됩니다. 프로필별 재정의를 사용하면 한 프로필은 원격 또는 클라우드 백엔드를 사용하고 다른 프로필은 로컬에 유지할 수 있습니다(**Use default gateway**로 재정의를 제거합니다).

### Settings → Connections: 다중 연결 레지스트리

위의 프로필별 연결 모드와 함께 **Settings → Connections**에서는 앱이 알고 있는 모든 에이전트 소스(로컬 런타임, 여러 LAN/Tailscale/인터넷 원격 게이트웨이, Hermes Cloud 인스턴스, SSH 호스트)를 이름이 지정된 레지스트리로 관리하며 모두 한곳에 저장합니다. 통합 에이전트 목록, `@name-device` 핸들, 전체 인스턴스 업데이트, 플러그인 SDK 화면을 포함한 전체 안내는 [여러 Hermes 인스턴스에 데스크톱 연결](./multi-connection-desktop.md)에서 확인할 수 있습니다.

- **모든 연결에는 고유한 이름이 필요합니다**("Homelab" 또는 "Work laptop"과 같은 장치 이름). 동일한 프로필 이름이 여러 등록 소스에 있으면 화면에 `@profile-device`(예: `@research-homelab`)로 구분됩니다.
- 패널에서 연결을 **추가 / 편집 / 제거 / 테스트**할 수 있습니다. 로컬 항목은 앱이 관리하므로 제거할 수 없습니다. **Test**는 연결 자체의 HTTP 및 WebSocket 경로를 직접 확인합니다.
- 레지스트리가 있는 빌드로 처음 실행할 때 기존 설정을 자동으로 **가져옵니다**. 현재 전역 연결과 프로필별 재정의가 이름이 지정된 항목으로 변환됩니다. 기존 설정 파일은 그대로 두므로 이전 빌드도 계속 작동합니다.
- 클라우드 항목은 위의 Hermes Cloud 로그인/검색 흐름에서 만들어지며 직접 입력한 URL에서 생성되지 않습니다.
- 토큰은 OS 키링에 암호화하여 저장됩니다(키링이 없는 Linux에서는 Settings → Gateway와 동일한 명시적 평문 저장 동의가 필요합니다).

나란히 보기 라우팅은 실시간으로 작동합니다. 등록된 각 소스가 필요할 때 자체 백엔드와 소켓에 연결하며(연결 + 프로필별로 키 지정), 플러그인 SDK는 통합 에이전트 목록(`host.agents()` / `host.ensureAgent()`)을 제공하고 **Update all instances**는 Connections 패널에서 자격이 있는 모든 소스에 `hermes update`를 동시에 전달합니다. Hermes Cloud 항목은 플랫폼이 업데이트하므로 건너뛰며, 각 인스턴스는 자체 결과를 보고합니다.

:::info 원격 백엔드는 실행 중인 `hermes serve` 프로세스입니다
"원격 백엔드"는 원격 컴퓨터에서 실행 중인 **`hermes serve`** 서버를 의미하며, 데스크톱 앱은 이 프로세스에 연결합니다. 백엔드가 실제로 실행 중이고 연결 가능한 상태가 아니면 이 섹션의 어떤 기능도 작동하지 않습니다. 데스크톱 앱은 백엔드를 대신 시작하지 않습니다. 사용자(또는 `systemd` 서비스)가 원격 호스트에서 `hermes serve`를 실행 상태로 유지하면 앱이 연결합니다. 메시징 채널(Telegram, Discord 등)도 사용한다면 **게이트웨이**는 별도로 시작하는 장시간 실행 프로세스입니다. 설정 단계 뒤의 참고 사항을 확인하세요.
:::

연결에는 두 부분이 있습니다. 백엔드에서는 **인증 제공자**로 보호하고, 앱에서는 백엔드 URL을 입력한 뒤 로그인합니다. 백엔드를 루프백이 아닌 주소에 바인딩하면 인증 게이트가 자동으로 활성화되며, 구성한 제공자가 데스크톱 앱을 통과시킵니다.

**백엔드 위치에 따라 제공자를 선택하세요.**

- **OAuth (Nous Portal) — 자신의 컴퓨터를 넘어 접근 가능한 경우 권장.** 로그인은 Nous 계정으로 확인되므로 VPS, 공개 호스트 또는 모든 원격 백엔드에 적합합니다. `hermes dashboard register`(또는 Portal의 [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) 페이지)로 대시보드를 등록하여 OAuth 클라이언트를 발급한 다음 앱에서 **Sign in with Nous Research**로 로그인합니다. 자체 ID 제공자를 운영한다면 자체 호스팅 OIDC 제공자도 같은 방식으로 사용할 수 있습니다.
- **사용자 이름/비밀번호 — 로컬 / 신뢰할 수 있는 네트워크에서만 사용.** 백엔드가 신뢰할 수 있는 동일 LAN에 있거나 VPN(예: Tailscale)을 통해서만 연결될 때 가장 간단한 옵션입니다. 외부 ID 제공자 없이 하나의 공유 자격 증명을 보호하므로 **공용 인터넷에 노출된 대시보드에는 사용하지 마세요** — 그 경우 OAuth를 사용하세요.

이 섹션의 나머지 부분에서는 신뢰할 수 있는 네트워크에서 가장 빠르게 구성할 수 있는 사용자 이름/비밀번호 경로를 설명합니다. OAuth 경로는 [웹 대시보드 → 기본 제공자: Nous Research](./features/web-dashboard.md#default-provider-nous-research)를 참고하세요.

### 백엔드에서 (원격 컴퓨터)

사용자 이름과 비밀번호를 설정한 다음 연결 가능한 주소에 바인딩하여 백엔드를 시작합니다. 자격 증명은 `~/.hermes/.env`(권한 0600인 비밀 파일)에 저장됩니다.

```bash
# 1. Set the dashboard login credentials.
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
# Without it a random key is generated per boot and you'll be logged out
# on every restart.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the backend bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes serve --host 0.0.0.0 --port 9119
```

데스크톱 앱이 연결할 수 있도록 하려는 동안 `hermes serve` 프로세스를 계속 실행하세요. 중지되면 앱이 더 이상 백엔드에 연결할 수 없습니다. 로그아웃과 재부팅 후에도 유지되도록 `systemd`, `tmux` 또는 원하는 프로세스 관리자로 실행하세요.

메시징 채널을 사용하는 경우 원격 호스트에서 **게이트웨이도 실행 중인지** 별도로 확인하세요. `hermes serve` 백엔드는 데스크톱 앱이 통신하는 대상이지만 Telegram/Discord/Slack 게이트웨이 세션은 별도의 프로세스이므로 따로 시작하고 계속 실행해야 합니다. 게이트웨이 설정은 [Messaging](./messaging/index.md)을 참고하세요.

평문 비밀번호를 디스크에 저장하고 싶지 않나요? 대신 `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH`를 scrypt 해시로 설정하세요. `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`로 계산할 수 있습니다. 전체 구성 화면(config.yaml 키, 모든 환경 변수, 속도 제한기)은 [웹 대시보드 → 사용자 이름/비밀번호 제공자](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp)를 참고하세요.

백엔드를 systemd 서비스로 실행하나요? 부팅 시 자격 증명이 환경에 들어가도록 유닛에 `EnvironmentFile=%h/.hermes/.env`를 지정하세요.

:::warning
백엔드는 `.env`(API 키, 비밀 정보)를 읽고 쓰며 에이전트 명령을 실행할 수 있습니다. 위에 제시한 **사용자 이름/비밀번호** 설정은 신뢰할 수 있는 네트워크용입니다. 비밀번호로 보호되는 백엔드를 개방형 인터넷에 직접 노출하지 말고 VPN 뒤에 두세요. [Tailscale](https://tailscale.com/)이 적합합니다. 컴퓨터의 Tailscale IP에 바인딩(`--host <tailscale-ip>`)하고 Remote URL에 `http://<tailscale-ip>:9119`를 사용하면 tailnet에서만 접근할 수 있습니다. 공용 인터넷을 통해 백엔드에 접근하려면 **OAuth (Nous Portal)** 제공자를 사용하세요.
:::

### 앱에서

**설정 → 게이트웨이 → 원격 게이트웨이:**

1. **Remote URL** — `http://<backend-host>:9119` (리버스 프록시 앞에 둘 때 `/hermes`와 같은 경로 접두사를 사용할 수 있습니다.)
2. **Sign in** — 앱이 백엔드가 알리는 제공자를 감지하고 버튼을 조정합니다. 사용자 이름/비밀번호 백엔드에는 자격 증명 양식을 여는 **Sign in** 버튼이 표시됩니다(1단계의 자격 증명을 입력). OAuth 백엔드에는 **Sign in with `<provider>`**(예: *Sign in with Nous Research*)가 표시되며 제공자의 브라우저 로그인을 실행합니다. 어느 경우든 앱은 백엔드에 인증된 세션을 갖게 됩니다.
3. **Save and reconnect** — 데스크톱 셸을 원격 백엔드로 전환합니다. 세션은 자동으로 갱신되며 `HERMES_DASHBOARD_BASIC_AUTH_SECRET`가 설정되어 있으면 재시작 후에도 로그인 상태가 유지됩니다.

UI를 사용하지 않고 앱을 실행하기 전에 `HERMES_DESKTOP_REMOTE_URL` 환경 변수로 백엔드 URL을 설정할 수도 있습니다(앱 내 설정을 덮어씀). 그래도 Gateway 설정 패널에서 로그인해야 합니다.

:::note 프로필별 원격 호스트
원격 게이트웨이 호스트는 [프로필](./profiles.md)별로 구성되므로 각 프로필이 자체 원격 백엔드를 사용하거나 로컬 백엔드에 남을 수 있습니다. 프로필을 전환하면 앱이 연결하는 원격 호스트도 전환됩니다.
:::

### 문제 해결

- **401 / "Invalid credentials"로 로그인이 실패함** — 사용자 이름 또는 비밀번호가 백엔드의 `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`와 일치하지 않습니다. 백엔드는 알 수 없는 사용자와 잘못된 비밀번호에 동일한 일반 오류를 반환하여 사용자 열거를 방지하므로 둘 다 다시 확인하세요. `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'`로 게이트가 켜져 있는지 확인하세요. `true`를 보고하고 `"basic"`을 포함해야 합니다.
- **"Sign in" 버튼이 없고 대신 세션 토큰을 요청함** — 백엔드의 사용자 이름/비밀번호 제공자가 활성화되지 않았습니다. `/api/status`의 `auth_providers`에 `"basic"`이 표시되지 않습니다. `~/.hermes/.env`에 사용자 이름과 비밀번호(또는 비밀번호 해시)가 모두 설정되어 있고 대시보드 프로세스가 실제로 이를 읽었는지 확인하세요.
- **재시작할 때마다 로그아웃됨** — `HERMES_DASHBOARD_BASIC_AUTH_SECRET`를 안정적인 값으로 설정하세요. 설정하지 않으면 부팅마다 토큰 서명 키가 재생성되어 모든 세션이 무효화됩니다.
- **연결이 거부되거나 시간 초과됨** — 백엔드가 `127.0.0.1`(기본값)에 바인딩되었거나 방화벽/VPN이 포트를 차단하고 있습니다. `0.0.0.0` 또는 Tailscale IP에 바인딩하고 신뢰할 수 있는 네트워크에 포트를 개방하세요.

웹 대시보드 관점에서 동일한 설정을 확인하려면 [웹 대시보드 → Hermes Desktop을 원격 백엔드에 연결](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)을 참고하세요. 환경 변수는 [환경 변수 → 웹 대시보드 및 Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop)에 정리되어 있습니다.

## 데스크톱 앱 확장

데스크톱 앱은 기여를 기반으로 합니다. 창, 페이지, 사이드바 탐색, 상태 표시줄 항목, 팔레트 명령, 키 바인드, 테마가 모두 하나의 SDK를 통해 등록되며 직접 추가할 수도 있습니다. 플러그인은 `$HERMES_HOME/desktop-plugins/<id>/plugin.js`에 넣는 단일 ESM 파일입니다. 앱은 몇 초 안에 이를 로드하고 저장할 때마다 핫 리로드합니다. 설치된 플러그인은 **Settings → Plugins**에서 실시간으로 관리할 수 있습니다.

전체 참조는 [Desktop Plugin SDK](../developer-guide/desktop-plugin-sdk.md)를 참고하세요. (이는 [웹 대시보드 플러그인 시스템](./features/extending-the-dashboard.md)과 별개입니다.)

## 문제 해결

부팅 로그는 `HERMES_HOME/logs/desktop.log`에 저장됩니다(백엔드 출력과 최근 Python 트레이스백 포함). 앱에서 부팅 실패를 보고하면 먼저 이 로그를 확인하세요. CLI에서도 다음 명령으로 끝까지 확인할 수 있습니다.

```bash
hermes logs gui -f
```

일반적인 초기화 방법:

```bash
# Force a clean first-launch setup (macOS/Linux)
rm "$HOME/.hermes/hermes-agent/.hermes-bootstrap-complete"

# Rebuild a broken Python venv (macOS/Linux)
rm -rf "$HOME/.hermes/hermes-agent/venv"

# Reset a stuck macOS microphone prompt
tccutil reset Microphone com.nousresearch.hermes
```

### Electron 다운로드에서 "Build desktop app"이 멈춤

빌드는 `github.com/electron/electron/releases`에서 Electron 런타임(약 114&nbsp;MB)을 다운로드합니다. 실시간 출력에 `retrying attempt=…`가 반복되며 **Build desktop app** 단계에서 설치 프로그램이 멈춘다면 네트워크에서 GitHub가 차단되었거나 제한된 것입니다(방화벽, 프록시 또는 지역 문제).

설치 프로그램은 이를 자동으로 복구합니다. 빌드 실패 시 (1) 손상된 Electron 캐시 zip을 지우고 재시도한 다음, (2) 여전히 실패하고 `ELECTRON_MIRROR`를 설정하지 않았다면 `npmmirror.com`을 통해 한 번 더 시도합니다. 이는 사실상 Electron 커뮤니티의 표준 미러입니다. `@electron/get`은 다운로드를 SHASUM으로 확인하지만 체크섬도 동일한 미러에서 가져옵니다. 따라서 손상되거나 불완전한 다운로드는 감지하지만 손상된 미러 자체는 감지하지 못합니다. 타사 호스트를 신뢰하고 싶지 않다면 아래처럼 직접 `ELECTRON_MIRROR`를 지정하세요. 빌드는 직접 지정한 값을 절대 덮어쓰지 않습니다.

**자체 미러를 선택**하려면(예: 기업 또는 신뢰하는 미러) 설치 전에 `ELECTRON_MIRROR`를 설정하거나 수동으로 다시 빌드하세요. 빌드는 해당 값을 적용하고 덮어쓰지 않습니다.

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  bash -c 'cd "$HOME/.hermes/hermes-agent/apps/desktop" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack'
```

손상된 캐시 zip을 직접 삭제하려면:

```bash
rm -f "$HOME/Library/Caches/electron"/electron-*.zip   # macOS
rm -f "$HOME/.cache/electron"/electron-*.zip            # Linux
```

## 소스에서 빌드

앱 자체를 수정하려면 먼저 저장소 루트에서 작업 공간 의존성을 한 번 설치한 다음 `apps/desktop`에서 개발 서버를 실행합니다.

```bash
npm install          # from repo root — links apps/desktop, web, apps/shared
cd apps/desktop
npm run dev          # Vite renderer + Electron, which boots the Python backend
```

특정 체크아웃을 앱에 지정하거나 실제 설정과 격리할 수 있습니다.

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=/tmp/throwaway npm run dev
npm run dev:fake-boot   # exercise the startup overlay with deterministic delays
```

설치 프로그램 빌드:

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ (no installer)
```

관련 자격 증명이 환경에 있으면 macOS/Windows 서명 및 공증이 자동으로 실행됩니다(macOS의 `CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*`, Windows의 `WIN_CSC_*`).

### macOS 권한 및 로컬 재빌드(TCC)

macOS는 경로가 아니라 앱의 *코드 서명 ID*에 대해 권한 부여(전체 디스크 접근, 데스크톱/다운로드/문서, 손쉬운 사용, 자동화, 마이크)를 기억합니다. 로컬에서 빌드하고 자체 업데이트하는 앱은 안정적인 식별자 고정 임시 서명으로 서명되므로 기본적으로 업데이트 후에도 권한이 유지됩니다.

가장 강력한 보장을 원한다면(yabai/skhd 사용자가 의존하는 것과 같은 인증서 기반 ID) 자체 서명 코드 서명 인증서를 한 번 만들고 Hermes가 이를 사용하도록 지정하세요.

1. 키체인 접근 → 인증서 지원 → **인증서 생성…**
2. 이름: `Hermes Local Signing`, ID 유형: *자체 서명 루트*, 인증서 유형: **코드 서명**.
3. `hermes config set desktop.macos_signing_identity "Hermes Local Signing"`

다음 업데이트에서는 해당 인증서로 재빌드된 앱에 다시 서명하므로 모든 TCC 권한 부여가 유지됩니다. Apple Developer 계정은 필요하지 않습니다. 공증된 릴리스 빌드는 감지되어 다시 서명되지 않습니다.

일회성 참고: 서명 ID를 변경하면(이 수정 후 첫 업데이트 포함) 앱의 ID가 한 번 변경되므로 macOS가 마지막으로 한 번 더 요청합니다. 이후 권한은 안정적으로 유지됩니다. 권한이 멈추면 `tccutil reset All com.nousresearch.hermes`로 초기화하고 다시 부여하세요.

## 함께 보기

- [CLI 안내](./cli.md) — 터미널 인터페이스
- [TUI](./tui.md) — `hermes --tui` 및 대시보드 채팅 탭에서 사용하는 최신 터미널 UI
- [웹 대시보드](./features/web-dashboard.md) — 채팅 탭이 삽입된 브라우저 관리 패널
- [구성](./configuration.md) — 데스크톱 앱이 읽고 쓰는 설정
- [Windows (Native)](./windows-native.md) — 네이티브 Windows 설치 경로
