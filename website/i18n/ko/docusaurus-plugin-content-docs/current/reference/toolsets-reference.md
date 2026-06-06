---
sidebar_position: 12
title: 번들 도구 (Bundled Toolsets)
description: Hermes 내장 코어 도구 세트 및 시스템 기능에 대한 참조입니다.
---

# 번들 도구 (Bundled Toolsets)

Hermes는 외부 스킬을 설치하기 전에도, 운영 체제와 상호 작용하고 웹에 접근하며 코드 작업을 할 수 있도록 하는 여러 핵심 "도구 세트(Toolsets)"가 기본적으로 제공됩니다.

다음은 `~/.hermes/config.yaml` 파일의 `toolsets` 하위에서 또는 `hermes tools` 명령어를 통해 활성화/비활성화할 수 있는 내장 도구 세트입니다. 도구 세트가 켜지면, 그 안에 포함된 모든 도구가 에이전트에 등록됩니다.

## 💻 쉘 및 시스템 (Shell & System)

### `shell_execution`
에이전트에게 셸 스크립트 실행 기능을 부여하여 터미널에서 사용자가 할 수 있는 모든 작업을 수행할 수 있게 합니다.

- **도구 (Tools):**
  - `run_script`: bash/zsh 스크립트 실행
  - `run_python`: Python 스크립트 실행 (Hermes의 자체 환경 사용)
  - `run_applescript`: (macOS 전용) OS 자동화를 위한 AppleScript 실행
- **참고:** 매우 강력합니다. 모델이 시스템을 변경할 수 있도록 하려면 활성화해야 하지만, 실행하기 전에 민감한 스크립트를 수락/거부할 수 있는 권한 승인(Approvals) 프롬프트를 함께 사용하는 것이 좋습니다.

### `file_system`
직접적인 셸 명령어를 실행하지 않고도 파일과 디렉토리를 안전하게 조작할 수 있는 전용 도구입니다.

- **도구 (Tools):**
  - `read_file`: 텍스트 및 기본 바이너리/이미지 콘텐츠 읽기
  - `write_file`: 파일 덮어쓰기 또는 생성
  - `append_file`: 파일 끝에 추가
  - `list_directory`: 폴더 내용 탐색
  - `search_files`: 디렉터리 내 정규식/글로벌 패턴(Glob) 검색
  - `edit_file_lines`: 전체 파일을 다시 쓰지 않고 특정 줄 바꾸기

## 🌐 웹 브라우징 (Web Browsing)

### `web_search`
에이전트가 외부 검색 엔진 API를 사용하여 최신 정보, 뉴스 또는 문서를 찾아볼 수 있게 합니다. API 키(`.env`)가 필요합니다.

- **도구 (Tools):**
  - `search_web`: 검색을 실행하고 요약/스니펫을 반환
- **지원 제공자:** SerpAPI, Tavily, Google Custom Search (설정에서 구성 가능)

### `web_fetch`
에이전트가 URL을 읽고 파싱할 수 있게 합니다. 간단한 요청을 위한 가벼운 HTTP 클라이언트가 내장되어 있습니다.

- **도구 (Tools):**
  - `read_url`: 텍스트 전용 마크다운 파싱을 사용한 기본 HTTP GET
  - `download_file`: 나중에 분석하기 위해 큰 페이로드를 디스크에 저장
- **참고:** 심층적인 브라우저 자동화(JavaScript 평가, 로그인)가 필요한 경우, 이 도구 대신 공식 `browser-automation/playwright` 스킬을 설치하세요.

## 🧑‍💻 개발자 유틸리티 (Developer Utilities)

### `git_tools`
셸 실행에 덜 의존하고 저장소를 검사하는 전용 Git 도구입니다.

- **도구 (Tools):**
  - `git_status`
  - `git_diff`
  - `git_log`
  - `git_blame`

### `github_tools`
에이전트가 코드를 푸시하거나 브라우저를 열지 않고도 GitHub API와 직접 통신할 수 있게 합니다. `GITHUB_TOKEN`이 필요합니다.

- **도구 (Tools):**
  - `create_pull_request`
  - `read_issue`
  - `add_issue_comment`
  - `search_code`

### `computer_use` (macOS 전용, 베타)
에이전트가 그래픽 인터페이스를 제어할 수 있도록 마우스, 키보드, 화면 캡처 권한을 부여합니다.
업스트림(Upstream) `cua-driver` 바이너리가 필요합니다 (`hermes computer-use install`).

- **도구 (Tools):**
  - `mouse_move`, `mouse_click`, `mouse_drag`
  - `keyboard_type`, `keyboard_shortcut`
  - `take_screenshot`, `find_on_screen`
- **참고:** 비전(Vision) 지원 모델(예: Claude 3.5 Sonnet)과 함께 사용할 때 가장 잘 작동합니다. 매우 실험적인 기능입니다.

## ⚙️ 에이전트 워크플로우 (Agent Workflow)

이 도구 세트들은 에이전트의 내부 관리 및 상태 저장을 처리합니다. 일반적으로 Hermes의 기능에 중요하므로 이 도구 세트는 **항상 활성화된 상태로 유지해야 합니다.**

### `memory_management`
에이전트가 장기 저장소에 대한 핵심 지식을 읽고, 쓰고, 업데이트할 수 있게 합니다.

- **도구 (Tools):**
  - `update_memory`: `MEMORY.md` 덮어쓰기
  - `append_memory`: 단기 큐(queue)에 추가

### `human_interaction`
에이전트가 수동적으로 응답하는 것을 넘어, 사용자가 개입하거나 피드백을 제공하도록 프롬프트를 띄울 수 있게 합니다.

- **도구 (Tools):**
  - `ask_user_for_input`: 에이전트 루프를 일시 중지하고 사용자에게 특정 질문이나 확인을 요구하는 메시지를 보냅니다.
  - `request_approval`: 명령을 실행하기 위해 예/아니요(Yes/No) 다이얼로그를 트리거합니다.

### `background_tasks`
에이전트가 시간이 많이 걸리는 작업을 생성하고 관리할 수 있게 합니다.

- **도구 (Tools):**
  - `spawn_task`: 백그라운드 셸 프로세스를 시작합니다.
  - `check_task_status`: 백그라운드 작업이 완료되었는지 확인합니다.
  - `kill_task`: 중단된 프로세스를 종료합니다.
  - `notify_when_done`: 백그라운드 작업이 완료되면 에이전트를 깨우는 콜백 훅을 설정합니다.

---

## 도구 비활성화 (Disabling Tools)

모델이 특정 도구에 너무 의존하거나 프롬프트 공간을 절약하고 싶다면 CLI를 사용하여 도구를 끌 수 있습니다:

```bash
hermes tools off github_tools
```

또는 설정 파일(`config.yaml`)을 편집하세요:

```yaml
toolsets:
  github_tools: false
  web_search: true
```

채팅 세션 중에 임시로 이 작업을 수행하려면 다음 슬래시 명령어를 사용하세요:

```text
/tools off github_tools
```
