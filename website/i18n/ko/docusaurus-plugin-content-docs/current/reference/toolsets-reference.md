---
sidebar_position: 4
title: "도구 세트 레퍼런스"
description: "Hermes의 핵심, 복합, 플랫폼 및 동적 도구 세트 레퍼런스"
---

# 도구 세트 레퍼런스

도구 세트는 에이전트가 수행할 수 있는 작업을 제어하는 이름이 지정된 도구 묶음입니다. 플랫폼, 세션 또는 작업별로 도구 사용 가능 여부를 구성하는 기본 메커니즘입니다.

## 도구 세트 작동 방식

모든 도구는 정확히 하나의 도구 세트에 속합니다. 도구 세트를 활성화하면 해당 묶음의 모든 도구를 에이전트가 사용할 수 있습니다. 도구 세트에는 세 가지 종류가 있습니다.

- **핵심** — 관련 도구를 하나의 논리적 그룹으로 묶습니다(예: `file`은 `read_file`, `write_file`, `patch`, `search_files`를 묶음).
- **복합** — 일반적인 시나리오를 위해 여러 핵심 도구 세트를 결합합니다(예: `debugging`은 file, terminal, web 도구를 묶음).
- **플랫폼** — 특정 배포 컨텍스트를 위한 완전한 도구 구성입니다(예: `hermes-cli`는 대화형 CLI 세션의 기본값).

## 도구 세트 구성

### 세션별(CLI)

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # composite — expands to file + terminal + web
hermes chat --toolsets all              # everything
```

### 플랫폼별(config.yaml)

```yaml
toolsets:
  - hermes-cli          # default for CLI
  # - hermes-telegram   # override for Telegram gateway
```

### 대화형 관리

```bash
hermes tools                            # curses UI to enable/disable per platform
```

또는 세션 중에 다음을 사용합니다.

```
/tools list
/tools disable browser
/tools enable homeassistant
```

## 핵심 도구 세트

| 도구 세트 | 도구 | 용도 |
|---------|-------|---------|
| `browser` | `browser_back`, `browser_cdp`, `browser_click`, `browser_console`, `browser_dialog`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `web_search` | 핵심 브라우저 자동화입니다. 빠른 조회를 위한 대체 수단으로 `web_search`를 포함합니다. `browser_cdp`와 `browser_dialog`는 런타임에서 게이트됩니다. 즉, 세션 시작 시 CDP 엔드포인트에 연결할 수 있을 때(`/browser connect`, `browser.cdp_url` 설정, Browserbase 또는 Camofox를 통해서만) 등록됩니다. `browser_dialog`는 CDP 감독자가 연결되었을 때 `browser_snapshot`이 추가하는 `pending_dialogs` 및 `frame_tree` 필드와 함께 작동합니다. |
| `clarify` | `clarify` | 에이전트에 명확한 정보가 필요할 때 사용자에게 질문합니다. |
| `code_execution` | `execute_code` | Hermes 도구를 프로그래밍 방식으로 호출하는 Python 스크립트를 실행합니다. |
| `coding` | composite (`file` + `terminal` + `search` + `web` + `skills` + `browser` + `todo` + `memory` + `session_search` + `clarify` + `code_execution` + `delegation` + `vision`) | 파일 편집, 터미널, 검색, 웹 문서, 스킬, 브라우저, 할 일, 메모리, 세션 검색, 명확화, 위임 및 코드 실행을 포함하는 코딩 중심 번들입니다. |
| `cronjob` | `cronjob` | 반복 작업을 예약하고 관리합니다. |
| `debugging` | composite (`file` + `terminal` + `web`) | 디버그 번들입니다. 파일, 프로세스/터미널, 웹 추출/검색을 포함합니다. |
| `delegation` | `delegate_task` | 병렬 작업을 위해 격리된 하위 에이전트 인스턴스를 생성합니다. |
| `discord` | `discord` | 핵심 Discord 텍스트/임베드/DM 작업입니다(게이트웨이 전용). `hermes-discord` 도구 세트에서 활성화됩니다. |
| `discord_admin` | `discord_admin` | Discord 관리 작업입니다(차단, 역할 변경, 채널 관리). `hermes-discord` 도구 세트에서 활성화되며, 봇이 관련 Discord 권한을 보유해야 합니다. |
| `feishu_doc` | `feishu_doc_read` | Feishu/Lark 문서 콘텐츠를 읽습니다. Feishu 문서 댓글 지능형 답변 핸들러에서 사용됩니다. |
| `feishu_drive` | `feishu_drive_add_comment`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment` | Feishu/Lark 드라이브 댓글 작업입니다. 댓글 에이전트로 범위가 제한되며 `hermes-cli` 또는 다른 메시징 도구 세트에는 노출되지 않습니다. |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | 파일 읽기, 쓰기, 검색 및 편집입니다. |
| `homeassistant` | `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services` | Home Assistant를 통한 스마트 홈 제어입니다. `HASS_TOKEN`이 설정된 경우에만 사용할 수 있습니다. |
| `computer_use` | `computer_use` | 백그라운드 데스크톱 제어입니다. 커서나 포커스를 빼앗지 않습니다. 도구 사용이 가능한 모든 모델에서 작동하며, macOS, Windows 및 Linux를 지원합니다. `$PATH`에 `cua-driver`가 있어야 합니다. |
| `context_engine` | (varies) | 활성 컨텍스트 엔진 플러그인이 노출하는 런타임 도구입니다(플러그인이 도구를 채울 때까지 비어 있음). |
| `image_gen` | `image_generate` | FAL.ai를 통한 텍스트-이미지 생성입니다(선택적으로 OpenAI / xAI 백엔드를 사용할 수 있음). |
| `video_gen` | `video_generate`, `xai_video_edit`, `xai_video_extend` | 플러그인에 등록된 백엔드(xAI Grok-Imagine, FAL.ai Veo 3.1 / Pixverse v6 / Kling O3)를 통한 텍스트-비디오 및 이미지-비디오 생성입니다. 이미지에 애니메이션을 적용하려면 `image_url`을 전달하고, 텍스트-비디오에는 생략합니다. `xai_video_edit` / `xai_video_extend`는 xAI Imagine 자격 증명이 있을 때 게이트되는 제공자별 편집/확장 도구입니다. |
| `kanban` | `kanban_attach`, `kanban_attach_url`, `kanban_attachments`, `kanban_block`, `kanban_comment`, `kanban_complete`, `kanban_create`, `kanban_heartbeat`, `kanban_link`, `kanban_list`, `kanban_request_changes`, `kanban_request_review`, `kanban_show`, `kanban_unblock` | 다중 에이전트 조정 도구입니다. 디스패처가 생성한 작업 워커(`HERMES_KANBAN_TASK`)와 이름으로 `kanban` 도구 세트를 명시적으로 나열한 프로필에 등록됩니다(`all`/`*` 와일드카드는 활성화하지 않음). 워커는 작업 완료, 첫 번째 클래스 리뷰 요청, 차단, 하트비트, 댓글 작성 및 후속 작업 생성/연결을 수행하며, 오케스트레이터 프로필에는 목록/차단 해제와 같은 보드 라우팅 도구가 추가로 제공됩니다. `delegate_task` 하위 에이전트는 Kanban 실행 소유자가 아닙니다. 해당 도구의 스키마는 이 도구 세트를 제거/비활성화하며, 부모의 `HERMES_KANBAN_*` 환경 변수가 있어도 런타임 가드가 직접적인 보드 변경을 거부합니다. |
| `memory` | `memory` | 세션 간 영구 메모리를 관리합니다. |
| `desktop_ui` | `close_terminal`, `focus_pane`, `open_preview`, `react_to_message`, `read_preview`, `read_terminal`, `read_window_below` | Hermes 데스크톱 앱 자체에서 작동하는 기능입니다. 내장 터미널 패널 읽기/닫기, 인앱 브라우저 열기/읽기, 앱 뒤의 OS 창 식별, 패널 표시, 메시지에 반응하기를 지원합니다. 연결된 백엔드가 로컬, SSH, URL 또는 Hermes Cloud 중 무엇이든 데스크톱 앱에서 시작된 세션에 활성화됩니다. CLI, TUI, 메시징 또는 cron 세션에는 절대 표시되지 않습니다. |
| `project` | `project_create`, `project_list`, `project_switch` | 데스크톱 [Projects](../user-guide/cli.md)(이름이 지정된 다중 폴더 작업공간)를 생성하고 전환합니다. GUI/데스크톱 세션 전용입니다. |
| `safe` | `image_generate`, `vision_analyze`, `web_extract`, `web_search` (via `includes`) | 읽기 전용 연구 및 미디어 생성입니다. 파일 쓰기, 터미널, 코드 실행은 포함하지 않습니다. |
| `search` | `web_search` | 추출 없이 웹 검색만 수행합니다. |
| `session_search` | `session_search` | 과거 대화 세션을 검색합니다. |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | 스킬 CRUD 및 탐색입니다. |
| `spotify` | `spotify_albums`, `spotify_devices`, `spotify_library`, `spotify_playback`, `spotify_playlists`, `spotify_queue`, `spotify_search` | 기본 제공 `spotify` 플러그인이 등록하는 네이티브 Spotify 제어(재생, 대기열, 검색, 플레이리스트, 앨범, 라이브러리)입니다. |
| `terminal` | `process`, `terminal` | 셸 명령 실행 및 백그라운드 프로세스 관리입니다. |
| `todo` | `todo` | 세션 내 작업 목록을 관리합니다. |
| `tts` | `text_to_speech` | 음성 합성 오디오를 생성합니다. |
| `vision` | `vision_analyze` | 비전 기능을 지원하는 모델을 통한 이미지 분석입니다. |
| `video` | `video_analyze` | 비디오 분석 및 이해 도구입니다(선택 기능이며 기본 도구 세트에는 없음 — `--toolsets`를 통해 명시적으로 추가). |
| `web` | `web_extract`, `web_search` | 웹 검색 및 페이지 콘텐츠 추출입니다. |
| `x_search` | `x_search` | xAI의 기본 제공 `x_search` Responses 도구를 통한 읽기 전용 공개 X 탐색입니다. 인증된 X API 읽기 및 계정 작업에는 `xurl` 스킬을 사용합니다. 기본적으로 꺼져 있으며 `hermes tools`를 통해 선택적으로 활성화합니다. xAI 자격 증명(SuperGrok OAuth 또는 `XAI_API_KEY`)이 구성된 경우에만 스키마가 등록됩니다. |
| `yuanbao` | `yb_query_group_info`, `yb_query_group_members`, `yb_search_sticker`, `yb_send_dm`, `yb_send_sticker` | Yuanbao DM/그룹 작업 및 스티커 검색입니다. `hermes-yuanbao`에서만 등록됩니다. |

## 플랫폼 도구 세트

플랫폼 도구 세트는 배포 대상에 대한 전체 도구 구성을 정의합니다. 대부분의 메시징 플랫폼은 `hermes-cli`와 동일한 세트를 사용합니다.

| 도구 세트 | `hermes-cli`와의 차이 |
|---------|-------------------------------|
| `hermes-cli` | 전체 도구 세트입니다. 파일, 터미널, 웹, 브라우저, 메모리, 스킬, 비전, image_gen, todo, tts, delegation, code_execution, cronjob, session_search, clarify, computer_use, Home Assistant 및 kanban 도구를 포함합니다(모두 런타임에서 `check_fn`으로 게이트됨). 대화형 CLI 세션의 기본값입니다. |
| `hermes-acp` | `clarify`, `cronjob`, `image_generate`, `text_to_speech`, `computer_use`, 네 가지 Home Assistant 도구 및 kanban 도구를 제외합니다. IDE 컨텍스트의 코딩 작업에 집중합니다. |
| `hermes-api-server` | `clarify`, `text_to_speech`, `computer_use` 및 kanban 도구를 제외합니다. 그 외에는 모두 유지하며, 사용자 상호작용이 불가능한 프로그래밍 방식의 접근에 적합합니다. |
| `hermes-cron` | `hermes-cli`와 동일합니다. |
| `hermes-telegram` | `hermes-cli`와 동일합니다. |
| `hermes-discord` | `hermes-cli`에 `discord` 및 `discord_admin`을 추가합니다. |
| `hermes-slack` | `hermes-cli`와 동일합니다. |
| `hermes-whatsapp` | `hermes-cli`와 동일합니다. |
| `hermes-signal` | `hermes-cli`와 동일합니다. |
| `hermes-matrix` | `hermes-cli`와 동일합니다. |
| `hermes-mattermost` | `hermes-cli`와 동일합니다. |
| `hermes-email` | `hermes-cli`와 동일합니다. |
| `hermes-sms` | `hermes-cli`와 동일합니다. |
| `hermes-bluebubbles` | `hermes-cli`와 동일합니다. |
| `hermes-dingtalk` | `hermes-cli`와 동일합니다. |
| `hermes-feishu` | 다섯 개의 `feishu_doc_*` / `feishu_drive_*` 도구를 추가합니다(일반 채팅 어댑터가 아니라 문서 댓글 핸들러에서만 사용). |
| `hermes-qqbot` | `hermes-cli`와 동일합니다. |
| `hermes-wecom` | `hermes-cli`와 동일합니다. |
| `hermes-wecom-callback` | `hermes-cli`와 동일합니다. |
| `hermes-weixin` | `hermes-cli`와 동일합니다. |
| `hermes-yuanbao` | `hermes-cli`에 다섯 개의 `yb_*` 도구(DM/그룹/스티커)를 추가합니다. |
| `hermes-homeassistant` | `hermes-cli`와 동일합니다(Home Assistant 도구는 기본적으로 이미 포함되어 있으며 `HASS_TOKEN`이 설정되면 활성화됨). |
| `hermes-webhook` | 제한된 안전 하위 집합입니다. `web_search`, `web_extract`, `vision_analyze` 및 `clarify`만 포함합니다. 웹훅으로 트리거된 실행에는 터미널, 파일 또는 브라우저 접근 권한이 없습니다. |
| `hermes-gateway` | 내부 게이트웨이 오케스트레이터 도구 세트입니다. 모든 `hermes-<platform>` 도구 세트의 합집합이며, 게이트웨이가 모든 메시지 소스를 수락해야 할 때 사용합니다. |

## 동적 도구 세트

### MCP 서버 도구 세트

구성된 각 MCP 서버는 런타임에 `mcp-<server>` 도구 세트를 생성합니다. 예를 들어 `github` MCP 서버를 구성하면 해당 서버가 노출하는 모든 도구를 포함하는 `mcp-github` 도구 세트가 생성됩니다.

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

생성된 도구 세트는 `--toolsets` 또는 플랫폼 설정에서 참조할 수 있는 `mcp-github`입니다.

### 플러그인 도구 세트

플러그인은 초기화 중 `ctx.register_tool()`을 통해 자체 도구 세트를 등록할 수 있습니다. 이러한 도구 세트는 기본 제공 도구 세트와 함께 표시되며 동일한 방식으로 활성화/비활성화할 수 있습니다.

### 사용자 지정 도구 세트

`config.yaml`에서 사용자 지정 도구 세트를 정의하여 프로젝트별 묶음을 생성할 수 있습니다.

```yaml
toolsets:
  - hermes-cli
custom_toolsets:
  data-science:
    - file
    - terminal
    - code_execution
    - web
    - vision
```

### 와일드카드

- `all` 또는 `*` — 등록된 모든 도구 세트(기본 제공 + 동적 + 플러그인)로 확장됩니다.

일부 도구는 도구 세트 소속 여부 외에 추가 가용성 확인이 필요하므로 `all`/`*`만으로는 활성화되지 않습니다.

- **기능 게이트 도구**(browser, `computer_use`, `code_execution`, Feishu, Home Assistant, cronjob)는 백엔드/자격 증명 전제 조건이 구성된 경우에만 표시됩니다.
- **워크플로 게이트 도구** — `kanban` 도구 세트는 의도적으로 선택 기능입니다. `all`/`*`로는 kanban이 활성화되지 않으며, `kanban`을 명시적으로 나열하거나 `HERMES_KANBAN_TASK`가 설정된 디스패처 생성 워커여야 합니다. Kanban 도구는 공유 보드 상태를 변경하므로 `all` 아래에서도 기본적으로 꺼져 있습니다.

## `hermes tools`와의 관계

`hermes tools` 명령은 플랫폼별로 개별 도구를 켜고 끄는 curses 기반 UI를 제공합니다. 이는 도구 세트보다 세밀한 도구 수준에서 작동하며 설정을 `config.yaml`에 저장합니다. 도구 세트가 활성화되어 있어도 비활성화된 도구는 필터링됩니다.

개별 도구와 해당 매개변수의 전체 목록은 [Tools Reference](./tools-reference.md)를 참조하세요.
