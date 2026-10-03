---
sidebar_position: 3
title: "내장 도구 레퍼런스"
description: "도구 세트별로 분류한 Hermes 내장 도구의 공식 레퍼런스"
---

# 내장 도구 레퍼런스

이 페이지에서는 도구 세트별로 분류한 Hermes의 내장 도구를 설명합니다. 사용 가능 여부는 플랫폼, 자격 증명, 활성화된 도구 세트에 따라 달라집니다.

**빠른 개수 확인(현재 레지스트리):** 약 83개 도구 — 브라우저 도구 10개(코어) + CDP로 게이트되는 브라우저 도구 2개, 파일 도구 4개, Home Assistant 도구 4개, 터미널 도구 2개(`terminal`, `process`), 데스크톱 GUI 도구 7개(`read_terminal`, `close_terminal`, `open_preview`, `read_preview`, `read_window_below`, `focus_pane`, `react_to_message` — 데스크톱 앱 세션에서만 사용 가능), 웹 도구 2개, Feishu 도구 5개, Spotify 도구 7개(번들로 제공되는 `spotify` 플러그인이 등록), Yuanbao 도구 5개, 칸반 도구 12개(칸반 디스패처가 에이전트를 생성할 때 등록), 프로젝트 도구 3개(데스크톱/GUI 세션), Discord 도구 2개, 비디오 도구 3개(`video_generate`, `xai_video_edit`, `xai_video_extend`), 그리고 여러 독립 실행형 도구(`memory`, `clarify`, `delegate_task`, `execute_code`, `cronjob`, `session_search`, `skill_view`/`skill_manage`/`skills_list`, `text_to_speech`, `image_generate`, `vision_analyze`, `video_analyze`, `todo`, `computer_use`, `x_search`).

:::tip MCP 도구
내장 도구 외에도 Hermes는 MCP 서버에서 도구를 동적으로 불러올 수 있습니다. MCP 도구에는 `mcp__<server>__` 접두사가 붙습니다(예: `github` MCP 서버의 `mcp__github__create_issue`). 구성 방법은 [MCP 통합](/user-guide/features/mcp)을 참조하세요.
:::

## `browser` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `browser_back` | 브라우저 기록에서 이전 페이지로 돌아갑니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_click` | 스냅샷의 ref ID(예: '@e5')로 식별되는 요소를 클릭합니다. ref ID는 스냅샷 출력에서 대괄호 안에 표시됩니다. 먼저 browser_navigate와 browser_snapshot을 호출해야 합니다. | — |
| `browser_console` | 현재 페이지의 브라우저 콘솔 출력과 JavaScript 오류를 가져옵니다. console.log/warn/error/info 메시지와 처리되지 않은 JavaScript 예외를 반환합니다. 조용히 발생하는 JavaScript 오류, 실패한 API 호출, 애플리케이션 경고를 감지할 때 사용합니다. Requi… | — |
| `browser_get_images` | 현재 페이지의 모든 이미지와 해당 URL 및 alt 텍스트 목록을 가져옵니다. vision 도구로 분석할 이미지를 찾을 때 유용합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_navigate` | 브라우저에서 URL로 이동합니다. 세션을 초기화하고 페이지를 불러옵니다. 다른 브라우저 도구를 호출하기 전에 반드시 호출해야 합니다. 단순 정보 검색에는 web_search 또는 web_extract가 더 빠르고 저렴하므로 이를 우선 사용하세요. 브라우저 도구는 다음과 같은 경우에 사용합니다… | — |
| `browser_press` | 키보드 키를 누릅니다. 양식 제출(Enter), 탐색(Tab), 키보드 단축키에 유용합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_scroll` | 특정 방향으로 스크롤합니다. 현재 뷰포트 아래나 위에 있는 내용을 표시할 때 사용합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_snapshot` | 현재 페이지의 접근성 트리를 텍스트 기반 스냅샷으로 가져옵니다. full=false(기본값): 대화형 요소가 포함된 간결한 보기입니다. full=true: comp… | — |
| `browser_type` | ref ID로 식별되는 입력 필드에 텍스트를 입력합니다. 먼저 필드를 지운 후 텍스트를 입력합니다. browser_navigate와 browser_snapshot을 먼저 호출해야 합니다. | — |
| `browser_vision` | 현재 페이지의 스크린샷을 찍어 시각적으로 검사합니다. 특히 CAPTCHA, 시각적 확인 문제, 복잡한 레이아웃을 이해해야 하거나 텍스트 스냅샷에 누락된 시각 정보가 있을 때 사용합니다. 네이티브 vision 모델에서는 스크린샷이 직접 첨부되고, 그 외에는 보조 vision 모… | — |

## `browser` 도구 세트(CDP로 게이트됨)

이 두 도구는 세션 시작 시 Chrome DevTools Protocol 엔드포인트에 연결할 수 있을 때만 `browser` 도구 세트에 등록됩니다 — `/browser connect`, `browser.cdp_url` 구성, Browserbase 세션 또는 Camofox를 통해 연결할 수 있습니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `browser_cdp` | 상위 수준의 `browser_*` 도구로 지원되지 않는 브라우저 작업을 위한 우회 수단으로, 원시 Chrome DevTools Protocol 명령을 보냅니다. https://chromedevtools.github.io/devtools-protocol/ 을 참조하세요. | CDP 엔드포인트 |
| `browser_dialog` | 네이티브 JavaScript 대화상자(alert / confirm / prompt / beforeunload)에 응답합니다. 먼저 `browser_snapshot`을 호출하세요 — 대기 중인 대화상자는 `pending_dialogs` 필드에 표시됩니다. 그런 다음 `browser_dialog(action='accept'\|'dismiss')`를 호출하세요. | CDP 엔드포인트 |

## `clarify` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `clarify` | 진행하기 전에 설명, 피드백 또는 결정을 요청할 때 사용자에게 질문합니다. 세 가지 모드를 지원합니다. 1. **단일 선택 객관식** — 최대 4개 선택지이며, 사용자는 하나를 고르거나 다섯 번째 '기타' 선택지를 통해 직접 입력할 수 있습니다. 2. **복수 선택 객관식** — `multi_select=true`로 체크박스를 표시하고 선택한 항목 목록을 반환합니다. 3. **개방형** — 선택지 없이 자유 형식으로 입력합니다. 선택지는 가장 적합한 순서로 정렬되므로 모든 화면에서 첫 번째 항목에 `(Recommended)`가 표시되고 기본 선택으로 강조됩니다. 이 레이블은 표시용일 뿐이며 에이전트가 읽는 답변에서는 제거됩니다. 클래식 CLI에서 복수 선택은 Space로 체크박스를 전환합니다. 네이티브 체크박스 UI가 없는 메시징 플랫폼에서는 사용자가 쉼표/공백으로 구분된 번호(예: "1, 3") 또는 선택지 텍스트로 답합니다. | — |

## `code_execution` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `execute_code` | 프로그래밍 방식으로 Hermes 도구를 호출할 수 있는 Python 스크립트를 실행합니다. 처리 로직이 포함된 도구 호출을 3개 이상 수행해야 하거나, 큰 도구 출력을 컨텍스트에 넣기 전에 필터링/축소해야 하거나, 조건부 분기를 사용해야 할 때 이용합니다… | — |

## `cronjob` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `cronjob` | 통합 예약 작업 관리자입니다. `action="create"`, `"list"`, `"update"`, `"pause"`, `"resume"`, `"run"` 또는 `"remove"`를 사용해 작업을 관리합니다. 하나 이상의 연결된 스킬이 있는 스킬 기반 작업과, 연결된 스킬을 제거하는 업데이트의 `skills=[]`를 지원합니다. Cron 실행은 현재 대화 컨텍스트가 없는 새 세션에서 이루어집니다. | — |

## `delegation` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `delegate_task` | 격리된 컨텍스트에서 하위 에이전트를 생성합니다. 각 하위 에이전트에는 자체 대화, 터미널 세션, 도구 세트가 제공되며 최종 요약만 반환됩니다. 단일 작업에는 'goal'을, 병렬 배치에는 'tasks'를 제공합니다(제한 및 중첩 규칙… | — |

## `feishu_doc` 도구 세트

Feishu 문서 댓글 지능형 답변 처리기(`gateway/platforms/feishu_comment.py`) 전용입니다. `hermes-cli` 또는 일반 Feishu 채팅 어댑터에는 노출되지 않습니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `feishu_doc_read` | file_type과 token을 지정해 Feishu/Lark 문서(Docx, Doc 또는 Sheet)의 전체 텍스트 내용을 읽습니다. | Feishu 앱 자격 증명 |

## `feishu_drive` 도구 세트

Feishu 문서 댓글 처리기 전용입니다. 드라이브 파일의 댓글 읽기/쓰기 작업을 처리합니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `feishu_drive_add_comment` | Feishu/Lark 문서 또는 파일에 최상위 댓글을 추가합니다. | — |
| `feishu_drive_list_comments` | Feishu/Lark 파일의 전체 문서 댓글을 최신순으로 나열합니다. | — |
| `feishu_drive_list_comment_replies` | 특정 Feishu 댓글 스레드(문서 전체 또는 로컬 선택 영역)의 답글을 나열합니다. | — |
| `feishu_drive_reply_comment` | 선택적으로 `@` 멘션을 포함해 Feishu 댓글 스레드에 답글을 게시합니다. | — |

## `file` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `patch` | 파일에서 지정한 부분을 찾아 바꾸는 편집을 수행합니다. 터미널 대신 이 도구를 사용하세요. 퍼지 매칭(9가지 전략)을 사용하므로 사소한 공백/들여쓰기 차이가 있어도 실패하지 않습니다. 편집 후 구문 검사를 자동으로 실행합니다… | — |
| `read_file` | 줄 번호와 페이지 나누기를 사용해 텍스트 파일을 읽습니다. 출력 형식은 'LINE_NUM\|CONTENT'입니다. 파일을 찾지 못하면 유사한 파일 이름을 제안합니다. 큰 파일에는 offset과 limit을 사용합니다. 약 100K자를 초과하는 읽기는 줄 경계에서 잘리며 next_offset을 반환합니다. Jupyter 노트북(.ipynb), Word 문서(.docx), Excel 통합 문서(.xlsx)도… | — |
| `search_files` | 파일 내용에서 검색하거나 이름으로 파일을 찾습니다. 셸 명령보다 빠른 Ripgrep 기반입니다. 내용 검색(target='content'): 파일 내부의 정규식을 검색합니다. 출력 모드: 일치 항목 전체… | — |
| `write_file` | 파일 내용을 완전히 교체해 씁니다. 상위 디렉터리를 자동으로 생성합니다. 전체 파일을 덮어쓰므로 대상 편집에는 `patch`를 사용하세요. 편집 후 구문 검사를 자동으로 실행합니다… | — |

## `homeassistant` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `ha_call_service` | Home Assistant 서비스를 호출해 장치를 제어합니다. 각 도메인에 사용할 수 있는 서비스와 매개변수를 확인하려면 ha_list_services를 사용하세요. | — |
| `ha_get_state` | 밝기, 색상, 온도 설정값, 센서 측정값 등 모든 속성을 포함한 단일 Home Assistant 엔터티의 상세 상태를 가져옵니다. | — |
| `ha_list_entities` | Home Assistant 엔터티를 나열합니다. 도메인(light, switch, climate, sensor, binary_sensor, cover, fan 등) 또는 영역 이름(거실, 주방, 침실 등)으로 선택적으로 필터링할 수 있습니다. | — |
| `ha_list_services` | 장치 제어에 사용할 수 있는 서비스를 나열합니다. ha_list_entities에서 찾은 장치 유형마다 수행할 수 있는 작업을 확인할 때 사용합니다. | — |

## `computer_use` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `computer_use` | cua-driver를 통한 백그라운드 데스크톱 제어 — 스크린샷(SOM / vision / AX), 클릭 / 드래그 / 스크롤 / 입력 / 키 / 대기, list_apps, focus_app을 지원합니다. 사용자의 커서나 키보드 포커스를 빼앗지 않습니다. 도구를 사용할 수 있는 모든 모델에서 작동합니다. macOS, Windows, Linux를 지원합니다. | `$PATH`에 `cua-driver`(설치하려면 `hermes tools`) |


:::note
**Honcho 도구**(`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_reasoning`, `honcho_conclude`)는 더 이상 내장되지 않습니다. Honcho 메모리 제공자 플러그인 `plugins/memory/honcho/`를 통해 사용할 수 있습니다. 설치 및 사용 방법은 [메모리 제공자](../user-guide/features/memory-providers.md)를 참조하세요.
:::

## `image_gen` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `image_generate` | 사용자가 구성한 백엔드(FAL.ai, OpenAI, OpenAI Codex auth, xAI, Krea)를 통해 텍스트 프롬프트로 이미지를 생성하거나(text-to-image), 기존 이미지를 편집/변환합니다(image-to-image). 이미지를 편집하려면 `image_url`을, 스타일 참조에는 `reference_image_urls`를 전달하고, text-to-image에는 둘 다 생략합니다. 모델은 사용자가 구성하며 에이전트가 선택할 수 없습니다. 단일 이미지 URL 또는 로컬 경로를 반환합니다. | FAL_KEY / OPENAI_API_KEY / Codex OAuth / xAI OAuth / KREA_API_KEY |

## `kanban` 도구 세트

에이전트가 (a) 칸반 디스패처에 의해 생성되었거나(`HERMES_KANBAN_TASK` 환경 변수 설정), (b) `kanban` 도구 세트를 명시적으로 활성화한 프로필에서 실행 중일 때 등록됩니다. 작업 범위가 지정된 워커는 할당된 작업을 위한 수명 주기 도구를 사용하며, 오케스트레이터 프로필에는 `kanban_list` 및 `kanban_unblock` 같은 보드 라우팅 도구도 추가됩니다. 전체 작업 흐름은 [칸반 멀티 에이전트](/user-guide/features/kanban)를 참조하세요.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `kanban_show` | 이 워커에 할당된 활성 칸반 작업(제목, 설명, 댓글, 종속성)을 표시합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_list` | 필터를 사용해 보드 작업을 나열합니다. 오케스트레이터 전용이며 디스패처가 생성한 작업 워커에게는 숨겨집니다. | `kanban` 도구 세트가 있는 프로필 |
| `kanban_complete` | 구조화된 인계 페이로드(결과, 산출물, 후속 작업)와 함께 현재 작업을 완료로 표시합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_block` | 사용자에게 질문하는 현재 작업을 차단합니다 — 디스패처는 작업을 일시 중지하고 질문을 표시하며, 사람이 답하면 재개합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_request_review` | `summary`, 선택적 구조화 `metadata`, 선택적 리뷰어 프로필과 함께 구현을 리뷰어에게 넘깁니다. 동일한 작업을 `review`로 이동하며, 차단이 아니므로 차단 루프 수에 영향을 주지 않습니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_request_changes` | 적극적으로 클레임된 리뷰 실행에 대한 리뷰어 판정입니다. 리뷰 실행을 종료하고 부모 게이트를 다시 적용한 뒤, 차단 없이 작업을 원래 구현자에게 돌려보냅니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_heartbeat` | 디스패처가 워커가 아직 실행 중임을 알 수 있도록 장시간 작업 중 진행 하트비트를 보냅니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_comment` | 상태를 변경하지 않고 작업 스레드에 댓글을 추가합니다 — 중간 결과를 알릴 때 유용합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_create` | 현재 작업에서 하위 작업을 분배합니다. 오케스트레이터와 후속 작업을 생성하는 워커가 사용합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_link` | 부모 → 자식 종속성 간선을 사용해 작업을 연결합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_unblock` | 모든 부모가 완료되면 차단된 작업을 `ready`로, 부모 중 하나라도 열려 있으면 `todo`로 이동합니다. 오케스트레이터 전용이며 디스패처가 생성한 작업 워커에게는 숨겨집니다. | `kanban` 도구 세트가 있는 프로필 |
| `kanban_attach` | 바이트를 인라인(base64)으로 전달해 파일을 작업에 첨부합니다. 작업의 attachments 디렉터리에 실제 첨부 파일로 저장되며 25MB로 제한됩니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_attach_url` | URL로 파일을 작업에 첨부합니다 — Hermes가 서버 측에서 다운로드해 실제 첨부 파일로 저장하며 25MB로 제한됩니다. http/https URL만 허용됩니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_attachments` | 작업에 첨부된 파일을 나열합니다: id, filename, content_type, size, uploader 및 디스크상의 절대 경로입니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |

## `project` 도구 세트

데스크톱 [프로젝트](../user-guide/cli.md) — 이름이 지정된 다중 폴더 작업 공간을 제어하는 도구입니다. `project` 도구 세트가 활성화된 경우(주로 데스크톱 앱/대시보드 표면) 등록됩니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `project_create` | 데스크톱 프로젝트(이름이 지정된 작업 공간)를 만들고 이 채팅을 해당 프로젝트로 전환합니다. `path`를 전달해 저장소/폴더에 기준을 둘 수 있습니다. | — |
| `project_list` | 데스크톱 프로젝트와 현재 활성 프로젝트를 나열합니다. | — |
| `project_switch` | 기존 프로젝트(이름, slug 또는 id)를 전환합니다. 세션 작업 공간을 프로젝트의 기본 폴더로 이동합니다. | — |

## `memory` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `memory` | 세션 간에도 유지되는 영구 메모리에 중요한 정보를 저장합니다. 저장한 메모리는 세션 시작 시 시스템 프롬프트에 표시되며 사용자의 환경과 사용자를 기억하는 방법입니다. 사용 시… | — |

## `session_search` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `session_search` | 로컬 세션 DB에 저장된 과거 세션을 검색하거나 하나의 세션 안에서 스크롤합니다. FTS5 기반 검색으로 실제 DB의 메시지를 반환하며 LLM 호출은 없습니다. 네 가지 형태가 있습니다: 검색(`query` 전달), 스크롤(`session_id` + `around_message_id`), 읽기(`session_id`만 전달), 탐색(인자 없음). | — |

## `skills` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `skill_manage` | 스킬을 관리(생성, 업데이트, 삭제)합니다. 스킬은 반복 작업을 위한 재사용 가능한 접근 방식인 절차적 메모리입니다. 새 스킬은 ~/.hermes/skills/에 생성되며 기존 스킬은 어디에 있든 수정할 수 있습니다. 작업: create(전체 SKILL.m… | — |
| `skill_view` | 스킬을 사용하면 특정 작업과 워크플로에 대한 정보 및 연결된 파일을 불러올 수 있습니다. 첫 호출은 SKILL.md 내용과… | — |
| `skills_list` | 사용 가능한 스킬(이름 + 설명)을 나열합니다. 전체 내용을 불러오려면 skill_view(name)를 사용하세요. | — |

## `terminal` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `process` | terminal(background=true)로 시작한 백그라운드 프로세스를 관리합니다. 작업: 'list'(모든 프로세스 표시), 'poll'(상태 + 새 출력 확인), 'log'(페이지 나누기가 있는 전체 출력), 'wait'(완료될 때까지 또는 시간 제한까지 대기), 'kill'(종료), 'write'(전… | — |
| `terminal` | Linux 환경에서 셸 명령을 실행합니다. 파일 시스템은 호출 간에 유지됩니다. 장시간 실행에는 `background=true`를 설정하세요. 완료 시 자동 알림을 받으려면 `background=true`와 함께 `notify_on_complete=true`를 설정하세요. `cat`/`head`/`tail`은 사용하지 말고 `read_file`을 사용하세요. `grep`/`rg`/`find`는 사용하지 말고 `search_files`를 사용하세요. | — |

## `desktop_ui` 도구 세트

Hermes 데스크톱 앱에서 시작된 세션인 경우, 연결된 백엔드가 로컬, SSH, URL 또는 Hermes Cloud 중 무엇이든 활성화됩니다. CLI, TUI, 메시징, cron 세션에는 없습니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `read_terminal` | Hermes 데스크톱 GUI의 인앱 터미널 창(이 채팅 옆에 포함된 셸)에 현재 표시된 내용을 읽습니다. | — |
| `close_terminal` | 백그라운드 프로세스의 읽기 전용 터미널 탭을 Hermes 데스크톱 GUI에서 닫습니다. 프로세스를 종료하지는 않으며 탭/보기를 제거할 뿐입니다 — 프로세스를 중지하려면 process(action='kill')을 사용하세요. | — |
| `open_preview` | Hermes 데스크톱 앱의 채팅 옆 미리보기 창에서 웹 URL, localhost 개발 서버 URL 또는 파일 경로를 엽니다. | — |
| `read_preview` | Hermes 데스크톱 앱의 미리보기 창에 현재 표시된 내용을 읽습니다 — 인앱 브라우저의 페이지 텍스트(URL + 제목 + 렌더링된 텍스트, `start`/`count`로 페이지 지정) 또는 파일/산출물 탭의 식별자입니다. | — |
| `read_window_below` | Hermes 데스크톱 창 바로 아래에 있는 OS 창을 식별합니다 — 앱 이름, 제목, 경계(메타데이터만, 픽셀은 절대 포함하지 않음)를 반환합니다. macOS에서는 화면 기록 권한이 이미 부여된 경우에만 다른 앱의 제목이 표시됩니다. 도구가 권한을 요청하지는 않습니다. | — |
| `focus_pane` | Hermes 데스크톱 앱의 창(채팅, 파일, 터미널, 리뷰, 세션)을 표시하고 포커스를 맞춥니다. | — |
| `react_to_message` | 단일 이모지로 메시지에 반응합니다(iMessage 탭백 방식). 설정 → 모양(`display.message_reactions`)에서 선택적으로 활성화할 수 있습니다. | — |

## `todo` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `todo` | 현재 세션의 작업 목록을 관리합니다. 매개변수 없이 호출하면 현재 목록을 읽습니다. 쓰기: 작업 목록을 생성/업데이트하려면 'todos' 배열을 제공합니다 — merge=… | — |

## `vision` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `vision_analyze` | AI vision으로 이미지를 분석합니다. vision을 지원하는 기본 모델에서는 원시 이미지 픽셀이 멀티모달 도구 결과로 직접 반환되어 다음 턴에 모델이 이를 볼 수 있습니다. 텍스트 전용 기본 모델에서는 보조 vision 모델로 대체되어 설명을 텍스트로 반환합니다. 어느 경우든 도구 시그니처는 동일합니다. | — |

## `video` 도구 세트

선택적 도구 세트입니다(기본 `hermes-cli` 세트에는 로드되지 않음). `--toolsets video`로 추가하거나 구성의 `toolsets:`에 `video`를 포함하세요.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `video_analyze` | URL 또는 파일 경로의 비디오를 분석합니다 — 자막, 장면 분석, 주요 타임스탬프, 시각적 설명을 제공합니다. | — |

## `video_gen` 도구 세트

선택적 도구 세트입니다(기본 `hermes-cli` 세트에는 로드되지 않음). `--toolsets video_gen`으로 추가하거나 `hermes tools` → Video Generation에서 활성화하면 백엔드 선택 과정도 안내합니다.

백엔드는 `plugins/video_gen/<name>/` 아래에 플러그인으로 제공됩니다:

- **xAI Grok-Imagine** — text-to-video와 image-to-video(SuperGrok OAuth 또는 `XAI_API_KEY`).
- **FAL.ai** — Veo 3.1, Pixverse v6, Kling O3(`FAL_KEY` 필요).

단일 `video_generate` 도구가 두 양식을 모두 처리합니다 — 정지 이미지를 애니메이션화하려면 `image_url`을 전달하고, 텍스트만으로 생성하려면 생략하세요. 활성 백엔드는 적절한 엔드포인트로 자동 라우팅합니다. 세션 시작 시 도구 설명은 활성 백엔드의 실제 기능(양식, 화면 비율, 해상도, 길이 범위, 참조 이미지 최대 수, 오디오 지원)을 반영하도록 다시 구성됩니다. 백엔드 작성 방법은 [비디오 생성 제공자 플러그인](/developer-guide/video-gen-provider-plugin)을 참조하세요.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `video_generate` | 텍스트 프롬프트로 비디오를 생성하거나(text-to-video), 정지 이미지를 애니메이션화합니다(image-to-video). 사용자가 구성한 비디오 생성 백엔드를 사용합니다. 이미지를 애니메이션화하려면 `image_url`을 전달하고, 텍스트만으로 생성하려면 생략하세요. 백엔드는 적절한 엔드포인트로 자동 라우팅합니다. `video` 필드에 HTTP URL 또는 절대 파일 경로를 반환합니다. | 활성 `video_gen` 플러그인 + 해당 자격 증명(예: `XAI_API_KEY`, `FAL_KEY`) |
| `xai_video_edit` | xAI Imagine으로 기존 비디오를 편집합니다. 제공자별 도구이며(`video_generate`와 별개), `video_url`은 이전 Imagine 결과의 공개 HTTPS MP4 URL이어야 합니다. | xAI Imagine 자격 증명(SuperGrok OAuth 또는 `XAI_API_KEY`) |
| `xai_video_extend` | xAI Imagine으로 기존 비디오를 연장합니다. 제공자별 도구이며(`video_generate`와 별개), `video_url`은 이전 Imagine 결과의 공개 HTTPS MP4 URL이어야 합니다. | xAI Imagine 자격 증명(SuperGrok OAuth 또는 `XAI_API_KEY`) |

## `web` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `web_search` | 웹에서 정보를 검색합니다. 기본적으로 제목, URL, 설명과 함께 최대 5개의 결과를 반환합니다. 선택적 `limit`(1-100, 기본값 5)을 허용합니다. 쿼리는 구성된 백엔드로 전달되므로 백엔드가 지원하는 경우 `site:domain`, `filetype:pdf`, `intitle:word`, `-term`, `"exact phrase"` 같은 연산자를 사용할 수 있습니다. | EXA_API_KEY 또는 PARALLEL_API_KEY 또는 FIRECRAWL_API_KEY 또는 TAVILY_API_KEY |
| `web_extract` | 웹 페이지 URL에서 내용을 추출합니다. LLM 요약 없이 정리된 페이지 콘텐츠를 markdown/text로 반환하므로 빠릅니다. PDF URL(arxiv 논문, 문서)에도 사용할 수 있으며 PDF 링크를 직접 전달하세요. 문자 예산(기본값 15000) 이내의 페이지는 전체를 반환하고, 더 큰 페이지는 전체 텍스트가 디스크에 저장된 위치를 가리키는 푸터와 함께 앞부분+뒷부분 창을 반환합니다. 호출당 최대 URL 5개입니다. | EXA_API_KEY 또는 PARALLEL_API_KEY 또는 FIRECRAWL_API_KEY 또는 TAVILY_API_KEY |

## `x_search` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `x_search` | xAI의 내장 `x_search` Responses 도구를 사용해 X(Twitter) 게시물, 프로필, 스레드를 검색합니다. 공개 X에서 현재 논의, 반응 또는 주장에 대한 읽기 전용 탐색이며, 일반 웹 페이지용이 아닙니다. 게시, 답글, 좋아요, DM, 미디어 업로드, 삭제 또는 인증된 X 계정 검사는 수행하지 않습니다 — 이러한 작업에는 별도의 인증된 X API 표면(예: `xurl` 스킬)이 필요합니다. 기본적으로 꺼져 있으며 `hermes tools` → 🐦 X (Twitter) Search에서 선택적으로 활성화할 수 있습니다. xAI 자격 증명이 구성된 경우에만(check_fn으로 게이트됨) 스키마가 등록됩니다. | XAI_API_KEY **또는** xAI OAuth(SuperGrok / Premium+) 로그인 |

## `tts` 도구 세트

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `text_to_speech` | 텍스트를 음성 오디오로 변환합니다. 플랫폼이 음성 메시지로 전달하는 MEDIA: 경로를 반환합니다. Telegram에서는 음성 말풍선으로 재생되고 Discord/WhatsApp에서는 오디오 첨부 파일로 전달됩니다. CLI 모드에서는 ~/voice-memos/에 저장합니다. 음성 및 제공자… | — |

## `discord` 도구 세트

`hermes-discord` 플랫폼 도구 세트에 등록됩니다(게이트웨이 전용). 메시징 어댑터와 동일한 봇 토큰을 사용합니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `discord` | Discord 서버를 읽고 참여합니다. 작업에는 `search_members`, `fetch_messages`, `send_message`, `react`, `fetch_channel`, `list_channels` 등이 포함됩니다. | `DISCORD_BOT_TOKEN` |

## `discord_admin` 도구 세트

`hermes-discord` 플랫폼 도구 세트에 등록됩니다. 중재 작업에는 봇이 일치하는 Discord 권한을 보유해야 합니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `discord_admin` | REST API를 통해 Discord 서버를 관리합니다: 길드/채널/역할 나열, 채널 생성/편집/삭제, 역할 부여 관리, 타임아웃, 킥, 밴을 수행합니다. | `DISCORD_BOT_TOKEN` + 봇 권한 |

## `spotify` 도구 세트

번들로 제공되는 `spotify` 플러그인이 등록합니다. OAuth 토큰이 필요합니다 — 한 번 인증하려면 `hermes auth spotify`를 실행하세요.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `spotify_playback` | Spotify 재생을 제어하고, 현재 재생 상태를 확인하거나, 최근 재생한 트랙을 가져옵니다. | Spotify OAuth |
| `spotify_devices` | Spotify Connect 장치를 나열하거나 재생을 다른 장치로 전송합니다. | Spotify OAuth |
| `spotify_queue` | 사용자의 Spotify 대기열을 확인하거나 항목을 추가합니다. | Spotify OAuth |
| `spotify_search` | 트랙, 앨범, 아티스트, 재생 목록, 쇼 또는 에피소드의 Spotify 카탈로그를 검색합니다. | Spotify OAuth |
| `spotify_playlists` | Spotify 재생 목록을 나열, 확인, 생성, 업데이트 및 수정합니다. | Spotify OAuth |
| `spotify_albums` | Spotify 앨범 메타데이터 또는 앨범 트랙을 가져옵니다. | Spotify OAuth |
| `spotify_library` | 사용자가 저장한 Spotify 트랙 또는 앨범을 나열, 저장 또는 삭제합니다. | Spotify OAuth |

## `hermes-yuanbao` 도구 세트

`hermes-yuanbao` 플랫폼 도구 세트에만 등록됩니다. Yuanbao는 Tencent의 채팅 앱이며, 이 도구는 DM/그룹/스티커 API를 제어합니다.

| 도구 | 설명 | 필요한 환경 |
|------|-------------|------|
| `yb_query_group_info` | 그룹의 기본 정보(앱에서 "派/Pai"라고 부름)를 조회합니다: 이름, 소유자, 구성원 수입니다. | Yuanbao 자격 증명 |
| `yb_query_group_members` | 그룹 구성원을 조회합니다(`@` 멘션, 이름으로 사용자 찾기, 봇 목록에 사용). | Yuanbao 자격 증명 |
| `yb_send_dm` | 선택적 미디어 파일과 함께 그룹 내 사용자에게 비공개/직접 메시지를 보냅니다. | Yuanbao 자격 증명 |
| `yb_search_sticker` | 키워드로 기본 제공 Yuanbao 스티커(TIM 얼굴) 카탈로그를 검색합니다. | Yuanbao 자격 증명 |
| `yb_send_sticker` | 현재 Yuanbao 채팅에 기본 제공 스티커를 보냅니다. | Yuanbao 자격 증명 |
