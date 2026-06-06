---
sidebar_position: 3
title: "내장 도구 레퍼런스"
description: "도구 세트별로 그룹화된 Hermes 내장 도구에 대한 공식 레퍼런스"
---

# 내장 도구 레퍼런스 (Built-in Tools Reference)

이 페이지는 도구 세트별로 그룹화된 Hermes의 내장 도구를 설명합니다. 사용 가능 여부는 플랫폼, 자격 증명(credentials), 활성화된 도구 세트에 따라 다릅니다.

**빠른 카운트 (현재 레지스트리):** 약 64개 도구 — 10개의 브라우저 도구 (핵심) + 2개의 CDP 제한 브라우저 도구, 4개의 파일 도구, 4개의 Home Assistant 도구, 2개의 터미널 도구, 2개의 웹 도구, 5개의 Feishu 도구, 7개의 Spotify 도구 (내장된 `spotify` 플러그인에 의해 등록됨), 5개의 Yuanbao 도구, 9개의 칸반 도구 (칸반 디스패처가 에이전트를 생성할 때 등록됨), 2개의 Discord 도구, 그리고 다수의 독립 실행형 도구 (`memory`, `clarify`, `delegate_task`, `execute_code`, `cronjob`, `session_search`, `skill_view`/`skill_manage`/`skills_list`, `text_to_speech`, `image_generate`, `video_generate`, `vision_analyze`, `video_analyze`, `mixture_of_agents`, `send_message`, `todo`, `computer_use`, `process`).

:::tip MCP 도구
내장 도구 외에도 Hermes는 MCP 서버에서 동적으로 도구를 로드할 수 있습니다. MCP 도구는 `mcp_<server>_` 접두사가 붙어 나타납니다 (예: `github` MCP 서버의 경우 `mcp_github_create_issue`). 구성 방법은 [MCP 통합](/user-guide/features/mcp)을 참조하세요.
:::

## `browser` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `browser_back` | 브라우저 기록의 이전 페이지로 뒤로 탐색합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_click` | 스냅샷의 참조 ID(예: '@e5')로 식별된 요소를 클릭합니다. 참조 ID는 스냅샷 출력의 대괄호 안에 표시됩니다. 먼저 browser_navigate와 browser_snapshot을 호출해야 합니다. | — |
| `browser_console` | 현재 페이지의 브라우저 콘솔 출력 및 JavaScript 오류를 가져옵니다. console.log/warn/error/info 메시지와 포착되지 않은 JS 예외를 반환합니다. 이를 사용하여 조용한 JavaScript 오류, 실패한 API 호출 및 애플리케이션 경고를 감지할 수 있습니다. | — |
| `browser_get_images` | 현재 페이지의 모든 이미지 목록을 URL과 대체 텍스트(alt text)와 함께 가져옵니다. vision 도구로 분석할 이미지를 찾을 때 유용합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_navigate` | 브라우저에서 URL로 이동합니다. 세션을 초기화하고 페이지를 로드합니다. 다른 브라우저 도구를 사용하기 전에 호출해야 합니다. 단순한 정보 검색의 경우 web_search 또는 web_extract를 사용하는 것이 더 빠르고 저렴합니다. | — |
| `browser_press` | 키보드 키를 누릅니다. 폼 제출(Enter), 탐색(Tab) 또는 키보드 단축키에 유용합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_scroll` | 특정 방향으로 페이지를 스크롤합니다. 현재 뷰포트의 아래나 위에 있을 수 있는 내용을 드러내기 위해 사용합니다. 먼저 browser_navigate를 호출해야 합니다. | — |
| `browser_snapshot` | 현재 페이지의 접근성 트리에 대한 텍스트 기반 스냅샷을 가져옵니다. browser_click 및 browser_type을 위한 참조 ID(@e1, @e2 등)가 포함된 대화형 요소를 반환합니다. full=false (기본값): 대화형 요소만 있는 간결한 보기. full=true: 컴팩트한 보기... | — |
| `browser_type` | 참조 ID로 식별된 입력 필드에 텍스트를 입력합니다. 필드를 먼저 지운 다음 새 텍스트를 입력합니다. 먼저 browser_navigate와 browser_snapshot을 호출해야 합니다. | — |
| `browser_vision` | 현재 페이지의 스크린샷을 찍고 AI 비전으로 분석합니다. 페이지에 무엇이 있는지 시각적으로 이해해야 할 때 사용합니다 - 특히 캡차(CAPTCHA), 시각적 확인 챌린지, 복잡한 레이아웃 등에 유용합니다. | — |

## `browser` 도구 세트 (CDP 제한 도구)

이 두 도구는 `browser` 도구 세트에 속하지만, `/browser connect`, `browser.cdp_url` 설정, Browserbase 세션 또는 Camofox를 통해 세션 시작 시 Chrome DevTools Protocol 엔드포인트에 도달할 수 있을 때만 등록됩니다.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `browser_cdp` | 원시 Chrome DevTools Protocol 명령을 보냅니다. 고수준 `browser_*` 도구에서 지원하지 않는 브라우저 작업을 위한 비상구(Escape hatch)입니다. 참조: https://chromedevtools.github.io/devtools-protocol/ | CDP 엔드포인트 |
| `browser_dialog` | 기본 JavaScript 대화 상자 (alert / confirm / prompt / beforeunload)에 응답합니다. 먼저 `browser_snapshot`을 호출하면 대기 중인 대화 상자가 `pending_dialogs` 필드에 나타납니다. 그 후 `browser_dialog(action='accept'\|'dismiss')`를 호출하세요. | CDP 엔드포인트 |

## `clarify` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `clarify` | 진행하기 전에 명확성, 피드백 또는 결정이 필요할 때 사용자에게 질문합니다. 두 가지 모드를 지원합니다: 1. **객관식** — 최대 4개의 선택지를 제공합니다. 사용자는 하나를 선택하거나 5번째 '기타' 옵션을 통해 직접 답을 입력할 수 있습니다. 2. **주관식**... | — |

## `code_execution` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `execute_code` | Hermes 도구를 프로그래밍 방식으로 호출할 수 있는 Python 스크립트를 실행합니다. 3개 이상의 도구 호출과 그 사이의 처리 로직이 필요할 때, 대량의 도구 출력을 컨텍스트에 넣기 전에 필터링/축소해야 할 때, 조건 분기가 필요할 때 사용합니다. | — |

## `cronjob` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `cronjob` | 통합된 예약 작업 관리자입니다. `action="create"`, `"list"`, `"update"`, `"pause"`, `"resume"`, `"run"` 또는 `"remove"`를 사용하여 작업을 관리합니다. 하나 이상의 기술이 첨부된 기술 기반(skill-backed) 작업을 지원하며, 업데이트 시 `skills=[]`를 설정하면 첨부된 기술이 지워집니다. 크론 실행은 현재 채팅 컨텍스트가 없는 새로운 세션에서 수행됩니다. | — |

## `delegation` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `delegate_task` | 격리된 컨텍스트에서 작업을 수행할 하나 이상의 하위 에이전트(subagent)를 생성합니다. 각 하위 에이전트는 고유한 대화, 터미널 세션 및 도구 세트를 갖습니다. 최종 요약만 반환되며, 중간 도구 결과는 결코 컨텍스트 창에 입력되지 않습니다. | — |

## `feishu_doc` 도구 세트

Feishu 문서 댓글 지능형 답변 핸들러 (`gateway/platforms/feishu_comment.py`) 범위에 속합니다. `hermes-cli` 또는 일반 Feishu 채팅 어댑터에는 노출되지 않습니다.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `feishu_doc_read` | 파일 유형(file_type) 및 토큰이 주어진 Feishu/Lark 문서(Docx, Doc 또는 Sheet)의 전체 텍스트 내용을 읽습니다. | Feishu 앱 자격 증명 |

## `feishu_drive` 도구 세트

Feishu 문서 댓글 핸들러 범위에 속합니다. 드라이브 파일에 대한 댓글 읽기/쓰기 작업을 주도합니다.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `feishu_drive_add_comment` | Feishu/Lark 문서 또는 파일에 최상위 댓글을 추가합니다. | Feishu 앱 자격 증명 |
| `feishu_drive_list_comments` | Feishu/Lark 파일의 전체 문서 댓글을 가장 최근 항목부터 나열합니다. | Feishu 앱 자격 증명 |
| `feishu_drive_list_comment_replies` | 특정 Feishu 댓글 스레드(전체 문서 또는 로컬 선택)에 대한 답글을 나열합니다. | Feishu 앱 자격 증명 |
| `feishu_drive_reply_comment` | 선택적인 `@` 멘션과 함께 Feishu 댓글 스레드에 답글을 게시합니다. | Feishu 앱 자격 증명 |

## `file` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `patch` | 파일의 특정 부분을 찾아 수정(find-and-replace)합니다. 터미널의 sed/awk 대신 이것을 사용하세요. 퍼지 매칭(9가지 전략)을 사용하므로 약간의 공백/들여쓰기 차이로 인해 중단되지 않습니다. 통합된 비교(unified diff)를 반환합니다. | — |
| `read_file` | 줄 번호 및 페이지 지정 기능과 함께 텍스트 파일을 읽습니다. 터미널의 cat/head/tail 대신 사용하세요. 출력 형식: 'LINE_NUM\|CONTENT'. 파일을 찾을 수 없는 경우 비슷한 파일명을 제안합니다. 대용량 파일의 경우 offset과 limit을 사용하세요. 참고: 이미지는 읽을 수 없습니다. | — |
| `search_files` | 파일 내용을 검색하거나 이름으로 파일을 찾습니다. 터미널의 grep/rg/find/ls 대신 사용하세요. Ripgrep 기반으로, 셸 도구보다 빠릅니다. 내용 검색(target='content'): 파일 내부에서 정규식 검색을 수행합니다. | — |
| `write_file` | 파일에 내용을 작성하여 기존 내용을 완전히 대체합니다. 터미널의 echo/cat heredoc 대신 사용하세요. 상위 디렉터리를 자동으로 생성합니다. 전체 파일을 덮어쓰므로 대상 편집에는 'patch'를 사용하세요. | — |

## `homeassistant` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `ha_call_service` | 기기를 제어하기 위해 Home Assistant 서비스를 호출합니다. 사용 가능한 서비스 및 각 도메인의 매개변수를 찾으려면 ha_list_services를 사용하세요. | — |
| `ha_get_state` | 속성(밝기, 색상, 온도 설정값, 센서 판독값 등)을 포함하여 단일 Home Assistant 엔티티의 상세 상태를 가져옵니다. | — |
| `ha_list_entities` | Home Assistant 엔티티를 나열합니다. 도메인(light, switch, climate, sensor, binary_sensor, cover, fan 등) 또는 영역 이름(거실, 주방, 침실 등)으로 선택적으로 필터링할 수 있습니다. | — |
| `ha_list_services` | 기기 제어에 사용할 수 있는 Home Assistant 서비스(액션)를 나열합니다. 각 기기 유형에서 수행할 수 있는 액션 및 허용되는 매개변수를 보여줍니다. ha_list_entities를 통해 찾은 기기를 제어하는 방법을 찾으려면 이 도구를 사용하세요. | — |

## `computer_use` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `computer_use` | cua-driver를 통한 백그라운드 macOS 데스크톱 제어 — 스크린샷 (SOM / vision / AX), 클릭 / 드래그 / 스크롤 / 타이핑 / 키 / 대기, 앱 목록, 앱 포커스. 사용자의 커서나 키보드 포커스를 빼앗지 않습니다. 도구를 사용할 수 있는 모든 모델과 작동합니다. macOS 전용입니다. | `$PATH`의 `cua-driver` (`hermes tools`를 통해 설치). |

:::note
**Honcho 도구** (`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_reasoning`, `honcho_conclude`)는 더 이상 기본 제공되지 않습니다. `plugins/memory/honcho/`의 Honcho 메모리 제공자 플러그인을 통해 사용할 수 있습니다. 설치 및 사용 방법은 [메모리 제공자](../user-guide/features/memory-providers.md)를 참조하세요.
:::

## `image_gen` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `image_generate` | FAL.ai를 사용하여 텍스트 프롬프트에서 고품질 이미지를 생성합니다. 기본 모델은 사용자 구성 (기본값: FLUX 2 Klein 9B, 1초 미만 생성)이며 에이전트가 선택할 수 없습니다. 단일 이미지 URL을 반환합니다. | FAL_KEY |

## `kanban` 도구 세트

에이전트가 (a) 칸반 디스패처(`HERMES_KANBAN_TASK` 환경 변수 설정)에 의해 생성되거나 (b) `kanban` 도구 세트를 명시적으로 활성화하는 프로필에서 실행될 때 등록됩니다. 작업 범위의 워커(worker)는 할당된 작업에 대한 생명주기 도구를 사용하며, 오케스트레이터(orchestrator) 프로필은 추가적으로 `kanban_list` 및 `kanban_unblock`과 같은 보드 라우팅 도구를 갖습니다. 전체 워크플로우는 [칸반 다중 에이전트](/user-guide/features/kanban)를 참조하세요.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `kanban_show` | 이 워커에 할당된 활성 칸반 작업(제목, 설명, 댓글, 종속성)을 표시합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_list` | 필터가 있는 보드 작업을 나열합니다. 오케스트레이터 전용으로, 디스패처가 생성한 작업 워커에게는 숨겨집니다. | `kanban` 도구 세트가 있는 프로필 |
| `kanban_complete` | 구조화된 핸드오프 페이로드(결과, 아티팩트, 후속 조치)와 함께 현재 작업을 완료로 표시합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_block` | 사용자에 대한 질문으로 인해 현재 작업을 차단합니다. 디스패처는 일시 중지하고 질문을 표면화한 후 사람이 답변하면 재개합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_heartbeat` | 장기 실행 작업 중에 진행 상황 하트비트를 보내어 디스패처가 워커가 아직 살아 있음을 알 수 있게 합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_comment` | 상태를 변경하지 않고 작업 스레드에 댓글을 추가합니다. 중간 발견 사항을 드러내는 데 유용합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_create` | 현재 작업에서 하위 작업을 생성합니다. 오케스트레이터와 후속 작업을 생성하는 워커에 의해 사용됩니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_link` | 부모 → 자식 종속성 에지로 작업들을 연결합니다. | `HERMES_KANBAN_TASK` 또는 `kanban` 도구 세트 |
| `kanban_unblock` | 차단된 작업을 `ready` 상태로 되돌립니다. 오케스트레이터 전용으로, 디스패처가 생성한 작업 워커에게는 숨겨집니다. | `kanban` 도구 세트가 있는 프로필 |

## `memory` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `memory` | 세션 간에 지속되는 영구 메모리에 중요한 정보를 저장합니다. 메모리는 세션 시작 시 시스템 프롬프트에 나타납니다 — 이를 통해 사용자 및 환경에 대한 내용을 대화 간에 기억합니다. | — |

## `messaging` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `send_message` | 연결된 메시징 플랫폼으로 메시지를 보내거나 사용 가능한 대상을 나열합니다. 중요: 사용자가 특정 채널이나 사람에게 보내라고 요청할 때, 사용 가능한 대상을 보려면 먼저 send_message(action='list')를 호출해야 합니다. | — |

## `moa` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `mixture_of_agents` | 다수의 프론티어 LLM을 통해 어려운 문제를 공동으로 해결합니다. 5개의 API 호출(4개의 참조 모델 + 1개의 집계기)을 수행하여 최대한의 추론 노력을 기울입니다. 진정으로 어려운 문제에만 드물게 사용하세요. 복잡한 수학, 고급 알고리즘 등에 가장 적합합니다. | OPENROUTER_API_KEY |

## `session_search` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `session_search` | 로컬 세션 DB에 저장된 과거 세션을 검색하거나 하나 안에서 스크롤합니다. FTS5 기반 검색; (LLM 호출 없이) DB에서 실제 메시지를 반환합니다. 세 가지 형태: 발견(`query` 전달), 스크롤(`session_id` + `around_message_id` 전달), 찾아보기(인수 없음). | — |

## `skills` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `skill_manage` | 기술(skill)을 관리합니다(생성, 업데이트, 삭제). 기술은 반복적인 작업 유형에 대한 재사용 가능한 접근 방식인 에이전트의 절차적 기억입니다. 새 기술은 ~/.hermes/skills/로 이동하며, 기존 기술은 어디에 있든 수정할 수 있습니다. | — |
| `skill_view` | 기술을 통해 특정 작업 및 워크플로우는 물론 스크립트와 템플릿에 대한 정보를 로드할 수 있습니다. 기술의 전체 내용을 로드하거나 연결된 파일(참조, 템플릿, 스크립트)에 액세스합니다. 첫 번째 호출은 SKILL.md 내용과 함께 추가 파일 목록을 반환합니다. | — |
| `skills_list` | 사용 가능한 기술 목록(이름 + 설명)을 나열합니다. 전체 내용을 로드하려면 skill_view(name)를 사용하세요. | — |

## `terminal` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `process` | terminal(background=true)로 시작된 백그라운드 프로세스를 관리합니다. 액션: 'list' (모두 표시), 'poll' (상태 + 새 출력 확인), 'log' (페이지 매김이 포함된 전체 출력), 'wait' (완료되거나 시간 초과될 때까지 차단), 'kill' (종료), 'write' (전송). | — |
| `terminal` | Linux 환경에서 셸 명령을 실행합니다. 호출 간에 파일 시스템이 유지됩니다. 장기 실행 서버의 경우 `background=true`로 설정하세요. 프로세스가 끝나면 자동으로 알림을 받으려면 `notify_on_complete=true`로 설정하세요. | — |

## `todo` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `todo` | 현재 세션에 대한 작업 목록을 관리합니다. 3단계 이상의 복잡한 작업이나 사용자가 여러 작업을 제공할 때 사용합니다. 현재 목록을 읽으려면 매개변수 없이 호출하세요. 쓰기: - 항목을 생성/업데이트하려면 'todos' 배열을 제공 - merge=... | — |

## `vision` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `vision_analyze` | AI 비전을 사용하여 이미지를 분석합니다. 비전 기능이 있는 주 모델의 경우 원본 이미지 픽셀을 멀티모달 도구 결과로 반환하여 다음 턴에 모델이 기본적으로 볼 수 있게 합니다. 텍스트 전용 주 모델의 경우 이미지를 설명하고 텍스트로 설명을 반환하는 보조 비전 모델로 폴백(fallback)합니다. 도구 서명은 두 경우 모두 동일합니다. | — |

## `video` 도구 세트

선택적 도구 세트입니다 (기본 `hermes-cli` 세트에 로드되지 않음). `--toolsets video`를 통해 추가하거나 `toolsets:` 구성에 `video`를 포함하세요.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `video_analyze` | URL 또는 파일 경로의 비디오 콘텐츠 — 캡션, 장면 분류, 주요 타임스탬프 및 시각적 설명을 분석합니다. | — |

## `video_gen` 도구 세트

선택적 도구 세트입니다 (기본 `hermes-cli` 세트에 로드되지 않음). `--toolsets video_gen`을 통해 추가하거나 백엔드 선택 과정을 안내하는 `hermes tools` → Video Generation에서 활성화하세요.

백엔드는 `plugins/video_gen/<name>/` 아래에 플러그인으로 제공됩니다:

- **xAI Grok-Imagine** — 텍스트-비디오 및 이미지-비디오 생성 (SuperGrok OAuth 또는 `XAI_API_KEY`).
- **FAL.ai** — Veo 3.1, Pixverse v6, Kling O3 (`FAL_KEY` 필요).

단일 `video_generate` 도구는 두 가지 모달리티를 모두 다룹니다 — 스틸 이미지를 애니메이션화하려면 `image_url`을 전달하고, 텍스트에서만 생성하려면 이를 생략하세요. 활성 백엔드가 적절한 엔드포인트로 자동 라우팅합니다. 도구의 설명은 세션 시작 시 활성 백엔드의 실제 기능(모달리티, 종횡비, 해상도, 지속 시간 범위, 최대 참조 이미지, 오디오 지원)을 반영하여 재구성됩니다. 백엔드 작성에 대해서는 [비디오 생성 제공자 플러그인](/developer-guide/video-gen-provider-plugin)을 참조하세요.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `video_generate` | 사용자가 구성한 비디오 생성 백엔드를 사용하여 텍스트 프롬프트에서 비디오를 생성하거나(text-to-video) 스틸 이미지를 애니메이션화합니다(image-to-video). 이미지를 애니메이션화하려면 `image_url`을 전달하고, 텍스트에서만 생성하려면 생략하세요. 백엔드가 올바른 엔드포인트로 자동 라우팅합니다. `video` 필드에 HTTP URL 또는 절대 파일 경로를 반환합니다. | 활성 `video_gen` 플러그인 + 해당 자격 증명(예: `XAI_API_KEY`, `FAL_KEY`) |

## `web` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `web_search` | 정보를 찾기 위해 웹을 검색합니다. 기본적으로 제목, URL 및 설명이 포함된 최대 5개의 결과를 반환합니다. 선택적 `limit`을 받습니다(1-100, 기본값 5). 쿼리는 구성된 백엔드로 통과되므로 백엔드가 지원하는 경우 `site:domain`, `filetype:pdf`, `intitle:word`, `-term`, 및 `"exact phrase"`와 같은 연산자가 작동할 수 있습니다. | EXA_API_KEY 또는 PARALLEL_API_KEY 또는 FIRECRAWL_API_KEY 또는 TAVILY_API_KEY |
| `web_extract` | 웹 페이지 URL에서 내용을 추출합니다. 페이지 내용을 마크다운 형식으로 반환합니다. PDF URL에서도 작동합니다 — PDF 링크를 직접 전달하면 마크다운 텍스트로 변환됩니다. 5000자 미만의 페이지는 전체 마크다운을 반환하고, 더 큰 페이지는 LLM에 의해 요약됩니다. | EXA_API_KEY 또는 PARALLEL_API_KEY 또는 FIRECRAWL_API_KEY 또는 TAVILY_API_KEY |

## `x_search` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `x_search` | xAI의 내장 `x_search` Responses 도구를 사용하여 X(Twitter) 게시물, 프로필 및 스레드를 검색합니다. 일반 웹 페이지가 아닌 X의 현재 토론, 반응 또는 주장을 파악할 때 사용하세요. 기본적으로 꺼져 있으며, `hermes tools` → 🐦 X (Twitter) Search를 통해 활성화할 수 있습니다. 스키마는 xAI 자격 증명이 구성된 경우에만 등록됩니다(check_fn 제한). | XAI_API_KEY **또는** xAI Grok OAuth (SuperGrok / Premium+) 로그인 |

## `tts` 도구 세트

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `text_to_speech` | 텍스트를 음성 오디오로 변환합니다. 플랫폼에서 음성 메시지로 전달하는 MEDIA: 경로를 반환합니다. Telegram에서는 음성 버블로 재생되고, Discord/WhatsApp에서는 오디오 첨부 파일로 재생됩니다. CLI 모드에서는 ~/voice-memos/에 저장됩니다. | — |

## `discord` 도구 세트

`hermes-discord` 플랫폼 도구 세트에 등록됩니다(게이트웨이 전용). 메시징 어댑터와 동일한 봇 토큰을 사용합니다.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `discord` | Discord 서버를 읽고 참여합니다. 액션에는 `search_members`, `fetch_messages`, `send_message`, `react`, `fetch_channel`, `list_channels` 등이 있습니다. | `DISCORD_BOT_TOKEN` |

## `discord_admin` 도구 세트

`hermes-discord` 플랫폼 도구 세트에 등록됩니다. 조정(Moderation) 액션을 수행하려면 봇이 일치하는 Discord 권한을 가지고 있어야 합니다.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `discord_admin` | REST API를 통해 Discord 서버를 관리합니다: 길드/채널/역할 나열, 채널 생성/편집/삭제, 역할 부여 관리, 타임아웃, 추방 및 차단. | `DISCORD_BOT_TOKEN` + 봇 권한 |

## `spotify` 도구 세트

내장된 `spotify` 플러그인에 의해 등록됩니다. OAuth 토큰이 필요합니다 — 권한을 부여하려면 `hermes spotify setup`을 한 번 실행하세요.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `spotify_playback` | Spotify 재생을 제어하거나, 활성 재생 상태를 검사하거나, 최근에 재생된 트랙을 가져옵니다. | Spotify OAuth |
| `spotify_devices` | Spotify Connect 기기를 나열하거나 다른 기기로 재생을 전송합니다. | Spotify OAuth |
| `spotify_queue` | 사용자의 Spotify 대기열을 검사하거나 대기열에 항목을 추가합니다. | Spotify OAuth |
| `spotify_search` | Spotify 카탈로그에서 트랙, 앨범, 아티스트, 재생 목록, 프로그램 또는 에피소드를 검색합니다. | Spotify OAuth |
| `spotify_playlists` | Spotify 재생 목록을 나열, 검사, 생성, 업데이트 및 수정합니다. | Spotify OAuth |
| `spotify_albums` | Spotify 앨범 메타데이터 또는 앨범 트랙을 가져옵니다. | Spotify OAuth |
| `spotify_library` | 사용자의 저장된 Spotify 트랙이나 앨범을 나열, 저장 또는 제거합니다. | Spotify OAuth |

## `hermes-yuanbao` 도구 세트

`hermes-yuanbao` 플랫폼 도구 세트에만 등록됩니다. Yuanbao는 텐센트의 채팅 앱이며, 이러한 도구는 DM/그룹/스티커 API를 구동합니다.

| 도구 | 설명 | 필요 환경 |
|------|-------------|----------------------|
| `yb_query_group_info` | 그룹(앱에서는 "파(派)/Pai"로 불림)에 대한 기본 정보: 이름, 소유자, 구성원 수를 쿼리합니다. | Yuanbao 자격 증명 |
| `yb_query_group_members` | 그룹의 구성원을 쿼리합니다(`@` 멘션, 이름으로 사용자 찾기, 봇 목록 조회 등에 사용). | Yuanbao 자격 증명 |
| `yb_send_dm` | 그룹 내 사용자에게 미디어 파일(선택 사항)과 함께 비공개/다이렉트 메시지를 보냅니다. | Yuanbao 자격 증명 |
| `yb_search_sticker` | 키워드로 내장 Yuanbao 스티커(TIM face) 카탈로그를 검색합니다. | Yuanbao 자격 증명 |
| `yb_send_sticker` | 내장 스티커를 현재 Yuanbao 채팅으로 보냅니다. | Yuanbao 자격 증명 |
