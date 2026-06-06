---
title: Codex App-Server Runtime (선택 사항)
sidebar_label: Codex App-Server Runtime
---

# Codex App-Server Runtime

Hermes는 선택적으로 자체 도구 루프를 실행하는 대신 `openai/*` 및 `openai-codex/*` 턴을 [Codex CLI app-server](https://github.com/openai/codex)에 넘길 수 있습니다. 활성화되면 터미널 명령어, 파일 편집, 샌드박싱 및 MCP 도구 호출이 모두 Codex의 런타임 내부에서 실행되며, Hermes는 그 주변의 셸이 됩니다(세션 DB, 슬래시 명령어, 게이트웨이, 메모리 및 스킬 리뷰).

이것은 **선택 사항(opt-in) 전용**입니다. 플래그를 변경하지 않는 한 기본 Hermes 동작은 변하지 않습니다. Hermes는 절대로 당신을 이 런타임으로 자동 라우팅하지 않습니다.

:::tip
OpenAI Codex를 사용하지 않으시나요? `hermes setup --portal`은 단 한 단계로 Claude/Gemini 등과 함께 비(非) Codex 백엔드를 구성합니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 왜 사용하는가 (Why)

- Codex CLI가 사용하는 것과 동일한 인증 흐름을 사용하여 API 키 없이 **ChatGPT 구독**을 통해 OpenAI 에이전트 턴을 실행합니다.
- **Codex 자체 도구 모음과 샌드박스**를 사용합니다 — 터미널/읽기/쓰기/검색을 위한 `shell`, 구조화된 편집을 위한 `apply_patch`, 계획을 위한 `update_plan` 등이 모두 seatbelt/landlock 샌드박싱 내에서 실행됩니다.
- `codex plugin`을 통해 설치된 **네이티브 Codex 플러그인**(Linear, GitHub, Gmail, Calendar, Canva 등)이 Hermes 세션으로 자동 마이그레이션되어 활성화됩니다.
- **Hermes의 더 풍부한 도구들이 함께 제공됩니다** — web_search, web_extract, 브라우저 자동화, 비전, 이미지 생성, 스킬 및 TTS가 MCP 콜백을 통해 작동합니다. Codex는 내장되어 있지 않은 도구를 위해 Hermes로 다시 콜백합니다.
- **메모리와 스킬 넛지(nudge)가 계속 작동합니다** — Codex의 이벤트가 Hermes의 메시지 형태로 투영되므로 자기 개선 루프는 일반적인 형태의 트랜스크립트를 보게 됩니다.

## 모델이 실제로 가지는 도구들

이 부분은 대부분의 사용자가 가장 먼저 알고 싶어하는 내용입니다. 이 런타임이 켜지면 턴을 실행하는 모델은 세 가지 독립적인 도구 소스를 가집니다.

### 1. Codex의 내장 툴셋 (항상 켜짐)

이들은 Hermes의 개입, MCP 또는 플러그인 없이 `codex app-server` 자체에 탑재되어 제공됩니다. 런타임이 시작되는 즉시 5개 모두를 사용할 수 있습니다.

- **`shell`** — 샌드박스 내에서 임의의 쉘 명령을 실행합니다. 모델은 이를 통해 파일을 읽고(`cat`, `head`, `tail`), 쓰고(`echo > foo`, 히어독), 검색하고(`find`, `rg`, `grep`), 디렉토리를 이동하고(`ls`, `cd`), 빌드를 실행하고, 프로세스를 관리하는 등 bash에서 할 수 있는 모든 작업을 수행합니다.
- **`apply_patch`** — Codex의 패치 형식으로 구조화된 다중 파일 차이점(diff)을 적용합니다. 모델은 간단하지 않은 코드 편집(함수 추가, 파일 간 리팩토링)에 이를 사용합니다. 일회성 쓰기에는 쉘 히어독(heredocs)을 여전히 사용할 수 있습니다.
- **`update_plan`** — codex의 내부 할 일 / 계획 추적기입니다. Hermes의 `todo` 도구에 해당하지만 전적으로 codex의 런타임 내부에서 관리됩니다.
- **`view_image`** — 모델이 이미지를 볼 수 있도록 로컬 이미지 파일을 대화에 로드합니다.
- **`web_search`** — 구성된 경우 codex는 자체 내장 웹 검색을 갖습니다. Hermes는 아래의 콜백을 통해 (Firecrawl 지원) `web_search`도 노출하며, 모델은 선호하는 것을 선택합니다.

따라서 터미널을 통해 **읽기/쓰기/검색/찾기/실행할 모든 작업을 codex가 네이티브하게 수행**합니다. 샌드박스 프로필(런타임을 활성화할 때 기본값은 `:workspace`)은 쓰기 권한이 있는 대상을 제어합니다.

### 2. 네이티브 Codex 플러그인 (`codex plugin` 설치에서 자동 마이그레이션됨)

런타임을 활성화하면 Hermes는 codex의 `plugin/list` RPC에 질의하고 설치한 모든 플러그인에 대해 `[plugins."<name>@openai-curated"]` 항목을 기록합니다. 플러그인 자체는 codex에 의해 관리되며 codex의 자체 UI를 통해 한 번 인증됩니다.

예시 (OpenClaw 스레드에서 "YouTube 비디오에 나올 만한" 것으로 강조된 것들):

- **Linear** — 이슈 찾기/업데이트
- **GitHub** — 코드 검색, PR 보기, 댓글 작성
- **Gmail** — 메일 읽기/보내기
- **Google Calendar** — 이벤트 생성/찾기
- **Outlook calendar/email** — Microsoft 커넥터를 통한 동일한 형태
- **Canva** — 디자인 생성
- ...`codex plugin marketplace add openai-curated` + `codex plugin install ...`을 통해 설치한 모든 것

마이그레이션되지 않는 것:
- 아직 설치하지 않은 플러그인 — 먼저 Codex에 설치하세요.
- ChatGPT 앱 마켓플레이스 항목(`app/list`) — 이는 계정 인증 덕분에 이미 codex 내부에서 활성화되어 있습니다.

### 3. Hermes 도구 콜백 (`~/.codex/config.toml`에 등록된 MCP 서버)

Hermes는 codex가 함께 제공되지 않는 도구에 대해 콜백할 수 있도록 자체를 MCP 서버로 등록합니다. 콜백을 통해 다음을 사용할 수 있습니다.

- **`web_search`** / **`web_extract`** — Firecrawl 기반. 구조화된 콘텐츠를 스크랩하는 것보다 더 깨끗한 결과를 제공하는 경향이 있습니다.
- **`browser_navigate` / `browser_click` / `browser_type` / `browser_press` / `browser_snapshot` / `browser_scroll` / `browser_back` / `browser_get_images` / `browser_console` / `browser_vision`** — Camofox 또는 Browserbase를 통한 전체 브라우저 자동화.
- **`vision_analyze`** — 별도의 비전 모델을 호출하여 이미지를 검사합니다(이미지를 대화에 로드하는 codex의 `view_image`와 다름).
- **`image_generate`** — Hermes의 image_gen 플러그인 체인을 통한 이미지 생성.
- **`skill_view` / `skills_list`** — Hermes의 스킬 라이브러리에서 읽어옵니다.
- **`text_to_speech`** — 구성된 Hermes 공급자를 통한 TTS.

모델이 이 중 하나를 원할 때 codex는 stdio MCP를 통해 `hermes_tools_mcp_server` 서브프로세스를 생성하고, 호출은 `model_tools.handle_function_call()`(Hermes의 기본 런타임과 동일한 코드 경로)을 통해 디스패치되며, 결과는 다른 MCP 응답처럼 codex로 반환됩니다.

### 이 런타임에서 사용할 수 없는 것

다음 네 가지 Hermes 도구는 디스패치하기 위해 실행 중인 AIAgent 컨텍스트(루프 중간 상태)가 필요하며, 상태 없는 MCP 콜백은 이를 구동할 수 없습니다. 이 도구 중 하나가 필요하면 기본 런타임(`/codex-runtime auto`)으로 다시 전환하세요.

- **`delegate_task`** — 하위 에이전트 생성
- **`memory`** — Hermes의 영구 메모리 저장소
- **`session_search`** — 세션 간 검색
- **`todo`** — Hermes의 할 일 저장소 (codex의 `update_plan`은 런타임 내부에서 해당하는 도구임)

## 워크플로 기능 (`/goal`, kanban, cron)

### `/goal` (Ralph 루프)

**이 런타임에서 작동합니다.** 목표는 세션 ID를 키로 하여 `state_meta`에 지속되며, 연속 프롬프트는 `run_conversation()`을 통해 일반적인 사용자 메시지처럼 피드백되고 codex는 다음 턴을 네이티브하게 실행합니다. 목표 판정자(judge)는 런타임 활성화 여부와 관계없이 보조 클라이언트(config.yaml의 `auxiliary.goal_judge`를 통해 구성됨)를 통해 실행됩니다. 판정자의 "차단됨, 사용자 입력 필요" 판정은 codex가 승인 대기 중에 멈출 경우 벗어날 수 있는 깨끗한 탈출구입니다.

**알아두어야 할 한 가지:** 각 연속 프롬프트는 완전히 새로운 codex 턴이므로 codex가 명령어 승인 정책을 처음부터 다시 평가한다는 것을 의미합니다. 쓰기가 많은 장기 실행 목표를 수행하는 경우 단일 세션 내 작업에서 볼 수 있는 것보다 더 많은 승인 프롬프트가 나타날 것으로 예상해야 합니다. 단순한 작업 공간 쓰기에서 프롬프트가 필요하지 않도록 `default_permissions = ":workspace"`(런타임을 활성화할 때 Hermes가 자동으로 수행함)를 설정하세요.

### Kanban (다중 에이전트 워크트리 디스패치)

**이 런타임에서 작동하지만 한 가지 미묘한 종속성이 있습니다.** 칸반 디스패처는 각 워커를 사용자의 구성을 읽는 별도의 `hermes chat -q` 하위 프로세스로 생성합니다. 즉, `model.openai_runtime: codex_app_server`가 전역적으로 설정되어 있으면 워커도 codex 런타임에 올라옵니다.

codex 런타임 워커 내에서 작동하는 것:
- Codex의 전체 툴셋(shell, apply_patch, update_plan, view_image, web_search) — 워커는 실제 작업 업무를 네이티브하게 수행합니다.
- 마이그레이션된 codex 플러그인 — Linear, GitHub 등
- browser_*, vision, image_gen, skills, TTS를 위한 Hermes 도구 콜백

MCP 콜백이 노출하기 때문에 작동하는 것:
- **`kanban_complete` / `kanban_block` / `kanban_comment` / `kanban_heartbeat`** — 워커 핸드오프 도구. 이들은 환경 변수(디스패처가 설정함)에서 `HERMES_KANBAN_TASK`를 읽고, 액세스를 올바르게 제어하며, `HERMES_KANBAN_DB`에 의해 핀으로 고정된 보드별 SQLite DB에 씁니다. 콜백에 이들이 없으면 이 런타임의 워커는 자신의 작업을 수행할 수 있지만 보고할 수는 없으며 디스패처의 시간 초과 시점까지 멈춥니다.
- **`kanban_show` / `kanban_list`** — 워커가 자신의 컨텍스트를 확인할 수 있는 읽기 전용 보드 쿼리입니다.
- **`kanban_create` / `kanban_unblock` / `kanban_link`** — 오케스트레이터 전용 작업. 새 작업을 디스패치해야 하는 codex 런타임에서 실행되는 오케스트레이터 에이전트에 사용할 수 있습니다.

칸반 도구는 디스패처가 설정한 `HERMES_KANBAN_TASK` 환경 변수에 의해 게이트됩니다 — 이 변수는 codex 서브프로세스로 전파되고(codex가 환경을 상속함) 거기서 생성된 `hermes-tools` MCP 서버 서브프로세스로 전파됩니다. 따라서 도구는 올바른 작업 ID를 확인하고 올바르게 게이트합니다. Codex app-server 워커의 경우, Hermes는 `HERMES_KANBAN_TASK`가 있을 때 좁은 app-server 샌드박스 재정의도 전달합니다: `workspace-write` 샌드박싱을 유지하고, **보드 DB 디렉토리와 디스패처가 핀으로 고정한 모든 Kanban 경로**를 쓰기 가능한 추가 루트로 더하며(`HERMES_KANBAN_WORKSPACES_ROOT`, `HERMES_KANBAN_WORKSPACE`, 레거시 `HERMES_KANBAN_ROOT` — 중복 제거, DB 디렉토리가 먼저 옴), 네트워크는 기본적으로 비활성화된 상태로 유지합니다. 이렇게 하면 불안정한 `:danger-no-sandbox` 우회 방법을 피하면서도 `kanban_complete` / `kanban_block`이 보드 DB를 업데이트할 수 있고 **동시에** 워커가 DB 디렉토리 외부에 있는 작업 공간 마운트(예: 다른 드라이브의 `/media/.../kanban-workspaces/...` — [이슈 #27941](https://github.com/NousResearch/hermes-agent/issues/27941)) 아래에 보고서/아티팩트를 쓸 수 있습니다.

### Cron 작업

**특별히 테스트되지는 않았습니다.** Cron 작업은 CLI와 동일한 코드 경로인 `cronjob` → `AIAgent.run_conversation`을 통해 실행됩니다. cron 작업의 구성에 `openai_runtime: codex_app_server`가 있으면 codex에서 실행됩니다. 동일한 도구 가용성 규칙이 적용됩니다 — codex 내장 + 플러그인 + MCP 콜백은 작동하고, 에이전트 루프 도구(delegate_task, memory, session_search, todo)는 작동하지 않습니다. cron 작업이 이에 의존하는 경우 cron 범위를 기본 런타임을 사용하는 프로필로 제한하세요.

## 절충점 (Trade-offs)

| | Hermes 기본 런타임 | Codex app-server (선택 사항) |
|---|---|---|
| `delegate_task` 하위 에이전트 | 가능 | 사용할 수 없음 — 에이전트 루프 컨텍스트 필요 |
| `memory`, `session_search`, `todo` | 가능 | 사용할 수 없음 — 에이전트 루프 컨텍스트 필요 |
| `web_search`, `web_extract` | 가능 | 가능 (MCP 콜백을 통해) |
| 브라우저 자동화 (Camofox/Browserbase) | 가능 | 가능 (MCP 콜백을 통해) |
| `vision_analyze`, `image_generate` | 가능 | 가능 (MCP 콜백을 통해) |
| `skill_view`, `skills_list` | 가능 | 가능 (MCP 콜백을 통해) |
| `text_to_speech` | 가능 | 가능 (MCP 콜백을 통해) |
| Codex `shell` (터미널/읽기/쓰기/검색/찾기/실행) | — | 가능 (Codex 내장) |
| Codex `apply_patch` (구조화된 다중 파일 편집) | — | 가능 (Codex 내장) |
| Codex `update_plan` (런타임 내 할 일) | — | 가능 (Codex 내장) |
| Codex `view_image` (대화에 이미지 로드) | — | 가능 (Codex 내장) |
| Codex 샌드박스 (seatbelt/landlock, 프로필) | — | 가능 (Codex 내장) |
| ChatGPT 구독 인증 | — | 가능 (`openai-codex` 공급자를 통해) |
| 네이티브 Codex 플러그인 (Linear, GitHub 등) | — | 가능 (자동 마이그레이션됨) |
| 사용자 MCP 서버 | 가능 | 가능 (codex로 자동 마이그레이션됨) |
| 메모리 + 스킬 리뷰 (백그라운드) | 가능 | 가능 (항목 투영을 통해) |
| 다중 턴 대화 | 가능 | 가능 |
| `/goal` (Ralph 루프) | 가능 | 가능 |
| Kanban 워커 디스패치 | 가능 | 가능 (콜백을 통해) |
| Kanban 오케스트레이터 도구 | 가능 | 가능 (콜백을 통해) |
| 모든 게이트웨이 플랫폼 | 가능 | 가능 |
| 비(非) OpenAI 공급자 | 가능 | 해당 없음 — OpenAI/Codex 범위로 제한 |

## 전제 조건 (Prerequisites)

1. **Codex CLI 설치:**
   ```bash
   npm i -g @openai/codex
   codex --version   # 0.130.0 이상
   ```
2. **Codex OAuth 로그인.** codex 서브프로세스는 `~/.codex/auth.json`을 읽습니다. 이를 채우는 두 가지 방법:
   ```bash
   codex login                  # 토큰을 ~/.codex/auth.json에 씁니다.
   ```
   Hermes 자체의 `hermes auth login codex`는 `~/.hermes/auth.json`에 씁니다 — 이는 별도의 세션입니다. 아직 하지 않았다면 **별도로 `codex login`을 실행**하세요.

3. **(선택 사항) 원하는 Codex 플러그인 설치.** 런타임을 활성화하면 Hermes는 Codex CLI를 통해 이미 설치한 큐레이션된 플러그인을 자동 마이그레이션합니다:
   ```bash
   codex plugin marketplace add openai-curated
   # 그런 다음 codex의 TUI를 통해 Linear / GitHub / Gmail 등을 설치합니다.
   ```
   Hermes가 이를 발견하고 `~/.codex/config.toml`에 `[plugins."<name>@openai-curated"]` 항목을 자동으로 작성합니다.

## 활성화하기

Hermes 세션 내에서:

```
/codex-runtime codex_app_server
```

이 명령어는:
- `codex` CLI가 설치되어 있는지 확인합니다(설치되어 있지 않은 경우 설치 힌트와 함께 차단).
- `model.openai_runtime: codex_app_server`를 config.yaml에 유지(persist)합니다.
- 사용자 MCP 서버를 `~/.hermes/config.yaml`에서 `~/.codex/config.toml`로 마이그레이션합니다.
- Codex의 `plugin/list` RPC에 질의하여 **설치된 네이티브 Codex 플러그인을 발견하고 마이그레이션**합니다(Linear, GitHub, Gmail, Calendar, Canva 등).
- codex 서브프로세스가 codex에 포함되지 않은 도구에 대해 콜백할 수 있도록 **Hermes 자체 도구를 MCP 서버로 등록**합니다.
- **`default_permissions = ":workspace"`를 작성**하여 모든 작업마다 묻지 않고 샌드박스가 작업 공간 내에서 쓰기를 허용하도록 합니다.
- 무엇이 마이그레이션되었는지 알려줍니다. **다음** 세션에 적용됩니다 — 현재 캐시된 에이전트는 프롬프트 캐시를 유효하게 유지하기 위해 이전 런타임을 유지합니다.

동의어: `/codex-runtime on`, `/codex-runtime off`, `/codex-runtime auto`.

아무것도 변경하지 않고 현재 상태를 확인하려면:
```
/codex-runtime
```

`~/.hermes/config.yaml`에서 수동으로 설정할 수도 있습니다.
```yaml
model:
  openai_runtime: codex_app_server   # 기본값은 "auto" (= Hermes 런타임)
```

## 자기 개선 루프 (메모리 + 스킬 넛지)

Hermes의 백그라운드 자기 개선은 카운터 임계값에서 시작됩니다.

- 10번의 사용자 프롬프트마다 → 포크된 리뷰 에이전트가 대화를 검토하고 메모리에 저장할 내용이 있는지 결정합니다.
- 단일 턴 내에서 도구 반복이 10번 될 때마다 → 같은 개념이지만 스킬에 대한 것입니다 (`skill_manage` 쓰기).

**이 둘 다 codex 런타임에서 계속 작동합니다.** codex 경로는 완료된 각 `commandExecution` / `fileChange` / `mcpToolCall` / `dynamicToolCall` 항목을 합성 `assistant tool_call` + `tool` 결과 메시지로 투영합니다. 따라서 리뷰가 실행될 때쯤에는 기본 Hermes 런타임에서 보는 것과 동일한 모양을 보게 됩니다.

연결이 동등하게 유지되는 방법:

| | 기본 런타임 | Codex 런타임 |
|---|---|---|
| `_turns_since_memory` 증가 | 사용자 프롬프트당, run_conversation 루프 전에서 | 조기 반환(early-return) 전, 동일한 코드 경로 |
| `_iters_since_skill` 증가 | 채팅 완성 루프의 도구 반복당 | codex 턴이 반환된 후 `turn.tool_iterations`만큼 |
| 메모리 트리거 (`_turns_since_memory >= _memory_nudge_interval`) | 루프 전에서 계산되고, 응답 후 발생 | 루프 전에서 계산되고, codex 헬퍼로 전달됨 |
| 스킬 트리거 (`_iters_since_skill >= _skill_nudge_interval`) | 루프 후 계산됨 | codex 턴 후 계산됨 |
| `_spawn_background_review(messages_snapshot=..., review_memory=..., review_skills=...)` | 어느 트리거든 발생할 때 호출됨 | 어느 트리거든 발생할 때 동일하게 호출됨 |

한 가지 세부 사항: 리뷰 포크(fork) 자체는 Hermes의 에이전트 루프 도구(`memory`, `skill_manage`)를 호출해야 하며, 이는 Hermes 자체의 디스패치가 필요합니다. 따라서 부모 에이전트가 `codex_app_server`에 있을 때 리뷰 포크는 **`codex_responses`로 다운그레이드**됩니다 — 동일한 OAuth 자격 증명, 동일한 `openai-codex` 공급자이지만 OpenAI의 Responses API와 직접 통신하므로 Hermes가 루프를 소유하고 에이전트 루프 도구가 작동합니다. 이것은 사용자에게 보이지 않습니다.

결과적인 효과: codex 런타임을 활성화하면 메모리와 스킬 넛지가 평소와 똑같이 계속 발생합니다.

## 승인 방식 (How approvals work)

Codex는 명령을 실행하거나 패치를 적용하기 전에 승인을 요청합니다. 이들은 Hermes의 표준 "위험한 명령(Dangerous Command)" 프롬프트로 번역됩니다:

```
╭───────────────────────────────────────╮
│ Dangerous Command                     │
│                                       │
│ /bin/bash -lc 'echo hello > foo.txt'  │
│                                       │
│ ❯ 1. Allow once                       │
│   2. Allow for this session           │
│   3. Deny                             │
│                                       │
│ Codex requests exec in /your/cwd      │
╰───────────────────────────────────────╯
```

- **Allow once** → 이 단일 명령을 승인합니다.
- **Allow for this session** → Codex는 유사한 명령에 대해 다시 묻지 않습니다.
- **Deny** → 명령이 거부됩니다; Codex는 읽기 전용 모드로 계속 진행합니다.

`apply_patch`(파일 편집) 승인의 경우, codex가 해당하는 `fileChange` 항목을 통해 데이터를 제공하면 Hermes는 변경 사항의 요약(`1 add, 1 update: /tmp/new.py, /tmp/old.py`)을 표시합니다.

## 권한 프로필 (Permission profiles)

Codex에는 세 가지 내장 권한 프로필이 있습니다:
- `:read-only` — 쓰기 불가; 모든 쉘 명령은 승인이 필요합니다.
- `:workspace` — 현재 작업 공간 내의 쓰기는 묻지 않고 허용됩니다 (런타임을 활성화할 때 Hermes의 기본값)
- `:danger-no-sandbox` — 샌드박스 전혀 없음 (이해하지 못한다면 사용하지 마세요)

Hermes의 관리 블록 외부에 있는 `~/.codex/config.toml`에서 기본값을 재정의할 수 있습니다.

```toml
default_permissions = ":read-only"
```

(Hermes는 `# managed by hermes-agent` 마커 외부에 있는 한 다시 마이그레이션할 때 재정의를 보존합니다.)

## 보조 작업 및 ChatGPT 구독 토큰 비용

이 런타임이 `openai-codex` 공급자와 함께 켜져 있을 때, **보조 작업(제목 생성, 컨텍스트 압축, 비전 자동 감지, 백그라운드 자기 개선 리뷰 포크)도 기본적으로 ChatGPT 구독을 통해 흐릅니다.** 작업별 재정의가 설정되지 않은 경우 Hermes의 보조 클라이언트가 기본 공급자/모델을 사용하기 때문입니다.

이것은 `codex_app_server`에만 국한된 것이 아니라 기존의 `codex_responses` 경로에도 해당하지만, 구독 결제를 명시적으로 옵트인하기 때문에 여기에서 더 눈에 띕니다.

특정 보조 작업을 더 저렴한 / 다른 모델로 라우팅하려면 `~/.hermes/config.yaml`에 명시적 재정의를 설정하세요.

```yaml
auxiliary:
  title_generation:
    provider: openrouter
    model: google/gemini-3-flash-preview
  context_compression:
    provider: openrouter
    model: google/gemini-3-flash-preview
  vision_detect:
    provider: openrouter
    model: google/gemini-3-flash-preview
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

자기 개선 리뷰 포크는 `_current_main_runtime()`을 통해 기본 런타임을 상속하며 Hermes는 इसे 자동으로 `codex_app_server`에서 `codex_responses`로 다운그레이드합니다 (포크가 실제로 Hermes 자체의 에이전트 루프 도구인 `memory` 및 `skill_manage`를 호출할 수 있도록). 보조 작업을 다른 곳으로 라우팅하지 않는 한 그 포크는 여전히 구독 인증을 사용합니다.

## `~/.codex/config.toml` 안전하게 편집하기

Hermes는 관리하는 모든 것을 두 마커 주석 사이에 래핑합니다:

```toml
# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section
default_permissions = ":workspace"
[mcp_servers.filesystem]
...
[plugins."github@openai-curated"]
...
# end hermes-agent managed section
```

그 블록 **외부**에 있는 모든 것은 당신의 몫입니다. 마이그레이션을 다시 실행하면(또는 `/codex-runtime codex_app_server`를 통해 또는 런타임을 켤 때마다) 관리 블록이 제자리에서 교체되지만 그 위아래에 있는 사용자 콘텐츠는 문자 그대로 보존됩니다. 즉, 다음을 할 수 있습니다.

- Hermes가 모르는 자신의 MCP 서버 추가
- 프롬프트가 표시되는 것을 선호하는 경우 `default_permissions`를 `:read-only`로 재정의
- codex 전용 옵션 구성(모델, 공급자, otel 등)
- `[permissions.<name>]` 테이블에 사용자 정의 권한 프로필 추가

관리 블록 **내부**에 추가하는 것은 다음 마이그레이션 때 덮어쓰여집니다. 관리 블록 편집이 필요한 조정이 필요한 경우 이슈를 제기하면 조절기를 추가해 드리겠습니다.

## 다중 프로필 / 다중 테넌트 설정

기본적으로 Hermes는 활성화된 Hermes 프로필이 무엇이든 codex 서브프로세스를 `~/.codex/`에 가리키도록 합니다. 즉, `hermes -p work`와 `hermes -p personal`은 동일한 Codex 인증, 플러그인 및 구성을 공유합니다. 대부분의 사용자에게 이것은 올바른 동작입니다 — `codex` CLI를 직접 실행하는 것과 일치합니다.

프로필별 Codex 격리(별도 인증, 별도 설치된 플러그인, 별도 구성)를 원하면 프로필별로 `CODEX_HOME`을 명시적으로 설정하세요. 가장 깔끔한 방법은 `HERMES_HOME` 아래의 디렉토리를 가리키는 것입니다.

```bash
# work 프로필 내부에서 hermes를 래핑할 수 있습니다:
CODEX_HOME=~/.hermes/profiles/work/codex hermes chat
```

OAuth 토큰이 프로필 범위의 위치에 저장되도록 설정된 `CODEX_HOME`으로 `codex login`을 한 번 다시 실행해야 합니다. 그 후 `hermes -p work`는 격리된 Codex 상태에서 작동합니다.

기존 사용자의 `~/.codex/`를 이동하면 이전의 Codex CLI 인증이 조용히 무효화되므로 이 범위를 자동 지정하지 않습니다 — 이미 `codex login`을 실행한 사람은 누구든 다시 인증해야 할 것입니다. 선택적(Opt-in)인 것이 사용자를 놀라게 하는 것보다 더 안전하게 느껴집니다.

## HOME 환경 변수 패스스루

Hermes는 codex app-server 서브프로세스를 생성할 때 `HOME`을 다시 쓰지 **않습니다** (`os.environ.copy()`를 사용하고 `CODEX_HOME`과 `RUST_LOG`만 오버레이합니다). 이 의미는:

- codex가 `shell` 도구를 통해 실행하는 명령은 실제 사용자 `HOME`을 보고 `~/.gitconfig`, `~/.gh/`, `~/.aws/`, `~/.npmrc` 등을 올바르게 찾습니다.
- Codex의 내부 상태는 `CODEX_HOME`(기본적으로 `~/.codex/`를 가리킴)을 통해 격리된 상태로 유지됩니다.

이것은 OpenClaw가 초기 실험 후에 도달한 경계와 일치합니다: Codex의 상태를 격리하고 사용자의 홈을 내버려 둡니다. (참조: openclaw/openclaw#81562)

## MCP 서버 마이그레이션

Hermes의 `mcp_servers` 구성은 Codex가 예상하는 TOML 형식으로 자동 변환됩니다. 마이그레이션은 런타임을 활성화할 때마다 실행되며 멱등적(idempotent)입니다 — 재실행 시 관리 섹션은 대체되지만 사용자가 편집한 Codex 구성은 모두 보존됩니다.

변환되는 것:

| Hermes (`config.yaml`) | Codex (`config.toml`) |
|---|---|
| `command` + `args` + `env` | stdio 전송 |
| `url` + `headers` | streamable_http 전송 |
| `timeout` | `tool_timeout_sec` |
| `connect_timeout` | `startup_timeout_sec` |
| `enabled: false` | `enabled = false` |

마이그레이션되지 않는 것:
- `sampling`과 같은 Hermes 특정 키(Codex의 MCP 클라이언트에는 해당 항목이 없으며 서버당 경고와 함께 삭제됩니다).

## 네이티브 Codex 플러그인 마이그레이션

`codex plugin`을 통해 설치된 플러그인(Linear, GitHub, Gmail, Calendar, Canva 등)은 Codex의 `plugin/list` RPC를 통해 검색됩니다. `installed: true`인 각 플러그인에 대해 Hermes는 Hermes 세션에서 활성화하는 `[plugins."<name>@openai-curated"]` 블록을 씁니다.

즉: 친구가 "Codex CLI에 Calendar와 GitHub를 설정했어"라고 말하고 Hermes의 codex 런타임을 활성화하면 Hermes가 자동으로 활성화합니다. 재구성이 필요하지 않습니다.

마이그레이션되지 않는 것:
- 아직 설치하지 않은 플러그인 — 먼저 Codex에 설치하세요.
- codex가 `availability != AVAILABLE`이라고 보고하는 플러그인 (손상된 설치, 만료된 OAuth, 마켓플레이스에서 제거됨 등). 활성화 시 실패할 구성을 쓰지 않도록 건너뜁니다.
- ChatGPT 앱 마켓플레이스 항목(계정별 `app/list` 결과 — 이는 계정 인증 덕분에 이미 codex 내에서 활성화되어 있습니다).
- 플러그인 OAuth — Codex 자체에서 각 플러그인을 한 번 승인합니다. Hermes는 자격 증명을 건드리지 않습니다.

## Hermes 도구 콜백 (새로운 MCP 서버)

Codex의 내장 툴셋은 쉘/파일 작업/패치를 다루지만 웹 검색, 브라우저 자동화, 비전, 이미지 생성 등은 없습니다. codex 턴에서 이들을 사용 가능하게 유지하기 위해 Hermes는 `~/.codex/config.toml`에 자체를 MCP 서버로 등록합니다.

```toml
[mcp_servers.hermes-tools]
command = "/path/to/python"
args = ["-m", "agent.transports.hermes_tools_mcp_server"]
env = { HERMES_HOME = "/your/.hermes", PYTHONPATH = "...", HERMES_QUIET = "1" }
startup_timeout_sec = 30.0
tool_timeout_sec = 600.0
```

모델이 `web_search`(또는 노출된 다른 Hermes 도구)를 호출할 때 codex는 stdio를 통해 `hermes_tools_mcp_server` 서브프로세스를 생성하고, 요청은 `model_tools.handle_function_call()`을 통해 디스패치되며, 결과는 다른 MCP 응답처럼 codex에 투영됩니다.

**콜백을 통해 사용 가능한 도구:** `web_search`, `web_extract`, `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`, `vision_analyze`, `image_generate`, `skill_view`, `skills_list`, `text_to_speech`.

**사용할 수 없는 도구:** `delegate_task`, `memory`, `session_search`, `todo`. 이들은 디스패치하기 위해 실행 중인 AIAgent 컨텍스트(루프 중간 상태)가 필요하며 상태 비저장 MCP 콜백은 이를 제공할 수 없습니다. 이들이 필요할 때는 기본 Hermes 런타임(`/codex-runtime auto`)을 사용하세요.

## 비활성화하기

언제든지 다시 전환하세요:

```
/codex-runtime auto
```

다음 세션에 적용됩니다. Codex 관리 블록은 `~/.codex/config.toml`에 유지되므로 나중에 구성 손실 없이 다시 활성화할 수 있습니다 — 또는 원한다면 수동으로 제거하세요.

## 한계 사항 (Limitations)

이 런타임은 **옵트인 베타(opt-in beta)** 입니다. Hermes Agent 2026.5 + Codex CLI 0.130.0 기준으로 작동하는 항목:

- 다중 턴 대화
- Hermes UI를 통한 `commandExecution` 및 `fileChange` (apply_patch) 승인
- MCP 도구 호출 (`@modelcontextprotocol/server-filesystem` 및 새로운 `hermes-tools` 콜백에 대해 검증됨)
- 네이티브 Codex 플러그인 마이그레이션 (Linear / GitHub / Calendar 인벤토리에 대해 검증됨)
- 거부/취소 경로
- 토글 켜기/끄기 주기
- 메모리 및 스킬 넛지 카운터 (통합 테스트를 통해 실시간 검증됨)
- codex를 통한 Hermes web_search (실시간 검증: "OpenAI Codex CLI – Getting Started"가 끝에서 끝까지 반환됨)

알려진 한계:

- **Hermes 인증과 codex 인증은 별개의 세션입니다.** 가장 깔끔한 UX를 위해서는 `codex login`과 `hermes auth login codex`가 모두 필요합니다 (런타임은 LLM 호출에 codex의 세션을 사용합니다). 이것은 Hermes의 `_import_codex_cli_tokens`에 있는 의도적인 설계 선택입니다 — Hermes는 토큰 새로 고침 시 서로를 덮어쓰지 않도록 codex CLI와 OAuth 상태를 공유하지 않습니다.
- **`delegate_task`, `memory`, `session_search`, `todo`는 이 런타임에서 사용할 수 없습니다.** 이들은 상태 비저장 MCP 콜백이 제공할 수 없는 실행 중인 AIAgent 컨텍스트가 필요합니다. 이들이 필요할 때는 `/codex-runtime auto`를 사용하세요.
- **codex가 변경 사항 집합(changeset)을 추적하지 않을 때 승인 프롬프트에 인라인 패치 미리보기가 표시되지 않습니다.** Codex의 `fileChange` 승인 매개변수는 항상 변경 사항 집합을 전달하지는 않습니다. Hermes는 가능할 때 해당하는 `item/started` 알림의 데이터를 캐시하지만 항목이 스트리밍되기 전에 승인이 도착하면 프롬프트는 codex가 제공하는 모든 `reason`으로 폴백합니다.
- **초 미만(Sub-second)의 취소는 보장되지 않습니다.** 스트림 중간 인터럽트(codex가 응답하는 동안 Ctrl+C)는 `turn/interrupt`를 통해 전송되지만 codex가 이미 마지막 메시지를 플러시한 경우 응답을 계속 받게 됩니다.

버그를 발견하면 `hermes logs --since 5m`의 출력과 함께 [이슈를 오픈](https://github.com/NousResearch/hermes-agent/issues)하세요. 제목에 `codex-runtime`을 언급하면 분류하기 쉽습니다.

## 아키텍처

```
                ┌─── Hermes 쉘 (CLI / TUI / 게이트웨이) ───┐
                │  세션 DB · 슬래시 명령어 · 메모리        │
                │  & 스킬 리뷰 · cron · 세션 선택기        │
                └──┬──────────────────────────────────────┬┘
                   │ user_message               최종      │
                   ▼                            텍스트 +  │
        ┌──────────────────────────────────┐   투영된     │
        │  AIAgent.run_conversation()       │   메시지들  │
        │   if api_mode == codex_app_server │              │
        │     → CodexAppServerSession       │              │
        │   else: chat_completions / codex_responses (기본값)
        └────┬─────────────────────────────┘              │
             │ stdio 상의 JSON-RPC                        │
             ▼                                            │
        ┌──────────────────────────────────┐              │
        │  codex app-server (서브프로세스)  │──────────────┘
        │   thread/start, turn/start        │
        │   item/* 알림                     │
        │   shell + apply_patch + update_plan│
        │   view_image + sandbox            │
        │   ┌─────────────────────────┐     │
        │   │  MCP 클라이언트         │     │
        │   │  ├─ 사용자 MCP 서버     │     │
        │   │  ├─ 네이티브 플러그인   │     │
        │   │  │   (linear, github,   │     │
        │   │  │    gmail, calendar,  │     │
        │   │  │    canva, ...)       │     │
        │   │  └─ hermes-tools ───────┼─────────────────┐
        │   │       (Hermes의 더     │     │           │
        │   │        풍부한 도구로의 │     │           │
        │   │        콜백)           │     │           │
        │   └─────────────────────────┘     │           │
        └──────────────────────────────────┘           │
                                                        │
                                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  hermes_tools_mcp_server.py (온디맨드 서브프로세스)        │
        │   web_search, web_extract, browser_*, vision_analyze,    │
        │   image_generate, skill_view, skills_list, text_to_speech│
        └──────────────────────────────────────────────────────────┘
```

구현 세부 정보는 [PR #24182](https://github.com/NousResearch/hermes-agent/pull/24182) 및 [Codex app-server 프로토콜 README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)를 참조하세요.
