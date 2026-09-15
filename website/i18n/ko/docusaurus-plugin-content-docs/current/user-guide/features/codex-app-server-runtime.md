---
title: Codex App-Server 런타임 (선택 사항)
sidebar_label: Codex App-Server 런타임
---

# Codex App-Server 런타임

Hermes는 자체 도구 루프를 실행하는 대신 [Codex CLI app-server](https://github.com/openai/codex)에 `openai/*` 및 `openai-codex/*` 턴을 선택적으로 맡길 수 있습니다. 활성화하면 터미널 명령, 파일 편집, 샌드박싱, MCP 도구 호출이 모두 Codex 런타임 안에서 실행되고, Hermes는 그 바깥의 셸(세션 DB, 슬래시 명령, 게이트웨이, 메모리 및 스킬 검토) 역할을 합니다.

이는 **옵트인으로만 활성화됩니다**. 플래그를 전환하지 않는 한 Hermes의 기본 동작은 변하지 않습니다. Hermes는 사용자를 이 런타임으로 자동 라우팅하지 않습니다.

:::tip
OpenAI Codex를 사용하지 않나요? `hermes setup --portal`은 한 단계로 Claude/Gemini 등을 사용하는 비-Codex 백엔드를 구성합니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 이유

- **ChatGPT 구독**으로 OpenAI 에이전트 턴을 실행합니다(API 키 필요 없음). Codex CLI가 사용하는 것과 동일한 인증 흐름을 사용합니다.
- **Codex 자체 도구 세트와 샌드박스**를 사용합니다. 터미널/읽기/쓰기/검색에는 `shell`, 구조화된 편집에는 `apply_patch`, 계획에는 `update_plan`을 사용하며, 모두 seatbelt/landlock 샌드박싱 안에서 실행됩니다.
- **네이티브 Codex 플러그인** — `codex plugin`으로 설치한 Linear, GitHub, Gmail, Calendar, Canva 등이 자동으로 마이그레이션되어 Hermes 세션에서 활성화됩니다.
- **Hermes의 더 풍부한 도구도 함께 사용됩니다** — web_search, web_extract, 브라우저 자동화, vision, 이미지 생성, 스킬, TTS가 MCP 콜백을 통해 작동합니다. Codex에 기본 제공되지 않는 도구는 모델이 Hermes로 콜백합니다.
- **메모리 및 스킬 알림도 계속 작동합니다** — Codex 이벤트가 Hermes의 메시지 형태로 투영되므로 자기 개선 루프가 일반적인 대화 기록처럼 보이는 트랜스크립트를 확인합니다.

## 모델이 실제로 사용할 수 있는 도구

이 부분은 대부분의 사용자가 가장 먼저 알고 싶어 하는 내용입니다. 이 런타임이 켜져 있을 때 턴을 실행하는 모델에는 서로 독립적인 세 가지 도구 공급원이 있습니다.

### 1. Codex 내장 도구 세트 (항상 활성화)

이 도구들은 `codex app-server` 자체에 포함되어 있습니다. Hermes, MCP, 플러그인이 필요하지 않습니다. 런타임이 시작되는 즉시 다섯 가지를 모두 사용할 수 있습니다.

- **`shell`** — 샌드박스 안에서 임의의 셸 명령을 실행합니다. 모델이 파일을 읽고(`cat`, `head`, `tail`), 쓰고(`echo > foo`, heredoc), 검색하고(`find`, `rg`, `grep`), 디렉터리를 탐색하고(`ls`, `cd`), 빌드를 실행하고, 프로세스를 관리하는 등 bash에서 할 수 있는 모든 작업을 수행하는 방법입니다.
- **`apply_patch`** — Codex 패치 형식의 구조화된 다중 파일 diff를 적용합니다. 모델은 사소하지 않은 코드 편집(함수 추가, 여러 파일에 걸친 리팩터링)에 이를 사용하며, 일회성 쓰기에는 셸 heredoc도 사용할 수 있습니다.
- **`update_plan`** — Codex 내부 todo/계획 추적기입니다. Hermes의 `todo` 도구와 동등하지만 Codex 런타임 안에서 전적으로 관리됩니다.
- **`view_image`** — 로컬 이미지 파일을 대화에 불러와 모델이 볼 수 있게 합니다.
- **`web_search`** — 구성된 경우 Codex 자체 웹 검색을 사용할 수 있습니다. Hermes도 아래의 콜백을 통해 `web_search`(Firecrawl 기반)를 제공합니다. 모델은 원하는 쪽을 선택합니다.

따라서 터미널로 할 수 있는 **읽기/쓰기/검색/찾기/실행 작업은 무엇이든 Codex가 기본으로 수행**합니다. 샌드박스 프로필(런타임을 활성화하면 기본값은 `:workspace`)이 쓰기 가능한 범위를 제어합니다.

### 2. 네이티브 Codex 플러그인 (설치된 `codex plugin`에서 자동 마이그레이션)

런타임을 활성화하면 Hermes가 Codex의 `plugin/list` RPC를 조회하고, 설치된 각 플러그인에 대해 `[plugins."<name>@openai-curated"]` 항목을 기록합니다. 플러그인 자체는 Codex가 관리하며 Codex 자체 UI를 통해 한 번 승인합니다.

예시는 다음과 같습니다(OpenClaw 스레드에서 "YouTube 영상으로 만들 만하다"고 강조한 항목들입니다).

- **Linear** — 이슈 찾기/업데이트
- **GitHub** — 코드 검색, PR 보기, 댓글 작성
- **Gmail** — 메일 읽기/보내기
- **Google Calendar** — 이벤트 생성/찾기
- **Outlook calendar/email** — Microsoft 커넥터를 통한 동일한 형태의 기능
- **Canva** — 디자인 생성
- Codex CLI를 통해 `codex plugin marketplace add openai-curated` + `codex plugin install ...`으로 설치한 그 밖의 항목

마이그레이션되지 않는 항목:
- 아직 설치하지 않은 플러그인 — 먼저 Codex에 설치하세요.
- ChatGPT 앱 마켓플레이스 항목(`app/list`) — 계정 인증 덕분에 이미 Codex 안에서 활성화되어 있습니다.

### 3. Hermes 도구 콜백 (MCP 서버, `~/.codex/config.toml`에 등록)

Hermes는 Codex에 기본 제공되지 않는 도구를 Codex가 호출할 수 있도록 자체적으로 MCP 서버를 등록합니다. 콜백을 통해 다음을 사용할 수 있습니다.

- **`web_search`** / **`web_extract`** — Firecrawl 기반이며, 구조화된 콘텐츠를 스크래핑하는 것보다 깔끔한 결과를 제공하는 경향이 있습니다.
- **`browser_navigate` / `browser_click` / `browser_type` / `browser_press` / `browser_snapshot` / `browser_scroll` / `browser_back` / `browser_get_images` / `browser_console` / `browser_vision`** — Camofox 또는 Browserbase를 통한 완전한 브라우저 자동화입니다.
- **`vision_analyze`** — 별도의 vision 모델을 호출해 이미지를 검사합니다(Codex의 이미지를 대화에 불러오는 `view_image`와는 다릅니다).
- **`image_generate`** — Hermes의 image_gen 플러그인 체인을 통한 이미지 생성입니다.
- **`skill_view` / `skills_list`** — Hermes 스킬 라이브러리에서 읽습니다.
- **`text_to_speech`** — Hermes에 구성된 프로바이더를 통한 TTS입니다.

모델이 이 중 하나를 요청하면 Codex는 stdio MCP를 통해 `hermes_tools_mcp_server` 하위 프로세스를 실행하고, 호출은 `model_tools.handle_function_call()`(Hermes 기본 런타임과 동일한 코드 경로)를 통해 전달되며 결과는 다른 MCP 응답과 동일하게 Codex로 반환됩니다.

### 이 런타임에서 사용할 수 없는 항목

다음 네 가지 Hermes 도구는 실행 중인 AIAgent 컨텍스트(루프 중간 상태)가 있어야 디스패치할 수 있으며, 상태가 없는 MCP 콜백으로는 구동할 수 없습니다. 필요할 때는 기본 런타임(`/codex-runtime auto`)으로 전환하세요.

- **`delegate_task`** — 하위 에이전트를 생성합니다.
- **`memory`** — Hermes의 영구 메모리 저장소입니다.
- **`session_search`** — 세션 간 검색입니다.
- **`todo`** — Hermes의 todo 저장소입니다(Codex의 `update_plan`은 런타임 내부에서 이에 상응하는 기능입니다).

## 워크플로 기능 (`/goal`, kanban, cron)

### `/goal` (Ralph 루프)

**이 런타임에서 작동합니다.** 목표는 세션 ID를 키로 `state_meta`에 저장되고, 연속 프롬프트는 `run_conversation()`을 통해 일반 사용자 메시지로 되돌아가며, Codex가 다음 턴을 기본으로 실행합니다. 목표 판정기는 활성 런타임과 무관하게 보조 클라이언트(`auxiliary.goal_judge` in config.yaml)를 통해 실행됩니다. 판정기의 "차단됨, 사용자 입력 필요" 판정은 Codex가 승인에서 멈출 때 빠져나오는 깔끔한 방법입니다.

**알아둘 점 한 가지:** 각 연속 프롬프트는 새로운 Codex 턴이므로 Codex가 매번 명령 승인 정책을 처음부터 다시 평가합니다. 쓰기가 많은 장기 목표를 수행한다면 단일 세션 작업보다 승인 요청이 더 많이 표시될 수 있습니다. `default_permissions = ":workspace"`를 설정하면(런타임 활성화 시 Hermes가 자동으로 설정) 간단한 워크스페이스 쓰기에 매번 승인을 요청하지 않습니다.

### Kanban (다중 에이전트 worktree 디스패치)

**이 런타임에서 작동하지만 한 가지 미묘한 의존성이 있습니다.** kanban 디스패처는 사용자의 설정을 읽는 별도 `hermes chat -q` 하위 프로세스로 각 워커를 생성합니다. 즉 전역으로 `model.openai_runtime: codex_app_server`가 설정되어 있으면 워커도 Codex 런타임으로 시작합니다.

Codex 런타임 워커 안에서 작동하는 항목:
- Codex의 전체 도구 세트(shell, apply_patch, update_plan, view_image, web_search) — 워커가 실제 작업을 기본으로 수행합니다.
- 마이그레이션된 Codex 플러그인 — Linear, GitHub 등
- 브라우저, vision, image_gen, 스킬, TTS를 위한 Hermes 도구 콜백

MCP 콜백이 노출하므로 작동하는 항목:
- **`kanban_complete` / `kanban_block` / `kanban_comment` / `kanban_heartbeat`** — 워커 인계 도구입니다. 이들은 환경 변수에서 `HERMES_KANBAN_TASK`를 읽고(디스패처가 설정), 올바르게 접근을 제한하며, `HERMES_KANBAN_DB`로 고정된 보드별 SQLite DB에 씁니다. 이 콜백이 없으면 이 런타임의 워커는 작업을 수행할 수는 있어도 보고할 수 없어 디스패처의 타임아웃까지 멈춥니다.
- **`kanban_show` / `kanban_list`** — 워커가 자신의 컨텍스트를 확인할 때 사용하는 읽기 전용 보드 조회입니다.
- **`kanban_create` / `kanban_unblock` / `kanban_link`** — Codex 런타임에서 새 작업을 디스패치해야 하는 오케스트레이터 에이전트가 사용할 수 있는 오케스트레이터 전용 작업입니다.

kanban 도구는 디스패처가 설정하는 `HERMES_KANBAN_TASK` 환경 변수로 제한됩니다. 이 변수는 Codex 하위 프로세스로 전달되고(Codex는 환경을 상속), 그곳에서 생성된 `hermes-tools` MCP 서버 하위 프로세스에도 전달됩니다. 따라서 도구는 올바른 작업 ID를 확인하고 접근을 제한합니다. Codex app-server 워커의 경우 Hermes는 `HERMES_KANBAN_TASK`가 있을 때 좁은 app-server 샌드박스 재정의도 전달합니다. `workspace-write` 샌드박싱을 유지하고 **보드 DB 디렉터리와 디스패처가 고정한 모든 Kanban 경로**(`HERMES_KANBAN_WORKSPACES_ROOT`, `HERMES_KANBAN_WORKSPACE`, 레거시 `HERMES_KANBAN_ROOT` — 중복 제거, DB 디렉터리 우선)를 추가 쓰기 루트로 넣으며 네트워크는 기본적으로 비활성화합니다. 이렇게 하면 별도 드라이브의 `/media/.../kanban-workspaces/...`처럼 워크스페이스 마운트가 DB 디렉터리 바깥에 있어도, 취약한 `:danger-no-sandbox` 우회 없이 `kanban_complete` / `kanban_block`이 보드 DB를 업데이트하고 워커가 보고서/아티팩트를 쓸 수 있습니다 — [issue #27941](https://github.com/NousResearch/hermes-agent/issues/27941).

### Cron 작업

**특별히 테스트되지는 않았습니다.** Cron 작업은 CLI와 동일한 코드 경로인 `cronjob` → `AIAgent.run_conversation`을 통해 실행됩니다. Cron 작업의 설정에 `openai_runtime: codex_app_server`가 있으면 Codex에서 실행됩니다. 동일한 도구 사용 가능성 규칙이 적용됩니다. Codex 내장 도구 + 플러그인 + MCP 콜백은 작동하지만, 에이전트 루프 도구(delegate_task, memory, session_search, todo)는 작동하지 않습니다. 작업이 이러한 도구에 의존한다면 기본 런타임을 사용하는 프로필로 cron 범위를 지정하세요.

## 장단점

|  | Hermes 기본 런타임 | Codex app-server (옵트인) |
|---|---|---|
| `delegate_task` 하위 에이전트 | 예 | 사용할 수 없음 — 에이전트 루프 컨텍스트 필요 |
| `memory`, `session_search`, `todo` | 예 | 사용할 수 없음 — 에이전트 루프 컨텍스트 필요 |
| `web_search`, `web_extract` | 예 | 예 (MCP 콜백을 통해) |
| 브라우저 자동화 (Camofox/Browserbase) | 예 | 예 (MCP 콜백을 통해) |
| `vision_analyze`, `image_generate` | 예 | 예 (MCP 콜백을 통해) |
| `skill_view`, `skills_list` | 예 | 예 (MCP 콜백을 통해) |
| `text_to_speech` | 예 | 예 (MCP 콜백을 통해) |
| Codex `shell` (터미널/읽기/쓰기/검색/찾기/실행) | — | 예 (Codex 기본 제공) |
| Codex `apply_patch` (구조화된 다중 파일 편집) | — | 예 (Codex 기본 제공) |
| Codex `update_plan` (런타임 내부 todo) | — | 예 (Codex 기본 제공) |
| Codex `view_image` (이미지를 대화에 불러오기) | — | 예 (Codex 기본 제공) |
| Codex 샌드박스 (seatbelt/landlock, 프로필) | — | 예 (Codex 기본 제공) |
| ChatGPT 구독 인증 | — | 예 (`openai-codex` 프로바이더를 통해) |
| 네이티브 Codex 플러그인 (Linear, GitHub 등) | — | 예 (자동 마이그레이션) |
| 사용자 MCP 서버 | 예 | 예 (Codex로 자동 마이그레이션) |
| 메모리 + 스킬 검토 (백그라운드) | 예 | 예 (항목 투영을 통해) |
| 다중 턴 대화 | 예 | 예 |
| `/goal` (Ralph 루프) | 예 | 예 |
| Kanban 워커 디스패치 | 예 | 예 (콜백을 통해) |
| Kanban 오케스트레이터 도구 | 예 | 예 (콜백을 통해) |
| 모든 게이트웨이 플랫폼 | 예 | 예 |
| 비-OpenAI 프로바이더 | 예 | 해당 없음 — OpenAI/Codex 범위 |

### 실시간 표시

에이전트 루프가 Codex 하위 프로세스 안에서 실행되더라도 런타임은 Codex 이벤트 스트림을 기본 런타임이 사용하는 동일한 표시 경로로 연결합니다.

- 실시간 어시스턴트 델타, 추론(요약 델타 포함), 안정적인 ID를 가진 도구 시작/완료 이벤트가 턴이 실행되는 동안 TUI, 데스크톱, 메시징 게이트웨이에 표시됩니다. 완료 시점에만 동작하는 기록 프로젝터는 별도로 유지되므로, 재개한 세션에도 턴 중에 표시된 동일한 도구 카드가 채워집니다.
- 토큰 스트리밍이 비활성화되어도 게이트웨이 commentary가 계속 표시되고, 승인 요청보다 먼저 비워진 알림에 대해서도 실시간 도구 이벤트가 전달됩니다. commentary는 `display.show_commentary`를 따릅니다.

## 사전 요구 사항

1. **Codex CLI 설치:**
   ```bash
   npm i -g @openai/codex
   codex --version   # 0.130.0 or newer
   ```
2. **Codex OAuth 로그인.** Codex 하위 프로세스는 `~/.codex/auth.json`을 읽습니다. 이를 채우는 방법은 두 가지입니다.
   ```bash
   codex login                  # writes tokens to ~/.codex/auth.json
   ```
   Hermes 자체의 `hermes auth add openai-codex`는 `~/.hermes/auth.json`에 기록하며 이는 별도의 세션입니다. 아직 로그인하지 않았다면 **`codex login`을 별도로 실행하세요**.

3. **(선택 사항) 원하는 Codex 플러그인을 설치하세요.** 런타임을 활성화하면 Hermes가 Codex CLI를 통해 이미 설치한 큐레이션 플러그인을 자동으로 마이그레이션합니다.
   ```bash
   codex plugin marketplace add openai-curated
   # then via codex's TUI, install Linear / GitHub / Gmail / etc.
   ```
   Hermes가 이를 찾아 `~/.codex/config.toml`에 `[plugins."<name>@openai-curated"]` 항목을 자동으로 기록합니다.

## 활성화

Hermes 세션에서:

```
/codex-runtime codex_app_server
```

이 명령은 다음을 수행합니다.
- `codex` CLI가 설치되어 있는지 확인합니다(설치되어 있지 않으면 설치 안내와 함께 중단합니다).
- `model.openai_runtime: codex_app_server`를 config.yaml에 저장합니다.
- `~/.hermes/config.yaml`의 사용자 MCP 서버를 `~/.codex/config.toml`로 마이그레이션합니다.
- **설치된 네이티브 Codex 플러그인을 발견하고 마이그레이션합니다**(Linear, GitHub, Gmail, Calendar, Canva 등). Codex의 `plugin/list` RPC를 조회합니다.
- **Hermes 자체 도구를 MCP 서버로 등록**하여 Codex 하위 프로세스가 Codex에 기본 제공되지 않는 도구를 콜백할 수 있게 합니다.
- **`default_permissions = ":workspace"`를 기록**하여 샌드박스가 매번 작업을 승인받지 않고 워크스페이스 안에 쓸 수 있게 합니다.
- 무엇이 마이그레이션되었는지 알려줍니다. 다음 세션부터 적용되며, 프롬프트 캐시를 유효하게 유지하기 위해 현재 캐시된 에이전트는 이전 런타임을 계속 사용합니다.

동의어: `/codex-runtime on`, `/codex-runtime off`, `/codex-runtime auto`.

변경하지 않고 현재 상태를 확인하려면:
```
/codex-runtime
```

`~/.hermes/config.yaml`에서 수동으로 설정할 수도 있습니다.
```yaml
model:
  openai_runtime: codex_app_server   # default is "auto" (= Hermes runtime)
```

## 자기 개선 루프 (메모리 + 스킬 알림)

Hermes의 백그라운드 자기 개선은 카운터 임계값에서 실행됩니다.

- 사용자 프롬프트 10회마다 분기된 검토 에이전트가 대화를 살펴보고 메모리에 저장할 것이 있는지 결정합니다.
- 단일 턴 안에서 도구 반복 10회마다 같은 방식으로 스킬을 검토합니다(`skill_manage`가 기록합니다).

**둘 다 Codex 런타임에서 계속 작동합니다.** Codex 경로는 완료된 각 `commandExecution` / `fileChange` / `mcpToolCall` / `dynamicToolCall` 항목을 합성 `assistant tool_call` + `tool` 결과 메시지로 투영하므로, 검토가 실행될 때에는 기본 Hermes 런타임에서 보는 것과 동일한 형태를 확인합니다.

배선이 동일하게 유지되는 방식:

| | 기본 런타임 | Codex 런타임 |
|---|---|---|
| `_turns_since_memory` 증가 | run_conversation 사전 루프에서 사용자 프롬프트마다 | 조기 반환 전에 동일한 코드 경로 |
| `_iters_since_skill` 증가 | chat-completions 루프에서 도구 반복마다 | Codex 턴 반환 후 `turn.tool_iterations`만큼 |
| 메모리 트리거 (`_turns_since_memory >= _memory_nudge_interval`) | 사전 루프에서 계산하고 응답 후 실행 | 사전 루프에서 계산하여 Codex 헬퍼로 전달 |
| 스킬 트리거 (`_iters_since_skill >= _skill_nudge_interval`) | 루프 후 계산 | Codex 턴 후 계산 |
| `_spawn_background_review(messages_snapshot=..., review_memory=..., review_skills=...)` | 어느 트리거든 실행되면 호출 | 어느 트리거든 실행되면 동일하게 호출 |

한 가지 세부 사항: 검토 분기 자체는 Hermes의 에이전트 루프 도구(`memory`, `skill_manage`)를 호출해야 하며, 이 도구들은 Hermes 자체 디스패치가 필요합니다. 따라서 부모 에이전트가 `codex_app_server`에 있을 때 검토 분기는 `codex_responses`로 **강등됩니다**. 동일한 OAuth 자격 증명과 동일한 `openai-codex` 프로바이더를 사용하지만, Hermes가 루프와 에이전트 루프 도구를 소유하도록 OpenAI의 Responses API와 직접 통신합니다. 이는 사용자에게 보이지 않습니다.

결과적으로 Codex 런타임을 활성화하면 메모리 + 스킬 알림이 이전과 정확히 동일하게 계속 실행됩니다.

## 승인 작동 방식

Codex는 명령을 실행하거나 패치를 적용하기 전에 승인을 요청합니다. 이 요청은 Hermes의 표준 "위험한 명령" 프롬프트로 변환됩니다.

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

- **한 번 허용** → 이 단일 명령을 승인합니다.
- **이 세션에 허용** → Codex가 유사한 명령에 대해 다시 묻지 않습니다.
- **거부** → 명령이 거부되고 Codex는 읽기 전용 모드로 계속합니다.

`apply_patch`(파일 편집) 승인에서는 Codex가 해당 `fileChange` 항목을 통해 데이터를 제공할 때 Hermes가 변경 요약(`1 add, 1 update: /tmp/new.py, /tmp/old.py`)을 표시합니다.

## 권한 프로필

Codex에는 세 가지 기본 제공 권한 프로필이 있습니다.
- `:read-only` — 쓰기 금지, 모든 셸 명령에 승인 필요
- `:workspace` — 현재 워크스페이스 안의 쓰기는 프롬프트 없이 허용(Hermes가 런타임 활성화 시 사용하는 기본값)
- `:danger-no-sandbox` — 샌드박스 없음(이해하고 있는 경우가 아니면 사용하지 마세요)

Hermes의 관리 블록 바깥에서 `~/.codex/config.toml`의 기본값을 재정의할 수 있습니다.

```toml
default_permissions = ":read-only"
```

(`# managed by hermes-agent` 마커 바깥에 있는 한 Hermes는 재마이그레이션 시 재정의를 유지합니다.)

## 보조 작업과 ChatGPT 구독 토큰 비용

`openai-codex` 프로바이더로 이 런타임을 켜면 **보조 작업(제목 생성, 컨텍스트 압축, vision 자동 감지, 백그라운드 자기 개선 검토 분기)도 기본적으로 ChatGPT 구독을 통해 실행됩니다.** 작업별 재정의가 설정되지 않은 경우 Hermes의 보조 클라이언트가 주 프로바이더/모델을 사용하기 때문입니다.

이는 `codex_app_server`에만 해당하지 않습니다. 기존 `codex_responses` 경로에도 적용됩니다. 다만 여기서는 구독 과금에 명시적으로 옵트인하기 때문에 더 눈에 띕니다.

특정 보조 작업을 더 저렴하거나 다른 모델로 라우팅하려면 `~/.hermes/config.yaml`에 명시적인 재정의를 설정하세요.

```yaml
auxiliary:
  title_generation:
    provider: openrouter
    model: google/gemini-3-flash-preview
  compression:
    provider: openrouter
    model: google/gemini-3-flash-preview
  vision:
    provider: openrouter
    model: google/gemini-3-flash-preview
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

자기 개선 검토 분기는 `_current_main_runtime()`을 통해 주 런타임을 상속하고, Hermes는 이를 `codex_app_server`에서 `codex_responses`로 자동 강등합니다(분기가 `memory`와 `skill_manage` — Hermes 자체 에이전트 루프 도구 — 를 실제로 호출할 수 있도록 하기 위해서입니다). 보조 작업을 다른 곳으로 라우팅하지 않았다면 이 분기도 여전히 구독 인증을 사용합니다.

## `~/.codex/config.toml` 안전하게 편집하기

Hermes는 관리하는 모든 내용을 두 마커 주석 사이에 감쌉니다.

```toml
# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section
default_permissions = ":workspace"
[mcp_servers.filesystem]
...
[plugins."github@openai-curated"]
...
# end hermes-agent managed section
```

이 블록 **바깥**의 모든 내용은 사용자의 것입니다. 마이그레이션을 다시 실행하면(`/codex-runtime codex_app_server`를 통해 또는 런타임을 켤 때마다) 관리 블록을 그 자리에서 교체하지만 위와 아래의 사용자 콘텐츠는 그대로 보존합니다. 따라서 다음을 할 수 있습니다.

- Hermes가 모르는 자체 MCP 서버 추가
- 프롬프트를 받고 싶다면 `default_permissions`를 `:read-only`로 재정의
- Codex 전용 옵션(모델, 프로바이더, otel 등) 구성
- `[permissions.<name>]` 테이블에 사용자 정의 권한 프로필 추가

관리 블록 **안**에 추가한 내용은 다음 마이그레이션에서 덮어쓰입니다. 관리 블록 편집이 필요한 조정이라면 이슈를 등록해 주세요. 필요한 설정을 추가하겠습니다.

## 다중 프로필 / 다중 테넌트 설정

기본적으로 Hermes는 활성 Hermes 프로필과 관계없이 Codex 하위 프로세스를 `~/.codex/`로 지정합니다. 즉 `hermes -p work`와 `hermes -p personal`은 동일한 Codex 인증, 플러그인, 설정을 공유합니다. 대부분의 사용자에게 이는 올바른 동작이며, `codex` CLI를 직접 실행할 때와 동일합니다.

프로필별 Codex 격리(별도의 인증, 설치된 플러그인, 설정)를 원한다면 프로필별로 `CODEX_HOME`을 명시적으로 설정하세요. 가장 깔끔한 방법은 `HERMES_HOME` 아래의 디렉터리를 가리키는 것입니다.

```bash
# Inside the work profile, you might wrap hermes:
CODEX_HOME=~/.hermes/profiles/work/codex hermes chat
```

`CODEX_HOME`을 설정한 상태에서 `codex login`을 한 번 다시 실행해야 OAuth 토큰이 프로필 범위 위치에 저장됩니다. 이후 `hermes -p work`는 격리된 Codex 상태에서 작동합니다.

기존 사용자의 `~/.codex/`를 옮기면 Codex CLI 인증이 조용히 무효화되어 이미 `codex login`을 실행한 사용자가 다시 인증해야 하기 때문에 이를 자동으로 프로필 범위에 지정하지 않습니다. 사용자를 놀라게 하는 것보다 옵트인이 안전합니다.

## HOME 환경 변수 전달

Hermes는 Codex app-server 하위 프로세스를 생성할 때 `HOME`을 다시 쓰지 않습니다(`os.environ.copy()`를 사용하고 `CODEX_HOME`과 `RUST_LOG`만 덮어씁니다). 따라서:

- Codex가 `shell` 도구로 실행하는 명령은 실제 사용자의 `HOME`을 보고 `~/.gitconfig`, `~/.gh/`, `~/.aws/`, `~/.npmrc` 등을 올바르게 찾습니다.
- Codex의 내부 상태는 `CODEX_HOME`(기본적으로 `~/.codex/`를 가리킴)을 통해 격리됩니다.

이는 초기 실험 이후 OpenClaw가 도달한 경계와 일치합니다. Codex 상태는 격리하고 사용자의 홈은 그대로 둡니다. (openclaw/openclaw#81562 참고.)

## MCP 서버 마이그레이션

Hermes의 `mcp_servers` 설정은 Codex가 기대하는 TOML 형식으로 자동 변환됩니다. 마이그레이션은 런타임을 활성화할 때마다 실행되며 멱등적입니다. 다시 실행하면 관리 섹션을 교체하지만 사용자가 편집한 Codex 설정은 보존합니다.

변환되는 항목:

| Hermes (`config.yaml`) | Codex (`config.toml`) |
|---|---|
| `command` + `args` + `env` | stdio 전송 |
| `url` + `headers` | streamable_http 전송 |
| `timeout` | `tool_timeout_sec` |
| `connect_timeout` | `startup_timeout_sec` |
| `enabled: false` | `enabled = false` |

변환되지 않는 항목:
- `sampling` 같은 Hermes 전용 키(Codex의 MCP 클라이언트에는 동등한 기능이 없음 — 서버별 경고와 함께 삭제됩니다).

## 네이티브 Codex 플러그인 마이그레이션

`codex plugin`으로 설치한 플러그인(Linear, GitHub, Gmail, Calendar, Canva 등)은 Codex의 `plugin/list` RPC를 통해 발견됩니다. `installed: true`인 각 플러그인에 대해 Hermes는 `[plugins."<name>@openai-curated"]` 블록을 기록하여 Hermes 세션에서 활성화합니다.

즉 친구가 "Codex CLI에서 Calendar와 GitHub를 설정했어"라고 말한 뒤 Hermes의 Codex 런타임을 활성화하면 Hermes가 이를 자동으로 활성화합니다. 다시 구성할 필요가 없습니다.

변환되지 않는 항목:
- 아직 설치하지 않은 플러그인 — 먼저 Codex에 설치하세요.
- Codex가 `availability != AVAILABLE`로 보고하는 플러그인(설치 손상, OAuth 만료, 마켓플레이스에서 제거됨 등). 활성화 시 실패할 설정 기록을 피하기 위해 건너뜁니다.
- ChatGPT 앱 마켓플레이스 항목(계정별 `app/list` 결과) — 계정 인증 덕분에 이미 Codex 안에서 활성화되어 있습니다.
- 플러그인 OAuth — 각 플러그인을 Codex 자체에서 한 번 승인하며 Hermes는 자격 증명을 건드리지 않습니다.

## Hermes 도구 콜백 (새 MCP 서버)

Codex의 내장 도구 세트는 셸/파일 작업/패치를 다루지만 웹 검색, 브라우저 자동화, vision, 이미지 생성 등은 제공하지 않습니다. Codex 턴에서도 이를 사용할 수 있도록 Hermes는 `~/.codex/config.toml`에 자체 MCP 서버를 등록합니다.

```toml
[mcp_servers.hermes-tools]
command = "/path/to/python"
args = ["-m", "agent.transports.hermes_tools_mcp_server"]
env = { HERMES_HOME = "/your/.hermes", PYTHONPATH = "...", HERMES_QUIET = "1" }
startup_timeout_sec = 30.0
tool_timeout_sec = 600.0
```

모델이 `web_search`(또는 노출된 다른 Hermes 도구)를 호출하면 Codex가 stdio를 통해 `hermes_tools_mcp_server` 하위 프로세스를 생성하고, 요청은 `model_tools.handle_function_call()`을 통해 전달되며 결과는 다른 MCP 응답과 동일하게 Codex로 투영됩니다.

**콜백을 통해 사용할 수 있는 도구:** `web_search`, `web_extract`, `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`, `vision_analyze`, `image_generate`, `skill_view`, `skills_list`, `text_to_speech`.

**사용할 수 없는 도구:** `delegate_task`, `memory`, `session_search`, `todo`. 실행 중인 AIAgent 컨텍스트(루프 중간 상태)가 있어야 디스패치할 수 있으며 상태가 없는 MCP 콜백으로는 구동할 수 없습니다. 이 도구들이 필요하면 기본 Hermes 런타임(`/codex-runtime auto`)을 사용하세요.

## 비활성화

언제든 다음으로 전환할 수 있습니다.

```
/codex-runtime auto
```

다음 세션부터 적용됩니다. 나중에 설정을 잃지 않고 다시 활성화할 수 있도록 Codex 관리 블록은 `~/.codex/config.toml`에 남아 있습니다. 원한다면 수동으로 삭제할 수도 있습니다.

## 제한 사항

이 런타임은 **옵트인 베타**입니다. Hermes Agent 2026.5 + Codex CLI 0.130.0 기준으로 다음이 작동합니다.

- 다중 턴 대화
- Hermes UI를 통한 `commandExecution` 및 `fileChange`(apply_patch) 승인
- MCP 도구 호출(`@modelcontextprotocol/server-filesystem` 및 새 `hermes-tools` 콜백으로 확인)
- 네이티브 Codex 플러그인 마이그레이션(Linear / GitHub / Calendar 인벤토리로 확인)
- 거부/취소 경로
- 켜기/끄기 사이클
- 메모리 및 스킬 알림 카운터(통합 테스트에서 실시간 확인)
- Hermes web_search를 Codex를 통해 사용(실시간 확인: "OpenAI Codex CLI – Getting Started"가 종단 간 반환됨)

알려진 제한 사항:

- **Hermes 인증과 Codex 인증은 별도의 세션입니다.** 가장 원활하게 사용하려면 `codex login`과 `hermes auth add openai-codex`를 모두 실행해야 합니다(런타임은 LLM 호출에 Codex 세션을 사용합니다). 이는 Hermes의 `_import_codex_cli_tokens`에서 의도적으로 선택한 설계입니다. Hermes는 토큰 갱신 시 서로 덮어쓰는 일을 피하기 위해 Codex CLI와 OAuth 상태를 공유하지 않습니다.
- **`delegate_task`, `memory`, `session_search`, `todo`는 이 런타임에서 사용할 수 없습니다.** 상태가 없는 MCP 콜백이 제공할 수 없는 실행 중인 AIAgent 컨텍스트가 필요합니다. 필요할 때는 `/codex-runtime auto`를 사용하세요.
- **Codex가 변경 세트를 추적하지 않을 때 승인 프롬프트에 인라인 패치 미리보기가 없습니다.** Codex의 `fileChange` 승인 매개변수에 변경 세트가 항상 포함되는 것은 아닙니다. Hermes는 가능한 경우 해당 `item/started` 알림의 데이터를 캐시하지만, 항목이 스트리밍되기 전에 승인이 도착하면 프롬프트는 Codex가 제공하는 `reason`으로 대체됩니다.
- **1초 미만의 취소는 보장되지 않습니다.** Codex가 응답하는 도중의 인터럽트(Ctrl+C)는 `turn/interrupt`를 통해 전송되지만, Codex가 최종 메시지를 이미 내보냈다면 응답을 받게 됩니다.

버그를 발견하면 `hermes logs --since 5m`의 출력과 함께 [이슈를 등록](https://github.com/NousResearch/hermes-agent/issues)하세요. 쉽게 분류할 수 있도록 제목에 `codex-runtime`을 언급하세요.

## 아키텍처

```
                ┌─── Hermes shell (CLI / TUI / gateway) ───┐
                │  sessions DB · slash commands · memory   │
                │  & skill review · cron · session pickers │
                └──┬──────────────────────────────────────┬┘
                   │ user_message               final     │
                   ▼                            text +    │
        ┌──────────────────────────────────┐   projected  │
        │  AIAgent.run_conversation()       │   messages   │
        │   if api_mode == codex_app_server │              │
        │     → CodexAppServerSession       │              │
        │   else: chat_completions / codex_responses (default)
        └────┬─────────────────────────────┘              │
             │ JSON-RPC over stdio                        │
             ▼                                            │
        ┌──────────────────────────────────┐              │
        │  codex app-server (subprocess)    │──────────────┘
        │   thread/start, turn/start        │
        │   item/* notifications            │
        │   shell + apply_patch + update_plan│
        │   view_image + sandbox            │
        │   ┌─────────────────────────┐     │
        │   │  MCP client             │     │
        │   │  ├─ user MCP servers    │     │
        │   │  ├─ native plugins      │     │
        │   │  │   (linear, github,   │     │
        │   │  │    gmail, calendar,  │     │
        │   │  │    canva, ...)       │     │
        │   │  └─ hermes-tools ───────┼─────────────────┐
        │   │       (callback to     │     │           │
        │   │        Hermes' richer  │     │           │
        │   │        tools)          │     │           │
        │   └─────────────────────────┘     │           │
        └──────────────────────────────────┘           │
                                                        │
                                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  hermes_tools_mcp_server.py (subprocess on demand)        │
        │   web_search, web_extract, browser_*, vision_analyze,    │
        │   image_generate, skill_view, skills_list, text_to_speech│
        └──────────────────────────────────────────────────────────┘
```

구현 세부 정보는 [PR #24182](https://github.com/NousResearch/hermes-agent/pull/24182)와 [Codex app-server protocol README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)를 참고하세요.
