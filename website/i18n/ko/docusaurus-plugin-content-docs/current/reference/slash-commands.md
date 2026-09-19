---
sidebar_position: 2
title: "슬래시 명령어 레퍼런스"
description: "대화형 CLI 및 메시징 슬래시 명령어 전체 레퍼런스"
---

# 슬래시 명령어 레퍼런스

Hermes에는 `hermes_cli/commands.py`의 중앙 `COMMAND_REGISTRY`가 구동하는 두 가지 슬래시 명령어 표면이 있습니다.

- **대화형 CLI 슬래시 명령어** — `cli.py`가 디스패치하며, 레지스트리에서 자동 완성 기능을 가져옵니다.
- **메시징 슬래시 명령어** — `gateway/run.py`가 디스패치하며, 도움말과 플랫폼 메뉴가 레지스트리에서 생성됩니다.

설치된 스킬은 두 표면 모두에서 동적 슬래시 명령어로도 노출됩니다. 여기에는 `/plan`과 같은 번들 스킬도 포함되며, `/plan`은 계획 모드를 열고 활성 workspace/backend 작업 디렉터리를 기준으로 `.hermes/plans/` 아래에 Markdown 계획을 저장합니다.

## 권한 및 관리자/일반 사용자 구분

사용자별 허용 목록을 지원하는 모든 메시징 플랫폼(Telegram, Discord, Slack, Matrix, Mattermost, Signal, …)은 슬래시 명령어를 두 계층으로 나누는 기능도 지원합니다. **관리자**는 등록된 모든 명령어를 사용할 수 있고, **일반 사용자**는 `user_allowed_commands`에 나열한 명령어만 사용할 수 있습니다(항상 허용되는 최소 명령어 `/help` 및 `/whoami`는 예외). 플랫폼의 `extra:` 블록 안에서 `allow_admin_from` 및 `user_allowed_commands`(그룹별 대응 항목은 `group_allow_admin_from` / `group_user_allowed_commands`)를 `~/.hermes/gateway-config.yaml`에 설정합니다.

플랫폼별 문서에서 예시를 확인하세요. 구조는 모든 플랫폼에서 동일합니다.

- [Telegram](../user-guide/messaging/telegram.md#slash-command-access-control)
- [Discord](../user-guide/messaging/discord.md)
- [Slack](../user-guide/messaging/slack.md)
- [Matrix](../user-guide/messaging/matrix.md)
- [Mattermost](../user-guide/messaging/mattermost.md)
- [Signal](../user-guide/messaging/signal.md)

범위에 `allow_admin_from`이 설정되지 않으면 해당 범위는 제한 없는 하위 호환 모드로 유지됩니다. 즉, 허용된 모든 사용자가 모든 명령어를 실행할 수 있습니다.

## 대화형 CLI 슬래시 명령어

CLI에서 `/`를 입력하면 자동 완성 메뉴가 열립니다. 기본 제공 명령어는 대소문자를 구분하지 않습니다.

### 세션

| 명령어 | 설명 |
|---------|-------------|
| `/new [name]` (별칭: `/reset`) | 새 세션을 시작합니다(새 세션 ID + 기록). 선택 사항인 `[name]`은 최초 세션 제목을 설정합니다. 예를 들어 `/new my-experiment`는 제목이 이미 `my-experiment`로 지정된 새 세션을 열므로 나중에 `/resume` 또는 `/sessions`로 쉽게 찾을 수 있습니다. 확인 모달을 건너뛰려면 `now`, `--yes` 또는 `-y`를 덧붙입니다. 예: `/reset now`, `/new --yes my-experiment`. |
| `/clear` | 화면을 지우고 새 세션을 시작합니다. |
| `/history` | 대화 기록을 표시합니다(`/timestamps` 설정을 따릅니다). |
| `/save` | 현재 대화를 저장합니다. |
| `/prompt` (별칭: `/compose`) | 인라인 입력 대신 `$EDITOR`(Markdown)에서 다음 프롬프트를 작성합니다. 길거나 여러 줄이거나 형식을 세심하게 지정해야 하는 프롬프트에 유용합니다. |
| `/retry` | 마지막 메시지를 다시 시도합니다(에이전트에 재전송). |
| `/undo` | 마지막 사용자/어시스턴트 교환을 제거합니다. |
| `/title` | 현재 세션의 제목을 설정합니다(사용법: /title My Session Name). |
| `/compress [here [N] \| focus topic]` | 대화 컨텍스트를 수동으로 압축합니다(메모리를 비우고 요약). `/compress here [N]`은 가장 최근의 N개 교환(기본값 2)을 제외한 모든 내용을 요약하고, 해당 교환은 원문 그대로 유지합니다. 원하는 압축 경계를 직접 정할 수 있습니다. focus topic을 지정하면 전체 요약에서 보존할 내용을 좁힐 수 있습니다. |
| `/rollback` | 파일 시스템 체크포인트를 나열하거나 복원합니다(사용법: /rollback [number]). |
| `/diff [staged\|all\|session] [--stat] [path...]` | 작업 디렉터리의 git 변경 사항을 표시합니다. 기본값은 스테이징되지 않은 변경 사항과 추적되지 않은 파일입니다. `staged`는 커밋을 위해 스테이징된 항목을, `all`은 HEAD 이후의 모든 항목을, `session`은 Hermes가 이곳에서 변경한 모든 내용의 누적 diff를 표시합니다(가장 오래 유지된 체크포인트 기준선부터 계산하며 체크포인트를 활성화해야 합니다. `/rollback diff <N>`을 보완합니다). `--stat`은 변경된 파일 요약만 출력하고, 경로 인수로 diff 범위를 제한할 수 있습니다. |
| `/snapshot [create\|restore <id>\|prune]` (별칭: `/snap`) | Hermes 구성/상태의 상태 스냅샷을 생성하거나 복원합니다. `create [label]`은 스냅샷을 저장하고, `restore <id>`는 해당 스냅샷으로 되돌리며, `prune [N]`은 오래된 스냅샷을 제거합니다. 인수 없이 실행하면 모두 나열합니다. |
| `/stop` | 실행 중인 모든 백그라운드 프로세스를 종료합니다. |
| `/queue <prompt>` (별칭: `/q`) | 현재 에이전트 응답을 중단하지 않고 다음 턴에 실행할 프롬프트를 대기열에 넣습니다. |
| `/steer <prompt>` | 현재 실행 중인 에이전트에 다음 도구 호출 **후** 도착하는 중간 메모를 주입합니다. 중단이나 새 사용자 턴은 발생하지 않습니다. 현재 도구가 완료되면 텍스트가 마지막 도구 결과의 content에 추가되어, 현재 도구 호출 루프를 끊지 않고 에이전트에 새로운 컨텍스트를 제공합니다. 예를 들어 테스트 실행 중인 에이전트에 “focus on the auth module”이라고 알릴 때 사용합니다. |
| `/goal <text>` | Hermes가 여러 턴에 걸쳐 달성할 지속 목표를 설정합니다. Ralph 루프를 활용한 방식입니다. 매 턴 후 보조 judge 모델이 목표 달성 여부를 결정하며, 미완료 시 Hermes가 자동으로 계속 실행합니다. 하위 명령어: `/goal status`, `/goal pause`, `/goal resume`, `/goal clear`. 예산 기본값은 20턴(`goals.max_turns`)입니다. 실제 사용자 메시지가 연속 실행 루프를 선점하며, 상태는 `/resume` 후에도 유지됩니다. 자세한 전체 안내는 [Persistent Goals](/user-guide/features/goals)를 참조하세요. |
| `/subgoal <text>` | 활성 목표에 사용자가 제공한 기준을 추가합니다. 연속 실행 프롬프트에는 모든 하위 목표가 원문 그대로 표시되고, judge는 이를 DONE/CONTINUE 판정에 반영합니다. 따라서 원래 목표와 모든 하위 목표가 충족될 때까지 목표가 완료로 표시되지 않습니다. 하위 명령어: `/subgoal`(목록), `/subgoal remove <N>`, `/subgoal clear`. 활성 `/goal`이 필요합니다. |
| `/heartbeat every <interval> <prompt>` (별칭: `/hb`) | 세션이 유휴 상태이고 지정한 간격이 지나면 **이 세션**에 일반 사용자 턴으로 다시 들어오는 반복 프롬프트를 설정합니다(최소 60초, 누락된 tick은 합쳐짐). 하위 명령어: `/heartbeat status`, `/heartbeat pause`, `/heartbeat resume`, `/heartbeat clear`. 세션 범위이며 프로세스 내에서만 동작합니다. 영속적이고 격리된 일정에는 `hermes cron`을 사용하세요. 자세한 내용은 [Session Heartbeats](/user-guide/features/heartbeat)를 참조하세요. |
| `/refine [focus]` | 자동 post-turn 트리거를 기다리지 않고 백그라운드 메모리/스킬 자기 개선 검토를 **지금** 실행합니다. 선택적 focus 텍스트로 검토 방향을 지정할 수 있습니다(예: `/refine save the deploy workflow as a skill`). 대화 스냅샷을 기준으로 백그라운드 fork에서 실행되므로 라이브 세션과 prompt cache는 영향을 받지 않습니다. 완료되면 결과가 보고됩니다. |
| `/moa <prompt>` | 기본 [Mixture of Agents](/user-guide/features/mixture-of-agents) 프리셋으로 단일 프롬프트를 실행한 뒤 현재 모델을 복원합니다. 한 번만 실행되며 세션 모델은 변경하지 않습니다. |
| `/resume [name]` | 이전에 이름을 지정한 세션을 재개합니다. |
| `/sessions` (TUI 별칭: `/switch`) | Classic CLI에서는 대화형 선택기에서 이전 세션을 찾아 재개합니다. TUI에서는 현재 열려 있는 TUI 세션의 실시간 세션 전환기를 엽니다. TUI에서 `/sessions new`를 사용하면 즉시 다른 실시간 세션을 시작할 수 있습니다. |
| `/egress [status]` | Docker egress proxy 상태를 표시합니다 — 활성화/구성/실행 상태, credential source, token mappings, uncovered providers 및 다음 remediation step을 보여줍니다. CLI, TUI, Desktop chat 및 messaging gateway에서 작동합니다. |
| `/redraw` | 전체 UI를 강제로 다시 그립니다(tmux 크기 조정, 마우스 선택으로 인한 아티팩트 등으로 터미널 표시가 어긋난 경우 복구). |
| `/status` | 세션 정보(모델, provider, profile, session ID, working directory, title, created/updated timestamps, token totals, agent-running state)를 표시한 뒤 로컬 **Session recap** 블록을 표시합니다. 여기에는 최근 사용자/어시스턴트 턴 수, 도구 결과 수, 가장 많이 사용한 도구, 최근 수정한 파일, 최신 사용자 프롬프트 및 최신 어시스턴트 답변이 포함됩니다. recap은 메모리 내 대화에서 로컬로 계산되며 LLM 호출이나 prompt-cache 영향이 없습니다. |
| `/context [all]` (별칭: `/ctx`) | 시각적 컨텍스트 창 분석을 표시합니다. CLI/TUI에서는 5×20 glyph block grid(각 셀은 모델 창의 약 1%)와 범주별 예상 표를 표시합니다 — system prompt, tool definitions, rules, skills index, MCP, subagents, memory, conversation과 남은 여유 공간을 비교합니다. 메시징 플랫폼에서는 자동 압축 threshold/headroom, 압축 통계, 누적 throughput 및 동일한 범주별 표를 일반 텍스트 usage gauge로 표시합니다. `/context all`을 추가하면 스킬별 및 toolset별 비용 목록(index cost와 SKILL.md load cost, toolset별 schema tokens)이 붙습니다. 읽기 전용이며 로컬에서 계산됩니다 — LLM 호출이나 prompt-cache 영향이 없습니다. |
| `/agents` (별칭: `/tasks`) | 현재 세션 전체의 활성 에이전트와 실행 중인 작업을 표시합니다. |
| `/background <prompt>` (별칭: `/bg`, `/btw`) | 별도의 백그라운드 세션에서 프롬프트를 실행합니다. 에이전트는 프롬프트를 독립적으로 처리하므로 현재 세션은 다른 작업을 위해 계속 사용할 수 있습니다. 작업이 완료되면 결과가 패널로 표시됩니다. 자세한 내용은 [CLI Background Sessions](/user-guide/cli#background-sessions)를 참조하세요. |
| `/branch [name]` (별칭: `/fork`) | 현재 세션을 분기합니다(다른 경로 탐색). |
| `/journey [list\|delete <id>\|edit <id>]` (별칭: `/learning`, `/memory-graph`) | **CLI 전용.** 학습 여정 타임라인을 엽니다. |
| `/handoff <platform>` | **CLI 전용.** 현재 세션을 메시징 플랫폼(Telegram, Discord, Slack, WhatsApp, Signal, Matrix)으로 넘깁니다. gateway가 즉시 인계받고, 스레드를 지원하는 플랫폼에서는 새 스레드를 생성하며(Telegram topics, Discord text-channel threads, Slack message-anchored threads), 대상 세션을 CLI session_id에 다시 연결해 전체 role-aware transcript를 재생하고, 에이전트가 새 위치에서 작업 중임을 확인하도록 synthetic user turn을 생성합니다. 성공하면 CLI가 `/resume` 안내와 함께 정상 종료됩니다. `/resume <title>`로 언제든 로컬에서 재개할 수 있습니다. 턴 중에는 거부됩니다. gateway가 실행 중이어야 하며 대상 플랫폼에 home channel이 구성되어 있어야 합니다(대상 채팅에서 `/sethome` 사용). 자세한 내용은 [Cross-Platform Handoff](/user-guide/sessions#cross-platform-handoff)를 참조하세요. |
| `/journey [list\|delete <id>\|edit <id>]` (별칭: `/learning`, `/memory-graph`) | 학습한 스킬과 메모리의 학습 여정 타임라인을 엽니다. Classic CLI, TUI overlay 및 desktop app(Star Map panel)에서 작동합니다. 메시징 플랫폼에서는 사용할 수 없습니다. 자세한 내용은 [Learning Journey](/user-guide/features/memory#learning-journey-journey)를 참조하세요. |

### 구성

| 명령어 | 설명 |
|---------|-------------|
| `/config` | 현재 구성을 표시합니다. |
| `/model [model-name]` | 현재 모델을 표시하거나 변경합니다. `/model claude-sonnet-4`, `/model provider:model`(provider 전환), `/model custom:model`(custom endpoint), `/model custom:name:model`(이름이 지정된 custom provider), `/model custom`(endpoint에서 자동 감지) 및 사용자 정의 별칭(`/model fav`, `/model grok` — [Custom model aliases](#custom-model-aliases) 참조)을 지원합니다. 플래그: `--global`은 변경 사항을 config.yaml에 영속화하고, `--session`은 세션 전용으로 강제하며, `--once`는 다음 턴에만 적용하고, `--refresh`는 provider의 model list를 다시 가져오며, `--provider <name>`은 backend를 전환합니다(`--global`이 아니면 세션 전용). 일반 `/model <name>`은 `model.persist_switch_by_default: true`가 설정되지 않은 한 세션 전용입니다. **참고:** `/model`은 이미 구성된 provider 사이에서만 전환할 수 있습니다. 새 provider를 추가하려면 세션을 종료하고 터미널에서 `hermes model`을 실행하세요. **비용 참고:** 대화 중 모델을 전환하면 prompt cache가 초기화됩니다 — cache key에 모델이 포함되므로 다음 턴에는 약 75% 할인된 cached rate가 아니라 전체 입력 비용으로 전체 대화를 다시 읽습니다. 예상된 동작이며 피할 수 없지만, 긴 세션에서는 알아둘 가치가 있습니다. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | OpenAI/Codex 모델을 위한 선택적 [Codex app-server runtime](../user-guide/features/codex-app-server-runtime)을 전환합니다. `auto`(기본값)는 Hermes의 표준 chat completions를 사용하고, `codex_app_server`는 native shell, apply_patch, ChatGPT subscription auth 및 migrated Codex plugins를 지원하는 `codex app-server` subprocess에 턴을 넘깁니다. 다음 세션부터 적용됩니다. |
| `/personality` | 미리 정의된 personality를 설정합니다. `/personality none`(또는 `default` / `neutral`)은 overlay를 지우고 기본 동작으로 돌아갑니다. |
| `/verbose` | 도구 진행 표시를 순환합니다: off → new → all → verbose. config를 통해 [messaging에 활성화](#notes)할 수 있습니다. |
| `/focus [on\|off\|status]` | **focus view**를 전환합니다 — 프롬프트와 최종 응답만 표시하는 출력 축소 모드입니다. `/verbose`와 함께 사용할 수 있습니다. 켜면 도구 진행 상태를 `off`로 설정하고 이전 모드를 기억하며, `/focus off`를 실행하면 복원합니다. 각 턴이 끝날 때 흐리게 표시되는 복구 문구(`⋯ 7 tool lines hidden · /focus off to show`)가 나오고, 축소 보기 상태임을 항상 알 수 있도록 상태 표시줄에 `◉ focus` 배지가 계속 표시됩니다. 모델에 전송되는 내용은 달라지지 않습니다 — 세부 정보는 버려지지 않고 숨겨질 뿐입니다. |
| `/fast [normal\|fast\|status]` | fast mode를 전환합니다 — OpenAI Priority Processing / Anthropic Fast Mode입니다. 옵션: `normal`, `fast`, `status`. |
| `/reasoning [level\|show\|hide\|full\|clamp] [--global]` | reasoning effort 및 표시를 관리합니다. 레벨에는 `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra`가 있습니다. `show` / `hide`(또는 `on` / `off`)는 reasoning 표시를 전환하고, `full` 및 `clamp`는 reasoning 표시 방식을 조정합니다. `--global`은 effort를 config에 영속화합니다. |
| `/skin` | display skin/theme을 표시하거나 변경합니다. |
| `/export [profile] [-o out.tar.gz]` | **CLI 전용.** profile을 공유 가능한 `.tar.gz`로 묶습니다 — skills, memory, persona, crons, plugins, settings 및 (desktop에서) themes와 layout을 포함합니다. credentials(`auth.json`, `.env`)는 제거됩니다. 기본값은 활성 profile과 현재 디렉터리의 `<name>.tar.gz`입니다. `hermes profile export`와 동일한 archive를 생성합니다. 버전 관리되고 업데이트 가능한 공유에는 대신 [profile distribution](../user-guide/profile-distributions.md)을 사용하세요. |
| `/import <archive.tar.gz> [--name <name>]` | **CLI 전용.** profile archive를 새 profile로 설치합니다. archive에서 이름을 추론하며 `--name`을 지정하면 해당 이름을 사용합니다. 기존 profile을 덮어쓰지 않으며 `default`로 가져올 수 없습니다. 이름이 사용 가능하면 shell wrapper를 생성합니다. [Export and import a profile file](../user-guide/profile-distributions.md#export-and-import-a-profile-file)을 참조하세요. |
| `/statusbar` (별칭: `/sb`) | context/model status bar를 켜거나 끕니다. |
| `/battery [on\|off\|status]` | 색상으로 구분된 battery read-out을 status-bar의 첫 번째 요소로 전환합니다(기본값은 off이며 battery가 없으면 no-op). |
| `/voice [on\|off\|tts\|status]` | CLI voice mode와 음성 재생을 전환합니다. 녹음에는 `voice.record_key`(기본값: `Ctrl+B`)를 사용합니다. |
| `/yolo` | YOLO mode를 전환합니다 — 모든 위험한 명령어 승인 프롬프트를 건너뜁니다. |
| `/approvals [manual\|smart\|off]` | 영속적인 위험한 명령어 승인 모드를 표시하거나 설정합니다. |
| `/footer [on\|off\|status]` | 최종 답변에서 gateway runtime-metadata footer를 전환합니다(model, context % 및 cwd 표시). |
| `/busy [queue\|steer\|interrupt\|status]` | CLI 전용: Hermes가 작업 중일 때 Enter 키를 눌렀을 때의 동작을 제어합니다 — 새 메시지를 queue에 넣거나, mid-turn을 steer하거나, 즉시 interrupt합니다. |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | CLI 전용: TUI busy-indicator style을 선택합니다. |
| `/timestamps [on\|off\|status]` | CLI 전용: 메시지와 `/history`에 `[HH:MM]` timestamps를 표시할지 전환합니다. |
| `/wake [on\|off\|status]` | CLI 전용: "Hey Hermes" wake word listener를 전환합니다. |
### 도구 및 스킬

| 명령어 | 설명 |
|---------|-------------|
| `/tools [list\|disable\|enable] [name...]` | 도구를 관리합니다. 현재 세션에서 사용 가능한 도구를 나열하거나, 특정 도구를 비활성화 또는 활성화할 수 있습니다. 도구를 비활성화하면 에이전트의 도구 세트에서 제거되고 세션이 재설정됩니다. |
| `/toolsets` | 사용 가능한 도구 세트를 나열합니다. |
| `/browser [connect\|disconnect\|status]` | 로컬 Chromium 계열 CDP 연결을 관리합니다. `connect`는 실행 중인 Chrome, Brave, Chromium 또는 Edge에 연결합니다(기본값: `http://127.0.0.1:9222`). `disconnect`는 연결을 해제합니다. `status`는 현재 연결 상태를 표시합니다. 디버거가 감지되지 않으면 지원되는 Chromium 계열 브라우저를 자동으로 실행합니다. |
| `/skills` | 온라인 레지스트리에서 스킬을 검색, 설치, 검사 또는 관리합니다. 스킬 작성 승인 게이트를 검토하는 화면이기도 합니다. `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>`, `/skills approval on\|off`를 사용할 수 있습니다. [스킬 작성 승인 게이트](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval)를 참고하세요. |
| `/memory [pending\|approve\|reject\|approval]` | 작성 승인 게이트에 의해 보류된 메모리 기록을 검토하고(`memory.write_approval`), 해당 게이트를 켜거나 끕니다. [메모리 기록 관리](/user-guide/features/memory#controlling-memory-writes-write_approval)를 참고하세요. |
| `/bundles` | 구성된 스킬 번들을 나열합니다. 여러 스킬을 한 번에 미리 불러오는 `/<name>` 슬래시 별칭입니다. `~/.hermes/config.yaml`의 `bundles:` 아래에서 구성합니다. [스킬 번들](/user-guide/features/skills#skill-bundles)을 참고하세요. |
| `/learn <what to learn from>` | 설명한 내용에서 재사용 가능한 스킬을 추출합니다. 디렉터리, URL, 방금 에이전트와 진행한 작업 흐름 또는 붙여 넣은 메모를 지정할 수 있습니다. 에이전트가 자체 도구로 소스를 수집하고 하우스 작성 표준에 맞는 `SKILL.md`를 작성합니다. CLI, 메시징 게이트웨이, TUI, 대시보드 Skills 페이지에서 사용할 수 있습니다. |
| `/init [notes]` | 저장소를 검사해 프로젝트 지침 `AGENTS.md`를 생성하거나 업데이트합니다(Codex `/init`의 이식판). 에이전트는 읽기 전용 도구로 매니페스트, 레이아웃, 도구 체인 설정을 검사한 뒤 간결한 `AGENTS.md`를 작성합니다. 파일이 이미 있으면 기존 내용을 보존하면서 병합해 업데이트합니다. 선택 사항인 메모로 강조할 내용을 지정할 수 있습니다. CLI, 메시징 게이트웨이, TUI에서 사용할 수 있습니다. |
| `/cron` | 예약 작업을 관리합니다(목록, 추가/생성, 편집, 일시 중지, 재개, 실행, 제거). |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` (별칭: `/suggest`) | 추천 자동화를 검토합니다. `/suggestions`는 보류 중인 추천을 나열하고, `/suggestions accept <id>`는 제안된 자동화를 생성하며, `/suggestions dismiss <id>`는 거부하고, `/suggestions catalog`는 엄선된 시작 자동화를 추가하며, `/suggestions clear`는 해결된 추천 기록을 삭제합니다. 수락한 작업은 현재 표면을 전달 출처로 유지합니다. |
| `/blueprint [name] [slot=value ...]` (별칭: `/bp`) | 블루프린트 템플릿에서 자동화를 설정합니다. 인수 없이 `/blueprint`를 실행하면 카탈로그를 나열하고, `/blueprint <name>`은 다음 에이전트 턴에 안내형 슬롯 입력 흐름을 시작하며, `/blueprint <name> slot=value ...`는 작업을 직접 생성합니다. |
| `/curator` | 백그라운드 스킬 유지 관리를 수행합니다(`status`, `run`, `pin`, `archive`). [Curator](/user-guide/features/curator)를 참고하세요. |
| `/kanban <action>` | 채팅을 벗어나지 않고 여러 프로필과 프로젝트의 협업 보드를 조작합니다. 전체 `hermes kanban` 인터페이스를 사용할 수 있습니다: `/kanban list`, `/kanban show t_abc`, `/kanban create "title" --assignee X`, `/kanban comment t_abc "text"`, `/kanban unblock t_abc`, `/kanban dispatch` 등. 여러 보드도 지원합니다: `/kanban boards list`, `/kanban boards create <slug>`, `/kanban boards switch <slug>`, `/kanban --board <slug> <action>`. [Kanban 슬래시 명령어](/user-guide/features/kanban#kanban-slash-command)를 참고하세요. |
| `/reload-mcp` (별칭: `/reload_mcp`) | `config.yaml`에서 MCP 서버를 다시 불러옵니다. |
| `/reload-skills` (별칭: `/reload_skills`) | 새로 설치되거나 제거된 스킬을 확인하기 위해 `~/.hermes/skills/`를 다시 검색합니다. |
| `/reload` | 실행 중인 세션에 `.env` 변수를 다시 불러옵니다(재시작하지 않고 새 API 키를 반영합니다). |
| `/plugins` | 설치된 플러그인과 상태를 나열합니다. |
| `/pet [list\|<slug>]` | [petdex](/user-guide/features/pets) 마스코트를 전환하거나 입양합니다. `/pet`은 패널을 전환하고, `/pet list`는 설치된 펫을 표시하며, `/pet <slug>`은 특정 펫을 입양합니다. |
| `/hatch <description>` (별칭: `/generate-pet`) | 구성된 이미지 백엔드(OpenRouter / Nous Portal)를 사용해 텍스트 설명으로 완전히 새로운 petdex 펫을 생성합니다. [Pets](/user-guide/features/pets)를 참고하세요. |

### 정보

| 명령어 | 설명 |
|---------|-------------|
| `/help` | 이 도움말 메시지를 표시합니다. |
| `/version` | Hermes Agent의 버전, 빌드 및 환경 정보를 표시합니다. |
| `/whoami` | 슬래시 명령어 접근 수준(관리자 / 사용자)을 표시합니다. |
| `/usage` | 토큰 사용량, 비용 내역, 세션 기간을 표시하고, 활성 공급자가 제공하는 경우 공급자 API에서 실시간으로 가져온 남은 할당량 / 크레딧 / 플랜 사용량을 포함한 **계정 한도** 섹션을 표시합니다. |
| `/topup` | Nous 잔액을 표시하고 포털에서 결제를 관리합니다(기존 `/credits` 및 `/billing` 명령어를 대체). |
| `/subscription` (별칭: `/upgrade`) | **CLI 전용.** Nous 플랜을 확인하고 브라우저에서 변경합니다. |
| `/insights` | 사용량 인사이트와 분석을 표시합니다(최근 30일). |
| `/update` | Hermes Agent를 최신 버전으로 업데이트합니다. |
| `/platforms` (별칭: `/gateway`) | 게이트웨이/메시징 플랫폼 상태를 표시합니다(CLI 전용 요약 화면). |
| `/paste` | 클립보드 이미지를 첨부합니다. |
| `/copy [number]` | 마지막 어시스턴트 응답을 클립보드에 복사합니다(숫자를 지정하면 뒤에서 N번째 응답을 복사). CLI 전용입니다. |
| `/image <path>` | 다음 프롬프트에 사용할 로컬 이미지 파일을 첨부합니다. |
| `/debug` | 디버그 보고서(시스템 정보 + 로그)를 업로드하고 공유 가능한 링크를 받습니다. 메시징에서도 사용할 수 있습니다. |
| `/update` | Hermes Agent를 최신 버전으로 업데이트합니다. |
| `/profile` | 활성 프로필 이름과 홈 디렉터리를 표시합니다. |

### 종료

| 명령어 | 설명 |
|---------|-------------|
| `/quit` | CLI를 종료합니다(또한 `/exit`). |

### 동적 CLI 슬래시 명령어

| 명령어 | 설명 |
|---------|-------------|
| `/<skill-name>` | 설치된 스킬을 주문형 명령어로 불러옵니다. 예: `/gif-search`, `/github-pr-workflow`, `/excalidraw`. |
| `/skills ...` | 레지스트리와 공식 optional-skills 카탈로그에서 스킬을 검색, 탐색, 검사, 설치, 감사, 게시 및 구성합니다. |

### 빠른 명령어

사용자 정의 빠른 명령어는 짧은 슬래시 명령어를 셸 명령어 또는 다른 슬래시 명령어에 매핑합니다. `~/.hermes/config.yaml`에서 구성합니다.

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  deploy:
    type: exec
    command: scripts/deploy.sh
  inbox:
    type: alias
    target: /gmail unread
```

그런 다음 CLI 또는 메시징 플랫폼에서 `/status`, `/deploy`, `/inbox`를 입력합니다. 빠른 명령어는 디스패치 시점에 확인되므로 모든 기본 자동 완성/도움말 표에 표시되지 않을 수 있습니다.

문자열만 사용하는 프롬프트 단축키는 빠른 명령어로 지원되지 않습니다. 재사용할 긴 프롬프트는 스킬에 넣거나 `type: alias`를 사용해 기존 슬래시 명령어를 가리키세요.

### 사용자 지정 모델 별칭

자주 사용하는 모델에 짧은 이름을 직접 정의한 다음 CLI 또는 모든 메시징 플랫폼에서 `/model <alias>`로 접근할 수 있습니다. 별칭은 세션 전용(기본값) 스위치와 `--global` 스위치 모두에서 동일하게 작동합니다.

두 가지 설정 형식을 지원합니다.

**전체 형식** — 정확한 모델, 공급자 및 선택적 기본 URL을 고정합니다. `~/.hermes/config.yaml`에 다음을 입력합니다.

```yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
  ollama-qwen:
    model: qwen3-coder:30b
    provider: custom
    base_url: http://localhost:11434/v1
```

**축약 형식** — 하나의 문자열에 `provider/model`을 사용합니다. YAML을 편집하지 않고 셸에서 설정할 수 있습니다.

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

그런 다음 채팅에서 다음과 같이 입력합니다.

```
/model fav            # session-only
/model grok --global  # also persists current-model change to config.yaml
```

사용자 별칭은 기본 제공 짧은 이름보다 우선하므로 별칭 이름을 `sonnet`, `kimi`, `opus` 등으로 지정하면 기본 제공 이름을 가립니다. 별칭 이름은 대소문자를 구분하지 않습니다.

### 별칭 확인

명령어는 접두사 일치를 지원합니다. `/h`를 입력하면 `/help`로, `/mod`를 입력하면 `/model`로 확인됩니다. 접두사가 모호하여 여러 명령어와 일치하면 레지스트리 순서에서 가장 먼저 나오는 항목이 선택됩니다. 전체 명령어 이름과 등록된 별칭은 접두사 일치보다 항상 우선합니다.

## 메시징 슬래시 명령어

> **Slack 스레드 명령어(`!` 접두사):**
> Slack 자체는 메시지 스레드 안에서 기본 슬래시 명령어를 차단하며("/queue is not supported in threads. Sorry!"), 이를 Hermes에 전달하지 않습니다. Slack 스레드 안에서는 대신 `!` 접두사를 사용하세요(`!stop`, `!new`, `!status`). 게이트웨이는 이를 슬래시 형식과 동일하게 디스패치합니다. 스레드에서는 `@Hermes !stop`과 `@Hermes /stop`도 작동합니다. 알려진 명령어 목록과 비교하는 것은 첫 번째 토큰뿐이므로 `!nice work` 같은 메시지는 변경 없이 에이전트로 전달됩니다. 자세한 내용은 [스레드 안에서 명령어 사용](/user-guide/messaging/slack#using-commands-inside-threads-the-cmd-prefix)을 참고하세요.

메시징 게이트웨이는 Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant 및 Teams 채팅에서 다음 기본 제공 명령어를 지원합니다.

| 명령어 | 설명 |
|---------|-------------|
| `/start` | 플랫폼 프로토콜 명령어입니다. 많은 채팅 플랫폼(Telegram, Discord, …)은 사용자가 봇과의 대화를 처음 열 때 `/start`를 자동으로 보냅니다. Hermes는 에이전트 응답이나 세션 소모 없이 이 핑을 조용히 확인하므로 최초 접속 핸드셰이크가 턴을 낭비하지 않습니다. 게이트웨이에 연결할 수 있는지 확인하기 위해 직접 보낼 수도 있습니다. |
| `/new [name]` (별칭: `/reset`) | 새 세션을 시작합니다(새 세션 ID + 기록). 선택 사항인 `[name]`은 초기 세션 제목을 설정합니다. `now`, `--yes` 또는 `-y`를 덧붙이면 확인 모달을 건너뜁니다(예: `/reset now`, `/new --yes my-experiment`). |
| `/status` | 세션 정보를 표시한 뒤 로컬 **세션 요약** 블록을 표시합니다(최근 턴 수, 가장 많이 사용한 도구, 변경된 파일, 최신 프롬프트 + 응답). |
| `/stop` | 실행 중인 모든 백그라운드 프로세스를 종료하고 실행 중인 에이전트를 중단합니다. |
| `/model [provider:model]` | 모델을 표시하거나 변경합니다. 공급자 전환(`/model zai:glm-5`), 사용자 지정 엔드포인트(`/model custom:model`), 이름이 지정된 사용자 지정 공급자(`/model custom:local:qwen`), 자동 감지(`/model custom`), 사용자 정의 별칭(`/model fav`, `/model grok` — [사용자 지정 모델 별칭](#custom-model-aliases) 참고)을 지원합니다. `--global`을 사용하면 변경 사항을 config.yaml에 저장합니다. **참고:** `/model`은 이미 구성된 공급자 사이에서만 전환할 수 있습니다. 새 공급자를 추가하거나 API 키를 설정하려면 채팅 세션 외부의 터미널에서 `hermes model`을 사용하세요. **비용 참고:** 세션 중간에 모델을 전환하면 프롬프트 캐시가 재설정됩니다(캐시 키에 모델이 포함됨). 따라서 다음 메시지에서는 전체 대화를 입력 비용을 전부 지불하고 다시 읽습니다. |
| `/codex-runtime [auto\|codex_app_server\|on\|off]` | 선택 사항인 [Codex app-server 런타임](../user-guide/features/codex-app-server-runtime)을 전환합니다. `model.openai_runtime`에 저장하고 캐시된 에이전트를 제거하므로 다음 메시지에서 새 런타임을 사용합니다. 다음 세션부터 적용됩니다. |
| `/personality [name]` | 세션에 성격 오버레이를 설정합니다. `/personality none`(또는 `default` / `neutral`)은 이를 지웁니다. |
| `/fast [normal\|fast\|status]` | 빠른 모드를 전환합니다 — OpenAI Priority Processing / Anthropic Fast Mode입니다. |
| `/retry` | 마지막 메시지를 다시 시도합니다. |
| `/undo` | 마지막 대화를 제거합니다. |
| `/sethome` (별칭: `/set-home`) | 현재 채팅을 전달을 위한 플랫폼 홈 채널로 지정합니다. |
| `/compress [here [N] \| focus topic]` | 대화 컨텍스트를 수동으로 압축합니다. `/compress here [N]`은 최근 N개 대화(기본값 2)를 원문 그대로 유지하고 나머지를 요약합니다. 초점 주제를 지정하면 전체 요약에서 보존할 내용의 범위를 좁힙니다. |
| `/topic [off\|help\|session-id]` | **Telegram DM 전용.** 사용자가 관리하는 다중 세션 토픽 모드를 관리합니다. `/topic`은 모드를 활성화하거나 상태를 표시하고, `/topic off`는 비활성화하고 바인딩을 지우며, `/topic help`는 사용법을 표시하고, 토픽 안에서 `/topic <session-id>`를 사용하면 이전 세션을 복원합니다. [다중 세션 DM 모드](/user-guide/messaging/telegram#multi-session-dm-mode-topic)를 참고하세요. |
| `/title [name]` | 세션 제목을 설정하거나 표시합니다. |
| `/resume [name]` | 이전에 이름을 지정한 세션을 재개합니다. |
| `/sessions [all] [search <query>]` | 이 채팅의 이전 세션을 나열합니다. `/sessions search <query>`는 제목/ID 일치로 필터링하며(최근 활성 순), `/sessions all`은 모든 출처의 세션을 나열합니다(관리자 전용). |
| `/usage` | 토큰 사용량, 예상 비용 내역(입력/출력), 컨텍스트 창 상태, 세션 기간을 표시하고, 활성 공급자가 제공하는 경우 공급자 API에서 실시간으로 가져온 남은 할당량 / 크레딧을 포함한 **계정 한도** 섹션을 표시합니다. |
| `/topup` | Nous 잔액을 표시하고 포털에서 결제를 관리합니다. |
| `/whoami` | 슬래시 명령어 접근 수준(관리자 / 사용자)을 표시합니다. |
| `/insights [days]` | 사용량 분석을 표시합니다. |
| `/reasoning [level\|show\|hide\|full\|clamp] [--global]` | 추론 수준(`max` / `ultra`까지)을 변경하거나 추론 표시를 전환합니다(`full` / `clamp` 포함). `--global`은 설정에 저장합니다. |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | 채팅에서 음성 응답을 제어합니다. `join`/`channel`/`leave`는 Discord 음성 채널 모드를 관리합니다. |
| `/rollback [number]` | 파일 시스템 체크포인트를 나열하거나 복원합니다. |
| `/diff [staged\|all\|session] [--stat]` | 작업 디렉터리의 git 변경 사항을 표시합니다(플랫폼 메시지 제한에 맞춰 펜스로 감싸고 잘라냄). `session`은 Hermes가 변경한 모든 항목의 누적 diff를 표시하며, `--stat`은 요약만 표시합니다. |
| `/background <prompt>` | 별도의 백그라운드 세션에서 프롬프트를 실행합니다. 작업이 끝나면 결과를 같은 채팅으로 전달합니다. [메시징 백그라운드 세션](/user-guide/messaging/#background-sessions)을 참고하세요. |
| `/queue <prompt>` (별칭: `/q`) | 현재 턴을 중단하지 않고 다음 턴에 실행할 프롬프트를 대기열에 넣습니다. |
| `/steer <prompt>` | 중단 없이 다음 도구 호출 뒤에 메시지를 삽입합니다. 모델은 새 턴이 아니라 다음 반복에서 이를 가져옵니다. |
| `/goal <text>` | 턴을 거쳐 Hermes가 계속 수행할 지속 목표를 설정합니다(Ralph 루프를 Hermes식으로 구현한 것입니다). 판정 모델이 각 턴 후 완료 여부를 확인하며, 목표를 달성하지 못하면 목표를 일시 중지/삭제하거나 턴 예산(기본값 20)에 도달할 때까지 Hermes가 자동으로 계속합니다. `/goal status`, `/goal pause`, `/goal resume`, `/goal clear` 하위 명령어를 사용할 수 있습니다. 에이전트 실행 중에도 상태/일시 중지/삭제는 안전하게 수행할 수 있지만, 새 목표를 설정하려면 먼저 `/stop`을 실행해야 합니다. [지속 목표](/user-guide/features/goals)를 참고하세요. |
| `/subgoal <text>` | 활성 `/goal`에 중간에 기준을 추가합니다(`/subgoal`, `/subgoal remove <N>`, `/subgoal clear`). |
| `/heartbeat every <interval> <prompt>` (별칭: `/hb`) | 유휴 상태일 때 이 세션에 다시 진입하는 반복 프롬프트를 설정합니다. 하위 명령어는 `status`, `pause`, `resume`, `clear`입니다. Slack에서는 `/hermes heartbeat …`를 사용합니다. |
| `/refine [focus]` | 지금 메모리/스킬 자기 개선 검토를 실행하며, 선택적으로 초점 지침을 지정할 수 있습니다. Slack에서는 `/hermes refine …`를 사용합니다. |
| `/moa <prompt>` | 기본 [Mixture of Agents](/user-guide/features/mixture-of-agents) 프리셋으로 프롬프트를 한 번 실행한 다음 세션 모델을 복원합니다. |
| `/branch [name]` (별칭: `/fork`) | 현재 세션을 분기합니다(다른 경로를 탐색). |
| `/agents` (별칭: `/tasks`) | 활성 에이전트와 실행 중인 작업을 표시합니다. |
| `/sessions` | 이전 세션을 탐색하고 재개합니다. |
| `/context [all]` (별칭: `/ctx`) | 컨텍스트 창 사용량 게이지와 범주별 내역을 표시합니다(메시징에 적합한 텍스트 형식). `/context all`은 스킬/도구 세트별 비용 세부 정보를 추가합니다. |
| `/egress [status]` | Docker egress 프록시 상태를 표시합니다. |
| `/init [notes]` | 저장소를 검사해 `AGENTS.md`를 생성하거나 업데이트합니다. |
| `/learn <what to learn from>` | 설명한 내용에서 재사용 가능한 스킬을 추출합니다. |
| `/bundles` | 구성된 스킬 번들을 나열합니다(여러 스킬을 미리 불러오는 `/<name>` 별칭). |
| `/reload-skills` (별칭: `/reload_skills`) | 새로 설치되거나 제거된 스킬을 확인하기 위해 `~/.hermes/skills/`를 다시 검색합니다. |
| `/footer [on\|off\|status]` | 최종 답변의 런타임 메타데이터 푸터를 전환합니다(모델, 컨텍스트 %, cwd를 표시). |
| `/curator [status\|run\|pin\|archive]` | 백그라운드 스킬 유지 관리 제어 기능입니다. |
| `/suggestions [accept\|dismiss N\|catalog\|clear]` | 채팅에서 바로 추천 자동화를 검토합니다. `/suggestions`는 보류 중인 추천을 나열하고, `catalog`는 엄선된 시작 자동화를 추가하며, `clear`는 해결된 추천을 정리합니다. 수락한 추천은 이 채팅/스레드를 작업 전달 출처로 유지합니다. |
| `/blueprint [name] [slot=value ...]` | cron 블루프린트를 탐색하고, 안내형 슬롯 입력 대화를 시작하거나, 블루프린트 작업을 직접 생성합니다. 직접 생성한 작업은 현재 채팅/스레드로 결과를 전달합니다. |
| `/memory [pending\|approve\|reject\|approval]` | 작성 승인 게이트에 의해 보류된 메모리 기록(`memory.write_approval`)을 검토하고 채팅에서 바로 승인 또는 거부합니다. `/memory approval on\|off`로 게이트를 전환할 수 있습니다. [메모리 기록 관리](/user-guide/features/memory#controlling-memory-writes-write_approval)를 참고하세요. |
| `/skills [pending\|approve\|reject\|diff\|approval]` | 작성 승인 게이트에 의해 보류된 **스킬** 기록(`skills.write_approval`)을 검토합니다. 보류된 기록마다 한 줄 요약을 표시하며, `/skills diff <id>`는 채팅용으로 잘립니다. 전체 diff는 CLI 또는 `~/.hermes/pending/skills/<id>.json`에서 확인하세요. 게이트가 켜져 있거나 보류된 기록이 남아 있을 때만 표시됩니다. 검색/설치는 CLI 전용입니다. |
| `/kanban <action>` | 채팅에서 여러 프로필과 프로젝트의 협업 보드를 조작합니다 — CLI와 동일한 인수 인터페이스를 사용합니다. 실행 중인 에이전트 보호를 우회하므로 `/kanban unblock t_abc`, `/kanban comment t_abc "…"`, `/kanban list --mine`, `/kanban boards switch <slug>` 등을 턴 중에도 사용할 수 있습니다. `/kanban create …`는 요청을 보낸 채팅을 새 작업의 터미널 이벤트에 자동으로 구독합니다. [Kanban 슬래시 명령어](/user-guide/features/kanban#kanban-slash-command)를 참고하세요. |
| `/platform <list\|pause\|resume> [name]` | 채팅에서 실행 중인 게이트웨이 플랫폼을 직접 조작합니다. `/platform list`는 모든 어댑터와 상태(실행 중, 차단기 일시 중지, 수동 일시 중지)를 표시하고, `/platform pause <name>`은 어댑터를 언로드하지 않고 새 메시지 디스패치를 중지하며, `/platform resume <name>`은 다시 활성화하고 업스트림이 정상 상태가 되면 트립된 회로 차단기를 해제합니다. |
| `/reload-mcp` (별칭: `/reload_mcp`) | 설정에서 MCP 서버를 다시 불러옵니다. |
| `/verbose` | 도구 진행 상황 표시를 순환합니다. **메시징에서는 기본적으로 꺼져 있습니다** — `config.yaml`에서 `display.tool_progress_command: true`로 활성화하세요. |
| `/yolo` | YOLO 모드를 전환합니다 — 위험한 명령어의 모든 승인 프롬프트를 건너뜁니다. |
| `/commands [page]` | 모든 명령어와 스킬을 페이지 단위로 탐색합니다. |
| `/approve [session\|always]` | 보류 중인 위험한 명령어를 승인하고 실행합니다. `session`은 현재 세션에만 승인하고, `always`는 영구 허용 목록에 추가합니다. |
| `/deny` | 보류 중인 위험한 명령어를 거부합니다. |
| `/update` | Hermes Agent를 최신 버전으로 업데이트합니다. |
| `/restart` | 활성 실행을 정리한 뒤 게이트웨이를 정상적으로 재시작합니다. 게이트웨이가 다시 온라인 상태가 되면 요청자의 채팅/스레드로 확인 메시지를 보냅니다. |
| `/debug` | 디버그 보고서(시스템 정보 + 로그)를 업로드하고 공유 가능한 링크를 받습니다. |
| `/help` | 메시징 도움말을 표시합니다. |
| `/<skill-name>` | 이름으로 설치된 스킬을 호출합니다. |
## 참고

- `/skin`, `/snapshot`, `/export`, `/import`, `/reload`, `/tools`, `/toolsets`, `/browser`, `/config`, `/cron`, `/platforms`, `/paste`, `/image`, `/statusbar`, `/battery`, `/focus`, `/plugins`, `/busy`, `/indicator`, `/wake`, `/journey`, `/redraw`, `/clear`, `/history`, `/save`, `/copy`, `/handoff`, `/prompt`, `/pet`, `/hatch`, `/timestamps`, `/subscription`, `/quit`은 **CLI 전용** 명령입니다.
- `/skills`는 **검색/탐색/설치 시 CLI 전용**입니다. 단, 쓰기 승인 검토 하위 명령(`pending`, `approve`, `reject`, `diff`, `approval`)은 `skills.write_approval`이 켜져 있으면 메시징 플랫폼에서도 작동합니다. `/memory`는 **두 표면 모두**에서 작동합니다.
- `/verbose`는 기본적으로 **CLI 전용**이지만, `config.yaml`에서 `display.tool_progress_command: true`를 설정하면 메시징 플랫폼에서도 활성화할 수 있습니다. 활성화하면 `display.tool_progress` 모드를 순환하며 설정에 저장합니다.
- `/focus`와 `/verbose`는 하나의 억제 경로(`display.tool_progress`)를 공유하므로 서로 모순될 수 없습니다. `/focus on`은 도구 진행 표시를 `off`로 고정하고 모드를 `display.focus_saved_tool_progress` 아래에 저장하며, `/focus off`는 이를 복원합니다. 포커스가 켜진 상태에서 `/verbose`를 순환하면 모드가 다시 변경되고 포커스 배지가 지워집니다. 포커스 보기는 표시만 제어하며 대화 기록, 시스템 프롬프트 또는 모델에 전송되는 내용을 변경하지 않으므로 프롬프트 캐시에 전혀 영향을 주지 않습니다.
- `/sethome`, `/restart`, `/approve`, `/deny`, `/topic`, `/platform`, `/commands`는 **메시징 전용** 명령입니다.
- `/status`, `/egress`, `/version`, `/whoami`, `/background`, `/queue`, `/steer`, `/voice`, `/reload-mcp`, `/reload-skills`, `/rollback`, `/diff`, `/debug`, `/fast`, `/approvals`, `/footer`, `/curator`, `/kanban`, `/topup`, `/suggestions`, `/blueprint`, `/learn`, `/init`, `/sessions`, `/yolo`는 **CLI와 메시징 게이트웨이 모두**에서 작동합니다.
- `/voice join`, `/voice channel`, `/voice leave`는 Discord에서만 의미가 있습니다.
- TUI에서 `/sessions`는 현재 TUI 프로세스의 활성 세션을 표시합니다. 저장되었거나 닫힌 대화 기록에는 `/resume [name]` 또는 `hermes --tui --resume <id-or-title>`을 사용하세요.

## 파괴적 명령의 확인 프롬프트

CLI는 저장되지 않은 세션 상태를 버리는 슬래시 명령을 실행하기 전에 확인을 요청합니다. 현재 파괴적 명령 집합은 다음과 같습니다.

| 명령 | 삭제되는 항목 |
|---------|------------------|
| `/clear` | 화면을 지우고 새 세션을 시작합니다. 현재 세션 ID와 메모리 내 기록이 사라집니다. |
| `/new` / `/reset` | 새 세션을 시작합니다(새 세션 ID + 빈 기록). |
| `/undo` | 기록에서 마지막 사용자/어시스턴트 교환을 제거합니다. |
| `/exit --delete` / `/quit --delete` | 종료하고 현재 세션의 SQLite 기록과 디스크에 저장된 대화 기록을 영구적으로 삭제합니다. |

각 명령에 대해 CLI는 세 가지 선택지가 있는 모달을 엽니다. **한 번 승인**(이번에만 진행), **항상 승인**(진행하고 `approvals.destructive_slash_confirm: false`를 저장하여 이후 파괴적 명령을 확인 없이 실행), **취소**.

**인라인 건너뛰기:** 한 번의 호출에서 모달을 우회하려면 `now`, `--yes` 또는 `-y`를 덧붙이세요. 예: `/reset now`, `/new --yes my-session`, `/clear -y`, `/undo -y`. 터미널에서 모달이 올바르게 표시되지 않을 때(예: 기본 Windows PowerShell의 [issue #30768](https://github.com/NousResearch/hermes-agent/issues/30768)) 또는 CLI를 대상으로 스크립트를 작성할 때 유용합니다.

`~/.hermes/config.yaml`에서 `approvals.destructive_slash_confirm: false`를 설정하면 확인 프롬프트를 전역으로 비활성화하고, 다시 `true`로 설정하면 재활성화합니다. 자세한 내용은 [보안 — 파괴적 슬래시 명령 확인](../user-guide/security.md#dangerous-command-approval)을 참고하세요.
