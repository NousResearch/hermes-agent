---
sidebar_position: 12
sidebar_label: "기본 제공 플러그인"
title: "기본 제공 플러그인"
description: "수명 주기 훅을 통해 자동으로 실행되는 Hermes Agent 기본 제공 플러그인 — disk-cleanup 등"
---

# 기본 제공 플러그인

Hermes는 저장소에 함께 포함된 소수의 플러그인을 제공합니다. 이 플러그인은 `<repo>/plugins/<name>/` 아래에 있으며, `~/.hermes/plugins/`에 설치한 사용자 플러그인과 함께 자동으로 로드됩니다. 서드 파티 플러그인과 동일한 플러그인 인터페이스(훅, 도구, 슬래시 명령)를 사용하지만 저장소 안에서 유지 관리됩니다.

일반적인 플러그인 시스템은 [플러그인](/user-guide/features/plugins) 페이지를, 직접 작성하는 방법은 [Hermes 플러그인 빌드](/developer-guide/plugins)를 참조하세요.

## 검색 작동 방식

`PluginManager`는 다음 네 가지 소스를 순서대로 검색합니다.

1. **번들** — `<repo>/plugins/<name>/` (이 페이지에서 설명하는 대상)
2. **사용자** — `~/.hermes/plugins/<name>/`
3. **프로젝트** — `./.hermes/plugins/<name>/` (`HERMES_ENABLE_PROJECT_PLUGINS=1` 필요)
4. **Pip 엔트리 포인트** — `hermes_agent.plugins`

이름이 충돌하면 나중에 검색된 소스가 우선합니다. 즉, `disk-cleanup`이라는 사용자 플러그인이 번들 플러그인을 대체합니다.

`plugins/memory/`와 `plugins/context_engine/`은 번들 검색에서 의도적으로 제외됩니다. 이 디렉터리들은 자체 검색 경로를 사용합니다. 메모리 제공자와 컨텍스트 엔진은 `hermes memory setup` / config의 `context.engine`을 통해 구성되는 단일 선택 제공자이기 때문입니다.

## 번들 플러그인은 선택적으로 활성화됩니다

번들 플러그인은 비활성화된 상태로 제공됩니다. 검색을 통해 찾을 수 있고(`hermes plugins list`와 대화형 `hermes plugins` UI에 표시됨), 명시적으로 활성화하기 전에는 로드되지 않습니다.

```bash
hermes plugins enable disk-cleanup
```

또는 `~/.hermes/config.yaml`을 통해 활성화할 수 있습니다.

```yaml
plugins:
  enabled:
    - disk-cleanup
```

사용자 설치 플러그인과 동일한 메커니즘입니다. 번들 플러그인은 절대로 자동 활성화되지 않습니다. 새로 설치할 때도, 기존 사용자가 새 Hermes로 업그레이드할 때도 마찬가지입니다. 항상 명시적으로 선택해 활성화해야 합니다.

번들 플러그인을 다시 끄려면 다음을 실행하세요.

```bash
hermes plugins disable disk-cleanup
# or: remove it from plugins.enabled in config.yaml
```

## 현재 제공되는 플러그인

저장소는 `plugins/` 아래에 다음 번들 플러그인을 제공합니다. 모두 선택 사항이며 `hermes plugins enable <name>`으로 활성화할 수 있습니다.

| 플러그인 | 종류 | 용도 |
|---|---|---|
| `disk-cleanup` | hooks + slash command | 임시 파일을 자동으로 추적하고 세션 종료 시 정리 |
| `security-guidance` | hooks | `write_file`/`patch`에서 위험한 코드를 패턴 매칭하고 보안 경고를 추가(또는 차단) — 25개 규칙 (Anthropic의 `claude-plugins-official` 패턴을 Apache-2.0으로 포크) |
| `observability/langfuse` | hooks | Hermes의 턴 / LLM 호출 / 도구를 [Langfuse](https://langfuse.com)로 추적 |
| `observability/nemo_relay` | hooks | 관측성 이벤트(턴 / LLM 호출 / 도구)를 NVIDIA NeMo 엔드포인트로 릴레이 |
| `teams_pipeline` | standalone | Microsoft Teams 회의 파이프라인 — Graph 기반, 트랜스크립트 우선 회의 요약 |
| `spotify` | backend (7 tools) | Spotify 재생, 대기열, 검색, 재생 목록, 앨범, 라이브러리 네이티브 지원 |
| `google_meet` | standalone | Meet 통화 참여, 실시간 자막 변환, 선택적 실시간 양방향 오디오 |
| `image_gen/openai` | image backend | OpenAI `gpt-image-2` 이미지 생성 백엔드 (FAL의 대안) |
| `image_gen/openai-codex` | image backend | Codex OAuth를 통한 OpenAI 이미지 생성 |
| `image_gen/xai` | image backend | xAI `grok-2-image` 백엔드 |
| `hermes-achievements` | dashboard tab | 실제 Hermes 세션 기록에서 생성한 Steam 스타일 수집형 배지 |
| `kanban/dashboard` | dashboard tab | 멀티 에이전트 디스패처용 칸반 보드 UI — 작업, 댓글, 팬아웃, 보드 전환. [Kanban 멀티 에이전트](./kanban.md)를 참조하세요. |

메모리 제공자(`plugins/memory/*`)와 컨텍스트 엔진(`plugins/context_engine/*`)은 [메모리 제공자](./memory-providers.md)에 별도로 나열되어 있습니다. `hermes memory`와 `hermes plugins`를 통해 각각 관리됩니다. 다음에는 장시간 실행되는 두 훅 기반 플러그인의 상세 내용을 설명합니다.

### disk-cleanup

세션 중 생성된 임시 파일(테스트 스크립트, 임시 출력, cron 로그, 오래된 chrome 프로필)을 자동으로 추적하고 제거하므로, 에이전트가 도구를 호출해야 한다는 사실을 기억할 필요가 없습니다.

**작동 방식:**

| 훅 | 동작 |
|---|---|
| `post_tool_call` | `write_file` / `terminal` / `patch`가 `HERMES_HOME` 또는 `/tmp/hermes-*` 내부에 `test_*`, `tmp_*`, `*.test.*`와 일치하는 파일을 만들면 이를 `test` / `temp` / `cron-output`으로 조용히 추적합니다. |
| `on_session_end` | 해당 턴 중 테스트 파일이 자동 추적되었다면 안전한 `quick` 정리를 실행하고 한 줄 요약을 기록합니다. 그렇지 않으면 조용히 넘어갑니다. |

**삭제 규칙:**

| 범주 | 기준 | 확인 |
|---|---|---|
| `test` | 모든 세션 종료 시 | 없음 |
| `temp` | 추적 후 7일 초과 | 없음 |
| `cron-output` | 추적 후 14일 초과 | 없음 |
| `HERMES_HOME` 아래 빈 디렉터리 | 항상 | 없음 |
| `research` | 30일 초과 및 최신 10개 이후 | 항상 (deep에서만) |
| `chrome-profile` | 추적 후 14일 초과 | 항상 (deep에서만) |
| 500 MB 초과 파일 | 자동 정리하지 않음 | 항상 (deep에서만) |

**슬래시 명령** — `/disk-cleanup`은 CLI와 게이트웨이 세션에서 모두 사용할 수 있습니다.

```
/disk-cleanup status                     # breakdown + top-10 largest
/disk-cleanup dry-run                    # preview without deleting
/disk-cleanup quick                      # run safe cleanup now
/disk-cleanup deep                       # quick + list items needing confirmation
/disk-cleanup track <path> <category>    # manual tracking
/disk-cleanup forget <path>              # stop tracking (does not delete)
```

**상태** — 모든 항목은 `$HERMES_HOME/disk-cleanup/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `tracked.json` | 범주, 크기, 타임스탬프가 포함된 추적 경로 |
| `tracked.json.bak` | 위 파일의 원자적 쓰기 백업 |
| `cleanup.log` | 모든 추적 / 건너뜀 / 거부 / 삭제 작업의 추가 전용 감사 기록 |

**안전성** — 정리는 `HERMES_HOME` 또는 `/tmp/hermes-*` 아래의 경로만 대상으로 합니다. Windows 마운트(`/mnt/c/...`)는 거부됩니다. 잘 알려진 최상위 상태 디렉터리(`logs/`, `memories/`, `sessions/`, `cron/`, `cache/`, `skills/`, `plugins/`, `disk-cleanup/` 자체)는 비어 있더라도 절대 제거되지 않습니다. 따라서 첫 세션 종료 시 새 설치가 초기화되지 않습니다.

**활성화:** `hermes plugins enable disk-cleanup` (또는 `hermes plugins`에서 확인란을 선택하세요).

**다시 비활성화:** `hermes plugins disable disk-cleanup`.

### security-guidance

파일 쓰기 시 빠른 패턴 기반 보안 경고를 제공합니다. 에이전트의 `write_file` / `patch` / `skill_manage` 호출에 알려진 위험 코드 패턴인 `pickle.load`, `SafeLoader` 없는 `yaml.load`, `eval(`, `os.system`, `shell=True`인 `subprocess(...)`, JS `child_process.exec`, React `dangerouslySetInnerHTML`, 원시 `.innerHTML =` / `.outerHTML =` / `document.write`, Node `crypto.createCipher`, AES ECB 모드, TLS 검증 비활성화, XXE 취약 `xml.etree` / `minidom` 파서, SRI 없는 `<script src="//..." >`, `weights_only=True` 없는 `torch.load`, GitHub Actions `${{ github.event.* }}` 인젝션이 포함되어 있으면 플러그인이 도구 결과에 `⚠️ Security guidance` 블록을 추가합니다.

파일은 여전히 작성됩니다. 모델은 다음 턴의 도구 메시지에서 경고를 읽고 코드를 수정하거나 해당 구문이 이 컨텍스트에서 안전한 이유를 문서화할 수 있습니다. 패턴 매칭은 오탐률이 무시할 수 없는 수준이므로 기본값은 차단이 아닌 경고입니다.

**적용 범위:** 총 25개 규칙으로 안전하지 않은 역직렬화, 명령 인젝션, XSS 싱크, 암호화 실수, XXE, 공급망(SRI), CI/CD 워크플로 인젝션을 다룹니다. 패턴 데이터는 [Anthropic의 `claude-plugins-official`](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/security-guidance/hooks)을 Apache-2.0으로 그대로 포크한 것이며, 저작자 표시는 플러그인의 `LICENSE`와 `NOTICE` 파일을 참조하세요.

**모드:**

| 환경 변수 | 효과 |
|---|---|
| (설정되지 않음) | **warn mode** (기본값) — 파일을 작성하고 결과에 경고를 추가 |
| `SECURITY_GUIDANCE_BLOCK=1` | **block mode** — 쓰기를 거부하고 경고를 차단 사유로 반환 |
| `SECURITY_GUIDANCE_DISABLE=1` | 킬 스위치 — 플러그인은 로드되지만 아무 작업도 하지 않음 |

**활성화:** `hermes plugins enable security-guidance` (또는 `hermes plugins`에서 확인란을 선택하세요).

**다시 비활성화:** `hermes plugins disable security-guidance`.

**아직 수행하지 않는 작업:** 업스트림 Anthropic 플러그인에는 두 가지 계층이 더 있습니다. 파일을 수정한 각 에이전트 턴의 LLM diff 검토와 파일 간 데이터 흐름을 추적하는 에이전트 기반 커밋 시점 검토입니다. 둘 다 아직 포팅되지 않았습니다. 에이전트는 이미 `delegate_task`를 통해 이러한 검토를 필요할 때 실행할 수 있습니다.

### observability/langfuse

[Langfuse](https://langfuse.com) — 오픈 소스 LLM 관측성 플랫폼 — 로 Hermes 턴, LLM 호출, 도구 호출을 추적합니다. 턴마다 하나의 span, API 호출마다 하나의 generation, 도구 호출마다 하나의 tool observation을 만듭니다. 사용량 합계, 유형별 토큰 수, 비용 추정치는 Hermes의 표준 `agent.usage_pricing` 수치에서 나오므로 Langfuse 대시보드에는 `hermes logs`에 표시되는 것과 동일한 분류(입력 / 출력 / `cache_read_input_tokens` / `cache_creation_input_tokens` / `reasoning_tokens`)가 표시됩니다.

플러그인은 fail-open 방식입니다. SDK가 설치되지 않았거나, 인증 정보가 없거나, 일시적인 Langfuse 오류가 발생해도 훅에서는 모두 조용한 no-op으로 처리됩니다. 에이전트 루프에는 절대 영향을 주지 않습니다.

**설정(대화형 — 권장):**

```bash
hermes tools          # → Langfuse Observability → Cloud or Self-Hosted
```

마법사가 키를 수집하고 `langfuse` SDK를 `pip install`하며 `observability/langfuse`를 `plugins.enabled`에 추가합니다. Hermes를 다시 시작하면 다음 턴부터 trace가 전송됩니다.

**설정(수동):**

```bash
pip install langfuse
hermes plugins enable observability/langfuse
```

그런 다음 인증 정보를 `~/.hermes/.env`에 넣습니다.

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

**작동 방식:**

| 훅 | 동작 |
|---|---|
| `pre_api_request` / `pre_llm_call` | 턴별 루트 span "Hermes turn"을 열거나 재사용합니다. 직렬화한 최근 메시지를 입력으로 사용해 이 API 호출의 `generation` 하위 observation을 시작합니다. |
| `post_api_request` / `post_llm_call` | generation을 닫고 `usage_details`, `cost_details`, `finish_reason`, 어시스턴트 출력 및 도구 호출을 연결합니다. 도구 호출이 없고 콘텐츠가 비어 있지 않으면 턴을 닫습니다. |
| `pre_tool_call` | 정제된 `args`와 함께 `tool` 하위 observation을 시작합니다. |
| `post_tool_call` | 정제된 `result`와 함께 도구 observation을 닫습니다. `read_file` 페이로드는 요약(head + tail + 생략된 줄 수)되어 큰 파일을 읽어도 `HERMES_LANGFUSE_MAX_CHARS` 아래로 유지됩니다. |

세션 그룹화는 `langfuse.propagate_attributes`를 통해 Hermes 세션 ID(또는 서브 에이전트의 경우 task ID)를 기준으로 합니다. 따라서 하나의 `hermes chat` 세션에 속한 모든 항목은 하나의 Langfuse 세션 아래에 저장됩니다.

**확인:**

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # check the Langfuse UI for a "Hermes turn" trace
```

**선택적 조정** (`.env`에서):

| 변수 | 기본값 | 용도 |
|---|---|---|
| `HERMES_LANGFUSE_ENV` | — | trace의 환경 태그(`production`, `staging`, …) |
| `HERMES_LANGFUSE_RELEASE` | — | 릴리스/버전 태그 |
| `HERMES_LANGFUSE_SAMPLE_RATE` | `1.0` | SDK에 전달할 샘플링 비율(0.0–1.0) |
| `HERMES_LANGFUSE_MAX_CHARS` | `12000` | 메시지 콘텐츠 / 도구 인수 / 도구 결과의 필드별 잘라내기 길이 |
| `HERMES_LANGFUSE_DEBUG` | `false` | `agent.log`에 플러그인 로그를 자세히 기록 |

Hermes 접두사가 붙은 SDK 표준 환경 변수(`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`)도 모두 허용됩니다. 둘 다 설정된 경우 Hermes 접두사 버전이 우선합니다.

**성능:** Langfuse 클라이언트는 첫 훅 호출 이후 캐시됩니다. 인증 정보나 SDK가 없는 경우 그 판단 역시 캐시되므로 이후 훅은 환경 변수를 다시 확인하거나 설정을 다시 로드하지 않고 빠르게 반환됩니다.

**비활성화:** `hermes plugins disable observability/langfuse`. 플러그인 모듈은 계속 검색되지만 다시 활성화할 때까지 모듈 코드는 실행되지 않습니다.

### observability/nemo_relay

Hermes 실행 경계(세션, 턴, LLM 호출, 도구 호출)를 [NVIDIA NeMo Relay](https://docs.nvidia.com/nemo/relay/about-nemo-relay/overview) 엔드포인트로 릴레이합니다. Hermes 코어가 Relay 세션/턴/LLM/도구 범위를 관리하고, 플러그인은 내보내기 도구(ATOF JSONL, ATIF trajectories, OpenTelemetry)를 구성하며 승인 및 위임된 서브 에이전트용 observer mark를 추가합니다. 전체 내보내기 설정은 `plugins/observability/nemo_relay/`의 플러그인 `README.md`에 있습니다.

**활성화:**

```bash
hermes plugins enable observability/nemo_relay
```

#### 세션 span 분할(연속 세션)

Relay 내보내기는 close-driven 방식입니다. scope가 pop될 때 span이 내보내집니다. 일반적인 Telegram/Slack 에이전트처럼 연속 게이트웨이 세션은 세션 scope를 며칠 또는 몇 주 동안 열린 상태로 유지하므로, 세션 루트 span과 여기에 연결된 mark는 `/new` 또는 유휴 종료 시점까지 내보내지지 않습니다. 또한 충돌이나 재배포가 발생하면 열려 있던 전체 세그먼트를 잃습니다. 턴 span은 턴마다 이미 내보내지므로 영향을 받지 않습니다.

선택적 분할을 사용하면 `config.yaml`에서 턴 경계마다 세션 scope를 순환시킬 수 있습니다.

```yaml
gateway:
  telemetry:
    session_segments:
      on_compaction: false   # rotate the session scope when the session compacts
      max_turns: 0           # 0 = unlimited; N = rotate after N turns per segment
```

| 키 | 기본값 | 동작 |
|---|---|---|
| `on_compaction` | `false` | 컨텍스트 압축이 완료된 후 세션 scope를 닫고 다시 엽니다(다음 턴 경계에서, 절대 턴 중간에는 실행하지 않음). |
| `max_turns` | `0` | 세그먼트 안에서 N턴마다 순환합니다. `0`은 제한을 비활성화합니다. |

두 기본값은 모두 꺼져 있습니다. 설정하지 않으면 scope 수명 주기는 이전 릴리스와 동일합니다(세션 수명 동안 하나의 세션 scope).

순환된 세그먼트는 동일한 `session_id` 속성을 유지하고 `hermes.session.segment`(0부터 시작하는 인덱스)와 `hermes.session.segment_reason`(`compaction` 또는 `max_turns`)를 추가하므로, `session_id`로 그룹화하는 대시보드는 영향을 받지 않습니다. 순환은 턴 경계에서만 발생하며 다른 모든 네이티브 Relay 호출과 동일한 제한된 scope-op 실행기를 사용합니다. 멈춘 내보내기는 에이전트가 아니라 하나의 세그먼트 span에만 비용을 발생시킵니다.

**비활성화:** `session_segments` 블록을 제거하거나 두 키를 기본값으로 되돌리세요.

### google_meet

에이전트가 **Google Meet 통화에 참여하고, 기록하고, 대화에 참여**할 수 있게 합니다. 회의 메모 작성, 통화 후 대화 요약, 특정 요점에 대한 후속 조치, 그리고 (선택적으로) TTS를 통한 통화 내 답변 발화를 지원합니다.

**추가되는 기능:**

- 브라우저 자동화를 사용해 Meet URL에 참여하는 헤드리스 가상 참가자
- 구성된 STT 제공자를 통한 회의 오디오 실시간 기록
- 통화에 참여하고, 실시간 트랜스크립트를 조회하며, 들은 내용에 따라 행동하도록 에이전트가 호출하는 `meet_join` / `meet_status` / `meet_transcript` / `meet_leave` / `meet_say` 도구 세트
- `~/.hermes/workspace/meetings/<meeting_id>/` 아래에 저장되는 회의 후 아티팩트(트랜스크립트, 상태)

**설정:**

```bash
hermes plugins enable google_meet
hermes meet setup   # preflight: playwright, chromium, auth file
hermes meet auth    # opens a browser to sign into Google and saves session state —
                    # needs a Google account with Meet access. Host approval may be
                    # required if the meeting enforces "only invited participants can join".
```

**채팅에서 사용:**

> "meet.google.com/abc-defg-hij에 참여해서 메모해 줘. 통화가 끝나면 실행 항목이 포함된 요약을 보내 줘."

에이전트는 회의 참여를 시작하고, 통화가 진행되는 동안 트랜스크립트를 컨텍스트로 스트리밍하며, 회의가 끝나거나 중지하라고 말하면 구조화된 요약을 생성합니다.

**사용할 때:** 비동기 참석자를 위해 봇이 기록과 요약을 해 주길 원하는 정기 스탠드업, 구조화된 메모가 필요한 증언 형식의 인터뷰, Fireflies / Otter / Grain이 필요했던 모든 경우에 유용합니다. AI가 듣는 것을 원하지 않는다면 활성화하지 마세요.

**비활성화:** `hermes plugins disable google_meet`. 저장된 트랜스크립트는 직접 제거할 때까지 `~/.hermes/workspace/meetings/`에 남아 있습니다.

### hermes-achievements

**대시보드에 Steam 스타일 업적 탭을 추가**합니다. 실제 Hermes 세션 기록에서 생성된 60개 이상의 수집형 티어 배지를 제공합니다. 도구 체인 성과, 디버깅 패턴, 바이브 코딩 연속 기록, 스킬/메모리 사용, 모델/제공자 다양성, 주말 및 야간 세션 같은 생활 패턴을 포함합니다. 원래 [@PCinkusz](https://github.com/PCinkusz)가 외부 플러그인으로 작성했으며, Hermes 기능 변경과 보조를 맞출 수 있도록 저장소에 포함되었습니다.

**작동 방식:**

- 대시보드 백엔드에서 전체 `~/.hermes/state.db` 세션 기록을 스캔합니다.
- 세션별 통계를 `(started_at, last_active)` fingerprint로 캐시하므로, 이후 스캔에서는 새 세션이나 변경된 세션만 다시 분석합니다.
- 최초 스캔은 백그라운드 스레드에서 실행되므로 수천 개의 세션이 있는 데이터베이스에서도 대시보드가 이를 기다리며 멈추지 않습니다.
- 해제 상태는 `$HERMES_HOME/plugins/hermes-achievements/state.json`에 저장됩니다.

**티어 진행:** Copper → Silver → Gold → Diamond → Olympian. 각 카드에는 추적 중인 정확한 지표를 나열하는 "집계 기준" 섹션이 표시됩니다.

**업적 상태:**

| 상태 | 의미 |
|---|---|
| Unlocked | 하나 이상의 티어를 달성함 |
| Discovered | 알려진 업적이며 진행 상황이 보이지만 아직 획득하지 않음 |
| Secret | Hermes가 기록에서 관련 신호를 처음 감지할 때까지 숨겨짐 |

**API** — 경로는 `/api/plugins/hermes-achievements/` 아래에 마운트됩니다.

| 엔드포인트 | 용도 |
|---|---|
| `GET /achievements` | 배지별 해제 상태가 포함된 전체 카탈로그(최초 콜드 스캔 중에는 대기 중 자리 표시자 반환) |
| `GET /scan-status` | 백그라운드 스캐너 상태: `idle` / `running` / `failed`, 마지막 소요 시간, 실행 횟수 |
| `GET /recent-unlocks` | 최근 해제된 배지 20개(최신순) |
| `GET /sessions/{id}/badges` | 특정 한 세션에서 주로 획득한 배지 |
| `POST /rescan` | 수동 동기 재스캔(차단됨; 사용자가 재스캔 버튼을 클릭할 때 사용) |
| `POST /reset-state` | 해제 기록과 캐시된 스냅샷 삭제 |

**상태 파일** — `$HERMES_HOME/plugins/hermes-achievements/` 아래에 있습니다.

| 파일 | 내용 |
|---|---|
| `state.json` | 해제 기록: 획득한 배지와 획득 시점. Hermes 업데이트 후에도 유지됩니다. |
| `scan_snapshot.json` | 마지막으로 완료된 스캔 페이로드(대시보드 로드 시 즉시 제공) |
| `scan_checkpoint.json` | fingerprint를 키로 사용하는 세션별 통계 캐시(웜 재스캔을 빠르게 함) |

**성능 참고:**

- 약 8,000개 세션의 콜드 스캔에는 몇 분이 걸립니다. 최초 대시보드 요청 시 백그라운드 스레드에서 실행되며, UI에는 대기 중 자리 표시자가 표시되고 `/scan-status`를 폴링합니다.
- **콜드 스캔 중 증분 결과** — 스캐너는 약 250개 세션마다 부분 스냅샷을 게시하므로 대시보드를 새로 고칠 때마다 스캔 진행에 따라 더 많은 배지가 해제된 것으로 표시됩니다. 0만 표시된 화면을 몇 분 동안 바라볼 필요가 없습니다.
- 웜 재스캔은 checkpoint와 `started_at` + `last_active` fingerprint가 일치하는 모든 세션에 대해 세션별 통계를 재사용하므로, 기록이 커도 몇 초 안에 완료됩니다.
- 메모리 내 스냅샷 TTL은 120초입니다. 오래된 요청은 이전 스냅샷을 즉시 제공하고 백그라운드 새로 고침을 시작합니다. TTL이 만료되었다는 이유만으로 스피너를 기다릴 필요가 없습니다.

**활성화:** 활성화할 필요가 없습니다. `hermes-achievements`는 대시보드 전용 플러그인(수명 주기 훅과 모델에 표시되는 도구 없음)이며, 최초 실행 시 `hermes dashboard`의 탭으로 자동 등록됩니다. `plugins.enabled` 설정은 수명 주기/도구 플러그인만 제어하고, 대시보드 플러그인은 `dashboard/manifest.json`을 통해서만 검색됩니다.

**선택 해제:** `plugins/hermes-achievements/dashboard/manifest.json`을 삭제하거나 이름을 바꾸세요. 또는 `~/.hermes/plugins/hermes-achievements/`에 같은 이름으로 대시보드가 없는 사용자 플러그인을 만들어 이를 덮어쓸 수 있습니다. `$HERMES_HOME/plugins/hermes-achievements/` 아래의 플러그인 상태 파일은 유지되므로 재설치해도 해제 기록이 보존됩니다.

## 번들 플러그인 추가

번들 플러그인은 다른 Hermes 플러그인과 정확히 같은 방식으로 작성합니다. [Hermes 플러그인 빌드](/developer-guide/plugins)를 참조하세요. 차이점은 다음과 같습니다.

- 디렉터리는 `~/.hermes/plugins/<name>/` 대신 `<repo>/plugins/<name>/`에 있습니다.
- 매니페스트 소스는 `hermes plugins list`에서 `bundled`로 표시됩니다.
- 이름이 같은 사용자 플러그인이 번들 버전을 재정의합니다.

다음 조건을 만족하는 플러그인은 번들로 제공하기에 적합합니다.

- 선택적 의존성이 없거나(`pip install .[all]` 의존성에 이미 포함된 경우)
- 대부분의 사용자에게 유용하고 선택 해제 방식이 선택 방식보다 적합한 경우
- 에이전트가 호출해야 한다는 사실을 기억하지 않아도 되는 수명 주기 훅과 연결되는 로직인 경우
- 모델에 표시되는 도구 표면을 확장하지 않고 핵심 기능을 보완하는 경우

반대로 API 키가 필요한 서드 파티 통합, 틈새 워크플로, 대규모 의존성 트리, 기본적으로 에이전트 동작을 의미 있게 바꾸는 기능은 번들이 아니라 사용자가 설치하는 플러그인으로 유지해야 합니다.
