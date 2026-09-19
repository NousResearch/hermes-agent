---
title: "Kanban Video Orchestrator — 다중 에이전트 영상 제작 파이프라인을 계획하고 실행하기"
sidebar_label: "Kanban Video Orchestrator"
description: "다중 에이전트 영상 제작 파이프라인을 계획하고 실행하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Kanban Video Orchestrator

다중 에이전트 영상 제작 파이프라인을 계획하고 실행합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/kanban-video-orchestrator`로 설치 |
| 경로 | `optional-skills/creative/kanban-video-orchestrator` |
| 버전 | `1.0.0` |
| 작성자 | ['SHL0MS', 'alt-glitch'] |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `video`, `kanban`, `multi-agent`, `orchestration`, `production-pipeline` |
| 관련 스킬 | [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`comfyui`](/docs/user-guide/skills/bundled/creative/creative-comfyui), [`touchdesigner-mcp`](/docs/user-guide/skills/bundled/creative/creative-touchdesigner-mcp), [`pixel-art`](/docs/user-guide/skills/optional/creative/creative-pixel-art), [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art), [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music), [`heartmula`](/docs/user-guide/skills/optional/creative/creative-heartmula), [`songsee`](/docs/user-guide/skills/bundled/media/media-songsee), [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram), [`concept-diagrams`](/docs/user-guide/skills/optional/creative/creative-concept-diagrams), [`baoyu-comic`](/docs/user-guide/skills/optional/creative/creative-baoyu-comic), [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic), [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer), [`gif-search`](/docs/user-guide/skills/bundled/media/media-gif-search), [`meme-generation`](/docs/user-guide/skills/optional/creative/creative-meme-generation) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 활성화된 스킬이 있을 때 에이전트가 보게 되는 내용입니다.
:::

# Kanban Video Orchestrator

15초 제품 티저부터 5분 내러티브 단편, 뮤직비디오, ASCII 루프까지 모든 영상 요청을 Hermes Kanban 파이프라인으로 감싸고, 작업을 전문 에이전트 프로필에 분해합니다.

이 스킬은 자체적으로 아무것도 렌더링하지 않습니다. 대신 다음을 수행하는 메타 파이프라인입니다.

1. **범위를 정합니다** — 목적에 맞춘 탐색을 통해
2. **적절한 팀을 설계합니다** — 스타일에 따라 역할과 각 역할의 도구를 결정
3. **설정 스크립트를 생성합니다** — Hermes 프로필, 프로젝트 작업 공간, 초기 kanban 작업을 생성
4. **디렉터 프로필에 넘깁니다** — 디렉터가 kanban을 통해 작업을 분해
5. **실행을 모니터링합니다** — 작업이 멈추거나 실패할 때 개입을 지원

실제 렌더링은 실행 중인 kanban 내부에서, 장면에 맞는 기존 스킬과 도구를 사용해 이루어집니다 — `ascii-video`, `manim-video`, `p5js`, `comfyui`, `touchdesigner-mcp`, `songwriting-and-ai-music`, `heartmula`, 외부 API 또는 PIL + ffmpeg를 사용하는 일반 Python.

## 이 스킬을 사용하지 않을 때

- 전문 담당자가 필요 없는 하나의 연속적인 절차형 프로젝트인 경우. 코드를 직접 작성하세요.
- 사용자가 빠른 일회성 변환(예: "이 mp4를 GIF로 변환해 줘")을 원하는 경우 — ffmpeg를 직접 사용하세요.
- 결과물이 정적 이미지, GIF 또는 오디오만인 경우 — 이에 맞는 특정 스킬(`ascii-art`, `gifs`, `meme-generation`, `songwriting-and-ai-music`)을 사용하세요.
- 작업이 하나의 기존 스킬에 깔끔하게 맞는 경우(예: 순수 ASCII 영상 — `ascii-video`만 사용).

## 워크플로

```
DISCOVER  →  BRIEF  →  TEAM DESIGN  →  SETUP  →  EXECUTE  →  MONITOR
```

### 1단계 — 발견(올바른 질문을 하기)

발견 과정은 **적응형**입니다. 실제로 필요한 것만 질문하세요. 전체적인 형태를 파악하기 위해 항상 다음 세 질문으로 시작합니다.

- **영상은 무엇인가요?** (한 문장으로 된 개요)
- **길이는 얼마인가요?** (5-30초 티저 / 30-90초 단편 / 90초-3분 설명 영상 / 3-10분 영화 / 그 이상)
- **화면 비율과 대상 플랫폼은 무엇인가요?** (1:1 / 9:16 / 16:9; X, IG, YouTube, 내부용 등)

답변을 바탕으로 스타일 범주를 분류합니다. 스타일에 따라 어떤 후속 질문을 할지가 결정됩니다. **모든 질문을 한꺼번에 하지 마세요.** 한 번에 2-4개씩 질문하고, 답변을 들은 뒤 진행합니다. 사용자가 암시한 답은 합리적으로 가정하세요.

전체 입력 패턴과 스타일별 질문 목록은 **[references/intake.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/intake.md)**를 참고하세요.

### 2단계 — 브리프

충분한 정보를 얻으면 `assets/brief.md.tmpl` 템플릿을 사용해 구조화된 `brief.md`를 작성합니다. 단계는 다음과 같습니다.

1. **콘셉트** — 한 문장 피치 + 감정적 핵심
2. **범위** — 길이, 화면 비율, 플랫폼, 마감일
3. **스타일** — 시각적 레퍼런스, 브랜드 제약, 분위기
4. **장면** — 박자별 분석(길이, 내용, 대상 도구)
5. **오디오** — 내레이션 / 음악 / SFX / 무음(필요한 경우 장면별)
6. **납품물** — 파일 형식, 해상도, 선택적 대체본(세로 버전, GIF 등)

팀을 설계하기 전에 브리프를 사용자에게 보여 주고 확인받습니다. **브리프가 계약입니다** — 이후의 모든 작업은 브리프를 참조합니다.

### 3단계 — 팀 설계

이 영상에 맞는 역할 라이브러리에서 역할 원형을 선택합니다. **복제하지 말고 조합하세요.** 대부분의 영상에는 4-7개의 프로필이 필요합니다. 디렉터는 항상 포함하며, 나머지는 브리프에 실제로 필요한 역할을 선택합니다.

역할 라이브러리와 스타일별 팀 구성은 **[references/role-archetypes.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/role-archetypes.md)**를 참고하세요.

역할 → 로드할 Hermes 스킬과 도구 세트 매핑은 **[references/tool-matrix.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/tool-matrix.md)**를 참고하세요.

### 4단계 — 설정

설정 스크립트(`setup.sh`)를 생성하고 실행합니다. 스크립트는 다음을 수행합니다.

1. 프로젝트 작업 공간(`~/projects/video-pipeline/<slug>/`)을 생성
2. 제공된 자산을 `taste/`, `audio/`, `assets/`에 복사
3. `hermes profile create --clone`으로 각 Hermes 프로필을 생성
4. 프로필별 `SOUL.md`(성격 + 역할 정의)를 작성
5. 프로필 YAML(toolsets, always_load 스킬, cwd)을 구성
6. `brief.md`, `TEAM.md`, `taste/` 콘텐츠를 작성
7. 디렉터에게 할당된 초기 `hermes kanban create` 작업을 실행

브리프와 팀 설계 JSON에서 setup.sh를 생성하려면 `scripts/bootstrap_pipeline.py`를 사용하세요. 설정 스크립트의 구조, 프로필 설정 패턴, 그리고 중요한 "공유 작업 공간" 규칙은 **[references/kanban-setup.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/kanban-setup.md)**를 참고하세요.

### 5단계 — 실행

`setup.sh`를 실행합니다. 그런 다음 사용자에게 모니터링 명령을 제공합니다.

```bash
hermes kanban watch --tenant <project-tenant>     # live events
hermes kanban list  --tenant <project-tenant>     # board snapshot
hermes dashboard                                   # visual board UI
```

이제 디렉터 프로필이 작업을 분해하고 kanban 도구 세트를 통해 전문 프로필에 작업을 라우팅합니다.

### 6단계 — 모니터링 및 개입

계속 관여하세요 — kanban은 자율적으로 실행되지만, 멈춘 작업이나 잘못된 결과에는 사람(또는 AI)의 판단이 필요합니다.

모니터링 패턴: `kanban list`를 주기적으로 폴링하고, 예상 시간을 초과한 RUNNING 작업이 있으면 `kanban show <id>`로 검사하며, 하트비트를 확인합니다. 작업자의 결과가 검토에 실패했을 때 표준 개입 방법은 다음과 같습니다.

1. 구체적인 피드백을 작업자의 작업에 댓글로 남기기(`kanban_comment`)
2. 원래 작업을 부모로 하는 재실행 작업 생성
3. 브리프의 범위를 조정하고 디렉터가 다시 분해하도록 하기

진단 패턴, 개입 레시피, 그리고 "작업이 멈췄을 때" 플레이북은 **[references/monitoring.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/monitoring.md)**를 참고하세요.

## 참고: 작업 예시

서로 매우 다른 영상 스타일을 다루는 여섯 가지 구체적인 파이프라인 — 내러티브 영화, 제품/마케팅, 뮤직비디오, 수학/알고리즘 설명 영상, ASCII 영상, 실시간 설치 작품 — 을 통해 동일한 워크플로가 어떻게 서로 다른 팀과 작업 그래프를 만들어 내는지 보여 줍니다. **[references/examples.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/examples.md)**를 참고하세요.

## 핵심 규칙

1. **행동 전에 발견.** 최소 세 가지 기본 질문을 묻기 전에는 절대로 브리프나 팀 생성을 시작하지 않습니다. 잘못된 브리프는 파이프라인 전체로 연쇄됩니다.

2. **영상에 맞게 팀을 구성.** 모든 작업에 동일한 4개 프로필 구성을 재사용하지 마세요. 비트 분석 프로필이 없는 뮤직비디오는 제대로 작동하지 않습니다. 작가 프로필이 없는 내러티브 영화는 일관성 없는 장면을 만들어 냅니다. `references/role-archetypes.md`를 참고하세요.

3. **프로젝트마다 하나의 작업 공간.** 주어진 영상의 모든 프로필은 동일한 `dir:` 작업 공간을 공유합니다. 작업은 공유 파일 시스템과 구조화된 인계로 아티팩트를 전달합니다. **모든** `kanban_create` 호출에는 `workspace_kind="dir"` + `workspace_path="<absolute project path>"`를 전달합니다.

4. **모든 프로젝트에 테넌트 지정.** 프로젝트별 테넌트(`--tenant <project-slug>`)를 사용합니다. 대시보드의 범위를 유지하고 다른 진행 중인 kanban과 섞이지 않게 합니다.

5. **기존 스킬을 존중.** 장면이 기존 스킬에 해당하면 관련 렌더러가 작업에서 `--skill <name>`을 사용하거나 프로필의 `always_load`를 통해 해당 스킬을 로드해야 합니다. 스킬이 이미 제공하는 것을 다시 만들지 마세요.

6. **디렉터는 절대 실행하지 않습니다.** 전체 `kanban + terminal + file` 도구 세트를 사용하더라도 디렉터의 `SOUL.md` 규칙은 디렉터가 직접 작업을 실행하지 못하도록 합니다. 디렉터는 분해하고 라우팅만 합니다 — 모든 구체적인 작업은 전문 프로필을 대상으로 하는 `hermes kanban create` 호출이 됩니다. 모든 kanban 작업자의 시스템 프롬프트에 자동 주입되는 kanban 오케스트레이션 지침이 이를 더 자세히 설명합니다.

7. **과도하게 분해하지 마세요.** 30초짜리 제품 영상에 작업 20개가 필요하지는 않습니다. 병렬화가 잘되고 적절한 사람 검토 게이트가 드러나는 최소 작업 그래프를 목표로 합니다.

8. **실행 전에 API 키를 확인.** 외부 API(TTS, 이미지 생성, 이미지-영상 변환)는 `${HERMES_HOME:-~/.hermes}/.env` 또는 사용자의 비밀 저장소에 키가 필요합니다. 키가 없는 작업자는 작업 슬롯을 낭비합니다. 설정 스크립트의 `check_key` 도우미는 필수 키가 없으면 정상적으로 중단합니다.

## 파일 구성

```
SKILL.md                            ← this file (workflow + rules)
references/
  intake.md                         ← discovery question banks per style
  role-archetypes.md                ← role library (writer, designer, animator, …)
  tool-matrix.md                    ← skill + toolset mapping per role
  kanban-setup.md                   ← setup script structure & profile config
  monitoring.md                     ← watch + intervene patterns
  examples.md                       ← six worked pipelines
assets/
  brief.md.tmpl                     ← brief skeleton
  setup.sh.tmpl                     ← setup script skeleton
  soul.md.tmpl                      ← profile personality skeleton
scripts/
  bootstrap_pipeline.py             ← generate setup.sh from brief + team JSON
  monitor.py                        ← polling + intervention helpers
```
