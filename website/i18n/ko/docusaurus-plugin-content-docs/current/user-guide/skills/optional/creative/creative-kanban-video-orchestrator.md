---
title: "Kanban Video Orchestrator — Hermes Kanban을 기반으로 한 다중 에이전트 비디오 제작 파이프라인의 계획, 설정 및 모니터링"
sidebar_label: "Kanban Video Orchestrator"
description: "Hermes Kanban을 기반으로 한 다중 에이전트 비디오 제작 파이프라인의 계획, 설정 및 모니터링"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Video Orchestrator

Hermes Kanban이 지원하는 다중 에이전트 비디오 제작 파이프라인을 계획하고, 설정하고, 모니터링합니다. 사용자가 모든 종류의 비디오(서사 영화, 제품/마케팅 영상, 뮤직비디오, 설명 영상, ASCII/터미널 아트, 추상/생성형 루프, 만화, 3D, 실시간/설치물)를 만들고자 할 때, 그리고 그 작업이 칸반 보드를 통해 조정되는 전문화된 프로필(작가, 디자이너, 애니메이터, 렌더러, 성우, 편집자 등)로 분해될 가치가 있을 때 사용합니다. 적응형 디스커버리(adaptive discovery)를 수행하여 브리프(brief)의 범위를 정하고, 요청된 스타일에 적절한 팀을 설계하며, Hermes 프로필과 초기 칸반 작업을 생성하는 설정 스크립트를 생성한 다음, 작업이 정체되거나 실패할 때 실행을 모니터링하고 개입하도록 돕습니다. 각 비트에 맞는 Hermes 렌더링 / 오디오 / 디자인 스킬(`ascii-video`, `manim-video`, `p5js`, `comfyui`, `touchdesigner-mcp`, `blender-mcp`, `pixel-art`, `baoyu-comic`, `claude-design`, `excalidraw`, `songsee`, `heartmula` 등)과 필요에 따라 TTS, 이미지 생성 및 이미지-비디오 변환을 위한 외부 API로 장면을 라우팅합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/kanban-video-orchestrator`로 설치 |
| Path | `optional-skills/creative/kanban-video-orchestrator` |
| Version | `1.0.0` |
| Author | ['SHL0MS', 'alt-glitch'] |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `video`, `kanban`, `multi-agent`, `orchestration`, `production-pipeline` |
| Related skills | [`kanban-orchestrator`](/docs/user-guide/skills/bundled/devops/devops-kanban-orchestrator), [`kanban-worker`](/docs/user-guide/skills/bundled/devops/devops-kanban-worker), [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js), [`comfyui`](/docs/user-guide/skills/bundled/creative/creative-comfyui), [`touchdesigner-mcp`](/docs/user-guide/skills/bundled/creative/creative-touchdesigner-mcp), [`blender-mcp`](/docs/user-guide/skills/optional/creative/creative-blender-mcp), [`pixel-art`](/docs/user-guide/skills/bundled/creative/creative-pixel-art), [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art), [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music), [`heartmula`](/docs/user-guide/skills/bundled/media/media-heartmula), [`songsee`](/docs/user-guide/skills/bundled/media/media-songsee), [`spotify`](/docs/user-guide/skills/bundled/media/media-spotify), [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content), [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw), [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram), [`concept-diagrams`](/docs/user-guide/skills/optional/creative/creative-concept-diagrams), [`baoyu-comic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-comic), [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic), [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer), [`gif-search`](/docs/user-guide/skills/bundled/media/media-gif-search), [`meme-generation`](/docs/user-guide/skills/optional/creative/creative-meme-generation) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Kanban Video Orchestrator

15초짜리 제품 티저부터 5분짜리 단편 서사, 뮤직비디오, ASCII 루프에 이르기까지 모든 비디오 요청을 Hermes Kanban 파이프라인으로 감싸서 작업을 전문 에이전트 프로필로 분해합니다.

이 스킬은 직접 아무것도 렌더링하지 **않습니다**. 다음을 수행하는 메타 파이프라인입니다:

1. 타겟팅된 디스커버리(discovery)를 통해 요청의 **범위를 지정(Scope)**합니다.
2. 스타일에 기반하여 적절한 팀(어떤 역할, 역할당 어떤 도구)을 **설계(Design)**합니다.
3. Hermes 프로필, 프로젝트 작업 공간 및 초기 칸반 작업을 생성하는 설정 스크립트를 **생성(Generate)**합니다.
4. 칸반을 통해 작업을 분해하는 디렉터(director) 프로필로 **인계(Hand off)**합니다.
5. 실행을 **모니터링(Monitor)**하고 작업이 정체되거나 실패할 때 개입을 돕습니다.

실제 렌더링은 칸반이 실행되면 해당 장면에 맞는 기존 스킬 + 도구(`ascii-video`, `manim-video`, `p5js`, `comfyui`, `touchdesigner-mcp`, `blender-mcp`, `songwriting-and-ai-music`, `heartmula`, 외부 API 또는 PIL + ffmpeg를 사용한 순수 Python 등)를 통해 내부에서 발생합니다.

## 이 스킬을 사용하지 말아야 할 때

- 비디오가 전문가가 필요 없는, 절차적으로 생성되는 하나의 연속적인 프로젝트인 경우. 그냥 코드를 직접 작성하세요.
- 사용자가 빠른 1회성 변환을 원할 때 (예: "이 mp4를 GIF로 변환해줘") — ffmpeg를 직접 사용하세요.
- 출력물이 정적 이미지, GIF 또는 오디오 전용 아티팩트인 경우 — 일치하는 특정 스킬(`ascii-art`, `gifs`, `meme-generation`, `songwriting-and-ai-music`)을 사용하세요.
- 작업이 기존 단일 스킬에 깔끔하게 들어맞을 때 (예: 순수 ASCII 비디오 — 그냥 `ascii-video`를 사용하세요).

## 워크플로우

```
DISCOVER(발견)  →  BRIEF(브리프 작성)  →  TEAM DESIGN(팀 설계)  →  SETUP(설정)  →  EXECUTE(실행)  →  MONITOR(모니터링)
```

### 1단계 — 발견 (올바른 질문하기)

디스커버리 프로세스는 **적응형(adaptive)**입니다. 실제로 필요한 것만 질문하세요. 큰 틀을 파악하기 위해 항상 세 가지 질문으로 시작하세요:

- **어떤 비디오인가요?** (한 문장 브리프)
- **길이는 어느 정도인가요?** (5-30초 티저 / 30-90초 숏폼 / 90초-3분 설명 영상 / 3-10분 영화 / 더 긴 영상)
- **화면 비율과 타겟 플랫폼은 무엇인가요?** (1:1 / 9:16 / 16:9; X, 인스타그램, 유튜브, 내부용 등)

답변을 바탕으로 스타일 카테고리를 분류합니다. 스타일은 후속으로 어떤 질문을 할지 결정합니다. **모든 질문을 한 번에 하지 마세요.** 한 번에 2-4가지를 묻고, 듣고, 진행하세요. 사용자가 답변을 암시할 때는 합리적인 가정을 세우세요.

전체 정보 수집 패턴과 스타일별 질문 뱅크는 **[references/intake.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/intake.md)**를 참조하세요.

### 2단계 — 브리프

충분히 파악되면 `assets/brief.md.tmpl`에 있는 템플릿을 사용하여 구조화된 `brief.md`를 작성합니다. 단계:

1. **컨셉 (Concept)** — 한 줄짜리 피치(pitch) + 감정적 지향점(emotional north star)
2. **범위 (Scope)** — 지속 시간, 화면 비율, 플랫폼, 마감일
3. **스타일 (Style)** — 시각적 참고 자료, 브랜드 제약, 톤
4. **장면 (Scenes)** — 비트(beat) 단위 분석 (지속 시간, 내용, 대상 도구)
5. **오디오 (Audio)** — 내레이션 / 음악 / SFX / 무음 (필요 시 장면별)
6. **결과물 (Deliverables)** — 파일 형식, 해상도, 선택적 대체본 (세로 컷, GIF 등)

팀을 설계하기 전에 사용자에게 브리프를 보여주고 확인을 받으세요. **브리프는 계약서입니다** — 하위의 모든 작업이 이를 참조합니다.

### 3단계 — 팀 설계

이 비디오에 맞는 역할 원형(role archetypes)을 라이브러리에서 고르세요. **복제하지 말고 구성하세요.** 대부분의 비디오에는 4-7개의 프로필이 필요합니다. 디렉터(director)는 항상 존재하며, 나머지는 브리프에 실제로 필요한 역할에 따라 선택됩니다.

역할 라이브러리와 스타일별 팀 구성은 **[references/role-archetypes.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/role-archetypes.md)**를 참조하세요.

역할에 따라 어떤 Hermes 스킬과 도구 세트를 로드할지 매핑하는 내용은 **[references/tool-matrix.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/tool-matrix.md)**를 참조하세요.

### 4단계 — 설정

설정 스크립트(`setup.sh`)를 생성하고 실행합니다. 스크립트는 다음을 수행합니다:

1. 프로젝트 작업 공간 생성 (`~/projects/video-pipeline/<slug>/`)
2. 제공된 에셋을 `taste/`, `audio/`, `assets/`로 복사
3. `hermes profile create --clone`을 통해 각 Hermes 프로필 생성
4. 프로필별 `SOUL.md` (성격 + 역할 정의) 작성
5. 프로필 YAML 구성 (toolsets, always_load 스킬, cwd)
6. `brief.md`, `TEAM.md`, 및 `taste/` 콘텐츠 작성
7. 디렉터에게 할당되는 초기 `hermes kanban create` 작업 실행

`scripts/bootstrap_pipeline.py`를 사용하여 브리프 + 팀 설계 JSON에서 setup.sh를 생성하세요. 설정 스크립트 구조, 프로필 구성 패턴 및 핵심적인 "공유 작업 공간" 규칙은 **[references/kanban-setup.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/kanban-setup.md)**를 참조하세요.

### 5단계 — 실행

`setup.sh`를 실행합니다. 그런 다음 사용자에게 모니터링 명령어를 제공합니다:

```bash
hermes kanban watch --tenant <project-tenant>     # 실시간 이벤트
hermes kanban list  --tenant <project-tenant>     # 보드 스냅샷
hermes dashboard                                   # 시각적 보드 UI
```

이제부터 디렉터 프로필이 넘겨받아 작업을 분해하고 칸반 도구 세트를 통해 전문가 프로필로 작업을 라우팅합니다.

### 6단계 — 모니터링 및 개입

계속 관여하세요 — 칸반은 자율적으로 실행되지만 정체된 작업이나 잘못된 결과물에는 사람(또는 AI)의 판단이 필요합니다.

모니터링 패턴: `kanban list`를 주기적으로 폴링(poll)하고, 예상 지속 시간을 초과하여 실행 중(RUNNING)인 작업이 있으면 `kanban show <id>`로 검사하며, 하트비트를 확인합니다. 작업자의 결과물이 검토에 실패했을 때 표준 개입 방법은 다음과 같습니다:

1. 작업자의 태스크에 구체적인 피드백을 댓글로 남깁니다 (`kanban_comment`)
2. 원래 태스크를 부모로 삼아 재실행 태스크를 만듭니다
3. 브리프의 범위를 조정하고 디렉터가 다시 분해하도록 합니다

진단 패턴, 개입 레시피 및 "작업이 멈췄을 때"의 플레이북은 **[references/monitoring.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/monitoring.md)**를 참조하세요.

## 참조: 작업 예시

서사 영화, 제품/마케팅, 뮤직비디오, 수학/알고리즘 설명 영상, ASCII 비디오, 실시간 설치물 등 매우 다양한 비디오 스타일을 다루는 6개의 구체적인 파이프라인. 동일한 워크플로우가 어떻게 매우 다른 팀과 작업 그래프(task graph)를 생성하는지 보여줍니다. **[references/examples.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/examples.md)**를 참조하세요.

## 핵심 규칙

1. **실행 전 디스커버리.** 최소한 3가지 기본 질문을 하기 전에는 브리프나 팀 생성을 절대 시작하지 마세요. 잘못된 브리프는 전체 파이프라인에 도미노처럼 영향을 미칩니다.

2. **비디오에 맞는 팀 구성.** 모든 작업에 동일한 4개의 프로필 구성을 재사용하지 마세요. 비트(beat) 분석 프로필이 없는 뮤직비디오는 실패합니다. 작가 프로필이 없는 서사 영화는 앞뒤가 안 맞는 장면을 만들어냅니다. `references/role-archetypes.md`를 참조하세요.

3. **프로젝트당 하나의 작업 공간.** 주어진 비디오의 모든 프로필은 동일한 `dir:` 작업 공간을 공유합니다. 태스크들은 공유 파일 시스템과 구조화된 인수인계(handoffs)를 통해 아티팩트를 전달합니다. **모든** `kanban_create` 호출은 `workspace_kind="dir"` + `workspace_path="<절대 프로젝트 경로>"`를 전달합니다.

4. **모든 프로젝트에 테넌트 할당.** 프로젝트별 테넌트를 사용하세요 (`--tenant <project-slug>`). 대시보드의 범위를 한정하고 진행 중인 다른 칸반과 섞이는 것을 방지합니다.

5. **기존 스킬 존중.** 장면이 기존 스킬에 적합한 경우, 해당 렌더러는 자신의 태스크에 `--skill <name>`을 사용하거나 프로필의 `always_load`를 통해 해당 스킬을 로드해야 합니다. 스킬이 이미 제공하는 것을 다시 파생시키지 마세요.

6. **디렉터는 절대 직접 실행하지 않음.** 완전한 `kanban + terminal + file` 도구 세트를 가지고 있더라도 디렉터의 `SOUL.md` 규칙은 디렉터가 직접 작업을 실행하는 것을 금지합니다. 오직 분해하고 라우팅할 뿐입니다 — 구체적인 모든 작업은 전문가 프로필로 향하는 `hermes kanban create` 호출이 됩니다. `kanban-orchestrator` 스킬이 이를 더 자세히 설명합니다.

7. **과도한 작업 분해 금지.** 30초짜리 제품 비디오에 20개의 작업이 필요하지는 않습니다. 적절히 병렬화되면서 올바른 사람-검토 게이트(human-review gates)를 노출하는 가장 작은 단위의 작업 그래프를 목표로 하세요.

8. **실행 전 API 키 확인.** 외부 API(TTS, 이미지 생성, 이미지-비디오 변환)에는 `~/.hermes/.env` 또는 사용자의 비밀 저장소에 키가 필요합니다. 누락된 키 오류에 부딪히는 작업자는 작업 슬롯을 낭비하게 됩니다. 설정 스크립트의 `check_key` 도우미는 필요한 키가 없을 때 깔끔하게 중단합니다.

## 파일 맵

```
SKILL.md                            ← 이 파일 (워크플로우 + 규칙)
references/
  intake.md                         ← 스타일별 디스커버리 질문 뱅크
  role-archetypes.md                ← 역할 라이브러리 (작가, 디자이너, 애니메이터 등)
  tool-matrix.md                    ← 역할별 스킬 + 도구 세트 매핑
  kanban-setup.md                   ← 설정 스크립트 구조 및 프로필 구성
  monitoring.md                     ← 모니터링 + 개입 패턴
  examples.md                       ← 6개의 작업 예시 파이프라인
assets/
  brief.md.tmpl                     ← 브리프 골격
  setup.sh.tmpl                     ← 설정 스크립트 골격
  soul.md.tmpl                      ← 프로필 성격 골격
scripts/
  bootstrap_pipeline.py             ← 브리프 + 팀 JSON에서 setup.sh 생성
  monitor.py                        ← 폴링 + 개입 도우미
```
