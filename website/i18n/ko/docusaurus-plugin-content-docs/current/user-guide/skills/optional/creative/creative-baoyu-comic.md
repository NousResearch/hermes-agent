---
title: "Baoyu Comic — 지식 만화 (知识漫画): 교육, 전기, 튜토리얼"
sidebar_label: "Baoyu Comic"
description: "지식 만화 (知识漫画): 교육, 전기, 튜토리얼"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Comic

지식 만화 (知识漫画): 교육, 전기, 튜토리얼.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/baoyu-comic`로 설치 |
| Path | `optional-skills/creative/baoyu-comic` |
| Version | `1.56.1` |
| Author | 宝玉 (JimLiu) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `comic`, `knowledge-comic`, `creative`, `image-generation` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Knowledge Comic Creator

Hermes Agent의 도구 생태계를 위해 [baoyu-comic](https://github.com/JimLiu/baoyu-skills)에서 조정되었습니다.

유연한 아트 스타일과 톤의 조합으로 독창적인 지식 만화를 만듭니다.

## 사용 시기

사용자가 지식/교육 만화, 전기 만화, 튜토리얼 만화를 만들어 달라고 요청하거나 "知识漫画", "教育漫画", "Logicomix-style"과 같은 용어를 사용할 때 이 스킬을 트리거하세요. 사용자는 콘텐츠(텍스트, 파일 경로, URL 또는 주제)를 제공하고 선택적으로 아트 스타일, 톤, 레이아웃, 화면 비율 또는 언어를 지정합니다.

## 참조 이미지

Hermes의 `image_generate` 도구는 **프롬프트 전용**입니다 — 텍스트 프롬프트와 화면 비율만 허용하고 이미지 URL을 반환합니다. 이 도구는 참조 이미지를 **허용하지 않습니다**. 사용자가 참조 이미지를 제공하는 경우, 이를 사용하여 매 페이지 프롬프트에 포함될 **특징을 텍스트로 추출**하세요:

**입력**: 사용자가 파일 경로를 제공하거나(또는 대화에 이미지를 붙여넣은 경우) 파일 경로를 수락합니다.
- 파일 경로 → 출처를 위해 만화 결과물 옆의 `refs/NN-ref-{slug}.{ext}`에 복사합니다.
- 경로 없이 붙여넣은 이미지 → `clarify`를 통해 사용자에게 경로를 묻거나 텍스트 대체 수단으로 구두로 스타일 특징을 추출합니다.
- 참조 없음 → 이 섹션을 건너뜁니다.

**사용 모드** (각 참조마다):

| 용도 | 효과 |
|-------|--------|
| `style` | 스타일 특징(선 처리, 질감, 무드)을 추출하여 모든 페이지의 프롬프트 본문에 추가합니다. |
| `palette` | 16진수(Hex) 색상을 추출하여 모든 페이지의 프롬프트 본문에 추가합니다. |
| `scene` | 장면 구도나 주제에 대한 노트를 추출하여 해당 페이지에 추가합니다. |

참조 이미지가 있을 때 **각 페이지 프롬프트의 Frontmatter에 기록**하세요:

```yaml
references:
  - ref_id: 01
    filename: 01-ref-scene.png
    usage: style
    traits: "muted earth tones, soft-edged ink wash, low-contrast backgrounds"
```

캐릭터 일관성은 모든 페이지 프롬프트(5단계)에 인라인으로 삽입되는 `characters/characters.md`의 **텍스트 설명**(3단계에서 작성됨)에 의해 유지됩니다. 7.1단계에서 생성된 선택적인 PNG 캐릭터 시트는 사람의 검토를 위한 결과물이며 `image_generate`의 입력이 아닙니다.

## 옵션

### 시각적 차원

| 옵션 | 값 | 설명 |
|--------|--------|-------------|
| Art | ligne-claire (기본값), manga, realistic, ink-brush, chalk, minimalist | 아트 스타일 / 렌더링 기법 |
| Tone | neutral (기본값), warm, dramatic, romantic, energetic, vintage, action | 분위기 / 무드 |
| Layout | standard (기본값), cinematic, dense, splash, mixed, webtoon, four-panel | 패널 배열 |
| Aspect | 3:4 (기본값, 세로), 4:3 (가로), 16:9 (와이드스크린) | 페이지 화면 비율 |
| Language | auto (기본값), zh, en, ja 등 | 출력 언어 |
| Refs | 파일 경로 | 스타일 / 팔레트 특징 추출에 사용되는 참조 이미지(이미지 모델에 전달되지 않음). 위의 [참조 이미지](#참조-이미지)를 참조하세요. |

### 부분 워크플로우 옵션

| 옵션 | 설명 |
|--------|-------------|
| Storyboard only | 스토리보드만 생성하고 프롬프트 및 이미지는 건너뜁니다. |
| Prompts only | 스토리보드 + 프롬프트를 생성하고 이미지는 건너뜁니다. |
| Images only | 기존 프롬프트 디렉토리에서 이미지를 생성합니다. |
| Regenerate N | 특정 페이지만 다시 생성합니다 (예: `3` 또는 `2,5,8`). |

자세한 내용: [references/partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/partial-workflows.md)

### 아트, 톤 및 프리셋 카탈로그

- **아트 스타일** (6): `ligne-claire`, `manga`, `realistic`, `ink-brush`, `chalk`, `minimalist`. 전체 정의는 `references/art-styles/<style>.md`에 있습니다.
- **톤** (7): `neutral`, `warm`, `dramatic`, `romantic`, `energetic`, `vintage`, `action`. 전체 정의는 `references/tones/<tone>.md`에 있습니다.
- **프리셋** (5) (단순한 아트+톤을 넘어서는 특별한 규칙 포함):

  | 프리셋 | 상응하는 조합 | 특징(Hook) |
  |--------|-----------|------|
  | `ohmsha` | manga + neutral | 시각적 은유, 말만 하는 인물 배제(no talking heads), 가젯(도구) 등장 |
  | `wuxia` | ink-brush + action | 기(Qi) 효과, 전투 시각화, 분위기 있음 |
  | `shoujo` | manga + romantic | 장식 요소, 눈 디테일, 로맨틱한 박자 |
  | `concept-story` | manga + warm | 시각적 기호 시스템, 성장 서사, 대화+행동 균형 |
  | `four-panel` | minimalist + neutral + four-panel layout | 기승전결(起承转合) 구조, 흑백 + 포인트 색상, 졸라맨 캐릭터 |

  전체 규칙은 `references/presets/<preset>.md`에 있습니다 — 프리셋이 선택되면 해당 파일을 로드하세요.

- **호환성 매트릭스** 및 **콘텐츠-신호 → 프리셋** 표는 [references/auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/auto-selection.md)에 있습니다. 2단계에서 조합을 추천하기 전에 이를 읽어보세요.

## 파일 구조

출력 디렉토리: `comic/{topic-slug}/`
- Slug: 주제에서 파생된 2-4개의 단어(kebab-case) (예: `alan-turing-bio`)
- 충돌 시: 타임스탬프 추가 (예: `turing-story-20260118-143052`)

**내용물**:
| 파일 | 설명 |
|------|-------------|
| `source-{slug}.md` | 저장된 소스 콘텐츠 (kebab-case 슬러그는 출력 디렉토리와 일치함) |
| `analysis.md` | 콘텐츠 분석 |
| `storyboard.md` | 패널 분할이 포함된 스토리보드 |
| `characters/characters.md` | 캐릭터 정의 |
| `characters/characters.png` | 캐릭터 참조 시트 (`image_generate`에서 다운로드됨) |
| `prompts/NN-{cover\|page}-[slug].md` | 생성 프롬프트 |
| `NN-{cover\|page}-[slug].png` | 생성된 이미지 (`image_generate`에서 다운로드됨) |
| `refs/NN-ref-{slug}.{ext}` | 사용자 제공 참조 이미지 (선택 사항, 출처 기록용) |

## 언어 처리

**감지 우선순위**:
1. 사용자가 지정한 언어 (명시적 옵션)
2. 사용자의 대화 언어
3. 소스 콘텐츠 언어

**규칙**: 모든 상호 작용에 사용자의 입력 언어를 사용하세요:
- 스토리보드 개요 및 장면 설명
- 이미지 생성 프롬프트
- 사용자 선택 옵션 및 확인
- 진행 상황 업데이트, 질문, 오류, 요약

기술 용어는 영어로 유지합니다.

## 워크플로우

### 진행 체크리스트

```
만화 제작 진행 상황:
- [ ] 1단계: 설정 및 분석
  - [ ] 1.1 콘텐츠 분석
  - [ ] 1.2 기존 디렉토리 확인
- [ ] 2단계: 확인 - 스타일 및 옵션 ⚠️ 필수
- [ ] 3단계: 스토리보드 + 캐릭터 생성
- [ ] 4단계: 개요 검토 (조건부)
- [ ] 5단계: 프롬프트 생성
- [ ] 6단계: 프롬프트 검토 (조건부)
- [ ] 7단계: 이미지 생성
  - [ ] 7.1 캐릭터 시트 생성 (필요한 경우) → characters/characters.png
  - [ ] 7.2 페이지 생성 (프롬프트에 캐릭터 설명 포함)
- [ ] 8단계: 완료 보고
```

### 흐름

```
입력 → 분석 → [기존 확인?] → [확인: 스타일 + 검토] → 스토리보드 → [검토?] → 프롬프트 → [검토?] → 이미지 → 완료
```

### 단계 요약

| 단계 | 작업 | 주요 출력물 |
|------|--------|------------|
| 1.1 | 콘텐츠 분석 | `analysis.md`, `source-{slug}.md` |
| 1.2 | 기존 디렉토리 확인 | 충돌 처리 |
| 2 | 스타일, 포커스, 대상 독자, 검토 여부 확인 | 사용자 선호 사항 |
| 3 | 스토리보드 + 캐릭터 생성 | `storyboard.md`, `characters/` |
| 4 | 개요 검토 (요청된 경우) | 사용자 승인 |
| 5 | 프롬프트 생성 | `prompts/*.md` |
| 6 | 프롬프트 검토 (요청된 경우) | 사용자 승인 |
| 7.1 | 캐릭터 시트 생성 (필요한 경우) | `characters/characters.png` |
| 7.2 | 페이지 생성 | `*.png` 파일들 |
| 8 | 완료 보고 | 요약 |

### 사용자 질문

`clarify` 도구를 사용하여 옵션을 확인하세요. `clarify`는 한 번에 하나의 질문만 처리하므로 가장 중요한 질문부터 순차적으로 진행하세요. 전체 2단계 질문 세트는 [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/workflow.md)를 참조하세요.

**타임아웃 처리 (매우 중요)**: `clarify`는 `"사용자가 시간 제한 내에 응답하지 않았습니다. 최선의 판단을 사용하여 선택하고 진행하십시오."`를 반환할 수 있습니다 — 이것은 모든 것을 기본값으로 해도 된다는 사용자의 동의가 **아닙니다**.

- **해당 한 질문에 대해서만** 기본값으로 처리하세요. 나머지 2단계 질문들을 순서대로 계속 물어보세요. 각 질문은 독립적인 동의 포인트입니다.
- **다음 메시지에서 기본값을 사용자에게 명확히 알리세요**. 그래야 사용자가 수정할 기회를 갖습니다: 예: `"스타일: ohmsha 프리셋으로 기본 설정되었습니다 (clarify 타임아웃). 변경하려면 말씀해주세요."` — 알리지 않은 기본값은 아예 묻지 않은 것과 구별할 수 없습니다.
- 한 번의 타임아웃 후에 2단계를 "모든 기본값 사용"이라는 하나의 패스로 축소하지 **마세요**. 사용자가 실제로 자리에 없다면 다섯 가지 질문 모두에 대해 부재 중일 것입니다 — 하지만 돌아왔을 때 보이는 기본값은 수정할 수 있어도 보이지 않는 기본값은 수정할 수 없습니다.

### 7단계: 이미지 생성

모든 이미지 렌더링에 Hermes의 내장 `image_generate` 도구를 사용하세요. 이 스키마는 `prompt`와 `aspect_ratio`(`landscape` | `portrait` | `square`)만 허용하며 로컬 파일이 아닌 **URL을 반환**합니다. 따라서 생성된 모든 페이지나 캐릭터 시트는 출력 디렉토리로 다운로드되어야 합니다.

**프롬프트 파일 요구 사항 (필수)**: `image_generate`를 호출하기 전에 각 이미지의 최종 프롬프트 전체를 `prompts/` 아래의 독립된 파일(명명 규칙: `NN-{type}-[slug].md`)에 기록하세요. 프롬프트 파일은 재현성 기록입니다.

**화면 비율 매핑** — 스토리보드의 `aspect_ratio` 필드는 `image_generate`의 형식에 다음과 같이 매핑됩니다:

| 스토리보드 비율 | `image_generate` 형식 |
|------------------|-------------------------|
| `3:4`, `9:16`, `2:3` | `portrait` |
| `4:3`, `16:9`, `3:2` | `landscape` |
| `1:1` | `square` |

**다운로드 단계** — 모든 `image_generate` 호출 이후:
1. 도구 결과에서 URL을 읽습니다.
2. **절대(absolute)** 출력 경로를 사용하여 이미지 바이트를 가져옵니다. 예:
   `curl -fsSL "<url>" -o /abs/path/to/comic/<slug>/NN-page-<slug>.png`
3. 다음 페이지로 진행하기 전에 파일이 해당 경로에 존재하고 비어 있지 않은지 확인합니다.

**`-o` 경로에 대해 쉘 CWD 지속성에 절대 의존하지 마세요.** 터미널 도구의 지속형 쉘 CWD는 배치(batch) 간에 변경될 수 있습니다(세션 만료, `TERMINAL_LIFETIME_SECONDS`, 잘못된 디렉토리에 남겨두는 실패한 `cd`). `curl -o relative/path.png`는 조용히 문제를 일으키는 치명적인 실수(silent footgun)입니다: CWD가 변경된 경우 오류 없이 다른 곳에 파일이 저장됩니다. **항상 완전히 지정된 절대 경로를 `-o`에 전달하거나**, 터미널 도구에 `workdir=<abs path>`를 전달하세요. (사례: 10페이지 분량 만화의 06-09페이지가 CWD가 꼬여서 엉뚱한 곳에 다운로드된 적이 있습니다).

**7.1 캐릭터 시트** — 반복되는 캐릭터가 있는 여러 페이지 만화일 때 생성합니다 (`characters/characters.png`에 `landscape` 비율로). 단순한 프리셋(예: 네 컷 미니멀리스트)이나 단일 페이지 만화인 경우 건너뜁니다. `image_generate`를 호출하기 전에 `characters/characters.md`의 프롬프트 파일이 존재해야 합니다. 렌더링된 PNG는 사람의 시각적 검증을 위한 **리뷰용 아티팩트**이자 향후 재생성이나 수동 프롬프트 편집을 위한 참고 자료이며, 7.2단계를 직접 구동하지는 **않습니다**. 페이지 프롬프트는 5단계에서 `characters/characters.md`의 **텍스트 설명**으로 작성됩니다; `image_generate`는 이미지를 시각적 입력으로 허용하지 않습니다.

**7.2 페이지** — 각 페이지의 프롬프트는 `image_generate`를 호출하기 전에 반드시 `prompts/NN-{cover|page}-[slug].md`에 있어야 합니다. `image_generate`가 프롬프트 전용이므로, 캐릭터 일관성은 **5단계에서 모든 페이지 프롬프트에 인라인으로 삽입된 (characters/characters.md에서 가져온) 캐릭터 설명**에 의해 유지됩니다. 삽입은 7.1에서 PNG 시트가 생성되었는지 여부와 무관하게 일관되게 수행되며; PNG는 오직 검토/재생성 보조 도구일 뿐입니다.

**백업 규칙**: 기존 `prompts/…md` 및 `…png` 파일 → 다시 생성하기 전에 `-backup-YYYYMMDD-HHMMSS` 접미사를 붙여 이름을 바꿉니다.

단계별 전체 워크플로우(분석, 스토리보드, 검토 게이트, 재생성 변형): [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/workflow.md).

## 참조 자료

**핵심 템플릿**:
- [analysis-framework.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/analysis-framework.md) - 심층 콘텐츠 분석
- [character-template.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/character-template.md) - 캐릭터 정의 형식
- [storyboard-template.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/storyboard-template.md) - 스토리보드 구조
- [ohmsha-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/ohmsha-guide.md) - Ohmsha 만화 세부 사항

**스타일 정의**:
- `references/art-styles/` - 아트 스타일 (ligne-claire, manga, realistic, ink-brush, chalk, minimalist)
- `references/tones/` - 톤 (neutral, warm, dramatic, romantic, energetic, vintage, action)
- `references/presets/` - 특별한 규칙이 있는 프리셋 (ohmsha, wuxia, shoujo, concept-story, four-panel)
- `references/layouts/` - 레이아웃 (standard, cinematic, dense, splash, mixed, webtoon, four-panel)

**워크플로우**:
- [workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/workflow.md) - 전체 워크플로우 세부 사항
- [auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/auto-selection.md) - 콘텐츠 신호 분석
- [partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/partial-workflows.md) - 부분 워크플로우 옵션

## 페이지 수정

| 작업 | 단계 |
|--------|-------|
| **편집** | **프롬프트 파일을 먼저 업데이트** → 이미지 재생성 → 새 PNG 다운로드 |
| **추가** | 위치에 프롬프트 생성 → 캐릭터 설명을 포함하여 생성 → 이후 번호 재부여 → 스토리보드 업데이트 |
| **삭제** | 파일 제거 → 이후 번호 재부여 → 스토리보드 업데이트 |

**중요**: 페이지를 업데이트할 때 이미지를 다시 생성하기 전에 **항상 프롬프트 파일(`prompts/NN-{cover|page}-[slug].md`)을 먼저 업데이트**하세요. 이렇게 해야 변경 사항이 문서화되고 재현 가능합니다.

## 주의 사항

- 이미지 생성: 페이지당 10-30초; 실패 시 1회 자동 재시도
- `image_generate`가 반환한 URL을 **항상 로컬 PNG로 다운로드**하세요 — 후속 작업(및 사용자의 검토)은 일시적인 URL이 아닌 출력 디렉토리의 파일을 필요로 합니다.
- **`curl -o`에 절대 경로를 사용하세요** — 배치 간에 지속형 쉘 CWD에 절대 의존하지 마세요. 소리 없는 치명적 실수(Silent footgun): 파일이 잘못된 디렉토리에 저장되고 나중에 올바른 경로에서 `ls`를 해도 아무것도 나오지 않을 수 있습니다. 7단계 "다운로드 단계" 참조.
- 민감한 공인에 대해서는 스타일화된 대안을 사용하세요.
- **2단계 확인 필수** - 건너뛰지 마세요.
- **4/6단계는 조건부** - 2단계에서 사용자가 요청한 경우에만 수행합니다.
- **7.1단계 캐릭터 시트** - 여러 페이지 만화에는 권장되지만, 단순한 프리셋에는 선택 사항입니다. PNG는 검토/재생성 보조 도구입니다; 페이지 프롬프트(5단계에서 작성됨)는 PNG가 아닌 `characters/characters.md`의 텍스트 설명을 사용합니다. `image_generate`는 시각적 입력으로 이미지를 받지 않습니다.
- **시크릿 정보 제거** — 출력 파일을 쓰기 전에 소스 콘텐츠에서 API 키, 토큰 또는 자격 증명을 스캔하세요.
