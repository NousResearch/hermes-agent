---
title: "Baoyu Comic — 지식 만화(知识漫画): 교육, 전기, 튜토리얼"
sidebar_label: "Baoyu Comic"
description: "지식 만화(知识漫画): 교육, 전기, 튜토리얼"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Baoyu Comic

지식 만화(知识漫画): 교육, 전기, 튜토리얼.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/baoyu-comic`으로 설치 |
| 경로 | `optional-skills/creative/baoyu-comic` |
| 버전 | `1.56.1` |
| 작성자 | 宝玉 (JimLiu) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `comic`, `knowledge-comic`, `creative`, `image-generation` |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 트리거될 때 Hermes가 로드하는 전체 skill 정의입니다. 활성화된 skill이 실행 중일 때 에이전트가 보게 되는 지침입니다.
:::

# 지식 만화 제작자

Hermes Agent의 도구 생태계에 맞게 [baoyu-comic](https://github.com/JimLiu/baoyu-skills)을 적용했습니다.

유연한 아트 스타일 × 톤 조합으로 독창적인 지식 만화를 제작합니다.

## 사용 시점

사용자가 지식/교육 만화, 전기 만화, 튜토리얼 만화를 만들어 달라고 요청하거나 "知识漫画", "教育漫画", "Logicomix-style"과 같은 용어를 사용할 때 이 skill을 트리거합니다. 사용자는 콘텐츠(텍스트, 파일 경로, URL 또는 주제)를 제공하며, 아트 스타일, 톤, 레이아웃, 가로세로 비율 또는 언어를 선택적으로 지정할 수 있습니다.

## 참고 이미지

Hermes의 `image_generate` 도구는 **prompt-only**입니다. 텍스트 prompt와 가로세로 비율을 받아 이미지 URL을 반환합니다. 참고 이미지는 받지 않습니다. 사용자가 참고 이미지를 제공하면 이미지의 특성을 **텍스트로 추출**하여 모든 페이지 prompt에 삽입합니다.

**입력**: 사용자가 제공한 파일 경로(또는 대화에 붙여 넣은 이미지)를 받습니다.
- 파일 경로 → 출처 추적을 위해 만화 출력물과 나란히 `refs/NN-ref-{slug}.{ext}`로 복사
- 경로 없이 붙여 넣은 이미지 → `clarify`로 사용자에게 경로를 묻거나, 텍스트 대체 수단으로 스타일 특성을 말로 추출
- 참고 없음 → 이 섹션 건너뛰기

**사용 모드**(참고 자료별):

| 사용 방식 | 효과 |
|-------|--------|
| `style` | 스타일 특성(선 처리, 질감, 분위기)을 추출하고 모든 페이지 prompt 본문에 추가 |
| `palette` | hex 색상을 추출하고 모든 페이지 prompt 본문에 추가 |
| `scene` | 장면 구성 또는 주제 메모를 추출하고 관련 페이지에 추가 |

참고 자료가 있으면 각 페이지 prompt의 frontmatter에 다음을 **기록**합니다.

```yaml
references:
  - ref_id: 01
    filename: 01-ref-scene.png
    usage: style
    traits: "muted earth tones, soft-edged ink wash, low-contrast backgrounds"
```

캐릭터 일관성은 `characters/characters.md`(3단계에서 작성)에 있는 **텍스트 설명**으로 유지하며, 이 설명은 각 페이지 prompt(5단계)에 인라인으로 삽입됩니다. 7.1단계에서 선택적으로 생성되는 PNG 캐릭터 시트는 사람이 검토하는 산출물이며 `image_generate`의 입력이 아닙니다.

## 옵션

### 시각적 차원

| 옵션 | 값 | 설명 |
|--------|--------|-------------|
| Art | ligne-claire (default), manga, realistic, ink-brush, chalk, minimalist | 아트 스타일 / 렌더링 기법 |
| Tone | neutral (default), warm, dramatic, romantic, energetic, vintage, action | 분위기 / 정서 |
| Layout | standard (default), cinematic, dense, splash, mixed, webtoon, four-panel | 패널 배치 |
| Aspect | 3:4 (default, portrait), 4:3 (landscape), 16:9 (widescreen) | 페이지 가로세로 비율 |
| Language | auto (default), zh, en, ja, etc. | 출력 언어 |
| Refs | File paths | 스타일 / 팔레트 특성 추출에 사용하는 참고 이미지 경로(`image_generate`에 전달하지 않음). 위의 [참고 이미지](#reference-images) 참조. |

### 부분 워크플로 옵션

| 옵션 | 설명 |
|-------------|-------------|
| Storyboard only | 스토리보드만 생성하고 prompt와 이미지는 건너뛰기 |
| Prompts only | 스토리보드 + prompt를 생성하고 이미지는 건너뛰기 |
| Images only | 기존 prompt 디렉터리에서 이미지 생성 |
| Regenerate N | 특정 페이지만 재생성(예: `3` 또는 `2,5,8`) |

세부 정보: [references/partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/partial-workflows.md)

### 아트, 톤 및 프리셋 카탈로그

- **아트 스타일**(6): `ligne-claire`, `manga`, `realistic`, `ink-brush`, `chalk`, `minimalist`. 전체 정의는 `references/art-styles/<style>.md`에 있습니다.
- **톤**(7): `neutral`, `warm`, `dramatic`, `romantic`, `energetic`, `vintage`, `action`. 전체 정의는 `references/tones/<tone>.md`에 있습니다.
- 일반적인 아트+톤 조합 외에 특별한 규칙이 있는 **프리셋**(5):

  | Preset | Equivalent | Hook |
  |--------|-----------|------|
  | `ohmsha` | manga + neutral | 시각적 은유, 말하는 머리 배제, 가젯 공개 |
  | `wuxia` | ink-brush + action | 기 효과, 전투 장면, 분위기 있는 연출 |
  | `shoujo` | manga + romantic | 장식 요소, 눈 디테일, 로맨틱한 순간 |
  | `concept-story` | manga + warm | 시각적 기호 체계, 성장 서사, 대화+행동 균형 |
  | `four-panel` | minimalist + neutral + four-panel layout | 기승전결 구조, 흑백 + 포인트 색상, 막대기 인물 |

  전체 규칙은 `references/presets/<preset>.md`에 있습니다. 프리셋을 선택하면 해당 파일을 로드하세요.

- **호환성 매트릭스** 및 **콘텐츠 신호 → 프리셋** 표는 [references/auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/auto-selection.md)에 있습니다. 2단계에서 조합을 추천하기 전에 읽으세요.

## 파일 구조

출력 디렉터리: `comic/{topic-slug}/`
- Slug: 주제에서 가져온 2~4개 단어의 kebab-case(예: `alan-turing-bio`)
- 충돌: 타임스탬프 추가(예: `turing-story-20260118-143052`)

**내용**:
| 파일 | 설명 |
|------|-------------|
| `source-{slug}.md` | 저장된 원본 콘텐츠(kebab-case slug는 출력 디렉터리와 일치) |
| `analysis.md` | 콘텐츠 분석 |
| `storyboard.md` | 패널별 분해가 포함된 스토리보드 |
| `characters/characters.md` | 캐릭터 정의 |
| `characters/characters.png` | 캐릭터 참고 시트(`image_generate`에서 다운로드) |
| `prompts/NN-{cover\|page}-[slug].md` | 생성 prompt |
| `NN-{cover\|page}-[slug].png` | 생성된 이미지(`image_generate`에서 다운로드) |
| `refs/NN-ref-{slug}.{ext}` | 사용자가 제공한 참고 이미지(선택 사항, 출처 추적용) |

## 언어 처리

**감지 우선순위**:
1. 사용자가 지정한 언어(명시적 옵션)
2. 사용자의 대화 언어
3. 원본 콘텐츠 언어

**규칙**: 모든 상호작용에 사용자의 입력 언어를 사용합니다.
- 스토리보드 개요 및 장면 설명
- 이미지 생성 prompt
- 사용자 선택 옵션 및 확인
- 진행 상황 업데이트, 질문, 오류, 요약

기술 용어는 English로 유지합니다.

## 워크플로

### 진행 체크리스트

```
Comic Progress:
- [ ] Step 1: Setup & Analyze
  - [ ] 1.1 Analyze content
  - [ ] 1.2 Check existing directory
- [ ] Step 2: Confirmation - Style & options ⚠️ REQUIRED
- [ ] Step 3: Generate storyboard + characters
- [ ] Step 4: Review outline (conditional)
- [ ] Step 5: Generate prompts
- [ ] Step 6: Review prompts (conditional)
- [ ] Step 7: Generate images
  - [ ] 7.1 Generate character sheet (if needed) → characters/characters.png
  - [ ] 7.2 Generate pages (with character descriptions embedded in prompt)
- [ ] Step 8: Completion report
```

### 흐름

```
Input → Analyze → [Check Existing?] → [Confirm: Style + Reviews] → Storyboard → [Review?] → Prompts → [Review?] → Images → Complete
```

### 단계 요약

| 단계 | 작업 | 주요 결과물 |
|------|--------|------------|
| 1.1 | 콘텐츠 분석 | `analysis.md`, `source-{slug}.md` |
| 1.2 | 기존 디렉터리 확인 | 충돌 처리 |
| 2 | 스타일, 초점, 대상, 검토 여부 확인 | 사용자 설정 |
| 3 | 스토리보드 + 캐릭터 생성 | `storyboard.md`, `characters/` |
| 4 | 개요 검토(요청된 경우) | 사용자 승인 |
| 5 | prompt 생성 | `prompts/*.md` |
| 6 | prompt 검토(요청된 경우) | 사용자 승인 |
| 7.1 | 캐릭터 시트 생성(필요한 경우) | `characters/characters.png` |
| 7.2 | 페이지 생성 | `*.png` 파일 |
| 8 | 완료 보고 | 요약 |

### 사용자 질문

옵션을 확인하려면 `clarify` 도구를 사용합니다. `clarify`는 한 번에 한 질문만 처리하므로 가장 중요한 질문을 먼저 하고 순차적으로 진행하세요. 전체 2단계 질문 세트는 [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/workflow.md)를 참조하세요.

**Timeout 처리(중요)**: `clarify`가 `"The user did not provide a response within the time limit. Use your best judgement to make the choice and proceed."`를 반환할 수 있습니다. 이는 모든 항목의 기본값을 사용해도 된다는 사용자의 동의가 **아닙니다**.

- 해당 질문 **하나에 대해서만** 기본값으로 처리합니다. 남은 2단계 질문은 순서대로 계속 묻습니다. 각 질문은 독립적인 동의 지점입니다.
- 다음 메시지에서 기본값을 사용자에게 명시적으로 보여 주어 사용자가 수정할 기회를 제공합니다. 예: `"Style: defaulted to ohmsha preset (clarify timed out). Say the word to switch."` — 보고되지 않은 기본값은 질문하지 않은 것과 구분할 수 없습니다.
- 첫 timeout 이후 2단계를 "모든 기본값 사용" 과정으로 축약하지 **마세요**. 사용자가 정말 자리를 비운 경우에도 다섯 질문 모두에 같은 방식으로 자리를 비울 것입니다. 하지만 돌아온 사용자는 표시된 기본값을 수정할 수 있고, 표시되지 않은 기본값은 수정할 수 없습니다.

### 7단계: 이미지 생성

모든 이미지 렌더링에는 Hermes의 내장 `image_generate` 도구를 사용합니다. 이 도구의 스키마는 `prompt`와 `aspect_ratio`(`landscape` | `portrait` | `square`)만 허용하며, URL을 반환하지 로컬 파일을 반환하지 않습니다. 따라서 생성된 모든 페이지 또는 캐릭터 시트를 출력 디렉터리에 다운로드해야 합니다.

**Prompt 파일 요구 사항(필수)**: 각 이미지의 전체 최종 prompt를 `image_generate`를 호출하기 **전에** `prompts/` 아래의 독립 파일(`NN-{type}-[slug].md` 명명 규칙)에 작성합니다. prompt 파일은 재현성을 기록하는 자료입니다.

**가로세로 비율 매핑** — 스토리보드의 `aspect_ratio` 필드는 다음과 같이 `image_generate`의 format에 매핑됩니다.

| 스토리보드 비율 | `image_generate` format |
|------------------|-------------------------|
| `3:4`, `9:16`, `2:3` | `portrait` |
| `4:3`, `16:9`, `3:2` | `landscape` |
| `1:1` | `square` |

**다운로드 단계** — `image_generate`를 호출할 때마다:
1. 도구 결과에서 URL을 읽습니다.
2. 다음과 같이 **절대** 출력 경로를 사용하여 이미지 바이트를 출력 디렉터리로 가져옵니다:
   `curl -fsSL "<url>" -o /abs/path/to/comic/<slug>/NN-page-<slug>.png`
3. 다음 페이지로 진행하기 전에 정확한 경로에 파일이 존재하고 비어 있지 않은지 확인합니다.

**7.1 캐릭터 시트** — 만화가 여러 페이지로 구성되고 반복 등장 캐릭터가 있으면 `characters/characters.png`(aspect `landscape`)로 생성합니다. 단순한 프리셋(예: four-panel minimalist)이나 단일 페이지 만화에는 생략합니다. `characters/characters.md`의 prompt 파일은 `image_generate`를 호출하기 전에 존재해야 합니다. 렌더링된 PNG는 **사람이 검토하는 산출물**(사용자가 캐릭터 디자인을 시각적으로 확인하기 위함)이자 이후 재생성 또는 수동 prompt 편집을 위한 참고 자료입니다. 7.2단계를 구동하지는 않습니다. 페이지 prompt는 5단계에서 `characters/characters.md`의 **텍스트 설명**을 사용해 이미 작성되며, `image_generate`는 이미지를 시각적 입력으로 받을 수 없습니다.

**7.2 페이지** — 각 페이지의 prompt는 `image_generate`를 호출하기 전에 반드시 `prompts/NN-{cover|page}-[slug].md`에 있어야 합니다. `image_generate`는 prompt-only이므로 5단계에서 `characters/characters.md`에 있는 캐릭터 설명을 각 페이지 prompt에 인라인으로 **삽입**하여 캐릭터 일관성을 유지합니다. PNG 시트를 7.1단계에서 생성하는지와 관계없이 삽입은 동일하게 수행합니다. PNG는 검토/재생성 보조 자료일 뿐입니다.

**백업 규칙**: 기존 `prompts/…md` 및 `…png` 파일은 재생성하기 전에 `-backup-YYYYMMDD-HHMMSS` 접미사를 붙여 이름을 변경합니다.

전체 단계별 워크플로(분석, 스토리보드, 검토 게이트, 재생성 변형): [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/workflow.md).

## 참고 자료

**핵심 템플릿**:
- [analysis-framework.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/analysis-framework.md) - 심층 콘텐츠 분석
- [character-template.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/character-template.md) - 캐릭터 정의 형식
- [storyboard-template.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/storyboard-template.md) - 스토리보드 구조
- [ohmsha-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/ohmsha-guide.md) - Ohmsha 만화 세부 사항

**스타일 정의**:
- `references/art-styles/` - 아트 스타일(ligne-claire, manga, realistic, ink-brush, chalk, minimalist)
- `references/tones/` - 톤(neutral, warm, dramatic, romantic, energetic, vintage, action)
- `references/presets/` - 특별한 규칙이 있는 프리셋(ohmsha, wuxia, shoujo, concept-story, four-panel)
- `references/layouts/` - 레이아웃(standard, cinematic, dense, splash, mixed, webtoon, four-panel)

**워크플로**:
- [workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/workflow.md) - 전체 워크플로 세부 정보
- [auto-selection.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/auto-selection.md) - 콘텐츠 신호 분석
- [partial-workflows.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-comic/references/partial-workflows.md) - 부분 워크플로 옵션

## 페이지 수정

| 작업 | 단계 |
|--------|-------|
| **Edit** | **먼저 prompt 파일 업데이트** → 이미지 재생성 → 새 PNG 다운로드 |
| **Add** | 해당 위치에 prompt 생성 → 캐릭터 설명을 삽입해 생성 → 이후 페이지 번호 변경 → 스토리보드 업데이트 |
| **Delete** | 파일 제거 → 이후 페이지 번호 변경 → 스토리보드 업데이트 |

**중요**: 페이지를 업데이트할 때는 **항상** 재생성 전에 prompt 파일(`prompts/NN-{cover|page}-[slug].md`)을 업데이트하세요. 이렇게 해야 변경 사항이 문서화되고 재현 가능합니다.

## 주의 사항

- 이미지 생성: 페이지당 10~30초, 실패하면 자동으로 한 번 재시도
- **항상 다운로드**: `image_generate`가 반환한 URL을 로컬 PNG로 다운로드 — 후속 도구(및 사용자의 검토)는 임시 URL이 아니라 출력 디렉터리의 파일을 기대합니다
- **`curl -o`에는 절대 경로 사용** — 배치 간 persistent-shell CWD에 의존하지 마세요. 조용한 함정: 파일이 잘못된 위치에 저장되고 이후 의도한 경로에서 `ls`를 실행하면 파일이 보이지 않습니다. 자세한 내용은 7단계의 "다운로드 단계"를 참조하세요.
- 민감한 공인을 위해서는 양식화된 대안을 사용

- **2단계 확인 필수** — 건너뛰지 마세요
- **4/6단계 조건부** — 사용자가 2단계에서 요청한 경우에만
- **7.1단계 캐릭터 시트** — 여러 페이지 만화에 권장, 단순한 프리셋에는 선택 사항. PNG는 검토/재생성 보조 자료이며, (5단계에서 작성되는) 페이지 prompt는 PNG가 아니라 `characters/characters.md`의 텍스트 설명을 사용합니다. `image_generate`는 이미지를 시각적 입력으로 받지 않습니다.
- **비밀 정보 제거** — 출력 파일을 작성하기 전에 원본 콘텐츠에서 API 키, 토큰 또는 자격 증명을 검사합니다.
