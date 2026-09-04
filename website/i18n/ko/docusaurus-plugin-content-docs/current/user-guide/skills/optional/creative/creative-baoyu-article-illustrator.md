---
title: "Baoyu Article Illustrator — 유형 × 스타일 × 팔레트 일관성을 갖춘 기사 일러스트"
sidebar_label: "Baoyu Article Illustrator"
description: "유형 × 스타일 × 팔레트 일관성을 갖춘 기사 일러스트"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Baoyu Article Illustrator

유형 × 스타일 × 팔레트 일관성을 갖춘 기사 일러스트입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/baoyu-article-illustrator`로 설치 |
| 경로 | `optional-skills/creative/baoyu-article-illustrator` |
| 버전 | `1.57.0` |
| 작성자 | 宝玉 (JimLiu) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `article-illustration`, `creative`, `image-generation` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# Article Illustrator

[baoyu-article-illustrator](https://github.com/JimLiu/baoyu-skills)에서 Hermes Agent의 도구 생태계에 맞게 조정했습니다.

기사를 분석하고, 일러스트를 배치할 위치를 식별하며, **유형 × 스타일 × 팔레트** 일관성을 갖춘 이미지를 생성합니다.

## 사용하는 경우

사용자가 기사에 일러스트를 넣거나, 기사에 이미지를 추가하거나, 콘텐츠용 일러스트를 생성해 달라고 요청하거나, "为文章配图", "illustrate article", "add images"와 같은 표현을 사용하면 이 스킬을 실행합니다. 사용자는 기사(파일 경로 또는 붙여넣은 내용)를 제공하며, 유형·스타일·팔레트·밀도를 선택적으로 지정할 수 있습니다.

## 세 가지 차원

| 차원 | 제어 대상 | 예시 |
|------|----------|------|
| **유형** | 정보 구조 | infographic, scene, flowchart, comparison, framework, timeline |
| **스타일** | 렌더링 방식 | notion, warm, minimal, blueprint, watercolor, elegant |
| **팔레트** | 색 구성(선택 사항) | macaron, warm, neon — 스타일의 기본 색상을 덮어씀 |

자유롭게 조합할 수 있습니다: `type=infographic, style=vector-illustration, palette=macaron`.

또는 프리셋을 사용할 수 있습니다: `edu-visual` → 한 번에 유형 + 스타일 + 팔레트. [style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/style-presets.md)를 참조하세요.

## 유형

| 유형 | 적합한 대상 |
|------|----------|
| `infographic` | 데이터, 지표, 기술 내용 |
| `scene` | 서사, 감정 |
| `flowchart` | 프로세스, 워크플로 |
| `comparison` | 나란히 비교, 선택지 |
| `framework` | 모델, 아키텍처 |
| `timeline` | 역사, 발전 과정 |

## 스타일

핵심 스타일, 전체 갤러리, 유형 × 스타일 호환성은 [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/styles.md)를 참조하세요.

## 출력 구조

<!-- ascii-guard-ignore -->
```
{output-dir}/
├── source-{slug}.{ext}    # Only for pasted content
├── outline.md
├── prompts/
│   └── NN-{type}-{slug}.md
└── NN-{type}-{slug}.png
```
<!-- ascii-guard-ignore-end -->

**기본 출력 디렉터리**:

| 입력 | 출력 디렉터리 | Markdown 삽입 경로 |
|------|-------------|-------------------|
| 기사 파일 경로 | `{article-dir}/imgs/` | `imgs/NN-{type}-{slug}.png` |
| 붙여넣은 내용 | `illustrations/{topic-slug}/` (cwd) | `illustrations/{topic-slug}/NN-{type}-{slug}.png` |

사용자가 다른 레이아웃(예: 기사 옆에 이미지 배치 또는 `illustrations/` 하위 디렉터리)을 요청하면 그에 따릅니다.

**Slug**: 2~4단어, kebab-case. **충돌**: `-YYYYMMDD-HHMMSS`를 추가합니다.

## 핵심 원칙

- **은유가 아니라 개념을 시각화하세요** — 기사에서 은유(예: "电锯切西瓜")를 사용하더라도 문자 그대로의 이미지를 그리지 말고 그 아래의 개념을 표현합니다.
- **레이블에는 기사 데이터를 사용하세요** — 일반적인 자리 표시자가 아니라 기사의 실제 숫자, 용어, 인용문을 사용합니다.
- **프롬프트 파일은 재현성 기록입니다** — 이미지를 생성하기 전에 모든 일러스트에 대해 `prompts/` 아래에 저장된 프롬프트 파일이 있어야 합니다.
- **비밀 정보를 제거하세요** — 디스크에 기록하기 전에 소스 콘텐츠에서 API 키, 토큰 또는 자격 증명을 검색합니다.

## 워크플로

```
- [ ] Step 1: Detect reference images (if provided)
- [ ] Step 2: Analyze content
- [ ] Step 3: Confirm settings (clarify tool, one question at a time)
- [ ] Step 4: Generate outline
- [ ] Step 5: Generate prompts
- [ ] Step 6: Generate images (image_generate)
- [ ] Step 7: Finalize
```

### 1단계: 참조 이미지 감지

사용자가 참조 이미지(인라인으로 붙여넣은 경로, 첨부 파일 또는 URL)를 제공하면 다음을 수행합니다.

1. 각 참조에 대해 경로/URL과 스타일, 팔레트, 구도, 대상에 관해 묻는 질문을 사용하여 `vision_analyze`를 호출합니다. 반환된 설명을 `{output-dir}/references/NN-ref-{slug}.md`에 `write_file`로 기록합니다.
2. **`write_file` / `read_file`로 바이너리를 복사하려 하지 마세요** — 이 도구들은 텍스트 전용입니다. 기록을 위해 로컬 복사본이 필요하면 `terminal`(`cp "$src" "{output-dir}/references/NN-ref-{slug}.{ext}"`)을 사용합니다. 스킬 자체는 바이너리를 읽지 않고 vision 설명을 사용합니다.
3. `image_generate`는 이미지 입력을 받지 않으므로, 5단계에서 프롬프트에 포함할 vision 설명을 사용합니다.

전체 절차: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-1-detect-reference-images)의 **1단계 참조 이미지 감지**를 확인하세요.

### 2단계: 분석

| 분석 항목 | 출력 |
|----------|--------|
| 콘텐츠 유형 | 기술 / 튜토리얼 / 방법론 / 서사 |
| 목적 | 정보 전달 / 시각화 / 상상 |
| 핵심 주장 | 주요 요점 2~5개 |
| 위치 | 일러스트가 가치를 더하는 지점 |

소스를 읽고(파일 경로 → `read_file`, 또는 붙여넣은 텍스트) `{output-dir}/analysis.md`에 `write_file`로 분석을 기록합니다.

전체 절차: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-2-analyze)의 **2단계 분석**을 확인하세요.

### 3단계: 설정 확인

`clarify` 도구를 사용합니다. `clarify`는 한 번에 한 질문만 처리하므로 가장 중요한 질문부터 합니다. 사용자의 요청에 이미 답이 포함된 질문은 건너뜁니다.

| 순서 | 질문 | 선택지 |
|------|------|--------|
| Q1 | **프리셋 또는 유형** | [권장 프리셋], [대체 프리셋], 또는 수동 선택: infographic, scene, flowchart, comparison, framework, timeline, mixed |
| Q2 | **밀도** | minimal (1-2), balanced (3-5), per-section (권장), rich (6+) |
| Q3 | **스타일** *(Q1에서 프리셋을 선택한 경우 생략)* | [권장], minimal-flat, sci-fi, hand-drawn, editorial, scene, poster |
| Q4 | **팔레트** *(선택 사항)* | Default (style colors), macaron, warm, neon |
| Q5 | **언어** *(기사 언어가 모호한 경우에만)* | 기사 언어 / 사용자 언어 |

`clarify` 질문을 연속으로 2~3개 넘게 하지 마세요. 사용자가 요청에서 이미 지정했다면 전부 생략합니다.

전체 절차: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-3-confirm-settings)의 **3단계 설정 확인**을 확인하세요.

### 4단계: 개요 생성 → `outline.md`

`write_file`을 사용해 `{output-dir}/outline.md`를 프런트매터(type, density, style, palette, image_count 포함)와 일러스트마다 하나의 항목으로 저장합니다.

```yaml
## Illustration 1
**Position**: [section/paragraph]
**Purpose**: [why]
**Visual Content**: [what to show]
**Filename**: 01-infographic-concept-name.png
```

전체 템플릿: [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md#step-4-generate-outline)를 확인하세요.

### 5단계: 프롬프트 생성

**차단 조건**: 이미지를 생성하기 전에 모든 일러스트에 저장된 프롬프트 파일이 있어야 합니다 — 프롬프트 파일은 재현성 기록입니다.

각 일러스트에 대해 다음을 수행합니다.

1. [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/prompt-construction.md)에 따라 일러스트마다 프롬프트 파일을 만듭니다.
2. YAML 프런트매터와 함께 `write_file`을 사용해 `{output-dir}/prompts/NN-{type}-{slug}.md`에 저장합니다.
3. 프롬프트는 유형별 템플릿과 구조화된 섹션(ZONES / LABELS / COLORS / STYLE / ASPECT)을 반드시 사용해야 합니다.
4. LABELS에는 기사에 특정한 데이터(실제 숫자, 용어, 지표, 인용문)를 반드시 포함해야 합니다.
5. 프롬프트 프런트매터에 따라 참조 자료(`direct`/`style`/`palette`)를 처리합니다 — `direct`를 사용하는 경우 `image_generate`가 참조 이미지 입력을 받지 않으므로 참조 이미지의 텍스트 설명을 프롬프트에 포함합니다.

### 6단계: 이미지 생성

각 프롬프트 파일에 대해 다음을 수행합니다.

1. `image_generate(prompt=..., aspect_ratio=...)`를 호출합니다. `image_generate`는 이미지 URL을 포함한 JSON 결과를 반환하며 디스크에 기록하지 않고 출력 경로도 받지 않습니다.
2. 프롬프트의 `ASPECT`를 `image_generate`의 enum에 매핑합니다: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`. 사용자 정의 비율은 가장 가까운 이름의 화면 비율로 매핑합니다.
3. `terminal`을 통해 반환된 URL을 `{output-dir}/NN-{type}-{slug}.png`로 다운로드합니다(예: `curl -sSL -o "{output-dir}/NN-{type}-{slug}.png" "{url}"`).
4. 생성에 실패하면 한 번 자동으로 재시도합니다.

참고: 기반 이미지 생성 백엔드는 사용자가 구성하며(기본값: FAL FLUX 2 Klein 9B), `image_generate`를 통해 에이전트가 선택할 수 없습니다. 라우팅을 기대하며 프롬프트에 모델 이름을 작성하지 마세요.

### 7단계: 마무리

해당 단락 다음에 `![description](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/{relative-path}/NN-{type}-{slug}.png)`를 삽입합니다. Alt 텍스트는 기사의 언어로 간결한 설명을 작성합니다.

다음과 같이 보고합니다.

```
Article Illustration Complete!
Article: [path] | Type: [type] | Density: [level] | Style: [style] | Palette: [palette or default]
Images: X/N generated
```

## 수정

| 작업 | 단계 |
|------|------|
| 편집 | 프롬프트 업데이트 → 재생성 → 참조 업데이트 |
| 추가 | 위치 지정 → 프롬프트 작성 → 생성 → 개요 업데이트 → 삽입 |
| 삭제 | 파일 삭제 → 참조 제거 → 개요 업데이트 |

## 참고 자료

| 파일 | 내용 |
|------|------|
| [references/workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/workflow.md) | 세부 절차 |
| [references/usage.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/usage.md) | 호출 예시 |
| [references/styles.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/styles.md) | 스타일 갤러리 + 팔레트 갤러리 |
| [references/style-presets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/style-presets.md) | 프리셋 단축키(유형 + 스타일 + 팔레트) |
| [references/prompt-construction.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/baoyu-article-illustrator/references/prompt-construction.md) | 프롬프트 템플릿 |

## 주의 사항

1. **데이터 무결성이 최우선입니다** — 소스 통계를 절대 요약하거나, 바꾸어 쓰거나, 변경하지 마세요. "73% increase"는 "73% increase"로 유지합니다.
2. **비밀 정보를 제거하세요** — 출력 파일에 포함하기 전에 API 키, 토큰 또는 자격 증명을 검색합니다.
3. **은유를 문자 그대로 표현하지 마세요** — 그 아래의 개념을 시각화합니다.
4. **프롬프트 파일은 필수입니다** — 저장된 프롬프트 파일 없이 이미지를 생성하지 마세요. 이 파일이 있어야 나중에 이미지를 재생성하거나 백엔드를 전환할 수 있습니다.
5. **`image_generate` 화면 비율** — 이 도구는 `landscape`, `portrait`, `square`를 지원합니다. 사용자 정의 비율은 가장 가까운 옵션으로 매핑합니다.
6. **`image_generate`는 로컬 파일이 아닌 URL을 반환합니다** — 로컬 이미지 경로를 기사에 삽입하기 전에 항상 `terminal`(`curl`)을 통해 다운로드합니다.
7. **에이전트는 백엔드를 선택할 수 없습니다** — `image_generate`는 사용자가 구성한 모델(기본값: FAL FLUX 2 Klein 9B)을 사용합니다. 라우팅을 기대하며 프롬프트에 `"use <model> to generate this"`를 작성하지 마세요.
