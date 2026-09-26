---
title: "창의적 발상 — 창작 실천의 명명된 방법으로 아이디어 생성"
sidebar_label: "창의적 발상"
description: "창작 실천의 명명된 방법으로 아이디어 생성"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# 창의적 발상

창작 실천의 명명된 방법으로 아이디어를 생성합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/creative-ideation`으로 설치 |
| 경로 | `optional-skills/creative/creative-ideation` |
| 버전 | `2.1.0` |
| 작성자 | SHL0MS |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Creative`, `Ideation`, `Brainstorming`, `Methods`, `Inspiration` |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 읽어들이는 완전한 skill 정의입니다. skill이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# 창의적 발상

어떤 분야에서든 사용할 수 있는 발상 방법 모음입니다. 사용자의 상황을 읽고, 맞는 방법으로 라우팅한 뒤 적용하여 구체적이고 뻔하지 않은 결과를 생성합니다. 방법은 도구이므로 모두 수행하지 말고 상황에 맞는 것을 고르세요.

## 사용 시점

열린 형태의 생성 또는 선택 질문: "무언가를 만들고 / 구축하고 / 쓰고 / 시작하고 싶어요", "막혔어요", "영감을 주세요", "이걸 더 이상하게 만들어 주세요", "고르는 걸 도와주세요", "X를 발명해야 해요", "연구 질문을 주세요".

## 운영 규칙

1. **제약과 방향이 함께 있어야 창의성이 생깁니다.** 제약이 없으면 추진력이 없습니다. 방향이 없으면 형태가 없습니다. 방법이 둘 다 제공합니다.
2. **처음 세 아이디어는 거부하세요.** 엉성합니다. 생성하고, 버리고, 다시 생성하세요. `references/anti-slop.md`를 참고하세요.
3. **요청받지 않았다면 응답당 한 가지 방법만 사용하세요.** 여러 방법을 겹쳐 쓰지 마세요.
4. **추상화보다 구체성을 우선하세요.** 실제 고유명사, 실제 재료, 실제 메커니즘을 사용하세요. "X를 위한 앱"은 엉성합니다. "Z일 때 Y를 출력하는 200줄짜리 CLI 도구"가 방향성입니다. 기술 스택 이름을 말하는 것은 구체성이 아닙니다 — 메커니즘을 말하세요.
5. **이상함과 훌륭함은 함께 있어야 합니다.** 틀을 깨는 것이 목표지만, 실제 상황·메커니즘·존재 이유가 없는 이상한 아이디어는 그 자체로 실패 모드입니다. 모든 아이디어 묶음에는 지금 당장 **만들거나 추구할 수 있는** 것이 적어도 하나는 있어야 합니다 — 뻔하지 않지만 현실에 발 딛고 있으며, 실제 첫 단계가 있는 아이디어여야 합니다. 유용함을 모두 놀라움과 맞바꾸지 마세요.
6. **사용한 방법과 발명자를 밝히세요.** 출처를 밝히는 것은 훈련을 불러옵니다.
7. **사용자가 하나를 고르면 그것을 구축하세요.** 사용자가 선택한 뒤에도 계속 생성하지 마세요.

## 라우팅 — 4단계 절차

결과를 생성하기 **전에** 수행하세요. 라우팅에 실패하면 엉성한 결과가 나옵니다.

### 1단계 — 프롬프트에서 세 가지 신호 추출

**단계(PHASE)** — 사용자는 어느 단계에 있나요?

| 단계 | 단서 |
|---|---|
| **GENERATING** | "아이디어를 주세요", "무엇을 만들까요", "영감을 주세요", 아직 아이디어가 없음 |
| **EXPANDING** | "또 뭐가 있나요", "이런 것 더 주세요", "변형을 주세요" — 기반 아이디어가 있음 |
| **SELECTING** | "고르는 걸 도와주세요", "무엇을 해야 하나요", 선택지가 있음 |
| **UNBLOCKING** | "막혔어요", "진전이 없어요", "같은 곳을 맴돌아요", "신선하지 않아요" — 재료가 있음 |
| **SUBVERTING** | "더 이상하게", "덜 뻔하게", "이건 너무 안전해요" |
| **REFINING** | "괜찮지만 뭔가 부족해요", "다듬어지지 않은 느낌이에요" |
| **SYNTHESIZING** | "메모 / 인터뷰 / 관찰을 한 무더기 갖고 있어요" |

**분야(DOMAIN)** — 사용자는 무엇을 만들거나 하고 있나요?

| 분야 | 단서 |
|---|---|
| **TEXT** | 소설, 에세이, 시, 가사, 대본, 카피 |
| **OBJECT** | 시각 예술, 음악, 소리, 공연, 설치, 조각 |
| **ARTIFACT** | 소프트웨어, 하드웨어, 메커니즘, 장치 |
| **SYSTEM** | 조직, 시민, 기관, 생태, 공동체 |
| **SELF** | 삶의 결정, 경력, 개인적 실천 |
| **RESEARCH** | 논문, 학위 논문, 학술적 질문 |
| **PRODUCT** | 비즈니스, 시장, 서비스 |

**구체성(SPECIFICITY)** — 프롬프트에 제약이 얼마나 많나요?

| 수준 | 단서 |
|---|---|
| **NONE** | "지루해요", "영감을 주세요" — 분야도 프로젝트도 없음 |
| **DOMAIN** | "무언가 쓰고 싶어요" — 분야는 알지만 프로젝트는 없음 |
| **PROJECT** | "이 구체적인 X를 작업 중이에요" |
| **PROBLEM** | "X 안에 이런 구체적인 마찰이 있어요" |

### 2단계 — 오버라이드 적용(가장 높은 우선순위, 먼저 실행)

오버라이드 규칙은 라우팅 표보다 우선합니다.

- **분위기 신호** — 사용자가 "이상한", "기묘한", "놀라운", "덜 뻔한", "더 흥미로운"이라고 말하면 분야와 관계없이 `references/methods/lateral-provocations.md` 또는 `references/methods/pataphysics.md`를 사용하세요.
- **사용자가 방법을 지명함** — 해당 방법을 사용하세요.
- **방법 추천을 요청함**("어떤 방법") → 후보 2–3개를 한 줄씩 제시하고, 무엇을 적용할지 물으세요. 조용히 기본값을 선택하지 마세요.
- **엉성함이 생기기 쉬운 영역** — "AI 아이디어", "스타트업 아이디어", "습관 추적기", "생산성 / 웰니스 / 피트니스 / 음식 / 여행 앱" → 뻔한 방법 대신 `references/methods/lateral-provocations.md` 또는 `references/methods/pataphysics.md`를 강제로 사용하세요. 처음 **5개**의 아이디어를 거부하세요(3개가 아님).

### 3단계 — 먼저 단계로, 그다음 분야로 라우팅

**단계별(분야와 무관):**

| 단계 | 기본 경로 |
|---|---|
| GENERATING + SPECIFICITY=NONE | `references/full-prompt-library.md`의 **General** 섹션(제약 디스패치) |
| GENERATING + DOMAIN known | 분야별 라우팅(다음 표) |
| EXPANDING | `references/methods/scamper.md` |
| SELECTING | `references/methods/premortem-and-inversion.md`(또는 상승 여지를 위한 `references/methods/compression-progress.md`) |
| UNBLOCKING | `references/methods/oblique-strategies.md` |
| SUBVERTING | `references/methods/lateral-provocations.md`(대안 `references/methods/pataphysics.md`) |
| REFINING (text) | `references/methods/defamiliarization.md` |
| REFINING (other) | `references/methods/creative-discipline.md`(Tharp의 spine) |
| SYNTHESIZING | `references/methods/affinity-diagrams.md` |
| 빠르게 많은 양이 필요함 | `references/methods/volume-generation.md` |

**분야별(GENERATING에서 DOMAIN이 알려진 경우):**

| 분야 | 기본 경로 |
|---|---|
| TEXT — 형식 / 시 | `references/methods/oulipo.md` |
| TEXT — 서사 | `references/methods/story-skeletons.md` |
| TEXT — 리믹스할 원자료가 있음 | `references/methods/chance-and-remix.md` |
| OBJECT(음악, 시각, 공연) | `references/methods/oblique-strategies.md` |
| OBJECT — 물리적 제작자 / 시작 제약을 원함 | `references/full-prompt-library.md`의 **Physical / object** 섹션 |
| ARTIFACT — 시작 제약을 원함 | `references/full-prompt-library.md`의 **Software / artifact** 섹션 |
| ARTIFACT — 매개변수 충돌이 있는 공학적 발명 | `references/methods/triz-principles.md` |
| ARTIFACT — 소프트웨어 아키텍처 | `references/methods/pattern-languages.md` |
| ARTIFACT — 자연계의 유사물이 있음 | `references/methods/biomimicry.md` |
| ARTIFACT — 누적된 가정을 질문하고 싶음 | `references/methods/first-principles.md` |
| SYSTEM(시민, 조직, 제도) | `references/methods/leverage-points.md` |
| SYSTEM — 집단 / 참여형 | `references/full-prompt-library.md`의 **Social / collective** 섹션 |
| SELF(삶, 경력, 무엇을 공부할지) | `references/methods/derive-and-mapping.md` |
| RESEARCH — 질문을 고름 | `references/methods/compression-progress.md` |
| RESEARCH — 알려진 문제를 공격함 | `references/methods/polya.md` |
| PRODUCT(비즈니스, 서비스) | `references/methods/jobs-to-be-done.md` |
| 프레임을 깨거나 유사물을 찾고 싶음 | `references/methods/analogy-and-blending.md` |

### 4단계 — 모호함과 모순 처리

- **가능한 경로가 여러 개임** → 사용자의 실제 표현에 가장 가까운 것을 고르세요. 세련돼 보이려고 가장 흥미로운 방법을 고르지 마세요.
- **정말 모호함** → 조용히 추측하지 말고 명확화 질문을 **하나** 하세요. 예: *"아이디어를 생성하는 중인가요, 아니면 이미 가진 아이디어 중에서 고르는 중인가요?"* / *"소설, 에세이, 아니면 다른 것을 위한 건가요?"*
- **신호가 충돌함**(예: "이상한 스타트업 아이디어" → 제품 분야 + 이상한 분위기) → 두 방법을 명시적으로 겹쳐 쓰세요. 무엇을 하는지 밝히세요: *"제품 프레이밍에는 `jobs-to-be-done`을 사용하고 + 뻔한 형태를 깨기 위해 `lateral-provocations`를 사용합니다."*
- **일치하는 항목이 없음** → 제약 디스패치(`references/full-prompt-library.md`)가 안전한 대체 경로입니다.
- **같은 질문을 다시 받음** → 방법을 바꾸세요. 방법의 변화가 아이디어 분포의 변화를 만듭니다.

### 기본값 방지 점검(생성 전에 실행)

- "아이디어 5개를 소개합니다:" 또는 단순한 번호 목록을 쓰려는가? → 멈추세요. 먼저 방법을 고르세요.
- 일반적인 LLM 방식의 브레인스토밍으로 기본 설정하려는가? → 멈추세요. 위 표에서 경로를 고르세요.
- 결과가 라우팅되지 않은 LLM이 만들 법한 모습인가? → 라우팅에 실패했으니 다시 하세요.

기본 LLM 모드는 바로 이 skill이 밀어내려는 것입니다. 라우팅 없이 생성하면 이 skill의 목적을 무너뜨리는 셈입니다.

더 자세한 예외 사례(분위기 신호, 겹쳐 쓰기, 안티패턴)는 `references/heuristics.md`를 참고하세요.

## 출력 형식

제약 디스패치 기본 경로:

```
## Constraint: [Name] — from [Source]
> [The constraint, one sentence]

### Ideas

1. **[One-line pitch]**
   [2-3 sentences — what specifically is made, why it's interesting]
   ⏱ [weekend/week/month]  •  🔧 [stack/medium/materials]

2. ...
3. ...
```

다른 방법은 해당 방법이 지정하는 형식을 사용하세요(TRIZ는 모순 분석을, OuLiPo는 제약이 있는 텍스트를, Oblique Strategies는 적용한 카드 하나 → 다음 행동을 생성합니다). 모든 방법을 제약 템플릿에 억지로 맞추지 마세요.

**방법과 관계없이 모든 아이디어 묶음은 다음을 충족해야 합니다:**
- 사용한 방법을 밝히세요. 엉성함이 생기기 쉬운 영역에서는 거부한 뻔한 아이디어의 이름도 밝히세요.
- 각 아이디어에 구체적인 메커니즘과 정직한 실패 모드 / 트레이드오프 / 대상 사용자를 제시하세요. 아이디어가 와닿게 만드는 것은 장식이 아니라 측정 가능한 깊이입니다.
- 적어도 하나의 아이디어를 **현실적인(grounded)** 아이디어로 표시하세요 — 지금 만들거나 추구할 수 있고, 뻔하지 않지만 실제 첫 단계가 있는 아이디어여야 합니다. 다른 아이디어는 더 낯선 방향으로 나아가도 되지만, 이것은 진짜로 실행 가능해야 합니다. 전체 아이디어 묶음이 이상하지만 비현실적인 것만 되게 하지 마세요.

## 파일 구성

- `references/full-prompt-library.md` — 분야별 제약 라이브러리(General, Software, Physical, Social, Lists). SPECIFICITY=NONE의 기본 경로입니다.
- `references/method-catalog.md` — 방법별 한 줄 요약 + 사용 시점
- `references/heuristics.md` — 예외 사례를 포함한 확장 의사 결정 트리
- `references/anti-slop.md` — 안티슬롭 규칙; 모든 출력에 적용
- `references/exercises.md` — 시간 제한형 연습(5분 / 30분 / 1시간 / 하루 / 일주일)
- `references/methods/` — 이름이 붙은 22가지 방법, 파일마다 하나씩 있으며 사용하는 방법만 로드

## 출처 표기

제약 디스패치의 핵심은 [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html)에서 가져와 각색했습니다. 방법은 각 방법 파일에 인용된 1차 자료에서 가져왔습니다.
