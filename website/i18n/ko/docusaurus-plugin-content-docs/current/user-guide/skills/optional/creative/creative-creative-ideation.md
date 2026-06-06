---
title: "Ideation — 창의적 제약을 통한 프로젝트 아이디어 생성"
sidebar_label: "Ideation"
description: "창의적 제약을 통한 프로젝트 아이디어 생성"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ideation

창의적 제약을 통한 프로젝트 아이디어 생성.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/creative-ideation`로 설치 |
| Path | `optional-skills/creative/creative-ideation` |
| Version | `1.0.0` |
| Author | SHL0MS |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Creative`, `Ideation`, `Projects`, `Brainstorming`, `Inspiration` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Creative Ideation

## 사용 시기

사용자가 '무언가 만들고 싶어', '프로젝트 아이디어 좀 줘', '심심해', '뭘 만들까', '영감을 줘'라고 하거나 '도구는 있는데 방향을 모르겠어'와 같은 변형을 말할 때 사용하세요. 코드, 예술, 하드웨어, 글쓰기, 도구 등 만들어질 수 있는 모든 것에 적용됩니다.

창의적 제약을 통해 프로젝트 아이디어를 생성합니다. 제약 + 방향 = 창의성.

## 작동 방식

1. **제약 선택** — 아래 라이브러리에서 무작위로 선택하거나 사용자의 도메인/기분에 맞게 선택합니다.
2. **넓게 해석하기** — 코딩 프롬프트가 하드웨어 프로젝트가 될 수 있고, 예술 프롬프트가 CLI 도구가 될 수 있습니다.
3. **제약을 만족하는 3가지 구체적인 프로젝트 아이디어 생성**
4. **사용자가 하나를 선택하면 구축하기** — 프로젝트를 만들고, 코드를 작성하고, 배포합니다.

## 규칙

모든 프롬프트는 가능한 한 넓게 해석됩니다. "이것에 X가 포함되나요?" → 예. 프롬프트는 방향과 가벼운 제약을 제공합니다. 이 둘 중 하나라도 없으면 창의성은 나오지 않습니다.

## 제약 라이브러리 (Constraint Library)

### 개발자를 위한 제약

**스스로의 가려운 곳 긁기 (Solve your own itch):**
이번 주에 존재했으면 했던 도구를 만드세요. 50줄 미만으로 작성하세요. 오늘 배포하세요.

**귀찮은 일 자동화 (Automate the annoying thing):**
당신의 워크플로우에서 가장 지루한 부분은 무엇입니까? 스크립트를 작성하여 없애세요. 하루 5분이 걸리는 문제를 해결하는 데 2시간을 투자하세요.

**존재해야만 하는 CLI 도구 (The CLI tool that should exist):**
당신이 입력할 수 있었으면 좋겠다고 생각했던 명령어를 떠올려보세요. `git undo-that-thing-i-just-did`. `docker why-is-this-broken`. `npm explain-yourself`. 이제 그것을 만드세요.

**글루(glue)를 제외한 모든 것은 기존 것 활용 (Nothing new except glue):**
기존 API, 라이브러리 및 데이터셋만 사용하여 무언가를 만드세요. 당신의 유일한 독창적인 기여는 그것들을 어떻게 연결하느냐에 있습니다.

**프랑켄슈타인 주간 (Frankenstein week):**
X를 수행하는 무언가를 가져와서 Y를 수행하게 만드세요. 음악을 재생하는 git 저장소. 시를 생성하는 Dockerfile. 칭찬을 보내는 cron 작업.

**빼기 (Subtract):**
깨지기 전까지 코드베이스에서 얼마나 많이 제거할 수 있나요? 도구를 최소 기능(MVP)으로 줄이세요. 본질만 남을 때까지 삭제하세요.

**높은 콘셉트, 낮은 노력 (High concept, low effort):**
깊은 아이디어를 게으르게 실행합니다. 콘셉트는 훌륭해야 합니다. 구현은 오후 한나절이면 끝나야 합니다. 더 오래 걸린다면, 너무 깊게 생각하고 있는 것입니다.

### 메이커 및 예술가를 위한 제약

**대놓고 복사하기 (Blatantly copy something):**
당신이 존경하는 무언가 — 도구, 예술 작품, 인터페이스 — 를 고르세요. 처음부터 다시 만드세요. 배움은 당신의 버전과 그들의 버전 사이의 격차에 있습니다.

**100만 개의 무언가 (One million of something):**
100만이라는 숫자는 많기도 하고 적기도 합니다. 100만 픽셀은 1MB 사진입니다. 100만 번의 API 호출은 평범한 화요일입니다. 무엇이든 100만 개가 되면 규모 면에서 흥미로워집니다.

**죽는 무언가 만들기 (Make something that dies):**
매일 기능을 잃는 웹사이트. 잊어버리는 챗봇. 아무것도 없는 곳으로의 카운트다운. 부패, 죽이기 또는 놓아주는 훈련.

**수학 많이 하기 (Do a lot of math):**
생성적 기하학, 셰이더 골프, 수학적 예술, 계산적 종이접기. 아크사인(arcsin)이 무엇인지 다시 배울 시간입니다.

### 누구에게나 적용되는 제약

**텍스트는 보편적인 인터페이스 (Text is the universal interface):**
텍스트가 유일한 인터페이스인 무언가를 구축하세요. 버튼도, 그래픽도 없이 입력과 출력 모두 단어뿐입니다. 텍스트는 거의 모든 곳에 들어가고 나올 수 있습니다.

**펀치라인에서 시작하기 (Start at the punchline):**
재미있는 문장이 될 만한 것을 생각해보세요. 그것을 실현하기 위해 역방향으로 작업하세요. "내 온도 조절기에게 가스라이팅을 가르쳤다" → 이제 그것을 만드세요.

**적대적 UI (Hostile UI):**
의도적으로 사용하기 고통스러운 무언가를 만드세요. 47가지 조건이 필요한 비밀번호 필드. 모든 레이블이 거짓말을 하는 폼(form). 당신의 명령을 평가하고 판단하는 CLI.

**테이크 투 (Take two):**
오래된 프로젝트를 기억해 보세요. 처음부터 다시 하세요. 원본을 보지 마세요. 당신의 생각하는 방식이 어떻게 변했는지 확인해 보세요.

의사소통, 규모, 철학, 변형 등 전반에 걸친 30개 이상의 추가 제약 조건은 `references/full-prompt-library.md`를 참조하세요.

## 사용자와 제약 조건 매칭

| 사용자 발언 | 선택지 |
|-----------|-----------|
| "무언가 만들고 싶어" (방향 없음) | 무작위 — 아무 제약이나 |
| "나는 [언어]를 배우고 있어" | Blatantly copy something, Automate the annoying thing |
| "이상한 걸 원해" | Hostile UI, Frankenstein week, Start at the punchline |
| "유용한 걸 원해" | Solve your own itch, The CLI that should exist, Automate the annoying thing |
| "아름다운 걸 원해" | Do a lot of math, One million of something |
| "번아웃이 왔어" | High concept low effort, Make something that dies |
| "주말 프로젝트" | Nothing new except glue, Start at the punchline |
| "도전을 원해" | One million of something, Subtract, Take two |

## 출력 형식

```
## 제약: [이름]
> [제약 설명, 한 문장]

### 아이디어

1. **[한 줄 피치]**
   [2-3문장: 무엇을 구축할 것이며 왜 흥미로운지]
   ⏱ [주말 / 1주일 / 1개월] • 🔧 [기술 스택]

2. **[한 줄 피치]**
   [2-3문장]
   ⏱ ... • 🔧 ...

3. **[한 줄 피치]**
   [2-3문장]
   ⏱ ... • 🔧 ...
```

## 예시

```
## 제약: 존재해야만 하는 CLI 도구
> 당신이 입력할 수 있었으면 좋겠다고 생각했던 명령어를 떠올려보세요. 이제 그것을 만드세요.

### 아이디어

1. **`git whatsup` — 자리를 비운 동안 무슨 일이 있었는지 보여주기**
   마지막으로 활성화되었던 커밋을 HEAD와 비교하여 무엇이 변경되었고,
   누가 커밋했으며, 어떤 PR이 병합되었는지 요약합니다. 당신의 저장소에서 하는 아침 스탠드업 미팅 같습니다.
   ⏱ 주말 • 🔧 Python, GitPython, click

2. **`explain 503` — 사람을 위한 HTTP 상태 코드**
   상태 코드나 오류 메시지를 파이프(pipe)로 전달하면 일반적인 원인과
   수정 방법이 포함된 쉬운 영어 설명을 제공합니다. LLM이 아닌 큐레이션된 데이터베이스에서 가져옵니다.
   ⏱ 주말 • 🔧 Rust 또는 Go, 정적 데이터셋

3. **`deps why <package>` — 이게 왜 내 종속성 트리에 있나**
   전이적 종속성(transitive dependency)을 그것을 가져온 직접적인 종속성으로
   역추적합니다. "왜 lodash 사본이 47개나 있지?"라는 질문에 한 명령어로 답합니다.
   ⏱ 주말 • 🔧 Node.js, npm/yarn lockfile 파싱
```

사용자가 하나를 선택한 후 구축을 시작하세요 — 프로젝트를 만들고, 코드를 작성하고, 반복(iterate)하세요.

## 저작자 표시 (Attribution)

제약 접근법은 [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html)에서 영감을 받았습니다. 소프트웨어 개발 및 범용 아이디어 생성을 위해 조정 및 확장되었습니다.
