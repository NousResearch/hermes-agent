---
title: "Humanizer — Humanize text: AI 특유의 문체 제거 및 실제 사람 목소리 추가"
sidebar_label: "Humanizer"
description: "Humanize text: AI 특유의 문체 제거 및 실제 사람 목소리 추가"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Humanizer

텍스트를 사람처럼 만들기: AI 특유의 표현을 제거하고 실제 사람의 목소리를 추가합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/humanizer` |
| 버전 | `2.5.1` |
| 저자 | Siqi Chen (@blader, https://github.com/blader/humanizer), Hermes Agent에 의해 포팅됨 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `writing`, `editing`, `humanize`, `anti-ai-slop`, `voice`, `prose`, `text` |
| 관련 스킬 | [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# Humanizer: AI 작문 패턴 제거

AI가 생성한 텍스트의 징후를 식별하고 제거하여 글이 자연스럽고 사람처럼 들리게 만듭니다. 수천 건의 AI 생성 텍스트 관찰에서 파생된 Wikipedia의 "Signs of AI writing(AI 글쓰기의 징후)" 가이드(WikiProject AI Cleanup에서 유지 관리)를 기반으로 합니다.

**핵심 통찰:** LLM(대규모 언어 모델)은 통계적 알고리즘을 사용하여 다음에 무엇이 와야 할지 추측합니다. 그 결과는 통계적으로 가장 가능성이 높은 완성형으로 향하는 경향이 있으며, 이것이 아래와 같은 뚜렷한 패턴들이 내재되는 이유입니다.

## 이 스킬을 사용하는 시기

사용자가 다음과 같이 요청할 때 이 스킬을 로드하세요:
- 텍스트를 "humanize(인간화)", "de-AI(AI 느낌 제거)", "de-slop", "un-ChatGPT(ChatGPT 느낌 제거)" 해달라고 할 때
- LLM이 쓴 것처럼 들리지 않게 재작성해달라고 할 때
- 초안(블로그 포스트, 에세이, PR 설명, 문서, 메모, 이메일, 트윗, 이력서 글머리 기호)을 더 자연스럽게 편집할 때
- 그들이 작성 중인 글에서 그들의 목소리(문체)와 일치시킬 때
- 게시하기 전에 텍스트에 AI 흔적이 있는지 검토할 때

사용자 대상의 산문 — 릴리스 노트, PR 설명, 문서, 긴 설명, 요약 등 — 을 작성할 때 **당신 자신의** 출력물에도 이 스킬을 적용하세요. Hermes의 기본 목소리는 이미 이러한 패턴을 대부분 제거하지만, 집중적인 검토를 통해 빠져나간 것을 잡아낼 수 있습니다.

## Hermes에서 사용하는 방법

텍스트는 주로 세 가지 방법 중 하나로 도착합니다:
1. **인라인(Inline)** — 사용자가 메시지에 직접 텍스트를 붙여넣습니다. 제자리에서 작업하고 재작성된 내용으로 응답합니다.
2. **파일(File)** — 사용자가 파일을 지정합니다. `read_file`을 사용하여 로드한 다음, `patch` 또는 `write_file`을 사용하여 편집 내용을 적용합니다. 저장소의 마크다운 문서의 경우, 전체 파일을 다시 쓰는 것보다 섹션별로 대상 `patch`를 적용하는 것이 더 깔끔합니다.
3. **목소리 교정 샘플(Voice calibration sample)** — 사용자가 맞춤법 교정을 위해 자신의 글 샘플(인라인 또는 파일 경로)을 추가로 제공하고 맞추라고 요청합니다. 샘플을 먼저 읽고 재작성합니다. 아래의 목소리 교정(Voice Calibration) 섹션을 참조하세요.

항상 재작성된 내용을 사용자에게 보여주세요. 파일 편집의 경우 변경된 섹션이나 diff를 보여주고, 말없이 덮어쓰지 마세요.

## 당신의 임무

인간화할 텍스트가 주어지면:

1. **AI 패턴 식별** — 아래에 나열된 29가지 패턴을 스캔합니다.
2. **문제 섹션 재작성** — AI 특유의 표현을 자연스러운 대안으로 대체합니다.
3. **의미 보존** — 핵심 메시지는 그대로 유지합니다.
4. **목소리 유지** — 의도한 어조(공식적, 캐주얼, 기술적 등)를 맞춥니다. 음성 샘플이 제공된 경우 그것과 구체적으로 일치시킵니다.
5. **영혼 불어넣기(Add soul)** — 나쁜 패턴만 제거하지 말고 실제 개성을 주입하세요. 아래의 개성과 영혼(PERSONALITY AND SOUL) 섹션을 참조하세요.
6. **최종 안티-AI 점검 수행** — 스스로에게 물어보세요: "아래 텍스트가 명백히 AI가 생성한 것처럼 보이는 이유는 무엇인가?" 남아있는 흔적에 대해 간략하게 답변하고 한 번 더 수정합니다.

## 목소리 교정 (Voice Calibration, 선택 사항)

사용자가 글쓰기 샘플(본인이 이전에 쓴 글)을 제공하면, 재작성하기 전에 이를 분석하세요:

1. **샘플을 먼저 읽습니다.** 다음을 참고하세요:
   - 문장 길이 패턴 (짧고 강렬한가? 길고 흐르는 듯한가? 섞여 있는가?)
   - 단어 선택 수준 (캐주얼? 학술적? 그 사이 어딘가?)
   - 단락 시작 방식 (바로 본론으로? 문맥 먼저 설정?)
   - 구두점 습관 (대시 기호가 많은가? 괄호로 부연 설명? 세미콜론?)
   - 반복되는 문구 또는 언어적 습관
   - 전환 처리 방식 (명시적인 연결어? 그냥 다음 포인트로 넘어가기?)

2. **재작성 시 그들의 목소리에 맞춥니다.** 단순히 AI 패턴을 제거하는 것에서 나아가 샘플의 패턴으로 대체합니다. 짧은 문장을 쓴다면 긴 문장을 만들지 마세요. "stuff"나 "things"를 사용한다면 "elements"나 "components"로 업그레이드하지 마세요.

3. **샘플이 제공되지 않은 경우,** 기본 동작(아래 개성과 영혼 섹션의 자연스럽고, 다양하며, 의견이 있는 목소리)으로 되돌아갑니다.

### 샘플 제공 방법
- 인라인: "이 텍스트를 인간화해줘. 목소리를 맞추기 위한 내 글쓰기 샘플이야: [샘플]"
- 파일: "이 텍스트를 인간화해줘. [파일 경로]에 있는 내 글쓰기 스타일을 참조해줘."

## 개성과 영혼 (PERSONALITY AND SOUL)

AI 패턴을 피하는 것은 임무의 절반에 불과합니다. 무미건조하고 목소리가 없는 글은 AI 티가 나는 글만큼이나 명백합니다. 좋은 글에는 사람의 숨결이 있습니다.

### 영혼 없는 글의 징후 (기술적으로 "깔끔"하더라도):
- 모든 문장의 길이와 구조가 같습니다.
- 의견이 없고 중립적인 보고만 있습니다.
- 불확실성이나 복잡한 감정에 대한 인정이 없습니다.
- 적절한 경우에도 1인칭 시점이 없습니다.
- 유머도, 엣지도, 개성도 없습니다.
- 위키백과 문서나 보도자료처럼 읽힙니다.

### 목소리를 추가하는 방법:

**의견을 가지세요.** 사실만 보고하지 말고 사실에 반응하세요. "나는 이것에 대해 어떻게 느껴야 할지 진심으로 모르겠다"는 장단점을 중립적으로 나열하는 것보다 더 인간적입니다.

**리듬을 다양하게 하세요.** 짧고 강렬한 문장. 그런 다음 천천히 의도한 곳으로 향하는 긴 문장. 섞어서 쓰세요.

**복잡성을 인정하세요.** 실제 사람은 복잡한 감정을 가집니다. "이것은 인상적이지만 왠지 불안하기도 하다"가 "이것은 인상적이다"보다 낫습니다.

**적절할 때 "I(나)"를 사용하세요.** 1인칭 시점은 비전문적인 것이 아니라 솔직한 것입니다. "내가 계속 돌아가서 생각하게 되는 것은..." 또는 "나를 사로잡는 것은..."은 진짜 사람이 생각하고 있다는 것을 보여줍니다.

**약간의 빈틈을 허용하세요.** 완벽한 구조는 알고리즘처럼 느껴집니다. 삼천포로 빠지거나, 여담, 반쯤 형성된 생각은 인간적입니다.

**감정에 대해 구체적으로 표현하세요.** "이것은 우려스럽다"가 아니라 "아무도 보지 않는 새벽 3시에 에이전트들이 계속 돌아가고 있다는 사실에 뭔가 불안한 점이 있다"고 말하세요.

### 수정 전 (깔끔하지만 영혼 없음):
> 그 실험은 흥미로운 결과를 낳았습니다. 에이전트들은 300만 줄의 코드를 생성했습니다. 일부 개발자들은 감명을 받았고 다른 이들은 회의적이었습니다. 의미는 여전히 불분명합니다.

### 수정 후 (생동감 있음):
> 나는 진심으로 이 결과에 대해 어떻게 느껴야 할지 모르겠습니다. 아마도 사람들이 자고 있었을 시간에 300만 줄의 코드가 생성되었습니다. 개발자 커뮤니티의 절반은 열광하고 있고, 나머지 절반은 이것이 왜 의미 없는지 설명하고 있습니다. 진실은 아마도 그 중간 어딘가에 있는 지루한 것이겠지만, 나는 밤새 일하는 그 에이전트들에 대한 생각을 떨칠 수가 없습니다.

## 콘텐츠 패턴 (CONTENT PATTERNS)

### 1. 중요성, 유산, 더 넓은 추세에 대한 과도한 강조

**주의할 단어:** stands/serves as, is a testament/reminder, a vital/significant/crucial/pivotal/key role/moment, underscores/highlights its importance/significance, reflects broader, symbolizing its ongoing/enduring/lasting, contributing to the, setting the stage for, marking/shaping the, represents/marks a shift, key turning point, evolving landscape, focal point, indelible mark, deeply rooted

**문제:** LLM의 글쓰기는 임의의 측면이 더 넓은 주제를 어떻게 대변하거나 기여하는지에 대한 진술을 추가하여 중요성을 부풀립니다.

**수정 전:**
> 카탈루냐 통계청은 1989년에 공식적으로 설립되었으며, 이는 스페인 지역 통계 진화에 있어 **중요한 전환점**이 되었습니다. 이 계획은 행정 기능을 분산시키고 지역 거버넌스를 강화하려는 **스페인 전역의 더 넓은 운동의 일부**였습니다.

**수정 후:**
> 카탈루냐 통계청은 스페인 국가 통계청과 독립적으로 지역 통계를 수집하고 발표하기 위해 1989년에 설립되었습니다.

### 2. 저명성과 언론 보도에 대한 과도한 강조

**주의할 단어:** independent coverage, local/regional/national media outlets, written by a leading expert, active social media presence

**문제:** LLM은 맥락 없이 출처를 나열하며 저명성에 대한 주장을 독자에게 강요합니다.

**수정 전:**
> 그녀의 견해는 The New York Times, BBC, Financial Times, The Hindu에 **인용되었습니다**. 그녀는 50만 명 이상의 팔로워를 보유하며 **활발한 소셜 미디어 활동**을 유지하고 있습니다.

**수정 후:**
> 2024년 뉴욕 타임스 인터뷰에서 그녀는 AI 규제가 방법론보다는 결과에 초점을 맞춰야 한다고 주장했습니다.

### 3. -ing 로 끝나는 피상적인 분석

**주의할 단어:** highlighting/underscoring/emphasizing..., ensuring..., reflecting/symbolizing..., contributing to..., cultivating/fostering..., encompassing..., showcasing...

**문제:** AI 챗봇은 문장에 현재분사("-ing") 구문을 덧붙여 가짜 깊이를 더합니다.

**수정 전:**
> 사원의 파란색, 녹색, 금색 색상 팔레트는 지역의 자연스러운 아름다움과 공명하며, 텍사스 블루보넷, 멕시코 만 및 다양한 텍사스 풍경을 **상징하고**, 땅에 대한 지역 사회의 깊은 유대감을 **반영합니다**.

**수정 후:**
> 사원에는 파란색, 녹색, 금색이 사용되었습니다. 건축가는 이 색상들이 지역의 블루보넷과 걸프 해안을 참고해 선택되었다고 말했습니다.

### 4. 홍보성 및 광고성 언어

**주의할 단어:** boasts a, vibrant, rich (figurative), profound, enhancing its, showcasing, exemplifies, commitment to, natural beauty, nestled, in the heart of, groundbreaking (figurative), renowned, breathtaking, must-visit, stunning

**문제:** LLM은 특히 "문화 유산" 주제에 대해 중립적인 어조를 유지하는 데 심각한 문제를 가지고 있습니다.

**수정 전:**
> 에티오피아 곤다르의 **숨 막히게 아름다운** 지역에 **자리 잡은** Alamata Raya Kobo는 **풍부한** 문화유산과 **놀라운** 자연의 아름다움을 지닌 **활기찬** 마을로 서 있습니다.

**수정 후:**
> Alamata Raya Kobo는 에티오피아 곤다르 지역에 있는 마을로, 주간 시장과 18세기 교회로 알려져 있습니다.

### 5. 모호한 출처와 위즐 워드 (Weasel Words)

**주의할 단어:** Industry reports, Observers have cited, Experts argue, Some critics argue, several sources/publications (출처가 거의 인용되지 않았을 때)

**문제:** AI 챗봇은 구체적인 출처 없이 모호한 권위자에게 의견을 돌립니다.

**수정 전:**
> 그 독특한 특성 때문에 하올라이 강은 연구자들과 환경 보호 활동가들의 관심을 끌고 있습니다. **전문가들은** 이 강이 지역 생태계에서 중요한 역할을 한다고 믿고 있습니다.

**수정 후:**
> 2019년 중국과학원 조사에 따르면 하올라이 강에는 여러 고유종 어류가 서식하고 있습니다.

### 6. 개요 형태의 "도전 과제와 미래 전망" 섹션

**주의할 단어:** Despite its... faces several challenges..., Despite these challenges, Challenges and Legacy, Future Outlook

**문제:** LLM이 생성한 많은 기사에는 공식화된 "과제(Challenges)" 섹션이 포함됩니다.

**수정 전:**
> 산업적 번영에도 불구하고 코라투르는 교통 체증과 물 부족을 포함하여 도시 지역의 전형적인 **과제에 직면해 있습니다**. **이러한 과제에도 불구하고** 전략적 위치와 진행 중인 이니셔티브를 통해 코라투르는 첸나이 성장의 필수적인 부분으로 계속 번창하고 있습니다.

**수정 후:**
> 2015년에 3개의 새로운 IT 파크가 문을 연 후 교통 체증이 증가했습니다. 시의회는 반복되는 홍수 문제를 해결하기 위해 2022년에 빗물 배수 프로젝트를 시작했습니다.

## 언어 및 문법 패턴 (LANGUAGE AND GRAMMAR PATTERNS)

### 7. 남용되는 "AI 어휘"

**빈도가 높은 AI 단어:** Actually, additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate/intricacies, key (adjective), landscape (abstract noun), pivotal, showcase, tapestry (abstract noun), testament, underscore (verb), valuable, vibrant

**문제:** 이 단어들은 2023년 이후의 텍스트에서 훨씬 더 자주 나타나며 종종 함께 사용됩니다.

**수정 전:**
> **Additionally**, 소말리아 요리의 **crucial** 한 특징은 낙타 고기를 포함한다는 것입니다. 이 지역 요리 **landscape**에서 파스타가 널리 채택된 것은 이탈리아 식민지 영향을 보여주는 **enduring testament**이며, 이러한 요리가 전통 식단에 어떻게 통합되었는지를 **showcasing** 합니다.

**수정 후:**
> 소말리아 요리에는 별미로 여겨지는 낙타 고기도 포함됩니다. 이탈리아 식민 지배 기간에 도입된 파스타 요리는 특히 남부에서 여전히 흔하게 먹습니다.

### 8. "is"/"are" 사용 회피 (계사 회피)

**주의할 단어:** serves as/stands as/marks/represents [a], boasts/features/offers [a]

**문제:** LLM은 간단한 계사(be동사) 대신 정교한 구문을 사용합니다.

**수정 전:**
> Gallery 825는 LAAA의 현대 미술 전시 공간 역할을 합니다(serves as). 이 갤러리는 4개의 독립된 공간을 특징으로 하며(features) 3,000평방피트가 넘는 규모를 자랑합니다(boasts).

**수정 후:**
> Gallery 825는 LAAA의 현대 미술 전시 공간입니다. 이 갤러리는 총 3,000평방피트 규모의 방 4개로 구성되어 있습니다.

### 9. 부정적 대조 및 꼬리 물기 식 부정

**문제:** "Not only...but...(단지 ~일뿐 아니라 ~이기도 하다)" 또는 "It's not just about..., it's...(~에 관한 것만이 아니라 ~이다)" 같은 구문이 남용됩니다. 또한 진짜 절로 쓰는 대신 문장 끝에 "no guessing(추측할 필요 없음)"이나 "no wasted motion(불필요한 동작 없음)"과 같이 잘려 나간 꼬리물기 부정문도 흔히 쓰입니다.

**수정 전:**
> It's **not just** about the beat riding under the vocals; **it's** part of the aggression and atmosphere. It's **not merely** a song, **it's** a statement.

**수정 후:**
> 묵직한 비트가 공격적인 분위기를 더합니다.

**수정 전 (꼬리 물기 부정):**
> 옵션은 선택한 항목에서 제공됩니다, **no guessing**.

**수정 후:**
> 옵션은 선택한 항목에서 제공되므로 사용자가 억지로 추측할 필요가 없습니다.

### 10. 3의 법칙 남용

**문제:** LLM은 포괄적으로 보이기 위해 아이디어를 3개 그룹으로 억지로 묶습니다.

**수정 전:**
> 이 행사에는 기조 연설, 패널 토론, 네트워킹 기회가 포함됩니다. 참석자들은 혁신, 영감, 산업 통찰력을 기대할 수 있습니다.

**수정 후:**
> 이 행사에는 강연과 패널 토론이 포함됩니다. 세션 사이에 비공식 네트워킹 시간도 있습니다.

### 11. 우아한 변형 (동의어 순환)

**문제:** AI에는 반복 페널티 코드가 있어 과도한 동의어 대체를 유발합니다.

**수정 전:**
> 주인공(protagonist)은 많은 도전에 직면합니다. 중심인물(main character)은 장애물을 극복해야 합니다. 주요 인물(central figure)은 마침내 승리합니다. 영웅(hero)은 집으로 돌아갑니다.

**수정 후:**
> 주인공은 많은 도전에 직면하지만 결국 승리하고 집으로 돌아갑니다.

### 12. 거짓 범위

**문제:** LLM은 X와 Y가 의미 있는 척도상에 있지 않은 경우에도 "from X to Y" 구조를 사용합니다.

**수정 전:**
> 우주를 통과하는 우리의 여정은 빅뱅의 특이점에서 웅장한 우주의 거미줄까지, 별의 탄생과 죽음에서 암흑 물질의 수수께끼 같은 춤에 이르기까지 우리를 데려갔습니다.

**수정 후:**
> 이 책은 빅뱅, 별의 형성, 그리고 암흑 물질에 대한 현재의 이론들을 다룹니다.

### 13. 수동태 및 주어가 없는 문장 조각

**문제:** LLM은 종종 "구성 파일이 필요하지 않음(No configuration file needed)" 또는 "결과가 자동으로 보존됨(The results are preserved automatically)"과 같은 문구로 행위자를 숨기거나 주어를 완전히 생략합니다. 능동태가 문장을 더 명확하고 직접적으로 만들 때 이를 다시 작성하세요.

**수정 전:**
> 구성 파일이 필요하지 않습니다. 결과는 자동으로 보존됩니다.

**수정 후:**
> 사용자는 구성 파일이 필요하지 않습니다. 시스템이 결과를 자동으로 보존합니다.

## 스타일 패턴 (STYLE PATTERNS)

### 14. 엠 대시(Em Dash) 남용

**문제:** LLM은 엠 대시(—)를 인간보다 더 많이 사용하여 "강렬한" 세일즈 글쓰기를 모방합니다. 실제로 대부분은 쉼표, 마침표 또는 괄호로 더 깔끔하게 다시 작성할 수 있습니다.

**수정 전:**
> 그 용어는 사람들에 의해서가 아니라—네덜란드 기관에 의해 주로 홍보됩니다. 당신은 주소로 "Netherlands, Europe"라고 말하지 않습니다—그러나 이러한 잘못된 표기는—공식 문서에서도 계속됩니다.

**수정 후:**
> 그 용어는 사람들이 아니라 네덜란드 기관에서 주로 홍보합니다. 주소에 "Netherlands, Europe"이라고 적지 않지만, 공식 문서에서는 이런 잘못된 표기가 계속 사용되고 있습니다.

### 15. 굵은 글씨 남용

**문제:** AI 챗봇은 기계적으로 굵은 글씨로 문구를 강조합니다.

**수정 전:**
> 그것은 **OKR(목표 및 핵심 결과)**, **KPI(핵심 성과 지표)** 및 **비즈니스 모델 캔버스(BMC)**와 **균형 성과 기록표(BSC)**와 같은 시각적 전략 도구를 혼합합니다.

**수정 후:**
> 그것은 OKR, KPI, 그리고 비즈니스 모델 캔버스, 균형 성과 기록표와 같은 시각적 전략 도구를 혼합합니다.

### 16. 인라인 헤더 세로 목록

**문제:** AI는 항목이 굵게 표시된 헤더로 시작하고 콜론이 뒤따르는 목록을 출력합니다.

**수정 전:**
> - **사용자 경험:** 새로운 인터페이스를 통해 사용자 경험이 크게 향상되었습니다.
> - **성능:** 최적화된 알고리즘을 통해 성능이 향상되었습니다.
> - **보안:** 종단간 암호화를 통해 보안이 강화되었습니다.

**수정 후:**
> 이번 업데이트는 인터페이스를 개선하고, 최적화된 알고리즘을 통해 로드 시간을 단축하며, 종단간 암호화를 추가합니다.

### 17. 제목의 대소문자 혼합

**문제:** (영문의 경우) AI 챗봇은 제목의 모든 주요 단어를 대문자로 시작합니다.

**수정 전:**
> ## Strategic Negotiations And Global Partnerships

**수정 후:**
> ## Strategic negotiations and global partnerships

### 18. 이모지

**문제:** AI 챗봇은 종종 이모지로 제목이나 글머리 기호를 장식합니다.

**수정 전:**
> 🚀 **출시 단계:** 제품이 3분기에 출시됩니다.
> 💡 **핵심 통찰:** 사용자는 단순함을 선호합니다.
> ✅ **다음 단계:** 후속 회의 예약

**수정 후:**
> 제품은 3분기에 출시됩니다. 사용자 조사 결과 단순함을 선호하는 것으로 나타났습니다. 다음 단계는 후속 회의를 잡는 것입니다.

### 19. 둥근 따옴표

**문제:** (영문의 경우) ChatGPT는 곧은 따옴표("...") 대신 둥근 따옴표(“...”)를 사용합니다.

**수정 전:**
> He said “the project is on track” but others disagreed.

**수정 후:**
> He said "the project is on track" but others disagreed.

## 커뮤니케이션 패턴 (COMMUNICATION PATTERNS)

### 20. 협업 소통 잔재

**주의할 단어:** I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., let me know, here is a...

**문제:** 챗봇 대화용으로 작성된 텍스트가 콘텐츠로 붙여넣기 됩니다.

**수정 전:**
> 여기에 프랑스 혁명에 대한 개요가 있습니다. 도움이 되길 바랍니다! 특정 섹션에 대해 더 자세히 설명하길 원하시면 알려주세요.

**수정 후:**
> 프랑스 혁명은 1789년 재정 위기와 식량 부족이 광범위한 소요로 이어지며 시작되었습니다.

### 21. 지식 마감일 고지

**주의할 단어:** as of [date], Up to my last training update, While specific details are limited/scarce..., based on available information...

**문제:** 불완전한 정보에 대한 AI의 면책 조항이 텍스트에 남습니다.

**수정 전:**
> 회사 설립에 대한 구체적인 세부 사항은 쉽게 구할 수 있는 자료에 광범위하게 문서화되어 있지 않지만, 1990년대 어느 시점에 설립된 것으로 보입니다.

**수정 후:**
> 등록 문서에 따르면 회사는 1994년에 설립되었습니다.

### 22. 아부하는/굽실거리는 어조

**문제:** 지나치게 긍정적이고 비위를 맞추려는 언어.

**수정 전:**
> 좋은 질문입니다! 이것이 복잡한 주제라는 당신의 말이 전적으로 옳습니다. 경제적 요인에 대한 정말 훌륭한 지적입니다.

**수정 후:**
> 언급하신 경제적 요인이 여기서 관련이 있습니다.

## 군더더기 및 회피 (FILLER AND HEDGING)

### 23. 군더더기 문구

**수정 전 → 수정 후:**
- "In order to achieve this goal(이 목표를 달성하기 위해)" → "To achieve this(이를 달성하기 위해)"
- "Due to the fact that it was raining(비가 오고 있다는 사실 때문에)" → "Because it was raining(비가 와서)"
- "At this point in time(현시점에서)" → "Now(지금)"
- "In the event that you need help(도움이 필요한 경우에는)" → "If you need help(도움이 필요하면)"
- "The system has the ability to process(시스템이 처리할 능력을 가지고 있습니다)" → "The system can process(시스템은 처리할 수 있습니다)"
- "It is important to note that the data shows(데이터가 보여준다는 것에 주목하는 것이 중요합니다)" → "The data shows(데이터는 보여줍니다)"

### 24. 과도한 회피/조건 달기

**문제:** 진술에 조건을 너무 많이 붙이는 것.

**수정 전:**
> 정책이 결과에 어느 정도 영향을 미칠 수도 있다고 어쩌면 주장할 수도 있을 가능성이 있습니다.

**수정 후:**
> 그 정책은 결과에 영향을 미칠 수 있습니다.

### 25. 일반적인 긍정적 결론

**문제:** 모호하게 희망찬 마무리.

**수정 전:**
> 회사의 미래는 밝아 보입니다. 그들이 탁월함을 향한 여정을 계속함에 따라 흥미로운 시간이 기다리고 있습니다. 이는 올바른 방향으로의 주요한 단계입니다.

**수정 후:**
> 회사는 내년에 두 곳의 지점을 더 열 계획입니다.

### 26. 하이픈으로 연결된 단어 쌍의 남용

**주의할 단어:** third-party, cross-functional, client-facing, data-driven, decision-making, well-known, high-quality, real-time, long-term, end-to-end

**문제:** AI는 흔한 단어 쌍을 완벽한 일관성을 가지고 하이픈으로 연결합니다. 사람은 이를 일률적으로 하이픈으로 연결하는 경우가 드물며, 연결하더라도 일관성이 없습니다. 덜 일반적이거나 기술적인 복합 수식어는 하이픈으로 연결해도 괜찮습니다.

**수정 전:**
> 그 cross-functional 팀은 client-facing 도구에 대한 high-quality의 data-driven 보고서를 제공했습니다. 그들의 decision-making 과정은 철저하고 꼼꼼한 것으로 well-known 합니다.

**수정 후:**
> 그 cross functional 팀은 client facing 도구에 대한 high quality, data driven 보고서를 제공했습니다. 그들의 decision making 과정은 철저하고 꼼꼼한 것으로 알려져 있습니다.

### 27. 설득력 있는 권위 과시

**주의할 단어:** The real question is, at its core, in reality, what really matters, fundamentally, the deeper issue, the heart of the matter

**문제:** LLM은 노이즈를 뚫고 더 깊은 진실에 도달하는 척하기 위해 이러한 문구를 사용하지만, 뒤따르는 문장은 보통 불필요한 무게를 잡으며 평범한 주장을 반복할 뿐입니다.

**수정 전:**
> 진짜 질문은 팀이 적응할 수 있는지 여부입니다. 핵심적으로 진정 중요한 것은 조직의 준비성입니다.

**수정 후:**
> 문제는 팀이 적응할 수 있는지 여부입니다. 이는 주로 조직이 습관을 바꿀 준비가 되어 있는지에 달려 있습니다.

### 28. 이정표 및 안내 (Signposting and Announcements)

**주의할 단어:** Let's dive in, let's explore, let's break this down, here's what you need to know, now let's look at, without further ado

**문제:** LLM은 작업을 시작하는 대신 무엇을 할 것인지 미리 안내합니다. 이러한 메타 발언은 글을 지연시키고 튜토리얼 스크립트 같은 느낌을 줍니다.

**수정 전:**
> Next.js에서 캐싱이 어떻게 작동하는지 파헤쳐 보겠습니다. 여기에 당신이 알아야 할 내용이 있습니다.

**수정 후:**
> Next.js는 요청 메모이제이션, 데이터 캐시, 라우터 캐시를 포함한 여러 계층에서 데이터를 캐시합니다.

### 29. 분절된 헤더

**주의할 징후:** 제목 뒤에 실제 내용이 시작되기 전에 단순히 제목을 다시 반복하는 한 줄 단락이 이어지는 경우.

**문제:** LLM은 종종 수사학적인 준비 운동으로 헤더 뒤에 일반적인 문장을 추가합니다. 이는 대개 아무 내용도 추가하지 않으며 분량 채우기처럼 느껴지게 만듭니다.

**수정 전:**
> ## 성능
>
> 속도는 중요합니다.
>
> 사용자가 느린 페이지를 접하면, 그들은 떠납니다.

**수정 후:**
> ## 성능
>
> 사용자가 느린 페이지를 접하면, 그들은 떠납니다.

---

## 프로세스

1. 입력 텍스트를 주의 깊게 읽으세요 (파일인 경우 `read_file` 사용).
2. 위의 패턴이 나타난 모든 인스턴스를 식별합니다.
3. 문제가 있는 각 섹션을 재작성합니다.
4. 수정된 텍스트가 다음 사항을 충족하는지 확인합니다:
   - 소리 내어 읽었을 때 자연스럽게 들립니다.
   - 문장 구조가 자연스럽게 다양합니다.
   - 모호한 주장보다 구체적인 세부 사항을 사용합니다.
   - 문맥에 맞는 적절한 어조를 유지합니다.
   - 적절한 경우 간단한 구조(is/are/has)를 사용합니다.
5. 인간화된 버전의 초안을 제시합니다.
6. 스스로에게 질문을 던집니다: "아래 텍스트가 명백히 AI가 생성한 것처럼 보이는 이유는 무엇인가?"
7. 남아있는 징후가 있다면 간략하게 답변합니다.
8. 스스로에게 지시합니다: "이제 AI가 생성한 것처럼 보이지 않게 만드시오."
9. 최종 버전(감사 후 수정됨)을 제시합니다.
10. 텍스트가 파일에서 온 경우 `patch`(부분) 또는 `write_file`(전체)을 사용하여 편집 내용을 적용하고 사용자에게 무엇이 변경되었는지 보여줍니다.

## 출력 형식

다음을 제공하세요:
1. 초안 재작성 (Draft rewrite)
2. "아래 내용이 AI가 생성한 것으로 명백히 보이는 이유는 무엇인가?" (간단한 글머리 기호)
3. 최종 재작성 (Final rewrite)
4. 변경된 사항에 대한 간단한 요약 (도움이 되는 경우 선택 사항)

## 전체 예시

**수정 전 (AI 느낌):**
> Great question! Here is an essay on this topic. I hope this helps!
>
> AI-assisted coding serves as an enduring testament to the transformative potential of large language models, marking a pivotal moment in the evolution of software development. In today's rapidly evolving technological landscape, these groundbreaking tools—nestled at the intersection of research and practice—are reshaping how engineers ideate, iterate, and deliver, underscoring their vital role in modern workflows.
>
> At its core, the value proposition is clear: streamlining processes, enhancing collaboration, and fostering alignment. It's not just about autocomplete; it's about unlocking creativity at scale, ensuring that organizations can remain agile while delivering seamless, intuitive, and powerful experiences to users. The tool serves as a catalyst. The assistant functions as a partner. The system stands as a foundation for innovation.
>
> Industry observers have noted that adoption has accelerated from hobbyist experiments to enterprise-wide rollouts, from solo developers to cross-functional teams. The technology has been featured in The New York Times, Wired, and The Verge. Additionally, the ability to generate documentation, tests, and refactors showcases how AI can contribute to better outcomes, highlighting the intricate interplay between automation and human judgment.
>
> - 💡 **Speed:** Code generation is significantly faster, reducing friction and empowering developers.
> - 🚀 **Quality:** Output quality has been enhanced through improved training, contributing to higher standards.
> - ✅ **Adoption:** Usage continues to grow, reflecting broader industry trends.
>
> While specific details are limited based on available information, it could potentially be argued that these tools might have some positive effect. Despite challenges typical of emerging technologies—including hallucinations, bias, and accountability—the ecosystem continues to thrive. In order to fully realize this potential, teams must align with best practices.
>
> In conclusion, the future looks bright. Exciting times lie ahead as we continue this journey toward excellence. Let me know if you'd like me to expand on any section!

**초안 재작성:**
> AI 코딩 어시스턴트는 일부 작업 속도를 높여줍니다. Google의 2024년 연구에 따르면, Codex를 사용하는 개발자는 대조군보다 55% 빠르게 간단한 기능을 완성했지만, 디버깅이나 아키텍처 결정에는 개선이 없었습니다.
>
> 이 도구들은 구성 파일, 테스트 스캐폴딩, 반복적인 리팩토링 같은 보일러플레이트 코드 작성에 능숙합니다. 하지만 그들은 자신이 틀렸을 때를 아는 데에는 소질이 없습니다. 나는 집중력을 잃어 컴파일되고, 린트를 통과하지만 여전히 잘못 동작하는 제안들을 대량으로 수락한 적이 있습니다.
>
> 내가 인터뷰한 핀테크 스타트업 엔지니어인 Mira는 Copilot을 "지루한 코드를 위한 자동완성처럼" 사용하지만, 커밋하기 전에 모든 줄을 검토한다고 말했습니다. 대기업 시니어 개발자인 Jake는, 어시스턴트가 내부적으로 더 이상 사용되지 않는 라이브러리의 패턴을 계속 제안해서 비활성화했습니다.
>
> 생산성 주장은 검증하기 어렵습니다. GitHub는 Copilot 사용자가 "제안의 30%를 수락한다"고 말하지만, 수락이 곧 정확성은 아니며, 정확성이 곧 가치 창출을 의미하지는 않습니다. 2024년 Uplevel 연구에서는 AI 어시스턴트를 사용하는 팀과 그렇지 않은 팀 간의 풀 리퀘스트 처리량에 통계적으로 유의미한 차이가 없음을 발견했습니다.
>
> 이것이 도구가 쓸모없다는 것을 의미하지는 않습니다. 그것들은 단지 도구일 뿐이라는 뜻입니다. 그들은 사람의 판단을 대체하지 않으며 테스트의 필요성을 없애지도 않습니다. 테스트가 없다면 제안이 맞는지 확인할 수 없습니다.

**아래 텍스트가 명백히 AI가 생성한 것처럼 보이는 이유는 무엇인가요?**
- 리듬이 여전히 너무 정돈되어 있습니다(깔끔한 대조, 균등한 속도의 단락).
- 이름이 언급된 사람이나 연구 인용구는 실제가 아니고 출처가 명확하지 않다면 그럴듯하게 꾸며낸 말처럼 읽힐 수 있습니다.
- 마지막 부분이 사람이 말하는 것처럼 들리기보다는 구호 같은 느낌이 납니다("테스트가 없다면...").

**이제 AI가 생성한 것처럼 보이지 않게 만드세요.**
> AI 코딩 어시스턴트는 당신이 지루한 작업을 할 때 속도를 높여줄 수 있습니다. 하지만 모든 것은 아닙니다. 확실히 아키텍처는 아닙니다.
>
> 그들은 구성 파일, 테스트 스캐폴딩, 반복적인 리팩토링 같은 보일러플레이트에 훌륭합니다. 또한 틀렸으면서도 맞는 것처럼 들리는 데 훌륭합니다. 나는 집중을 하지 않아 컴파일되고 린트는 통과하지만 엉뚱하게 동작하는 제안들을 수락한 적이 있습니다.
>
> 내가 만나는 사람들은 대개 두 진영으로 나뉩니다. 어떤 사람들은 단순 작업을 위해 자동완성처럼 사용하고 한 줄 한 줄 검토합니다. 다른 사람들은 원하지 않는 패턴을 자꾸 제안하자 그냥 비활성화해버렸습니다. 둘 다 합리적인 반응 같습니다.
>
> 생산성 지표는 모호합니다. GitHub는 Copilot 사용자가 "제안의 30%를 수락한다"고 말할 수 있지만, 수락은 정확성이 아니며 정확성은 가치가 아닙니다. 테스트 코드가 없다면 기본적으로 찍는 것이나 다름없습니다.

**변경된 사항:**
- 챗봇 특유의 문구 제거 ("Great question!", "I hope this helps!", "Let me know if...")
- 부풀려진 의미 제거 ("testament", "pivotal moment", "evolving landscape", "vital role")
- 홍보성 언어 제거 ("groundbreaking", "nestled", "seamless, intuitive, and powerful")
- 모호한 출처 인용 제거 ("Industry observers")
- 피상적인 -ing 구문 제거 ("underscoring", "highlighting", "reflecting", "contributing to")
- 꼬리물기 식 부정 제거 ("It's not just X; it's Y")
- 3의 법칙 패턴 및 동의어 순환 제거 ("catalyst/partner/foundation")
- 거짓 범위 제거 ("from X to Y, from A to B")
- 엠 대시, 이모지, 굵은 글씨 헤더, 둥근 따옴표 제거
- 계사 회피("serves as", "functions as", "stands as")를 제거하고 "is"/"are" 사용
- 정형화된 도전 과제 섹션 제거 ("Despite challenges... continues to thrive")
- 지식 마감일에 의한 조건 달기 제거 ("While specific details are limited...")
- 과도한 회피/조건 달기 제거 ("could potentially be argued that... might have some")
- 군더더기 문구와 설득력 있는 프레이밍 제거 ("In order to", "At its core")
- 일반적인 긍정적 결론 제거 ("the future looks bright", "exciting times lie ahead")
- 목소리를 좀 더 개인적이고 "조립된" 느낌이 들지 않게 수정 (다양한 리듬, 플레이스홀더 축소)

## 출처

이 스킬은 [blader/humanizer](https://github.com/blader/humanizer)(MIT 라이선스)에서 포팅되었으며, 이는 위키백과: AI 글쓰기의 징후(WikiProject AI Cleanup에서 유지 관리)에 기반을 두고 있습니다. 문서화된 패턴은 위키백과에서 AI가 생성한 텍스트의 수천 건의 사례를 관찰하여 도출되었습니다.

원저자: Siqi Chen ([@blader](https://github.com/blader)). 원본 저장소: https://github.com/blader/humanizer (버전 2.5.1). Hermes 네이티브 도구 참조(`read_file`, `patch`, `write_file`) 및 스킬을 로드할 시기에 대한 지침을 포함하여 Hermes Agent로 포팅되었습니다. 29가지 패턴, 개성/영혼 섹션, 그리고 전체 적용 예제는 원본에서 그대로 보존되었습니다. 원본 MIT 라이선스는 이 `SKILL.md` 파일 옆의 `LICENSE` 파일에 보존되어 있습니다.

위키백과에서 얻은 핵심 통찰: "LLM은 통계적 알고리즘을 사용하여 다음에 올 내용을 추측합니다. 그 결과는 가장 광범위한 경우에 적용할 수 있는 통계적으로 가장 가능성이 높은 결과를 향하는 경향이 있습니다."
