---
title: "근거 있는 인용 — 답변과 문서를 인용되고 검증 가능한 출처에 기반하기"
sidebar_label: "근거 있는 인용"
description: "답변과 문서를 인용되고 검증 가능한 출처에 기반하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# 근거 있는 인용

답변과 문서를 인용되고 검증 가능한 출처에 기반합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함됨 |
| 경로 | `skills/research/grounded-citations` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Research`, `Citations`, `Grounding`, `Sources`, `Web`, `Reports` |
| 관련 스킬 | [`research-paper-writing`](/docs/user-guide/skills/bundled/research/research-research-paper-writing), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 활성화된 스킬이 있을 때 에이전트가 보게 되는 내용입니다.
:::

# 근거 있는 인용

외부 출처에서 가져온 모든 주장에는 Perplexity 스타일의 인라인 번호 인용과
`Sources:` 목록을 붙입니다. 원장 스크립트가 `url → [n]` 매핑을 관리하므로
번호와 URL은 기억이 아니라 검색 결과에서 가져옵니다. 모델은 전달받은 작은
정수만 출력합니다.

중요도가 높은 작업에서는 동일한 원장이 팩트체인 검사에도 사용됩니다. 원문
인용은 각 출처에 연결되며(가져온 페이지의 텍스트에 실제로 그대로 나타나는
경우에만 허용됨), 모델 지식에서 나온 주장은 `[unverified]`로 표시되고,
`verify --evidence`는 인용된 출처에 근거가 없는 초안을 실패 처리합니다.

이 스킬은 채팅 답변, 작성 문서(markdown, PDF, docx, slides), 연구 보고서를
다룹니다. 학술 BibTeX 파이프라인은 다루지 않습니다. 학회 논문에는 이 스킬이
공급하는 `research-paper-writing` 스킬을 사용하세요(자세한 내용은
`references/citation-formats.md`).

## 사용 시점

가져온 정보가 아니라 알고 있던 정보에 기반한 답변이나 산출물이라면 사용하세요.

- 조사, 비교, 뉴스 요약, "X의 현재 상태는 무엇인가"
- 인용, 바꿔 쓰기 또는 외부 사실 보고가 포함된 디스크 저장 산출물 — 보고서, 브리프, 문서, 발표 자료, 위키 페이지
- 사용자가 작업 내용을 확인하고 싶어 할 팩트 수집
- 상충하는 출처에 귀속을 표시해야 하는 다중 출처 종합

검색이 다른 작업에 부수적으로 사용된 경우에는 인라인 인용을 생략하세요 —
코딩 중의 간단한 문법/버전 조회, 일상적인 대화, 창작 글쓰기. 사용자가 그
링크를 원할 법한 경우에만 URL을 언급하세요.

## 사전 조건

표준 도구 세트 외에는 아무것도 필요하지 않습니다. `scripts/sources.py`는
Python 3 표준 라이브러리만 사용합니다. 검색은 설정된 도구
`web_search`, `web_extract`, `browser_navigate` 또는 `terminal`(curl, CLI)에서 수행됩니다.

원장 위치: `$HERMES_HOME/cache/citations/ledger.json`(프로필에 따라 달라짐).
작업별로 `--ledger <path>` 또는 `HERMES_CITATION_LEDGER`로 재정의할 수 있습니다.

## 실행 방법

```bash
S=~/.hermes/skills/research/grounded-citations/scripts/sources.py

python3 "$S" reset                                  # start a clean ledger
python3 "$S" add https://example.com/a --title "A"  # prints: [1]
python3 "$S" add https://example.com/b --title "B"  # prints: [2]
python3 "$S" list                                   # ledger table
python3 "$S" render                                 # Sources: block
python3 "$S" verify draft.md                        # catch bad citations
```

`add`는 멱등적이며 URL을 정규화합니다. 동일한 페이지는 하나의 원장 안에서
항상 같은 ID를 반환하므로 여러 검색/추출 과정에서도 ID가 안정적으로 유지됩니다.

## 빠른 참고

| 작업 | 명령 |
|---|---|
| 새 작업을 위한 원장 초기화 | `sources.py reset` |
| 출처를 등록하고 ID 받기 | `sources.py add <url> [--title T]` |
| 한 번에 여러 출처 등록 | `sources.py add <url1> <url2> ...` |
| JSON 도구 출력에서 등록 | `sources.py ingest results.json` |
| 출처에 원문 근거 연결 | `sources.py quote <id> --text "exact wording" --from page.txt` |
| 원장 표시 | `sources.py list [--json]` |
| Sources 블록 생성 | `sources.py render [--style markdown\|plain\|footnotes\|bibtex\|evidence] [--only 1,3]` |
| 초안에 인용된 항목만 생성 | `sources.py render --cited-in draft.md` |
| 초안의 Sources 블록을 제자리에서 다시 작성 | `sources.py render --replace-in draft.md` |
| 초안의 인용 검사 | `sources.py verify draft.md [--strict] [--min-coverage 0.6] [--evidence]` |

## 절차

① 근거 있는 답변이나 문서를 만들 작업을 시작할 때 **원장을 초기화**합니다. 초안에 이미 ID가 있는 작업을 계속하는 경우에는 초기화를 건너뜁니다 — 원장을 재사용하면 번호가 안정적으로 유지됩니다.

② **검색 시점에 모든 출처를 등록**합니다. 각 `web_search` / `web_extract` / `browser_navigate` / fetch 후 URL을 `sources.py add`에 전달하거나 원시 JSON을 `sources.py ingest`로 파이프합니다. 산문을 작성하기 **전에** 수행하세요. 나중에 기억을 바탕으로 등록하는 것이 이 스킬이 막으려는 오류입니다.

③ **초안 작성 중 인용**합니다. 출처가 뒷받침하는 각 문장 바로 뒤에 대괄호 ID를 넣습니다.

```
Ice floats because it is less dense than liquid water.[1][2]
```

- 대괄호 앞에 공백을 넣지 않습니다. 각 ID는 별도의 대괄호에 넣습니다.
- 문장당 ID는 최대 3개입니다. 문장마다 인용하며, 끝에 몰아서 붙이지 않습니다.
- 원장이 반환한 ID만 사용합니다. ID나 URL을 절대 지어내지 않습니다.
- 자신의 지식에서 나온 주장에는 인용을 붙이지 않습니다.
- 출처가 충돌하면 각 해석에 자체 ID를 붙여 두 해석을 모두 제시합니다.
- 정확한 수치, 날짜, 이름은 출처의 표현 그대로 인용합니다. 빈틈을 명시적으로 표시하고("X에 대한 출처를 찾지 못함"), 매끄럽게 얼버무리지 않습니다.

④ `sources.py render --cited-in <draft>`로 **Sources 블록을 추가**합니다. ID → URL 매핑은 직접 다시 입력하지 말고 기계적으로 원장에서 생성해야 합니다. 마크다운이 아닌 대상은 맞는 `--style`을 선택하고 `references/citation-formats.md`에 따라 배치합니다(docx의 각주, PDF/LaTeX의 미주, 발표 자료의 Sources 슬라이드, 위키 출력의 페이지별 출처 목록).

⑤ **제공 전에 검증**합니다 — `sources.py verify <draft>`는 원장에 없는 ID, 원장과 일치하지 않는 Sources 블록, 또는(`--min-coverage` 사용 시) 인용이 너무 적은 산문이 있는 초안을 0이 아닌 종료 코드로 처리합니다. 수정 후 다시 실행하세요.

⑥ **채팅 답변**도 동일한 단계를 따르며, 답변에 초안을 둡니다. 출처를 등록하고 인라인으로 인용한 뒤, 생성된 `Sources:` 목록으로 끝냅니다. 짧은 답변은 파일을 작성하는 대신 `sources.py render --only <ids>`로 블록을 생성해도 됩니다.

## 팩트체크 모드

독자가 검증 경로를 확인할 수 있어야 하는 작업 — 의료, 법률, 금융, 안전, 논쟁적인 주장 또는 사용자가 팩트체크를 요청한 경우 — 인용에서 근거로 강화합니다.

① **출처마다 원문 인용을 연결**합니다. 페이지를 추출한 뒤 텍스트를 파일에 저장하고 각 주장에 해당하는 문장을 연결합니다.

```bash
python3 "$S" quote 1 --text "Ice is about 9% less dense than liquid water." --from page1.txt
```

인용문은 근거 텍스트에 그대로 나타나는 경우에만 허용됩니다(공백, 대소문자, 마크다운 표기에는 영향받지 않음 — 추출된 텍스트의 `_[ERAP1](https://…)_` 같은 인라인 링크도 독자가 보는 일반 산문과 일치합니다). 가져온 텍스트에서 복사해 붙여 넣고, 절대 다시 입력하지 않습니다. 독자가 보는 문장 그대로 인용하세요 — 일치 검사는 추출기의 표기를 알아서 건너뛰므로 링크 문법이나 이스케이프된 별표를 재현할 필요가 없습니다.

② **모델 지식에 기반한 주장에는 `[unverified]`를 표시**합니다. 출처를 찾을 수 없는 핵심 주장은 인용 대신 명시적인 표시를 붙입니다.

```
The refactor likely predates the 2.0 release.[unverified]
```

`verify --min-coverage`는 `[unverified]`가 붙은 문장을 충족된 것으로 계산합니다 — 목표는 모든 주장에 인용을 붙이는 것이 아니라 출처를 선언하는 것입니다. 확인할 수 있는 핵심 주장은 확인하세요. `[unverified]`는 정말 확인할 수 없는 내용에만 사용하며, 팩트체크 산출물 대부분에 이 표시가 있다면 요약에 그 사실을 밝혀야 합니다.

③ **논쟁적인 사실은 서로 독립적인 두 번째 출처와 교차 확인**합니다. 두 출처가 의견을 달리하면 각 해석에 자체 ID와 인용문을 붙이고, 어느 쪽을 왜 더 중요하게 보는지 설명합니다. 한 출처는 보도이고 두 개의 독립적인 출처는 보강입니다.

④ **근거 게이트로 검증하고 근거 블록을 생성**합니다.

```bash
python3 "$S" verify report.md --evidence --min-coverage 0.5
python3 "$S" render --style evidence --replace-in report.md
```

`--evidence`는 인용된 출처에 연결된 인용문이 하나라도 없으면 초안을 실패 처리합니다. `evidence` 렌더링 스타일은 각 출처의 URL 아래에 인용문을 출력하므로, 산출물은 추측 없이 주장 → 출처 → 정확한 뒷받침 텍스트를 보여 줍니다. `--replace-in <draft>`를 사용해 기존 Sources 블록을 제자리에서 다시 작성하세요(멱등적이며, 인용문을 추가한 뒤 안전하게 다시 실행 가능). `--cited-in`은 표준 출력으로 출력합니다. 둘 다 `## Sources` 제목을 출력합니다(`--style plain`은 `Sources:`를 출력).

**`--min-coverage`가 계산하는 것.** 적용 범위는 Sources 블록, 제목(`#`), 표 행(`|`), 펜스 코드 블록을 제거하고 인용문 블록 표식을 벗긴 뒤 4단어 이상인 비어 있지 않은 줄 조각인 산문 문장 중, 출처가 선언된 문장의 비율입니다. `[n]` 인용 또는 `[unverified]` 표시로 출처가 선언되며, 둘 다 있는 문장도 한 번만 계산됩니다. 임계값 없이 먼저 `verify`를 실행하고 `info: stats:` 줄을 읽어 통계를 확인한 다음 적절한 값을 선택하세요. 경고도 읽으세요. 종료 코드가 0이어도 등록했지만 인용되지 않은 출처는 대개 편집 중 귀속이 사라졌다는 뜻입니다.

## 주의할 점

- **작성 후 등록.** 원장은 도구 출력으로 채워야 하며 초안에서 복원해서는 안 됩니다 — 그러면 이 스킬이 막으려는 조작된 URL 위험이 다시 생깁니다.
- **작업 중 번호 재지정.** 초안의 ID를 직접 편집하지 않습니다. ID는 원장의 식별자입니다. 초안이 `[4]`를 인용한다면 `[4]`는 계속 그 출처여야 합니다. 작업 사이에만 `reset`을 실행합니다.
- **Sources 블록에 URL 재입력.** 항상 `render`를 사용합니다. 직접 입력한 URL은 검증되지 않은 주장입니다.
- **검색 스니펫을 페이지를 읽은 것처럼 인용.** 검색 결과 설명은 그 안에 문자 그대로 있는 내용만 뒷받침합니다. 본문이 필요한 주장은 먼저 페이지를 `web_extract`로 추출하세요.
- **과도한 인용.** 문장당 ID는 세 개가 상한입니다. 모든 절에 인용을 붙이면 읽기 어려워지고 어떤 출처가 핵심인지 가려집니다.
- **코드/설정 산출물에서 원장을 인용.** 출처 주석은 산문 산출물과 문서 헤더에 넣으며, 생성된 코드 안에는 넣지 않습니다.
- **병렬 서브에이전트.** 각 서브에이전트는 별도의 작업 디렉터리를 사용합니다. 결과를 합칠 경우 모두 같은 원장을 가리키도록 `--ledger`(또는 `HERMES_CITATION_LEDGER`)를 지정하지 않으면 ID가 충돌합니다.
- **페이지가 아닌 스니펫에서 인용.** 근거 인용은 검색 결과 설명이 아니라 추출한 페이지 텍스트에서 가져와야 합니다 — 먼저 `web_extract`하고, 텍스트를 저장한 다음 `quote --from`을 사용합니다.
- **`quote --text`에 바꿔 쓴 문장 사용.** 원문 검사가 이를 거부합니다. 해결책은 일치할 때까지 다시 표현하는 것이 아니라 실제 문장을 찾는 것입니다.
- **`[unverified]`를 빠져나갈 구멍으로 사용.** 정말 출처를 찾을 수 없는 드문 주장에 표시하는 것입니다. 대부분의 문장에 붙는다면 표시가 더 필요한 것이 아니라 검색이 더 필요했다는 뜻입니다.
- **Sources 블록 직접 편집.** `render --replace-in <draft>`를 사용합니다. 직접 잘라내면 오래되었거나 중복된 블록이 생겨 `verify`에서 다시 실패할 수 있습니다.

## 검증

```bash
python3 "$S" verify report.md --strict --min-coverage 0.5
```

통과하면 초안의 모든 `[n]`이 원장에 존재하고, Sources 블록이 원장의 URL과 함께 정확히 인용된 ID를 나열하며, 인용된 출처가 있는 문장의 비율이 기준을 충족한다는 뜻입니다. 종료 코드가 0이어도 경고를 읽으세요 — 등록했지만 인용되지 않은 출처는 대개 편집 중 귀속이 사라졌다는 뜻입니다.
