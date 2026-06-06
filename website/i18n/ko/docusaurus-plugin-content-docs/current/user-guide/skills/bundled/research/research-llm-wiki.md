---
title: "Llm Wiki — Karpathy의 LLM Wiki: 상호 연결된 마크다운 KB(Knowledge Base) 구축/조회"
sidebar_label: "Llm Wiki"
description: "Karpathy의 LLM Wiki: 상호 연결된 마크다운 KB 구축/조회"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llm Wiki

Karpathy의 LLM Wiki: 상호 연결된 마크다운 KB 구축/조회.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/research/llm-wiki` |
| 버전 | `2.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `wiki`, `knowledge-base`, `research`, `notes`, `markdown`, `rag-alternative` |
| 관련 스킬 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Karpathy's LLM Wiki

지속적이고 복합적인 지식 베이스를 상호 연결된 마크다운 파일로 구축하고 유지 관리합니다.
[Andrej Karpathy의 LLM Wiki 패턴](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)을 기반으로 합니다.

기존의 RAG(쿼리마다 지식을 처음부터 다시 탐색)와 달리, 위키는 지식을 한 번만 컴파일하고 최신 상태로 유지합니다. 상호 참조는 이미 존재합니다. 모순되는 내용은 이미 표시되어 있습니다. 통합된 정보는 수집된 모든 내용을 반영합니다.

**역할 분담:** 인간은 출처를 큐레이션하고 분석을 지시합니다. 에이전트는 요약, 상호 참조, 정리 및 일관성 유지를 담당합니다.

## 이 스킬의 활성화 시기

사용자가 다음과 같은 작업을 요청할 때 이 스킬을 사용합니다:
- 위키나 지식 베이스를 생성, 구축 또는 시작할 때
- 출처를 수집, 추가 또는 처리하여 위키에 넣을 때
- 질문을 하고 구성된 경로에 기존 위키가 존재할 때
- 위키를 린트(lint), 감사 또는 상태 점검(health-check)할 때
- 연구 맥락에서 위키, 지식 베이스 또는 "노트"를 언급할 때

## 위키 위치

**위치:** `WIKI_PATH` 환경 변수(예: `~/.hermes/.env`)를 통해 설정됩니다.

설정되지 않은 경우 기본값은 `~/wiki`입니다.

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
```

위키는 단순히 마크다운 파일들의 디렉토리입니다. Obsidian, VS Code 또는 어떠한 에디터에서든 열 수 있습니다. 데이터베이스나 특별한 도구가 필요하지 않습니다.

## 아키텍처: 세 가지 계층

<!-- ascii-guard-ignore -->
```
wiki/
├── SCHEMA.md           # 규칙, 구조 정의, 도메인 구성
├── index.md            # 한 줄 요약이 포함된 섹션별 콘텐츠 카탈로그
├── log.md              # 시간순 작업 로그 (추가 전용, 매년 순환)
├── raw/                # 1계층: 변경 불가능한 원본 자료
│   ├── articles/       # 웹 기사, 스크랩
│   ├── papers/         # PDF, arxiv 논문
│   ├── transcripts/    # 회의록, 인터뷰
│   └── assets/         # 소스에서 참조하는 이미지, 다이어그램
├── entities/           # 2계층: 엔티티 페이지 (인물, 조직, 제품, 모델)
├── concepts/           # 2계층: 개념/주제 페이지
├── comparisons/        # 2계층: 나란히 비교 분석한 페이지
└── queries/            # 2계층: 보관할 가치가 있는 쿼리 결과 파일
```
<!-- ascii-guard-ignore-end -->

**1계층 — 원본 소스 (Raw Sources):** 변경 불가능(Immutable)합니다. 에이전트는 이 파일들을 읽기만 하고 절대 수정하지 않습니다.
**2계층 — 위키 (The Wiki):** 에이전트 소유의 마크다운 파일들입니다. 에이전트에 의해 생성, 업데이트 및 상호 참조됩니다.
**3계층 — 스키마 (The Schema):** `SCHEMA.md`는 구조, 규칙 및 태그 분류 체계(taxonomy)를 정의합니다.

## 기존 위키 다시 시작하기 (매우 중요 — 매 세션마다 수행하세요)

사용자에게 기존 위키가 있는 경우, **어떤 작업을 수행하기 전에 항상 파악을 먼저 하세요**:

① **`SCHEMA.md` 읽기** — 도메인, 규칙, 태그 분류 체계를 이해합니다.
② **`index.md` 읽기** — 어떤 페이지가 존재하며 요약이 무엇인지 파악합니다.
③ **최근 `log.md` 스캔** — 최근 20~30개 항목을 읽어 최근 활동을 이해합니다.

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
# 세션 시작 시 파악을 위한 읽기
read_file "$WIKI/SCHEMA.md"
read_file "$WIKI/index.md"
read_file "$WIKI/log.md" offset=<최근 30줄>
```

이렇게 파악한 후에만 수집(ingest), 쿼리 또는 린트 작업을 해야 합니다. 이를 통해 다음을 방지할 수 있습니다:
- 이미 존재하는 엔티티에 대한 중복 페이지 생성
- 기존 콘텐츠에 대한 상호 참조 누락
- 스키마 규칙과의 충돌
- 이미 로그에 기록된 작업 반복

대규모 위키(100개 이상의 페이지)의 경우, 새로운 것을 생성하기 전에 당면한 주제에 대해 빠른 `search_files`를 실행하세요.

## 새 위키 초기화

사용자가 위키 생성이나 시작을 요청할 때:

1. 위키 경로를 결정합니다 (`$WIKI_PATH` 환경 변수에서 가져오거나 사용자에게 질문; 기본값은 `~/wiki`)
2. 위와 같은 디렉토리 구조를 생성합니다.
3. 사용자에게 위키가 다루는 도메인이 무엇인지 물어봅니다 — 구체적으로 확인하세요.
4. 해당 도메인에 맞춤화된 `SCHEMA.md`를 작성합니다 (아래 템플릿 참조).
5. 섹션 헤더가 있는 초기 `index.md`를 작성합니다.
6. 생성 항목이 있는 초기 `log.md`를 작성합니다.
7. 위키가 준비되었음을 확인하고 수집할 첫 번째 출처를 제안합니다.

### SCHEMA.md 템플릿

사용자의 도메인에 맞게 조정하세요. 스키마는 에이전트의 동작을 제어하고 일관성을 보장합니다:

```markdown
# 위키 스키마

## 도메인
[이 위키가 다루는 내용 — 예: "AI/ML 연구", "개인 건강", "스타트업 인텔리전스"]

## 규칙
- 파일명: 소문자, 하이픈 사용, 공백 없음 (예: `transformer-architecture.md`)
- 모든 위키 페이지는 YAML 프런트매터로 시작합니다 (아래 참조)
- 페이지 간 링크에는 `[[wikilinks]]`를 사용합니다 (페이지당 최소 2개의 아웃바운드 링크)
- 페이지 업데이트 시 항상 `updated` 날짜를 갱신합니다
- 모든 새 페이지는 올바른 섹션 아래에 `index.md`에 추가되어야 합니다
- 모든 작업은 `log.md`에 추가되어야 합니다
- **출처 표시기 (Provenance markers):** 3개 이상의 소스를 통합한 페이지에서는 특정 소스에서 비롯된 주장이 있는 단락 끝에 `^[raw/articles/source-file.md]`를 추가합니다. 이를 통해 독자는 원본 파일 전체를 다시 읽지 않고도 각 주장의 출처를 추적할 수 있습니다. `sources:` 프런트매터만으로 충분한 단일 소스 페이지에서는 선택 사항입니다.

## 프런트매터 (Frontmatter)
  ```yaml
  ---
  title: 페이지 제목
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [아래 분류 체계 참조]
  sources: [raw/articles/source-name.md]
  # 선택적 품질 신호:
  confidence: high | medium | low        # 주장이 얼마나 잘 뒷받침되는지
  contested: true                        # 페이지에 해결되지 않은 모순이 있을 때 설정
  contradictions: [other-page-slug]      # 이 페이지와 충돌하는 페이지들
  ---
  ```

`confidence`와 `contested`는 선택 사항이지만 의견이 분분하거나 빠르게 변화하는 주제에 권장됩니다. Lint 기능은 검토를 위해 `contested: true` 및 `confidence: low` 페이지를 표시하여 약한 주장이 조용히 굳어진 위키 사실이 되지 않도록 합니다.

### raw/ 프런트매터

원본 소스(Raw sources) 또한 약간의 프런트매터 블록을 가지므로, 재수집 시 변경(drift)을 감지할 수 있습니다:

```yaml
---
source_url: https://example.com/article   # 적용 가능한 경우 원본 URL
ingested: YYYY-MM-DD
sha256: <프런트매터 아래 원본 콘텐츠의 16진수 다이제스트>
---
```

`sha256:`을 사용하면 나중에 동일한 URL을 재수집할 때 콘텐츠가 변경되지 않은 경우 처리를 건너뛰고, 변경된 경우 변동을 표시할 수 있습니다. 프런트매터 자체가 아닌 닫는 `---` 이후의 본문에 대해서만 계산하세요.

## 태그 분류 체계 (Tag Taxonomy)
[도메인에 대한 10-20개의 최상위 태그를 정의합니다. 새 태그는 사용하기 전에 먼저 여기에 추가하세요.]

AI/ML 예시:
- Models: model, architecture, benchmark, training
- People/Orgs: person, company, lab, open-source
- Techniques: optimization, fine-tuning, inference, alignment, data
- Meta: comparison, timeline, controversy, prediction

규칙: 페이지의 모든 태그는 이 분류 체계에 있어야 합니다. 새 태그가 필요한 경우 먼저 여기에 추가한 다음 사용하세요. 이렇게 하면 태그가 무분별하게 늘어나는 것을 방지할 수 있습니다.

## 페이지 생성 기준 (Page Thresholds)
- **페이지 생성:** 엔티티/개념이 2개 이상의 소스에 나타나거나 한 소스의 핵심 내용일 때
- **기존 페이지 추가:** 소스에서 이미 다룬 내용을 언급할 때
- **페이지 생성 금지:** 스쳐 지나가는 언급, 사소한 세부 사항 또는 도메인 밖의 내용
- **페이지 분할:** 페이지가 ~200줄을 초과할 때 — 교차 링크가 있는 하위 주제로 분할
- **페이지 보관:** 콘텐츠가 완전히 대체되었을 때 — `_archive/`로 이동, 인덱스에서 제거

## 엔티티 페이지 (Entity Pages)
주목할 만한 엔티티당 하나의 페이지입니다. 포함 내용:
- 개요 / 무엇인지
- 주요 사실 및 날짜
- 다른 엔티티와의 관계 (`[[wikilinks]]`)
- 출처 참조

## 개념 페이지 (Concept Pages)
개념 또는 주제당 하나의 페이지입니다. 포함 내용:
- 정의 / 설명
- 현재 지식 상태
- 미해결 질문 또는 논쟁
- 관련 개념 (`[[wikilinks]]`)

## 비교 페이지 (Comparison Pages)
나란히 비교 분석한 내용입니다. 포함 내용:
- 비교 대상 및 이유
- 비교 차원 (표 형식 권장)
- 결론 또는 통합
- 출처

## 업데이트 정책 (Update Policy)
새로운 정보가 기존 콘텐츠와 충돌할 때:
1. 날짜 확인 — 최신 소스가 일반적으로 오래된 소스를 대체합니다.
2. 정말 모순되는 경우, 날짜와 출처를 포함하여 두 입장을 모두 기록합니다.
3. 프런트매터에 모순 표시: `contradictions: [page-name]`
4. 린트(lint) 보고서에서 사용자 검토를 위해 플래그 지정
```

### index.md 템플릿

인덱스는 유형별로 구분됩니다. 각 항목은 한 줄로 구성됩니다: 위키링크 + 요약.

```markdown
# 위키 인덱스 (Wiki Index)

> 콘텐츠 카탈로그입니다. 모든 위키 페이지가 한 줄 요약과 함께 해당 유형 아래에 나열됩니다.
> 모든 쿼리에 대해 관련 페이지를 찾으려면 이것을 먼저 읽으세요.
> 마지막 업데이트: YYYY-MM-DD | 총 페이지 수: N

## 엔티티 (Entities)
<!-- 섹션 내 알파벳 순 -->

## 개념 (Concepts)

## 비교 (Comparisons)

## 쿼리 (Queries)
```

**확장 규칙:** 섹션이 50개 항목을 초과하면 첫 글자나 하위 도메인별로 하위 섹션으로 분할합니다. 인덱스 전체 항목이 200개를 초과하면 더 빠른 탐색을 위해 주제별로 페이지를 그룹화하는 `_meta/topic-map.md`를 생성합니다.

### log.md 템플릿

```markdown
# 위키 로그 (Wiki Log)

> 모든 위키 작업의 시간순 기록입니다. 추가만 가능합니다 (Append-only).
> 형식: `## [YYYY-MM-DD] action | subject`
> 작업: ingest, update, query, lint, create, archive, delete
> 이 파일이 500개 항목을 초과하면 순환: log-YYYY.md로 이름 변경 후 새로 시작.

## [YYYY-MM-DD] create | 위키 초기화됨
- 도메인: [domain]
- SCHEMA.md, index.md, log.md로 구조 생성됨
```

## 핵심 작업 (Core Operations)

### 1. 수집 (Ingest)

사용자가 소스(URL, 파일, 붙여넣기)를 제공하면, 위키에 통합합니다:

① **원본 소스 캡처:**
   - URL → `web_extract`를 사용하여 마크다운으로 변환, `raw/articles/`에 저장
   - PDF → `web_extract` 사용(PDF 지원), `raw/papers/`에 저장
   - 붙여넣은 텍스트 → 적절한 `raw/` 하위 디렉토리에 저장
   - 파일 이름을 설명적으로 지정: `raw/articles/karpathy-llm-wiki-2026.md`
   - **원본 프런트매터 추가** (`source_url`, `ingested`, 본문의 `sha256`).
     동일한 URL 재수집 시: sha256을 다시 계산하고 저장된 값과 비교합니다 —
     동일하면 건너뛰고, 다르면 변동(drift)을 표시하고 업데이트합니다. 이는 모든 재수집 시 수행할 만큼 비용이 저렴하며 원본의 소리 없는 변경을 감지합니다.

② **사용자와 핵심 내용 논의** — 도메인에서 흥미로운 점, 중요한 점 등. (자동화/cron 컨텍스트에서는 생략하고 직접 진행하세요.)

③ **이미 존재하는 항목 확인** — `index.md`를 검색하고 `search_files`를 사용하여 언급된 엔티티/개념에 대한 기존 페이지를 찾습니다. 이것이 성장하는 위키와 중복 더미의 차이점입니다.

④ **위키 페이지 작성 또는 업데이트:**
   - **새 엔티티/개념:** SCHEMA.md의 페이지 생성 기준(2개 이상의 소스 언급, 또는 한 소스의 핵심)을 충족하는 경우에만 페이지를 생성합니다.
   - **기존 페이지:** 새 정보를 추가하고, 사실을 업데이트하고, `updated` 날짜를 갱신합니다. 새 정보가 기존 콘텐츠와 충돌하는 경우 업데이트 정책을 따릅니다.
   - **상호 참조:** 모든 새 페이지나 업데이트된 페이지는 `[[wikilinks]]`를 통해 적어도 2개의 다른 페이지에 링크해야 합니다. 기존 페이지들이 다시 링크되는지 확인합니다.
   - **태그:** SCHEMA.md의 분류 체계에 있는 태그만 사용합니다.
   - **출처 (Provenance):** 3개 이상의 소스를 통합한 페이지에서는 주장이 특정 소스에서 비롯된 단락에 `^[raw/articles/source.md]` 표시기를 추가합니다.
   - **신뢰도 (Confidence):** 의견이 분분하거나, 빠르게 변하거나, 단일 출처에 의존하는 주장의 경우 프런트매터에서 `confidence: medium` 또는 `low`를 설정합니다. 여러 소스에 의해 잘 뒷받침되지 않는 한 `high`로 표시하지 마세요.

⑤ **탐색(Navigation) 업데이트:**
   - 올바른 섹션 아래 알파벳순으로 `index.md`에 새 페이지 추가
   - 인덱스 헤더의 "총 페이지 수" 및 "마지막 업데이트" 날짜 갱신
   - `log.md`에 추가: `## [YYYY-MM-DD] ingest | 소스 제목`
   - 생성되거나 업데이트된 모든 파일을 로그 항목에 나열

⑥ **변경 사항 보고** — 생성되거나 업데이트된 모든 파일을 사용자에게 나열합니다.

단일 소스가 5-15개의 위키 페이지에 걸쳐 업데이트를 트리거할 수 있습니다. 이는 정상적이며 바람직한 현상입니다 — 바로 복합적 효과(compounding effect)입니다.

### 2. 쿼리 (Query)

사용자가 위키의 도메인에 대한 질문을 할 때:

① 관련 페이지를 식별하기 위해 **`index.md`를 읽습니다.**
② **100개 이상의 페이지가 있는 위키의 경우**, 인덱스만으로는 관련 콘텐츠를 놓칠 수 있으므로 모든 `.md` 파일에 걸쳐 핵심 용어로 `search_files`도 실행합니다.
③ `read_file`을 사용하여 **관련 페이지를 읽습니다.**
④ 통합된 지식을 바탕으로 **답변을 합성합니다.** 도출한 위키 페이지를 인용하세요: "[[page-a]] 및 [[page-b]]에 따르면..."
⑤ **가치 있는 답변은 보관합니다** — 답변이 상당한 수준의 비교, 심층 분석 또는 새로운 통합인 경우 `queries/` 또는 `comparisons/`에 페이지를 만듭니다. 단순 검색 결과는 보관하지 말고, 다시 유도하기 번거로운 답변만 보관하세요.
⑥ 쿼리 내용과 파일 보관 여부로 **log.md를 업데이트합니다.**

### 3. 린트 (Lint)

사용자가 위키의 린트, 상태 점검 또는 감사를 요청할 때:

① **고아 페이지(Orphan pages):** 다른 페이지로부터 인바운드 `[[wikilinks]]`가 없는 페이지 찾기.
```python
# 이를 위해 execute_code를 사용하세요 — 모든 위키 페이지에 대한 프로그래밍 방식 스캔
import os, re
from collections import defaultdict
wiki = "<WIKI_PATH>"
# entities/, concepts/, comparisons/, queries/ 의 모든 .md 파일 스캔
# 모든 [[wikilinks]] 추출 — 인바운드 링크 맵 구축
# 인바운드 링크가 0개인 페이지가 고아 페이지입니다
```

② **깨진 위키링크(Broken wikilinks):** 존재하지 않는 페이지를 가리키는 `[[links]]` 찾기.

③ **인덱스 완전성(Index completeness):** 모든 위키 페이지는 `index.md`에 나타나야 합니다. 파일 시스템과 인덱스 항목을 비교합니다.

④ **프런트매터 유효성 검사(Frontmatter validation):** 모든 위키 페이지에는 모든 필수 필드(title, created, updated, type, tags, sources)가 있어야 합니다. 태그는 분류 체계에 있어야 합니다.

⑤ **오래된 콘텐츠(Stale content):** `updated` 날짜가 동일한 엔티티를 언급하는 가장 최근의 소스보다 90일 이상 오래된 페이지 찾기.

⑥ **모순(Contradictions):** 상충되는 주장이 있는 같은 주제의 페이지. 태그/엔티티를 공유하지만 다른 사실을 명시하는 페이지를 찾습니다. 프런트매터에 `contested: true` 또는 `contradictions:`가 있는 모든 페이지를 사용자 검토 대상으로 표시합니다.

⑦ **품질 신호(Quality signals):** `confidence: low`인 페이지와 하나의 소스만 인용하지만 신뢰도 필드가 설정되지 않은 페이지 나열 — 이들은 확증을 찾거나 `confidence: medium`으로 강등할 대상입니다.

⑧ **원본 변경(Source drift):** `sha256:` 프런트매터가 있는 `raw/`의 각 파일에 대해 해시를 다시 계산하고 불일치에 플래그 지정. 불일치는 원본 파일이 편집되었거나(`raw/`는 변경 불가능해야 하므로 발생해서는 안 됨) 수집된 URL의 내용이 변경되었음을 나타냅니다. 심각한 오류는 아니지만 보고할 가치가 있습니다.

⑨ **페이지 크기:** 200줄이 넘는 페이지에 플래그 지정 — 분할 대상.

⑩ **태그 감사(Tag audit):** 사용 중인 모든 태그를 나열하고 SCHEMA.md 분류 체계에 없는 태그에 플래그 지정.

⑪ **로그 순환(Log rotation):** log.md가 500개 항목을 초과하면 순환시킵니다.

⑫ 특정 파일 경로와 권장 조치 사항을 심각도(깨진 링크 > 고아 페이지 > 원본 변경 > 논쟁 중인 페이지 > 오래된 콘텐츠 > 스타일 문제)에 따라 그룹화하여 **결과를 보고합니다.**

⑬ **log.md에 추가:** `## [YYYY-MM-DD] lint | N개 문제 발견`

## 위키 사용하기

### 검색 (Searching)

```bash
# 내용으로 페이지 찾기
search_files "transformer" path="$WIKI" file_glob="*.md"

# 파일 이름으로 페이지 찾기
search_files "*.md" target="files" path="$WIKI"

# 태그로 페이지 찾기
search_files "tags:.*alignment" path="$WIKI" file_glob="*.md"

# 최근 활동
read_file "$WIKI/log.md" offset=<최근 20줄>
```

### 일괄 수집 (Bulk Ingest)

한 번에 여러 소스를 수집할 때는 업데이트를 일괄 처리하세요:
1. 먼저 모든 소스를 읽습니다.
2. 모든 소스에서 모든 엔티티와 개념을 식별합니다.
3. 이들 모두에 대해 기존 페이지를 확인합니다 (N번이 아닌 1번의 검색 패스).
4. 한 번의 패스로 페이지를 생성/업데이트합니다 (중복 업데이트 방지).
5. 마지막에 한 번만 index.md를 업데이트합니다.
6. 전체 일괄 처리를 포괄하는 단일 로그 항목을 작성합니다.

### 보관 (Archiving)

콘텐츠가 완전히 대체되거나 도메인 범위가 변경될 때:
1. `_archive/` 디렉토리가 없으면 생성합니다.
2. 원래 경로와 함께 페이지를 `_archive/`로 이동합니다 (예: `_archive/entities/old-page.md`).
3. `index.md`에서 제거합니다.
4. 연결되었던 페이지를 업데이트합니다 — 위키링크를 일반 텍스트 + "(archived)"로 바꿉니다.
5. 보관 작업을 기록(log)합니다.

### Obsidian 통합 (Obsidian Integration)

위키 디렉토리는 별도 설정 없이 Obsidian 볼트(vault)로 작동합니다:
- `[[wikilinks]]`는 클릭 가능한 링크로 렌더링됩니다.
- 그래프 뷰(Graph View)는 지식 네트워크를 시각화합니다.
- YAML 프런트매터는 Dataview 쿼리를 구동합니다.
- `raw/assets/` 폴더는 `![[image.png]]`를 통해 참조되는 이미지를 보관합니다.

최상의 결과를 위해:
- Obsidian의 첨부 파일 폴더를 `raw/assets/`로 설정합니다.
- Obsidian 설정에서 "Wikilinks"를 활성화합니다 (일반적으로 기본 활성화됨).
- `TABLE tags FROM "entities" WHERE contains(tags, "company")`와 같은 쿼리를 위해 Dataview 플러그인을 설치합니다.

이 스킬과 함께 Obsidian 스킬을 사용하는 경우 `OBSIDIAN_VAULT_PATH`를 위키 경로와 동일한 디렉토리로 설정하세요.

### 헤드리스 Obsidian (서버 및 헤드리스 머신용)

디스플레이가 없는 머신에서는 데스크톱 앱 대신 `obsidian-headless`를 사용하세요.
GUI 없이 Obsidian Sync를 통해 볼트를 동기화합니다 — 이는 데스크톱 앱으로 다른 기기에서 읽는 동안 서버에서 실행되며 위키에 쓰는 에이전트에 적합합니다.

**설정:**
```bash
# Node.js 22+ 필요
npm install -g obsidian-headless

# 로그인 (Sync 구독이 있는 Obsidian 계정 필요)
ob login --email <email> --password '<password>'

# 위키용 원격 볼트 생성
ob sync-create-remote --name "LLM Wiki"

# 위키 디렉토리를 볼트에 연결
cd ~/wiki
ob sync-setup --vault "<vault-id>"

# 초기 동기화
ob sync

# 지속적 동기화 (포그라운드 — 백그라운드용으로는 systemd 사용)
ob sync --continuous
```

**systemd를 통한 지속적 백그라운드 동기화:**
```ini
# ~/.config/systemd/user/obsidian-wiki-sync.service
[Unit]
Description=Obsidian LLM Wiki Sync
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/path/to/ob sync --continuous
WorkingDirectory=/home/user/wiki
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now obsidian-wiki-sync
# 동기화가 로그아웃 후에도 지속되도록 linger 활성화:
sudo loginctl enable-linger $USER
```

이를 통해 에이전트는 서버의 `~/wiki`에 쓰고 사용자는 노트북/전화의 Obsidian에서 동일한 볼트를 찾아볼 수 있습니다 — 변경 사항은 몇 초 안에 나타납니다.

## 주의 사항 (Pitfalls)

- **`raw/`에 있는 파일을 절대 수정하지 마세요** — 소스는 변경할 수 없습니다. 수정 사항은 위키 페이지에 기재합니다.
- **항상 먼저 파악하세요** — 새 세션에서 어떠한 작업을 하기 전에 SCHEMA + 인덱스 + 최근 로그를 읽으세요. 이를 건너뛰면 중복 생성 및 상호 참조 누락이 발생합니다.
- **항상 index.md와 log.md를 업데이트하세요** — 이를 건너뛰면 위키의 품질이 저하됩니다. 이들은 탐색의 중추입니다.
- **스쳐 지나가는 언급에 대한 페이지를 만들지 마세요** — SCHEMA.md의 페이지 생성 기준(Page Thresholds)을 따르세요. 각주에 한 번 나타나는 이름은 엔티티 페이지를 만들 이유가 되지 않습니다.
- **상호 참조가 없는 페이지를 만들지 마세요** — 고립된 페이지는 보이지 않습니다. 모든 페이지는 2개 이상의 다른 페이지로 링크되어야 합니다.
- **프런트매터는 필수입니다** — 검색, 필터링, 오래된 내용(staleness) 감지를 가능하게 합니다.
- **태그는 반드시 분류 체계에 있어야 합니다** — 자유형 태그는 노이즈가 됩니다. 새 태그는 SCHEMA.md에 먼저 추가한 다음 사용하세요.
- **페이지를 훑어보기 쉽게 유지하세요** — 위키 페이지는 30초 안에 읽을 수 있어야 합니다. 200줄이 넘는 페이지는 분할하세요. 상세한 분석은 전용 심층 분석(deep-dive) 페이지로 옮기세요.
- **대량 업데이트 전에 물어보세요** — 수집 작업이 10개 이상의 기존 페이지를 변경할 경우, 사전에 사용자와 범위를 확인하세요.
- **로그 순환** — log.md가 500개 항목을 초과하면 `log-YYYY.md`로 이름을 바꾸고 새로 시작하세요. 에이전트는 린트(lint) 중에 로그 크기를 확인해야 합니다.
- **모순은 명시적으로 처리하세요** — 조용히 덮어쓰지 마세요. 날짜와 함께 양쪽의 주장을 모두 기록하고, 프런트매터에 표시하며, 사용자 검토 대상으로 지정하세요.

## 관련 도구 (Related Tools)

[llm-wiki-compiler](https://github.com/atomicmemory/llm-wiki-compiler)는 동일한 Karpathy의 영감을 받아 소스를 개념 위키로 컴파일하는 Node.js CLI입니다. Obsidian과 호환되므로 예약된/CLI 기반 컴파일 파이프라인을 원하는 사용자는 이 스킬이 관리하는 동일한 볼트를 가리키도록 할 수 있습니다. 절충안: 페이지 생성(페이지 생성에 대한 에이전트의 판단을 대체)을 담당하며 소규모 말뭉치(corpora)에 최적화되어 있습니다. 인간 개입(agent-in-the-loop) 큐레이션을 원할 때 이 스킬을 사용하세요; 소스 디렉토리의 일괄 컴파일을 원할 때 llmwiki를 사용하세요.
