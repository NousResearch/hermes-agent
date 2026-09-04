---
title: "Llm Wiki — Karpathy의 LLM Wiki: 상호 연결된 마크다운 KB 구축/질의"
sidebar_label: "Llm Wiki"
description: "Karpathy의 LLM Wiki: 상호 연결된 마크다운 KB 구축/질의"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Llm Wiki

Karpathy의 LLM Wiki: 상호 연결된 마크다운 지식 베이스를 구축하고 질의합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/research/llm-wiki` |
| 버전 | `2.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `wiki`, `knowledge-base`, `research`, `notes`, `markdown`, `rag-alternative` |
| 관련 스킬 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 내용입니다.
:::

# Karpathy의 LLM Wiki

상호 연결된 마크다운 파일로 지속적이고 누적되는 지식 베이스를 구축하고 유지 관리합니다.
[Andrej Karpathy의 LLM Wiki 패턴](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)을 기반으로 합니다.

기존 RAG는 질의할 때마다 지식을 처음부터 다시 발견하지만, 이 위키는 지식을 한 번 컴파일하고 최신 상태로 유지합니다. 교차 참조가 이미 존재합니다.
모순 사항도 이미 표시되어 있습니다. 종합 결과에는 수집된 모든 정보가 반영됩니다.

**역할 분담:** 사람은 소스를 선별하고 분석을 지시합니다. 에이전트는 요약하고, 상호 참조하고, 기록하고, 일관성을 유지 관리합니다.

## 이 스킬이 활성화되는 경우

다음과 같은 경우 이 스킬을 사용합니다.
- 사용자에게 위키 또는 지식 베이스를 생성, 구축 또는 시작해 달라는 요청이 있을 때
- 사용자의 위키에 소스를 수집, 추가 또는 처리해 달라는 요청이 있을 때
- 질문을 받았고 구성된 경로에 기존 위키가 있을 때
- 사용자가 위키의 린트, 감사 또는 상태 점검을 요청할 때
- 연구 맥락에서 사용자의 위키, 지식 베이스 또는 "노트"를 참조할 때

## 위키 위치

**위치:** `WIKI_PATH` 환경 변수로 설정합니다(예: `${HERMES_HOME:-~/.hermes}/.env`).

설정되지 않은 경우 기본값은 `~/wiki`입니다.

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
```

위키는 마크다운 파일로 이루어진 디렉터리일 뿐입니다. Obsidian, VS Code 또는 다른 편집기에서 열 수 있습니다. 데이터베이스도, 특별한 도구도 필요하지 않습니다.

## 아키텍처: 3개 계층

<!-- ascii-guard-ignore -->
```
wiki/
├── SCHEMA.md           # Conventions, structure rules, domain config
├── index.md            # Sectioned content catalog with one-line summaries
├── log.md              # Chronological action log (append-only, rotated yearly)
├── raw/                # Layer 1: Immutable source material
│   ├── articles/       # Web articles, clippings
│   ├── papers/         # PDFs, arxiv papers
│   ├── transcripts/    # Meeting notes, interviews
│   └── assets/         # Images, diagrams referenced by sources
├── entities/           # Layer 2: Entity pages (people, orgs, products, models)
├── concepts/           # Layer 2: Concept/topic pages
├── comparisons/        # Layer 2: Side-by-side analyses
└── queries/            # Layer 2: Filed query results worth keeping
```
<!-- ascii-guard-ignore-end -->

**계층 1 — 원본 소스:** 변경 불가합니다. 에이전트는 읽기만 하며 수정하지 않습니다.
**계층 2 — 위키:** 에이전트가 소유하는 마크다운 파일입니다. 에이전트가 생성하고, 업데이트하고, 상호 참조합니다.
**계층 3 — 스키마:** `SCHEMA.md`가 구조, 규칙 및 태그 분류 체계를 정의합니다.

## 기존 위키 재개(CRITICAL — 매 세션마다 수행)

사용자에게 기존 위키가 있으면 **항상 작업을 시작하기 전에 방향을 파악합니다**.

① **`SCHEMA.md` 읽기** — 도메인, 규칙 및 태그 분류 체계를 이해합니다.
② **`index.md` 읽기** — 어떤 페이지가 있는지와 각 페이지의 요약을 파악합니다.
③ 최근 `log.md` **스캔** — 최근 활동을 파악할 수 있도록 마지막 20~30개 항목을 읽습니다.

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
# Orientation reads at session start
read_file "$WIKI/SCHEMA.md"
read_file "$WIKI/index.md"
read_file "$WIKI/log.md" offset=<last 30 lines>
```

방향을 파악한 뒤에만 수집, 질의 또는 린트를 수행합니다. 이렇게 하면 다음을 방지할 수 있습니다.
- 이미 존재하는 엔터티에 대해 중복 페이지를 생성하는 일
- 기존 콘텐츠에 대한 상호 참조를 놓치는 일
- 스키마 규칙과 모순되는 일
- 로그에 이미 기록된 작업을 반복하는 일

대규모 위키(페이지 100개 이상)의 경우 새로 생성하기 전에 해당 주제에 대해 `search_files`로 빠르게 검색합니다.

## 새 위키 초기화

사용자가 위키를 생성하거나 시작해 달라고 요청하면 다음을 수행합니다.

1. 위키 경로 결정(`$WIKI_PATH` 환경 변수 또는 사용자에게 질문, 기본값 `~/wiki`)
2. 위의 디렉터리 구조 생성
3. 사용자에게 위키가 다룰 도메인 질문 — 구체적으로 질문
4. 도메인에 맞게 `SCHEMA.md` 작성(아래 템플릿 참조)
5. 섹션별 헤더가 있는 초기 `index.md` 작성
6. 생성 항목이 포함된 초기 `log.md` 작성
7. 위키가 준비되었음을 확인하고 먼저 수집할 소스 제안

### SCHEMA.md 템플릿

도메인에 맞게 조정합니다. 스키마는 에이전트 동작을 제한하고 일관성을 보장합니다.

```markdown
# Wiki Schema

## Domain
[What this wiki covers — e.g., "AI/ML research", "personal health", "startup intelligence"]

## Conventions
- File names: lowercase, hyphens, no spaces (e.g., `transformer-architecture.md`)
- Every wiki page starts with YAML frontmatter (see below)
- Use `[[wikilinks]]` to link between pages (minimum 2 outbound links per page)
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md` under the correct section
- Every action must be appended to `log.md`
- **Provenance markers:** On pages that synthesize 3+ sources, append `^[raw/articles/source-file.md]`
  at the end of paragraphs whose claims come from a specific source. This lets a reader trace each
  claim back without re-reading the whole raw file. Optional on single-source pages where the
  `sources:` frontmatter is enough.

## Frontmatter
  ```yaml
  ---
  title: Page Title
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [from taxonomy below]
  sources: [raw/articles/source-name.md]
  # Optional quality signals:
  confidence: high | medium | low        # how well-supported the claims are
  contested: true                        # set when the page has unresolved contradictions
  contradictions: [other-page-slug]      # pages this one conflicts with
  ---
  ```

`confidence` and `contested` are optional but recommended for opinion-heavy or fast-moving
topics. Lint surfaces `contested: true` and `confidence: low` pages for review so weak claims
don't silently harden into accepted wiki fact.

### raw/ Frontmatter

Raw sources ALSO get a small frontmatter block so re-ingests can detect drift:

```yaml
---
source_url: https://example.com/article   # original URL, if applicable
ingested: YYYY-MM-DD
sha256: &lt;hex digest of the raw content below the frontmatter>
---
```

The `sha256:` lets a future re-ingest of the same URL skip processing when content is unchanged,
and flag drift when it has changed. Compute over the body only (everything after the closing
`---`), not the frontmatter itself.

## Tag Taxonomy
[Define 10-20 top-level tags for the domain. Add new tags here BEFORE using them.]

Example for AI/ML:
- Models: model, architecture, benchmark, training
- People/Orgs: person, company, lab, open-source
- Techniques: optimization, fine-tuning, inference, alignment, data
- Meta: comparison, timeline, controversy, prediction

Rule: every tag on a page must appear in this taxonomy. If a new tag is needed,
add it here first, then use it. This prevents tag sprawl.

## Page Thresholds
- **Create a page** when an entity/concept appears in 2+ sources OR is central to one source
- **Add to existing page** when a source mentions something already covered
- **DON'T create a page** for passing mentions, minor details, or things outside the domain
- **Split a page** when it exceeds ~200 lines — break into sub-topics with cross-links
- **Archive a page** when its content is fully superseded — move to `_archive/`, remove from index

## Entity Pages
One page per notable entity. Include:
- Overview / what it is
- Key facts and dates
- Relationships to other entities ([[wikilinks]])
- Source references

## Concept Pages
One page per concept or topic. Include:
- Definition / explanation
- Current state of knowledge
- Open questions or debates
- Related concepts ([[wikilinks]])

## Comparison Pages
Side-by-side analyses. Include:
- What is being compared and why
- Dimensions of comparison (table format preferred)
- Verdict or synthesis
- Sources

## Update Policy
When new information conflicts with existing content:
1. Check the dates — newer sources generally supersede older ones
2. If genuinely contradictory, note both positions with dates and sources
3. Mark the contradiction in frontmatter: `contradictions: [page-name]`
4. Flag for user review in the lint report
```

### index.md 템플릿

색인은 유형별로 섹션을 나눕니다. 각 항목은 한 줄로 작성합니다: 위키 링크 + 요약.

```markdown
# Wiki Index

> Content catalog. Every wiki page listed under its type with a one-line summary.
> Read this first to find relevant pages for any query.
> Last updated: YYYY-MM-DD | Total pages: N

## Entities
<!-- Alphabetical within section -->

## Concepts

## Comparisons

## Queries
```

**확장 규칙:** 섹션의 항목이 50개를 초과하면 첫 글자 또는 하위 도메인별로 하위 섹션을 나눕니다. 전체 색인 항목이 200개를 초과하면 더 빠르게 탐색할 수 있도록 테마별로 페이지를 묶는 `_meta/topic-map.md`를 생성합니다.

### log.md 템플릿

```markdown
# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete
> When this file exceeds 500 entries, rotate: rename to log-YYYY.md, start fresh.

## [YYYY-MM-DD] create | Wiki initialized
- Domain: [domain]
- Structure created with SCHEMA.md, index.md, log.md
```

## 핵심 작업

### 1. 수집

사용자가 소스(URL, 파일, 붙여넣은 내용)를 제공하면 위키에 통합합니다.

① **원본 소스 캡처:**
   - URL → `web_extract`를 사용해 마크다운을 가져오고 `raw/articles/`에 저장
   - PDF → `web_extract` 사용(PDF 처리 가능), `raw/papers/`에 저장
   - 붙여넣은 텍스트 → 적절한 `raw/` 하위 디렉터리에 저장
   - 설명적인 파일명 사용: `raw/articles/karpathy-llm-wiki-2026.md`
   - **원본 프런트매터 추가**(`source_url`, `ingested`, 본문의 `sha256`). 동일한 URL을 재수집할 때 sha256을 다시 계산하고 저장된 값과 비교합니다 — 동일하면 건너뛰고, 변경되었으면 변경 사항을 표시하고 업데이트합니다. 매번 재수집할 때 수행해도 충분히 저렴하며 조용한 소스 변경을 포착합니다.

② 사용자와 **핵심 내용 논의** — 도메인에 흥미롭거나 중요한 내용이 무엇인지 확인합니다. (자동화/cron 컨텍스트에서는 건너뛰고 바로 진행합니다.)

③ **기존 항목 확인** — `index.md`를 검색하고 `search_files`를 사용해 언급된 엔터티/개념에 대한 기존 페이지를 찾습니다. 이것이 성장하는 위키와 중복 페이지 모음의 차이입니다.

④ **위키 페이지 작성 또는 업데이트:**
   - **새 엔터티/개념:** `SCHEMA.md`의 페이지 생성 기준(2개 이상의 소스에 언급되었거나 하나의 소스에서 핵심)에 해당하는 경우에만 페이지를 생성합니다.
   - **기존 페이지:** 새 정보를 추가하고, 사실을 업데이트하며, `updated` 날짜를 변경합니다. 새 정보가 기존 내용과 충돌하면 업데이트 정책을 따릅니다.
   - **상호 참조:** 새로 생성하거나 업데이트하는 모든 페이지는 최소 2개의 다른 페이지에 `[[wikilinks]]`로 연결해야 합니다. 기존 페이지가 역방향으로 연결되는지 확인합니다.
   - **태그:** `SCHEMA.md`의 분류 체계에 있는 태그만 사용합니다.
   - **출처 표시:** 3개 이상의 소스를 종합하는 페이지에서는 주장이 특정 소스에 근거하는 문단에 `^[raw/articles/source.md]` 표시를 추가합니다.
   - **신뢰도:** 의견이 많이 개입되거나 빠르게 변하는 주제, 또는 단일 소스에 근거한 주장에는 프런트매터에 `confidence: medium` 또는 `low`를 설정합니다. 여러 소스에서 충분히 뒷받침되지 않는 한 `high`로 표시하지 않습니다.

⑤ **탐색 기능 업데이트:**
   - 새 페이지를 적절한 섹션에 알파벳순으로 `index.md`에 추가
   - 색인 헤더의 "Total pages" 수와 "Last updated" 날짜 업데이트
   - `log.md`에 추가: `## [YYYY-MM-DD] ingest | Source Title`
   - 로그 항목에 생성하거나 업데이트한 모든 파일 나열

⑥ **변경 사항 보고** — 생성하거나 업데이트한 모든 파일을 사용자에게 나열합니다.

단일 소스가 5~15개의 위키 페이지를 업데이트하게 만들 수 있습니다. 이는 정상이며 바람직합니다 — 지식이 누적되는 효과입니다.
### 2. 질의

사용자가 위키의 도메인에 관해 질문하면:

① 관련 페이지를 식별하려면 `index.md`를 읽습니다.
② **페이지가 100개 이상인 위키의 경우**, 모든 `.md` 파일에서 핵심 용어를 대상으로 `search_files`도 실행합니다. 색인만으로는 관련 내용을 놓칠 수 있습니다.
③ `read_file`을 사용해 관련 페이지를 읽습니다.
④ 수집한 지식을 바탕으로 답변을 종합합니다. 참고한 위키 페이지를 다음과 같이 인용합니다: "[[page-a]]와 [[page-b]]를 바탕으로..."
⑤ 가치 있는 답변은 다시 기록합니다. 실질적인 비교, 심층 분석 또는 새롭게 종합한 내용인 경우 `queries/` 또는 `comparisons/`에 페이지를 만듭니다. 단순한 조회 결과는 기록하지 않습니다. 다시 도출하기 매우 어려운 답변만 기록합니다.
⑥ `log.md`를 업데이트하여 질의 내용과 기록 여부를 남깁니다.

### 3. 린트

사용자가 위키의 린트, 상태 점검 또는 감사를 요청하면:

① **고아 페이지:** 다른 페이지의 `[[wikilinks]]`에서 들어오는 링크가 없는 페이지를 찾습니다.
```python
# Use execute_code for this — programmatic scan across all wiki pages
import os, re
from collections import defaultdict
wiki = "<WIKI_PATH>"
# Scan all .md files in entities/, concepts/, comparisons/, queries/
# Extract all [[wikilinks]] — build inbound link map
# Pages with zero inbound links are orphans
```

② **깨진 wikilink:** 존재하지 않는 페이지를 가리키는 `[[links]]`를 찾습니다.

③ **색인 완전성:** 모든 위키 페이지가 `index.md`에 나타나는지 확인합니다. 파일 시스템과 색인 항목을 비교합니다.

④ **프런트매터 검증:** 모든 위키 페이지에 필수 필드(title, created, updated, type, tags, sources)가 있는지 확인합니다. 태그는 분류 체계에 속해야 합니다.

⑤ **오래된 콘텐츠:** 동일한 엔티티를 언급하는 최신 소스보다 `updated` 날짜가 90일 넘게 오래된 페이지를 찾습니다.

⑥ **모순:** 동일한 주제를 다루는 페이지에서 서로 충돌하는 주장을 찾습니다. 태그/엔티티가 겹치지만 서로 다른 사실을 말하는 페이지를 살펴봅니다. `contested: true` 또는 `contradictions:` 프런트매터가 있는 모든 페이지를 사용자 검토 대상으로 표시합니다.

⑦ **품질 신호:** `confidence: low`인 페이지와, 단일 소스만 인용하면서 confidence 필드가 설정되지 않은 페이지를 나열합니다. 이러한 페이지는 뒷받침 자료를 추가하거나 `confidence: medium`으로 낮출 후보입니다.

⑧ **소스 드리프트:** `sha256:` 프런트매터가 있는 `raw/`의 각 파일에 대해 해시를 다시 계산하고 불일치를 표시합니다. 불일치는 raw/가 수정되었거나(원래 수정해서는 안 됨), 이후 변경된 URL에서 수집되었음을 의미합니다. 중대한 오류는 아니지만 확인할 가치가 있습니다.

⑨ **페이지 크기:** 200줄이 넘는 페이지를 표시합니다. 분할 후보입니다.

⑩ **태그 감사:** 사용 중인 모든 태그를 나열하고, `SCHEMA.md`의 분류 체계에 없는 태그를 표시합니다.

⑪ **로그 순환:** `log.md`가 500개 항목을 넘으면 순환합니다.

⑫ **발견 사항 보고:** 구체적인 파일 경로와 권장 조치를 포함해 심각도별로 그룹화하여 보고합니다(깨진 링크 > 고아 페이지 > 소스 드리프트 > 논쟁 중인 페이지 > 오래된 콘텐츠 > 스타일 문제).

⑬ **`log.md`에 추가:** `## [YYYY-MM-DD] lint | N issues found`

## 위키 작업

### 검색

```bash
# Find pages by content
search_files "transformer" path="$WIKI" file_glob="*.md"

# Find pages by filename
search_files "*.md" target="files" path="$WIKI"

# Find pages by tag
search_files "tags:.*alignment" path="$WIKI" file_glob="*.md"

# Recent activity
read_file "$WIKI/log.md" offset=<last 20 lines>
```

### 일괄 수집

여러 소스를 한 번에 수집할 때는 일괄 처리합니다.
1. 먼저 모든 소스를 읽습니다.
2. 모든 소스에서 모든 엔티티와 개념을 식별합니다.
3. 한 번의 검색으로 모든 엔티티의 기존 페이지를 확인합니다(N번 검색하지 않음).
4. 한 번에 페이지를 생성/업데이트합니다(불필요한 업데이트를 방지).
5. 마지막에 `index.md`를 한 번 업데이트합니다.
6. 일괄 작업 전체를 포괄하는 로그 항목 하나를 작성합니다.

### 보관

콘텐츠가 완전히 대체되었거나 도메인 범위가 변경된 경우:
1. `_archive/` 디렉터리가 없으면 생성합니다.
2. 원래 경로를 유지한 채 페이지를 `_archive/`로 이동합니다(예: `_archive/entities/old-page.md`).
3. `index.md`에서 제거합니다.
4. 해당 페이지로 연결되는 모든 링크를 업데이트합니다 — wikilink를 일반 텍스트와 `"(archived)"`로 바꿉니다.
5. 보관 작업을 `log.md`에 기록합니다.

### Obsidian 통합

위키 디렉터리는 별도 설정 없이 Obsidian 볼트로 사용할 수 있습니다.
- `[[wikilinks]]`는 클릭 가능한 링크로 렌더링됩니다.
- YAML 프런트매터는 Dataview 쿼리를 구동합니다.
- `raw/assets/` 폴더에는 `![[image.png]]`로 참조되는 이미지가 저장됩니다.

최상의 결과를 얻으려면:

- Obsidian의 첨부 파일 폴더를 `raw/assets/`로 설정합니다.
- 보통 기본적으로 활성화되어 있는 Obsidian의 "Wikilinks"를 활성화합니다.
- Dataview 플러그인을 설치해 `TABLE tags FROM "entities" WHERE contains(tags, "company")`와 같은 쿼리를 사용합니다.

Obsidian 스킬을 함께 사용하는 경우 `OBSIDIAN_VAULT_PATH`를 위키 경로와 동일하게 설정합니다.

### Obsidian Headless(서버 및 헤드리스 머신)

디스플레이가 없는 머신에서는 데스크톱 앱 대신 `obsidian-headless`를 사용합니다.
Obsidian Sync를 통해 볼트를 동기화하므로, 서버에서 실행되는 에이전트가 사용하기에 적합하며 다른 기기의 Obsidian 데스크톱에서 읽을 수 있습니다.

**설정:**
```bash
# Requires Node.js 22+
npm install -g obsidian-headless

# Login (requires Obsidian account with Sync subscription)
ob login --email <email> --password '<password>'

# Create a remote vault for the wiki
ob sync-create-remote --name "LLM Wiki"

# Connect the wiki directory to the vault
cd ~/wiki
ob sync-setup --vault "<vault-id>"

# Initial sync
ob sync

# Continuous sync (foreground — use systemd for background)
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
WorkingDirectory=%h/wiki
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now obsidian-wiki-sync
# Enable linger so sync survives logout:
sudo loginctl enable-linger $USER
```

이렇게 하면 서버의 에이전트가 `~/wiki`에 파일을 쓰고, 사용자는 노트북/휴대폰의 Obsidian에서 동일한 볼트를 탐색할 수 있습니다. 변경 사항은 수 초 내에 나타납니다.

## 주의 사항

- **`raw/`의 파일은 절대 수정하지 않습니다** — 소스는 변경할 수 없습니다. 수정 사항은 위키 페이지에 반영합니다.
- **항상 먼저 현황을 파악합니다** — 새 세션에서 작업을 시작할 때는 SCHEMA, 색인, 최근 로그를 읽습니다. 이를 건너뛰면 중복이 생기고 상호 참조를 놓치게 됩니다.
- **항상 `index.md`와 `log.md`를 업데이트합니다** — 이를 건너뛰면 위키가 점차 훼손됩니다. 두 파일은 위키의 탐색을 지탱하는 핵심입니다.
- **단순 언급만으로 페이지를 만들지 않습니다** — SCHEMA.md의 페이지 기준을 따릅니다. 각주에 이름이 한 번 등장한 것만으로는 엔티티 페이지를 만들 근거가 되지 않습니다.
- **상호 참조 없이 페이지를 만들지 않습니다** — 고립된 페이지는 보이지 않습니다. 모든 페이지는 최소 2개의 다른 페이지로 연결되어야 합니다.
- **프런트매터는 필수입니다** — 검색, 필터링, 최신성 감지에 사용됩니다.
- **태그는 분류 체계에서 가져옵니다** — 자유 형식 태그는 관리되지 않는 잡음이 됩니다. 먼저 SCHEMA.md에 새 태그를 추가한 뒤 사용합니다.
- **페이지를 훑어보기 쉽게 유지합니다** — 위키 페이지는 30초 안에 읽을 수 있어야 합니다. 200줄이 넘는 페이지는 분할합니다. 상세한 분석은 별도의 심층 분석 페이지로 옮깁니다.
- **대량 업데이트 전에는 묻습니다** — 수집 작업이 기존 페이지 10개 이상을 수정한다면 먼저 사용자에게 범위를 확인합니다.
- **로그를 순환합니다** — `log.md`가 500개 항목을 넘으면 `log-YYYY.md`로 이름을 바꾸고 새 로그를 시작합니다. 린트 중 로그 크기를 확인해야 합니다.
- **모순을 명시적으로 처리합니다** — 조용히 덮어쓰지 않습니다. 두 주장을 날짜와 함께 기록하고, 프런트매터에 표시하며, 사용자 검토 대상으로 지정합니다.

## 관련 도구

[llm-wiki-compiler](https://github.com/atomicmemory/llm-wiki-compiler)는 동일한 Karpathy 영감을 바탕으로 소스를 개념 위키로 컴파일하는 Node.js CLI입니다. Obsidian과 호환되므로, 예약/CLI 기반 컴파일 파이프라인을 원하는 사용자는 동일한 볼트를 가리키도록 설정할 수 있습니다. 절충점은 페이지 생성을 직접 담당하므로 에이전트의 페이지 생성 판단을 대체하며, 소규모 코퍼스에 맞춰져 있다는 것입니다. 에이전트가 참여하는 큐레이션을 원하면 이 스킬을 사용하고, 소스 디렉터리의 일괄 컴파일을 원하면 llmwiki를 사용합니다.
