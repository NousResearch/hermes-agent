---
title: "Arxiv — 키워드, 저자, 범주 또는 ID로 arXiv 논문 검색"
sidebar_label: "Arxiv"
description: "키워드, 저자, 범주 또는 ID로 arXiv 논문 검색"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Arxiv

키워드, 저자, 범주 또는 ID로 arXiv 논문 검색.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/research/arxiv` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Research`, `Arxiv`, `Papers`, `Academic`, `Science`, `API` |
| 관련 스킬 | [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# arXiv Research

무료 REST API를 통해 arXiv에서 학술 논문을 검색하고 검색합니다. API 키, 종속성 없음 — curl만 있으면 됩니다.

## 빠른 참조

| 작업 | 명령어 |
|--------|---------|
| 논문 검색 | `curl "https://export.arxiv.org/api/query?search_query=all:QUERY&max_results=5"` |
| 특정 논문 가져오기 | `curl "https://export.arxiv.org/api/query?id_list=2402.03300"` |
| 초록 읽기 (웹) | `web_extract(urls=["https://arxiv.org/abs/2402.03300"])` |
| 전체 논문 읽기 (PDF) | `web_extract(urls=["https://arxiv.org/pdf/2402.03300"])` |

## 논문 검색

API는 Atom XML을 반환합니다. 깔끔한 출력을 위해 `grep`/`sed`로 파싱하거나 `python3`를 통해 파이프라인으로 연결하세요.

### 기본 검색

```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:GRPO+reinforcement+learning&max_results=5"
```

### 깔끔한 출력 (XML을 읽기 쉬운 형식으로 파싱)

```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:GRPO+reinforcement+learning&max_results=5&sortBy=submittedDate&sortOrder=descending" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
for i, entry in enumerate(root.findall('a:entry', ns)):
    title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
    arxiv_id = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
    published = entry.find('a:published', ns).text[:10]
    authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
    summary = entry.find('a:summary', ns).text.strip()[:200]
    cats = ', '.join(c.get('term') for c in entry.findall('a:category', ns))
    print(f'{i+1}. [{arxiv_id}] {title}')
    print(f'   Authors: {authors}')
    print(f'   Published: {published} | Categories: {cats}')
    print(f'   Abstract: {summary}...')
    print(f'   PDF: https://arxiv.org/pdf/{arxiv_id}')
    print()
"
```

## 검색어 구문

| 접두어 | 검색 대상 | 예시 |
|--------|----------|---------|
| `all:` | 모든 필드 | `all:transformer+attention` |
| `ti:` | 제목 | `ti:large+language+models` |
| `au:` | 저자 | `au:vaswani` |
| `abs:` | 초록 | `abs:reinforcement+learning` |
| `cat:` | 분류 | `cat:cs.AI` |
| `co:` | 댓글 | `co:accepted+NeurIPS` |

### 논리 연산자

```
# AND (+ 사용 시 기본값)
search_query=all:transformer+attention

# OR
search_query=all:GPT+OR+all:BERT

# AND NOT
search_query=all:language+model+ANDNOT+all:vision

# 정확한 문구 (Exact phrase)
search_query=ti:"chain+of+thought"

# 조합
search_query=au:hinton+AND+cat:cs.LG
```

## 정렬 및 페이지네이션

| 매개변수 | 옵션 |
|-----------|---------|
| `sortBy` | `relevance`, `lastUpdatedDate`, `submittedDate` |
| `sortOrder` | `ascending`, `descending` |
| `start` | 결과 오프셋 (0 기반) |
| `max_results` | 결과 수 (기본값 10, 최대 30000) |

```bash
# cs.AI 분야의 최근 논문 10개
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=10"
```

## 특정 논문 가져오기

```bash
# arXiv ID로 가져오기
curl -s "https://export.arxiv.org/api/query?id_list=2402.03300"

# 여러 논문 가져오기
curl -s "https://export.arxiv.org/api/query?id_list=2402.03300,2401.12345,2403.00001"
```

## BibTeX 생성

논문에 대한 메타데이터를 가져온 후, BibTeX 항목을 생성하세요:

&#123;% raw %&#125;
```bash
curl -s "https://export.arxiv.org/api/query?id_list=1706.03762" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
root = ET.parse(sys.stdin).getroot()
entry = root.find('a:entry', ns)
if entry is None: sys.exit('Paper not found')
title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
authors = ' and '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
year = entry.find('a:published', ns).text[:4]
raw_id = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
cat = entry.find('arxiv:primary_category', ns)
primary = cat.get('term') if cat is not None else 'cs.LG'
last_name = entry.find('a:author', ns).find('a:name', ns).text.split()[-1]
print(f'@article{{{last_name}{year}_{raw_id.replace(\".\", \"\")},')
print(f'  title     = {{{title}}},')
print(f'  author    = {{{authors}}},')
print(f'  year      = {{{year}}},')
print(f'  eprint    = {{{raw_id}}},')
print(f'  archivePrefix = {{arXiv}},')
print(f'  primaryClass  = {{{primary}}},')
print(f'  url       = {{https://arxiv.org/abs/{raw_id}}}')
print('}')
"
```
&#123;% endraw %&#125;

## 논문 내용 읽기

논문을 찾은 후, 내용을 읽어보세요:

```
# 초록 페이지 (빠름, 메타데이터 + 초록)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# 전체 논문 (Firecrawl을 통해 PDF → 마크다운 변환)
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
```

로컬 PDF 처리의 경우, `ocr-and-documents` 스킬을 참조하세요.

## 공통 카테고리 (Common Categories)

| 카테고리 | 분야 |
|----------|-------|
| `cs.AI` | Artificial Intelligence (인공지능) |
| `cs.CL` | Computation and Language (NLP) (자연어 처리) |
| `cs.CV` | Computer Vision (컴퓨터 비전) |
| `cs.LG` | Machine Learning (머신 러닝) |
| `cs.CR` | Cryptography and Security (암호학 및 보안) |
| `stat.ML` | Machine Learning (Statistics) (통계적 머신 러닝) |
| `math.OC` | Optimization and Control (최적화 및 제어) |
| `physics.comp-ph` | Computational Physics (계산 물리학) |

전체 목록: https://arxiv.org/category_taxonomy

## 도우미 스크립트

`scripts/search_arxiv.py` 스크립트는 XML 파싱을 처리하고 깔끔한 출력을 제공합니다:

```bash
python scripts/search_arxiv.py "GRPO reinforcement learning"
python scripts/search_arxiv.py "transformer attention" --max 10 --sort date
python scripts/search_arxiv.py --author "Yann LeCun" --max 5
python scripts/search_arxiv.py --category cs.AI --sort date
python scripts/search_arxiv.py --id 2402.03300
python scripts/search_arxiv.py --id 2402.03300,2401.12345
```

종속성 없음 — Python 표준 라이브러리만 사용합니다.

---

## Semantic Scholar (인용, 관련 논문, 저자 프로필)

arXiv는 인용 데이터나 추천을 제공하지 않습니다. 이를 위해 **Semantic Scholar API**를 사용하세요 — 무료이고, 기본적인 사용(1 req/sec)에는 키가 필요 없으며, JSON을 반환합니다.

### 논문 세부 정보 + 인용 가져오기

```bash
# arXiv ID로 가져오기
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300?fields=title,authors,citationCount,referenceCount,influentialCitationCount,year,abstract" | python3 -m json.tool

# Semantic Scholar 논문 ID 또는 DOI로 가져오기
curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/example?fields=title,citationCount"
```

### 논문의 인용 (어디서 이 논문을 인용했는지) 가져오기

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300/citations?fields=title,authors,year,citationCount&limit=10" | python3 -m json.tool
```

### 논문에서 참조한 목록 (이 논문이 인용한 것) 가져오기

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300/references?fields=title,authors,year,citationCount&limit=10" | python3 -m json.tool
```

### 논문 검색 (arXiv 검색의 대안, JSON 반환)

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=GRPO+reinforcement+learning&limit=5&fields=title,authors,year,citationCount,externalIds" | python3 -m json.tool
```

### 추천 논문 가져오기

```bash
curl -s -X POST "https://api.semanticscholar.org/recommendations/v1/papers/" \
  -H "Content-Type: application/json" \
  -d '{"positivePaperIds": ["arXiv:2402.03300"], "negativePaperIds": []}' | python3 -m json.tool
```

### 저자 프로필

```bash
curl -s "https://api.semanticscholar.org/graph/v1/author/search?query=Yann+LeCun&fields=name,hIndex,citationCount,paperCount" | python3 -m json.tool
```

### 유용한 Semantic Scholar 필드

`title`, `authors`, `year`, `abstract`, `citationCount`, `referenceCount`, `influentialCitationCount`, `isOpenAccess`, `openAccessPdf`, `fieldsOfStudy`, `publicationVenue`, `externalIds` (arXiv ID, DOI 등 포함)

---

## 완전한 연구 워크플로우

1. **발견**: `python scripts/search_arxiv.py "your topic" --sort date --max 10`
2. **영향력 평가**: `curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:ID?fields=citationCount,influentialCitationCount"`
3. **초록 읽기**: `web_extract(urls=["https://arxiv.org/abs/ID"])`
4. **전체 논문 읽기**: `web_extract(urls=["https://arxiv.org/pdf/ID"])`
5. **관련 연구 찾기**: `curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:ID/references?fields=title,citationCount&limit=20"`
6. **추천 받기**: Semantic Scholar recommendations 엔드포인트에 POST 요청
7. **저자 추적**: `curl -s "https://api.semanticscholar.org/graph/v1/author/search?query=NAME"`

## 속도 제한 (Rate Limits)

| API | 속도 | 인증 |
|-----|------|------|
| arXiv | ~1 req / 3초 | 필요 없음 |
| Semantic Scholar | 1 req / 초 | 필요 없음 (API 키 사용 시 100/sec) |

## 참고 사항

- arXiv는 Atom XML을 반환합니다 — 깔끔한 출력을 위해 도우미 스크립트나 파싱 스니펫을 사용하세요
- Semantic Scholar는 JSON을 반환합니다 — 가독성을 높이기 위해 `python3 -m json.tool`을 통해 파이프라인으로 연결하세요
- arXiv ID: 이전 형식 (`hep-th/0601001`) vs 새 형식 (`2402.03300`)
- PDF: `https://arxiv.org/pdf/{id}` — 초록: `https://arxiv.org/abs/{id}`
- HTML (가능한 경우): `https://arxiv.org/html/{id}`
- 로컬 PDF 처리의 경우, `ocr-and-documents` 스킬을 참조하세요

## ID 버전 관리 (ID Versioning)

- `arxiv.org/abs/1706.03762`는 항상 **최신** 버전으로 연결됩니다
- `arxiv.org/abs/1706.03762v1`는 **특정** 불변 버전을 가리킵니다
- 인용을 생성할 때, 실제로 읽은 버전 접미사를 보존하여 인용 변경(이후 버전에서 내용이 크게 변경될 수 있음)을 방지하세요
- API `<id>` 필드는 버전이 포함된 URL을 반환합니다 (예: `http://arxiv.org/abs/1706.03762v7`)

## 철회된 논문 (Withdrawn Papers)

논문은 제출 후 철회될 수 있습니다. 이런 경우:
- `<summary>` 필드에 철회 공지사항이 포함됩니다 ("withdrawn" 또는 "retracted" 확인)
- 메타데이터 필드가 불완전할 수 있습니다
- 결과를 유효한 논문으로 처리하기 전에 항상 초록을 확인하세요
