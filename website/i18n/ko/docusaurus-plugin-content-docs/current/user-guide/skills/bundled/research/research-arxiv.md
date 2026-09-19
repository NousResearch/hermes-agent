---
title: "Arxiv — 키워드, 저자, 카테고리 또는 ID로 arXiv 논문 검색"
sidebar_label: "Arxiv"
description: "키워드, 저자, 카테고리 또는 ID로 arXiv 논문 검색"
---

{/* 이 페이지는 skill의 SKILL.md에서 자동으로 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Arxiv

키워드, 저자, 카테고리 또는 ID로 arXiv 논문을 검색합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들됨(기본 설치) |
| 경로 | `skills/research/arxiv` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Research`, `Arxiv`, `Papers`, `Academic`, `Science`, `API` |
| 관련 스킬 | [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침도 이것입니다.
:::

# arXiv 연구

무료 REST API를 통해 arXiv에서 학술 논문을 검색하고 가져옵니다. API 키나 의존성이 필요하지 않으며 curl만 사용합니다.

## 빠른 참조

| 작업 | 명령 |
|--------|---------|
| 논문 검색 | `curl "https://export.arxiv.org/api/query?search_query=all:QUERY&max_results=5"` |
| 특정 논문 가져오기 | `curl "https://export.arxiv.org/api/query?id_list=2402.03300"` |
| 초록 읽기(웹) | `web_extract(urls=["https://arxiv.org/abs/2402.03300"])` |
| 논문 전체 읽기(PDF) | `web_extract(urls=["https://arxiv.org/pdf/2402.03300"])` |

## 논문 검색

API는 Atom XML을 반환합니다. `grep`/`sed`로 파싱하거나 `python3`을 통해 파이프하여 읽기 쉬운 출력으로 만들 수 있습니다.

### 기본 검색

```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:GRPO+reinforcement+learning&max_results=5"
```

### 깔끔한 출력(XML을 읽기 쉬운 형식으로 파싱)

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

## 검색 쿼리 구문

| 접두사 | 검색 대상 | 예시 |
|---------|---------|---------|
| `all:` | 모든 필드 | `all:transformer+attention` |
| `ti:` | 제목 | `ti:large+language+models` |
| `au:` | 저자 | `au:vaswani` |
| `abs:` | 초록 | `abs:reinforcement+learning` |
| `cat:` | 카테고리 | `cat:cs.AI` |
| `co:` | 주석 | `co:accepted+NeurIPS` |

### 불리언 연산자

```
# AND (default when using +)
search_query=all:transformer+attention

# OR
search_query=all:GPT+OR+all:BERT

# AND NOT
search_query=all:language+model+ANDNOT+all:vision

# Exact phrase
search_query=ti:"chain+of+thought"

# Combined
search_query=au:hinton+AND+cat:cs.LG
```

## 정렬 및 페이지 매김

| 매개변수 | 옵션 |
|---------|---------|
| `sortBy` | `relevance`, `lastUpdatedDate`, `submittedDate` |
| `sortOrder` | `ascending`, `descending` |
| `start` | 결과 오프셋(0부터 시작) |
| `max_results` | 결과 수(기본값 10, 최대 30000) |

```bash
# Latest 10 papers in cs.AI
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=10"
```

## 특정 논문 가져오기

```bash
# By arXiv ID
curl -s "https://export.arxiv.org/api/query?id_list=2402.03300"

# Multiple papers
curl -s "https://export.arxiv.org/api/query?id_list=2402.03300,2401.12345,2403.00001"
```

## BibTeX 생성

논문의 메타데이터를 가져온 후 BibTeX 항목을 생성합니다.

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

논문을 찾은 후 읽습니다.

```
# Abstract page (fast, metadata + abstract)
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# Full paper (PDF → markdown via Firecrawl)
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
```

로컬 PDF 처리에 대해서는 `ocr-and-documents` 스킬을 참조하세요.

## 일반 카테고리

| 카테고리 | 분야 |
|----------|-------|
| `cs.AI` | 인공지능 |
| `cs.CL` | 계산 및 언어(NLP) |
| `cs.CV` | 컴퓨터 비전 |
| `cs.LG` | 머신 러닝 |
| `cs.CR` | 암호학 및 보안 |
| `stat.ML` | 머신 러닝(통계) |
| `math.OC` | 최적화 및 제어 |
| `physics.comp-ph` | 계산 물리학 |

전체 목록: https://arxiv.org/category_taxonomy

## 도우미 스크립트

`scripts/search_arxiv.py` 스크립트는 XML 파싱을 처리하고 깔끔한 출력을 제공합니다.

```bash
python scripts/search_arxiv.py "GRPO reinforcement learning"
python scripts/search_arxiv.py "transformer attention" --max 10 --sort date
python scripts/search_arxiv.py --author "Yann LeCun" --max 5
python scripts/search_arxiv.py --category cs.AI --sort date
python scripts/search_arxiv.py --id 2402.03300
python scripts/search_arxiv.py --id 2402.03300,2401.12345
```

의존성이 없으며 Python 표준 라이브러리만 사용합니다.

---

## Semantic Scholar(인용, 관련 논문, 저자 프로필)

arXiv는 인용 데이터나 추천을 제공하지 않습니다. 이를 위해 **Semantic Scholar API**를 사용합니다. 기본 사용량은 무료이고 키가 필요하지 않으며(초당 1회 요청), JSON을 반환합니다.

### 논문 세부 정보 및 인용 가져오기

```bash
# By arXiv ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300?fields=title,authors,citationCount,referenceCount,influentialCitationCount,year,abstract" | python3 -m json.tool

# By Semantic Scholar paper ID or DOI
curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/example?fields=title,citationCount"
```

### 논문의 인용 가져오기(누가 인용했는가)

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300/citations?fields=title,authors,year,citationCount&limit=10" | python3 -m json.tool
```

### 논문에서 참조한 문헌 가져오기(무엇을 인용했는가)

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300/references?fields=title,authors,year,citationCount&limit=10" | python3 -m json.tool
```

### 논문 검색(arXiv 검색의 대안, JSON 반환)

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=GRPO+reinforcement+learning&limit=5&fields=title,authors,year,citationCount,externalIds" | python3 -m json.tool
```

### 논문 추천 가져오기

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

`title`, `authors`, `year`, `abstract`, `citationCount`, `referenceCount`, `influentialCitationCount`, `isOpenAccess`, `openAccessPdf`, `fieldsOfStudy`, `publicationVenue`, `externalIds`(arXiv ID, DOI 등을 포함)

---

## 전체 연구 워크플로

1. **발견**: `python scripts/search_arxiv.py "your topic" --sort date --max 10`
2. **영향 평가**: `curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:ID?fields=citationCount,influentialCitationCount"`
3. **초록 읽기**: `web_extract(urls=["https://arxiv.org/abs/ID"])`
4. **논문 전체 읽기**: `web_extract(urls=["https://arxiv.org/pdf/ID"])`
5. **관련 연구 찾기**: `curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:ID/references?fields=title,citationCount&limit=20"`
6. **추천 받기**: Semantic Scholar 추천 엔드포인트에 POST
7. **저자 추적**: `curl -s "https://api.semanticscholar.org/graph/v1/author/search?query=NAME"`

## 요청 제한

| API | 제한 | 인증 |
|------|------|------|
| arXiv | 약 3초당 1회 요청 | 필요 없음 |
| Semantic Scholar | 초당 1회 요청 | 없음(API 키 사용 시 초당 100회) |

## 참고

- arXiv는 Atom XML을 반환합니다. 깔끔한 출력을 위해 도우미 스크립트나 파싱 스니펫을 사용하세요.
- Semantic Scholar는 JSON을 반환합니다. 읽기 쉽게 하려면 `python3 -m json.tool`로 파이프하세요.
- arXiv ID: 이전 형식(`hep-th/0601001`)과 새 형식(`2402.03300`)
- PDF: `https://arxiv.org/pdf/{id}` — 초록: `https://arxiv.org/abs/{id}`
- HTML(제공되는 경우): `https://arxiv.org/html/{id}`
- 로컬 PDF 처리에 대해서는 `ocr-and-documents` 스킬을 참조하세요.

## ID 버전 관리

- `arxiv.org/abs/1706.03762`는 항상 최신 버전으로 연결됩니다.
- `arxiv.org/abs/1706.03762v1`은 특정 불변 버전을 가리킵니다.
- 인용을 생성할 때 실제로 읽은 버전 접미사를 유지해 인용 표류를 방지하세요(이후 버전은 내용이 크게 달라질 수 있습니다).
- API의 `<id>` 필드는 버전이 포함된 URL을 반환합니다(예: `http://arxiv.org/abs/1706.03762v7`).

## 철회된 논문

논문은 제출 후 철회될 수 있습니다. 이런 일이 발생하면:

- `<summary>` 필드에 철회 공지가 포함됩니다("withdrawn" 또는 "retracted"를 확인하세요).
- 메타데이터 필드가 불완전할 수 있습니다.
- 결과를 유효한 논문으로 취급하기 전에 항상 요약을 확인하세요.
