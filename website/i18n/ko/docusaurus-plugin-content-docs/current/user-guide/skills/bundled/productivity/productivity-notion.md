---
title: "Notion — Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, Workers"
sidebar_label: "Notion"
description: "Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, Workers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Notion

Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, Workers.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨(기본 설치) |
| 경로 | `skills/productivity/notion` |
| 버전 | `2.0.0` |
| 작성자 | 커뮤니티 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Notion`, `Productivity`, `Notes`, `Database`, `API`, `CLI`, `Workers` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보게 되는 지침입니다.
:::

# Notion

Notion과 통신하는 방법은 두 가지입니다. 두 방법 모두 동일한 통합 토큰을 사용하므로, 사용 가능한 방법을 선택하세요.

◆ **`ntn` CLI** — Notion의 공식 CLI입니다. 더 짧은 구문과 한 줄 파일 업로드를 지원하며 Workers에 필요합니다. 2026년 5월 기준 macOS + Linux에서만 사용할 수 있습니다(Windows 지원은 "coming soon"). **설치되어 있으면 기본값입니다.**
◆ **HTTP + curl** — Windows를 포함한 모든 환경에서 작동합니다. **`ntn`이 설치되지 않았을 때의 기본 대체 방법입니다.**

## 설정

### 1. 통합 토큰 받기(두 경로 모두 필수)

1. https://notion.so/my-integrations에서 통합을 생성합니다.
2. API 키를 복사합니다(`ntn_` 또는 `secret_`으로 시작).
3. `${HERMES_HOME:-~/.hermes}/.env`에 저장합니다.
   ```
   NOTION_API_KEY=ntn_your_key_here
   ```
4. Notion에서 **대상 페이지/데이터베이스를 통합과 공유**합니다. 페이지 메뉴 `...` → `Connect to` → 통합 이름을 선택합니다. 이렇게 하지 않으면 해당 페이지가 실제로 존재하더라도 API가 해당 페이지에 대해 404를 반환합니다.

### 2. `ntn` 설치(macOS / Linux에서 권장 경로)

```bash
# Recommended
curl -fsSL https://ntn.dev | bash

# Or via npm (needs Node 22+, npm 10+)
npm install --global ntn

ntn --version    # verify
```

**`ntn login`은 건너뛰고 통합 토큰을 사용하세요.** 브라우저가 필요 없으므로 헤드리스 환경에서도 작동합니다.
```bash
export NOTION_API_TOKEN=$NOTION_API_KEY      # ntn reads NOTION_API_TOKEN
export NOTION_KEYRING=0                       # don't try to use the OS keychain
```

모든 세션이 이 값을 상속하도록 셸 프로필(또는 `${HERMES_HOME:-~/.hermes}/.env`)에 해당 export를 추가합니다.

### 3. 런타임에 경로 선택

```bash
if command -v ntn >/dev/null 2>&1; then
  # use ntn
else
  # fall back to curl
fi
```

Windows 사용자는 네이티브 `ntn`이 출시될 때까지 2단계를 완전히 건너뛰세요. Path B는 문제없이 작동합니다. 지금 CLI 방식이 필요하다면 WSL2 안에 `ntn`을 설치하세요.

## API 기본 사항

모든 HTTP 요청에는 `Notion-Version: 2025-09-03`이 필요합니다. `ntn`이 이를 대신 처리합니다. 이 버전에서는 사용자가 "데이터베이스"라고 부르는 것이 API에서 **데이터 소스**라고 불립니다.

## 경로 A — `ntn` CLI(권장, macOS / Linux)

### 원시 API 호출(curl의 축약형)
```bash
ntn api v1/users                                  # GET
ntn api v1/pages parent[page_id]=abc123 \         # POST with inline body
  properties[title][0][text][content]="Notes"
ntn api v1/pages/abc123 -X PATCH archived:=true   # PATCH; := is non-string (bool/num/null)
```

구문 참고:
- `key=value` — 문자열 필드
- `key[nested]=value` — 중첩 객체 필드
- `key:=value` — 타입이 지정된 할당(부울, 숫자, null, 배열)

### 검색
```bash
ntn api v1/search query="page title"
```

### 페이지 메타데이터 읽기
```bash
ntn api v1/pages/{page_id}
```

### 페이지를 마크다운으로 읽기(에이전트 친화적)
```bash
ntn api v1/pages/{page_id}/markdown
```

### 페이지 콘텐츠를 블록으로 읽기
```bash
ntn api v1/blocks/{page_id}/children
```

### 마크다운에서 페이지 생성
```bash
ntn api v1/pages \
  parent[page_id]=xxx \
  properties[title][0][text][content]="Notes from meeting" \
  markdown="# Agenda

- Q3 roadmap
- Hiring"
```

### 마크다운으로 페이지 패치
```bash
ntn api v1/pages/{page_id}/markdown -X PATCH \
  markdown="## Update

Shipped the prototype."
```

### 데이터베이스 쿼리(데이터 소스)
```bash
ntn api v1/data_sources/{data_source_id}/query -X POST \
  filter[property]=Status filter[select][equals]=Active
```

`sorts`, 여러 필터 절 또는 복합 로직이 포함된 복잡한 쿼리는 JSON을 파이프로 전달합니다.
```bash
echo '{"filter": {"property": "Status", "select": {"equals": "Active"}}, "sorts": [{"property": "Date", "direction": "descending"}]}' | \
  ntn api v1/data_sources/{data_source_id}/query -X POST --json -
```

### 파일 업로드(한 줄 — CLI의 가장 큰 장점)
```bash
ntn files create < photo.png
ntn files create --external-url https://example.com/photo.png
ntn files list
```

3단계 HTTP 흐름(업로드 생성 → 바이트 PUT → 참조)과 비교해 보세요.

### 유용한 환경 변수
| 변수 | 효과 |
|---|---|
| `NOTION_API_TOKEN` | 인증 토큰(키체인보다 우선) — 통합 토큰으로 설정 |
| `NOTION_KEYRING=0` | OS 키체인 대신 `~/.config/notion/auth.json`의 파일 기반 자격 증명 사용 |
| `NOTION_WORKSPACE_ID` | 워크스페이스 선택기 프롬프트 건너뛰기 |

## 경로 B — HTTP + curl(크로스 플랫폼, Windows의 기본값)

모든 요청은 다음 패턴을 공유합니다.

```bash
curl -s -X GET "https://api.notion.com/v1/..." \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json"
```

Windows에서는 Windows 10 이상에 포함된 `curl`이 그대로 작동합니다. PowerShell 사용자는 `Invoke-RestMethod`도 사용할 수 있습니다.

### 검색
```bash
curl -s -X POST "https://api.notion.com/v1/search" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"query": "page title"}'
```

### 페이지 메타데이터 읽기
```bash
curl -s "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 페이지를 마크다운으로 읽기(에이전트 친화적)

블록 JSON보다 모델에 전달하기 쉽습니다.

```bash
curl -s "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 페이지 콘텐츠를 블록으로 읽기(구조가 필요할 때)
```bash
curl -s "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 마크다운에서 페이지 생성

`POST /v1/pages`는 `markdown` 본문 매개변수를 허용합니다.

```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"page_id": "xxx"},
    "properties": {"title": [{"text": {"content": "Notes from meeting"}}]},
    "markdown": "# Agenda\n\n- Q3 roadmap\n- Hiring\n\n## Decisions\n- Ship MVP Friday"
  }'
```

### 마크다운으로 페이지 패치
```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"markdown": "## Update\n\nShipped the prototype."}'
```

### 데이터베이스에 페이지 생성(타입이 지정된 속성)
```bash
curl -s -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"database_id": "xxx"},
    "properties": {
      "Name": {"title": [{"text": {"content": "New Item"}}]},
      "Status": {"select": {"name": "Todo"}}
    }
  }'
```

### 데이터베이스 쿼리(데이터 소스)
```bash
curl -s -X POST "https://api.notion.com/v1/data_sources/{data_source_id}/query" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"property": "Status", "select": {"equals": "Active"}},
    "sorts": [{"property": "Date", "direction": "descending"}]
  }'
```

### 데이터베이스 생성
```bash
curl -s -X POST "https://api.notion.com/v1/data_sources" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "parent": {"page_id": "xxx"},
    "title": [{"text": {"content": "My Database"}}],
    "properties": {
      "Name": {"title": {}},
      "Status": {"select": {"options": [{"name": "Todo"}, {"name": "Done"}]}},
      "Date": {"date": {}}
    }
  }'
```

### 페이지 속성 업데이트
```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"properties": {"Status": {"select": {"name": "Done"}}}}'
```

### 페이지에 블록 추가
```bash
curl -s -X PATCH "https://api.notion.com/v1/blocks/{page_id}/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{
    "children": [
      {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "Hello from Hermes!"}}]}}
    ]
  }'
```

### 파일 업로드(3단계 흐름)
```bash
# 1. Create upload
curl -s -X POST "https://api.notion.com/v1/file_uploads" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"filename": "photo.png", "content_type": "image/png"}'

# 2. PUT bytes to the upload_url returned above
curl -s -X PUT "{upload_url}" --data-binary @photo.png

# 3. Reference {file_upload_id} in a page/block payload
```

## 속성 유형

데이터베이스 항목의 일반적인 속성 형식:

- **제목:** `{"title": [{"text": {"content": "..."}}]}`
- **서식 있는 텍스트:** `{"rich_text": [{"text": {"content": "..."}}]}`
- **선택:** `{"select": {"name": "Option"}}`
- **다중 선택:** `{"multi_select": [{"name": "A"}, {"name": "B"}]}`
- **날짜:** `{"date": {"start": "2026-01-15", "end": "2026-01-16"}}`
- **체크박스:** `{"checkbox": true}`
- **숫자:** `{"number": 42}`
- **URL:** `{"url": "https://..."}`
- **이메일:** `{"email": "user@example.com"}`
- **관계:** `{"relation": [{"id": "page_id"}]}`

## API 버전 2025-09-03 — 데이터베이스와 데이터 소스

- **데이터베이스가 데이터 소스가 되었습니다.** 쿼리와 검색에는 `/data_sources/` 엔드포인트를 사용합니다.
- **데이터베이스마다 두 개의 ID가 있습니다:** `database_id`와 `data_source_id`.
  - 페이지 생성 시 `database_id`: `parent: {"database_id": "..."}`
  - 쿼리 시 `data_source_id`: `POST /v1/data_sources/{id}/query`
- 검색 결과는 `data_source_id` 필드와 함께 데이터베이스를 `"object": "data_source"`로 반환합니다.

## Notion Workers(고급, `ntn` 필요)

Workers는 Notion이 사용자를 대신해 호스팅하는 TypeScript 프로그램입니다. 하나의 Worker는 다음 기능을 원하는 조합으로 노출할 수 있습니다.
- **동기화** — 외부 API의 데이터를 일정에 따라(기본 30분) Notion 데이터베이스로 가져옵니다.
- **도구** — Notion의 Custom Agents 안에서 호출 가능한 도구로 표시됩니다.
- **웹훅** — 외부 서비스(GitHub, Stripe 등)에서 HTTP 이벤트를 받아 Notion에서 처리합니다.

**요금제 / 플랫폼 제한:**
- CLI는 모든 요금제에서 작동합니다. **Workers를 배포하려면 Business 또는 Enterprise가 필요합니다.**
- 2026년 5월 기준 `ntn`은 macOS/Linux에서만 사용할 수 있습니다. Windows 사용자는 WSL2를 사용하거나 네이티브 지원을 기다려야 합니다.
- 2026년 8월 11일까지 무료이며, 이후 Notion 크레딧에 따라 사용량이 과금됩니다.

### 최소 Worker

```bash
ntn workers new my-worker      # scaffold
cd my-worker
# Edit src/index.ts
ntn workers deploy --name my-worker
```

`src/index.ts`:
```typescript
import { Worker } from "@notionhq/workers";

const worker = new Worker();
export default worker;

worker.tool("greet", {
  title: "Greet a User",
  description: "Returns a friendly greeting",
  inputSchema: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
  execute: async ({ name }) => `Hello, ${name}!`,
});
```

### 웹훅 기능

```typescript
worker.webhook("onGithubPush", {
  title: "GitHub Push Handler",
  execute: async (events, { notion }) => {
    for (const event of events) {
      // event.body, event.rawBody (for signature verification), event.headers
      console.log("got delivery", event.deliveryId);
    }
  },
});
```

배포 후 `ntn workers webhooks list`에 Notion이 생성한 URL이 표시됩니다. 해당 URL을 비밀로 취급하세요. 서명 검증을 추가하지 않으면 URL을 아는 누구나 이벤트를 POST할 수 있습니다.

### Worker 수명 주기 명령

```bash
ntn workers deploy
ntn workers list
ntn workers exec <capability-key> -d '{"name": "world"}'
ntn workers sync trigger <key>            # run a sync now
ntn workers sync pause <key>
ntn workers env set GITHUB_WEBHOOK_SECRET=...
ntn workers runs list                     # recent invocations
ntn workers runs logs <run-id>
ntn workers webhooks list
```

Worker를 빌드해 달라는 요청을 받으면 `ntn workers new`로 스캐폴딩하고, `src/index.ts`에 코드를 작성하며, `ntn workers env set`으로 필요한 비밀을 설정한 다음 배포합니다. Notion 문서의 https://developers.notion.com/workers에서 전체 API를 다룹니다.

## Notion 스타일 마크다운(`/markdown` 엔드포인트에서 사용)

표준 CommonMark에 Notion 전용 블록을 위한 XML 유사 태그를 더한 형식입니다. 들여쓰기에는 **탭**을 사용합니다.

**CommonMark 외의 블록:**
```
<callout icon="🎯" color="blue_bg">
	Ship the MVP by **Friday**.
</callout>

<details color="gray">
<summary>Toggle title</summary>
	Children indented one tab
</details>

<columns>
	<column>Left side</column>
	<column>Right side</column>
</columns>

<table_of_contents color="gray"/>
```

**인라인:**
- 멘션: `<mention-user url="..."/>`, `<mention-page url="...">Title</mention-page>`, `<mention-date start="2026-05-15"/>`
- 밑줄: `<span underline="true">text</span>`
- 색상: `<span color="blue">text</span>` 또는 첫 줄의 블록 수준 `{color="blue"}`
- 수학: 인라인 `$x^2$`, 블록 `$$ ... $$`
- 인용: `[^https://example.com]`

**색상:** 배경용 변형인 `*_bg`와 함께 `gray brown orange yellow green blue purple pink red`를 사용할 수 있습니다.

제목 5/6은 H4로 접힙니다. 여러 `>` 줄은 별도의 인용 블록으로 렌더링되므로, 여러 줄 인용에는 하나의 `>` 안에 `<br>`을 사용하세요.

## 적절한 경로 선택

| 작업 | mac / Linux | Windows |
|---|---|---|
| 페이지 읽기/쓰기, 검색, 데이터베이스 쿼리 | `ntn api ...` | curl |
| 에이전트가 요약할 페이지 읽기 | `ntn api v1/pages/{id}/markdown` | curl `/markdown` 엔드포인트 |
| 파일 업로드 | `ntn files create < file` | 3단계 HTTP 흐름 |
| 일회성 API 탐색 | `ntn api ...` | curl |
| Notion이 호스팅하는 동기화 / 웹훅 / 에이전트 도구 빌드 | `ntn workers ...` | WSL2 + `ntn workers ...` |

## 참고 사항

- 페이지/데이터베이스 ID는 UUID입니다(대시 포함 여부 모두 허용).
- 속도 제한: 평균 초당 약 3개 요청입니다. CLI가 이를 우회하지는 않습니다.
- API에서는 데이터베이스 **보기** 필터를 설정할 수 없습니다. UI에서만 가능합니다.
- 데이터 소스를 페이지에 삽입하려면 생성 시 항상 `"is_inline": true`를 사용합니다.
- 진행률 표시줄을 억제하려면 항상 curl에 `-s`를 전달합니다(에이전트 출력이 더 깔끔해집니다).
- 읽을 때는 JSON을 `jq`로 파이프합니다: `... | jq '.results[0].properties'`.
- Notion은 이제 MCP 서버(`Notion MCP`, 이전 버전보다 DB 작업에서 토큰 효율이 약 91% 향상됨)도 제공합니다. 세션 내부에서 스트리밍 Notion 액세스를 원한다면 Hermes의 MCP 지원을 통해 연결할 수 있지만, 대부분의 일회성 작업에는 위 경로로 충분합니다.
