---
title: "Notion — Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, 워커"
sidebar_label: "Notion"
description: "Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, 워커"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Notion

Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, 워커(Workers).

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/notion` |
| Version | `2.0.0` |
| Author | community |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Notion`, `Productivity`, `Notes`, `Database`, `API`, `CLI`, `Workers` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Notion

두 가지 방법으로 Notion과 통신합니다. 동일한 통합 토큰이 두 방법 모두에 작동합니다 — 사용 가능한 것을 선택하세요.

◆ **`ntn` CLI** — Notion의 공식 CLI. 더 짧은 구문, 한 줄 파일 업로드, 워커(Workers)에 필수적입니다. 2026년 5월 현재 macOS + Linux 전용 (Windows 지원 "출시 예정"). **설치 시 기본값입니다.**
◆ **HTTP + curl** — Windows를 포함한 모든 곳에서 작동합니다. `ntn`이 설치되지 않은 경우 **기본 대체 수단(fallback)**입니다.

## 설정 (Setup)

### 1. 통합 토큰 받기 (두 경로 모두 필수)

1. https://notion.so/my-integrations 에서 통합(integration)을 생성합니다.
2. API 키를 복사합니다 (`ntn_` 또는 `secret_`으로 시작)
3. `~/.hermes/.env`에 저장합니다:
   ```
   NOTION_API_KEY=ntn_your_key_here
   ```
4. Notion에서 **대상 페이지/데이터베이스를 통합과 공유합니다**: 페이지 메뉴 `...` → `연결(Connect to)` → 통합 이름. 이 작업을 수행하지 않으면 API가 해당 페이지가 존재하더라도 404를 반환합니다.

### 2. `ntn` 설치 (macOS / Linux 권장 경로)

```bash
# 권장
curl -fsSL https://ntn.dev | bash

# 또는 npm 사용 (Node 22+, npm 10+ 필요)
npm install --global ntn

ntn --version    # 검증
```

**`ntn login`은 건너뛰고 — 대신 통합 토큰을 사용하세요.** 이 방법은 브라우저 없이 헤드리스(headless)로 작동합니다:
```bash
export NOTION_API_TOKEN=$NOTION_API_KEY      # ntn은 NOTION_API_TOKEN을 읽습니다
export NOTION_KEYRING=0                       # OS 키체인을 사용하지 마세요
```

이 export 문을 셸 프로필(또는 `~/.hermes/.env`)에 추가하여 모든 세션에서 상속되도록 하세요.

### 3. 런타임에 경로 선택

```bash
if command -v ntn >/dev/null 2>&1; then
  # ntn 사용
else
  # curl로 폴백
fi
```

Windows 사용자: 네이티브 `ntn`이 출시될 때까지 2단계를 완전히 건너뛰세요 — 경로 B(HTTP+curl)가 잘 작동합니다. 지금 바로 CLI의 편의성을 원한다면 WSL2 내부에 `ntn`을 설치하세요.

## API 기초 (API Basics)

모든 HTTP 요청에는 `Notion-Version: 2025-09-03`이 필요합니다. `ntn`은 이를 자동으로 처리합니다. 이 버전에서 사용자들이 "데이터베이스"라고 부르는 것은 API에서는 **데이터 소스(data sources)**라고 부릅니다.

## 경로 A — `ntn` CLI (권장, macOS / Linux)

### Raw API 호출 (curl의 약어)
```bash
ntn api v1/users                                  # GET
ntn api v1/pages parent[page_id]=abc123 \         # POST (인라인 본문)
  properties[title][0][text][content]="Notes"
ntn api v1/pages/abc123 -X PATCH archived:=true   # PATCH; :=는 문자열이 아님 (bool/num/null)
```

구문 참고:
- `key=value` — 문자열 필드
- `key[nested]=value` — 중첩된 객체 필드
- `key:=value` — 타입이 지정된 할당 (부울, 숫자, null, 배열)

### 검색 (Search)
```bash
ntn api v1/search query="page title"
```

### 페이지 메타데이터 읽기
```bash
ntn api v1/pages/{page_id}
```

### 마크다운으로 페이지 읽기 (에이전트 친화적)
```bash
ntn api v1/pages/{page_id}/markdown
```

### 블록으로 페이지 내용 읽기
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

### 마크다운으로 페이지 패치(Patch)
```bash
ntn api v1/pages/{page_id}/markdown -X PATCH \
  markdown="## Update

Shipped the prototype."
```

### 데이터베이스 (데이터 소스) 쿼리
```bash
ntn api v1/data_sources/{data_source_id}/query -X POST \
  filter[property]=Status filter[select][equals]=Active
```

`sorts`, 여러 필터 조건 또는 복합 논리가 있는 복잡한 쿼리의 경우 JSON을 파이프로 전달합니다:
```bash
echo '{"filter": {"property": "Status", "select": {"equals": "Active"}}, "sorts": [{"property": "Date", "direction": "descending"}]}' | \
  ntn api v1/data_sources/{data_source_id}/query -X POST --json -
```

### 파일 업로드 (원라이너 — CLI의 가장 큰 장점)
```bash
ntn files create < photo.png
ntn files create --external-url https://example.com/photo.png
ntn files list
```

3단계 HTTP 흐름(업로드 생성 → 바이트 PUT → 참조)과 비교해 보세요.

### 유용한 환경 변수
| 변수 | 효과 |
|---|---|
| `NOTION_API_TOKEN` | 인증 토큰 (키체인 무시) — 통합 토큰으로 설정 |
| `NOTION_KEYRING=0` | OS 키체인 대신 `~/.config/notion/auth.json`의 파일 기반 자격 증명 |
| `NOTION_WORKSPACE_ID` | 작업 공간 선택 프롬프트 건너뛰기 |

## 경로 B — HTTP + curl (크로스 플랫폼, Windows 기본값)

모든 요청은 이 패턴을 공유합니다:

```bash
curl -s -X GET "https://api.notion.com/v1/..." \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json"
```

Windows에서는 Windows 10+와 함께 제공되는 `curl`이 그대로 작동합니다. PowerShell 사용자는 `Invoke-RestMethod`를 사용할 수도 있습니다.

### 검색 (Search)
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

### 마크다운으로 페이지 읽기 (에이전트 친화적)

블록 JSON보다 모델에 공급하기 더 쉽습니다.

```bash
curl -s "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03"
```

### 블록으로 페이지 내용 읽기 (구조가 필요할 때)
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

### 마크다운으로 페이지 패치(Patch)
```bash
curl -s -X PATCH "https://api.notion.com/v1/pages/{page_id}/markdown" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"markdown": "## Update\n\nShipped the prototype."}'
```

### 데이터베이스에 페이지 생성 (타입이 지정된 속성)
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

### 데이터베이스 (데이터 소스) 쿼리
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

### 파일 업로드 (3단계 흐름)
```bash
# 1. 업로드 생성
curl -s -X POST "https://api.notion.com/v1/file_uploads" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -H "Content-Type: application/json" \
  -d '{"filename": "photo.png", "content_type": "image/png"}'

# 2. 위에서 반환된 upload_url로 바이트 PUT
curl -s -X PUT "{upload_url}" --data-binary @photo.png

# 3. 페이지/블록 페이로드에서 {file_upload_id} 참조
```

## 속성 타입 (Property Types)

데이터베이스 항목의 일반적인 속성 형식:

- **제목 (Title):** `{"title": [{"text": {"content": "..."}}]}`
- **리치 텍스트 (Rich text):** `{"rich_text": [{"text": {"content": "..."}}]}`
- **선택 (Select):** `{"select": {"name": "Option"}}`
- **다중 선택 (Multi-select):** `{"multi_select": [{"name": "A"}, {"name": "B"}]}`
- **날짜 (Date):** `{"date": {"start": "2026-01-15", "end": "2026-01-16"}}`
- **체크박스 (Checkbox):** `{"checkbox": true}`
- **숫자 (Number):** `{"number": 42}`
- **URL:** `{"url": "https://..."}`
- **이메일 (Email):** `{"email": "user@example.com"}`
- **관계형 (Relation):** `{"relation": [{"id": "page_id"}]}`

## API 버전 2025-09-03 — 데이터베이스 vs 데이터 소스

- **데이터베이스가 데이터 소스로 변경되었습니다.** 쿼리 및 검색을 위해 `/data_sources/` 엔드포인트를 사용하세요.
- **데이터베이스당 두 개의 ID:** `database_id` 및 `data_source_id`.
  - 페이지 생성 시 `database_id` 사용: `parent: {"database_id": "..."}`
  - 쿼리 시 `data_source_id` 사용: `POST /v1/data_sources/{id}/query`
- 검색은 `data_source_id` 필드와 함께 `"object": "data_source"`로 데이터베이스를 반환합니다.

## Notion Workers (고급, `ntn` 필요)

워커(Workers)는 Notion이 귀하를 위해 호스팅하는 TypeScript 프로그램입니다. 하나의 워커는 다음 조합을 노출할 수 있습니다:
- **Syncs** — 일정에 따라(기본값 30분) 외부 API에서 Notion 데이터베이스로 데이터를 가져옵니다.
- **Tools** — Notion의 사용자 지정 에이전트 내에서 호출 가능한 도구로 표시됩니다.
- **Webhooks** — 외부 서비스(GitHub, Stripe 등)로부터 HTTP 이벤트를 수신하고 Notion에서 작동합니다.

**플랜 / 플랫폼 제한:**
- CLI는 모든 플랜에서 작동합니다. **워커 배포에는 비즈니스 또는 엔터프라이즈 플랜이 필요합니다.**
- `ntn`은 2026년 5월 현재 macOS/Linux 전용입니다. Windows 사용자는 WSL2가 필요하거나 네이티브 지원을 기다려야 합니다.
- 2026년 8월 11일까지 무료이며; 이후에는 Notion 크레딧에 따라 과금됩니다.

### 최소한의 워커 (Minimal Worker)

```bash
ntn workers new my-worker      # 스캐폴딩
cd my-worker
# src/index.ts 편집
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

### 웹훅 기능 (Webhook capability)

```typescript
worker.webhook("onGithubPush", {
  title: "GitHub Push Handler",
  execute: async (events, { notion }) => {
    for (const event of events) {
      // event.body, event.rawBody (서명 검증용), event.headers
      console.log("got delivery", event.deliveryId);
    }
  },
});
```

배포 후: `ntn workers webhooks list`를 실행하면 Notion이 생성한 URL이 표시됩니다. 해당 URL을 비밀로 취급하세요 — 서명 검증을 추가하지 않으면 이 URL을 가진 사람은 누구나 이벤트를 POST할 수 있습니다.

### 워커 수명 주기 명령 (Worker lifecycle commands)

```bash
ntn workers deploy
ntn workers list
ntn workers exec <capability-key> -d '{"name": "world"}'
ntn workers sync trigger <key>            # 지금 동기화 실행
ntn workers sync pause <key>
ntn workers env set GITHUB_WEBHOOK_SECRET=...
ntn workers runs list                     # 최근 호출 내역
ntn workers runs logs <run-id>
ntn workers webhooks list
```

워커 구축을 요청받으면 `ntn workers new`로 스캐폴딩하고, `src/index.ts`에 코드를 작성하며, `ntn workers env set`으로 비밀을 설정하고 배포합니다. 전체 API 표면은 https://developers.notion.com/workers 의 Notion 문서에서 다룹니다.

## Notion 기반 마크다운 (`/markdown` 엔드포인트에서 사용됨)

표준 CommonMark에 Notion 전용 블록을 위한 XML과 유사한 태그가 추가되었습니다. 들여쓰기에는 **탭(tabs)**을 사용하세요.

**CommonMark 이상의 블록:**
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
- 색상: `<span color="blue">text</span>` 또는 첫 번째 줄의 블록 수준 `{color="blue"}`
- 수식: 인라인 `$x^2$`, 블록 `$$ ... $$`
- 인용(Citations): `[^https://example.com]`

**색상:** `gray brown orange yellow green blue purple pink red`, 배경용 `*_bg` 변형 포함.

제목 5/6은 H4로 축소됩니다. 여러 개의 `>` 줄은 별도의 인용 블록으로 렌더링됩니다 — 여러 줄 인용의 경우 단일 `>` 안에 `<br>`을 사용하세요.

## 올바른 경로 선택

| 작업 | mac / Linux | Windows |
|---|---|---|
| 페이지 읽기/쓰기, 검색, 데이터베이스 쿼리 | `ntn api ...` | curl |
| 에이전트 요약용 페이지 읽기 | `ntn api v1/pages/{id}/markdown` | curl `/markdown` 엔드포인트 |
| 파일 업로드 | `ntn files create < file` | 3단계 HTTP 흐름 |
| 일회성 API 탐색 | `ntn api ...` | curl |
| Notion 호스팅 sync / webhook / agent tool 빌드 | `ntn workers ...` | WSL2 + `ntn workers ...` |

## 참고 사항 (Notes)

- 페이지/데이터베이스 ID는 UUID입니다 (대시 유무 상관없이 모두 허용됨).
- 속도 제한(Rate limit): 평균 초당 약 3회 요청. CLI는 이를 우회하지 않습니다.
- API는 데이터베이스 **보기(view)** 필터를 설정할 수 없습니다 — UI 전용입니다.
- 페이지에 데이터 소스를 임베드하려면 데이터 소스 생성 시 `"is_inline": true`를 사용하세요.
- 진행률 표시줄을 억제하려면 항상 curl에 `-s`를 전달하세요 (더 깨끗한 에이전트 출력).
- 읽을 때 `jq`를 통해 JSON을 파이프하세요: `... | jq '.results[0].properties'`.
- Notion은 현재 MCP 서버도 제공합니다(`Notion MCP`, 이전 버전에 비해 DB 작업 시 토큰 효율이 약 91% 향상됨) — 세션 내부에서 스트리밍 방식으로 Notion에 접근하려면 Hermes의 MCP 지원을 통해 연결하세요. 그러나 대부분의 일회성 작업에는 위의 경로들로 충분합니다.
