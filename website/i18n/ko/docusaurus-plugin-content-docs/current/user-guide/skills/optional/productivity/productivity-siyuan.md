---
title: "Siyuan — API로 SiYuan 지식 베이스 조회 및 편집"
sidebar_label: "Siyuan"
description: "API로 SiYuan 지식 베이스를 조회하고 편집합니다"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Siyuan

API로 SiYuan 지식 베이스를 조회하고 편집합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Optional — install with `hermes skills install official/productivity/siyuan` |
| 경로 | `optional-skills/productivity/siyuan` |
| 버전 | `1.0.0` |
| 작성자 | FEUAZUR |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `SiYuan`, `Notes`, `Knowledge Base`, `PKM`, `API` |
| 관련 스킬 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트에 표시되는 내용입니다.
:::

# SiYuan 노트 API

curl을 사용해 [SiYuan](https://github.com/siyuan-note/siyuan) 커널 API로 셀프 호스팅 지식 베이스의 블록과 문서를 검색, 읽기, 생성, 업데이트, 삭제합니다. 추가 도구는 필요하지 않습니다 -- curl과 API 토큰만 있으면 됩니다.

## 사전 요구 사항

1. SiYuan 설치 및 실행(데스크톱 또는 Docker)
2. API 토큰 확인: **Settings > About > API token**
3. `${HERMES_HOME:-~/.hermes}/.env`에 저장:
   ```
   SIYUAN_TOKEN=your_token_here
   SIYUAN_URL=http://127.0.0.1:6806
   ```
   설정하지 않으면 `SIYUAN_URL`은 `http://127.0.0.1:6806`으로 기본 설정됩니다.

## API 기본 사항

모든 SiYuan API 호출은 **JSON 본문을 사용하는 POST**입니다. 모든 요청은 다음 패턴을 따릅니다.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/..." \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'
```

응답은 다음 구조의 JSON입니다.
```json
{"code": 0, "msg": "", "data": { ... }}
```
`code: 0`은 성공을 의미합니다. 그 외의 값은 오류이므로 자세한 내용은 `msg`를 확인하세요.

**ID 형식:** SiYuan ID는 `20210808180117-6v0mkxr`와 같은 형식입니다(14자리 타임스탬프 + 영숫자 7자).

## 빠른 참조

| 작업 | 엔드포인트 |
|-----------|----------|
| 전체 텍스트 검색 | `/api/search/fullTextSearchBlock` |
| SQL 쿼리 | `/api/query/sql` |
| 블록 읽기 | `/api/block/getBlockKramdown` |
| 하위 블록 읽기 | `/api/block/getChildBlocks` |
| 경로 가져오기 | `/api/filetree/getHPathByID` |
| 속성 가져오기 | `/api/attr/getBlockAttrs` |
| 노트북 목록 조회 | `/api/notebook/lsNotebooks` |
| 문서 목록 조회 | `/api/filetree/listDocsByPath` |
| 노트북 생성 | `/api/notebook/createNotebook` |
| 문서 생성 | `/api/filetree/createDocWithMd` |
| 블록 추가 | `/api/block/appendBlock` |
| 블록 업데이트 | `/api/block/updateBlock` |
| 문서 이름 변경 | `/api/filetree/renameDocByID` |
| 속성 설정 | `/api/attr/setBlockAttrs` |
| 블록 삭제 | `/api/block/deleteBlock` |
| 문서 삭제 | `/api/filetree/removeDocByID` |
| Markdown으로 내보내기 | `/api/export/exportMdContent` |

## 일반적인 작업

### 검색(전체 텍스트)

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/search/fullTextSearchBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "meeting notes", "page": 0}' | jq '.data.blocks[:5]'
```

### 검색(SQL)

블록 데이터베이스를 직접 쿼리합니다. SELECT 문만 안전합니다.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/query/sql" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stmt": "SELECT id, content, type, box FROM blocks WHERE content LIKE '\''%keyword%'\'' AND type='\''p'\'' LIMIT 20"}' | jq '.data'
```

유용한 열: `id`, `parent_id`, `root_id`, `box`(노트북 ID), `path`, `content`, `type`, `subtype`, `created`, `updated`.

### 블록 콘텐츠 읽기

블록 콘텐츠를 Kramdown(Markdown과 유사한) 형식으로 반환합니다.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/getBlockKramdown" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data.kramdown'
```

### 하위 블록 읽기

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/getChildBlocks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### 사람이 읽을 수 있는 경로 가져오기

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/getHPathByID" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### 블록 속성 가져오기

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/attr/getBlockAttrs" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "20210808180117-6v0mkxr"}' | jq '.data'
```

### 노트북 목록 조회

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/lsNotebooks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.data.notebooks[] | {id, name, closed}'
```

### 노트북의 문서 목록 조회

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/listDocsByPath" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"notebook": "NOTEBOOK_ID", "path": "/"}' | jq '.data.files[] | {id, name}'
```

### 문서 생성

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/createDocWithMd" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notebook": "NOTEBOOK_ID",
    "path": "/Meeting Notes/2026-03-22",
    "markdown": "# Meeting Notes\n\n- Discussed project timeline\n- Assigned tasks"
  }' | jq '.data'
```

### 노트북 생성

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/createNotebook" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My New Notebook"}' | jq '.data.notebook.id'
```

### 문서에 블록 추가

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/appendBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "parentID": "DOCUMENT_OR_BLOCK_ID",
    "data": "New paragraph added at the end.",
    "dataType": "markdown"
  }' | jq '.data'
```

다음도 사용할 수 있습니다. `/api/block/prependBlock`은 동일한 매개변수로 맨 앞에 삽입하며, `/api/block/insertBlock`은 `parentID` 대신 `previousID`를 사용해 특정 블록 뒤에 삽입합니다.

### 블록 콘텐츠 업데이트

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/updateBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "BLOCK_ID",
    "data": "Updated content here.",
    "dataType": "markdown"
  }' | jq '.data'
```

### 문서 이름 변경

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/filetree/renameDocByID" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID", "title": "New Title"}'
```

### 블록 속성 설정

사용자 지정 속성에는 `custom-` 접두사를 붙여야 합니다.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/attr/setBlockAttrs" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "BLOCK_ID",
    "attrs": {
      "custom-status": "reviewed",
      "custom-priority": "high"
    }
  }'
```

### 블록 삭제

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/block/deleteBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "BLOCK_ID"}'
```

문서 전체를 삭제하려면 `{"id": "DOC_ID"}`와 함께 `/api/filetree/removeDocByID`를 사용하세요.
노트북을 삭제하려면 `{"notebook": "NOTEBOOK_ID"}`와 함께 `/api/notebook/removeNotebook`을 사용하세요.

### 문서를 Markdown으로 내보내기

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/export/exportMdContent" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID"}' | jq -r '.data.content'
```

## 블록 유형

SQL 쿼리에서 사용하는 일반적인 `type` 값:

| 유형 | 설명 |
|------|-------------|
| `d` | 문서(루트 블록) |
| `p` | 문단 |
| `h` | 제목 |
| `l` | 목록 |
| `i` | 목록 항목 |
| `c` | 코드 블록 |
| `m` | 수학 블록 |
| `t` | 표 |
| `b` | 인용 블록 |
| `s` | 슈퍼 블록 |
| `html` | HTML 블록 |

## 주의 사항

- **모든 엔드포인트는 POST입니다** -- 읽기 전용 작업도 마찬가지입니다. GET을 사용하지 마세요.
- **SQL 안전성**: SELECT 쿼리만 사용하세요. INSERT/UPDATE/DELETE/DROP은 위험하므로 절대 전송하지 마세요.
- **ID 검증**: ID는 `YYYYMMDDHHmmss-xxxxxxx` 패턴과 일치합니다. 그 외의 값은 거부하세요.
- **오류 응답**: 데이터를 처리하기 전에 항상 응답에서 `code != 0`인지 확인하세요.
- **대규모 문서**: 블록 콘텐츠와 내보내기 결과는 매우 클 수 있습니다. SQL에서 `LIMIT`을 사용하고 `jq`로 필요한 내용만 추출하세요.
- **노트북 ID**: 특정 노트북으로 작업할 때는 먼저 `lsNotebooks`로 ID를 가져오세요.

## 대안: MCP 서버

curl 대신 네이티브 통합을 선호한다면 SiYuan MCP 서버를 설치하세요.

```yaml
# In ~/.hermes/config.yaml under mcp_servers:
mcp_servers:
  siyuan:
    command: npx
    args: ["-y", "@porkll/siyuan-mcp"]
    env:
      SIYUAN_TOKEN: "your_token"
      SIYUAN_URL: "http://127.0.0.1:6806"
```
