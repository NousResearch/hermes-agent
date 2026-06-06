---
title: "Siyuan"
sidebar_label: "Siyuan"
description: "curl을 통해 셀프 호스팅 지식 베이스에서 블록과 문서를 검색, 읽기, 생성, 관리하는 SiYuan Note API"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Siyuan

curl을 통해 셀프 호스팅 지식 베이스에서 블록과 문서를 검색, 읽기, 생성, 관리하는 SiYuan Note API입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택적(Optional) — `hermes skills install official/productivity/siyuan` 명령어로 설치 |
| 경로 | `optional-skills/productivity/siyuan` |
| 버전 | `1.0.0` |
| 작성자 | FEUAZUR |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `SiYuan`, `Notes`, `Knowledge Base`, `PKM`, `API` |
| 관련 스킬 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되어 있을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# SiYuan Note API

curl을 통한 [SiYuan](https://github.com/siyuan-note/siyuan) 커널 API를 사용하여 셀프 호스팅 지식 베이스에서 블록과 문서를 검색, 읽기, 생성, 업데이트 및 삭제합니다. 추가 도구 없이 curl과 API 토큰만 있으면 됩니다.

## 전제 조건

1. SiYuan 설치 및 실행 (데스크톱 또는 Docker)
2. API 토큰 발급: **Settings(설정) > About(정보) > API token**
3. `~/.hermes/.env` 파일에 저장:
   ```
   SIYUAN_TOKEN=your_token_here
   SIYUAN_URL=http://127.0.0.1:6806
   ```
   설정하지 않으면 `SIYUAN_URL`의 기본값은 `http://127.0.0.1:6806`입니다.

## API 기초

모든 SiYuan API 호출은 **JSON 본문(body)을 포함하는 POST** 방식입니다. 모든 요청은 다음 패턴을 따릅니다:

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/..." \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'
```

응답은 다음 구조를 가진 JSON입니다:
```json
{"code": 0, "msg": "", "data": { ... }}
```
`code: 0`은 성공을 의미합니다. 다른 모든 값은 오류를 나타냅니다 -- 자세한 내용은 `msg`를 확인하세요.

**ID 형식:** SiYuan ID는 `20210808180117-6v0mkxr` 형식 (14자리 타임스탬프 + 7자리 영숫자)입니다.

## 빠른 참조

| 작업 | 엔드포인트 |
|-----------|----------|
| 전체 텍스트 검색 | `/api/search/fullTextSearchBlock` |
| SQL 쿼리 | `/api/query/sql` |
| 블록 읽기 | `/api/block/getBlockKramdown` |
| 하위 블록 읽기 | `/api/block/getChildBlocks` |
| 경로 가져오기 | `/api/filetree/getHPathByID` |
| 속성 가져오기 | `/api/attr/getBlockAttrs` |
| 노트북 나열 | `/api/notebook/lsNotebooks` |
| 문서 나열 | `/api/filetree/listDocsByPath` |
| 노트북 생성 | `/api/notebook/createNotebook` |
| 문서 생성 | `/api/filetree/createDocWithMd` |
| 블록 덧붙이기(Append) | `/api/block/appendBlock` |
| 블록 업데이트 | `/api/block/updateBlock` |
| 문서 이름 변경 | `/api/filetree/renameDocByID` |
| 속성 설정 | `/api/attr/setBlockAttrs` |
| 블록 삭제 | `/api/block/deleteBlock` |
| 문서 삭제 | `/api/filetree/removeDocByID` |
| 마크다운 내보내기 | `/api/export/exportMdContent` |

## 일반적인 작업

### 검색 (전체 텍스트)

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/search/fullTextSearchBlock" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "meeting notes", "page": 0}' | jq '.data.blocks[:5]'
```

### 검색 (SQL)

블록 데이터베이스에 직접 쿼리합니다. SELECT 문만 안전합니다.

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/query/sql" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stmt": "SELECT id, content, type, box FROM blocks WHERE content LIKE '\''%keyword%'\'' AND type='\''p'\'' LIMIT 20"}' | jq '.data'
```

유용한 열(Columns): `id`, `parent_id`, `root_id`, `box` (노트북 ID), `path`, `content`, `type`, `subtype`, `created`, `updated`.

### 블록 내용 읽기

블록 내용을 Kramdown (마크다운과 유사한) 형식으로 반환합니다.

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

### 노트북 나열

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/notebook/lsNotebooks" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' | jq '.data.notebooks[] | {id, name, closed}'
```

### 노트북의 문서 나열

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

### 문서에 블록 덧붙이기 (Append)

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

또한 사용 가능: `/api/block/prependBlock` (동일한 파라미터, 맨 앞에 삽입) 및 `/api/block/insertBlock` (`parentID` 대신 `previousID`를 사용하여 특정 블록 뒤에 삽입).

### 블록 내용 업데이트

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

사용자 지정 속성은 반드시 `custom-` 접두사로 시작해야 합니다:

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

문서 전체를 삭제하려면: `{"id": "DOC_ID"}`와 함께 `/api/filetree/removeDocByID`를 사용하세요.
노트북을 삭제하려면: `{"notebook": "NOTEBOOK_ID"}`와 함께 `/api/notebook/removeNotebook`을 사용하세요.

### 문서를 마크다운으로 내보내기

```bash
curl -s -X POST "${SIYUAN_URL:-http://127.0.0.1:6806}/api/export/exportMdContent" \
  -H "Authorization: Token $SIYUAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": "DOCUMENT_ID"}' | jq -r '.data.content'
```

## 블록 유형 (Block Types)

SQL 쿼리에서 사용되는 일반적인 `type` 값:

| 유형 | 설명 |
|------|-------------|
| `d` | 문서 (루트 블록) |
| `p` | 단락 (Paragraph) |
| `h` | 제목 (Heading) |
| `l` | 목록 (List) |
| `i` | 목록 항목 (List item) |
| `c` | 코드 블록 (Code block) |
| `m` | 수식 블록 (Math block) |
| `t` | 표 (Table) |
| `b` | 인용구 (Blockquote) |
| `s` | 슈퍼 블록 (Super block) |
| `html` | HTML 블록 |

## 주의 사항 (Pitfalls)

- **모든 엔드포인트는 POST입니다** -- 읽기 전용 작업일지라도 마찬가지입니다. GET을 사용하지 마세요.
- **SQL 안전성**: SELECT 쿼리만 사용하세요. INSERT/UPDATE/DELETE/DROP은 위험하므로 절대 전송해서는 안 됩니다.
- **ID 검증**: ID는 `YYYYMMDDHHmmss-xxxxxxx` 패턴과 일치해야 합니다. 그 외의 것은 거부하세요.
- **오류 응답**: `data`를 처리하기 전에 항상 응답에서 `code != 0`인지 확인하세요.
- **대형 문서**: 블록 내용과 내보내기 결과는 매우 클 수 있습니다. SQL에서 `LIMIT`을 사용하고 필요한 것만 추출하기 위해 `jq`를 통해 파이프(pipe)하세요.
- **노트북 ID**: 특정 노트북으로 작업할 때, `lsNotebooks`를 통해 먼저 노트북 ID를 가져오세요.

## 대안: MCP 서버

curl 대신 네이티브 연동을 선호한다면 SiYuan MCP 서버를 설치하세요:

```yaml
# ~/.hermes/config.yaml 파일의 mcp_servers 아래에 추가:
mcp_servers:
  siyuan:
    command: npx
    args: ["-y", "@porkll/siyuan-mcp"]
    env:
      SIYUAN_TOKEN: "your_token"
      SIYUAN_URL: "http://127.0.0.1:6806"
```
