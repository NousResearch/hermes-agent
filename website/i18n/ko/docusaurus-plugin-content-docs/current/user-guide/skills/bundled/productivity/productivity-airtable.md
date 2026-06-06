---
title: "Airtable — curl을 통한 Airtable REST API"
sidebar_label: "Airtable"
description: "curl을 통한 Airtable REST API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Airtable

curl을 통한 Airtable REST API. 레코드 CRUD, 필터, 업서트(upserts).

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/airtable` |
| Version | `1.1.0` |
| Author | community |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Airtable`, `Productivity`, `Database`, `API` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Airtable — 베이스, 테이블 및 레코드 (Bases, Tables & Records)

`terminal` 도구를 사용하여 `curl`을 통해 Airtable의 REST API와 직접 작업합니다. MCP 서버, OAuth 흐름, Python SDK가 필요하지 않으며 — 오직 `curl`과 개인 액세스 토큰(PAT)만 있으면 됩니다.

## 사전 요구 사항 (Prerequisites)

1. https://airtable.com/create/tokens 에서 **개인 액세스 토큰(Personal Access Token, PAT)**을 생성합니다 (토큰은 `pat...`으로 시작함).
2. 다음 범위를 부여합니다 (최소):
   - `data.records:read` — 행 읽기
   - `data.records:write` — 행 생성 / 업데이트 / 삭제
   - `schema.bases:read` — 베이스 및 테이블 나열
3. **중요:** 동일한 토큰 UI에서 접근하려는 각 베이스를 토큰의 **Access** 목록에 추가합니다. PAT는 베이스 단위로 범위가 지정됩니다 — 잘못된 베이스에 유효한 토큰을 사용하면 `403`이 반환됩니다.
4. `~/.hermes/.env` (또는 `hermes setup`을 통해)에 토큰을 저장합니다:
   ```
   AIRTABLE_API_KEY=pat_your_token_here
   ```

> 참고: 기존의 `key...` 형태의 API 키는 2024년 2월에 더 이상 사용되지 않습니다. 이제 PAT와 OAuth 토큰만 작동합니다.

## API 기초 (API Basics)

- **엔드포인트:** `https://api.airtable.com/v0`
- **인증 헤더:** `Authorization: Bearer $AIRTABLE_API_KEY`
- **모든 요청**은 JSON을 사용합니다 (모든 POST/PATCH/PUT 본문에 대해 `Content-Type: application/json` 사용).
- **객체 ID:** 베이스 `app...`, 테이블 `tbl...`, 레코드 `rec...`, 필드 `fld...`. ID는 절대 변경되지 않지만 이름은 변경될 수 있습니다. 자동화에서는 ID를 사용하는 것을 선호하세요.
- **속도 제한:** 베이스당 5회 요청/초. `429` → 백오프(back off)하세요. 단일 베이스에 대한 버스트 요청은 제한될 것입니다.

기본 curl 패턴:
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=5" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

`-s`는 curl의 진행률 표시줄을 억제합니다 — 도구 출력이 Hermes를 위해 깔끔하게 유지되도록 모든 호출에 이 옵션을 설정해 두세요. 읽기 쉬운 JSON을 위해 `python3 -m json.tool` (항상 존재함) 또는 `jq` (설치된 경우)를 파이프합니다.

## 필드 유형 (요청 본문 형태) (Field Types)

| 필드 유형 | 쓰기 형태 |
|---|---|
| 한 줄 텍스트 | `"Name": "hello"` |
| 긴 텍스트 | `"Notes": "multi\nline"` |
| 숫자 | `"Score": 42` |
| 체크박스 | `"Done": true` |
| 단일 선택 | `"Status": "Todo"` (`typecast: true`가 아닌 이상 이름이 이미 존재해야 함) |
| 다중 선택 | `"Tags": ["urgent", "bug"]` |
| 날짜 | `"Due": "2026-04-01"` |
| 날짜 및 시간 (UTC) | `"At": "2026-04-01T14:30:00.000Z"` |
| URL / 이메일 / 전화 | `"Link": "https://…"` |
| 첨부파일 | `"Files": [{"url": "https://…"}]` (Airtable이 가져와서 재호스팅함) |
| 연결된 레코드 | `"Owner": ["recXXXXXXXXXXXXXX"]` (레코드 ID 배열) |
| 사용자 | `"AssignedTo": {"id": "usrXXXXXXXXXXXXXX"}` |

생성/업데이트 본문의 최상위 수준에서 `"typecast": true`를 전달하면 Airtable이 값을 자동으로 강제 변환하도록 할 수 있습니다 (예: 즉석에서 새 선택 옵션 생성, `"42"` → `42`로 변환).

## 일반적인 쿼리 (Common Queries)

### 토큰이 볼 수 있는 베이스 나열
```bash
curl -s "https://api.airtable.com/v0/meta/bases" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 베이스의 테이블 + 스키마 나열
```bash
curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
수정하기 전에 이것을 사용하세요 — 정확한 필드 이름과 ID를 확인하고, 선택 필드에 대한 `options.choices`를 표시하며, 기본 필드(primary-field) 이름을 표시합니다.

### 레코드 나열 (처음 10개)
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=10" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 단일 레코드 가져오기
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 레코드 필터링 (filterByFormula)
Airtable 수식은 URL 인코딩되어야 합니다. Python 표준 라이브러리를 사용하세요 — 절대 수동으로 인코딩하지 마세요:
```bash
FORMULA="{Status}='Todo'"
ENC=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$FORMULA")
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?filterByFormula=$ENC&maxRecords=20" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

유용한 수식 패턴:
- 정확히 일치: `{Email}='user@example.com'`
- 포함: `FIND('bug', LOWER({Title}))`
- 여러 조건: `AND({Status}='Todo', {Priority}='High')`
- 또는(Or): `OR({Owner}='alice', {Owner}='bob')`
- 비어 있지 않음: `NOT({Assignee}='')`
- 날짜 비교: `IS_AFTER({Due}, TODAY())`

### 정렬 + 특정 필드 선택
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?sort%5B0%5D%5Bfield%5D=Priority&sort%5B0%5D%5Bdirection%5D=asc&fields%5B%5D=Name&fields%5B%5D=Status" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
쿼리 매개변수의 대괄호는 반드시 URL 인코딩되어야 합니다 (`%5B` / `%5D`).

### 명명된 뷰(Named view) 사용
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?view=Grid%20view&maxRecords=50" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
뷰는 저장된 필터 + 정렬을 서버 측에 적용합니다.

## 일반적인 데이터 변경 (Common Mutations)

### 레코드 생성
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Name":"New task","Status":"Todo","Priority":"High"}}' | python3 -m json.tool
```

### 한 번의 호출로 최대 10개 레코드 생성
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "typecast": true,
    "records": [
      {"fields": {"Name": "Task A", "Status": "Todo"}},
      {"fields": {"Name": "Task B", "Status": "In progress"}}
    ]
  }' | python3 -m json.tool
```
일괄(Batch) 엔드포인트는 **요청당 10개 레코드**로 제한됩니다. 더 큰 삽입의 경우 베이스당 초당 5개 요청을 존중하기 위해 짧은 절전(sleep)과 함께 10개 단위의 배치로 반복하세요.

### 레코드 업데이트 (PATCH — 병합, 변경되지 않은 필드 보존)
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}' | python3 -m json.tool
```

### 병합 필드로 업서트 (ID 불필요)
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "performUpsert": {"fieldsToMergeOn": ["Email"]},
    "records": [
      {"fields": {"Email": "user@example.com", "Status": "Active"}}
    ]
  }' | python3 -m json.tool
```
`performUpsert`는 병합 필드(merge-field) 값이 새로운 레코드를 생성하고, 병합 필드 값이 이미 존재하는 레코드는 패치(patch)합니다. 멱등성 동기화에 유용합니다.

### 레코드 삭제
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 한 번의 호출로 최대 10개 레코드 삭제
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE?records%5B%5D=rec1&records%5B%5D=rec2" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

## 페이지네이션 (Pagination)

리스트 엔드포인트는 페이지당 최대 **100개의 레코드**를 반환합니다. 응답에 `"offset": "..."`이 포함된 경우 다음 호출 시 다시 전달하세요. 필드가 없어질 때까지 반복합니다:

```bash
OFFSET=""
while :; do
  URL="https://api.airtable.com/v0/$BASE_ID/$TABLE?pageSize=100"
  [ -n "$OFFSET" ] && URL="$URL&offset=$OFFSET"
  RESP=$(curl -s "$URL" -H "Authorization: Bearer $AIRTABLE_API_KEY")
  echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(r["id"], r["fields"].get("Name","")) for r in d["records"]]'
  OFFSET=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("offset",""))')
  [ -z "$OFFSET" ] && break
done
```

## 일반적인 Hermes 워크플로우 (Typical Hermes Workflow)

1. **인증 확인.** `curl -s -o /dev/null -w "%{http_code}\n" https://api.airtable.com/v0/meta/bases -H "Authorization: Bearer $AIRTABLE_API_KEY"` — `200`을 기대합니다.
2. **베이스 찾기.** 베이스를 나열(위 단계)하거나, 토큰에 `schema.bases:read` 권한이 없는 경우 사용자에게 `app...` ID를 직접 요청합니다.
3. **스키마 검사.** `GET /v0/meta/bases/$BASE_ID/tables` — 수정하기 전에 세션 로컬에 정확한 필드 이름과 기본 필드 이름을 캐시합니다.
4. **쓰기 전에 읽기.** "Y가 조건인 X를 업데이트" 하려면 먼저 `filterByFormula`를 사용하여 `rec...` ID를 확인한 다음 `PATCH /v0/$BASE_ID/$TABLE/$RECORD_ID`를 실행합니다. 레코드 ID를 절대 추측하지 마세요.
5. **일괄 쓰기.** 초당 5개 요청 예산 내에 유지하기 위해 관련된 생성 작업을 하나의 10개 레코드 POST로 결합합니다.
6. **파괴적인 작업.** 삭제는 API를 통해 취소할 수 없습니다. 사용자가 "모든 X를 삭제"하라고 할 경우, 필터 + 레코드 수를 에코백(echo back)하고 실행하기 전에 확인을 받으세요.

## 주의사항 (Pitfalls)

- **`filterByFormula`는 반드시 URL 인코딩되어야 합니다.** 공백이나 ASCII가 아닌 문자가 포함된 필드 이름도 인코딩이 필요합니다(`{My Field}` → `%7BMy%20Field%7D`). Python 표준 라이브러리(위의 패턴)를 사용하세요 — 절대 수동으로 이스케이프하지 마세요.
- **빈 필드는 응답에서 생략됩니다.** `"Assignee"` 키가 누락된 것은 필드가 존재하지 않는다는 의미가 아닙니다 — 이 레코드의 값이 비어 있다는 의미입니다. 필드가 누락되었다고 결론을 내리기 전에 스키마(3단계)를 확인하세요.
- **PATCH vs PUT.** `PATCH`는 제공된 필드를 레코드에 병합합니다. `PUT`은 레코드를 완전히 교체하고 포함하지 않은 필드는 모두 지웁니다. `PATCH`를 기본값으로 사용하세요.
- **단일 선택 옵션이 존재해야 합니다.** 선택 필드의 옵션 목록에 없는 `"Status": "Shipping"`을 작성하면 `"typecast": true`(옵션을 자동 생성함)를 전달하지 않는 한 `INVALID_MULTIPLE_CHOICE_OPTIONS` 오류가 발생합니다.
- **베이스 단위 토큰 범위(scoping).** 한 베이스에서는 작동하지만 다른 베이스에서 `403`이 발생하는 경우, 토큰의 액세스 목록에 해당 베이스가 포함되지 않은 것입니다 — 범위나 인증 문제가 아닙니다. 사용자를 https://airtable.com/create/tokens 로 보내서 권한을 부여하도록 하세요.
- **속도 제한은 토큰당이 아니라 베이스당 적용됩니다.** `baseA`에서 초당 5개 요청, `baseB`에서 초당 5개 요청은 괜찮습니다; `baseA`에만 초당 6개 요청을 하면 제한됩니다. `429` 발생 시 `Retry-After` 헤더를 모니터링하세요.

## Hermes를 위한 중요 참고 사항 (Important Notes for Hermes)

- **항상 `curl`과 함께 `terminal` 도구를 사용하세요.** `web_extract`(인증 헤더를 보낼 수 없음)나 `browser_navigate`(UI 인증이 필요하고 느림)를 사용하지 **마세요**.
- **이 스킬이 로드될 때 `AIRTABLE_API_KEY`는 `~/.hermes/.env`에서 하위 프로세스로 자동으로 흐릅니다** — 각 `curl` 호출 전에 다시 export할 필요가 없습니다.
- **수식에서 중괄호를 주의 깊게 이스케이프하세요.** 히어독(heredoc) 본문에서는 `{Status}`가 리터럴입니다. 쉘 인수에서는 `{Status}`가 `{...}` 중괄호 확장 컨텍스트 외부에 있으면 안전하지만 — 동적 문자열은 URL에 붙여넣기 전에 `python3 urllib.parse.quote`를 통과하도록 하세요.
- **`jq` (선택 사항)보다는 `python3 -m json.tool` (항상 존재함)로 예쁘게 출력(Pretty-print)하세요.** 필터링/프로젝션이 필요할 때만 `jq`를 사용하세요.
- **페이지네이션은 전역이 아니라 페이지 단위입니다.** Airtable의 100개 레코드 제한은 엄격한 한도입니다; 이를 늘릴 방법은 없습니다. 필드가 없어질 때까지 `offset`으로 반복하세요.
- **2xx가 아닌 응답에 대한 `errors` 배열을 읽으세요** — Airtable은 무엇이 잘못되었는지 정확히 알려주는 `AUTHENTICATION_REQUIRED`, `INVALID_PERMISSIONS`, `MODEL_ID_NOT_FOUND`, `INVALID_MULTIPLE_CHOICE_OPTIONS`와 같은 구조화된 오류 코드를 반환합니다.
