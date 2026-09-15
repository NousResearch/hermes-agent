---
title: "Airtable — curl을 통한 Airtable REST API"
sidebar_label: "Airtable"
description: "curl을 통한 Airtable REST API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Airtable

curl을 통한 Airtable REST API입니다. 레코드 CRUD, 필터, 업서트를 지원합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨(기본 설치) |
| 경로 | `skills/productivity/airtable` |
| 버전 | `1.1.0` |
| 작성자 | 커뮤니티 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Airtable`, `Productivity`, `Database`, `API` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화된 동안 에이전트가 보게 되는 내용입니다.
:::

# Airtable — 베이스, 테이블 및 레코드

`terminal` 도구를 사용해 `curl`로 Airtable의 REST API를 직접 다룹니다. MCP 서버도, OAuth 플로도, Python SDK도 필요 없이 `curl`과 개인 액세스 토큰만 사용합니다.

## 사전 준비

1. https://airtable.com/create/tokens에서 **개인 액세스 토큰(PAT)** 을 생성합니다(토큰은 `pat...`으로 시작합니다).
2. 다음 스코프를 부여합니다(최소 권한):
   - `data.records:read` — 행 읽기
   - `data.records:write` — 행 생성 / 업데이트 / 삭제
   - `schema.bases:read` — 베이스와 테이블 목록 조회
3. **중요:** 같은 토큰 UI에서 액세스하려는 각 베이스를 토큰의 **Access** 목록에 추가합니다. PAT는 베이스별로 범위가 지정되므로, 올바른 토큰이라도 잘못된 베이스에 사용하면 `403`을 반환합니다.
4. 토큰을 `${HERMES_HOME:-~/.hermes}/.env`에 저장하거나 `hermes setup`을 사용합니다:
   ```
   AIRTABLE_API_KEY=pat_your_token_here
   ```

> 참고: 기존 `key...` API 키는 2024년 2월에 더 이상 사용되지 않습니다. 이제 PAT와 OAuth 토큰만 작동합니다.

## API 기본 사항

- **엔드포인트:** `https://api.airtable.com/v0`
- **인증 헤더:** `Authorization: Bearer $AIRTABLE_API_KEY`
- **모든 요청**은 JSON을 사용합니다(POST/PATCH/PUT 본문에는 `Content-Type: application/json`을 사용).
- **객체 ID:** 베이스는 `app...`, 테이블은 `tbl...`, 레코드는 `rec...`, 필드는 `fld...`입니다. ID는 변경되지 않지만 이름은 변경될 수 있습니다. 자동화에서는 ID를 우선 사용합니다.
- **요청 제한:** 베이스당 초당 5개 요청입니다. `429`가 반환되면 대기합니다. 단일 베이스에 요청을 몰아서 보내면 제한됩니다.

기본 curl 패턴:
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=5" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

`-s`는 curl의 진행률 표시줄을 숨깁니다. Hermes의 도구 출력이 깔끔하게 유지되도록 모든 호출에서 계속 사용합니다. 읽기 쉬운 JSON을 위해 `python3 -m json.tool`(항상 존재) 또는 `jq`(설치된 경우)를 파이프로 연결합니다.

## 필드 유형(요청 본문 형태)

| 필드 유형 | 쓰기 형태 |
|---|---|
| 한 줄 텍스트 | `"Name": "hello"` |
| 긴 텍스트 | `"Notes": "multi\nline"` |
| 숫자 | `"Score": 42` |
| 체크박스 | `"Done": true` |
| 단일 선택 | `"Status": "Todo"` (`typecast: true`가 아닌 경우 이름이 이미 존재해야 함) |
| 다중 선택 | `"Tags": ["urgent", "bug"]` |
| 날짜 | `"Due": "2026-04-01"` |
| 날짜/시간(UTC) | `"At": "2026-04-01T14:30:00.000Z"` |
| URL / 이메일 / 전화번호 | `"Link": "https://…"` |
| 첨부 파일 | `"Files": [{"url": "https://…"}]` (Airtable이 가져와 다시 호스팅함) |
| 연결된 레코드 | `"Owner": ["recXXXXXXXXXXXXXX"]` (레코드 ID 배열) |
| 사용자 | `"AssignedTo": {"id": "usrXXXXXXXXXXXXXX"}` |

생성/업데이트 본문의 최상위에 `"typecast": true`를 전달하면 Airtable이 값을 자동으로 변환합니다(예: 즉석에서 새 선택 옵션을 만들거나 `"42"`를 `42`로 변환).

## 일반적인 쿼리

### 토큰이 볼 수 있는 베이스 목록 조회
```bash
curl -s "https://api.airtable.com/v0/meta/bases" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 베이스의 테이블과 스키마 목록 조회
```bash
curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
변경하기 전에 이것을 사용합니다. 정확한 필드 이름과 ID를 확인하고, 선택 필드의 `options.choices`를 표시하며, 기본 필드 이름을 보여줍니다.

### 레코드 목록 조회(처음 10개)
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=10" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 단일 레코드 가져오기
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### 레코드 필터링(filterByFormula)

Airtable 수식은 URL 인코딩해야 합니다. Python 표준 라이브러리에 맡기고 직접 인코딩하지 않습니다:
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
- 또는: `OR({Owner}='alice', {Owner}='bob')`
- 비어 있지 않음: `NOT({Assignee}='')`
- 날짜 비교: `IS_AFTER({Due}, TODAY())`

### 정렬하고 특정 필드만 선택
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?sort%5B0%5D%5Bfield%5D=Priority&sort%5B0%5D%5Bdirection%5D=asc&fields%5B%5D=Name&fields%5B%5D=Status" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
쿼리 매개변수의 대괄호는 반드시 URL 인코딩해야 합니다(`%5B` / `%5D`).

### 이름이 지정된 뷰 사용
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?view=Grid%20view&maxRecords=50" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
뷰는 저장된 필터와 정렬을 서버 측에서 적용합니다.

## 일반적인 변경 작업

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
배치 엔드포인트는 요청당 **레코드 10개**로 제한됩니다. 더 많이 삽입하려면 초당 5개 요청/베이스 제한을 지키도록 짧게 대기하면서 10개 단위로 반복합니다.

### 레코드 업데이트(PATCH — 병합하며 변경하지 않은 필드는 보존)
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}' | python3 -m json.tool
```

### 병합 필드로 업서트(ID 불필요)
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
`performUpsert`는 병합 필드 값이 새로운 레코드를 생성하고, 이미 존재하는 병합 필드 값의 레코드를 패치합니다. 멱등적인 동기화에 적합합니다.

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

## 페이지 매김

목록 엔드포인트는 페이지당 최대 **레코드 100개**를 반환합니다. 응답에 `"offset": "..."`이 포함되어 있으면 다음 호출에 다시 전달합니다. 필드가 없어질 때까지 반복합니다:

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

## 일반적인 Hermes 워크플로

1. **인증 확인.** `curl -s -o /dev/null -w "%{http_code}\n" https://api.airtable.com/v0/meta/bases -H "Authorization: Bearer $AIRTABLE_API_KEY"` — `200`을 예상합니다.
2. **베이스 찾기.** 베이스 목록을 조회하거나(위 단계 참조), 토큰에 `schema.bases:read`가 없으면 사용자에게 `app...` ID를 직접 요청합니다.
3. **스키마 확인.** `GET /v0/meta/bases/$BASE_ID/tables` — 변경하기 전에 정확한 필드 이름과 기본 필드 이름을 세션에 로컬로 캐시합니다.
4. **쓰기 전에 읽기.** "Y 조건에 맞는 X 업데이트"의 경우 먼저 `filterByFormula`로 `rec...` ID를 확인한 다음 `PATCH /v0/$BASE_ID/$TABLE/$RECORD_ID`를 사용합니다. 레코드 ID를 절대 추측하지 않습니다.
5. **배치 쓰기.** 관련 생성 작업을 하나의 10레코드 POST로 합쳐 초당 5개 요청 예산을 지킵니다.
6. **파괴적 작업.** API로 삭제한 내용은 되돌릴 수 없습니다. 사용자가 "모든 X 삭제"라고 하면 삭제를 실행하기 전에 필터와 레코드 수를 되풀이해 확인을 받습니다.

## 주의할 점

- **`filterByFormula`는 반드시 URL 인코딩해야 합니다.** 공백이나 ASCII가 아닌 문자가 포함된 필드 이름도 인코딩해야 합니다(`{My Field}` → `%7BMy%20Field%7D`). 위 패턴처럼 Python 표준 라이브러리를 사용하고 직접 이스케이프하지 않습니다.
- **빈 필드는 응답에서 생략됩니다.** 누락된 `"Assignee"` 키는 필드가 없다는 뜻이 아니라 해당 레코드의 값이 비어 있다는 뜻입니다. 필드가 없다고 결론 내리기 전에 3단계의 스키마를 확인합니다.
- **PATCH와 PUT.** `PATCH`는 제공된 필드를 레코드에 병합합니다. `PUT`은 레코드를 완전히 대체하며 포함하지 않은 필드를 모두 지웁니다. 기본값으로 `PATCH`를 사용합니다.
- **단일 선택 옵션은 존재해야 합니다.** `Shipping`이 필드의 옵션 목록에 없을 때 `"Status": "Shipping"`을 쓰면 `typecast": true`를 전달하지 않는 한 `INVALID_MULTIPLE_CHOICE_OPTIONS` 오류가 발생합니다(이 경우 옵션이 자동 생성됨).
- **베이스별 토큰 범위.** 한 베이스에서는 작동하지만 다른 베이스에서 `403`이 발생한다면 토큰의 Access 목록에 해당 베이스가 없는 것입니다. 스코프나 인증 문제가 아닙니다. 액세스 권한을 부여하려면 사용자를 https://airtable.com/create/tokens로 안내합니다.
- **요청 제한은 토큰이 아니라 베이스별입니다.** `baseA`에서 초당 5개, `baseB`에서 초당 5개를 요청하는 것은 괜찮지만 `baseA` 하나에서 초당 6개를 요청하면 제한됩니다. `429`에서 `Retry-After` 헤더를 모니터링합니다.

## Hermes 관련 중요 참고 사항

- **항상 `curl`과 함께 `terminal` 도구를 사용합니다.** `web_extract`는 인증 헤더를 보낼 수 없고 `browser_navigate`는 UI 인증이 필요하며 느리므로 사용하지 않습니다.
- **`AIRTABLE_API_KEY`는 `${HERMES_HOME:-~/.hermes}/.env`에서 이 스킬이 로드될 때 자동으로 서브프로세스로 전달됩니다.** 각 `curl` 호출 전에 다시 내보낼 필요가 없습니다.
- **수식에서 중괄호를 주의해서 이스케이프합니다.** heredoc 본문에서 `{Status}`는 그대로 사용합니다. 셸 인수에서 `{Status}`는 `{...}` 중괄호 확장 문맥 밖에서는 안전하지만, 동적 문자열은 URL에 삽입하기 전에 `python3 urllib.parse.quote`로 처리합니다.
- **`jq`(선택 사항)보다 `python3 -m json.tool`(항상 존재)로 보기 좋게 출력합니다.** 필터링/프로젝션이 필요할 때만 `jq`를 사용합니다.
- **페이지 매김은 전체가 아니라 페이지 단위입니다.** Airtable의 100레코드 제한은 고정되어 있어 늘릴 수 없습니다. 필드가 없어질 때까지 `offset`을 사용해 반복합니다.
- **2xx가 아닌 응답에서는 `errors` 배열을 읽습니다.** Airtable은 무엇이 잘못되었는지 정확히 알려주는 `AUTHENTICATION_REQUIRED`, `INVALID_PERMISSIONS`, `MODEL_ID_NOT_FOUND`, `INVALID_MULTIPLE_CHOICE_OPTIONS` 같은 구조화된 오류 코드를 반환합니다.
