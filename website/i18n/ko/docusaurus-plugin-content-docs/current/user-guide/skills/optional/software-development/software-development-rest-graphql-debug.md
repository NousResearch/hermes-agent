---
title: "Rest Graphql Debug — REST/GraphQL API 디버깅: 상태 코드, 인증, 스키마, 재현"
sidebar_label: "Rest Graphql Debug"
description: "REST/GraphQL API 디버깅: 상태 코드, 인증, 스키마, 재현"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Rest Graphql Debug

REST/GraphQL API를 디버깅합니다: 상태 코드, 인증, 스키마, 재현.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/software-development/rest-graphql-debug`로 설치 |
| 경로 | `optional-skills/software-development/rest-graphql-debug` |
| 버전 | `1.2.0` |
| 작성자 | eren-karakus0 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `api`, `rest`, `graphql`, `http`, `debugging`, `testing`, `curl`, `integration` |
| 관련 스킬 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# API 테스트 및 디버깅

Hermes 도구를 통해 REST 및 GraphQL 진단을 진행합니다 — `terminal`은 `curl`, `execute_code`는 Python `requests`, `web_extract`는 공급업체 문서에 사용합니다. 수정 방법을 추측하기 전에 실패한 계층을 분리하세요.

## 사용 시점

- API가 예상치 못한 상태 코드나 본문을 반환할 때
- 인증이 실패할 때 (토큰 갱신, OAuth, API 키 이후에도 401/403)
- Postman에서는 작동하지만 코드에서는 실패할 때
- 웹훅/콜백 통합을 디버깅할 때
- API 통합 테스트를 작성하거나 검토할 때
- 속도 제한 또는 페이지 매김 문제가 있을 때

UI 렌더링, DB 쿼리 튜닝, DNS/방화벽 인프라에는 사용하지 마세요(에스컬레이션).

## 핵심 원칙

**계층을 분리한 다음 수정하세요.** 200 OK에도 깨진 데이터가 숨어 있을 수 있습니다. 500은 인증 오타 한 글자를 가릴 수 있습니다. 순서대로 체인을 따라가고, 단계를 건너뛰지 마세요.

```
1. Connectivity   → can we reach the host at all?
1.5 Timeouts      → connect-slow vs read-slow?
2. TLS/SSL        → cert valid and trusted?
3. Auth           → credentials correct and unexpired?
4. Request format → payload shape match server expectations?
5. Response parse → does our code accept what came back?
6. Semantics      → does the data mean what we assume?
```

## 5분 빠른 시작

### 터미널에서 REST 사용

```python
# Verbose request/response exchange
terminal('curl -v https://api.example.com/users/1')

# POST with JSON
terminal("""curl -X POST https://api.example.com/users \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"name":"test","email":"test@example.com"}'""")

# Headers only
terminal('curl -sI https://api.example.com/health')

# Pretty-print JSON
terminal('curl -s https://api.example.com/users | python3 -m json.tool')
```

### 터미널에서 GraphQL 사용

```python
terminal("""curl -X POST https://api.example.com/graphql \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"query":"{ user(id: 1) { name email } }"}'""")
```

**GraphQL의 함정:** 쿼리가 실패해도 서버가 HTTP 200을 반환하는 경우가 많습니다. 상태 코드와 관계없이 항상 `errors` 필드를 확인하세요.

```python
execute_code('''
import os, requests
resp = requests.post(
    "https://api.example.com/graphql",
    json={"query": "{ user(id: 1) { name email } }"},
    headers={"Authorization": f"Bearer {os.environ['TOKEN']}"},
    timeout=10,
)
data = resp.json()
if data.get("errors"):
    for err in data["errors"]:
        print(f"GraphQL error: {err['message']} (path: {err.get('path')})")
print(data.get("data"))
''')
```

### execute_code를 통한 Python (`requests`) 사용

```python
execute_code('''
import requests
resp = requests.get(
    "https://api.example.com/users/1",
    headers={"Authorization": "Bearer <TOKEN>"},
    timeout=(3.05, 30),  # (connect, read)
)
print(resp.status_code, dict(resp.headers))
print(resp.text[:500])
''')
```

## 계층별 디버깅 흐름

### 1단계 — 연결

```python
terminal('nslookup api.example.com')
terminal('curl -v --connect-timeout 5 https://api.example.com/health')
```

실패 원인: DNS가 해석되지 않음, 방화벽, VPN 필요, 프록시 누락.

### 1.5단계 — 타임아웃

*연결할 수 없음*과 *연결되었지만 느림*을 구분하세요.

```python
terminal('''curl -w "dns:%{time_namelookup}s connect:%{time_connect}s tls:%{time_appconnect}s ttfb:%{time_starttransfer}s total:%{time_total}s\\n" \\
  -o /dev/null -s https://api.example.com/endpoint''')
```

Python에서는 항상 튜플 타임아웃을 전달하세요 — `requests`에는 기본값이 없으며 영원히 멈출 수 있습니다.

```python
execute_code('''
import requests
from requests.exceptions import ConnectTimeout, ReadTimeout
try:
    requests.get(url, timeout=(3.05, 30))
except ConnectTimeout:
    print("Cannot reach host — DNS, firewall, VPN")
except ReadTimeout:
    print("Connected but server is slow")
''')
```

진단: `time_connect`가 높으면 네트워크/방화벽 문제이고, `time_connect`가 낮은데 `time_starttransfer`가 높으면 서버가 느린 것입니다.

### 2단계 — TLS/SSL

```python
terminal('curl -vI https://api.example.com 2>&1 | grep -E "SSL|subject|expire|issuer"')
```

실패 원인: 인증서 만료, 자체 서명, 호스트 이름 불일치, CA 번들 누락. 임시 디버깅에만 `-k`를 사용하고, 코드에서는 절대 사용하지 마세요.

### 3단계 — 인증

```python
# Token validity check
terminal('curl -s -o /dev/null -w "%{http_code}\\n" -H "Authorization: Bearer $TOKEN" https://api.example.com/me')

# Decode JWT exp claim — handles base64url padding correctly
execute_code('''
import json, base64, os
tok = os.environ["TOKEN"]
payload = tok.split(".")[1]
payload += "=" * (-len(payload) % 4)
print(json.dumps(json.loads(base64.urlsafe_b64decode(payload)), indent=2))
''')
```

체크리스트:
- 토큰이 만료되었나요? (JWT의 `exp` 클레임)
- 올바른 스킴인가요? Bearer 대 Basic 대 Token 대 `X-Api-Key`
- 올바른 환경인가요? 프로덕션에서 스테이징 키를 사용하는 것은 흔한 문제입니다.
- API 키를 헤더와 쿼리 파라미터(`?api_key=…`) 중 올바른 위치에 넣었나요?

### 4단계 — 요청 형식

```python
terminal("""curl -v -X POST https://api.example.com/endpoint \\
  -H 'Content-Type: application/json' \\
  -d '{"key":"value"}' 2>&1""")
```

**Content-Type/body 불일치 — 조용히 발생하는 415/400:**

```python
# WRONG — data= sends form-encoded, header lies
requests.post(url, data='{"k":"v"}', headers={"Content-Type": "application/json"})

# RIGHT — json= auto-sets header AND serializes
requests.post(url, json={"k": "v"})

# WRONG — Accept says XML, code calls .json()
requests.get(url, headers={"Accept": "text/xml"})

# RIGHT — let requests build multipart with boundary
requests.post(url, files={"file": open("doc.pdf", "rb")})
```

일반적인 문제: form-encoded와 JSON의 혼동, 필수 필드 누락, 잘못된 HTTP 메서드, 인코딩되지 않은 쿼리 파라미터.

### 5단계 — 응답 파싱

항상 `.json()`을 호출하기 전에 콘텐츠 타입을 확인하세요.

```python
execute_code('''
import requests
resp = requests.post(url, json=payload, timeout=10)
print(f"status={resp.status_code}")
print(f"headers={dict(resp.headers)}")
ct = resp.headers.get("Content-Type", "")
if "application/json" in ct:
    print(resp.json())
else:
    print(f"unexpected content-type {ct!r}, body={resp.text[:500]!r}")
''')
```

실패 원인: JSON을 예상했지만 HTML 오류 페이지가 반환됨, 빈 본문, 잘못된 문자 집합.

### 6단계 — 의미 검증

파싱은 깔끔하게 되었지만 — 데이터가 *올바른가요*?

- `"status": "active"`가 코드가 생각하는 의미인가요?
- 응답의 ID가 요청한 ID와 일치하나요?
- 타임스탬프가 예상한 시간대인가요?
- 페이지 매김이 모든 결과를 반환하나요, 아니면 1페이지만 반환하나요?

## HTTP 상태 코드 대응표

### 401 Unauthorized — 인증 정보가 없거나 유효하지 않음

1. `Authorization` 헤더가 실제로 포함되어 있나요? (`curl -v`로 확인)
2. 토큰이 올바르고 만료되지 않았나요?
3. 올바른 인증 스킴인가요? (`Bearer` 대 `Basic` 대 `Token`)
4. 일부 API는 헤더 대신 쿼리 파라미터(`?api_key=…`)를 사용합니다.

### 403 Forbidden — 인증되었지만 권한이 없음

1. 토큰에 필요한 범위/권한이 있나요?
2. 리소스가 다른 계정 소유인가요?
3. IP 허용 목록이 차단하고 있나요?
4. 브라우저에서 CORS 문제가 있나요? (`Access-Control-Allow-Origin` 확인)

### 404 Not Found — 리소스가 없거나 URL이 잘못됨

1. 경로가 올바른가요? (trailing slash, 오타, 버전 접두사)
2. 리소스 ID가 존재하나요?
3. 올바른 API 버전인가요? (`/v1/` 대 `/v2/`)
4. 올바른 기본 URL인가요? (스테이징 대 프로덕션)

### 409 Conflict — 상태 충돌

1. 리소스가 이미 존재하나요(중복 생성)?
2. 오래된 `ETag`/`If-Match`인가요?
3. 다른 프로세스가 동시에 수정했나요?

### 422 Unprocessable Entity — JSON은 유효하지만 데이터가 유효하지 않음

오류 본문에는 보통 잘못된 필드가 표시됩니다. 다음을 확인하세요.
- 필드 타입 (문자열 대 정수, 날짜 형식)
- 필수 대 선택 사항
- 허용된 집합 안의 열거형 값

### 429 Too Many Requests — 속도 제한에 걸림

`Retry-After` 및 `X-RateLimit-*` 헤더를 확인하세요. 지수 백오프:

```python
execute_code('''
import time, requests

def with_backoff(method, url, **kwargs):
    for attempt in range(5):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code != 429:
            return resp
        wait = int(resp.headers.get("Retry-After", 2 ** attempt))
        time.sleep(wait)
    return resp
''')
```

### 5xx — 서버 측 오류, 보통 사용자의 잘못이 아님

- **500** — 서버 버그입니다. 상관관계 ID를 수집하여 공급업체에 신고하세요.
- **502** — 업스트림이 다운되었습니다. 백오프 후 재시도하세요.
- **503** — 과부하/유지보수 중입니다. 상태 페이지를 확인하세요.
- **504** — 업스트림 타임아웃입니다. 페이로드를 줄이거나 타임아웃을 늘리세요.

모든 5xx에 대해 지터가 포함된 백오프를 사용하고, 지속되면 알림을 보내세요.

## 페이지 매김 및 멱등성

**페이지 매김.** 모든 결과를 받고 있는지 확인하세요. `next_cursor`, `next_page`, `total_count`를 찾아보세요. 두 가지 패턴이 있습니다.
- 오프셋 (`?limit=100&offset=200`) — 단순하지만 데이터가 변경되면 항목을 건너뛸 수 있습니다.
- 커서 (`?cursor=abc123`) — 실시간 또는 대규모 데이터셋에 선호됩니다.

**멱등성.** 멱등적이지 않은 작업(POST)에는 `Idempotency-Key: <uuid>`를 전송하여 재시도로 인해 이중 청구/이중 생성이 발생하지 않도록 하세요. 결제와 주문에는 필수입니다.

## 계약 검증

스키마 변경이 프로덕션에 도달하기 전에 감지하세요.

```python
execute_code('''
import requests

def validate_user(data: dict) -> list[str]:
    errors = []
    required = {"id": int, "email": str, "created_at": str}
    for field, expected in required.items():
        if field not in data:
            errors.append(f"missing field: {field}")
        elif not isinstance(data[field], expected):
            errors.append(f"{field}: want {expected.__name__}, got {type(data[field]).__name__}")
    return errors

resp = requests.get(f"{BASE}/users/1", headers=HEADERS, timeout=10)
issues = validate_user(resp.json())
if issues:
    print(f"contract violations: {issues}")
''')
```

API 업그레이드 후, 새로운 타사 연동 시 또는 CI 스모크 테스트에서 실행하세요.

## 상관관계 ID

항상 공급업체의 요청 ID를 수집하세요 — 공급업체 지원팀에 문의하는 가장 빠른 방법입니다.

```python
execute_code('''
import requests
resp = requests.post(url, json=payload, headers=headers, timeout=10)
request_id = (
    resp.headers.get("X-Request-Id")
    or resp.headers.get("X-Trace-Id")
    or resp.headers.get("CF-Ray")  # Cloudflare
)
if resp.status_code >= 400:
    print(f"failed status={resp.status_code} req_id={request_id} ts={resp.headers.get('Date')}")
''')
```

**공급업체 버그 신고 템플릿:**

```
Endpoint:    POST /api/v1/orders
Request ID:  req_abc123xyz
Timestamp:   2026-03-17T14:30:00Z
Status:      500
Expected:    201 with order object
Actual:      500 {"error":"internal server error"}
Repro:       curl -X POST … (auth: <REDACTED>)
```

## 회귀 테스트 템플릿

이 내용을 `tests/`에 넣고 `terminal('pytest tests/test_api_smoke.py -v')`로 실행하세요.

```python
import os, requests, pytest

BASE_URL = os.environ.get("API_BASE_URL", "https://api.example.com")
TOKEN    = os.environ.get("API_TOKEN", "")
HEADERS  = {"Authorization": f"Bearer {TOKEN}"}

class TestAPISmoke:
    def test_health(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200

    def test_list_users_returns_array(self):
        resp = requests.get(f"{BASE_URL}/users", headers=HEADERS, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("data", data), list)

    def test_get_user_required_fields(self):
        resp = requests.get(f"{BASE_URL}/users/1", headers=HEADERS, timeout=10)
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            user = resp.json()
            assert "id" in user and "email" in user

    def test_invalid_auth_returns_401(self):
        resp = requests.get(
            f"{BASE_URL}/users",
            headers={"Authorization": "Bearer invalid-token"},
            timeout=10,
        )
        assert resp.status_code == 401
```

## 보안

### 토큰 처리
- 전체 토큰을 기록하지 마세요. 다음처럼 수정하세요: `Bearer <REDACTED>`.
- 스크립트에 토큰을 하드코딩하지 마세요. 환경 변수(`os.environ["API_TOKEN"]`) 또는 `${HERMES_HOME:-~/.hermes}/.env`에서 읽으세요.
- 로그, 오류 메시지 또는 git 이력에 토큰이 노출되면 즉시 교체하세요.

### 안전한 로깅

```python
def redact_auth(headers: dict) -> dict:
    sensitive = {"authorization", "x-api-key", "cookie", "set-cookie"}
    return {k: ("<REDACTED>" if k.lower() in sensitive else v) for k, v in headers.items()}
```

### 유출 체크리스트

- [ ] **URL의 인증 정보.** 쿼리 문자열의 API 키는 서버 로그, 브라우저 기록, 리퍼러 헤더에 남습니다 — 헤더를 사용하세요.
- [ ] **오류 응답의 PII.** `/users/123`에 대한 404가 사용자의 존재 여부를 노출해서는 안 됩니다(열거).
- [ ] **프로덕션의 스택 트레이스.** 500 오류가 파일 경로와 프레임워크 버전을 노출해서는 안 됩니다.
- [ ] **내부 호스트 이름/IP.** 오류 본문에 `10.x.x.x`, `internal-api.corp.local`이 포함되어서는 안 됩니다.
- [ ] **반환되는 토큰.** 일부 API는 오류 세부 정보에 인증 토큰을 포함합니다. 그렇지 않은지 확인하세요.
- [ ] **상세한 `Server`/`X-Powered-By`.** 스택 정보가 유출됩니다. 보안 검토 시 참고하세요.

## Hermes 도구 패턴

### terminal — curl, dig, openssl에 사용

```python
terminal('curl -sI https://api.example.com')
terminal('openssl s_client -connect api.example.com:443 -servername api.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates')
```

### execute_code — 여러 단계의 Python 흐름에 사용

디버깅이 인증 → 가져오기 → 페이지 매김 → 검증으로 이어질 때 `execute_code`를 사용하세요. 변수는 스크립트에서 유지되고, 결과는 stdout에 출력되며, 컨텍스트에 토큰이 넘쳐날 위험이 없습니다.

```python
execute_code('''
import os, requests

token = os.environ["API_TOKEN"]
base  = "https://api.example.com"
H     = {"Authorization": f"Bearer {token}"}

# 1. auth
me = requests.get(f"{base}/me", headers=H, timeout=10)
print(f"auth {me.status_code}")

# 2. paginate
all_users, cursor = [], None
while True:
    params = {"cursor": cursor} if cursor else {}
    r = requests.get(f"{base}/users", headers=H, params=params, timeout=10)
    body = r.json()
    all_users.extend(body["data"])
    cursor = body.get("next_cursor")
    if not cursor:
        break
print(f"users={len(all_users)}")
''')
```

### web_extract — 공급업체 API 문서에 사용

추측하는 대신 디버깅 중인 엔드포인트의 사양을 가져오세요.

```python
web_extract(urls=["https://docs.example.com/api/v1/users"])
```

### delegate_task — 전체 CRUD 테스트 점검에 사용

```python
delegate_task(
    goal="Test all CRUD endpoints for /api/v1/users",
    context="""
Follow the rest-graphql-debug skill (optional-skills/software-development/rest-graphql-debug).
Base URL: https://api.example.com
Auth: Bearer token from API_TOKEN env var.

For each verb (POST, GET, PATCH, DELETE):
  - happy path: assert status + response schema
  - error cases: 400, 404, 422
  - log a repro curl for any failure (redact tokens)

Output: pass/fail per endpoint + correlation IDs for failures.
""",
    toolsets=["terminal", "file"],
)
```

## 출력 형식

발견 사항을 보고할 때:

```
## Finding
Endpoint: POST /api/v1/users
Status:   422 Unprocessable Entity
Req ID:   req_abc123xyz

## Repro
curl -X POST https://api.example.com/api/v1/users \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <REDACTED>' \
  -d '{"name":"test"}'

## Root Cause
Missing required field `email`. Server validation rejects before processing.

## Fix
-d '{"name":"test","email":"test@example.com"}'
```

## 관련 항목

- `systematic-debugging` — 실패한 API 계층을 분리한 뒤 코드의 근본 원인을 분석할 때
- `test-driven-development` — 수정 사항을 배포하기 전에 회귀 테스트를 작성할 때
