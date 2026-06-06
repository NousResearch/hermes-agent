---
title: "Rest Graphql Debug — REST/GraphQL API 디버깅: 상태 코드, 인증, 스키마, 재현"
sidebar_label: "Rest Graphql Debug"
description: "REST/GraphQL API 디버깅: 상태 코드, 인증, 스키마, 재현"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Rest Graphql Debug

REST/GraphQL API 버그 디버깅: 상태 코드, 인증, 스키마 불일치 및 오류 재현.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/software-development/rest-graphql-debug`로 설치 |
| 경로 | `optional-skills/software-development/rest-graphql-debug` |
| 버전 | `1.2.0` |
| 작성자 | eren-karakus0 |
| 라이선스 | MIT |
| 태그 | `api`, `rest`, `graphql`, `http`, `debugging`, `testing`, `curl`, `integration` |
| 관련 스킬 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# API 테스팅 및 디버깅

Hermes 도구들을 사용해 REST 및 GraphQL 진단을 주도하십시오 — `curl` 실행엔 `terminal`, Python `requests` 모듈엔 `execute_code`, 공급업체의 문서를 볼 땐 `web_extract`를 활용합니다. 짐작으로 문제를 고치려 들지 말고 실패한 계층을 먼저 정확히 격리하십시오.

## 언제 사용하나요

- API가 예상치 못한 상태 코드나 응답 본문을 반환할 때
- 인증이 실패할 때 (토큰 갱신 후 401/403, OAuth, API 키 문제 등)
- Postman에서는 정상 동작하지만 코드에서는 실패할 때
- Webhook / 콜백 통합 디버깅
- API 통합 테스트를 구축하거나 검토할 때
- 속도 제한(Rate limiting)이나 페이지네이션 문제가 있을 때

UI 렌더링, DB 쿼리 튜닝, 또는 DNS/방화벽 인프라 문제의 경우(담당자 에스컬레이션) 이 스킬을 건너뛰십시오.

## 핵심 원칙

**계층을 먼저 격리한 후 해결하십시오.** 200 OK 상태 코드 뒤에 깨진 데이터가 숨어있을 수 있습니다. 500 오류가 인증의 단일 문자 오타 때문에 발생한 것일 수도 있습니다. 체인을 순서대로 따라가며, 단계를 절대 건너뛰지 마십시오.

```
1. 연결성 (Connectivity)    → 호스트에 도달할 수 있는가?
1.5. 타임아웃 (Timeouts)    → 연결 지연인가 읽기 지연인가?
2. TLS/SSL                  → 인증서가 유효하고 신뢰할 수 있는가?
3. 인증 (Auth)              → 자격 증명이 올바르고 만료되지 않았는가?
4. 요청 형식 (Format)       → 페이로드 형태가 서버가 기대하는 것과 일치하는가?
5. 응답 파싱 (Parse)        → 돌아온 데이터를 우리 코드가 수용할 수 있는가?
6. 의미론 (Semantics)       → 데이터가 우리가 가정한 의미와 일치하는가?
```

## 5분 빠른 시작 (Quickstart)

### 터미널을 통한 REST

```python
# 상세한 요청/응답 교환 출력
terminal('curl -v https://api.example.com/users/1')

# JSON을 포함한 POST 요청
terminal("""curl -X POST https://api.example.com/users \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"name":"test","email":"test@example.com"}'""")

# 헤더만 조회
terminal('curl -sI https://api.example.com/health')

# JSON 깔끔하게 출력(Pretty-print)
terminal('curl -s https://api.example.com/users | python3 -m json.tool')
```

### 터미널을 통한 GraphQL

```python
terminal("""curl -X POST https://api.example.com/graphql \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"query":"{ user(id: 1) { name email } }"}'""")
```

**GraphQL 주의사항:** 서버는 쿼리가 실패하더라도 자주 HTTP 200을 반환합니다. 상태 코드와 무관하게 항상 `errors` 필드를 검사하십시오:

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

### execute_code를 통한 Python (requests)

```python
execute_code('''
import requests
resp = requests.get(
    "https://api.example.com/users/1",
    headers={"Authorization": "Bearer <TOKEN>"},
    timeout=(3.05, 30),  # (연결 타임아웃, 읽기 타임아웃)
)
print(resp.status_code, dict(resp.headers))
print(resp.text[:500])
''')
```

## 계층별 디버그 흐름 (Layered Debug Flow)

### 1단계 — 연결성 (Connectivity)

```python
terminal('nslookup api.example.com')
terminal('curl -v --connect-timeout 5 https://api.example.com/health')
```

실패 원인: DNS 분석 불가, 방화벽, VPN 필요, 프록시 누락.

### 1.5단계 — 타임아웃 (Timeouts)

*도달할 수 없음*과 *도달하지만 느림*을 구분하십시오:

```python
terminal('''curl -w "dns:%{time_namelookup}s connect:%{time_connect}s tls:%{time_appconnect}s ttfb:%{time_starttransfer}s total:%{time_total}s\\n" \\
  -o /dev/null -s https://api.example.com/endpoint''')
```

Python 환경에서는 항상 튜플 형태의 타임아웃을 전달하십시오 — `requests`에는 기본 타임아웃 값이 없어 무한정 대기할 수 있습니다:

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

진단: 높은 `time_connect`는 네트워크/방화벽 문제입니다. 낮은 `time_connect`와 높은 `time_starttransfer`는 서버가 느린 것입니다.

### 2단계 — TLS/SSL

```python
terminal('curl -vI https://api.example.com 2>&1 | grep -E "SSL|subject|expire|issuer"')
```

실패 원인: 만료된 인증서, 자체 서명(self-signed) 인증서, 호스트명 불일치, CA 번들 누락. `-k` 옵션은 즉흥적인 디버그 용도로만 사용하고, 코드에 포함해서는 안 됩니다.

### 3단계 — 인증 (Authentication)

```python
# 토큰 유효성 검사
terminal('curl -s -o /dev/null -w "%{http_code}\\n" -H "Authorization: Bearer $TOKEN" https://api.example.com/me')

# JWT exp(만료일) 클레임 디코딩 — base64url 패딩을 제대로 처리함
execute_code('''
import json, base64, os
tok = os.environ["TOKEN"]
payload = tok.split(".")[1]
payload += "=" * (-len(payload) % 4)
print(json.dumps(json.loads(base64.urlsafe_b64decode(payload)), indent=2))
''')
```

체크리스트:
- 토큰이 만료되었습니까? (JWT의 `exp` 클레임 확인)
- 스킴이 올바른가요? Bearer vs Basic vs Token vs `X-Api-Key`
- 올바른 환경에 사용 중인가요? (운영 환경에서 스테이징 키를 사용하는 것이 고전적인 실수입니다)
- API 키가 헤더에 위치해야 합니까 아니면 쿼리 매개변수(`?api_key=…`)에 위치해야 합니까?

### 4단계 — 요청 형식 (Request Format)

```python
terminal("""curl -v -X POST https://api.example.com/endpoint \\
  -H 'Content-Type: application/json' \\
  -d '{"key":"value"}' 2>&1""")
```

**Content-Type / 본문 불일치 — 조용한 415/400 오류:**

```python
# 틀림 — data= 는 form 인코딩 전송을 하지만 헤더는 json이라 속임
requests.post(url, data='{"k":"v"}', headers={"Content-Type": "application/json"})

# 맞음 — json= 은 헤더를 자동으로 설정하고 직렬화를 같이 수행함
requests.post(url, json={"k": "v"})

# 틀림 — Accept 에 XML을 요구해놓고 코드에선 .json()을 호출함
requests.get(url, headers={"Accept": "text/xml"})

# 맞음 — requests 가 boundary가 포함된 multipart를 스스로 생성하게 함
requests.post(url, files={"file": open("doc.pdf", "rb")})
```

흔한 문제: form 인코딩 vs JSON 헷갈림, 필수 필드 누락, 잘못된 HTTP 메서드, 인코딩되지 않은 쿼리 파라미터.

### 5단계 — 응답 파싱 (Response Parsing)

`.json()`을 호출하기 전에 항상 content-type을 검사하십시오:

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

실패 원인: JSON을 기대했지만 반환된 HTML 에러 페이지, 비어 있는 본문, 잘못된 문자셋.

### 6단계 — 의미론적 검증 (Semantic Validation)

깔끔하게 파싱되었다 하더라도 그 데이터가 정말 *올바른* 정보인가요?

- `"status": "active"`가 당신의 코드가 생각하는 의미와 같은가요?
- 응답의 ID가 요청한 ID와 일치합니까?
- 타임스탬프가 예상한 시간대에 맞춰져 있나요?
- 페이지네이션이 전체 결과를 반환했나요 아니면 페이지 1만 반환했나요?

## HTTP 상태 코드 플레이북

### 401 Unauthorized — 자격 증명이 누락되었거나 올바르지 않음

1. `Authorization` 헤더가 실제로 존재하는지 확인 (`curl -v` 로 확인)
2. 토큰이 올바르고 만료되지 않았는지 확인
3. 올바른 인증 스킴인지 확인 (`Bearer` vs `Basic` vs `Token`)
4. 특정 API들은 헤더 대신 쿼리 파라미터(`?api_key=…`)를 요구하기도 합니다.

### 403 Forbidden — 인증은 되었지만 권한이 없음

1. 토큰이 필요한 스코프나 권한을 가지고 있는지 확인
2. 리소스가 다른 계정 소유인지 확인
3. IP 허용 목록(allowlist)이 차단하고 있는지 확인
4. 브라우저에서 발생하는 경우 CORS 문제인지 확인 (`Access-Control-Allow-Origin` 확인)

### 404 Not Found — 리소스가 존재하지 않거나 URL이 잘못됨

1. 경로가 맞는지 확인 (후행 슬래시, 오타, 버전 접두사 등)
2. 리소스 ID가 실제로 존재하는지 확인
3. 올바른 API 버전인지 확인 (`/v1/` vs `/v2/`)
4. 올바른 기본 URL인지 확인 (스테이징 vs 운영 환경)

### 409 Conflict — 상태 충돌

1. 리소스가 이미 존재하는지 (중복 생성 시도)
2. `ETag` / `If-Match` 가 오래된 상태인지
3. 다른 프로세스에 의한 동시 수정이 발생했는지 확인

### 422 Unprocessable Entity — JSON은 유효하나 데이터가 부적절함

에러 본문에 잘못된 필드 이름이 명시되는 경우가 많습니다. 다음을 확인하세요:
- 필드 타입 (문자열 vs 정수, 날짜 포맷)
- 필수(Required) vs 선택(Optional) 여부
- 열거형(Enum) 값이 허용된 집합 안에 있는지 여부

### 429 Too Many Requests — 속도 제한(Rate limit) 초과

`Retry-After` 와 `X-RateLimit-*` 헤더를 확인하십시오. 지수 백오프(Exponential backoff)를 적용하세요:

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

### 5xx — 서버 측 에러, 대개 사용자의 잘못이 아님

- **500** — 서버 버그입니다. 코릴레이션(Correlation) ID를 캡처하여 제공자 측에 신고하십시오.
- **502** — 업스트림 서비스가 다운되었습니다. 백오프 적용 후 재시도하십시오.
- **503** — 과부하 / 점검 중입니다. 상태(Status) 페이지를 확인하십시오.
- **504** — 업스트림 타임아웃입니다. 페이로드 크기를 줄이거나 타임아웃 설정값을 높이십시오.

모든 5xx 에러에 대해: 지터를 포함한 백오프(backoff with jitter)를 수행하고 지속될 경우 경고(Alert)를 띄웁니다.

## 페이지네이션 & 멱등성 (Pagination & Idempotency)

**페이지네이션.** *모든* 결과를 받아오고 있는지 확인하십시오. `next_cursor`, `next_page`, `total_count` 같은 속성을 찾으십시오. 대표적인 두 패턴:
- Offset 방식 (`?limit=100&offset=200`) — 단순하지만 데이터 변동 시 아이템을 건너뛸 수 있습니다.
- Cursor 방식 (`?cursor=abc123`) — 라이브 데이터나 대규모 데이터셋에 적합합니다.

**멱등성.** 멱등성이 보장되지 않는 작업(POST 등)의 경우 재시도 시 이중 결제나 이중 생성을 방지하기 위해 `Idempotency-Key: <uuid>`를 전송하십시오. 결제 및 주문 시스템에서는 필수입니다.

## 계약 검증 (Contract Validation)

프로덕션 환경에 타격을 주기 전에 스키마 변동(drift)을 잡아내십시오:

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

API 업그레이드 후, 새 타사(third-party) 서비스를 연동할 때, 혹은 CI 스모크 테스트 과정 중에 실행하십시오.

## 코릴레이션(상관) ID (Correlation IDs)

제공자의 Request ID를 항상 캡처해두십시오. 이는 공급업체 고객지원에 대응하는 가장 빠른 길입니다:

```python
execute_code('''
import requests
resp = requests.post(url, json=payload, headers=headers, timeout=10)
request_id = (
    resp.headers.get("X-Request-Id")
    or resp.headers.get("X-Trace-Id")
    or resp.headers.get("CF-Ray")  # Cloudflare 환경
)
if resp.status_code >= 400:
    print(f"failed status={resp.status_code} req_id={request_id} ts={resp.headers.get('Date')}")
''')
```

**공급업체 버그 보고 템플릿:**

```
Endpoint:    POST /api/v1/orders
Request ID:  req_abc123xyz
Timestamp:   2026-03-17T14:30:00Z
Status:      500
Expected:    201 with order object
Actual:      500 {"error":"internal server error"}
Repro:       curl -X POST … (auth: <REDACTED>)
```

## 회귀 테스트 템플릿 (Regression Test Template)

아래 내용을 `tests/` 에 두고 `terminal('pytest tests/test_api_smoke.py -v')` 로 실행하십시오:

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

## 보안 (Security)

### 토큰 관리
- 전체 토큰을 로깅하지 마십시오. `Bearer <REDACTED>` 처럼 검열하십시오.
- 스크립트에 토큰을 하드코딩하지 마십시오. 환경변수(`os.environ["API_TOKEN"]`) 나 `~/.hermes/.env` 에서 읽어오십시오.
- 만약 로그, 에러 메시지, 깃 히스토리 등에 토큰이 표면으로 드러난 경우 즉시 폐기/갱신(rotate)하십시오.

### 안전한 로깅

```python
def redact_auth(headers: dict) -> dict:
    sensitive = {"authorization", "x-api-key", "cookie", "set-cookie"}
    return {k: ("<REDACTED>" if k.lower() in sensitive else v) for k, v in headers.items()}
```

### 정보 유출 점검 체크리스트

- [ ] **URL에 포함된 자격증명.** 쿼리 문자열에 담긴 API 키는 서버 로그, 브라우저 기록, 리퍼러(referrer) 헤더에 모두 남습니다 — 대신 헤더를 사용하십시오.
- [ ] **에러 응답에 포함된 개인 정보(PII).** `/users/123 에 대한 404` 에러가 사용자의 존재 여부를 짐작하게 해서는 안 됩니다 (열거 공격 위험).
- [ ] **운영 환경의 스택 트레이스.** 500 에러에서 서버 내부의 파일 경로, 프레임워크 버전 등이 유출되어서는 안 됩니다.
- [ ] **내부 호스트명/IP 주소.** 에러 본문에 `10.x.x.x` 나 `internal-api.corp.local` 이 드러나선 안 됩니다.
- [ ] **에코 백(Echo back)된 토큰.** 일부 API들은 에러 상세 정보에 인증 토큰을 포함시켜 반환하곤 합니다. 이런 현상이 없는지 확인하십시오.
- [ ] **장황한 `Server` / `X-Powered-By` 헤더.** 기술 스택 정보가 유출됩니다. 보안 리뷰 시 이 점을 노트하십시오.

## Hermes 도구 패턴

### terminal — curl, dig, openssl 용

```python
terminal('curl -sI https://api.example.com')
terminal('openssl s_client -connect api.example.com:443 -servername api.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates')
```

### execute_code — 다단계 Python 흐름 용

디버깅이 인증 → 가져오기 → 페이지네이션 → 검증 과정에 걸쳐 이루어질 때 `execute_code`를 활용하십시오. 스크립트 실행 동안 변수 상태가 유지되고, 결과는 표준 출력(stdout)으로 찍히며, 사용자의 컨텍스트가 토큰들로 오염될 위험이 없습니다:

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

### web_extract — 공급업체 API 문서 참조용

짐작하려 들지 말고, 당신이 지금 디버깅하고 있는 엔드포인트에 해당하는 명세(spec)를 직접 추출하십시오:

```python
web_extract(urls=["https://docs.example.com/api/v1/users"])
```

### delegate_task — 전체 CRUD 테스트 점검

```python
delegate_task(
    goal="/api/v1/users 에 대한 모든 CRUD 엔드포인트를 테스트하십시오",
    context="""
rest-graphql-debug 스킬(optional-skills/software-development/rest-graphql-debug)을 따르십시오.
기본 URL: https://api.example.com
인증: API_TOKEN 환경변수로부터의 Bearer 토큰.

각 HTTP 메서드 (POST, GET, PATCH, DELETE)에 대해:
  - 해피 패스(happy path): 상태 코드 및 응답 스키마 검증(assert)
  - 에러 케이스: 400, 404, 422
  - 실패 시 재현을 위한 curl 명령어 로그 기록 (토큰은 레드액트할 것)

출력: 엔드포인트별 성공/실패 여부 + 실패한 경우의 상관(Correlation) ID.
""",
    toolsets=["terminal", "file"],
)
```

## 출력 형식 (Output Format)

발견한 사항을 보고할 때의 형식:

```
## 발견 사항 (Finding)
엔드포인트: POST /api/v1/users
상태:   422 Unprocessable Entity
Req ID:   req_abc123xyz

## 재현 (Repro)
curl -X POST https://api.example.com/api/v1/users \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <REDACTED>' \
  -d '{"name":"test"}'

## 근본 원인 (Root Cause)
필수 필드인 `email`이 누락되었습니다. 처리되기 전 서버의 검증 과정에서 거부되었습니다.

## 수정 (Fix)
-d '{"name":"test","email":"test@example.com"}'
```

## 연관 스킬

- `systematic-debugging` — 실패하는 API 계층을 격리하고 나면, 당신 코드의 근본 원인을 찾으십시오
- `test-driven-development` — 수정본을 릴리스하기 전에 회귀 테스트를 먼저 작성하십시오
