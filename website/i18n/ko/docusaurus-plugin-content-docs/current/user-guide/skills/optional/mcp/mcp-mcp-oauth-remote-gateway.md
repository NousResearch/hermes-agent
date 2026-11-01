---
title: "Mcp Oauth Remote Gateway — 헤드리스 게이트웨이에서 원격 MCP 서버를 위한 수동 OAuth"
sidebar_label: "Mcp Oauth Remote Gateway"
description: "헤드리스 게이트웨이에서 원격 MCP 서버를 위한 수동 OAuth"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Mcp Oauth Remote Gateway

헤드리스 게이트웨이에서 원격 MCP 서버를 위한 수동 OAuth입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mcp/mcp-oauth-remote-gateway`로 설치 |
| 경로 | `optional-skills/mcp/mcp-oauth-remote-gateway` |
| 버전 | `1.0.0` |
| 작성자 | Ben Barclay (benbarclay), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `MCP`, `OAuth`, `PKCE`, `Remote-Deployment` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`mcporter`](/docs/user-guide/skills/optional/mcp/mcp-mcporter), [`fastmcp`](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보는 내용입니다.
:::

# 원격 Hermes 게이트웨이에서의 MCP OAuth

## 개요

Hermes에 내장된 MCP OAuth 클라이언트는 Hermes 프로세스 내부의
`127.0.0.1:<port>`에서 일회성 HTTP 리스너를 실행하고 해당 루프백 주소를 OAuth
`redirect_uri`로 등록합니다. 이는 사용자의 컴퓨터에서 실행되는 로컬 CLI에서는 완벽하게
작동합니다. 하지만 Hermes가 원격 게이트웨이(컨테이너, VPS,
메시징 봇)로 실행되면 완전히 작동하지 않습니다. 사용자의 브라우저는 `127.0.0.1`을 원격
컨테이너가 아니라 사용자 자신의 노트북으로 해석하기 때문에 인증 코드가 Hermes에 도달하지
않습니다.

이 스킬은 OAuth 절차를 수동으로 수행하고 결과 토큰을 Hermes의 토큰 저장소가 기대하는
정확한 파일에 기록합니다. 그러면 이후 `/reload-mcp`가 캐시된 토큰을 찾아 브라우저 흐름을
완전히 건너뜁니다.

## 사용 시점

다음 조건을 **모두** 충족할 때 이 스킬을 사용하세요.

1. 사용자가 OAuth를 요구하는 원격 HTTP MCP 서버를 추가하려 합니다(정적 Bearer 토큰이 아님).
2. Hermes가 **원격 게이트웨이**(컨테이너, VPS, Docker, 관리형 서비스)로 실행 중입니다 — 사용자의 노트북에서 실행되는 로컬 CLI가 **아닙니다**.
3. 서버가 PKCE를 사용하는 OAuth 2.1과 RFC 7591 동적 클라이언트 등록을 지원합니다(대부분의 최신 MCP 서버가 지원합니다 — Better Stack, Linear, Cloudflare, Datadog 등). DCR을 지원하지 않는 경우(GitHub가 대표적인 예)에는 이 스킬을 적용할 수 없습니다 — 사전 등록된 OAuth App 또는 Personal Access Token을 대신 사용하세요.

다음에는 사용하지 마세요.
- **로컬 CLI Hermes** — `mcp_servers.<name>`에 `auth: oauth`를 설정하고 `/reload-mcp`를 실행하면 됩니다. 내장 흐름이 브라우저를 열고 localhost에서 콜백을 캡처합니다. 완벽하게 작동합니다.
- **정적 Bearer 토큰(API 키)을 받는 서버** — 사용자가 원한다면 항상 `headers.Authorization: "Bearer <token>"`을 우선 사용하세요. 더 간단하고 갱신 절차가 없습니다.
- **GitHub Copilot MCP**(`api.githubcopilot.com/mcp/`) — GitHub는 DCR을 노출하지 않습니다. PAT 또는 사전 등록된 OAuth App을 사용하세요(함정 12 참조).

## 원격 게이트웨이에서 내장 OAuth 흐름이 실패하는 이유

Hermes의 기본 MCP OAuth 클라이언트(`tools/mcp_oauth.py`)는 다음과 같이 동작합니다.

1. 사용 가능한 로컬 포트 `P`를 선택합니다.
2. AS에 동적 OAuth 클라이언트를 등록하며 `redirect_uri = http://127.0.0.1:P/callback`을 보냅니다.
3. Hermes 프로세스 **내부**의 `127.0.0.1:P`에서 HTTP 서버를 시작합니다.
4. 인증 URL을 출력하고 로컬 엔드포인트에서 코드를 기다립니다.

Hermes가 원격으로 실행되면 `redirect_uri`의 `127.0.0.1`은 사용자가 아닌 원격
컨테이너의 루프백입니다. 인증이 끝나면 사용자의 브라우저는
`http://127.0.0.1:P/callback?code=...`으로 302 리디렉션되며, 이 주소는 사용자의
노트북으로 해석되어 연결에 실패합니다. 콜백은 Hermes 프로세스에 도달하지 않고 흐름은
시간 초과되며, `/reload-mcp`는 세부 정보 없이 "No MCP tools available"를 반환합니다.

확인할 증상: hermes 사용자 아래의 `[xdg-open] <defunct>` 프로세스, 비어 있거나 없는
토큰 디렉터리(`$HERMES_HOME/mcp-tokens/`), 그리고 `change_detail`에
"Added/Reconnected: X" 행이 전혀 없는 reload 응답입니다.

## 먼저 시도할 간단한 대안: 내장 흐름 자체의 우회 방법

수동으로 토큰을 조작하기 전에 내장 흐름의 대안이 배포 환경을 이미 지원하는지 확인하세요.
Hermes가 원격 세션을 감지하면 인증 URL과 함께 두 가지 옵션을 출력합니다(`tools/mcp_oauth.py`).

1. **붙여넣기 반환** — 대화형 TTY에서는 stdin 리더와 HTTP 리스너가 경쟁합니다. 사용자가 인증하고 브라우저가 `127.0.0.1:<port>` 연결에 실패하면, 주소 표시줄의 전체 URL(`?code=...&state=...`)을 프롬프트에 붙여 넣습니다. SSH로 접속한 CLI 세션에서 작동합니다.
2. **SSH 포트 포워딩** — `ssh -N -L <port>:127.0.0.1:<port> <user>@<host>`를 사용하면 리디렉션이 원격 리스너에 정상적으로 도달합니다.

두 방법 모두 Hermes 호스트에 대화형 터미널이 필요합니다. 이 스킬의 나머지 부분은
대화형 TTY가 **전혀 없는** 경우, 즉 Hermes가 순수한 메시징 게이트웨이/봇으로 실행되고
누구도 프롬프트에 응답하지 않는 경우를 위한 것입니다.

## 권장 진입점: Hermes 대시보드(수동 토큰 조작 전에 먼저 시도)

원격 Hermes 게이트웨이는 종종 **별도** 프로세스로 대시보드 웹 UI도 실행합니다(예:
`hermes dashboard --host 0.0.0.0 --port <port>`; `ps aux | grep 'hermes dashboard'`로
확인). 이 UI에는 커넥터/MCP 콘솔이 있으며 `/api/mcp/servers`, `/api/mcp/status`,
`/connectors` 같은 엔드포인트가 있습니다(모두 로그인이 필요하며, 쿠키 없이 401/302를
반환하는 curl 응답은 해당 엔드포인트가 존재한다는 뜻입니다).

**대시보드가 핵심 문제를 해결하는 이유:** 사용자가 자신의 브라우저에서 대시보드를 통해
OAuth를 진행하면 리디렉션이 대시보드가 캡처할 수 있는 컨텍스트에 도달합니다. 따라서
CLI/수동 흐름을 깨뜨리는 `127.0.0.1` 콜백 실패를 피할 수 있습니다. 그러므로
"원격 게이트웨이에서 OAuth MCP 서버 추가 또는 재인증"의 올바른 진행 순서는 다음과 같습니다.

1. **사용자의 브라우저에서 대시보드** — 의도된 진입점입니다. 서버 추가, OAuth 실행, reload를 모두 사용자로 인증된 상태에서 수행합니다. 콜백 복사-붙여넣기 절차도, 토큰 파일 수동 작성도 필요 없습니다.
2. **수동 토큰 조작(이 스킬의 나머지 부분)** — 대시보드에 브라우저 세션이 없을 때의 대안입니다(순수 채팅/헤드리스 컨텍스트).

**대시보드의 공개 URL 찾기.** 대시보드는 내부적으로 `0.0.0.0:<port>`에 바인딩하지만,
사용자에게는 외부에서 접근 가능한 URL이 필요합니다. 대부분의 배포 플랫폼은 이를 환경에
주입하므로 사용자가 직접 찾게 하지 말고 grep하세요.

```bash
env | grep -iE "HERMES_DASHBOARD_PUBLIC_URL|RAILWAY_PUBLIC_DOMAIN|RAILWAY_STATIC_URL|RAILWAY_SERVICE_.*_URL|PUBLIC_URL|BASE_URL|DOMAIN" \
  | sed -E 's/(TOKEN|SECRET|KEY|PASSWORD)=.*/\1=***REDACTED***/I'
```

`HERMES_DASHBOARD_PUBLIC_URL`이 있으면 이를 기준으로 삼습니다. Railway에서는
`RAILWAY_PUBLIC_DOMAIN` / `RAILWAY_STATIC_URL`(`*.up.railway.app` 호스트)과
`RAILWAY_SERVICE_*_URL` 변수도 확인하세요. 이 변수에는 더 이해하기 쉬운 사용자 지정
도메인이 들어 있는 경우가 있습니다. 사용자에게 전체 `https://` URL을 전달하고
Connectors/MCP 섹션으로 안내하세요. 이 환경 변수 grep은 `*_TOKEN`/`*_SECRET` 변수와
나란히 실행되므로 반드시 위의 `sed` 마스킹을 파이프하세요.

**대시보드로 해결되지 않는 것(여전히 호스트 측/셸 작업):** 셸 인증 상태가 필요한 stdio
서버(재시작 후에도 자격 증명이 유지되지 않을 수 있는 CLI `login` 명령)와
`$HERMES_HOME/.env`에서 자격 증명을 읽는 모든 항목입니다. 대시보드의 범위 밖입니다.

## 우회 방법

OAuth 절차를 수동으로 수행한 다음, Hermes의 `HermesTokenStorage`가 기록했을 정확한 파일에
결과 토큰을 기록하세요. 그러면 `/reload-mcp`에서 Hermes가 캐시된 토큰을 찾아 브라우저
흐름을 완전히 건너뜁니다.

아래 셸 명령은 게이트웨이 호스트의 `terminal` 도구로 실행하고, Python 단계(PKCE 생성,
토큰 교환, 파일 기록)는 `execute_code` 또는 `terminal`의 python3 호출로 실행하세요 —
파일 기록은 토큰 교환과 **동일한** 코드 블록에서 수행해야 합니다(함정 16 참조).

### 1. 원격 게이트웨이인지 확인

```bash
env | grep -iE "HERMES|RAILWAY|CONTAINER"
echo "$DISPLAY $WAYLAND_DISPLAY $SSH_CLIENT"
```

디스플레이가 없고 원격 표시가 있으면 원격 게이트웨이입니다. `tools/mcp_oauth.py::_can_open_browser()`는 동일한 환경 변수를 사용하므로, Hermes의 자동 감지가 "headless"라고 판단하면 내장 흐름은 작동하지 않습니다.

### 2. HERMES_HOME과 config 경로 찾기

```bash
HERMES_HOME=$(python3 -c 'from hermes_constants import get_hermes_home; print(get_hermes_home())')
echo "config: $HERMES_HOME/config.yaml"
echo "tokens: $HERMES_HOME/mcp-tokens/"
```

### 3. MCP 서버에서 OAuth 메타데이터 검색

MCP 서버는 RFC 9728(OAuth 2.0 Protected Resource Metadata)을 통해 OAuth 설정을
알립니다. 401 응답의 `WWW-Authenticate` 헤더에서 조회 위치를 알려 줍니다.

```bash
curl -sI https://mcp.example.com | grep -i www-authenticate
# → Bearer realm="mcp", resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"
```

**모든 서버가 `WWW-Authenticate`를 반환하지는 않습니다.** 일부 서버는 인증 검색
힌트가 없는 `{"errors":["Unauthorized"]}` 401을 반환합니다. 그럴 때는 well-known 경로를
직접 확인하세요.

```bash
for p in \
  /.well-known/oauth-protected-resource \
  /.well-known/oauth-authorization-server \
  /.well-known/openid-configuration ; do
  echo "=== $p ==="
  curl -s -A "python-httpx/0.27" "https://mcp.example.com$p" | head -c 400; echo
done
```

리소스 메타데이터를 가져와 `authorization_servers`를 확인한 다음, AS의
`/.well-known/oauth-authorization-server`를 가져와 `authorization_endpoint`,
`token_endpoint`, `registration_endpoint`를 확인하세요.

함정: 많은 서버가 Cloudflare 뒤에 있으며, 기본 `urllib` 사용자 에이전트에는 403을
반환합니다. 이 흐름의 요청에는 항상 `User-Agent: python-httpx/0.27`(또는 유사한 값)을
설정하세요.

### 4. 동적 클라이언트 등록(RFC 7591)

다음과 같이 `registration_endpoint`에 POST하세요.

```json
{
  "client_name": "Hermes Agent (manual OAuth)",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "<scopes_from_resource_metadata>"
}
```

AS의 `scopes_supported`가 비어 있으면 `scope`를 완전히 생략하세요 — 5단계의 함정을
참조하세요. 포트 `8765`(또는 아무 포트나)를 사용하세요 — 아무것도 수신 대기하지
않습니다. `token_endpoint_auth_method: none`은 공개 PKCE 클라이언트임을 의미합니다.
반환된 `client_id`를 저장하세요.

### 5. PKCE로 인증 URL 만들기

다음을 생성하세요.
- `code_verifier`: `secrets.token_urlsafe(64)[:128]`
- `code_challenge`: `base64url(sha256(code_verifier))`(패딩 없음)
- `state`: `secrets.token_urlsafe(24)`

쿼리 매개변수는 `response_type=code`, `client_id`, `redirect_uri`, `code_challenge`,
`code_challenge_method=S256`, `state`, 그리고 `resource=<mcp_server_url>`(RFC 8707 —
많은 서버가 특정 MCP 리소스에 토큰을 바인딩하기 위해 요구함)입니다. `scope`는 AS
메타데이터의 `scopes_supported`가 비어 있지 않은 배열이고 리소스 메타데이터가 특정
스코프를 선언한 경우에만 `<space-separated>` 형식으로 포함하세요. `scopes_supported: []`이면
`scope` 매개변수를 생략하세요 — 서버가 자체적으로 전체 기본 집합을 부여합니다. 빈
`scopes_supported`에 대해 스코프 문자열을 임의로 만들면 일부 AS에서 `invalid_scope`
오류가 발생할 수 있습니다.

**`code_verifier`와 `state`를 디스크에 임시 저장하세요**(예: `/tmp/.mcp-oauth-work/<server>.json`,
권한 0600). 7단계에서 필요하며 여러 채팅 턴에 걸쳐 사용될 수 있습니다.

### 6. 사용자에게 인증 URL 제공

```
Open this URL in your browser:
<authorize_url>

After approving, your browser will try to load http://127.0.0.1:8765/callback
and fail to connect — THAT'S EXPECTED. Just copy the entire URL from the
address bar (it will contain ?code=...&state=...) and paste it back here.
```

### 7. 코드를 토큰으로 교환

사용자가 콜백 URL을 붙여 넣으면 다음을 수행하세요.

1. 쿼리 문자열에서 `code`와 `state`를 파싱합니다.
2. **`state`가 임시 저장한 값과 일치하는지 확인합니다**(CSRF 검사 — 건너뛰지 마세요).
3. `token_endpoint`에 `application/x-www-form-urlencoded`로 POST합니다.
   - `grant_type=authorization_code`
   - `code=<from callback>`
   - `redirect_uri=<same as step 4>`
   - `client_id=<from step 4>`
   - `code_verifier=<stashed>`
   - `resource=<mcp_server_url>`(AS가 5단계에서 이를 요구했다면 여기도 포함)
4. 응답에는 `access_token`, `refresh_token`, `token_type`, `expires_in`, `scope`가 포함됩니다.

### 8. Hermes의 정확한 스키마로 토큰 기록

`tools/mcp_oauth.py::HermesTokenStorage`는 `$HERMES_HOME/mcp-tokens/` 아래에 두 파일을
요구합니다(디렉터리는 `0o700`, 파일은 `0o600`으로 생성).

**`<server_name>.json`** — `OAuthToken` pydantic 모델:
```json
{
  "access_token": "...",
  "token_type": "Bearer",
  "expires_in": 7200,
  "refresh_token": "...",
  "scope": "read write"
}
```

**`<server_name>.client.json`** — `OAuthClientInformationFull` 모델:
```json
{
  "client_id": "...",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "read write",
  "client_name": "..."
}
```

각 파일은 `json.dumps(..., indent=2)`로 기록하세요. 파일명은
`re.sub(r'[^\w\-]', '_', server_name)[:128]`로 정리하세요 — 이는 Hermes 토큰 저장소의
`_safe_filename()`과 일치합니다.

### 9. config.yaml에 서버 추가

```yaml
mcp_servers:
  <name>:
    url: "https://mcp.example.com"
    auth: oauth
    timeout: 180
    connect_timeout: 60
```

### 10. 사용자에게 reload를 요청하기 전에 토큰 스모크 테스트

토큰이 종단 간 작동하는지 확인하기 위해 MCP `initialize` 요청을 수동으로 POST하세요 —
이는 사용자가 또 다른 "No MCP tools available" reload 때문에 혼란을 겪기 전에 스코프
설정 오류, 잘못된 `resource` 값, CF 차단을 찾아냅니다.

```python
body = json.dumps({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "hermes-debug", "version": "1.0"},
    },
}).encode()
# POST to the MCP URL with:
#   Authorization: Bearer <access_token>
#   Accept: application/json, text/event-stream
#   Content-Type: application/json
#   MCP-Protocol-Version: 2025-06-18
#   User-Agent: python-httpx/0.27
```

`Content-Type: text/event-stream`인 HTTP 200과 `serverInfo` 및 `capabilities`를 포함한
JSON-RPC 결과를 기대하세요. **기본 UA를 사용하는 `urllib`은 사용하지 마세요** —
Cloudflare가 403을 반환하지만 Hermes(httpx 사용)는 성공합니다.
`scripts/diagnose-oauth-mcp.py`가 이 스모크 테스트를 자동화합니다.

### 11. 사용자에게 `/reload-mcp` 실행 안내

reload 시 Hermes는 `auth: oauth`를 확인하고 `HermesTokenStorage.get_tokens()`를 호출해
캐시된 토큰을 찾은 뒤 브라우저 흐름을 건너뛰고 `mcp_<name>_*` 도구를 등록합니다.
`expires_in`이 만료되기 전에 자동으로 갱신됩니다.

## 함정과 교훈

1. **"headless"라고 해서 OAuth가 불가능하다고 가정하지 마세요.** 내장 흐름은 로컬 CLI에서 잘 작동합니다. 문제는 사용자의 브라우저와 Hermes 프로세스가 서로 다른 컴퓨터에 있는 원격 배포에만 해당합니다. OAuth가 선택지가 아니라고 말하기 전에 실행 환경을 확인하세요.

2. **스킬 문서만 읽지 말고 소스를 읽으세요.** `tools/mcp_oauth.py`와 `website/docs/`의 MCP config reference가 권위 있는 참고 자료입니다. 기능이 "존재하지 않는다"고 말하기 전에 트리 전체를 grep하세요.

3. **Cloudflare UA 필터.** 많은 MCP/OAuth 제공업체가 인프라를 Cloudflare 앞에 두며, 공개 메타데이터 엔드포인트에서도 `python-urllib/*` 사용자 에이전트에 403을 반환합니다. 이 흐름의 모든 요청에 `User-Agent: python-httpx/0.27`(또는 브라우저와 유사한 문자열)을 설정하세요. Hermes 자체는 httpx를 사용하므로 실제 연결 경로에서는 문제가 되지 않습니다.

4. **authorize 요청과 token 요청 모두에 `resource`를 포함하세요.** RFC 8707 리소스 지시자는 대부분의 최신 MCP 서버에서 선택 사항이 아닙니다 — 발급된 토큰을 특정 MCP 리소스 URL에 바인딩합니다. 생략해도 작동하는 경우가 있지만, 이후 MCP 서버에서 스코프/대상 오류가 발생하는 토큰이 발급될 수 있습니다.

5. **후행 슬래시가 중요합니다.** 일부 서버는 리소스를 `https://mcp.example.com/`처럼 후행 슬래시와 함께 광고하며, 슬래시가 없는 변형에 대해 발급된 토큰을 거부합니다. `.well-known/oauth-protected-resource` 응답의 `resource` 값을 그대로 복사하세요.

6. **`/reload-mcp`는 실패 시 조용합니다.** reload에 `change_detail` 행 없이 "No MCP tools available"이 표시되면, 서버가 config에 있지만 연결에 실패했고 오류가 전파되지 않은 것입니다. 오류 로그를 tail하고 수동 `initialize` POST로 토큰을 직접 스모크 테스트하세요. 모든 것이 정상으로 보이면 전체 프로세스 재시작을 요청하세요.

7. **Circuit breaker는 `/reload-mcp` 후에도 유지될 수 있습니다.** `tools/mcp_tool.py`는 작은 임계값을 가진 모듈 수준 오류 카운트 딕셔너리를 유지합니다. 토큰 만료 후 여러 번 연속 실패하는 등 breaker가 작동하면, 도구 핸들러가 서버를 호출하기 전에 단락될 수 있으므로 성공적인 호출이 카운터를 초기화하지 못합니다. 증상은 "Reconnected: X"라고 표시되지만 같은 대화에서 이후 호출이 계속 "server unreachable"로 실패하는 것입니다. 복구 순서: 먼저 `/reload-mcp`를 시도하세요(저렴하고 채팅 프로세스 중단이 없음) — 현재 빌드에서는 카운터를 지울 수 있습니다. reload 후에도 실제 호출이 여전히 단락될 때만 전체 게이트웨이 프로세스 재시작으로 확대하세요. 처음부터 "반드시 재시작해야 한다"고 말하지 마세요.

8. **만료된 access_token과 작동한 breaker의 조합은 교착 상태입니다.** 자동 갱신 로직은 MCP 호출 경로 내부에서 실행되며, breaker가 작동하면 이 경로가 단락됩니다. 디스크의 토큰을 수동으로 갱신하는 것만으로는 도움이 되지 않습니다 — 수동 토큰 갱신은 `/reload-mcp`가 아니라 전체 재시작과 함께 수행하세요.

9. **수동 refresh에서 `invalid_grant`가 발생하면 refresh token이 죽은 것입니다 — 반복하지 말고 재인증만이 해결책입니다.** access_token이 충분히 오래 만료되면 refresh_token도 서버 측에서 폐기되거나 만료될 수 있습니다. 그러면 `grant_type=refresh_token` POST가 HTTP 400 `{"error":"invalid_grant",...}`를 반환합니다(문구는 "Grant not found", "Token expired", "refresh token is invalid" 등으로 다를 수 있음). 게이트웨이 측에서 복구할 방법은 **없습니다**. 사용자에게 두 가지 선택지를 제시하세요: (a) 전체 수동 OAuth 절차(3–10단계)를 다시 실행하거나, (b) 제공업체가 정적 개인 API 키를 제공한다면 그것으로 전환합니다 — 갱신/만료 주기가 없고 무인 원격 게이트웨이에 더 안정적입니다. 조기에 감지하려면 OAuth MCP에 대한 생성/업데이트 작업 전에 `expires_at`과 `time.time()`을 비교하세요. 이미 만료되었으면 먼저 refresh를 시도하고 `invalid_grant`를 즉시 표시하여 작업 중간에 실패하지 않게 하세요.

10. **성공적인 refresh 후에도 토큰이 거부되면 서버 측 SESSION 폐기입니다. 이를 해결하는 유일한 방법은 새 authorization_code 흐름입니다.** 함정 9와는 다릅니다. 저장된 토큰 파일이 정상으로 보이고(`expires_at`이 충분히 남아 있고 refresh_token도 있음) 라이브 `initialize` POST가 JSON-RPC 본문 `{"error":{"code":-32002,"message":"Session expired. Please re-authenticate."}}`과 함께 `401 invalid_token`을 반환할 수 있습니다. `grant_type=refresh_token` POST는 **성공할 수 있지만**(HTTP 200, 새 access_token) 새 토큰도 동일한 `-32002`를 반환합니다. 제공업체가 기본 MCP *세션*을 서버 측에서 폐기한 것입니다. OAuth refresh 체인은 자격 증명을 다시 발급할 뿐 폐기된 세션을 재설정하지 못합니다. OAuth MCP가 "not connected"를 보고할 때의 판단 규칙: (1) 수동 `initialize` POST로 저장된 access_token을 스모크 테스트합니다. (2) `401 invalid_token`이면 refresh를 시도하고 새 토큰을 스모크 테스트합니다. (3a) 새 토큰이 작동하면 기록하고 breaker를 해제하기 위해 재시작합니다. (3b) 새 토큰도 여전히 `-32002`/"Session expired"를 반환하면 중단합니다. 이는 세션 폐기이므로 사용자에게 전체 재인증을 위한 authorize URL을 전달합니다. `scripts/diagnose-oauth-mcp.py`가 1–2단계를 자동화하고 현재 분기를 출력합니다. 세션이 계속 폐기되는 무인 게이트웨이에는 정적 Personal API key를 우선 사용하세요. 매주 폐기하는 제공업체의 실제 사례는 `references/stripe-mcp-oauth-revocation.md`를 참조하세요.

11. **클라이언트 정보 파일은 선택 사항이 아닙니다.** Hermes는 refresh grant에 필요한 `client_id`를 알기 위해 `<server>.client.json`이 필요합니다. 이를 생략하면 첫 refresh가 실패하고 사용자는 재인증해야 합니다 — 두 파일을 모두 기록하는 것이 이 스킬의 핵심입니다.

12. **사용자가 열 redirect URL을 직접 입력하지 마세요.** `urllib.parse.urlencode()`로 인증 URL을 프로그래밍 방식으로 생성하세요. 스코프의 공백과 `state`의 특수 문자가 문자열을 이어 붙인 URL을 깨뜨립니다.

13. **보안: 임시 저장 파일에는 `code_verifier`가 들어 있습니다.** 토큰 교환이 성공하면 즉시 `/tmp/.mcp-oauth-work/<server>.json`을 삭제하세요. 사용된 신원 증명 비밀을 계속 보관할 이유가 없습니다.

14. **토큰 엔드포인트가 실제로 반환한 값을 기록하세요.** AS는 요청한 것보다 좁거나 넓은 스코프를 부여할 수 있습니다. 5단계에서 요청한 값이 아니라 토큰 교환 응답의 `scope`를 `<server>.json`에 기록하세요. `scopes_supported: []`인 경우 명시적으로 전송하는 스코프 목록이 양쪽에서 권위 있는 값입니다. 일부 서버는 최소 권한을 위해 지정한 좁은 스코프를 정확히 부여하고, 사용자가 모든 기능을 필요로 하면 전체 집합을 열거해야 합니다. 또 일부는 등록 시 허용된 스코프를 응답하지 않을 수 있습니다 — 토큰 교환 응답만 권위 있는 값입니다.

15. **OAuth 토큰은 제공업체의 공개 REST API에 대한 Bearer 토큰으로도 자주 사용할 수 있습니다.** `<server>.json`의 access_token은 흔히 "MCP 전용"이 아닙니다 — 해당 리소스 스코프가 부여되면 제공업체의 문서화된 REST API에 `Authorization: Bearer <token>`으로 요청할 수 있습니다. 이는 제공업체의 특이한 동작이 아니라 OAuth 2.0 사양입니다. MCP 서버가 읽기 전용이지만 쓰기 작업이 필요하다면 별도 API 키를 제안하기 전에 OAuth 토큰으로 제공업체 REST API를 직접 호출할 수 있는지 확인하세요.

16. **비밀 마스킹이 도구 출력에서 토큰을 가릴 수 있습니다.** 비밀 마스킹이 활성화되면 토큰과 긴 불투명 문자열이 도구 결과 출력에서 `***`으로 표시되므로, 여러 턴에 걸쳐 access_token을 보이게 하려고 `print(response)`를 사용할 수 없습니다. authorization_code grant의 일회성 `code` 값과 결합되면 토큰 교환 응답을 출력하다가 토큰을 잃고 코드를 소비할 수 있어 새 인증 URL로 다시 시작해야 합니다. **토큰 교환을 수행하는 동일한 코드 블록에서 access_token을 최종 대상 파일에 직접 기록하세요.** 디버깅을 위해 출력해야 한다면 `len(access_token)`, `token_type`, `scope`, `expires_in`만 출력하고 비밀 값은 절대 출력하지 마세요.

17. **GitHub MCP(`api.githubcopilot.com/mcp/`)는 DCR + PKCE-public이 아니라 사전 등록된 confidential OAuth App을 사용합니다.** 해당 클라이언트 정보에는 실제 `client_secret`과 `token_endpoint_auth_method: client_secret_post`가 들어 있습니다. `https://github.com/login/oauth/access_token`으로 보내는 토큰 교환 POST에는 `client_id`, `code`, `code_verifier`, `redirect_uri`와 함께 `client_secret`을 form 필드로 포함해야 합니다(PKCE는 secret과 함께 여전히 적용됨). redirect URI는 **OAuth App 설정에서 고정**되어 있어 변경할 수 없으므로 수동 리스너 포트 방식은 적용되지 않습니다. 사용자는 브라우저가 해당 포트 연결에 실패하도록 둔 다음 주소 표시줄 URL을 복사해 붙여 넣으면 됩니다.

## 하지 말아야 할 일

- **대안으로 `mcp-remote`를 사용하지 마세요.** 원격 컨테이너의 localhost에 OAuth 콜백 서버를 두는 npx 하위 프로세스를 실행하므로 동일한 문제가 발생합니다. `mcp-remote`는 MCP 클라이언트가 원격 HTTP를 전혀 지원하지 않을 때만 도움이 됩니다(Hermes는 기본적으로 지원함).
- **사용자가 OAuth를 명시적으로 요청했는데 "API 토큰을 붙여 넣으면 헤더를 추가하겠다"고 밀어붙이지 마세요.** 원격 배포에서 기본 OAuth 흐름이 실패하는 이유를 설명한 뒤에만 정적 토큰 대안을 제시하세요. 순환 없는 scope 제한 액세스를 위해 추가 작업을 선택할 사용자의 결정을 존중하세요.
- **소스를 읽지 않고 Hermes가 어떤 기능을 지원하지 않는다고 말하지 마세요.** 기능에 대해 말하기 전에 소스 트리를 grep하세요.

## 빠른 참조 파일

- `scripts/diagnose-oauth-mcp.py` — 재실행 가능하며 기본적으로 읽기 전용인 진단 도구입니다. 서버 이름을 받아 저장된 access_token을 스모크 테스트하고, refresh를 시도하고, 새 토큰을 스모크 테스트한 뒤 어느 복구 분기인지 정확히 출력합니다(`TOKEN_OK` = breaker/재시작, `REFRESH_FIXED` = 저장+재시작, `SESSION_REVOKED` = 전체 재인증, `REFRESH_DEAD` = 전체 재인증/API key). `--write`를 전달하면 작동하는 갱신 토큰을 원자적으로 저장합니다. 비밀 값은 절대 출력하지 않습니다. **OAuth MCP 서버가 "not connected"라고 보고할 때 가장 먼저 실행하세요** — 함정 7/9/10의 판단 트리를 구현합니다.
- `references/stripe-mcp-oauth-revocation.md` — OAuth 세션을 주기적으로 폐기하는 제공업체(Stripe)의 실제 사례와 지속 가능한 해결책인 정적 제한 API 키로의 전환을 설명하는 예시입니다.

## 관련 항목

- `native-mcp` — Hermes에서 MCP를 설정하는 일반 안내서입니다. 권위 있는 config reference는 여기에 있습니다.
- `mcporter` — Hermes 설정 외부에서 임시 MCP 호출을 위한 외부 CLI 브리지입니다.
