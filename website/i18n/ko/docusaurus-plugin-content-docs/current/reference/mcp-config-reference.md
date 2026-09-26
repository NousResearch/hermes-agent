---
sidebar_position: 8
title: "MCP 구성 참조"
description: "Hermes Agent MCP 구성 키, 필터링 의미 체계 및 유틸리티 도구 정책에 대한 참조"
---

# MCP 구성 참조

이 페이지는 기본 MCP 문서를 보완하는 간결한 참조 자료입니다.

개념에 대한 안내는 다음을 참조하세요:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Hermes에서 MCP 사용하기](/guides/use-mcp-with-hermes)

## 루트 구성 형태

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio servers
    args: []
    env: {}

    # OR
    url: "..."          # HTTP servers
    headers: {}

    # Optional HTTP/SSE TLS settings:
    ssl_verify: true                # bool or path to a CA bundle (PEM)
    client_cert: "/path/to/cert.pem"  # mTLS client certificate (see below)
    # client_key: "/path/to/key.pem"  # optional, when key lives in a separate file

    enabled: true
    timeout: 120
    connect_timeout: 60
    supports_parallel_tool_calls: false
    tools:
      include: []
      exclude: []
      resources: true
      prompts: true
```

## 서버 키

| 키 | 유형 | 적용 대상 | 의미 |
|---|---|---|---|
| `command` | string | stdio | 실행할 실행 파일 |
| `args` | list | stdio | 하위 프로세스에 전달할 인수 |
| `env` | mapping | stdio | 하위 프로세스에 전달할 환경 |
| `url` | string | HTTP | 원격 MCP 엔드포인트 |
| `headers` | mapping | HTTP | 원격 서버 요청에 사용할 헤더 |
| `ssl_verify` | bool or string | HTTP | TLS 검증. `true`(기본값)는 시스템 CA를 사용하고, `false`는 검증을 비활성화하며(안전하지 않음), 문자열은 사용자 지정 CA 번들(PEM)의 경로입니다 |
| `client_cert` | string or list | HTTP | mTLS 클라이언트 인증서. 문자열 = 인증서와 키가 포함된 PEM 파일 경로. 목록 `[cert, key]` = 별도 파일. 목록 `[cert, key, password]` = 암호화된 키 |
| `client_key` | string | HTTP | `client_cert`가 문자열이고 키가 별도 파일에 있을 때 클라이언트 개인 키의 경로 |
| `enabled` | bool | both | `false`인 경우 서버 전체를 건너뜁니다 |
| `timeout` | number | both | 도구 호출 제한 시간(초)(기본값: `300`) |
| `connect_timeout` | number | both | 초기 연결 제한 시간(초)(기본값: `60`) |
| `supports_parallel_tool_calls` | bool | both | 이 서버의 도구를 동시에 실행하도록 허용 |
| `skip_preflight` | bool | HTTP | HEAD/GET 응답의 콘텐츠 유형이 MCP가 아닌, 유효한 Streamable HTTP 엔드포인트에 대한 빠른 실패 콘텐츠 유형 프로브를 우회합니다(기본값: `false`) |
| `transport` | string | HTTP | Streamable HTTP 대신 SSE 전송을 사용하려면 `sse`로 설정 |
| `keepalive_interval` | number | both | 활성 상태 핑 주기(초)(기본값: `180`, 최솟값 5초). 유휴 세션을 빠르게 정리하는 서버에서는 서버의 세션 TTL보다 작은 값으로 설정하세요 |
| `idle_timeout_seconds` | number | stdio | 유휴 시간 후 stdio 서버를 선택적으로 재활용(`0`은 비활성화). `lifecycle:` 매핑 아래에 둘 수도 있습니다 |
| `max_lifetime_seconds` | number | stdio | 수명 초과 후 stdio 서버를 선택적으로 재활용(`0`은 비활성화). `lifecycle:` 매핑 아래에 둘 수도 있습니다 |
| `tools` | mapping | both | 필터링 및 유틸리티 도구 정책 |
| `auth` | string | HTTP | 인증 방식. OAuth 2.1과 PKCE를 활성화하려면 `oauth`로 설정 |
| `sampling` | mapping | both | 서버가 시작하는 LLM 요청 정책(MCP 안내 참조) |
| `elicitation` | mapping | both | 서버가 시작하는 사용자 입력 요청. `enabled`(기본값 `true`) 및 초 단위 `timeout`(기본값 `300`). 양식 모드 요청은 승인 화면으로 전달되고 URL 모드는 거부됩니다(MCP 안내 참조) |
| `trust` | string | both | 신뢰 수준: `full`(기본값) 또는 `untrusted`. `untrusted` 서버에서는 쓰기 가능한 모든 도구 호출(`readOnlyHint: true` 주석이 없는 모든 도구)을 실행하기 전에 표준 승인 화면을 통한 사용자 승인이 필요합니다. `readOnlyHint`는 서버가 제공하는 *힌트*입니다. 거짓말하는 서버는 읽기 전용이라고 주장한 도구의 승인을 건너뛸 수 있을 뿐이며 추가 액세스 권한을 얻을 수는 없습니다. 따라서 완전히 제어하지 않는 서버는 모두 `untrusted`로 표시하세요. 인식할 수 없는 값은 `untrusted`로 처리됩니다(페일 클로즈) |

## 환경 변수 참조

서버 항목의 어느 위치에서든(`env`, `headers`, `args`, `url` 등) 문자열 값은 `${VAR}` 또는 Cursor 스타일 SecretRef 형식인 `${env:VAR}`로 환경 변수를 참조할 수 있습니다. 둘 다 동일한 변수로 확인되므로 Cursor/Claude 구성에서 복사한 MCP 스니펫이 수정 없이 작동합니다.

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${env:GITHUB_TOKEN}"   # same as "${GITHUB_TOKEN}"
```

값은 활성 프로필의 비밀 범위에서 확인됩니다(프로세스 환경으로 대체). 따라서 비밀을 `~/.hermes/.env`에 넣으세요. 설정되지 않은 변수는 리터럴 플레이스홀더를 그대로 유지합니다.

### 컨텍스트 변수

환경 변수 외에도 Cursor 스타일의 컨텍스트 변수도 보간됩니다(이름은 대소문자를 구분합니다).

| 변수 | 확인되는 값 |
|---|---|
| `${userHome}` | 현재 사용자의 홈 디렉터리 |
| `${workspaceFolder}` | 세션 작업 공간 루트(알려진 경우 세션 터미널 cwd, 그렇지 않으면 프로세스 cwd) |
| `${workspaceFolderBasename}` | `${workspaceFolder}`의 기본 이름 |
| `${pathSeparator}` / `${/}` | OS 경로 구분자(`os.sep`) |

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    env:
      CACHE_DIR: "${userHome}${/}.cache${/}mcp"
```

그 밖의 `${...}` 참조는 위의 환경 변수 조회로 전달됩니다.

## `tools` 정책 키

| 키 | 유형 | 의미 |
|---|---|---|
| `include` | string or list | 서버 네이티브 MCP 도구의 허용 목록. 항목은 정확한 이름 또는 fnmatch 스타일 글로브(`*_radar_*`, `get_zones_*`)일 수 있습니다 |
| `exclude` | string or list | 서버 네이티브 MCP 도구의 차단 목록. `include`와 동일한 정확한 이름/글로브 의미 체계 사용 |
| `resources` | bool-like | `list_resources` + `read_resource` 활성화/비활성화 |
| `prompts` | bool-like | `list_prompts` + `get_prompt` 활성화/비활성화 |

## 필터링 의미 체계

### `include`

`include`가 설정되면 해당 서버 네이티브 MCP 도구만 등록됩니다.

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude`

`exclude`가 설정되고 `include`가 설정되지 않은 경우, 해당 이름을 제외한 모든 서버 네이티브 MCP 도구가 등록됩니다.

```yaml
tools:
  exclude: [delete_customer]
```

### 우선순위

둘 다 설정된 경우 `include`가 우선합니다.

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

결과:
- `create_issue`는 계속 허용됩니다
- `delete_issue`는 `include`가 우선하므로 무시됩니다

## 유틸리티 도구 정책

Hermes는 MCP 서버별로 다음 유틸리티 래퍼를 등록할 수 있습니다.

리소스:
- `list_resources`
- `read_resource`

프롬프트:
- `list_prompts`
- `get_prompt`

### 리소스 비활성화

```yaml
tools:
  resources: false
```

### 프롬프트 비활성화

```yaml
tools:
  prompts: false
```

### 기능 인식 등록

`resources: true` 또는 `prompts: true`인 경우에도 MCP 세션이 실제로 해당 기능을 노출할 때만 Hermes가 해당 유틸리티 도구를 등록합니다.

따라서 다음은 정상입니다:
- 프롬프트를 활성화함
- 하지만 프롬프트 유틸리티가 표시되지 않음
- 서버가 프롬프트를 지원하지 않기 때문

## `enabled: false`

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

동작:
- 연결 시도 없음
- 검색 없음
- 도구 등록 없음
- 나중에 재사용할 수 있도록 구성이 그대로 유지됨

## 빈 결과 동작

필터링으로 모든 서버 네이티브 도구가 제거되고 유틸리티 도구도 등록되지 않으면 Hermes는 해당 서버를 위한 빈 MCP 런타임 도구 세트를 생성하지 않습니다.

## 구성 예시

### 안전한 GitHub 허용 목록

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      resources: false
      prompts: false
```

### Stripe 차단 목록

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### 리소스 전용 문서 서버

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

### TLS 클라이언트 인증서(mTLS)

클라이언트 인증서가 필요한 HTTP/SSE 서버의 경우 `client_cert`(선택적으로 `client_key`)를 설정하세요.

```yaml
mcp_servers:
  # Combined cert + key in a single PEM file
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # Separate cert and key files
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # Encrypted key with a passphrase (3-element list form)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # Custom CA bundle (private CA / self-signed server)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

참고:
- 경로는 `~` 확장을 지원합니다. 파일이 없으면 연결 시 서버 범위 오류 메시지와 함께 즉시 실패합니다.
- `ssl_verify: false`는 서버 인증서 검증을 완전히 비활성화합니다. 실제 서비스에서는 사용하지 마세요.
- Streamable HTTP 및 SSE 전송 모두에서 작동합니다.

## 구성 다시 로드

MCP 구성을 변경한 후 다음 명령으로 서버를 다시 로드합니다.

```text
/reload-mcp
```

## 도구 이름 지정

서버 네이티브 MCP 도구는 다음과 같은 이름이 됩니다.

```text
mcp__<server>__<tool>
```

예시:
- `mcp__github__create_issue`
- `mcp__filesystem__read_file`
- `mcp__my_api__query_data`

유틸리티 도구도 동일한 접두사 패턴을 따릅니다.
- `mcp__<server>__list_resources`
- `mcp__<server>__read_resource`
- `mcp__<server>__list_prompts`
- `mcp__<server>__get_prompt`

이중 밑줄 구분자(`mcp__…__…`)는 Claude Code, Codex 및 OpenCode에서 사용하는 규칙과 일치하며, 어느 구성 요소에 밑줄이 포함되어 있더라도 서버와 도구의 경계를 명확히 구분합니다.

### 이름 정규화

서버 이름과 도구 이름에서 문자, 숫자 또는 밑줄이 아닌 모든 문자(하이픈, 점, 공백 등)는 등록 전에 밑줄로 대체됩니다. 이를 통해 도구 이름이 LLM 함수 호출 API에서 유효한 식별자가 됩니다.

예를 들어 `my-api`라는 서버가 `list-items.v2`라는 도구를 노출하면 다음과 같이 됩니다.

```text
mcp__my_api__list_items_v2
```

`include`/`exclude` 필터를 작성할 때 이 점을 기억하세요. 정규화된 버전이 아니라 **원래의** MCP 도구 이름(하이픈/점 포함)을 사용해야 합니다.

## OAuth 2.1 인증

OAuth가 필요한 HTTP 서버의 경우 서버 항목에 `auth: oauth`를 설정합니다.

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

동작:
- Hermes는 MCP SDK의 OAuth 2.1 PKCE 흐름(메타데이터 검색, 동적 클라이언트 등록, 토큰 교환 및 갱신)을 사용합니다
- 처음 연결할 때 인증을 위한 브라우저 창이 열립니다
- 토큰은 `~/.hermes/mcp-tokens/<server>.json`에 저장되며 세션 간 재사용됩니다
- 토큰 갱신은 자동으로 수행되며, 갱신에 실패할 때만 다시 인증합니다
- HTTP/StreamableHTTP 전송(`url` 기반 서버)에만 적용됩니다
