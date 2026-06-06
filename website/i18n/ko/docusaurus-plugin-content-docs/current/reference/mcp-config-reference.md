---
sidebar_position: 8
title: "MCP 구성 레퍼런스"
description: "Hermes 에이전트 MCP 구성 키, 필터링 의미 체계 및 유틸리티 도구 정책에 대한 레퍼런스"
---

# MCP 구성 레퍼런스 (MCP Config Reference)

이 페이지는 기본 MCP 문서의 핵심 요약 레퍼런스입니다.

개념적인 안내는 다음을 참조하세요:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Hermes와 함께 MCP 사용하기](/guides/use-mcp-with-hermes)

## 루트 구성 형태 (Root config shape)

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio 서버
    args: []
    env: {}

    # OR
    url: "..."          # HTTP 서버
    headers: {}

    # 선택적 HTTP/SSE TLS 설정:
    ssl_verify: true                # boolean 또는 CA 번들 (PEM) 경로
    client_cert: "/path/to/cert.pem"  # mTLS 클라이언트 인증서 (아래 참조)
    # client_key: "/path/to/key.pem"  # 선택 사항, 키가 별도 파일에 있는 경우

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

## 서버 키 (Server keys)

| 키 | 타입 | 적용 대상 | 의미 |
|---|---|---|---|
| `command` | string | stdio | 실행할 실행 파일 |
| `args` | list | stdio | 하위 프로세스에 대한 인수 |
| `env` | mapping | stdio | 하위 프로세스에 전달할 환경 변수 |
| `url` | string | HTTP | 원격 MCP 엔드포인트 |
| `headers` | mapping | HTTP | 원격 서버 요청을 위한 헤더 |
| `ssl_verify` | bool 또는 string | HTTP | TLS 유효성 검사. `true` (기본값)는 시스템 CA를 사용하고, `false`는 유효성 검사를 비활성화(보안에 취약)하며, 문자열의 경우 사용자 정의 CA 번들(PEM) 경로를 의미 |
| `client_cert` | string 또는 list | HTTP | mTLS 클라이언트 인증서. 문자열 = 인증서 + 키가 포함된 PEM 파일 경로. 목록 `[cert, key]` = 별도의 파일들. 목록 `[cert, key, password]` = 암호화된 키 |
| `client_key` | string | HTTP | `client_cert`가 문자열이고 키가 별도 파일에 있는 경우, 클라이언트 개인 키의 경로 |
| `enabled` | bool | 둘 다 | false인 경우 서버를 완전히 건너뜀 |
| `timeout` | number | 둘 다 | 도구 호출 시간 제한(timeout) |
| `connect_timeout` | number | 둘 다 | 초기 연결 시간 제한 |
| `supports_parallel_tool_calls` | bool | 둘 다 | 이 서버의 도구를 동시에 실행할 수 있도록 허용 |
| `tools` | mapping | 둘 다 | 필터링 및 유틸리티 도구 정책 |
| `auth` | string | HTTP | 인증 방법. PKCE와 함께 OAuth 2.1을 활성화하려면 `oauth`로 설정 |
| `sampling` | mapping | 둘 다 | 서버가 시작하는 LLM 요청 정책 (MCP 가이드 참조) |

## `tools` 정책 키

| 키 | 타입 | 의미 |
|---|---|---|
| `include` | string 또는 list | 서버 기본 MCP 도구 화이트리스트 |
| `exclude` | string 또는 list | 서버 기본 MCP 도구 블랙리스트 |
| `resources` | bool-like | `list_resources` + `read_resource` 활성화/비활성화 |
| `prompts` | bool-like | `list_prompts` + `get_prompt` 활성화/비활성화 |

## 필터링 의미론 (Filtering semantics)

### `include`

`include`가 설정된 경우 해당 서버 기본 MCP 도구들만 등록됩니다.

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude`

`exclude`가 설정되고 `include`가 설정되지 않은 경우, 해당 이름을 제외한 모든 서버 기본 MCP 도구가 등록됩니다.

```yaml
tools:
  exclude: [delete_customer]
```

### 우선순위 (Precedence)

두 개 모두 설정된 경우 `include`가 우선 적용됩니다.

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

결과:
- `create_issue`는 여전히 허용됩니다.
- `include`가 우선 적용되므로 `delete_issue`는 무시됩니다.

## 유틸리티 도구 정책 (Utility-tool policy)

Hermes는 각 MCP 서버에 대해 다음의 유틸리티 래퍼(wrappers)를 등록할 수 있습니다:

리소스 (Resources):
- `list_resources`
- `read_resource`

프롬프트 (Prompts):
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

### 기능을 인식하는 등록 (Capability-aware registration)

`resources: true` 또는 `prompts: true`인 경우에도 MCP 세션이 실제로 해당 기능을 노출하는 경우에만 Hermes가 해당 유틸리티 도구를 등록합니다.

따라서 다음의 상황은 정상입니다:
- 프롬프트를 활성화함
- 그러나 프롬프트 유틸리티가 표시되지 않음
- 이유는 서버가 프롬프트를 지원하지 않기 때문임

## `enabled: false`

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

동작:
- 연결 시도를 하지 않음
- 검색을 하지 않음
- 도구 등록을 하지 않음
- 나중에 재사용하기 위해 구성은 제자리에 유지됨

## 빈 결과 동작 (Empty result behavior)

필터링을 통해 모든 서버 기본 도구가 제거되고 유틸리티 도구도 등록되지 않은 경우, Hermes는 해당 서버에 대한 빈 MCP 런타임 도구 세트를 생성하지 않습니다.

## 구성 예시 (Example configs)

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

### Stripe 블랙리스트

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

### TLS 클라이언트 인증서 (mTLS)

클라이언트 인증서가 필요한 HTTP/SSE 서버의 경우 `client_cert` (및 선택적으로 `client_key`)를 설정하세요:

```yaml
mcp_servers:
  # 단일 PEM 파일에 결합된 인증서 + 키
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # 별도의 인증서 및 키 파일
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # 암호가 있는 암호화된 키 (3개 요소 리스트 형식)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # 사용자 지정 CA 번들 (프라이빗 CA / 자체 서명 서버)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

참고:
- 경로는 `~` 확장을 지원합니다. 파일이 누락된 경우 연결 시점에 서버 범위의 오류 메시지와 함께 즉시 실패합니다.
- `ssl_verify: false`는 서버 인증서 유효성 검사를 완전히 비활성화합니다. 실제 서비스에서는 이 설정을 사용하지 마세요.
- 스트리밍 가능한 HTTP 및 SSE 전송 방식 모두에서 작동합니다.

## 구성 다시 로드 (Reloading config)

MCP 구성을 변경한 후 다음 명령으로 서버를 다시 로드하세요:

```text
/reload-mcp
```

## 도구 명명 규칙 (Tool naming)

서버 기본 MCP 도구는 다음과 같이 지정됩니다:

```text
mcp_<server>_<tool>
```

예시:
- `mcp_github_create_issue`
- `mcp_filesystem_read_file`
- `mcp_my_api_query_data`

유틸리티 도구도 동일한 접두사 패턴을 따릅니다:
- `mcp_<server>_list_resources`
- `mcp_<server>_read_resource`
- `mcp_<server>_list_prompts`
- `mcp_<server>_get_prompt`

### 이름 정리 (Name sanitization)

서버 이름과 도구 이름에 있는 하이픈(`-`)과 마침표(`.`)는 등록하기 전에 밑줄(`_`)로 교체됩니다. 이렇게 하면 도구 이름이 LLM 함수 호출 API에 유효한 식별자가 됩니다.

예를 들어, `list-items.v2`라는 도구를 노출하는 `my-api`라는 서버는 다음과 같이 됩니다:

```text
mcp_my_api_list_items_v2
```

`include` / `exclude` 필터를 작성할 때 이 점을 유의하세요 — 정리된 버전이 아닌, 하이픈/마침표가 있는 **원본** MCP 도구 이름을 사용해야 합니다.

## OAuth 2.1 인증

OAuth가 필요한 HTTP 서버의 경우 서버 항목에 `auth: oauth`를 설정합니다:

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

동작:
- Hermes는 MCP SDK의 OAuth 2.1 PKCE 흐름(메타데이터 검색, 동적 클라이언트 등록, 토큰 교환 및 새로 고침)을 사용합니다.
- 처음 연결 시 인증을 위해 브라우저 창이 열립니다.
- 토큰은 `~/.hermes/mcp-tokens/<server>.json`에 유지되며 세션 전반에 걸쳐 재사용됩니다.
- 토큰 새로 고침은 자동입니다; 새로 고침이 실패할 때만 재인증이 발생합니다.
- HTTP/StreamableHTTP 전송 방식(`url` 기반 서버)에만 적용됩니다.
