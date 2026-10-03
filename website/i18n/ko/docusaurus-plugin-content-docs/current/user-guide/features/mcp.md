---
sidebar_position: 4
title: "MCP (Model Context Protocol)"
description: "MCP를 통해 Hermes Agent를 외부 도구 서버에 연결하고, Hermes가 로드하는 MCP 도구를 정확히 제어하세요"
---

# MCP (Model Context Protocol)

MCP를 사용하면 Hermes Agent를 외부 도구 서버에 연결하여 에이전트가 Hermes 외부에 있는 도구(GitHub, 데이터베이스, 파일 시스템, 브라우저 스택, 내부 API 등)를 사용할 수 있습니다.

어딘가에 이미 존재하는 도구를 Hermes에서 사용하고 싶었던 적이 있다면, MCP가 대개 가장 깔끔한 방법입니다.

:::tip Claude Code에서 오셨나요?
`~/.claude.json`의 `mcpServers` 블록은 Hermes의 `config.yaml`에서 `mcp_servers`에 대응하며, `hermes import-agent claude-code`가 스킬과 지침을 포함해 자동으로 마이그레이션합니다. [다른 에이전트에서 가져오기](../import-from-other-agents.md)를 참고하세요.
:::

## MCP로 얻을 수 있는 것

- 먼저 네이티브 Hermes 도구를 작성하지 않고도 외부 도구 생태계에 접근
- 동일한 설정에서 로컬 stdio 서버와 원격 HTTP MCP 서버 사용
- 시작 시 자동 도구 검색 및 등록
- 서버가 지원하는 경우 MCP 리소스와 프롬프트를 위한 유틸리티 래퍼
- 서버별 필터링으로 Hermes가 실제로 보게 할 MCP 도구만 노출

## 빠른 시작

1. MCP 지원은 표준 설치에 포함되어 있으므로 추가 단계가 필요하지 않습니다.

2. `~/.hermes/config.yaml`에 MCP 서버를 추가합니다.

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
```

3. Hermes를 시작합니다.

```bash
hermes chat
```

4. Hermes에 MCP 기반 기능을 사용하도록 요청합니다.

예:

```text
List the files in /home/user/projects and summarize the repo structure.
```

Hermes는 MCP 서버의 도구를 검색하고 다른 도구와 동일한 방식으로 사용합니다.

## 카탈로그: Nous 승인 MCP 원클릭 설치

Hermes에는 Nous 직원이 검토하고 병합한 엄선된 MCP 서버 카탈로그가 포함되어 있습니다. 기본적으로 비활성화되어 있으므로 실제로 원하는 것만 설치하세요.

```bash
hermes mcp                # interactive picker (default)
hermes mcp catalog        # plain-text list, scriptable
hermes mcp install n8n    # install a catalog entry by name
```

선택기에는 각 항목의 현재 상태가 표시됩니다.

```
n8n          available              Manage and inspect n8n workflows from Hermes
linear       enabled                Linear issue/project management (remote OAuth)
github       installed (disabled)   GitHub repo + PR tools
```

행에서 `Enter`를 누르면 설치(필요한 인증 정보 입력 포함), 활성화, 비활성화 또는 제거할 수 있습니다. 카탈로그 항목은 hermes-agent 저장소의 `optional-mcps/` 아래에 저장되며, 해당 디렉터리에 존재한다는 것은 Nous의 승인을 받았음을 의미합니다. 커뮤니티 제출 단계는 없으며 PR을 병합하여 항목을 추가합니다.

카탈로그 항목에는 다음이 필요할 수 있습니다.

- **API 키** — Hermes가 설치 시 입력을 요청하고 값을 `~/.hermes/.env`에 기록합니다. 기본 URL 같은 비밀이 아닌 값도 같은 파일에 저장됩니다.
- **OAuth**(원격 MCP) — 설정에 `auth: oauth`로 기록되며, MCP 클라이언트가 최초 연결 시 브라우저를 엽니다.
- **OAuth**(Google/GitHub 같은 서드파티 제공자) — 아직 인증하지 않았다면 Hermes가 `hermes auth <provider>`를 안내합니다.

### 설치 시 도구 선택

인증 정보를 설정한 후 Hermes는 MCP 서버를 탐색하여 서버가 노출하는 모든 도구를 나열하고 체크리스트를 표시합니다.

```
Select tools for 'linear' (SPACE toggle, ENTER confirm)
  [x] find_issues       Find issues matching a query
  [x] get_issue         Get a single issue
  [x] create_issue      Create a new issue
  [ ] delete_workspace  Delete a Linear workspace
  ...
```

미리 선택된 행은 다음에서 가져옵니다.

1. **이전 선택** — 이 항목을 전에 설치했다면(재설치 시 매니페스트의 기본값이 덮어쓰지 않고 기존 선택을 보존함)
2. **매니페스트의 `tools.default_enabled`** — 항목에 해당 선언이 있는 경우(일부 카탈로그 항목은 변경 작업 도구나 자주 사용하지 않는 도구를 미리 제외함)
3. **모든 도구** — 앞의 두 조건이 모두 적용되지 않는 경우

`ENTER`로 체크리스트를 제출합니다. 체크한 도구만 `mcp_servers.<name>.tools.include`에 들어갑니다. 모두 선택하면 필터가 기록되지 않으며(가장 깔끔한 설정 형태이고 동작은 동일함),

**탐색에 실패한 경우**(서버에 연결할 수 없음, OAuth가 아직 완료되지 않음, 백업 서비스가 실행 중이 아님)에도 설치는 성공합니다. 선언되어 있으면 매니페스트의 `tools.default_enabled`가 직접 적용되고, 그렇지 않으면 필터가 기록되지 않습니다. 서버에 연결할 수 있게 된 후 `hermes mcp configure <name>`을 다시 실행하여 조정하세요.

### 신뢰 모델

카탈로그 항목을 설치하면 매니페스트가 지정한 모든 작업(예: `git clone`, 항목의 `bootstrap` 명령(`pip install`, `npm install` 등), 그리고 궁극적으로 MCP 서버 자체의 코드 실행)이 수행됩니다. 매니페스트는 hermes-agent 저장소에 PR 검토를 거쳐 등록되므로 Nous가 각 항목을 출시 전에 검토했지만, **설치하기 전에 매니페스트를 직접 읽어야 합니다.** 특히 `source:` 필드의 저장소, `install.bootstrap:` 명령, `transport.command:` 호출을 확인하세요.

매니페스트는 GitHub의 [`optional-mcps/<name>/manifest.yaml`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)에 있습니다. 선택기는 설치 시 매니페스트의 `source:` URL도 출력하므로 업스트림 저장소를 빠르게 확인할 수 있습니다. 웹 대시보드의 MCP 페이지에도 카탈로그 항목별 동일한 세부 정보가 표시됩니다. 전송 방식, 인증 유형, 엔드포인트 URL(HTTP) 또는 명령과 인자(stdio), git 설치 소스/참조 및 부트스트랩 명령, 설정 참고 사항을 확인할 수 있으며, `source:`는 클릭 가능한 링크로 표시됩니다. 따라서 Install을 클릭하기 전에 항목이 정확히 무엇에 연결되거나 무엇을 실행하는지 검사할 수 있습니다.

### 매니페스트 버전 호환성

매니페스트는 `manifest_version`을 고정합니다. 카탈로그는 이전 버전과 호환됩니다. PR에서 설치된 Hermes가 이해하는 것보다 새로운 `manifest_version`의 항목을 추가하면 선택기가 해당 항목을 조용히 숨기는 대신 경고(`⚠ '<name>' requires a newer Hermes`)를 표시합니다. 이 경고가 보이면 `hermes update`를 실행하여 최신 Hermes를 설치하세요.

### 런타임 `${ENV_VAR}` 치환

항목의 `transport.command`, `transport.args`, `transport.url` 및 `headers` 안의 `${VAR}` 자리표시자는 서버 연결 시 환경 변수에서 해석됩니다(`~/.hermes/.env`의 모든 값 포함). 이는 카탈로그 항목이 사용자가 다른 곳에서 설정한 값을 참조하려 할 때 유용합니다. 예를 들면 `${HOME}/foo` 또는 `${MY_PROVIDER_TOKEN}`입니다.

Cursor 스타일 컨텍스트 변수도 치환됩니다(대소문자를 구분함): `${userHome}`(홈 디렉터리), `${workspaceFolder}`(세션 작업 공간 루트), `${workspaceFolderBasename}`, `${pathSeparator}` / `${/}`(OS 경로 구분자)입니다. 자세한 내용은 [MCP 설정 참조](/docs/reference/mcp-config-reference)를 참고하세요.

이는 카탈로그 매니페스트의 `${INSTALL_DIR}`과는 다릅니다. `${INSTALL_DIR}`은 설치 시 카탈로그가 항목의 저장소를 복제한 경로로 치환됩니다.

### 나중에 도구 선택 업데이트

```bash
hermes mcp configure linear
```

현재 선택을 미리 체크한 동일한 체크리스트를 다시 엽니다. 더 많은 도구를 활성화하고 싶거나 서버에 새 도구가 추가되어 사용하고 싶을 때 사용하세요.

### 카탈로그 매니페스트 업데이트

MCP는 자동으로 업데이트되지 않습니다. Hermes 업데이트 후 매니페스트 버전이 변경되었다면 `hermes mcp install <name>`을 다시 실행하여 새로 고치세요.

카탈로그에 MCP를 추가하려면 [`optional-mcps/`](https://github.com/NousResearch/hermes-agent/tree/main/optional-mcps)에 대한 PR을 여세요.

### 제안 메타데이터(`suggest:`)

매니페스트에는 `keywords:` 및/또는 `hosts:` 목록이 포함된 선택적 `suggest:` 블록을 선언할 수 있습니다. UI 표면(현재는 Desktop 앱의 작성기)은 초안에 키워드 중 하나가 완전한 단어로 언급되거나 붙여넣은 링크의 호스트 이름이 호스트 접미사 중 하나로 끝날 때 원클릭 "Add &lt;server&gt;" 필을 제공합니다. 이는 단순한 참고 정보입니다. 설치는 동일한 검증된 카탈로그/설정 경로를 거치며, 대부분의 호스팅 원격 항목(Atlassian, Sentry, Notion, Stripe, Vercel, Supabase 등)이 이를 선언합니다.

GitHub는 의도적으로 카탈로그에 포함되지 않습니다. 호스팅 MCP는 각 클라이언트가 자체 OAuth 앱을 제공해야 하고(일반 동적 클라이언트 등록은 거부됨), Hermes에 번들된 `github/*` 스킬이 `gh` CLI를 구동하는 더 강력한 통합이기 때문입니다. Desktop에서는 `gh`에 아직 로그인하지 않은 경우 GitHub 언급에 대신 `github-auth` 스킬이 제안됩니다.

## MCP 서버의 두 가지 유형

### Stdio 서버

Stdio 서버는 로컬 하위 프로세스로 실행되며 stdin/stdout을 통해 통신합니다.

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
```

다음과 같은 경우 stdio 서버를 사용하세요.

- 서버가 로컬에 설치되어 있음
- 로컬 리소스에 낮은 지연 시간으로 접근하고 싶음
- `command`, `args`, `env`를 보여주는 MCP 서버 문서를 따르고 있음

### HTTP 서버

HTTP MCP 서버는 Hermes가 직접 연결하는 원격 엔드포인트입니다.

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ***"
```

다음과 같은 경우 HTTP 서버를 사용하세요.

- MCP 서버가 다른 곳에서 호스팅됨
- 조직이 내부 MCP 엔드포인트를 제공함
- 해당 통합을 위해 Hermes가 로컬 하위 프로세스를 생성하는 것을 원하지 않음

### OAuth 인증 HTTP 서버

호스팅 MCP 서버 대부분(Linear, Sentry, Atlassian, Asana, Figma, Stripe 등)은 정적 bearer 토큰 대신 OAuth 2.1을 요구합니다. `auth: oauth`를 설정하면 Hermes가 MCP Python SDK를 통해 검색, 동적 클라이언트 등록, PKCE, 토큰 교환, 갱신 및 추가 인증을 처리합니다.

:::tip Figma 원격 MCP
Figma의 호스팅 엔드포인트(`https://mcp.figma.com/mcp`)는 정확한 `client_name`으로 동적 클라이언트 등록을 허용 목록에 등록합니다. 일반적인 `"Hermes Agent"`는 403이 반환되지만 `"Claude Code"`와 `"Codex"`는 성공합니다. Hermes는 `mcp.figma.com`에 `oauth.client_name: "Claude Code"`를 자동 설정하므로 별도 방법 없이 설치/로그인이 작동합니다.

```yaml
mcp_servers:
  figma:
    url: "https://mcp.figma.com/mcp"
    auth: oauth
```

또는 `hermes mcp install figma`를 실행한 다음 `hermes mcp login figma`를 실행하세요.
:::

```yaml
mcp_servers:
  linear:
    url: "https://mcp.linear.app/mcp"
    auth: oauth
```

처음 연결할 때 Hermes는 인증 URL을 출력하고 가능한 경우 브라우저를 연 다음 로컬 루프백 포트에서 OAuth 콜백을 기다립니다. 토큰은 0o600 권한으로 `~/.hermes/mcp-tokens/<server>.json`에 캐시되며, 이후 실행에서는 갱신에 실패할 때까지 조용히 재사용합니다.

**원격/헤드리스 호스트.** Hermes가 브라우저와 다른 컴퓨터에서 실행되면 루프백 콜백이 노트북에 도달할 수 없습니다. 흐름을 완료하는 방법은 다음과 같습니다.

- **붙여넣기(설정 불필요):** 대화형 터미널에서 Hermes는 인증 URL과 함께 "Or paste the redirect URL here…"를 출력합니다. 브라우저에서 URL을 열고 승인한 다음, 브라우저가 최종적으로 이동한 전체 URL을 복사하여 프롬프트에 붙여넣으세요(리디렉션에 연결 오류가 표시되는 것은 정상입니다). `?code=…&state=…` 형식의 쿼리 문자열만 입력해도 됩니다.
- **SSH 포트 포워딩:** 별도 터미널에서 `ssh -N -L <port>:127.0.0.1:<port> user@host`를 실행한 다음 리디렉션 흐름을 정상적으로 진행합니다.
- **프록시 콜백(`redirect_uri`):** 공개 HTTPS 엔드포인트가 호스트로 전달되는 경우(예: 콜백 포트를 가리키는 Tailscale Funnel 또는 리버스 프록시), `oauth.redirect_uri`를 설정하면 브라우저 리디렉션이 터널이나 붙여넣기 없이 Hermes에 도달합니다.

```yaml
mcp_servers:
  myserver:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      redirect_port: 8765                                # fixed port for the proxy to target
      redirect_uri: "https://oauth.example.ts.net/callback"
```

완전한 헤드리스 게이트웨이(메시징 봇, 대화형 터미널 없음)의 경우 선택 사항인 [`mcp-oauth-remote-gateway` 스킬](../skills/optional/mcp/mcp-mcp-oauth-remote-gateway.md)이 흐름을 수동으로 완료하고 Hermes가 예상하는 위치에 토큰을 기록하는 과정을 안내합니다.

**주의 — WAF가 `127.0.0.1` 리디렉션 URI를 거부함.** 일부 제공자는 쿼리 문자열에 리터럴 `127.0.0.1`이 포함된 인증 요청을 403으로 거부하는 WAF를 인증 서버 앞에 둡니다(Reclaim.ai의 AWS API Gateway가 알려진 예이며, 모든 시도가 OAuth 앱에 도달하기 전에 `{"message":"Forbidden"}`을 반환합니다). `oauth.redirect_host: localhost`를 설정하면 대신 `http://localhost:<port>/callback`을 사용합니다. 어느 경우든 콜백 리스너는 `127.0.0.1`에 바인딩됩니다.

[OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md#mcp-servers)에서 DCR을 사용하지 않는 서버(예: Slack), 사전 등록된 `client_id`/`client_secret`, 범위 사용자 지정, `hermes mcp login <server>`를 통한 재인증을 포함한 전체 안내를 확인하세요.

**주의 — 자동 등록을 지원하지 않는 제공자(Google Drive, Atlassian).** 일부 서버는 일반적인 `auth: oauth`가 의존하는 동적 클라이언트 등록 단계(RFC 7591)를 거부합니다. Google의 공식 Drive 서버(`https://drivemcp.googleapis.com/mcp/v1`)는 `400 Bad Request`를 반환하므로 OAuth 클라이언트가 생성되지 않고 토큰도 발급되지 않습니다. 증상이 미묘합니다. 이러한 서버는 인증 없이도 `tools/list`를 제공하므로 `hermes mcp login`이 도구를 나열하여 성공한 것처럼 보이지만, 이후 실제 도구 호출은 모두 시간 초과됩니다. `hermes mcp login`은 이제 토큰이 실제로 디스크에 기록되었는지 확인하여 이를 감지하고 자체 OAuth 클라이언트를 제공하라고 안내합니다. 제공자의 콘솔에서 하나를 만든 뒤 설정에 추가하세요.

```yaml
mcp_servers:
  googledrive:
    url: "https://drivemcp.googleapis.com/mcp/v1"
    auth: oauth
    oauth:
      client_id: "<your-oauth-client-id>"
      client_secret: "<your-oauth-client-secret>"
```

그런 다음 `hermes mcp login googledrive`를 실행하세요. 사전 등록된 클라이언트를 사용하면 Hermes가 등록을 건너뛰고 일반 브라우저 인증 흐름을 실행합니다.

**주의 — 설정 자동 재로드 경합.** 실행 중인 Hermes 세션 내부에서 `~/.hermes/config.yaml`을 편집하면 CLI가 30초 제한으로 MCP 연결을 자동 재로드합니다. 대화형 OAuth 흐름에는 충분하지 않습니다. 항목을 추가한 다음 새 터미널에서 `hermes mcp login <server>`를 실행하세요. 인증을 완료할 때까지 최대 5분을 기다립니다.

## mTLS / 클라이언트 인증서

상호 TLS(클라이언트 인증서 인증)가 필요한 원격 HTTP MCP 서버는 `client_cert` / `client_key`를 통해 지원됩니다. Hermes는 TLS 핸드셰이크를 위해 해석된 인증서를 기반 HTTP 클라이언트에 전달합니다.

`client_cert`는 다음 세 가지 형태를 지원합니다.

- **하나로 합친 PEM 경로** — 인증서와 개인 키가 모두 들어 있는 하나의 파일:

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/.certs/mcp-client.pem"
```

- **`[cert, key]` 2-튜플** — 인증서와 키가 별도 파일에 있는 경우(`client_cert` + `client_key`를 설정하는 것과 동일함):

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key"]
```

- **`[cert, key, password]` 3-튜플** — 개인 키가 암호화된 경우 세 번째 요소가 키 암호입니다.

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: ["~/.certs/mcp-client.crt", "~/.certs/mcp-client.key", "${MCP_KEY_PASSWORD}"]
```

`client_cert`(합친 PEM)와 명시적인 `client_key`를 사용하여 인증서와 키를 완전히 분리할 수도 있습니다. 경로는 `~` 확장을 지원하며, 파일이 없으면 불분명한 TLS 핸드셰이크 실패 대신 서버 범위가 명확한 오류가 발생합니다.

## 사용자별 ID 헤더

호출자 ID에 따라 동작을 결정하는 원격 HTTP/SSE MCP 서버(사용자별 속도 제한, 감사 추적, 멀티테넌트 라우팅 등)에는 `identity_header`를 통해 모든 요청에 ID 헤더를 보낼 수 있습니다.

```yaml
mcp_servers:
  team_api:
    url: "https://mcp.team.example.com/mcp"
    identity_header:
      name: "X-User-Id"
      value_from: "static"   # "static" (default) or "profile"
      value: "alice"         # required for static
```

- `value_from: static`은 config.yaml의 리터럴 `value`를 전송합니다.
- `value_from: profile`은 활성 Hermes 프로필 이름을 연결 시 한 번 해석하여 전송합니다. 한 컴퓨터의 여러 프로필이 같은 서버에 연결되고 이를 구분해야 할 때 유용합니다.

서버의 `headers` 매핑에 같은 이름(대소문자 무관)의 명시적 항목이 있으면 항상 그 항목이 우선합니다. ID 헤더가 직접 설정한 헤더를 덮어쓰는 일은 없습니다. 잘못된 `identity_header` 블록은 경고 후 무시되며 서버 연결을 차단하지 않습니다. stdio 서버에서는 헤더가 없으므로 해당 키를 무시하고 경고합니다.

## 기본 설정 참조

Hermes는 `mcp_servers` 아래의 `~/.hermes/config.yaml`에서 MCP 설정을 읽습니다.

### 공통 키

| 키 | 유형 | 의미 |
|---|---|---|
| `command` | 문자열 | stdio MCP 서버의 실행 파일 |
| `args` | 목록 | stdio 서버의 인자 |
| `env` | 매핑 | stdio 서버에 전달할 환경 변수 |
| `url` | 문자열 | HTTP MCP 엔드포인트 |
| `headers` | 매핑 | 원격 서버용 HTTP 헤더 |
| `client_cert` | 문자열 \| 목록 | mTLS용 클라이언트 인증서 — 합친 PEM 경로 또는 `[cert, key]` / `[cert, key, password]` |
| `client_key` | 문자열 | 클라이언트 개인 키 PEM 경로(`client_cert`와 분리된 경우) |
| `identity_header` | 매핑 | HTTP/SSE 서버용 선택적 사용자별 ID 헤더 — `{name, value_from: static\|profile, value}` |
| `timeout` | 숫자 | 도구 호출 제한 시간 |
| `connect_timeout` | 숫자 | 초기 연결 제한 시간(MCP `initialize` 핸드셰이크에도 적용) |
| `idle_timeout_seconds` | 숫자 | 도구 호출 없이 이 시간이 지나면 stdio 서버 재활용(`0` = 사용 안 함, 기본값). 다음 도구 호출 시 서버가 투명하게 재시작됩니다. |
| `max_lifetime_seconds` | 숫자 | 총 수명이 이 시간에 도달하면 stdio 서버 재활용(`0` = 사용 안 함, 기본값). 다음 사용 시 투명하게 재시작됩니다. |
| `enabled` | 불리언 | `false`이면 Hermes가 서버 전체를 건너뜀 |
| `supports_parallel_tool_calls` | 불리언 | `true`이면 이 서버의 도구를 동시에 실행할 수 있음 |
| `tools` | 매핑 | 서버별 도구 필터링 및 유틸리티 정책 |

### 최소 stdio 예시

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### 메모리를 많이 사용하는 stdio 서버 재활용

브라우저 기반 MCP 서버(예: `@playwright/mcp`)는 첫 도구 호출 후 전체 Chromium을 메모리에 유지하며 수백 MB를 계속 점유합니다. 자동 재활용을 선택하면 유휴/수명 제한 이후 서버가 종료되고 다음 도구 호출 시 투명하게 재시작됩니다(도구는 전체 기간 등록된 상태로 유지됨).

```yaml
mcp_servers:
  playwright:
    command: "npx"
    args: ["-y", "@playwright/mcp@latest", "--headless"]
    idle_timeout_seconds: 900     # recycle after 15 min without a tool call
    max_lifetime_seconds: 86400   # and at least once a day regardless
```

### 최소 HTTP 예시

```yaml
mcp_servers:
  company_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
```

## 기본 제공 프리셋

잘 알려진 MCP 서버의 경우 `hermes mcp add`가 `--preset` 플래그를 받아 전송 세부 정보를 채우므로 명령과 인자를 직접 찾을 필요가 없습니다. 프리셋은 기본값만 제공합니다. 같은 명령줄에서 전달한 다른 항목(환경 변수, 헤더, 필터링 등)이 있으면 해당 항목이 우선합니다.

| 프리셋 | 연결하는 대상 |
|---|---|
| `codex` | Codex CLI의 MCP 서버(stdio를 통한 `codex mcp-server`). PATH에 `codex` CLI가 필요합니다. |

```bash
# Add Codex CLI as an MCP server in one line
hermes mcp add codex --preset codex
```

다음과 같은 설정이 기록됩니다.

```yaml
mcp_servers:
  codex:
    command: "codex"
    args: ["mcp-server"]
```

아무 로컬 이름이나 선택할 수 있습니다(`hermes mcp add my-codex --preset codex`도 가능함). 프리셋은 `command`/`args` 기본값만 제공합니다.

## Hermes의 MCP 도구 등록 방식

Hermes는 기본 제공 이름과 충돌하지 않도록 MCP 도구에 접두사를 붙입니다.

```text
mcp_<server_name>_<tool_name>
```

예:

| 서버 | MCP 도구 | 등록 이름 |
|---|---|---|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

실제로는 보통 접두사가 붙은 이름을 직접 호출할 필요가 없습니다. Hermes가 도구를 확인하고 일반적인 추론 과정에서 선택합니다.

## MCP 유틸리티 도구

지원되는 경우 Hermes는 MCP 리소스와 프롬프트를 중심으로 유틸리티 도구도 등록합니다.

- `list_resources`
- `read_resource`
- `list_prompts`
- `get_prompt`

이 도구들은 동일한 접두사 패턴으로 서버별 등록됩니다. 예:

- `mcp_github_list_resources`
- `mcp_github_get_prompt`

### 중요

이 유틸리티 도구는 이제 기능을 인식합니다.
- Hermes는 MCP 세션이 실제로 리소스 작업을 지원할 때만 리소스 유틸리티를 등록합니다.
- Hermes는 MCP 세션이 실제로 프롬프트 작업을 지원할 때만 프롬프트 유틸리티를 등록합니다.

따라서 호출 가능한 도구는 노출하지만 리소스/프롬프트는 제공하지 않는 서버에는 이러한 추가 래퍼가 생성되지 않습니다.

## 서버별 필터링

각 MCP 서버가 Hermes에 제공하는 도구를 제어하여 도구 네임스페이스를 세밀하게 관리할 수 있습니다.

### 서버 전체 비활성화

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

`enabled: false`이면 Hermes는 서버 전체를 건너뛰며 연결을 시도하지도 않습니다.

### 서버 도구 허용 목록

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues]
```

해당 MCP 서버의 도구만 등록됩니다.

### 서버 도구 차단 목록

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    tools:
      exclude: [delete_customer]
```

제외된 도구를 제외한 모든 서버 도구가 등록됩니다.

### Glob 패턴

두 목록 모두 정확한 이름과 함께 fnmatch 스타일 glob을 허용합니다. 제품 영역을 엔드포인트 하나씩 제외하기 어려운 Cloudflare의 API MCP(`?codemode=false`, 약 3,300개 도구)처럼 매우 큰 평면형 표면에 필수적입니다.

```yaml
mcp_servers:
  cloudflare:
    url: "https://mcp.cloudflare.com/mcp?codemode=false"
    auth: oauth
    tools:
      exclude: ["*_radar_*", "*_accounts_dlp_*", "*_zones_web3_*"]
```

Glob 메타문자(`*`, `?`, `[`)가 없는 항목은 정확히 일치합니다. `docs`는 `docs_search`가 아니라 `docs`라는 이름의 도구만 제외합니다.

### 우선순위 규칙

둘 다 있는 경우:

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

`include`가 우선합니다.

### 유틸리티 도구도 필터링

Hermes가 추가한 유틸리티 래퍼를 별도로 비활성화할 수도 있습니다.

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

이는 다음을 의미합니다.
- `tools.resources: false`는 `list_resources`와 `read_resource`를 비활성화합니다.
- `tools.prompts: false`는 `list_prompts`와 `get_prompt`를 비활성화합니다.

### 전체 예시

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [create_issue, list_issues, search_code]
      prompts: false

  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer]
      resources: false

  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

## 모든 항목이 필터링되면 어떻게 되나요?

설정이 호출 가능한 도구를 모두 필터링하고 지원되는 유틸리티도 모두 비활성화하거나 생략하면 Hermes는 해당 서버에 대해 빈 런타임 MCP 도구 세트를 생성하지 않습니다.

따라서 도구 목록을 깔끔하게 유지할 수 있습니다.

## 런타임 동작

### 검색 시점

Hermes는 시작 시 MCP 서버를 검색하고 해당 도구를 일반 도구 레지스트리에 등록합니다.

### 동적 도구 검색

MCP 서버는 `notifications/tools/list_changed` 알림을 보내 런타임에 사용 가능한 도구가 변경되었음을 Hermes에 알릴 수 있습니다. Hermes가 이 알림을 받으면 서버의 도구 목록을 자동으로 다시 가져와 레지스트리를 업데이트합니다. 수동으로 `/reload-mcp`를 실행할 필요가 없습니다.

이는 새 데이터베이스 스키마가 로드될 때 도구를 추가하거나 서비스가 오프라인일 때 도구를 제거하는 서버처럼 기능이 동적으로 변경되는 MCP 서버에 유용합니다.

새로 고침은 잠금으로 보호되므로 동일한 서버에서 빠르게 연속된 알림이 와도 새로 고침이 겹치지 않습니다. 프롬프트 및 리소스 변경 알림(`prompts/list_changed`, `resources/list_changed`)은 수신되지만 아직 처리하지 않습니다.

### 다시 로드

MCP 설정을 변경했다면 다음을 사용하세요.

```text
/reload-mcp
```

이 명령은 설정에서 MCP 서버를 다시 로드하고 사용 가능한 도구 목록을 새로 고칩니다. 서버 자체가 보낸 런타임 도구 변경은 위의 [동적 도구 검색](#dynamic-tool-discovery)을 참고하세요.

### 도구 세트

설정된 각 MCP 서버는 등록된 도구를 하나 이상 제공할 때 런타임 도구 세트도 생성합니다.

```text
mcp-<server>
```

이를 통해 도구 세트 수준에서 MCP 서버를 더 쉽게 파악할 수 있습니다.

## 보안 모델

### Stdio 환경 변수 필터링

stdio 서버의 경우 Hermes는 전체 셸 환경을 무조건 전달하지 않습니다.

명시적으로 설정한 `env`와 안전한 기본 환경만 전달됩니다. 이를 통해 실수로 비밀 정보가 유출되는 일을 줄입니다.

### 설정 수준 노출 제어

새로운 필터링 지원은 보안 제어 기능이기도 합니다.
- 모델이 보지 않기를 원하는 위험한 도구 비활성화
- 민감한 서버에는 최소한의 허용 목록만 노출
- 해당 표면을 노출하지 않으려는 경우 리소스/프롬프트 래퍼 비활성화

## 사용 사례

### 최소한의 이슈 관리 표면을 갖춘 GitHub 서버

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue]
      prompts: false
      resources: false
```

다음과 같이 사용합니다.

```text
Show me open issues labeled bug, then draft a new issue for the flaky MCP reconnection behavior.
```

### 위험한 작업을 제거한 Stripe 서버

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

다음과 같이 사용합니다.

```text
Look up the last 10 failed payments and summarize common failure reasons.
```

### 단일 프로젝트 루트를 위한 파일 시스템 서버

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

다음과 같이 사용합니다.

```text
Inspect the project root and explain the directory layout.
```

## 문제 해결

### MCP 서버가 연결되지 않음

확인:

```bash
# Verify MCP deps are installed (already included in standard install)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

node --version
npx --version
```

그런 다음 설정을 확인하고 Hermes를 다시 시작하세요.

### 도구가 나타나지 않음

가능한 원인:
- 서버 연결에 실패함
- 검색에 실패함
- 필터 설정에서 도구를 제외함
- 해당 서버에 유틸리티 기능이 존재하지 않음
- `enabled: false`로 서버가 비활성화됨

의도적으로 필터링한 경우라면 정상입니다.

### 리소스 또는 프롬프트 유틸리티가 나타나지 않은 이유

Hermes는 다음 두 조건이 모두 참일 때만 해당 래퍼를 등록하기 때문입니다.
1. 설정에서 허용함
2. 서버 세션이 실제로 해당 기능을 지원함

이는 의도된 동작이며 도구 목록을 실제 상태에 맞게 유지합니다.

## 병렬 도구 호출

기본적으로 MCP 도구는 한 번에 하나씩 순차적으로 실행됩니다. MCP 서버가 동시에 실행해도 안전한 도구(예: 읽기 전용 쿼리, 서로 독립적인 API 호출)를 노출한다면 병렬 실행을 선택할 수 있습니다.

```yaml
mcp_servers:
  docs:
    command: "docs-server"
    supports_parallel_tool_calls: true
```

`supports_parallel_tool_calls`가 `true`이면 Hermes는 기본 제공 읽기 전용 도구(web_search, read_file)와 마찬가지로 하나의 도구 호출 배치 안에서 해당 서버의 여러 도구를 동시에 실행할 수 있습니다.

:::caution
동시에 실행해도 안전한 MCP 서버의 도구에만 병렬 호출을 활성화하세요. 도구가 공유 상태, 파일, 데이터베이스 또는 외부 리소스를 읽고 쓰는 경우 이 설정을 활성화하기 전에 읽기/쓰기 경합 조건을 검토하세요.
:::

## MCP 샘플링 지원

MCP 서버는 `sampling/createMessage` 프로토콜을 통해 Hermes에 LLM 추론을 요청할 수 있습니다. 이를 통해 MCP 서버는 자체 모델 액세스 권한이 없어도 Hermes에 텍스트 생성을 요청할 수 있습니다. LLM 기능이 필요하지만 자체 모델 액세스 권한이 없는 서버에 유용합니다.

샘플링은 모든 MCP 서버에서 기본적으로 **활성화**되어 있습니다(MCP SDK가 지원하는 경우). `sampling` 키 아래 서버별로 설정합니다.

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    sampling:
      enabled: true            # Enable sampling (default: true)
      model: "openai/gpt-4o"  # Override model for sampling requests (optional)
      max_tokens_cap: 4096     # Max tokens per sampling response (default: 4096)
      timeout: 30              # Timeout in seconds per request (default: 30)
      max_rpm: 10              # Rate limit: max requests per minute (default: 10)
      max_tool_rounds: 5       # Max tool-use rounds in sampling loops (default: 5)
      allowed_models: []       # Allowlist of model names the server may request (empty = any)
      log_level: "info"        # Audit log level: debug, info, or warning (default: info)
```

샘플링 핸들러에는 폭주하는 사용량을 방지하기 위한 슬라이딩 윈도우 속도 제한, 요청별 제한 시간, 도구 루프 깊이 제한이 포함되어 있습니다. 측정값(요청 수, 오류, 사용 토큰)은 서버 인스턴스별로 추적됩니다.

특정 서버에서 샘플링을 비활성화하려면:

```yaml
mcp_servers:
  untrusted_server:
    url: "https://mcp.example.com"
    sampling:
      enabled: false
```

## MCP Elicitation 지원

MCP 서버는 `elicitation/create` 프로토콜(mcp Python SDK ≥ 1.11.0)을 통해 도구 호출 중간에 사용자에게 구조화된 입력을 요청할 수 있습니다. Hermes는 **form 모드** elicitation을 기존 승인 표면으로 전달합니다. CLI/TUI에서는 대화형 프롬프트로, Telegram과 Slack 같은 게이트웨이 플랫폼에서는 승인 버튼으로 전달되므로 세션이 어디에 있든 요청을 받을 수 있습니다. **URL 모드** elicitation(서버가 외부 URL로 안내하는 방식)은 지원되지 않아 거부됩니다.

Elicitation은 서버별로 **기본 활성화**되어 있습니다. `elicitation` 키 아래에서 설정합니다.

```yaml
mcp_servers:
  my_server:
    command: "my-mcp-server"
    elicitation:
      enabled: true    # default: true
      timeout: 300     # seconds to wait for your answer (default: 300)
```

5분 기본 제한 시간은 게이트웨이 승인 기본값과 동일하여 비동기 표면의 사용자가 서버가 포기하기 전에 응답할 시간을 확보합니다. 요청, 승인, 거부, 오류에 대한 서버별 측정값은 핸들러에서 추적됩니다.

## Hermes를 MCP 서버로 실행

MCP 서버에 **연결하는 것**뿐 아니라 Hermes가 MCP 서버가 될 수도 있습니다. 이를 통해 다른 MCP 지원 에이전트(Claude Code, Cursor, Codex 또는 모든 MCP 클라이언트)가 Hermes의 메시징 기능을 사용할 수 있습니다. 대화 목록 조회, 메시지 기록 읽기, 연결된 모든 플랫폼에서 메시지 전송이 가능합니다.

### 사용 시점

- Claude Code, Cursor 또는 다른 코딩 에이전트로 Hermes를 통해 Telegram/Discord/Slack 메시지를 보내고 읽고 싶을 때
- Hermes에 연결된 모든 메시징 플랫폼을 한 번에 연결하는 단일 MCP 서버가 필요할 때
- 연결된 플랫폼이 있는 실행 중인 Hermes 게이트웨이가 이미 있을 때

### 빠른 시작

```bash
hermes mcp serve
```

이 명령은 stdio MCP 서버를 시작합니다. 프로세스 수명 주기는 MCP 클라이언트가 관리합니다.

### MCP 클라이언트 설정

MCP 클라이언트 설정에 Hermes를 추가합니다. 예를 들어 Claude Code의 `~/.claude/claude_desktop_config.json`에 다음을 추가합니다.

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

또는 Hermes를 특정 위치에 설치했다면:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/home/user/.hermes/hermes-agent/venv/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

### 사용 가능한 도구

MCP 서버는 OpenClaw의 채널 브리지 표면과 Hermes 전용 채널 브라우저에 대응하는 10개 도구를 노출합니다.

| 도구 | 설명 |
|------|-------------|
| `conversations_list` | 활성 메시징 대화를 나열합니다. 플랫폼별로 필터링하거나 이름으로 검색할 수 있습니다. |
| `conversation_get` | 세션 키로 하나의 대화에 대한 상세 정보를 가져옵니다. |
| `messages_read` | 대화의 최근 메시지 기록을 읽습니다. |
| `attachments_fetch` | 특정 메시지에서 텍스트가 아닌 첨부 파일(이미지, 미디어)을 추출합니다. |
| `events_poll` | 커서 위치 이후의 새 대화 이벤트를 폴링합니다. |
| `events_wait` | 다음 이벤트가 도착할 때까지 롱 폴링/대기합니다(거의 실시간). |
| `messages_send` | 플랫폼을 통해 메시지를 보냅니다(예: `telegram:123456`, `discord:#general`). |
| `channels_list` | 모든 플랫폼에서 사용 가능한 메시징 대상을 나열합니다. |
| `permissions_list_open` | 이 브리지 세션 중 확인된 대기 중인 승인 요청을 나열합니다. |
| `permissions_respond` | 대기 중인 승인 요청을 허용하거나 거부합니다. |

### 이벤트 시스템

MCP 서버에는 Hermes의 세션 데이터베이스에서 새 메시지를 폴링하는 실시간 이벤트 브리지가 포함되어 있습니다. 이를 통해 MCP 클라이언트는 수신 대화를 거의 실시간으로 인지할 수 있습니다.

```
# Poll for new events (non-blocking)
events_poll(after_cursor=0)

# Wait for next event (blocks up to timeout)
events_wait(after_cursor=42, timeout_ms=30000)
```

이벤트 유형: `message`, `approval_requested`, `approval_resolved`

이벤트 큐는 메모리에 있으며 브리지가 연결될 때 시작됩니다. 이전 메시지는 `messages_read`를 통해 사용할 수 있습니다.

### 옵션

```bash
hermes mcp serve              # Normal mode
hermes mcp serve --verbose    # Debug logging on stderr
```

### 작동 방식

MCP 서버는 Hermes의 세션 저장소에서 직접 대화 데이터를 읽습니다. `~/.hermes/state.db`가 기본 소스이며 `sessions.json`은 레거시 폴백으로만 유지됩니다. 백그라운드 스레드가 데이터베이스에서 새 메시지를 폴링하고 메모리 내 이벤트 큐를 유지합니다. 메시지 전송에는 cron 전달과 `hermes send` CLI를 구동하는 동일한 내부 전송 엔진(`tools/send_message_tool.py`)을 사용합니다.

읽기 작업(대화 목록 조회, 기록 읽기, 이벤트 폴링)에는 게이트웨이가 실행 중일 필요가 없습니다. 전송 작업에는 플랫폼 어댑터가 활성 연결을 필요로 하므로 게이트웨이가 실행 중이어야 합니다.

### 현재 제한 사항

- 내장된 `hermes mcp serve`는 현재 **stdio 전용** MCP 서버를 노출합니다. HTTP MCP 서버가 필요하면 별도 어댑터를 실행하거나, 더 일반적으로 stdio와 HTTP(`mcp_servers.yaml` / `config.yaml`의 `url` + `headers`)를 모두 지원하는 Hermes의 MCP **클라이언트** 측을 사용하세요. 위의 [HTTP 서버](#http-servers)를 참고하세요.
- mtime 최적화 DB 폴링으로 약 200ms 간격 이벤트 폴링(파일이 변경되지 않으면 작업을 건너뜀)
- 아직 `claude/channel` 푸시 알림 프로토콜은 지원하지 않음
- 텍스트만 전송 가능(`messages_send`를 통한 미디어/첨부 파일 전송은 불가)

## 관련 문서

- [Hermes에서 MCP 사용](/guides/use-mcp-with-hermes)
- [CLI 명령](/reference/cli-commands)
- [슬래시 명령](/reference/slash-commands)
- [FAQ](/reference/faq)
