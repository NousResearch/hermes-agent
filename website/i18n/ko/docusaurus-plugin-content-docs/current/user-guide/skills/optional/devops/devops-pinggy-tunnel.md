---
title: "Pinggy Tunnel — Pinggy를 통한 설치가 필요 없는 SSH 기반 localhost 터널"
sidebar_label: "Pinggy Tunnel"
description: "Pinggy를 통한 설치가 필요 없는 SSH 기반 localhost 터널"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinggy Tunnel

Pinggy를 통한 설치가 필요 없는 SSH 기반 localhost 터널입니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/pinggy-tunnel` |
| Path | `optional-skills/devops/pinggy-tunnel` |
| Version | `0.1.0` |
| Author | Teknium (teknium1), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Pinggy`, `Tunnel`, `Networking`, `SSH`, `Webhook`, `Localhost` |
| Related skills | `cloudflared-quick-tunnel`, [`webhook-subscriptions`](/docs/user-guide/skills/bundled/devops/devops-webhook-subscriptions) |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Pinggy Tunnel Skill

Pinggy SSH 역방향 터널을 사용하여 로컬 서비스(개발 서버, 웹훅 수신기, MCP 엔드포인트, 데모)를 퍼블릭 인터넷에 노출합니다. 설치할 데몬이 없습니다 — 사용자의 기본 SSH 클라이언트가 `a.pinggy.io:443`에 연결하고 Pinggy가 퍼블릭 HTTP/HTTPS URL을 반환합니다.

무료 티어: 60분 터널, 무작위 하위 도메인, 가입 불필요. 프로 티어($3/월)는 토큰을 사용하여 선택적으로 사용할 수 있습니다.

## When to Use

- 사용자가 "이것을 로컬로 노출시켜 줘", "내 개발 서버를 공유해 줘", "이 URL을 공개적으로 만들어 줘", "포트 N번을 터널링해 줘", "웹훅을 위한 공개 URL을 가져와 줘"라고 요청할 때
- 로컬 작업 중 외부 서비스(Stripe, GitHub, Discord, AgentMail 등)로부터 웹훅 콜백을 받아야 할 때
- 단발성 HTTP 데모(MCP 서버, Ollama/vLLM 엔드포인트, 대시보드)를 원격의 상대방과 공유할 때
- 호스트에 SSH는 있지만 `cloudflared` / `ngrok` 바이너리가 없으며, 이를 설치하는 것이 과도할 때

호스트에 이미 `cloudflared`가 구성되어 있는 경우 `cloudflared-quick-tunnel` 스킬을 선호하십시오 — Cloudflare 퀵 터널은 60분 후에도 만료되지 않습니다.

## Prerequisites

- PATH에 있는 `ssh` (`ssh -V`로 확인). Linux, macOS, Windows 10+의 기본값입니다. 다른 설치는 필요하지 않습니다.
- 터널이 시작되기 전에 `127.0.0.1:<port>`에서 수신 대기 중인 로컬 서비스. Pinggy는 URL을 반환하지만 로컬 오리진(origin)이 실행될 때까지 502 오류를 반환합니다.

선택 사항:

- 유료 프로 기능(영구 하위 도메인, 사용자 지정 도메인, 다중 터널, 60분 제한 없음)을 위한 `PINGGY_TOKEN` 환경 변수. 무료 티어는 자격 증명이 필요하지 않습니다.

## Quick Reference

```bash
# 포트 8000에 대한 일반 HTTP/HTTPS 터널 (무료 티어)
ssh -p 443 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R0:localhost:8000 free@a.pinggy.io

# TCP 터널 (데이터베이스, 원시 SSH 등)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:5432 tcp@a.pinggy.io

# TLS 터널 (Pinggy는 암호화를 해제할 수 없음 — 오리진에서 직접 인증서를 준비해야 함)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:443 tls@a.pinggy.io

# 기본 인증(Basic auth) 게이트 (b:user:pass)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "b:admin:secret+free@a.pinggy.io"

# Bearer 토큰 게이트 (k:token)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "k:mysecrettoken+free@a.pinggy.io"

# IP 화이트리스트 (w:CIDR)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "w:203.0.113.0/24+free@a.pinggy.io"

# CORS 활성화 + HTTPS 리디렉션 강제
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "co+x:https+free@a.pinggy.io"

# 프로 티어 (영구 URL, 60분 제한 없음)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 "$PINGGY_TOKEN+a.pinggy.io"
```

## Procedure — Start a Tunnel and Get the URL

모델은 **반드시** `terminal` 도구를 사용해야 합니다. 터널은 공유하는 동안 계속 유지되어야 하므로 백그라운드 프로세스로 실행하고 stdout에서 공개 URL을 파싱하세요.

### 1. Confirm a local origin is up

```bash
curl -sI http://127.0.0.1:8000/ | head -1
# HTTP/1.x 200 (또는 연결 거부가 아닌 모든 응답)을 예상함
```

아직 수신 대기 중인 항목이 없다면 먼저 시작하세요 (예: `python3 -m http.server 8000 --bind 127.0.0.1`). Pinggy는 아무것도 없는 대상을 가리키는 URL을 반환할 수 있지만, 사용자는 로컬 오리진이 켜질 때까지 502 오류를 보게 됩니다.

### 2. Launch the tunnel as a background process

`terminal(background=True)`를 사용하고 출력을 로그 파일로 캡처하세요 (Pinggy는 stdout에 URL을 출력한 다음 연결을 열어 둡니다):

```bash
LOG=/tmp/pinggy-8000.log
nohup ssh -p 443 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R0:localhost:8000 free@a.pinggy.io \
    > "$LOG" 2>&1 &
echo $! > /tmp/pinggy-8000.pid
```

`StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null`은 첫 실행 시의 호스트 키 프롬프트를 건너뜁니다. `ServerAliveInterval=30`은 유휴 NAT에 의해 SSH 세션이 끊어지는 것을 방지합니다.

### 3. Parse the URL out of the log

```bash
sleep 4
grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/pinggy-8000.log | head -1
```

예상되는 출력은 다음과 같습니다:

```
You are not authenticated.
Your tunnel will expire in 60 minutes.
http://yqycl-98-162-69-48.a.free.pinggy.link
https://yqycl-98-162-69-48.a.free.pinggy.link
```

사용자에게 `https://...pinggy.link` URL을 전달하세요.

### 4. Verify

```bash
curl -sI https://<the-url>/ | head -3
# 200/302 또는 로컬 오리진이 실제로 반환하는 응답을 예상함
```

`502 Bad Gateway`가 표시되면 SSH 세션은 열려 있지만 로컬 오리진이 수신 대기하지 않는 것입니다 — 먼저 1단계를 수정하세요.

### 5. Teardown

```bash
kill "$(cat /tmp/pinggy-8000.pid)"
# 또는, pid 파일이 손실된 경우:
pkill -f 'ssh -p 443 .* free@a\.pinggy\.io'
```

`terminal(background=True)`에서 가져온 session_id가 있는 경우 `process(action='kill', session_id=...)`를 선호하세요.

## Access Control via Username Keywords

Pinggy는 `+`로 구분된 SSH 사용자 이름에 제어 플래그를 쌓습니다. 사용자 이름 인수에 `+`가 포함된 경우 항상 `user@host` 인수를 전체 따옴표로 묶으세요:

| Keyword | Effect |
|---------|--------|
| `b:user:pass` | HTTP Basic auth 게이트 |
| `k:token` | Bearer-token 헤더 게이트 (`Authorization: Bearer <token>`) |
| `w:CIDR` | IP 화이트리스트 (단일 IP 또는 CIDR, 반복 가능) |
| `co` | `Access-Control-Allow-Origin: *` 추가 (CORS) |
| `x:https` | HTTPS 강제 — HTTP를 HTTPS로 자동 리디렉션 |
| `a:Name:Value` | 요청 헤더 추가 |
| `u:Name:Value` | 요청 헤더 업데이트 |
| `r:Name` | 요청 헤더 제거 |
| `qr` | stdout에 URL의 QR 코드 출력 (모바일 공유에 유용함) |

자유롭게 조합 가능: `"b:admin:secret+co+x:https+free@a.pinggy.io"`.

## Web Debugger (optional)

Pinggy는 인바운드 트래픽을 `localhost:4300`으로 미러링하여 검사할 수 있습니다. SSH 명령에 로컬 포워딩을 추가하세요:

```bash
ssh -p 443 -L4300:localhost:4300 -R0:localhost:8000 free@a.pinggy.io
```

그런 다음 브라우저에서 `http://localhost:4300`을 열어 실시간 요청/응답 쌍을 확인하세요.

## Pitfalls

- **무료 티어의 60분 엄격한 제한.** SSH 세션은 60분 지점에서 종료되며 URL은 죽습니다. 더 오래 공유하려면 `PINGGY_TOKEN` (프로)을 사용하거나 쉘 루프를 통해 자동 재시작하세요 (참고: 무료 티어의 경우 재시작할 때마다 URL이 변경됨).
- **무료 티어 URL은 무작위이며 재시작할 때마다 변경됩니다.** 북마크하거나 구성 파일에 붙여넣지 마세요. 매번 로그에서 다시 파싱하세요.
- **동시 무료 터널은 소스 IP당 하나로 제한됩니다.** 같은 컴퓨터에서 두 번째 터널을 시작하면 보통 첫 번째 터널이 종료됩니다. 프로 티어는 이 제한을 해제합니다.
- **사용자 이름의 `+`는 따옴표로 묶어야 합니다.** 따옴표가 없는 `ssh ... b:admin:secret+free@a.pinggy.io`는 bash에서는 작동하지만 `+`를 특수하게 처리하거나 프로그래밍 방식으로 조립된 쉘에서는 손상됩니다. 항상 큰따옴표로 감싸세요.
- **접근 제어 플래그 없이 민감한 항목을 터널링하지 마세요.** 순수 HTTP 터널은 URL을 아는 누구나 접근할 수 있습니다. 공개되지 않은 서비스의 경우 `b:`, `k:` 또는 `w:`를 사용하세요.
- **`process(action='log')`는 SSH 배너 출력을 놓칠 수 있습니다.** Pinggy는 URL을 출력한 다음 SSH 세션이 인터랙티브 상태로 전환됩니다. 항상 로그 파일로 리디렉션하고 파일을 직접 `grep`하세요 — `cloudflared-quick-tunnel`과 같은 패턴입니다.
- **첫 실행 시 호스트 키 프롬프트.** 기본 OpenSSH 구성은 사용자에게 Pinggy의 호스트 키를 수락할지 묻습니다. 무인(unattended) 실행을 위해서는 항상 `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`을 전달하세요.
- **TCP 및 TLS 터널은 https URL이 아닌 `<subdomain>.a.pinggy.online:<port>` 쌍을 반환합니다.** 다른 정규식으로 파싱하세요 (`tcp://`와 포트). 모든 Pinggy 터널이 HTTP라고 가정하지 마세요.
- **프로 모드에서는 토큰을 플래그가 아닌 사용자 이름으로 요구합니다.** `"$PINGGY_TOKEN+a.pinggy.io"`를 사용하세요 (`free@` 제외). 토큰을 사용하면 안정적인 하위 도메인을 위해 `:persistent`를 추가할 수도 있습니다 — 자세한 내용은 `pinggy.io/docs/`를 참조하세요.

## Recipes

로컬 오리진과 Pinggy 터널을 결합하는 복합 패턴. 각 레시피는 자체 완결형입니다 — 오리진 시작, 터널 시작, URL 파싱, 사용자에게 전달.

### Recipe 1 — Receive a webhook callback

외부 서비스(Stripe, GitHub, Discord, AgentMail 등)가 로컬 작업 중에 퍼블릭 접근이 가능한 URL로 POST 요청을 해야 할 때 사용합니다.

```bash
# 1. 작은 캡처링 서버: 모든 요청이 /tmp/webhook-hits.log에 추가됨
cat >/tmp/webhook-server.py <<'PY'
import http.server, json, datetime, pathlib
LOG = pathlib.Path("/tmp/webhook-hits.log")
class H(http.server.BaseHTTPRequestHandler):
    def _capture(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        rec = {"t": datetime.datetime.utcnow().isoformat(), "path": self.path,
               "method": self.command, "headers": dict(self.headers), "body": body}
        with LOG.open("a") as f: f.write(json.dumps(rec) + "\n")
        self.send_response(200); self.send_header("content-type","application/json")
        self.end_headers(); self.wfile.write(b'{"ok":true}\n')
    def do_GET(self): self._capture()
    def do_POST(self): self._capture()
    def log_message(self,*a,**k): pass
http.server.HTTPServer(("127.0.0.1", 18080), H).serve_forever()
PY
nohup python3 /tmp/webhook-server.py >/tmp/webhook-server.log 2>&1 &
echo $! >/tmp/webhook-server.pid

# 2. 터널 — 임의의 요청들이 캡처 로그를 오염시키지 않도록 bearer 토큰으로 차단
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:18080 "k:$(openssl rand -hex 12)+free@a.pinggy.io" \
    >/tmp/webhook-pinggy.log 2>&1 &
echo $! >/tmp/webhook-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/webhook-pinggy.log | head -1)
echo "Webhook URL: $URL"

# 3. 에이전트가 작업하는 동안 요청 로그 확인
tail -f /tmp/webhook-hits.log
```

호출해야 하는 서비스에 `$URL`을 전달하세요. 해제(Teardown): `kill $(cat /tmp/webhook-server.pid) $(cat /tmp/webhook-pinggy.pid)`.

### Recipe 2 — Expose an MCP server over HTTP/SSE

원격 MCP 클라이언트(다른 컴퓨터에 있는 Claude Desktop, 팀원의 에디터 등)가 로컬 시스템에서 실행 중인 MCP 서버에 도달해야 할 때 사용합니다. HTTP 전송 방식을 사용하는 MCP 서버에서만 작동합니다 — stdio 모드 서버는 터널링할 수 없습니다.

```bash
# 1. HTTP 모드에서 MCP 서버 시작 (예: 포트 8765의 FastMCP 서버)
nohup python3 my_mcp_server.py --transport http --port 8765 \
    >/tmp/mcp-server.log 2>&1 &
echo $! >/tmp/mcp-server.pid

# 2. Bearer 토큰으로 터널링 — MCP 트래픽은 인터넷에 공개되어서는 안 됨
TOKEN=$(openssl rand -hex 16)
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:8765 "k:$TOKEN+free@a.pinggy.io" \
    >/tmp/mcp-pinggy.log 2>&1 &
echo $! >/tmp/mcp-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/mcp-pinggy.log | head -1)
echo "MCP URL: $URL"
echo "Bearer token: $TOKEN"
```

원격 클라이언트는 `Authorization: Bearer $TOKEN`과 함께 `$URL`에 연결합니다. Hermes 자체 기본 MCP 클라이언트 구성: `{"transport": "http", "url": "<URL>", "headers": {"Authorization": "Bearer <TOKEN>"}}`.

### Recipe 3 — Expose a local LLM endpoint (Ollama / vLLM / llama.cpp)

원격 호출자(다른 에이전트, 전화, 팀원)와 로컬 모델을 공유합니다. Ollama는 `:11434`에서, vLLM 및 llama.cpp는 일반적으로 `:8000`에서 수신 대기합니다.

```bash
# 사전 요건: 모델 서버가 이미 127.0.0.1:11434(Ollama 기본값)에서 실행 중이어야 함
TOKEN=$(openssl rand -hex 16)
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:11434 "k:$TOKEN+co+free@a.pinggy.io" \
    >/tmp/llm-pinggy.log 2>&1 &
echo $! >/tmp/llm-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/llm-pinggy.log | head -1)
echo "Endpoint: $URL"
echo "Token:    $TOKEN"

# 확인
curl -s "$URL/api/tags" -H "Authorization: Bearer $TOKEN" | head
```

`co`는 브라우저 호출자가 엔드포인트에 도달할 수 있도록 CORS를 활성화합니다. 백엔드 전용 호출자에게는 `co`를 삭제하세요. OpenAI 호환 vLLM/llama.cpp 엔드포인트의 경우, 호출자는 `Authorization: Bearer $TOKEN`과 함께 기본 URL `$URL/v1`을 사용합니다 — 하지만 Pinggy는 본문의 내용을 제거하거나 대체하지 않으므로 모델 서버 자체에서 Pinggy의 토큰을 볼 수 있습니다. 로컬 서버는 인증을 무시하고(`127.0.0.1`에 있으므로) Pinggy가 게이트웨이 역할을 하도록 구성되어야 합니다.

### Recipe 4 — Share a dev server with a one-shot password

"팀원에게 내 실행 중인 앱을 살펴볼 수 있게 하는" 가장 빠른 패턴입니다. 임의의 비밀번호가 1번 출력되며, Ctrl-C를 누르면 종료됩니다.

```bash
PASS=$(openssl rand -base64 12 | tr -d '+/=' | head -c 12)
echo "Dev server password: $PASS"
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:3000 "b:dev:$PASS+co+x:https+free@a.pinggy.io"
# 터미널에 URL이 출력됩니다. URL + 비밀번호를 공유하세요. Ctrl-C를 누르면 해제됩니다.
```

`b:dev:$PASS`는 HTTP 기본 인증으로 URL을 제한합니다. `x:https`는 TLS를 강제합니다. `co`는 SPA 프런트엔드에 대한 CORS를 추가합니다.

## Verification

```bash
# 종단 간(End-to-end): 아주 간단한 오리진을 실행하고, 터널링하고, 접근을 시도한 후, 종료합니다.
python3 -m http.server 18000 --bind 127.0.0.1 >/tmp/origin.log 2>&1 &
ORIGIN_PID=$!

nohup ssh -p 443 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -R0:localhost:18000 free@a.pinggy.io >/tmp/pinggy-verify.log 2>&1 &
SSH_PID=$!

sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/pinggy-verify.log | head -1)
echo "URL: $URL"
curl -sI "$URL/" | head -1

kill "$SSH_PID" "$ORIGIN_PID"
```

예상 결과: `pinggy.link` URL과 curl head의 `HTTP/2 200`.
