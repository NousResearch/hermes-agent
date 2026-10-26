---
title: "Pinggy Tunnel — Pinggy를 통한 SSH 기반 무설치 로컬호스트 터널"
sidebar_label: "Pinggy Tunnel"
description: "Pinggy를 통한 SSH 기반 무설치 로컬호스트 터널"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Pinggy Tunnel

Pinggy를 통한 SSH 기반 무설치 로컬호스트 터널입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/devops/pinggy-tunnel`로 설치 |
| 경로 | `optional-skills/devops/pinggy-tunnel` |
| 버전 | `0.1.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Pinggy`, `Tunnel`, `Networking`, `SSH`, `Webhook`, `Localhost` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Pinggy Tunnel 스킬

Pinggy SSH 리버스 터널을 사용해 로컬 서비스(개발 서버, 웹훅 수신기, MCP 엔드포인트, 데모)를 공개 인터넷에 노출합니다. 설치할 데몬은 없습니다. 사용자의 기본 SSH 클라이언트가 `a.pinggy.io:443`에 연결하면 Pinggy가 공개 HTTP/HTTPS URL을 반환합니다.

무료 요금제: 60분 터널, 무작위 서브도메인, 가입 불필요. Pro 요금제(월 $3)는 선택적으로 사용할 수 있으며 토큰이 필요합니다.

## 사용 시점

- 사용자가 "이걸 로컬에서 노출해 줘", "내 개발 서버를 공유해 줘", "이 URL을 공개해 줘", "포트 N을 터널링해 줘", "웹훅용 공개 URL을 가져와 줘"라고 요청한 경우
- 로컬 작업 중 웹훅 콜백을 받아야 하는 경우(Stripe, GitHub, Discord, AgentMail)
- 일회성 HTTP 데모(MCP 서버, Ollama/vLLM 엔드포인트, 대시보드)를 원격 사용자와 공유하는 경우
- 호스트에 SSH는 있지만 `cloudflared` / `ngrok` 바이너리가 없고, 이를 설치하는 것이 과한 경우

호스트에 이미 `cloudflared`가 구성되어 있다면 `cloudflared-quick-tunnel` 스킬을 우선 사용하세요. Cloudflare 퀵 터널은 60분 후 만료되지 않습니다.

## 사전 요구 사항

- PATH에 `ssh`가 있어야 합니다(`ssh -V`). Linux, macOS 및 Windows 10 이상에서는 기본 제공됩니다. 다른 설치는 필요하지 않습니다.
- 터널을 시작하기 전에 `127.0.0.1:<port>`에서 수신 대기 중인 로컬 서비스가 있어야 합니다. Pinggy는 URL을 반환하지만 로컬 오리진이 실행될 때까지는 502를 반환합니다.

선택 사항:

- 유료 Pro 기능(영구 서브도메인, 사용자 지정 도메인, 여러 터널, 60분 제한 없음)을 사용하려면 `PINGGY_TOKEN` 환경 변수를 설정합니다. 무료 요금제에는 자격 증명이 필요하지 않습니다.

## 빠른 참조

```bash
# Plain HTTP/HTTPS tunnel for port 8000 (free tier)
ssh -p 443 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R0:localhost:8000 free@a.pinggy.io

# TCP tunnel (databases, raw SSH, etc.)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:5432 tcp@a.pinggy.io

# TLS tunnel (Pinggy can't decrypt — bring your own certs at origin)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:443 tls@a.pinggy.io

# Basic auth gate (b:user:pass)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "b:admin:secret+free@a.pinggy.io"

# Bearer token gate (k:token)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "k:mysecrettoken+free@a.pinggy.io"

# IP whitelist (w:CIDR)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "w:203.0.113.0/24+free@a.pinggy.io"

# Enable CORS + force HTTPS redirect
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "co+x:https+free@a.pinggy.io"

# Pro tier (persistent URL, no 60-min cap)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 "$PINGGY_TOKEN+a.pinggy.io"
```

## 절차 — 터널을 시작하고 URL 가져오기

모델은 `terminal` 도구를 사용해야 합니다. 공유하는 동안 터널이 계속 실행되어야 하므로 백그라운드 프로세스로 실행하고 stdout에서 공개 URL을 파싱합니다.

### 1. 로컬 오리진이 실행 중인지 확인

```bash
curl -sI http://127.0.0.1:8000/ | head -1
# expect HTTP/1.x 200 (or any non-connection-refused response)
```

아직 수신 대기 중인 항목이 없다면 먼저 시작합니다(예: `python3 -m http.server 8000 --bind 127.0.0.1`). Pinggy는 아무것도 가리키지 않는 URL도 문제없이 반환하므로 오리진이 실행될 때까지 사용자는 502를 보게 됩니다.

### 2. 터널을 백그라운드 프로세스로 시작

`terminal(background=True)`를 사용하고 출력을 로그 파일에 캡처합니다(Pinggy는 stdout에 URL을 출력한 뒤 연결을 계속 열어 둡니다).

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

`StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null`은 최초 실행 시 호스트 키 프롬프트를 건너뜁니다. `ServerAliveInterval=30`은 유휴 NAT로 인해 SSH 세션이 종료되는 것을 방지합니다.

### 3. 로그에서 URL 파싱

```bash
sleep 4
grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/pinggy-8000.log | head -1
```

예상 출력은 다음과 같습니다.

```
You are not authenticated.
Your tunnel will expire in 60 minutes.
http://yqycl-98-162-69-48.a.free.pinggy.link
https://yqycl-98-162-69-48.a.free.pinggy.link
```

사용자에게 `https://...pinggy.link` URL을 전달합니다.

### 4. 확인

```bash
curl -sI https://<the-url>/ | head -3
# expect 200/302/whatever the local origin actually returns
```

`502 Bad Gateway`가 표시되면 SSH 세션은 실행 중이지만 로컬 오리진이 수신 대기 중이 아닌 것입니다. 먼저 1단계를 해결하세요.

### 5. 종료

```bash
kill "$(cat /tmp/pinggy-8000.pid)"
# or, if the pid file got lost:
pkill -f 'ssh -p 443 .* free@a\.pinggy\.io'
```

`terminal(background=True)`에서 `session_id`를 받은 경우 `process(action='kill', session_id=...)`을 우선 사용합니다.

## 사용자 이름 키워드를 통한 접근 제어

Pinggy는 `+`로 구분된 제어 플래그를 SSH 사용자 이름에 조합합니다. `+`가 포함된 전체 `user@host` 인수는 항상 따옴표로 묶으세요.

| 키워드 | 효과 |
|---------|--------|
| `b:user:pass` | HTTP Basic 인증 게이트 |
| `k:token` | Bearer 토큰 헤더 게이트(`Authorization: Bearer <token>`) |
| `w:CIDR` | IP 허용 목록(단일 IP 또는 CIDR, 반복 가능) |
| `co` | `Access-Control-Allow-Origin: *` 추가(CORS) |
| `x:https` | HTTPS 강제 — HTTP를 HTTPS로 자동 리디렉션 |
| `a:Name:Value` | 요청 헤더 추가 |
| `u:Name:Value` | 요청 헤더 업데이트 |
| `r:Name` | 요청 헤더 제거 |
| `qr` | URL의 QR 코드 출력(stdout)(모바일 공유에 유용) |

자유롭게 조합할 수 있습니다: `"b:admin:secret+co+x:https+free@a.pinggy.io"`.

## 웹 디버거(선택 사항)

Pinggy는 수신 트래픽을 검사를 위해 `localhost:4300`으로 미러링할 수 있습니다. SSH 명령에 로컬 포워딩을 추가합니다.

```bash
ssh -p 443 -L4300:localhost:4300 -R0:localhost:8000 free@a.pinggy.io
```

그런 다음 브라우저에서 `http://localhost:4300`을 열어 실시간 요청/응답 쌍을 확인합니다.

## 주의 사항

- **무료 요금제는 60분으로 엄격히 제한됩니다.** SSH 세션은 60분이 지나면 종료되고 URL이 작동하지 않습니다. 더 오래 공유하려면 `PINGGY_TOKEN`(Pro)을 사용하거나 셸 루프로 자동 재시작하세요(무료 요금제에서는 재시작할 때마다 URL이 변경됨).
- **무료 요금제 URL은 무작위이며 재시작할 때 변경됩니다.** 북마크하거나 구성 파일에 붙여 넣지 마세요. 매번 로그에서 다시 파싱하세요.
- **무료 터널의 동시 사용은 소스 IP당 하나로 제한됩니다.** 같은 컴퓨터에서 두 번째 터널을 시작하면 보통 첫 번째 터널이 종료됩니다. Pro 요금제에서는 이 제한이 해제됩니다.
- **사용자 이름의 `+`는 따옴표로 묶어야 합니다.** 일반적인 bash에서는 따옴표 없이 `ssh ... b:admin:secret+free@a.pinggy.io`도 작동하지만, `+`를 특별하게 처리하는 셸이나 프로그래밍 방식으로 조립할 때는 깨집니다. 항상 큰따옴표로 감싸세요.
- **접근 제어 플래그 없이 민감한 것을 터널링하지 마세요.** 기본 HTTP 터널은 URL을 아는 누구나 접근할 수 있습니다. 공개 서비스가 아닌 경우 `b:`, `k:`, 또는 `w:`를 사용하세요.
- **`process(action='log')`는 SSH 배너 출력을 놓칠 수 있습니다.** Pinggy는 URL을 출력한 뒤 SSH 세션을 대화형으로 전환합니다. 항상 로그 파일로 리디렉션하고 파일을 직접 `grep`하세요. `cloudflared-quick-tunnel`과 같은 패턴입니다.
- **최초 실행 시 호스트 키 프롬프트가 표시됩니다.** 기본 OpenSSH 구성은 사용자에게 Pinggy의 호스트 키를 수락할지 묻습니다. 무인 실행에서는 항상 `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`을 전달하세요.
- **TCP 및 TLS 터널은 https URL이 아니라 `<subdomain>.a.pinggy.online:<port>` 쌍을 반환합니다.** 다른 정규식(`tcp://` 및 포트)으로 파싱하세요. 모든 Pinggy 터널이 HTTP라고 가정하지 마세요.
- **Pro 모드에서는 플래그가 아니라 토큰을 사용자 이름으로 사용해야 합니다.** `"$PINGGY_TOKEN+a.pinggy.io"`를 사용하세요(`free@` 없음). 토큰을 사용하면 안정적인 서브도메인을 위해 `:persistent`를 추가할 수도 있습니다. 자세한 내용은 `pinggy.io/docs/`를 참고하세요.

## 레시피

로컬 오리진과 Pinggy 터널을 결합한 복합 패턴입니다. 각 레시피는 독립적으로 사용할 수 있습니다. 오리진을 시작하고, 터널을 시작하고, URL을 파싱한 다음 사용자에게 전달하세요.

### 레시피 1 — 웹훅 콜백 수신

외부 서비스(Stripe, GitHub, Discord, AgentMail 등)가 로컬 작업 중 공개적으로 접근 가능한 URL로 POST해야 할 때 사용합니다.

```bash
# 1. Tiny capturing server: every request gets appended to /tmp/webhook-hits.log
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

# 2. Tunnel — bearer-token-gate so randos can't pollute the capture log
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:18080 "k:$(openssl rand -hex 12)+free@a.pinggy.io" \
    >/tmp/webhook-pinggy.log 2>&1 &
echo $! >/tmp/webhook-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/webhook-pinggy.log | head -1)
echo "Webhook URL: $URL"

# 3. While the agent works, watch hits land
tail -f /tmp/webhook-hits.log
```

`$URL`을 호출해야 하는 서비스에 전달합니다. 종료: `kill $(cat /tmp/webhook-server.pid) $(cat /tmp/webhook-pinggy.pid)`.

### 레시피 2 — HTTP/SSE를 통한 MCP 서버 노출

원격 MCP 클라이언트(다른 컴퓨터의 Claude Desktop, 팀원의 에디터 등)가 로컬 컴퓨터에서 실행 중인 MCP 서버에 연결해야 할 때 사용합니다. HTTP 전송을 사용하는 MCP 서버에서만 작동하며, stdio 모드 서버는 터널링할 수 없습니다.

```bash
# 1. Start the MCP server in HTTP mode (example: a FastMCP server on port 8765)
nohup python3 my_mcp_server.py --transport http --port 8765 \
    >/tmp/mcp-server.log 2>&1 &
echo $! >/tmp/mcp-server.pid

# 2. Tunnel with a bearer token — MCP traffic should not be open to the internet
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

원격 클라이언트는 `Authorization: Bearer $TOKEN`과 함께 `$URL`에 연결합니다. Hermes 자체의 네이티브 MCP 클라이언트 구성: `{"transport": "http", "url": "<URL>", "headers": {"Authorization": "Bearer <TOKEN>"}}`.

### 레시피 3 — 로컬 LLM 엔드포인트 노출(Ollama / vLLM / llama.cpp)

로컬 모델을 원격 호출자(다른 에이전트, 휴대폰, 팀원)와 공유합니다. Ollama는 `:11434`에서 수신 대기하고, vLLM과 llama.cpp는 일반적으로 `:8000`에서 수신 대기합니다.

```bash
# Pre-req: the model server is already running on 127.0.0.1:11434 (Ollama default)
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

# Verify
curl -s "$URL/api/tags" -H "Authorization: Bearer $TOKEN" | head
```

`co`는 브라우저 호출자가 엔드포인트에 접근할 수 있도록 CORS를 활성화합니다. 백엔드 전용 호출자라면 `co`를 제거하세요. OpenAI 호환 vLLM/llama.cpp 엔드포인트의 경우 호출자는 `Authorization: Bearer $TOKEN`과 함께 기본 URL `$URL/v1`을 사용합니다. 단, Pinggy는 본문에서 아무것도 제거하거나 대체하지 않으므로 모델 서버 자체가 Pinggy의 토큰을 보게 됩니다. 로컬 서버는 이미 `127.0.0.1`에 있으므로 인증을 무시하도록 구성하고 Pinggy가 게이트 역할을 맡게 해야 합니다.
### 레시피 4 — 일회성 비밀번호로 개발 서버 공유하기

“팀원이 실행 중인 내 앱을 잠깐 사용해 보게 하는” 가장 빠른 방법입니다. 무작위 비밀번호를 생성해 한 번만 출력하고, Ctrl-C를 누르면 종료됩니다.

```bash
PASS=$(openssl rand -base64 12 | tr -d '+/=' | head -c 12)
echo "Dev server password: $PASS"
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:3000 "b:dev:$PASS+co+x:https+free@a.pinggy.io"
# URL prints to the terminal. Share URL + password. Ctrl-C to tear down.
```

`b:dev:$PASS`는 HTTP Basic 인증으로 URL을 보호합니다. `x:https`는 TLS를 강제합니다. `co`는 SPA 프론트엔드에 CORS를 추가합니다.

## 검증

```bash
# End-to-end: spin up a trivial origin, tunnel it, hit it, tear down
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

예상 결과: `pinggy.link` URL이 출력되고 curl 헤더의 첫 줄에 `HTTP/2 200`이 표시됩니다.
