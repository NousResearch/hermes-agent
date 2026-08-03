---
title: "Pinggy Tunnel — 通过 Pinggy 实现零安装 SSH localhost 隧道"
sidebar_label: "Pinggy Tunnel"
description: "通过 Pinggy 实现零安装 SSH localhost 隧道"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pinggy Tunnel

通过 Pinggy 实现零安装 SSH localhost 隧道。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/devops/pinggy-tunnel` 安装 |
| 路径 | `optional-skills/devops/pinggy-tunnel` |
| 版本 | `0.1.0` |
| 作者 | Teknium (teknium1), Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Pinggy`, `Tunnel`, `Networking`, `SSH`, `Webhook`, `Localhost` |
| 相关 skill | `cloudflared-quick-tunnel`, [`webhook-subscriptions`](/user-guide/skills/bundled/devops/devops-webhook-subscriptions) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Pinggy Tunnel Skill

使用 Pinggy SSH 反向隧道将本地服务（开发服务器、webhook 接收器、MCP 端点、演示）暴露到公共互联网。无需安装任何守护进程——用户的标准 SSH 客户端连接到 `a.pinggy.io:443`，Pinggy 返回一个公共 HTTP/HTTPS URL。

免费套餐：60 分钟隧道，随机子域名，无需注册。Pro 套餐（$3/月）需要 token，按需选用。

## 使用时机

- 用户要求"暴露本地服务"、"分享我的开发服务器"、"将此 URL 公开"、"隧道端口 N"、"为 webhook 获取公共 URL"
- 在本地任务期间需要接收 webhook 回调（Stripe、GitHub、Discord、AgentMail）
- 与远程方分享一次性 HTTP 演示（MCP 服务器、Ollama/vLLM 端点、仪表盘）
- 主机有 SSH 但没有 `cloudflared` / `ngrok` 二进制文件，安装一个又显得多余

如果主机已配置 `cloudflared`，优先使用 `cloudflared-quick-tunnel` skill——Cloudflare 快速隧道不会在 60 分钟后过期。

## 前提条件

- PATH 中有 `ssh`（`ssh -V`）。Linux、macOS 和 Windows 10+ 默认自带。无需其他安装。
- 隧道启动前，本地服务已在 `127.0.0.1:<port>` 上监听。Pinggy 会返回 URL，但在本地源服务启动之前访问会返回 502。

可选：

- `PINGGY_TOKEN` 环境变量，用于付费 Pro 功能（持久子域名、自定义域名、多隧道、无 60 分钟限制）。免费套餐无需凭据。

## 快速参考

```bash
# 端口 8000 的普通 HTTP/HTTPS 隧道（免费套餐）
ssh -p 443 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R0:localhost:8000 free@a.pinggy.io

# TCP 隧道（数据库、原始 SSH 等）
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:5432 tcp@a.pinggy.io

# TLS 隧道（Pinggy 无法解密——在源端自带证书）
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:443 tls@a.pinggy.io

# Basic auth 认证（b:user:pass）
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "b:admin:secret+free@a.pinggy.io"

# Bearer token 认证（k:token）
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "k:mysecrettoken+free@a.pinggy.io"

# IP 白名单（w:CIDR）
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "w:203.0.113.0/24+free@a.pinggy.io"

# 启用 CORS + 强制 HTTPS 重定向
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    "co+x:https+free@a.pinggy.io"

# Pro 套餐（持久 URL，无 60 分钟限制）
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 "$PINGGY_TOKEN+a.pinggy.io"
```

## 操作流程——启动隧道并获取 URL

模型应使用 `terminal` 工具。隧道在共享期间必须保持存活，因此以后台进程方式运行，并从 stdout 解析公共 URL。

### 1. 确认本地源服务已启动

```bash
curl -sI http://127.0.0.1:8000/ | head -1
# 期望返回 HTTP/1.x 200（或任何非连接拒绝的响应）
```

如果尚无服务在监听，先启动它（例如 `python3 -m http.server 8000 --bind 127.0.0.1`）。Pinggy 会正常返回 URL，但在本地源服务启动之前用户会看到 502。

### 2. 以后台进程方式启动隧道

使用 `terminal(background=True)` 并将输出捕获到日志文件（Pinggy 在 stdout 打印 URL 后保持连接）：

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

`StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null` 跳过首次运行的主机密钥确认提示。`ServerAliveInterval=30` 防止 SSH 会话因空闲 NAT 而被断开。

### 3. 从日志中解析 URL

```bash
sleep 4
grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/pinggy-8000.log | head -1
```

预期输出如下：

```
You are not authenticated.
Your tunnel will expire in 60 minutes.
http://yqycl-98-162-69-48.a.free.pinggy.link
https://yqycl-98-162-69-48.a.free.pinggy.link
```

将 `https://...pinggy.link` URL 提供给用户。

### 4. 验证

```bash
curl -sI https://<the-url>/ | head -3
# 期望返回 200/302/本地源服务实际返回的状态码
```

如果返回 `502 Bad Gateway`，说明 SSH 会话已建立但本地源服务未在监听——先修复步骤 1。

### 5. 关闭隧道

```bash
kill "$(cat /tmp/pinggy-8000.pid)"
# 或者，如果 pid 文件丢失：
pkill -f 'ssh -p 443 .* free@a\.pinggy\.io'
```

如果有来自 `terminal(background=True)` 的 session_id，优先使用 `process(action='kill', session_id=...)`。

## 通过用户名关键字进行访问控制

Pinggy 将控制标志以 `+` 分隔堆叠到 SSH 用户名中。当 `user@host` 参数包含 `+` 时，始终用引号括起整个参数：

| 关键字 | 效果 |
|---------|--------|
| `b:user:pass` | HTTP Basic auth 认证门控 |
| `k:token` | Bearer token 请求头门控（`Authorization: Bearer <token>`） |
| `w:CIDR` | IP 白名单（单个 IP 或 CIDR，可重复使用） |
| `co` | 添加 `Access-Control-Allow-Origin: *`（CORS） |
| `x:https` | 强制 HTTPS——自动将 HTTP 重定向到 HTTPS |
| `a:Name:Value` | 添加请求头 |
| `u:Name:Value` | 更新请求头 |
| `r:Name` | 删除请求头 |
| `qr` | 将 URL 的二维码打印到 stdout（便于移动端分享） |

可自由组合：`"b:admin:secret+co+x:https+free@a.pinggy.io"`。

## Web 调试器（可选）

Pinggy 可将入站流量镜像到 `localhost:4300` 以供检查。在 SSH 命令中添加本地转发：

```bash
ssh -p 443 -L4300:localhost:4300 -R0:localhost:8000 free@a.pinggy.io
```

然后在浏览器中打开 `http://localhost:4300`，查看实时请求/响应对。

## 注意事项

- **免费套餐有 60 分钟硬性限制。** SSH 会话在 60 分钟时终止，URL 失效。如需更长时间的共享，使用 `PINGGY_TOKEN`（Pro）或用 shell 循环自动重启（注意免费套餐每次重启 URL 都会变化）。
- **免费套餐 URL 是随机的，重启后会变化。** 不要收藏，不要粘贴到配置文件中。每次都从日志重新解析。
- **同一源 IP 的并发免费隧道限制为一个。** 从同一台机器启动第二个隧道通常会终止第一个。Pro 套餐取消此限制。
- **用户名中的 `+` 必须加引号。** 裸命令 `ssh ... b:admin:secret+free@a.pinggy.io` 在 bash 中可以工作，但在将 `+` 视为特殊字符的 shell 中或以编程方式组装时会出错。始终用双引号括起。
- **不加访问控制标志不要隧道任何敏感内容。** 裸 HTTP 隧道对任何知道 URL 的人都可访问。对非公开服务使用 `b:`、`k:` 或 `w:`。
- **`process(action='log')` 可能会遗漏 SSH banner 输出。** Pinggy 打印 URL 后 SSH 会话进入交互模式。始终重定向到日志文件并直接 `grep` 文件——与 `cloudflared-quick-tunnel` 相同的模式。
- **首次运行时的主机密钥提示。** 默认 OpenSSH 配置会要求用户接受 Pinggy 的主机密钥。无人值守运行时始终传入 `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`。
- **TCP 和 TLS 隧道返回 `<subdomain>.a.pinggy.online:<port>` 对，而非 https URL。** 使用不同的正则表达式解析（`tcp://` 加端口）。不要假设每个 Pinggy 隧道都是 HTTP。
- **Pro 模式需要将 token 作为用户名，而非标志。** 使用 `"$PINGGY_TOKEN+a.pinggy.io"`（无 `free@`）。使用 token 还可以添加 `:persistent` 获得稳定子域名——参见 `pinggy.io/docs/`。

## 示例配方

将本地源服务与 Pinggy 隧道结合的复合模式。每个配方均自包含——启动源服务、启动隧道、解析 URL、返回给用户。

### 配方 1——接收 webhook 回调

当外部服务（Stripe、GitHub、Discord、AgentMail 等）需要在本地任务期间 POST 到公开可达的 URL 时使用。

```bash
# 1. 简易捕获服务器：每个请求都追加到 /tmp/webhook-hits.log
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

# 2. 隧道——使用 bearer token 门控，防止无关请求污染捕获日志
nohup ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:18080 "k:$(openssl rand -hex 12)+free@a.pinggy.io" \
    >/tmp/webhook-pinggy.log 2>&1 &
echo $! >/tmp/webhook-pinggy.pid
sleep 5
URL=$(grep -oE 'https://[a-z0-9-]+\.[a-z]+\.pinggy\.link' /tmp/webhook-pinggy.log | head -1)
echo "Webhook URL: $URL"

# 3. 在 agent 工作期间，监视请求到达
tail -f /tmp/webhook-hits.log
```

将 `$URL` 提供给需要调用你的服务。关闭：`kill $(cat /tmp/webhook-server.pid) $(cat /tmp/webhook-pinggy.pid)`。

### 配方 2——通过 HTTP/SSE 暴露 MCP 服务器

当远程 MCP 客户端（另一台机器上的 Claude Desktop、队友的编辑器等）需要访问本地运行的 MCP 服务器时使用。仅适用于使用 HTTP transport 的 MCP 服务器——stdio 模式的服务器无法被隧道。

```bash
# 1. 以 HTTP 模式启动 MCP 服务器（示例：端口 8765 上的 FastMCP 服务器）
nohup python3 my_mcp_server.py --transport http --port 8765 \
    >/tmp/mcp-server.log 2>&1 &
echo $! >/tmp/mcp-server.pid

# 2. 使用 bearer token 建立隧道——MCP 流量不应对互联网开放
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

远程客户端使用 `Authorization: Bearer $TOKEN` 连接到 `$URL`。Hermes 原生 MCP 客户端配置：`{"transport": "http", "url": "<URL>", "headers": {"Authorization": "Bearer <TOKEN>"}}`。

### 配方 3——暴露本地 LLM 端点（Ollama / vLLM / llama.cpp）

与远程调用方（另一个 agent、手机、队友）共享本地模型。Ollama 监听 `:11434`，vLLM 和 llama.cpp 通常监听 `:8000`。

```bash
# 前提：模型服务器已在 127.0.0.1:11434 上运行（Ollama 默认端口）
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

# 验证
curl -s "$URL/api/tags" -H "Authorization: Bearer $TOKEN" | head
```

`co` 启用 CORS，使浏览器调用方可以访问端点。纯后端调用方可去掉 `co`。对于兼容 OpenAI 的 vLLM/llama.cpp 端点，调用方使用基础 URL `$URL/v1` 加 `Authorization: Bearer $TOKEN`——但请注意 Pinggy 不会修改请求体中的任何内容，因此本地服务器实际上会看到 Pinggy 的 token；本地服务器应配置为忽略认证（它已在 `127.0.0.1` 上），让 Pinggy 负责门控。

### 配方 4——用一次性密码共享开发服务器

最快的"让队友访问我正在运行的应用"模式。随机密码，打印一次，Ctrl-C 后终止。

```bash
PASS=$(openssl rand -base64 12 | tr -d '+/=' | head -c 12)
echo "Dev server password: $PASS"
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:3000 "b:dev:$PASS+co+x:https+free@a.pinggy.io"
# URL 打印到终端。分享 URL + 密码。Ctrl-C 关闭隧道。
```

`b:dev:$PASS` 使用 HTTP Basic auth 对 URL 进行门控。`x:https` 强制 TLS。`co` 为 SPA 前端添加 CORS。

## 验证

```bash
# 端到端：启动一个简单的源服务，建立隧道，访问它，然后关闭
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

预期结果：一个 `pinggy.link` URL 以及 curl 返回的 `HTTP/2 200`。