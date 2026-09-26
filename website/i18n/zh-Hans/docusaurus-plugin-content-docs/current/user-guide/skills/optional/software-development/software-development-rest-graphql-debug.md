---
title: "Rest Graphql Debug — 调试 REST/GraphQL API：状态码、认证、Schema、复现"
sidebar_label: "Rest Graphql Debug"
description: "调试 REST/GraphQL API：状态码、认证、Schema、复现"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Rest Graphql Debug

调试 REST/GraphQL API：状态码、认证、Schema、复现。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/software-development/rest-graphql-debug` 安装 |
| 路径 | `optional-skills/software-development/rest-graphql-debug` |
| 版本 | `1.2.0` |
| 作者 | eren-karakus0 |
| 许可证 | MIT |
| 标签 | `api`, `rest`, `graphql`, `http`, `debugging`, `testing`, `curl`, `integration` |
| 相关 skill | [`systematic-debugging`](/user-guide/skills/bundled/software-development/software-development-systematic-debugging)、[`test-driven-development`](/user-guide/skills/bundled/software-development/software-development-test-driven-development) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# API 测试与调试

通过 Hermes 工具驱动 REST 和 GraphQL 诊断 —— `terminal` 用于 `curl`，`execute_code` 用于 Python `requests`，`web_extract` 用于查阅厂商文档。在猜测修复方案之前，先隔离出故障层。

## 适用场景

- API 返回意外的状态码或响应体
- 认证（auth）失败（token 刷新后仍 401/403、OAuth、API key）
- Postman 中正常但代码中失败
- Webhook / 回调集成调试
- 构建或审查 API 集成测试
- 限流或分页问题

以下场景跳过本 skill（向上升级）：UI 渲染、DB 查询调优、DNS/防火墙基础设施。

## 核心原则

**先隔离层，再修复。** 200 OK 可能隐藏损坏的数据。500 可能掩盖一个字符的认证拼写错误。按顺序逐层排查，不要跳过任何步骤。

```
1. 连通性       → 能否访问到主机？
1.5 超时        → 连接慢还是读取慢？
2. TLS/SSL      → 证书是否有效且受信任？
3. 认证         → 凭据是否正确且未过期？
4. 请求格式     → payload 结构是否符合服务端预期？
5. 响应解析     → 代码是否能接受返回的内容？
6. 语义         → 数据含义是否符合我们的假设？
```

## 5 分钟快速上手

### 通过 terminal 调试 REST

```python
# 详细的请求/响应交互
terminal('curl -v https://api.example.com/users/1')

# 带 JSON 的 POST
terminal("""curl -X POST https://api.example.com/users \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"name":"test","email":"test@example.com"}'""")

# 仅查看响应头
terminal('curl -sI https://api.example.com/health')

# 格式化输出 JSON
terminal('curl -s https://api.example.com/users | python3 -m json.tool')
```

### 通过 terminal 调试 GraphQL

```python
terminal("""curl -X POST https://api.example.com/graphql \\
  -H 'Content-Type: application/json' \\
  -H "Authorization: Bearer $TOKEN" \\
  -d '{"query":"{ user(id: 1) { name email } }"}'""")
```

**GraphQL 注意事项：** 即使查询失败，服务端通常也会返回 HTTP 200。无论状态码如何，始终检查 `errors` 字段：

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

### 通过 execute_code 使用 Python（requests）

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

## 分层调试流程

### 第 1 步 — 连通性

```python
terminal('nslookup api.example.com')
terminal('curl -v --connect-timeout 5 https://api.example.com/health')
```

常见故障：DNS 无法解析、防火墙、需要 VPN、缺少代理。

### 第 1.5 步 — 超时

区分*无法到达*与*到达但响应慢*：

```python
terminal('''curl -w "dns:%{time_namelookup}s connect:%{time_connect}s tls:%{time_appconnect}s ttfb:%{time_starttransfer}s total:%{time_total}s\\n" \\
  -o /dev/null -s https://api.example.com/endpoint''')
```

在 Python 中，始终传入元组超时 —— `requests` 没有默认值，会永久挂起：

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

诊断：`time_connect` 高说明是网络/防火墙问题；`time_connect` 低但 `time_starttransfer` 高说明是服务端响应慢。

### 第 2 步 — TLS/SSL

```python
terminal('curl -vI https://api.example.com 2>&1 | grep -E "SSL|subject|expire|issuer"')
```

常见故障：证书过期、自签名证书、主机名不匹配、缺少 CA bundle。`-k` 仅用于临时调试，不得写入代码。

### 第 3 步 — 认证

```python
# 检查 token 有效性
terminal('curl -s -o /dev/null -w "%{http_code}\\n" -H "Authorization: Bearer $TOKEN" https://api.example.com/me')

# 解码 JWT exp 声明 — 正确处理 base64url 填充
execute_code('''
import json, base64, os
tok = os.environ["TOKEN"]
payload = tok.split(".")[1]
payload += "=" * (-len(payload) % 4)
print(json.dumps(json.loads(base64.urlsafe_b64decode(payload)), indent=2))
''')
```

检查清单：
- Token 是否过期？（JWT 中的 `exp` 声明）
- 认证方案是否正确？Bearer vs Basic vs Token vs `X-Api-Key`
- 环境是否正确？将 Staging 的 key 用于 prod 是常见错误
- API key 是放在请求头还是查询参数（`?api_key=…`）中？

### 第 4 步 — 请求格式

```python
terminal("""curl -v -X POST https://api.example.com/endpoint \\
  -H 'Content-Type: application/json' \\
  -d '{"key":"value"}' 2>&1""")
```

**Content-Type 与请求体不匹配 —— 静默的 415/400：**

```python
# 错误 — data= 发送表单编码，但 header 声明 JSON
requests.post(url, data='{"k":"v"}', headers={"Content-Type": "application/json"})

# 正确 — json= 自动设置 header 并序列化
requests.post(url, json={"k": "v"})

# 错误 — Accept 声明 XML，代码却调用 .json()
requests.get(url, headers={"Accept": "text/xml"})

# 正确 — 让 requests 自动构建带 boundary 的 multipart
requests.post(url, files={"file": open("doc.pdf", "rb")})
```

常见问题：表单编码 vs JSON、缺少必填字段、HTTP 方法错误、查询参数未编码。

### 第 5 步 — 响应解析

调用 `.json()` 前始终检查 content-type：

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

常见故障：期望 JSON 却收到 HTML 错误页、响应体为空、字符集错误。

### 第 6 步 — 语义验证

解析成功 —— 但数据*正确*吗？

- `"status": "active"` 的含义是否符合代码预期？
- 响应中的 ID 是否与请求的 ID 一致？
- 时间戳是否在预期时区？
- 分页是否返回了全部结果，还是只有第 1 页？

## HTTP 状态码处理手册

### 401 Unauthorized — 凭据缺失或无效

1. `Authorization` 请求头是否实际存在？（用 `curl -v` 确认）
2. Token 是否正确且未过期？
3. 认证方案是否正确？（`Bearer` vs `Basic` vs `Token`）
4. 部分 API 使用查询参数（`?api_key=…`）而非请求头。

### 403 Forbidden — 已认证但无权限

1. Token 是否具有所需的 scope/权限？
2. 资源是否属于其他账户？
3. IP 白名单是否将你拦截？
4. 浏览器中的 CORS 问题？（检查 `Access-Control-Allow-Origin`）

### 404 Not Found — 资源不存在或 URL 错误

1. 路径是否正确？（末尾斜杠、拼写错误、版本前缀）
2. 资源 ID 是否存在？
3. API 版本是否正确（`/v1/` vs `/v2/`）？
4. Base URL 是否正确（staging vs prod）？

### 409 Conflict — 状态冲突

1. 资源是否已存在（重复创建）？
2. `ETag` / `If-Match` 是否过期？
3. 是否有其他进程并发修改？

### 422 Unprocessable Entity — JSON 合法但数据无效

错误响应体通常会指出有问题的字段。检查：
- 字段类型（string vs int、日期格式）
- 必填 vs 可选
- 枚举值是否在允许范围内

### 429 Too Many Requests — 触发限流

检查 `Retry-After` 和 `X-RateLimit-*` 响应头。指数退避：

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

### 5xx — 服务端问题，通常不是你的错

- **500** — 服务端 bug。记录 correlation ID，向服务商提交工单。
- **502** — 上游服务宕机。退避后重试。
- **503** — 过载 / 维护中。查看状态页。
- **504** — 上游超时。减小 payload 或增大超时时间。

所有 5xx：带抖动的退避重试，持续出现时发出告警。

## 分页与幂等性

**分页。** 确认你获取了*全部*结果。查找 `next_cursor`、`next_page`、`total_count`。两种常见模式：
- 偏移量（`?limit=100&offset=200`）—— 简单，但数据变动时可能跳过条目。
- 游标（`?cursor=abc123`）—— 适用于实时或大数据集，推荐使用。

**幂等性。** 对于非幂等操作（POST），发送 `Idempotency-Key: <uuid>`，确保重试不会重复扣款或重复创建。支付和订单场景必须使用。

## 契约验证

在进入生产前捕获 schema 漂移：

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

在 API 升级后、接入新第三方时，或在 CI 冒烟测试中运行。

## Correlation ID

始终记录服务商的请求 ID —— 这是联系厂商支持的最快途径：

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

**厂商 bug 报告模板：**

```
Endpoint:    POST /api/v1/orders
Request ID:  req_abc123xyz
Timestamp:   2026-03-17T14:30:00Z
Status:      500
Expected:    201 with order object
Actual:      500 {"error":"internal server error"}
Repro:       curl -X POST … (auth: <REDACTED>)
```

## 回归测试模板

将以下内容放入 `tests/` 目录，通过 `terminal('pytest tests/test_api_smoke.py -v')` 运行：

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

## 安全

### Token 处理
- 不要记录完整 token。脱敏处理：`Bearer <REDACTED>`。
- 不要在脚本中硬编码 token。从环境变量（`os.environ["API_TOKEN"]`）或 `~/.hermes/.env` 读取。
- 如果 token 出现在日志、错误信息或 git 历史中，立即轮换。

### 安全日志记录

```python
def redact_auth(headers: dict) -> dict:
    sensitive = {"authorization", "x-api-key", "cookie", "set-cookie"}
    return {k: ("<REDACTED>" if k.lower() in sensitive else v) for k, v in headers.items()}
```

### 泄露检查清单

- [ ] **URL 中的凭据。** 查询字符串中的 API key 会出现在服务器日志、浏览器历史、Referer 请求头中 —— 请使用请求头传递。
- [ ] **错误响应中的 PII。** `404 on /users/123` 不应暴露该用户是否存在（枚举攻击）。
- [ ] **生产环境中的堆栈跟踪。** 500 响应不应泄露文件路径、框架版本。
- [ ] **内部主机名/IP。** 错误响应体中出现 `10.x.x.x`、`internal-api.corp.local`。
- [ ] **Token 被回显。** 部分 API 会在错误详情中包含认证 token。请验证其不会如此。
- [ ] **冗余的 `Server` / `X-Powered-By`。** 技术栈信息泄露。记录以供安全审查。

## Hermes 工具使用模式

### terminal — 用于 curl、dig、openssl

```python
terminal('curl -sI https://api.example.com')
terminal('openssl s_client -connect api.example.com:443 -servername api.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates')
```

### execute_code — 用于多步骤 Python 流程

当调试跨越认证 → 请求 → 分页 → 验证多个环节时，使用 `execute_code`。变量在脚本内持久存在，结果打印到 stdout，不会在上下文中产生 token 污染：

```python
execute_code('''
import os, requests

token = os.environ["API_TOKEN"]
base  = "https://api.example.com"
H     = {"Authorization": f"Bearer {token}"}

# 1. 认证
me = requests.get(f"{base}/me", headers=H, timeout=10)
print(f"auth {me.status_code}")

# 2. 分页
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

### web_extract — 用于查阅厂商 API 文档

直接拉取你正在调试的端点的规范，而不是靠猜测：

```python
web_extract(urls=["https://docs.example.com/api/v1/users"])
```

### delegate_task — 用于完整的 CRUD 测试扫描

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

## 输出格式

报告调试结论时：

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

## 相关 Skill

- `systematic-debugging` —— 隔离出故障 API 层后，对代码进行根因分析
- `test-driven-development` —— 在发布修复前先编写回归测试