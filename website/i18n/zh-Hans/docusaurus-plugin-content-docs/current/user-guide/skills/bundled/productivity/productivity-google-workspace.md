---
title: "Google Workspace — 通过 gws CLI 或 Python 使用 Gmail、Calendar、Drive、Docs、Sheets"
sidebar_label: "Google Workspace"
description: "通过 gws CLI 或 Python 使用 Gmail、Calendar、Drive、Docs、Sheets"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Google Workspace

通过 gws CLI 或 Python 使用 Gmail、Calendar、Drive、Docs、Sheets。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/google-workspace` |
| 版本 | `1.1.0` |
| 作者 | Nous Research |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Google`, `Gmail`, `Calendar`, `Drive`, `Sheets`, `Docs`, `Contacts`, `Email`, `OAuth` |
| 相关 skill | [`himalaya`](/user-guide/skills/bundled/email/email-himalaya) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Google Workspace

Gmail、Calendar、Drive、Contacts、Sheets 和 Docs —— 通过 Hermes 管理的 OAuth（开放授权）和轻量 CLI 封装器实现。若已安装 `gws`，该 skill 将以其作为执行后端以获得更广泛的 Google Workspace 覆盖；否则回退到内置的 Python 客户端实现。

## 参考资料

- `references/gmail-search-syntax.md` —— Gmail 搜索运算符（is:unread、from:、newer_than: 等）

## 脚本

- `scripts/setup.py` —— OAuth2 设置（运行一次以完成授权）
- `scripts/google_api.py` —— 兼容性封装 CLI。在可用时优先使用 `gws` 执行操作，同时保留 Hermes 现有的 JSON 输出契约。

## 首次设置

设置过程完全非交互式 —— 你逐步驱动它，使其在 CLI、Telegram、Discord 或任何平台上均可正常工作。

首先定义一个简写：

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
```

### 步骤 0：检查是否已完成设置

```bash
$GSETUP --check
```

若输出 `AUTHENTICATED`，跳至「使用方法」—— 设置已完成。

### 步骤 1：分流 —— 询问用户需求

在开始 OAuth 设置之前，向用户提出**两个**问题：

**问题 1："你需要哪些 Google 服务？仅需邮件，还是还需要 Calendar/Drive/Sheets/Docs？"**

- **仅邮件** → 根本不需要此 skill。改用 `himalaya` skill —— 它通过 Gmail 应用专用密码（设置 → 安全 → 应用专用密码）工作，2 分钟即可完成设置，无需 Google Cloud 项目。加载 himalaya skill 并按其设置说明操作。

- **邮件 + Calendar** → 继续使用此 skill，但在授权时使用 `--services email,calendar`，使同意界面仅请求实际需要的权限范围（scope）。

- **仅 Calendar/Drive/Sheets/Docs** → 继续使用此 skill，并使用更窄的 `--services` 集合，如 `calendar,drive,sheets,docs`。

- **完整 Workspace 访问** → 继续使用此 skill，并使用默认的 `all` 服务集合。

**问题 2："你的 Google 账号是否启用了高级保护（登录时需要硬件安全密钥）？如果不确定，很可能没有 —— 这是需要你主动注册的功能。"**

- **否 / 不确定** → 正常设置，继续以下步骤。
- **是** → 其 Workspace 管理员必须先将 OAuth 客户端 ID 添加到组织的允许应用列表，步骤 4 才能成功。请提前告知用户。

### 步骤 2：创建 OAuth 凭据（一次性，约 5 分钟）

告知用户：

> 你需要一个 Google Cloud OAuth 客户端。这是一次性设置：
>
> 1. 创建或选择一个项目：
>    https://console.cloud.google.com/projectselector2/home/dashboard
> 2. 在 API 库中启用所需 API：
>    https://console.cloud.google.com/apis/library
>    启用：Gmail API、Google Calendar API、Google Drive API、
>    Google Sheets API、Google Docs API、People API
> 3. 在此处创建 OAuth 客户端：
>    https://console.cloud.google.com/apis/credentials
>    凭据 → 创建凭据 → OAuth 2.0 客户端 ID
> 4. 应用类型选择「桌面应用」→ 创建
> 5. 若应用仍处于测试状态，在此处将用户的 Google 账号添加为测试用户：
>    https://console.cloud.google.com/auth/audience
>    受众群体 → 测试用户 → 添加用户
> 6. 下载 JSON 文件并告诉我文件路径
>
> Hermes CLI 重要提示：若文件路径以 `/` 开头，请勿在 CLI 中单独发送该裸路径，因为它可能被误识别为斜杠命令。请将其放在句子中发送，例如：
> `The JSON file path is: /home/user/Downloads/client_secret_....json`

用户提供路径后：

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

若用户粘贴的是原始客户端 ID / 客户端密钥值而非文件路径，请自行为其编写一个有效的桌面 OAuth JSON 文件，保存到明确的位置（例如 `~/Downloads/hermes-google-client-secret.json`），然后对该文件运行 `--client-secret`。

### 步骤 3：获取授权 URL

使用步骤 1 中选择的服务集合。示例：

```bash
$GSETUP --auth-url --services email,calendar --format json
$GSETUP --auth-url --services calendar,drive,sheets,docs --format json
$GSETUP --auth-url --services all --format json
```

此命令返回包含 `auth_url` 字段的 JSON，并将该 URL 保存至 `~/.hermes/google_oauth_last_url.txt`。

本步骤的 Agent 规则：
- 提取 `auth_url` 字段，将该确切 URL 以单行形式发送给用户。
- 告知用户，批准后浏览器很可能会在 `http://localhost:1` 上失败，这是预期行为。
- 告知用户从浏览器地址栏复制**完整**的重定向 URL。
- 若用户收到 `Error 403: access_denied`，直接将其引导至 `https://console.cloud.google.com/auth/audience` 以添加自己为测试用户。

### 步骤 4：交换授权码

用户将粘贴回形如 `http://localhost:1/?code=4/0A...&scope=...` 的 URL 或仅粘贴授权码字符串，两者均可。`--auth-url` 步骤会在本地存储一个临时待处理的 OAuth 会话，以便 `--auth-code` 稍后完成 PKCE 交换，即使在无头系统上也可正常工作：

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED" --format json
```

若 `--auth-code` 因授权码过期、已被使用或来自旧浏览器标签页而失败，它现在会返回一个新的 `fresh_auth_url`。在这种情况下，立即将新 URL 发送给用户，并让其仅使用最新的浏览器重定向重试。

### 步骤 5：验证

```bash
$GSETUP --check
```

应输出 `AUTHENTICATED`。设置完成 —— 此后 token（令牌）将自动刷新。

### 注意事项

- Token 存储于 `~/.hermes/google_token.json`，自动刷新。
- 待处理的 OAuth 会话状态/验证器临时存储于 `~/.hermes/google_oauth_pending.json`，直至交换完成。
- 若已安装 `gws`，`google_api.py` 会将其指向同一个 `~/.hermes/google_token.json` 凭据文件。用户无需单独运行 `gws auth login` 流程。
- 撤销授权：`$GSETUP --revoke`

## 使用方法

所有命令均通过 API 脚本执行。将 `GAPI` 设为简写：

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
```

### Gmail

```bash
# 搜索（返回包含 id、from、subject、date、snippet 的 JSON 数组）
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# 读取完整邮件（返回包含正文文本的 JSON）
$GAPI gmail get MESSAGE_ID

# 发送
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
$GAPI gmail send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"

# 回复（自动归入同一会话线程并设置 In-Reply-To）
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# 标签
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# 列出事件（默认为未来 7 天）
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# 创建事件（需要带时区的 ISO 8601 格式）
$GAPI calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
$GAPI calendar create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# 删除事件
$GAPI calendar delete EVENT_ID
```

### Drive

```bash
# 搜索现有文件
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5

# 获取单个文件的元数据
$GAPI drive get FILE_ID

# 上传本地文件（自动检测 MIME 类型）
$GAPI drive upload /path/to/report.pdf
$GAPI drive upload /path/to/image.png --name "Logo.png" --parent FOLDER_ID

# 下载（二进制文件原样下载；Google 原生文件导出为合理的默认格式 ——
# Docs→pdf、Sheets→csv、Slides→pdf、Drawings→png）
$GAPI drive download FILE_ID
$GAPI drive download DOC_ID --output ~/doc.pdf
$GAPI drive download DOC_ID --export-mime text/plain --output ~/doc.txt

# 创建文件夹
$GAPI drive create-folder "Reports"
$GAPI drive create-folder "Q4" --parent FOLDER_ID

# 共享
$GAPI drive share FILE_ID --email alice@example.com --role reader
$GAPI drive share FILE_ID --email alice@example.com --role writer --notify
$GAPI drive share FILE_ID --type anyone --role reader        # 任何拥有链接的人
$GAPI drive share FILE_ID --type domain --domain example.com --role reader

# 删除 —— 默认移至回收站（可恢复）。使用 --permanent 跳过回收站。
$GAPI drive delete FILE_ID
$GAPI drive delete FILE_ID --permanent
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# 创建新电子表格
$GAPI sheets create --title "Q4 Budget"
$GAPI sheets create --title "Inventory" --sheet-name "Stock"

# 读取
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# 写入
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# 追加行
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
# 读取
$GAPI docs get DOC_ID

# 创建新文档（可选择以正文文本初始化）
$GAPI docs create --title "Meeting Notes"
$GAPI docs create --title "Draft" --body "First paragraph..."

# 在现有文档末尾追加文本
$GAPI docs append DOC_ID --text "Additional content to append"
```

## 输出格式

所有命令均返回 JSON。可使用 `jq` 解析或直接读取。关键字段：

- **Gmail search**：`[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**：`{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**：`{status: "sent", id, threadId}`
- **Calendar list**：`[{id, summary, start, end, location, description, htmlLink}]`
- **Calendar create**：`{status: "created", id, summary, htmlLink}`
- **Drive search**：`[{id, name, mimeType, modifiedTime, webViewLink}]`
- **Drive get**：`{id, name, mimeType, modifiedTime, size, webViewLink, parents, owners}`
- **Drive upload**：`{status: "uploaded", id, name, mimeType, webViewLink}`
- **Drive download**：`{status: "downloaded", id, name, path, mimeType}`
- **Drive create-folder**：`{status: "created", id, name, webViewLink}`
- **Drive share**：`{status: "shared", permissionId, fileId, role, type}`
- **Drive delete**：`{status: "trashed" | "deleted", fileId, permanent}`
- **Contacts list**：`[{name, emails: [...], phones: [...]}]`
- **Sheets get**：`[[cell, cell, ...], ...]`
- **Sheets create**：`{status: "created", spreadsheetId, title, spreadsheetUrl}`
- **Docs create**：`{status: "created", documentId, title, url}`
- **Docs append**：`{status: "appended", documentId, inserted_at, characters}`

## 规则

1. **未经用户确认，绝不发送邮件、创建/删除日历事件、删除 Drive 文件、共享文件或修改 Docs/Sheets。** 展示将要执行的操作（收件人、文件 ID、内容、共享角色）并请求批准。对于 `drive delete`，优先使用默认的回收站（可恢复）而非 `--permanent`。
2. **首次使用前检查授权** —— 运行 `setup.py --check`。若失败，引导用户完成设置。
3. **对于复杂查询，使用 Gmail 搜索语法参考** —— 通过 `skill_view("google-workspace", file_path="references/gmail-search-syntax.md")` 加载。
4. **Calendar 时间必须包含时区** —— 始终使用带偏移量的 ISO 8601 格式（如 `2026-03-01T10:00:00-06:00`）或 UTC（`Z`）。
5. **遵守速率限制** —— 避免快速连续的 API 调用。尽可能批量读取。

## 故障排查

| 问题 | 解决方法 |
|---------|-----|
| `NOT_AUTHENTICATED` | 执行上述设置步骤 2-5 |
| `REFRESH_FAILED` | Token 已被撤销或过期 —— 重新执行步骤 3-5 |
| `HttpError 403: Insufficient Permission` | 缺少 API scope —— `$GSETUP --revoke` 后重新执行步骤 3-5 |
| `AUTHENTICATED (partial)` 或「Token missing scopes」 | 新的写入功能（Drive 写入/删除、Docs 创建/编辑）需要重新授权。`$GSETUP --revoke` 后重新执行步骤 3-5 以授予升级后的 scope。 |
| `HttpError 403: Access Not Configured` | API 未启用 —— 用户需在 Google Cloud Console 中启用 |
| `ModuleNotFoundError` | 运行 `$GSETUP --install-deps` |
| 高级保护阻止授权 | Workspace 管理员必须将 OAuth 客户端 ID 加入白名单 |

## 撤销访问权限

```bash
$GSETUP --revoke
```