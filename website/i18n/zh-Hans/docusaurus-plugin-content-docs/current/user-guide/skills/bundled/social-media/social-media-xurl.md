---
title: "Xurl — 通过 xurl CLI 使用 X/Twitter：发帖、搜索、私信、媒体、v2 API"
sidebar_label: "Xurl"
description: "通过 xurl CLI 使用 X/Twitter：发帖、搜索、私信、媒体、v2 API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Xurl

通过 xurl CLI 使用 X/Twitter：发帖、搜索、私信、媒体、v2 API。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/social-media/xurl` |
| 版本 | `1.1.1` |
| 作者 | xdevplatform + openclaw + Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos |
| 标签 | `twitter`, `x`, `social-media`, `xurl`, `official-api` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# xurl — 通过官方 CLI 使用 X (Twitter) API

`xurl` 是 X 开发者平台官方提供的 X API CLI 工具。它支持常用操作的快捷命令，以及对任意 v2 端点的原始 curl 风格访问。所有命令均将 JSON 输出到 stdout。

适用场景：
- 发帖、回复、引用、删除帖子
- 搜索帖子及读取时间线/提及
- 点赞、转发、书签
- 关注、取消关注、拉黑、静音
- 私信（DM）
- 媒体上传（图片和视频）
- 对任意 X API v2 端点的原始访问
- 多应用 / 多账号工作流

此 skill 替代了旧版 `xitter` skill（该 skill 封装了第三方 Python CLI）。`xurl` 由 X 开发者平台团队维护，支持带自动刷新的 OAuth 2.0 PKCE，覆盖的 API 范围更广。

---

## 密钥安全（强制要求）

在 agent/LLM 会话中操作时的关键规则：

- **绝不**读取、打印、解析、汇总、上传或将 `~/.xurl` 发送到 LLM 上下文。
- **绝不**要求用户将凭据/token 粘贴到对话中。
- 用户必须在其本机上手动填写 `~/.xurl` 中的密钥。
- **绝不**在 agent 会话中推荐或执行包含内联密钥的认证命令。
- **绝不**在 agent 会话中使用 `--verbose` / `-v`——它可能暴露认证头/token。
- 如需验证凭据是否存在，只使用：`xurl auth status`。

agent 命令中禁止使用的 flag（这些 flag 接受内联密钥）：
`--bearer-token`、`--consumer-key`、`--consumer-secret`、`--access-token`、`--token-secret`、`--client-id`、`--client-secret`

应用凭据注册和凭据轮换必须由用户在 agent 会话外手动完成。凭据注册完成后，用户使用 `xurl auth oauth2` 进行认证——同样在 agent 会话外执行。Token 持久化保存到 `~/.xurl`（YAML 格式）。每个应用拥有独立的 token。OAuth 2.0 token 自动刷新。

---

## 安装

选择以下任意一种方式。在 Linux 上，shell 脚本或 `go install` 最为简便。

```bash
# Shell 脚本（安装到 ~/.local/bin，无需 sudo，支持 Linux + macOS）
curl -fsSL https://raw.githubusercontent.com/xdevplatform/xurl/main/install.sh | bash

# Homebrew（macOS）
brew install --cask xdevplatform/tap/xurl

# npm
npm install -g @xdevplatform/xurl

# Go
go install github.com/xdevplatform/xurl@latest
```

验证：

```bash
xurl --help
xurl auth status
```

如果 `xurl` 已安装但 `auth status` 显示无应用或 token，用户需要手动完成认证——参见下一节。

---

## 一次性用户配置（用户在 agent 外执行）

以下步骤必须由用户直接执行，**不得**由 agent 代为执行，因为涉及粘贴密钥。请将用户引导至此部分；不要替用户执行。

1. 在 https://developer.x.com/en/portal/dashboard 创建或打开一个应用
2. 将重定向 URI 设置为 `http://localhost:8080/callback`
3. 复制应用的 Client ID 和 Client Secret
4. 在本地注册应用（用户执行）：
   ```bash
   xurl auth apps add my-app --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
   ```
5. 进行认证（指定 `--app` 将 token 绑定到你的应用）：
   ```bash
   xurl auth oauth2 --app my-app
   ```
   （这将打开浏览器进行 OAuth 2.0 PKCE 流程。）

   如果 X 在 OAuth 后的 `/2/users/me` 查询中返回 `UsernameNotFound` 错误或 403，请显式传入你的用户名（xurl v1.1.0+）：
   ```bash
   xurl auth oauth2 --app my-app YOUR_USERNAME
   ```
   这会将 token 绑定到你的用户名，并跳过有问题的 `/2/users/me` 调用。
6. 将该应用设为默认，使所有命令都使用它：
   ```bash
   xurl auth default my-app
   ```
7. 验证：
   ```bash
   xurl auth status
   xurl whoami
   ```

完成后，agent 即可使用以下所有命令，无需进一步配置。OAuth 2.0 token 自动刷新。

> **常见陷阱：** 如果在 `xurl auth oauth2` 时省略了 `--app my-app`，OAuth token 将保存到内置的 `default` 应用配置中——该配置没有 client-id 或 client-secret。即使 OAuth 流程看似成功，命令也会因认证错误而失败。如遇此情况，请重新运行 `xurl auth oauth2 --app my-app` 和 `xurl auth default my-app`。

---

## 快速参考

| 操作 | 命令 |
| --- | --- |
| 发帖 | `xurl post "Hello world!"` |
| 回复 | `xurl reply POST_ID "Nice post!"` |
| 引用 | `xurl quote POST_ID "My take"` |
| 删除帖子 | `xurl delete POST_ID` |
| 读取帖子 | `xurl read POST_ID` |
| 搜索帖子 | `xurl search "QUERY" -n 10` |
| 查看自己 | `xurl whoami` |
| 查找用户 | `xurl user @handle` |
| 主页时间线 | `xurl timeline -n 20` |
| 提及 | `xurl mentions -n 10` |
| 点赞 / 取消点赞 | `xurl like POST_ID` / `xurl unlike POST_ID` |
| 转发 / 撤销转发 | `xurl repost POST_ID` / `xurl unrepost POST_ID` |
| 书签 / 移除书签 | `xurl bookmark POST_ID` / `xurl unbookmark POST_ID` |
| 列出书签 / 点赞 | `xurl bookmarks -n 10` / `xurl likes -n 10` |
| 关注 / 取消关注 | `xurl follow @handle` / `xurl unfollow @handle` |
| 正在关注 / 粉丝 | `xurl following -n 20` / `xurl followers -n 20` |
| 拉黑 / 取消拉黑 | `xurl block @handle` / `xurl unblock @handle` |
| 静音 / 取消静音 | `xurl mute @handle` / `xurl unmute @handle` |
| 发送私信 | `xurl dm @handle "message"` |
| 列出私信 | `xurl dms -n 10` |
| 上传媒体 | `xurl media upload path/to/file.mp4` |
| 媒体状态 | `xurl media status MEDIA_ID` |
| 列出应用 | `xurl auth apps list` |
| 移除应用 | `xurl auth apps remove NAME` |
| 设置默认应用 | `xurl auth default APP_NAME [USERNAME]` |
| 单次请求指定应用 | `xurl --app NAME /2/users/me` |
| 认证状态 | `xurl auth status` |

注意：
- `POST_ID` 也接受完整 URL（如 `https://x.com/user/status/1234567890`）——xurl 会自动提取 ID。
- 用户名可带或不带前缀 `@`。

---

## 命令详情

### 发帖

```bash
xurl post "Hello world!"
xurl post "Check this out" --media-id MEDIA_ID
xurl post "Thread pics" --media-id 111 --media-id 222

xurl reply 1234567890 "Great point!"
xurl reply https://x.com/user/status/1234567890 "Agreed!"
xurl reply 1234567890 "Look at this" --media-id MEDIA_ID

xurl quote 1234567890 "Adding my thoughts"
xurl delete 1234567890
```

### 读取与搜索

```bash
xurl read 1234567890
xurl read https://x.com/user/status/1234567890

xurl search "golang"
xurl search "from:elonmusk" -n 20
xurl search "#buildinpublic lang:en" -n 15
```

### 用户、时间线、提及

```bash
xurl whoami
xurl user elonmusk
xurl user @XDevelopers

xurl timeline -n 25
xurl mentions -n 20
```

### 互动

```bash
xurl like 1234567890
xurl unlike 1234567890

xurl repost 1234567890
xurl unrepost 1234567890

xurl bookmark 1234567890
xurl unbookmark 1234567890

xurl bookmarks -n 20
xurl likes -n 20
```

### 社交关系

```bash
xurl follow @XDevelopers
xurl unfollow @XDevelopers

xurl following -n 50
xurl followers -n 50

# 查看其他用户的关系
xurl following --of elonmusk -n 20
xurl followers --of elonmusk -n 20

xurl block @spammer
xurl unblock @spammer
xurl mute @annoying
xurl unmute @annoying
```

### 私信

```bash
xurl dm @someuser "Hey, saw your post!"
xurl dms -n 25
```

### 媒体上传

```bash
# 自动检测类型
xurl media upload photo.jpg
xurl media upload video.mp4

# 显式指定类型/分类
xurl media upload --media-type image/jpeg --category tweet_image photo.jpg

# 视频需要服务端处理——检查状态（或轮询）
xurl media status MEDIA_ID
xurl media status --wait MEDIA_ID

# 完整工作流
xurl media upload meme.png                  # 返回 media id
xurl post "lol" --media-id MEDIA_ID
```

---

## 原始 API 访问

快捷命令覆盖了常用操作。对于其他需求，可使用原始 curl 风格模式访问任意 X API v2 端点：

```bash
# GET
xurl /2/users/me

# POST，带 JSON body
xurl -X POST /2/tweets -d '{"text":"Hello world!"}'

# DELETE / PUT / PATCH
xurl -X DELETE /2/tweets/1234567890

# 自定义请求头
xurl -H "Content-Type: application/json" /2/some/endpoint

# 强制流式传输
xurl -s /2/tweets/search/stream

# 完整 URL 同样有效
xurl https://api.x.com/2/users/me
```

---

## 全局 Flag

| Flag | 简写 | 说明 |
| --- | --- | --- |
| `--app` | | 使用指定的已注册应用（覆盖默认值） |
| `--auth` | | 强制指定认证类型：`oauth1`、`oauth2` 或 `app` |
| `--username` | `-u` | 指定使用哪个 OAuth2 账号（存在多个时） |
| `--verbose` | `-v` | **agent 会话中禁止使用**——会泄露认证头 |
| `--trace` | `-t` | 添加 `X-B3-Flags: 1` 追踪请求头 |

---

## 流式传输

流式端点会被自动检测。已知的流式端点包括：

- `/2/tweets/search/stream`
- `/2/tweets/sample/stream`
- `/2/tweets/sample10/stream`

对任意端点使用 `-s` 强制启用流式传输。

---

## 输出格式

所有命令将 JSON 输出到 stdout。结构与 X API v2 保持一致：

```json
{ "data": { "id": "1234567890", "text": "Hello world!" } }
```

错误同样以 JSON 形式输出：

```json
{ "errors": [ { "message": "Not authorized", "code": 403 } ] }
```

---

## 常见工作流

### 发布带图片的帖子
```bash
xurl media upload photo.jpg
xurl post "Check out this photo!" --media-id MEDIA_ID
```

### 回复某个对话
```bash
xurl read https://x.com/user/status/1234567890
xurl reply 1234567890 "Here are my thoughts..."
```

### 搜索并互动
```bash
xurl search "topic of interest" -n 10
xurl like POST_ID_FROM_RESULTS
xurl reply POST_ID_FROM_RESULTS "Great point!"
```

### 查看自己的动态
```bash
xurl whoami
xurl mentions -n 20
xurl timeline -n 20
```

### 多应用（凭据已手动预配置）
```bash
xurl auth default prod alice               # prod 应用，alice 用户
xurl --app staging /2/users/me             # 单次请求使用 staging
```

---

## 错误处理

- 任何错误均返回非零退出码。
- API 错误仍以 JSON 形式打印到 stdout，可直接解析。
- 认证错误 → 让用户在 agent 会话外重新运行 `xurl auth oauth2`。
- 需要调用方用户 ID 的命令（点赞、转发、书签、关注等）会通过 `/2/users/me` 自动获取。该处的认证失败会以认证错误的形式呈现。

---

## Agent 工作流

1. 验证前置条件：`xurl --help` 和 `xurl auth status`。
2. **检查默认应用是否有凭据。** 解析 `auth status` 输出。默认应用以 `▸` 标记。如果默认应用显示 `oauth2: (none)`，但另一个应用有有效的 oauth2 用户，请告知用户运行 `xurl auth default <that-app>` 修复。这是最常见的配置错误——用户添加了自定义名称的应用但从未将其设为默认，导致 xurl 一直尝试使用空的 `default` 配置。
3. 如果完全缺少认证，停止操作并将用户引导至"一次性用户配置"部分——不要尝试自行注册应用或传递密钥。
4. 先执行低成本的读取操作（`xurl whoami`、`xurl user @handle`、`xurl search ... -n 3`）以确认连通性。
5. 在执行任何写操作（发帖、回复、点赞、转发、私信、关注、拉黑、删除）前，确认目标帖子/用户及用户意图。
6. 直接使用 JSON 输出——每个响应均已结构化。
7. 绝不将 `~/.xurl` 内容粘贴回对话中。

---

## 故障排查

| 现象 | 原因 | 解决方法 |
| --- | --- | --- |
| OAuth 流程成功后仍出现认证错误 | Token 保存到了 `default` 应用（无 client-id/secret）而非命名应用 | 执行 `xurl auth oauth2 --app my-app`，然后 `xurl auth default my-app` |
| OAuth 期间出现 `unauthorized_client` | X 控制台中应用类型设置为"Native App" | 在用户认证设置中改为"Web app, automated app or bot" |
| OAuth 后 `/2/users/me` 返回 `UsernameNotFound` 或 403 | X 的 `/2/users/me` 返回用户名不稳定 | 重新运行 `xurl auth oauth2 --app my-app YOUR_USERNAME`（xurl v1.1.0+）显式传入用户名 |
| 每次请求均返回 401 | Token 已过期或默认应用错误 | 检查 `xurl auth status`——确认 `▸` 指向有 oauth2 token 的应用 |
| `client-forbidden` / `client-not-enrolled` | X 平台注册问题 | 控制台 → 应用 → 管理 → 切换到"Pay-per-use"套餐 → 生产环境 |
| `CreditsDepleted` | X API 余额为 $0 | 在开发者控制台 → 账单中充值（最低 $5） |
| 图片上传时 `media processing failed` | 默认分类为 `amplify_video` | 添加 `--category tweet_image --media-type image/png` |
| X 控制台中出现两个"Client Secret"值 | UI 问题——第一个实际上是 Client ID | 在"Keys and tokens"页面确认；ID 以 `MTpjaQ` 结尾 |

---

## 注意事项

- **速率限制：** X 对每个端点执行速率限制。429 表示需要等待后重试。写操作端点（发帖、回复、点赞、转发）的限制比读操作更严格。
- **权限范围（Scope）：** OAuth 2.0 token 使用宽泛的 scope。特定操作返回 403 通常意味着 token 缺少某个 scope——让用户重新运行 `xurl auth oauth2`。
- **Token 刷新：** OAuth 2.0 token 自动刷新，无需任何操作。
- **多应用：** 每个应用拥有独立的凭据/token。使用 `xurl auth default` 或 `--app` 切换。
- **每个应用的多账号：** 使用 `-u / --username` 选择，或通过 `xurl auth default APP USER` 设置默认值。
- **Token 存储：** `~/.xurl` 为 YAML 格式。绝不读取或将此文件发送到 LLM 上下文。
- **费用：** X API 访问在有实际使用量时通常需要付费。许多失败是套餐/权限问题，而非代码问题。

---

## 致谢

- 上游 CLI：https://github.com/xdevplatform/xurl（X 开发者平台团队，Chris Park 等）
- 上游 agent skill：https://github.com/openclaw/openclaw/blob/main/skills/xurl/SKILL.md
- Hermes 适配：按 Hermes skill 规范重新格式化；安全防护规则原文保留。