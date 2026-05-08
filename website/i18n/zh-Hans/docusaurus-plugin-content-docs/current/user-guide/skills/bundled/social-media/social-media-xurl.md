---
title: "Xurl — 通过 xurl CLI 使用 X/Twitter：发帖、搜索、私信、媒体、v2 API"
sidebar_label: "Xurl"
description: "通过 xurl CLI 使用 X/Twitter：发帖、搜索、私信、媒体、v2 API"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Xurl

通过 xurl CLI 使用 X/Twitter：发帖、搜索、私信、媒体、v2 API。

## 技能元数据

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
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# xurl — 通过官方 CLI 使用 X (Twitter) API

`xurl` 是 X 开发者平台的官方 X API CLI 工具。它支持常见操作的快捷命令以及对任何 v2 端点的原始 curl 风格访问。所有命令返回 JSON 到标准输出。

使用此技能进行：
- 发帖、回复、引用、删除帖子
- 搜索帖子和阅读时间线/提及
- 点赞、转发、收藏
- 关注、取消关注、屏蔽、静音
- 私信
- 媒体上传（图片和视频）
- 对任何 X API v2 端点的原始访问
- 多应用/多账户工作流

此技能取代了旧版 `xitter` 技能（该技能封装了第三方 Python CLI）。`xurl` 由 X 开发者平台团队维护，支持带自动刷新的 OAuth 2.0 PKCE，并覆盖了更广泛的 API 表面。

---

## 密钥安全（强制）

在代理/LLM 会话中操作时的关键规则：

- **永远不要** 读取、打印、解析、摘要、上传或将 `~/.xurl` 发送到 LLM 上下文。
- **永远不要** 要求用户将凭据/令牌粘贴到聊天中。
- 用户必须在自己的机器上手动填写 `~/.xurl` 中的密钥。
- **永远不要** 在代理会话中推荐或执行包含内联密钥的认证命令。
- **永远不要** 在代理会话中使用 `--verbose` / `-v` ——它可能暴露认证头/令牌。
- 要验证凭据是否存在，只能使用：`xurl auth status`。

代理命令中禁止使用的标志（它们接受内联密钥）：
`--bearer-token`, `--consumer-key`, `--consumer-secret`, `--access-token`, `--token-secret`, `--client-id`, `--client-secret`

应用凭据注册和凭据轮换必须由用户在代理会话外手动完成。凭据注册后，用户在代理会话外使用 `xurl auth oauth2` 进行认证。令牌以 YAML 格式持久化到 `~/.xurl`。每个应用有隔离的令牌。OAuth 2.0 令牌自动刷新。

---

## 安装

选择一种方法。在 Linux 上，shell 脚本或 `go install` 最简单。

```bash
# Shell 脚本（安装到 ~/.local/bin，无需 sudo，支持 Linux + macOS）
curl -fsSL https://raw.githubusercontent.com/xdevplatform/xurl/main/install.sh | bash

# Homebrew (macOS)
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

如果 `xurl` 已安装但 `auth status` 显示没有应用或令牌，用户需要手动完成认证——请参阅下一节。

---

## 一次性用户设置（用户在代理外执行）

这些步骤必须由用户直接执行，而非代理，因为涉及粘贴密钥。将用户引导到此块；不要为他们执行。

1. 在 https://developer.x.com/en/portal/dashboard 创建或打开应用
2. 将重定向 URI 设置为 `http://localhost:8080/callback`
3. 复制应用的 Client ID 和 Client Secret
4. 在本地注册应用（用户执行此操作）：
   ```bash
   xurl auth apps add my-app --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
   ```
5. 认证（指定 `--app` 将令牌绑定到你的应用）：
   ```bash
   xurl auth oauth2 --app my-app
   ```
   （这将打开浏览器进行 OAuth 2.0 PKCE 流程。）

   如果 X 返回 `UsernameNotFound` 错误或在 OAuth 后 `/2/users/me` 查找返回 403，显式传递你的用户名（xurl v1.1.0+）：
   ```bash
   xurl auth oauth2 --app my-app YOUR_USERNAME
   ```
   这将令牌绑定到你的用户名并跳过损坏的 `/2/users/me` 调用。
6. 将应用设为默认，以便所有命令使用它：
   ```bash
   xurl auth default my-app
   ```
7. 验证：
   ```bash
   xurl auth status
   xurl whoami
   ```

此后，代理可以使用以下任何命令而无需进一步设置。OAuth 2.0 令牌自动刷新。

> **常见陷阱：** 如果你在 `xurl auth oauth2` 中省略了 `--app my-app`，OAuth 令牌将保存到内置的 `default` 应用配置中——该配置没有 client-id 或 client-secret。即使 OAuth 流程看起来成功了，命令也会因认证错误而失败。如果遇到此问题，重新运行 `xurl auth oauth2 --app my-app` 和 `xurl auth default my-app`。

---

## 快速参考

| 操作 | 命令 |
| --- | --- |
| 发帖 | `xurl post "Hello world!"` |
| 回复 | `xurl reply POST_ID "Nice post!"` |
| 引用 | `xurl quote POST_ID "My take"` |
| 删除帖子 | `xurl delete POST_ID` |
| 阅读帖子 | `xurl read POST_ID` |
| 搜索帖子 | `xurl search "QUERY" -n 10` |
| 我是谁 | `xurl whoami` |
| 查找用户 | `xurl user @handle` |
| 主页时间线 | `xurl timeline -n 20` |
| 提及 | `xurl mentions -n 10` |
| 点赞/取消点赞 | `xurl like POST_ID` / `xurl unlike POST_ID` |
| 转发/取消转发 | `xurl repost POST_ID` / `xurl unrepost POST_ID` |
| 收藏/取消收藏 | `xurl bookmark POST_ID` / `xurl unbookmark POST_ID` |
| 列出收藏/点赞 | `xurl bookmarks -n 10` / `xurl likes -n 10` |
| 关注/取消关注 | `xurl follow @handle` / `xurl unfollow @handle` |
| 正在关注/粉丝 | `xurl following -n 20` / `xurl followers -n 20` |
| 屏蔽/取消屏蔽 | `xurl block @handle` / `xurl unblock @handle` |
| 静音/取消静音 | `xurl mute @handle` / `xurl unmute @handle` |
| 发送私信 | `xurl dm @handle "message"` |
| 列出私信 | `xurl dms -n 10` |
| 上传媒体 | `xurl media upload path/to/file.mp4` |
| 媒体状态 | `xurl media status MEDIA_ID` |
| 列出应用 | `xurl auth apps list` |
| 删除应用 | `xurl auth apps remove NAME` |
| 设置默认应用 | `xurl auth default APP_NAME [USERNAME]` |
| 单次请求使用指定应用 | `xurl --app NAME /2/users/me` |
| 认证状态 | `xurl auth status` |

注意：
- `POST_ID` 也接受完整 URL（例如 `https://x.com/user/status/1234567890`）——xurl 会提取 ID。
- 用户名带或不带前导 `@` 都可以。

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

### 阅读与搜索

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

### 社交图谱

```bash
xurl follow @XDevelopers
xurl unfollow @XDevelopers

xurl following -n 50
xurl followers -n 50

# 其他用户的图谱
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

# 显式指定类型/类别
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

快捷命令覆盖常见操作。对于其他操作，使用原始 curl 风格模式访问任何 X API v2 端点：

```bash
# GET
xurl /2/users/me

# POST with JSON body
xurl -X POST /2/tweets -d '{"text":"Hello world!"}'

# DELETE / PUT / PATCH
xurl -X DELETE /2/tweets/1234567890

# 自定义请求头
xurl -H "Content-Type: application/json" /2/some/endpoint

# 强制流式传输
xurl -s /2/tweets/search/stream

# 完整 URL 也可以
xurl https://api.x.com/2/users/me
```

---

## 全局标志

| 标志 | 简写 | 描述 |
| --- | --- | --- |
| `--app` | | 使用特定的已注册应用（覆盖默认） |
| `--auth` | | 强制认证类型：`oauth1`、`oauth2` 或 `app` |
| `--username` | `-u` | 使用哪个 OAuth2 账户（如果有多个） |
| `--verbose` | `-v` | **代理会话中禁止** ——泄露认证头 |
| `--trace` | `-t` | 添加 `X-B3-Flags: 1` 追踪头 |

---

## 流式传输

流式端点会自动检测。已知的包括：

- `/2/tweets/search/stream`
- `/2/tweets/sample/stream`
- `/2/tweets/sample10/stream`

使用 `-s` 在任何端点上强制流式传输。

---

## 输出格式

所有命令返回 JSON 到标准输出。结构映射到 X API v2：

```json
{ "data": { "id": "1234567890", "text": "Hello world!" } }
```

错误也是 JSON：

```json
{ "errors": [ { "message": "Not authorized", "code": 403 } ] }
```

---

## 常见工作流

### 带图片发帖
```bash
xurl media upload photo.jpg
xurl post "Check out this photo!" --media-id MEDIA_ID
```

### 回复对话
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

### 检查你的活动
```bash
xurl whoami
xurl mentions -n 20
xurl timeline -n 20
```

### 多应用（凭据已手动预配置）
```bash
xurl auth default prod alice               # prod 应用，alice 用户
xurl --app staging /2/users/me             # 一次性对 staging 操作
```

---

## 错误处理

- 任何错误时返回非零退出码。
- API 错误仍以 JSON 形式打印到标准输出，因此你可以解析它们。
- 认证错误 → 让用户在代理会话外重新运行 `xurl auth oauth2`。
- 需要调用者用户 ID 的命令（点赞、转发、收藏、关注等）会通过 `/2/users/me` 自动获取。那里的认证失败会作为认证错误显示。

---

## 代理工作流

1. 验证前提条件：`xurl --help` 和 `xurl auth status`。
2. **检查默认应用是否有凭据。** 解析 `auth status` 输出。默认应用标有 `▸`。如果默认应用显示 `oauth2: (none)` 但另一个应用有有效的 oauth2 用户，告诉用户运行 `xurl auth default <that-app>` 来修复。这是最常见的设置错误——用户添加了一个自定义名称的应用但从未将其设为默认，所以 xurl 一直在尝试空的 `default` 配置。
3. 如果完全缺少认证，停止并引导用户到"一次性用户设置"部分——不要尝试自己注册应用或传递密钥。
4. 以一次廉价的读取（`xurl whoami`、`xurl user @handle`、`xurl search ... -n 3`）开始，确认可达性。
5. 在任何写入操作（发帖、回复、点赞、转发、私信、关注、屏蔽、删除）之前，确认目标帖子/用户和用户的意图。
6. 直接使用 JSON 输出——每个响应已经是结构化的。
7. 永远不要将 `~/.xurl` 的内容粘贴回对话中。

---

## 故障排除

| 症状 | 原因 | 修复 |
| --- | --- | --- |
| OAuth 成功后出现认证错误 | 令牌保存到了 `default` 应用（没有 client-id/secret）而不是你命名的应用 | `xurl auth oauth2 --app my-app` 然后 `xurl auth default my-app` |
| OAuth 期间 `unauthorized_client` | X 控制台中应用类型设为"Native App" | 在用户认证设置中改为"Web app, automated app or bot" |
| OAuth 后 `/2/users/me` 返回 `UsernameNotFound` 或 403 | X 无法从 `/2/users/me` 可靠返回用户名 | 重新运行 `xurl auth oauth2 --app my-app YOUR_USERNAME`（xurl v1.1.0+）显式传递用户名 |
| 每个请求返回 401 | 令牌过期或默认应用错误 | 检查 `xurl auth status` ——验证 `▸` 指向一个有 oauth2 令牌的应用 |
| `client-forbidden` / `client-not-enrolled` | X 平台注册问题 | 控制台 → 应用 → 管理 → 移至"Pay-per-use"套餐 → 生产环境 |
| `CreditsDepleted` | X API 余额为 $0 | 在开发者控制台 → 计费中购买额度（最低 $5） |
| 图片上传 `media processing failed` | 默认类别为 `amplify_video` | 添加 `--category tweet_image --media-type image/png` |
| X 控制台中有两个"Client Secret"值 | UI bug ——第一个实际上是 Client ID | 在"Keys and tokens"页面确认；ID 以 `MTpjaQ` 结尾 |

---

## 注意事项

- **速率限制：** X 对每个端点强制执行速率限制。429 表示等待并重试。写入端点（发帖、回复、点赞、转发）的限制比读取更严格。
- **权限范围：** OAuth 2.0 令牌使用广泛的权限范围。特定操作的 403 通常表示令牌缺少权限范围——让用户重新运行 `xurl auth oauth2`。
- **令牌刷新：** OAuth 2.0 令牌自动刷新。无需任何操作。
- **多应用：** 每个应用有隔离的凭据/令牌。使用 `xurl auth default` 或 `--app` 切换。
- **每应用多账户：** 使用 `-u / --username` 选择，或使用 `xurl auth default APP USER` 设置默认。
- **令牌存储：** `~/.xurl` 是 YAML 格式。永远不要读取或将此文件发送到 LLM 上下文。
- **费用：** X API 访问通常需要付费才能有意义地使用。很多失败是计划/权限问题，而非代码问题。

---

## 致谢

- 上游 CLI：https://github.com/xdevplatform/xurl（X 开发者平台团队，Chris Park 等）
- 上游代理技能：https://github.com/openclaw/openclaw/blob/main/skills/xurl/SKILL.md
- Hermes 适配：为 Hermes 技能约定重新格式化；安全护栏原样保留。
