---
title: "Here.Now — 将静态站点发布到 {slug}"
sidebar_label: "Here.Now"
description: "将静态站点发布到 {slug}"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Here.Now

将静态站点发布到 &#123;slug&#125;.here.now，并将私有文件存储在云端 Drive 中，供 agent 间交接使用。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/productivity/here-now` 安装 |
| 路径 | `optional-skills/productivity/here-now` |
| 版本 | `1.15.3` |
| 作者 | here.now |
| 许可证 | MIT |
| 平台 | macos, linux |
| 标签 | `here.now`, `herenow`, `publish`, `deploy`, `hosting`, `static-site`, `web`, `share`, `URL`, `drive`, `storage` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# here.now

here.now 让 agent 能够发布网站并将私有文件存储在云端 Drive 中。

here.now 适用于两类任务：

- **Sites（站点）**：在 `{slug}.here.now` 发布网站和文件。
- **Drives（驱动器）**：在云端文件夹中存储 agent 私有文件。

## 当前文档

**在回答有关 here.now 功能、特性或工作流的问题之前，请先阅读当前文档：**

→ **https://here.now/docs**

在以下情况下阅读文档：

- 对话中首次出现与 here.now 相关的交互时
- 用户询问如何操作时
- 用户询问哪些功能可用、受支持或被推荐时
- 在告知用户某功能不受支持之前

需要参考当前文档的主题（不能仅依赖本地 skill 文本）：

- Drive 及 Drive 共享
- 自定义域名
- 付款与付款门控
- 分叉（forking）
- 代理路由（proxy routes）与服务变量
- 句柄（handles）与链接
- 限制与配额
- SPA 路由
- 错误处理与修复
- 功能可用性

**如果文档与实时 API 行为不一致，以实时 API 行为为准。**

如果文档获取失败或超时，继续使用本地 skill 和实时 API/脚本输出。对于活跃操作，优先以实时 API 行为为准。

## 依赖要求

- 必需的二进制文件：`curl`、`file`、`jq`
- 可选环境变量：`$HERENOW_API_KEY`
- 可选 Drive token 变量：`$HERENOW_DRIVE_TOKEN`
- 可选凭据文件：`~/.herenow/credentials`
- Skill 辅助脚本路径：
  - `${HERMES_SKILL_DIR}/scripts/publish.sh` 用于发布站点
  - `${HERMES_SKILL_DIR}/scripts/drive.sh` 用于私有 Drive 存储

## 创建站点

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --client hermes
```

输出实时 URL（例如 `https://bright-canvas-a7k2.here.now/`）。

底层流程分三步：创建/更新 -> 上传文件 -> 最终确认。站点在最终确认成功之前不会上线。

不使用 API key 时，将创建一个 **匿名站点**，24 小时后过期。
保存 API key 后，站点将永久保留。

**文件结构：** 对于 HTML 站点，请将 `index.html` 放在发布目录的根目录下，而非子目录中。目录内容将成为站点根目录。例如，发布 `my-site/`，其中存在 `my-site/index.html` — 不要发布包含 `my-site/` 的父目录。

也可以发布不含 HTML 的原始文件。单个文件会获得丰富的自动预览器（支持图片、PDF、视频、音频）。多个文件会自动生成带文件夹导航和图片画廊的目录列表。

## 更新已有站点

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --slug {slug} --client hermes
```

更新匿名站点时，脚本会自动从 `.herenow/state.json` 加载 `claimToken`。传入 `--claim-token {token}` 可覆盖此值。

已认证的更新需要保存的 API key。

## 使用 Drive

当用户需要为 agent 文件提供私有云存储时，使用 Drive：文档、上下文、记忆、计划、资产、媒体、研究、代码，以及任何需要持久化但不作为网站发布的内容。

每个已登录账户都有一个名为 `My Drive` 的默认 Drive。

```bash
DRIVE="${HERMES_SKILL_DIR}/scripts/drive.sh"
bash "$DRIVE" default
bash "$DRIVE" ls "My Drive"
bash "$DRIVE" put "My Drive" notes/today.md --from ./notes/today.md
bash "$DRIVE" cat "My Drive" notes/today.md
bash "$DRIVE" share "My Drive" --perms write --prefix notes/ --ttl 7d
```

使用有范围限制的 Drive token 进行 agent 间交接。如果收到 `herenow_drive` 共享块，将其 `token` 作为 `Authorization: Bearer <token>` 用于 `api_base`，存在 `pathPrefix` 时须遵守，写入时保留 ETag。`pathPrefix` 为 `null` 表示完整 Drive 访问权限。如果 skill 可用，优先使用 `drive.sh`；否则直接调用列出的 API 操作。

## API key 存储

发布脚本按以下来源读取 API key（先匹配先用）：

1. `--api-key {key}` 标志（仅用于 CI/脚本场景 — 交互式使用时请避免）
2. `$HERENOW_API_KEY` 环境变量
3. `~/.herenow/credentials` 文件（推荐 agent 使用）

要存储 key，将其写入凭据文件：

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

**重要**：收到 API key 后，立即保存 — 自行运行上述命令。不要让用户手动运行。在交互式会话中避免通过 CLI 标志（如 `--api-key`）传递 key；凭据文件是首选存储方式。

切勿将凭据或本地状态文件（`~/.herenow/credentials`、`.herenow/state.json`）提交到源代码控制。

## 获取 API key

从匿名（24 小时）升级为永久站点：

1. 向用户询问其电子邮件地址。
2. 请求一次性登录码：

```bash
curl -sS https://here.now/api/auth/agent/request-code \
  -H "content-type: application/json" \
  -d '{"email": "user@example.com"}'
```

3. 告知用户："请查收来自 here.now 的登录码邮件，并将其粘贴到此处。"
4. 验证登录码并获取 API key：

```bash
curl -sS https://here.now/api/auth/agent/verify-code \
  -H "content-type: application/json" \
  -d '{"email":"user@example.com","code":"ABCD-2345"}'
```

5. 自行保存返回的 `apiKey`（不要让用户操作）：

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

## 状态文件

每次站点创建/更新后，脚本会将内容写入工作目录下的 `.herenow/state.json`：

```json
{
  "publishes": {
    "bright-canvas-a7k2": {
      "siteUrl": "https://bright-canvas-a7k2.here.now/",
      "claimToken": "abc123",
      "claimUrl": "https://here.now/claim?slug=bright-canvas-a7k2&token=abc123",
      "expiresAt": "2026-02-18T01:00:00.000Z"
    }
  }
}
```

在创建或更新站点之前，可以检查此文件以查找之前的 slug。
将 `.herenow/state.json` 视为内部缓存。
切勿将此本地文件路径作为 URL 呈现，也不要将其作为认证模式、过期时间或 claim URL 的可信来源。

## 向用户说明的内容

对于已发布的站点：

- 始终分享当前脚本运行输出的 `siteUrl`。
- 读取并遵循脚本 stderr 中的 `publish_result.*` 行以确定认证模式。
- 当 `publish_result.auth_mode=authenticated` 时：告知用户站点是**永久的**，已保存到其账户。无需 claim URL。
- 当 `publish_result.auth_mode=anonymous` 时：告知用户站点将在 **24 小时后过期**。分享 claim URL（如果 `publish_result.claim_url` 非空且以 `https://` 开头），以便用户永久保留。提醒用户 claim token 仅返回一次，无法找回。
- 切勿让用户查看 `.herenow/state.json` 以获取 claim URL 或认证状态。

对于 Drive：

- 不要将 Drive 文件描述为公开 URL。
- 告知用户 Drive 内容是私有的，除非通过有范围限制的 token 共享。
- 与其他 agent 共享访问权限时，优先使用具有窄 `pathPrefix` 和短 TTL 的有范围 token。

## publish.sh 选项

| 标志 | 说明 |
| ---------------------- | -------------------------------------------- |
| `--slug {slug}` | 更新已有站点而非创建新站点 |
| `--claim-token {token}` | 覆盖匿名更新的 claim token |
| `--title {text}` | 预览器标题（非 HTML 站点） |
| `--description {text}` | 预览器描述 |
| `--ttl {seconds}` | 设置过期时间（仅限已认证用户） |
| `--client {name}` | 用于归因的 agent 名称（如 `hermes`） |
| `--base-url {url}` | API 基础 URL（默认：`https://here.now`） |
| `--allow-nonherenow-base-url` | 允许向非默认 `--base-url` 发送认证信息 |
| `--api-key {key}` | API key 覆盖（优先使用凭据文件） |
| `--spa` | 启用 SPA 路由（对未知路径返回 index.html） |
| `--forkable` | 允许他人分叉此站点 |

## publish.sh 之外的功能

Drive 操作请使用 `drive.sh` 或 Drive API。对于更广泛的账户和站点管理 — 删除、元数据、密码、付款、域名、句柄、链接、变量、代理路由、分叉、复制等 — 请参阅当前文档：

→ **https://here.now/docs**

完整文档：https://here.now/docs