---
title: "Here.Now — 发布静态网站到 {slug}"
sidebar_label: "Here.Now"
description: "发布静态网站到 {slug}"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Here.Now

发布静态网站到 {slug}.here.now 并在云端 Drive 中存储私有文件，用于代理间交接。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/here-now` |
| Path | `optional-skills/productivity/here-now` |
| Version | `1.15.3` |
| Author | here.now |
| License | MIT |
| Platforms | macos, linux |
| Tags | `here.now`, `herenow`, `publish`, `deploy`, `hosting`, `static-site`, `web`, `share`, `URL`, `drive`, `storage` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# here.now

here.now 让代理发布网站并在云端 Drive 中存储私有文件。

将 here.now 用于两项工作：

- **站点**：在 `{slug}.here.now` 发布网站和文件。
- **Drive**：在云端文件夹中存储私有代理文件。

## 当前文档

**在回答关于 here.now 功能、特性或工作流程的问题之前，请阅读当前文档：**

→ **https://here.now/docs**

阅读文档的时机：

- 在对话中首次涉及 here.now 的交互时
- 用户询问如何执行某操作时
- 用户询问什么是可能的、支持的或推荐的时
- 在告诉用户某功能不支持之前

需要查阅当前文档的主题（不要仅依赖本地技能文本）：

- Drive 和 Drive 共享
- 自定义域名
- 支付和付费门控
- 分叉
- 代理路由和服务变量
- 句柄和链接
- 限制和配额
- SPA 路由
- 错误处理和修复
- 功能可用性

**如果文档和实时 API 行为不一致，请以实时 API 行为为准。**

如果文档获取失败或超时，请继续使用本地技能和实时 API/脚本输出。对于活跃操作，请优先使用实时 API 行为。

## 要求

- 必需的二进制文件：`curl`、`file`、`jq`
- 可选环境变量：`$HERENOW_API_KEY`
- 可选 Drive 令牌变量：`$HERENOW_DRIVE_TOKEN`
- 可选凭证文件：`~/.herenow/credentials`
- 技能辅助脚本路径：
  - `${HERMES_SKILL_DIR}/scripts/publish.sh` 用于发布站点
  - `${HERMES_SKILL_DIR}/scripts/drive.sh` 用于私有 Drive 存储

## 创建站点

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --client hermes
```

输出实时 URL（例如 `https://bright-canvas-a7k2.here.now/`）。

底层是三步流程：创建/更新 → 上传文件 → 完成。站点在完成步骤成功之前不会上线。

没有 API 密钥时，这将创建一个**匿名站点**，24 小时后过期。
保存了 API 密钥时，站点是永久的。

**文件结构：** 对于 HTML 站点，请将 `index.html` 放在发布目录的根目录中，而不是子目录内。目录内容将成为站点根目录。例如，发布 `my-site/` 目录（其中 `my-site/index.html` 存在）——不要发布包含 `my-site/` 的父文件夹。

你还可以发布没有任何 HTML 的原始文件。单个文件会获得富自动查看器（图片、PDF、视频、音频）。多个文件会获得自动生成的目录列表，包含文件夹导航和图片画廊。

## 更新现有站点

```bash
PUBLISH="${HERMES_SKILL_DIR}/scripts/publish.sh"
bash "$PUBLISH" {file-or-dir} --slug {slug} --client hermes
```

脚本在更新匿名站点时从 `.herenow/state.json` 自动加载 `claimToken`。传入 `--claim-token {token}` 可覆盖。

认证更新需要已保存的 API 密钥。

## 使用 Drive

当用户希望为代理文件提供私有云存储时使用 Drive：文档、上下文、记忆、计划、资产、媒体、研究、代码以及任何应持久化但不应作为网站发布的内容。

每个已登录账户都有一个名为 `My Drive` 的默认 Drive。

```bash
DRIVE="${HERMES_SKILL_DIR}/scripts/drive.sh"
bash "$DRIVE" default
bash "$DRIVE" ls "My Drive"
bash "$DRIVE" put "My Drive" notes/today.md --from ./notes/today.md
bash "$DRIVE" cat "My Drive" notes/today.md
bash "$DRIVE" share "My Drive" --perms write --prefix notes/ --ttl 7d
```

使用受限范围的 Drive 令牌进行代理间交接。如果你收到一个 `herenow_drive` 共享块，请使用其 `token` 作为 `Authorization: Bearer <token>` 访问 `api_base`，在存在 `pathPrefix` 时予以遵守，并在写入时保留 ETag。`pathPrefix` 为 `null` 表示完全 Drive 访问。如果技能可用，优先使用 `drive.sh`；否则直接调用列出的 API 操作。

## API 密钥存储

发布脚本从以下来源读取 API 密钥（第一个匹配的生效）：

1. `--api-key {key}` 标志（仅限 CI/脚本使用——避免在交互式会话中使用）
2. `$HERENOW_API_KEY` 环境变量
3. `~/.herenow/credentials` 文件（推荐用于代理）

要存储密钥，将其写入凭证文件：

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

**重要**：收到 API 密钥后，请立即保存——你自己运行上述命令。不要让用户手动运行。在交互式会话中避免通过 CLI 标志传递密钥（如 `--api-key`）；凭证文件是首选的存储方式。

切勿将凭证或本地状态文件（`~/.herenow/credentials`、`.herenow/state.json`）提交到版本控制中。

## 获取 API 密钥

要从匿名（24 小时）升级到永久站点：

1. 询问用户的电子邮件地址。
2. 请求一次性登录代码：

```bash
curl -sS https://here.now/api/auth/agent/request-code \
  -H "content-type: application/json" \
  -d '{"email": "user@example.com"}'
```

3. 告诉用户："请检查收件箱中来自 here.now 的登录代码并粘贴到此处。"
4. 验证代码并获取 API 密钥：

```bash
curl -sS https://here.now/api/auth/agent/verify-code \
  -H "content-type: application/json" \
  -d '{"email":"user@example.com","code":"ABCD-2345"}'
```

5. 自己保存返回的 `apiKey`（不要让用户执行此操作）：

```bash
mkdir -p ~/.herenow && echo "{API_KEY}" > ~/.herenow/credentials && chmod 600 ~/.herenow/credentials
```

## 状态文件

每次站点创建/更新后，脚本会写入工作目录中的 `.herenow/state.json`：

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

在创建或更新站点之前，你可以检查此文件以查找先前的 slug。
将 `.herenow/state.json` 仅视为内部缓存。
切勿将此本地文件路径作为 URL 展示，也切勿将其用作认证模式、过期或 claim URL 的事实来源。

## 如何告知用户

对于已发布的站点：

- 始终分享当前脚本运行产生的 `siteUrl`。
- 读取并遵循脚本 stderr 中的 `publish_result.*` 行来确定认证模式。
- 当 `publish_result.auth_mode=authenticated` 时：告诉用户站点是**永久的**并已保存到其账户。不需要 claim URL。
- 当 `publish_result.auth_mode=anonymous` 时：告诉用户站点**24 小时后过期**。分享 claim URL（如果 `publish_result.claim_url` 非空且以 `https://` 开头）以便他们可以永久保留。警告 claim 令牌仅返回一次且无法恢复。
- 切勿告诉用户检查 `.herenow/state.json` 获取 claim URL 或认证状态。

对于 Drive：

- 不要将 Drive 文件描述为公开 URL。
- 告诉用户 Drive 内容是私有的，除非通过受限令牌共享。
- 与其他代理共享访问时，优先使用带有狭窄 `pathPrefix` 和短 TTL 的受限令牌。

## publish.sh 选项

| 标志 | 描述 |
| ---------------------- | -------------------------------------------- |
| `--slug {slug}` | 更新现有站点而非创建新站点 |
| `--claim-token {token}` | 覆盖匿名更新的 claim 令牌 |
| `--title {text}` | 查看器标题（非 HTML 站点） |
| `--description {text}` | 查看器描述 |
| `--ttl {seconds}` | 设置过期时间（仅限认证站点） |
| `--client {name}` | 代理名称用于归因（如 `hermes`） |
| `--base-url {url}` | API 基础 URL（默认：`https://here.now`） |
| `--allow-nonherenow-base-url` | 允许向非默认 `--base-url` 发送认证 |
| `--api-key {key}` | API 密钥覆盖（优先使用凭证文件） |
| `--spa` | 启用 SPA 路由（对未知路径返回 index.html） |
| `--forkable` | 允许他人分叉此站点 |

## publish.sh 之外

对于 Drive 操作，使用 `drive.sh` 或 Drive API。对于更广泛的账户和站点管理——删除、元数据、密码、支付、域名、句柄、链接、变量、代理路由、分叉、复制等——请参阅当前文档：

→ **https://here.now/docs**

完整文档：https://here.now/docs
