---
title: "Read X Articles — 无需 API 密钥，端到端阅读 X（Twitter）长文 Article"
sidebar_label: "Read X Articles"
description: "无需 API 密钥，端到端阅读 X（Twitter）长文 Article"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Read X Articles

无需 X API 密钥即可端到端阅读来自 x.com 链接的 X（Twitter）长文 Article。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/social-media/read-x-articles` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `x`, `twitter`, `articles`, `web`, `reading`, `long-form`, `web-extract` |

## 参考：完整 SKILL.md

:::info
以下为 Hermes 触发此 Skill 时加载的完整定义。
:::

# 阅读 X（Twitter）Article

把用户分享的任何 X 链接转换为完整文章正文。不要默认说「我读不了 X」——X 长文 Article 可以通过正确的 URL 和具备 JS 渲染能力的网页提取/浏览器工具顺利读取。

## 何时使用
- 用户给出 `x.com/...` 或 `twitter.com/...` 链接并要求阅读。
- 链接是 X **Article**（长文/访谈）——规范路径为 `/i/article/<ID>`。
- 需要把 X 长文作为工作素材（写作、视频脚本、分析）的原始出处。
- 有人说「X 内容读不了」——这正是行动信号，不是放弃的理由。

## 核心要点
- X **Article** 在规范 URL **`https://x.com/i/article/<ID>`** 下，由具备 JS 渲染能力的网页提取/浏览器工具（如 `web_extract`）读取时，能拿到**完整正文**。
- **裸 HTTP 请求（`urllib`/`curl`）往往会返回登录/JS 壳页面**，需要渲染工具。
- **`/status/<id>` 与个人主页**是 JS 渲染的，会遇到登录墙。解决方法是解析出 article URL 并使用正确的抓取工具。

## 步骤
1. **先试再说。** 拿到 X 链接立即用 `web_extract` 尝试。
2. **使用正确的抓取器：** 优先 `web_extract` 或浏览器渲染，而非裸 `curl`/`urllib`。
3. **若是 `/status/` URL 或返回登录/JS 壳**，解析出规范 article URL 后 `web_extract`：`https://x.com/i/article/<ID>`。
4. **校验正文**：真正的结果包含实质性段落，而不仅是「登录或注册 X」。
5. **备选：** 浏览器渲染（CDP 风格），或若账号有 X API 权限，用 `xurl` CLI 读取 `data.article.plain_text`。
6. **使用内容。** 这是合法的原始出处——引用并注明作者。

## 注意事项
- 不要假设 Article 被登录墙挡住；规范 `/i/article/<ID>` 对渲染工具是可达的。
- 裸 `curl`/`urllib` 返回登录壳页**不代表读不了**——改用 `web_extract`/浏览器。
- X API token 失效（401）与阅读 Article 无关——别因 `xurl` 鉴权失败而卡住任务。

## 验证
- 提取到的是文章真实正文还是壳页？是正文即视为已读。
- 报告所读的确切 URL 及简要要点，便于用户核对。