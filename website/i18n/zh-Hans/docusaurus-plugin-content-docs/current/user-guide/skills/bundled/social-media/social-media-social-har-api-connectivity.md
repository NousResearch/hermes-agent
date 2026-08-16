---
title: "Social HAR API Connectivity — 通过驱驱动 Chrome 捕获登录流程和网络流量来连接社交平台 API"
sidebar_label: "Social HAR API Connectivity"
description: "通过驱动 Chrome 捕获登录流程和网络流量来连接社交平台 API — 用户登录，代理捕获会话（仅授权使用）。"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 SKILL.md 自动生成。修改源 SKILL.md 而非此页面。 */}

# Social HAR API Connectivity

通过 Chrome DevTools Protocol (CDP) 驱动 Chrome，在用户登录时捕获网络流量，提取会话
令牌，并构建可重用的 API 客户端，将代理连接到任意社交平台的 API。

**这是一个互动工作流：代理编排，用户认证。**

## 工作流程
1. 代理提示用户："您想连接哪个社交平台？"
2. 代理启动 Chrome（可见模式）并导航至登录页面。
3. 代理通知用户登录页面已打开 — 用户在浏览器窗口中输入凭据并处理任何 MFA/CAPTCHA。
4. 代理通过 CDP 监控网络流量并检测登录后的重定向。
5. 代理停止捕获，提取会话 cookie、认证令牌和 API 主机，将数据保存至临时目录（chmod 600）。
6. 代理确认连接成功并构建可重用的发布客户端。

## 随附工具
`scripts/chrome_capture_client.py` — 一个自动执行 Chrome CDP 捕获的 Python 脚本。
代理为用户运行此脚本。

## 支持的平台（登录 URL）
| 平台 | 登录 URL | 优先使用官方 API？ |
|---|---|---|
| Bluesky | https://bsky.app/login | 是 — App Password + AT Protocol |
| Mastodon | [实例]/auth/sign_in | 是 — 令牌认证 |
| X/Twitter | https://x.com/login | 用于 API 层级未覆盖的端点 |
| LinkedIn | https://www.linkedin.com/login | 是 — OAuth |
| Instagram | https://www.instagram.com/accounts/login/ | 是 — Meta Graph API |
| Facebook | https://www.facebook.com/login | 是 — Meta Graph API |
| TikTok | https://www.tiktok.com/login | 反机器人措施强 — 可能不稳定 |
| Reddit | https://www.reddit.com/login | 是 — OAuth 脚本应用 |
| Pinterest | https://www.pinterest.com/login | 是 — 官方 API（如已批准） |
| Threads | https://www.threads.net/login | 用于未文档化的端点 |
| YouTube | https://accounts.google.com/ | 是 — YouTube Data API |

## 陷阱
- MFA 是预期的 — 可见的 Chrome 窗口用于让用户完成双重认证。
- 会话令牌会过期（数小时至数天） — 过期后重新捕获。
- 令牌本地存储 — 绝不硬编码，绝不提交。
- TikTok 的反机器人措施很严格 — 捕获可能失败或会话可能快速过期。
- 必须安装 `websockets` Python 包（`pip install websockets`）。