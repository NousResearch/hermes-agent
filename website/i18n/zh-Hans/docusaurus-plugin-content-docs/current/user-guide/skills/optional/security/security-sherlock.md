---
title: "Sherlock — 跨 400+ 社交网络的 OSINT 用户名搜索"
sidebar_label: "Sherlock"
description: "跨 400+ 社交网络的 OSINT 用户名搜索"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sherlock

跨 400+ 社交网络的 OSINT（开源情报）用户名搜索。通过用户名追踪社交媒体账号。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/security/sherlock` 安装 |
| 路径 | `optional-skills/security/sherlock` |
| 版本 | `1.0.0` |
| 作者 | unmodeled-tyler |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `osint`, `security`, `username`, `social-media`, `reconnaissance` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Sherlock OSINT 用户名搜索

使用 [Sherlock Project](https://github.com/sherlock-project/sherlock) 跨 400+ 社交网络通过用户名追踪社交媒体账号。

## 使用时机

- 用户要求查找与某用户名关联的账号
- 用户想检查用户名在各平台的可用性
- 用户正在进行 OSINT 或侦察研究
- 用户询问"这个用户名在哪里注册了？"或类似问题

## 前置要求

- 已安装 Sherlock CLI：`pipx install sherlock-project` 或 `pip install sherlock-project`
- 或者：可用的 Docker（`docker run -it --rm sherlock/sherlock`）
- 可访问网络以查询社交平台

## 操作流程

### 1. 检查 Sherlock 是否已安装

**在执行任何操作之前**，先验证 sherlock 是否可用：

```bash
sherlock --version
```

如果命令失败：
- 提议安装：`pipx install sherlock-project`（推荐）或 `pip install sherlock-project`
- **不要**尝试多种安装方式 — 选择一种并继续
- 如果安装失败，告知用户并停止

### 2. 提取用户名

**如果用户消息中明确说明了用户名，直接从中提取。**

以下情况**不应**使用 clarify（澄清）：
- "Find accounts for nasa" → 用户名为 `nasa`
- "Search for johndoe123" → 用户名为 `johndoe123`
- "Check if alice exists on social media" → 用户名为 `alice`
- "Look up user bob on social networks" → 用户名为 `bob`

**仅在以下情况使用 clarify：**
- 提到了多个可能的用户名（"search for alice or bob"）
- 表述模糊（"search for my username" 但未指定）
- 完全未提及用户名（"do an OSINT search"）

提取时，**原样**保留用户名 — 保留大小写、数字、下划线等。

### 3. 构建命令

**默认命令**（除非用户明确要求，否则使用此命令）：
```bash
sherlock --print-found --no-color "<username>" --timeout 90
```

**可选标志**（仅在用户明确要求时添加）：
- `--nsfw` — 包含 NSFW 站点（仅在用户要求时）
- `--tor` — 通过 Tor 路由（仅在用户要求匿名时）

**不要通过 clarify 询问选项** — 直接运行默认搜索。用户如有需要可自行请求特定选项。

### 4. 执行搜索

通过 `terminal` 工具运行。根据网络状况和站点数量，命令通常需要 30-120 秒。

**终端调用示例：**
```json
{
  "command": "sherlock --print-found --no-color \"target_username\"",
  "timeout": 180
}
```

### 5. 解析并呈现结果

Sherlock 以简单格式输出找到的账号。解析输出并呈现：

1. **摘要行：** "Found X accounts for username 'Y'"
2. **分类链接：** 如有帮助，按平台类型分组（社交、职业、论坛等）
3. **输出文件位置：** Sherlock 默认将结果保存至 `<username>.txt`

**输出解析示例：**
```
[+] Instagram: https://instagram.com/username
[+] Twitter: https://twitter.com/username
[+] GitHub: https://github.com/username
```

尽可能以可点击链接的形式呈现结果。

## 常见问题

### 未找到结果
如果 Sherlock 未找到任何账号，这通常是正确的 — 该用户名可能未在已检查的平台上注册。建议：
- 检查拼写或变体
- 使用 `?` 通配符尝试相似用户名：`sherlock "user?name"`
- 用户可能设置了隐私保护或已删除账号

### 超时问题
部分站点响应缓慢或屏蔽自动请求。使用 `--timeout 120` 增加等待时间，或使用 `--site` 限制搜索范围。

### Tor 配置
`--tor` 需要 Tor 守护进程运行。如果用户需要匿名但 Tor 不可用，建议：
- 安装 Tor 服务
- 使用 `--proxy` 配合其他代理

### 误报
部分站点由于响应结构问题始终返回"已找到"。对意外结果进行人工交叉核验。

### 速率限制
频繁搜索可能触发速率限制。批量用户名搜索时，在调用之间添加延迟，或使用 `--local` 配合缓存数据。

## 安装

### pipx（推荐）
```bash
pipx install sherlock-project
```

### pip
```bash
pip install sherlock-project
```

### Docker
```bash
docker pull sherlock/sherlock
docker run -it --rm sherlock/sherlock <username>
```

### Linux 软件包
适用于 Debian 13+、Ubuntu 22.10+、Homebrew、Kali、BlackArch。

## 合规使用

此工具仅用于合法的 OSINT 和研究目的。请提醒用户：
- 仅搜索自己拥有或有权调查的用户名
- 遵守各平台服务条款
- 不得用于骚扰、跟踪或非法活动
- 分享结果前请考虑隐私影响

## 验证

运行 sherlock 后，验证：
1. 输出列出了带 URL 的已找到站点
2. 如使用文件输出，已创建 `<username>.txt` 文件（默认输出）
3. 如使用 `--print-found`，输出应仅包含匹配的 `[+]` 行

## 交互示例

**用户：** "Can you check if the username 'johndoe123' exists on social media?"

**Agent 操作流程：**
1. 检查 `sherlock --version`（验证已安装）
2. 已提供用户名 — 直接继续
3. 运行：`sherlock --print-found --no-color "johndoe123" --timeout 90`
4. 解析输出并呈现链接

**响应格式：**
> Found 12 accounts for username 'johndoe123':
>
> • https://twitter.com/johndoe123
> • https://github.com/johndoe123
> • https://instagram.com/johndoe123
> • [... 其他链接]
>
> Results saved to: johndoe123.txt

---

**用户：** "Search for username 'alice' including NSFW sites"

**Agent 操作流程：**
1. 检查 sherlock 已安装
2. 已提供用户名及 NSFW 标志
3. 运行：`sherlock --print-found --no-color --nsfw "alice" --timeout 90`
4. 呈现结果