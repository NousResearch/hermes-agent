---
title: "Sherlock — 跨 400+ 社交网络的 OSINT 用户名搜索"
sidebar_label: "Sherlock"
description: "跨 400+ 社交网络的 OSINT 用户名搜索"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sherlock

跨 400+ 社交网络的 OSINT 用户名搜索。通过用户名查找社交媒体账户。

## 技能元数据

| | |
|---|---|
| 来源 | 可选技能 — 使用 `hermes skills install official/security/sherlock` 安装 |
| 路径 | `optional-skills/security/sherlock` |
| 版本 | `1.0.0` |
| 作者 | unmodeled-tyler |
| 许可证 | MIT |
| 标签 | `osint`, `security`, `username`, `social-media`, `reconnaissance` |

## 参考：完整的 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时智能体看到的指令。
:::

# Sherlock OSINT 用户名搜索

使用 [Sherlock Project](https://github.com/sherlock-project/sherlock) 通过用户名在 400+ 社交网络中查找社交媒体账户。

## 何时使用

- 用户要求查找与用户名关联的账户
- 用户想要检查跨平台的用户名可用性
- 用户正在进行 OSINT 或侦察研究
- 用户问"这个用户名在哪里注册？"或类似问题

## 要求

- 已安装 Sherlock CLI：`pipx install sherlock-project` 或 `pip install sherlock-project`
- 或者：Docker 可用（`docker run -it --rm sherlock/sherlock`）
- 网络访问以查询社交平台

## 程序

### 1. 检查 Sherlock 是否已安装

**在任何操作之前**，验证 sherlock 可用：

```bash
sherlock --version
```

如果命令失败：
- 提供安装：`pipx install sherlock-project`（推荐）或 `pip install sherlock-project`
- **不要**尝试多种安装方法 — 选择一种并继续
- 如果安装失败，通知用户并停止

### 2. 提取用户名

**直接从用户消息中提取用户名（如果明确说明）。**

不应该使用澄清的示例：
- "查找 nasa 的账户" → 用户名是 `nasa`
- "搜索 johndoe123" → 用户名是 `johndoe123`
- "检查 alice 是否存在于社交媒体上" → 用户名是 `alice`
- "在社交网络上查找用户 bob" → 用户名是 `bob`

**仅在以下情况下使用澄清：**
- 提到多个潜在用户名（"搜索 alice 或 bob"）
- 含糊的措辞（"搜索我的用户名"但未指定）
- 根本没有提到用户名（"进行 OSINT 搜索"）

提取时，按原样使用**确切**的用户名 — 保留大小写、数字、下划线等。

### 3. 构建命令

**默认命令**（除非用户特别要求，否则使用此命令）：
```bash
sherlock --print-found --no-color "<username>" --timeout 90
```

**可选参数**（仅在用户明确要求时添加）：
- `--nsfw` — 包含 NSFW 网站（仅在用户要求时）
- `--tor` — 通过 Tor 路由（仅在用户要求匿名时）

**不要通过澄清询问选项** — 只运行默认搜索。用户如有需要可以请求特定选项。

### 4. 执行搜索

通过 `terminal` 工具运行。命令通常需要 30-120 秒，取决于网络条件和站点数量。

**示例终端调用：**
```json
{
  "command": "sherlock --print-found --no-color \"target_username\"",
  "timeout": 180
}
```

### 5. 解析并呈现结果

Sherlock 以简单格式输出找到的账户。解析输出并呈现：

1. **摘要行：** "找到用户名 'Y' 的 X 个账户"
2. **分类链接：** 如有帮助，按平台类型分组（社交、专业、论坛等）
3. **输出文件位置：** Sherlock 默认保存结果到 `<username>.txt`

**示例输出解析：**
```
[+] Instagram: https://instagram.com/username
[+] Twitter: https://twitter.com/username
[+] GitHub: https://github.com/username
```

尽可能将发现呈现为可点击链接。

## 陷阱

### 未找到结果
如果 Sherlock 未找到账户，这通常是正确的 — 用户名可能未在检查的平台上注册。建议：
- 检查拼写/变体
- 尝试带 `?` 通配符的类似用户名：`sherlock "user?name"`
- 用户可能有隐私设置或已删除账户

### 超时问题
一些网站速度慢或阻止自动化请求。使用 `--timeout 120` 增加等待时间，或使用 `--site` 限制范围。

### Tor 配置
`--tor` 需要 Tor 守护进程运行。如果用户想要匿名但 Tor 不可用，建议：
- 安装 Tor 服务
- 使用 `--proxy` 和替代代理

### 误报
一些网站由于其响应结构总是返回"找到"。交叉引用意外结果与手动检查。

### 速率限制
激进的搜索可能触发速率限制。对于批量用户名搜索，在调用之间添加延迟或使用带缓存数据的 `--local`。

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

### Linux 包
可用于 Debian 13+、Ubuntu 22.10+、Homebrew、Kali、BlackArch。

## 道德使用

此工具仅用于合法的 OSINT 和研究目的。提醒用户：
- 只搜索他们拥有或有权调查的用户名
- 尊重平台服务条款
- 不要用于骚扰、跟踪或非法活动
- 在分享结果前考虑隐私影响

## 验证

运行 sherlock 后，验证：
1. 输出列出带 URL 的找到的站点
2. 创建了 `<username>.txt` 文件（如果使用文件输出）
3. 如果使用 `--print-found`，输出应仅包含匹配的 `[+]` 行

## 示例交互

**用户：** "能检查用户名 'johndoe123' 是否存在于社交媒体上吗？"

**智能体程序：**
1. 检查 `sherlock --version`（验证已安装）
2. 用户名已提供 — 直接继续
3. 运行：`sherlock --print-found --no-color "johndoe123" --timeout 90`
4. 解析输出并呈现链接

**响应格式：**
> 找到用户名 'johndoe123' 的 12 个账户：
>
> • https://twitter.com/johndoe123
> • https://github.com/johndoe123
> • https://instagram.com/johndoe123
> • [... 其他链接]
>
> 结果保存到：johndoe123.txt

---

**用户：** "搜索用户名 'alice' 包括 NSFW 网站"

**智能体程序：**
1. 检查 sherlock 已安装
2. 用户名 + NSFW 参数都已提供
3. 运行：`sherlock --print-found --no-color --nsfw "alice" --timeout 90`
4. 呈现结果
