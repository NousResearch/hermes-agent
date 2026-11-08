---
title: "Domain Intel — 使用 Python 标准库进行被动域名侦察"
sidebar_label: "Domain Intel"
description: "使用 Python 标准库进行被动域名侦察"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Domain Intel

使用 Python 标准库进行被动域名侦察。支持子域名发现、SSL 证书检查、WHOIS 查询、DNS 记录、域名可用性检测以及批量多域名分析。无需 API 密钥。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/research/domain-intel` 安装 |
| 路径 | `optional-skills/research/domain-intel` |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Domain Intelligence — 被动 OSINT

仅使用 Python 标准库进行被动域名侦察。
**零依赖。零 API 密钥。支持 Linux、macOS 和 Windows。**

## 辅助脚本

此 skill 包含 `scripts/domain_intel.py` — 一个涵盖所有域名情报操作的完整 CLI 工具。

```bash
# 通过证书透明度日志发现子域名
python3 SKILL_DIR/scripts/domain_intel.py subdomains example.com

# SSL 证书检查（有效期、加密套件、SAN、颁发者）
python3 SKILL_DIR/scripts/domain_intel.py ssl example.com

# WHOIS 查询（注册商、日期、名称服务器 — 支持 100+ 顶级域名）
python3 SKILL_DIR/scripts/domain_intel.py whois example.com

# DNS 记录（A、AAAA、MX、NS、TXT、CNAME）
python3 SKILL_DIR/scripts/domain_intel.py dns example.com

# 域名可用性检测（被动方式：DNS + WHOIS + SSL 信号）
python3 SKILL_DIR/scripts/domain_intel.py available coolstartup.io

# 批量分析 — 并行对多个域名执行多项检查
python3 SKILL_DIR/scripts/domain_intel.py bulk example.com github.com google.com
python3 SKILL_DIR/scripts/domain_intel.py bulk example.com github.com --checks ssl,dns
```

`SKILL_DIR` 为包含此 SKILL.md 文件的目录。所有输出均为结构化 JSON。

## 可用命令

| 命令 | 功能说明 | 数据来源 |
|---------|-------------|-------------|
| `subdomains` | 从证书日志中发现子域名 | crt.sh（HTTPS） |
| `ssl` | 检查 TLS 证书详情 | 直接 TCP:443 连接目标 |
| `whois` | 注册信息、注册商、日期 | WHOIS 服务器（TCP:43） |
| `dns` | A、AAAA、MX、NS、TXT、CNAME 记录 | 系统 DNS + Google DoH |
| `available` | 检查域名是否已注册 | DNS + WHOIS + SSL 信号 |
| `bulk` | 对多个域名执行多项检查 | 以上所有来源 |

## 何时使用此 skill 而非内置工具

- **使用此 skill** 处理基础设施相关问题：子域名、SSL 证书、WHOIS、DNS 记录、可用性检测
- **使用 `web_search`** 进行关于某个域名或公司的通用研究
- **使用 `web_extract`** 获取网页的实际内容
- **使用 `terminal` 配合 `curl -I`** 进行简单的"URL 是否可达"检查

| 任务 | 更合适的工具 | 原因 |
|------|-------------|-----|
| "example.com 是做什么的？" | `web_extract` | 获取页面内容，而非 DNS/WHOIS 数据 |
| "查找某公司的信息" | `web_search` | 通用研究，非域名专项 |
| "这个网站安全吗？" | `web_search` | 信誉检查需要 Web 上下文 |
| "检查某 URL 是否可达" | `terminal` 配合 `curl -I` | 简单 HTTP 检查 |
| "查找 X 的子域名" | **此 skill** | 唯一的被动来源 |
| "SSL 证书何时到期？" | **此 skill** | 内置工具无法检查 TLS |
| "谁注册了这个域名？" | **此 skill** | WHOIS 数据不在 Web 搜索结果中 |
| "coolstartup.io 可以注册吗？" | **此 skill** | 通过 DNS+WHOIS+SSL 进行被动可用性检测 |

## 平台兼容性

纯 Python 标准库（`socket`、`ssl`、`urllib`、`json`、`concurrent.futures`）。
无需任何依赖，在 Linux、macOS 和 Windows 上表现完全一致。

- **crt.sh 查询** 使用 HTTPS（443 端口） — 在大多数防火墙后均可正常工作
- **WHOIS 查询** 使用 TCP 43 端口 — 在限制性网络中可能被封锁
- **DNS 查询** 使用 Google DoH（HTTPS）解析 MX/NS/TXT — 对防火墙友好
- **SSL 检查** 连接目标的 443 端口 — 唯一的"主动"操作

## 数据来源

所有查询均为**被动**方式 — 不进行端口扫描，不进行漏洞测试：

- **crt.sh** — 证书透明度日志（子域名发现，仅 HTTPS）
- **WHOIS 服务器** — 直接 TCP 连接 100+ 权威 TLD 注册机构
- **Google DNS-over-HTTPS** — MX、NS、TXT、CNAME 解析（对防火墙友好）
- **系统 DNS** — A/AAAA 记录解析
- **SSL 检查** 是唯一的"主动"操作（TCP 连接目标:443）

## 注意事项

- WHOIS 查询使用 TCP 43 端口 — 在限制性网络中可能被封锁
- 部分 WHOIS 服务器会隐去注册人信息（GDPR 合规） — 请告知用户
- 对于非常热门的域名（拥有数千张证书），crt.sh 可能响应较慢 — 请设置合理预期
- 可用性检测基于启发式方法（3 个被动信号） — 并非像注册商 API 那样权威

---

*由 [@FurkanL0](https://github.com/FurkanL0) 贡献*