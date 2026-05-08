---
title: "Parallel Cli"
sidebar_label: "Parallel Cli"
description: "Parallel CLI 的可选供应商技能 — 智能体原生网络搜索、提取、深度研究、丰富、FindAll 和监控"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Parallel Cli

Parallel CLI 的可选供应商技能 — 智能体原生网络搜索、提取、深度研究、丰富、FindAll 和监控。优先使用 JSON 输出和非交互式流程。

## 技能元数据

| | |
|---|---|
| 来源 | 可选技能 — 使用 `hermes skills install official/research/parallel-cli` 安装 |
| 路径 | `optional-skills/research/parallel-cli` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `Research`, `Web`, `Search`, `Deep-Research`, `Enrichment`, `CLI` |
| 相关技能 | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`mcporter`](/docs/user-guide/skills/optional/mcp/mcp-mcporter) |

## 参考：完整的 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时智能体看到的指令。
:::

# Parallel CLI

当用户明确想要 Parallel 时，或当终端原生工作流程会从 Parallel 的供应商特定技术栈中受益时使用，用于网络搜索、提取、深度研究、丰富、实体发现或监控。

这是一个可选的第三方工作流程，不是 Hermes 核心功能。

重要预期：
- Parallel 是付费服务，有免费套餐，不是完全免费的本地工具。
- 它与 Hermes 原生 `web_search` / `web_extract` 重叠，因此不要默认偏好它进行普通查询。
- 当用户特别提到 Parallel 或需要 Parallel 的丰富、FindAll 或监控等功能时，优先使用此技能。

`parallel-cli` 专为智能体设计：
- 通过 `--json` 输出 JSON
- 非交互式命令执行
- 使用 `--no-wait`、`status` 和 `poll` 的异步长时间运行作业
- 使用 `--previous-interaction-id` 进行上下文链接
- 在一个 CLI 中进行搜索、提取、研究、丰富、实体发现和监控

## 何时使用

优先使用此技能当：
- 用户特别提到 Parallel 或 `parallel-cli`
- 任务需要比简单的一次性搜索/提取更丰富的工作流程
- 您需要可以启动并稍后轮询的异步深度研究作业
- 您需要结构化丰富、FindAll 实体发现或监控

当 Parallel 没有被特别请求时，对于快速一次性查询优先使用 Hermes 原生 `web_search` / `web_extract`。

## 安装

尝试为环境选择侵入性最小的安装路径。

### Homebrew

```bash
brew install parallel-web/tap/parallel-cli
```

### npm

```bash
npm install -g parallel-web-cli
```

### Python 包

```bash
pip install "parallel-web-tools[cli]"
```

### 独立安装程序

```bash
curl -fsSL https://parallel.ai/install.sh | bash
```

如果您想要隔离的 Python 安装，`pipx` 也可以工作：

```bash
pipx install "parallel-web-tools[cli]"
pipx ensurepath
```

## 身份验证

交互式登录：

```bash
parallel-cli login
```

无头 / SSH / CI：

```bash
parallel-cli login --device
```

API 密钥环境变量：

```bash
export PARALLEL_API_KEY="***"
```

验证当前认证状态：

```bash
parallel-cli auth
```

如果认证需要浏览器交互，使用 `pty=true` 运行。

## 核心规则集

1. 当需要机器可读输出时，始终优先使用 `--json`。
2. 优先使用显式参数和非交互式流程。
3. 对于长时间运行的作业，使用 `--no-wait` 然后 `status` / `poll`。
4. 仅引用 CLI 输出返回的 URL。
5. 当后续问题时，将大型 JSON 输出保存到临时文件。
6. 仅对真正长时间运行的工作流程使用后台进程；否则在前台运行。
7. 除非用户特别想要 Parallel 或需要 Parallel 独有的工作流程，否则优先使用 Hermes 原生工具。

## 快速参考

<!-- ascii-guard-ignore -->
```text
parallel-cli
├── auth
├── login
├── logout
├── search
├── extract / fetch
├── research run|status|poll|processors
├── enrich run|status|poll|plan|suggest|deploy
├── findall run|ingest|status|poll|result|enrich|extend|schema|cancel
└── monitor create|list|get|update|delete|events|event-group|simulate
```
<!-- ascii-guard-ignore-end -->

## 常用参数和模式

常用有用参数：
- `--json` 用于结构化输出
- `--no-wait` 用于异步作业
- `--previous-interaction-id <id>` 用于重用早期上下文的跟进任务
- `--max-results <n>` 用于搜索结果数量
- `--mode one-shot|agentic` 用于搜索行为
- `--include-domains domain1.com,domain2.com`
- `--exclude-domains domain1.com,domain2.com`
- `--after-date YYYY-MM-DD`

方便时从 stdin 读取：

```bash
echo "What is the latest funding for Anthropic?" | parallel-cli search - --json
echo "Research question" | parallel-cli research run - --json
```

## 搜索

用于带结构化结果的当前网络查询。

```bash
parallel-cli search "What is Anthropic's latest AI model?" --json
parallel-cli search "SEC filings for Apple" --include-domains sec.gov --json
parallel-cli search "bitcoin price" --after-date 2026-01-01 --max-results 10 --json
parallel-cli search "latest browser benchmarks" --mode one-shot --json
parallel-cli search "AI coding agent enterprise reviews" --mode agentic --json
```

有用的约束：
- `--include-domains` 缩小可信来源
- `--exclude-domains` 剥离嘈杂域名
- `--after-date` 用于时效性过滤
- `--max-results` 当需要更广泛覆盖时

如果您预期有后续问题，保存输出：

```bash
parallel-cli search "latest React 19 changes" --json -o /tmp/react-19-search.json
```

总结结果时：
- 首先给出答案
- 包括日期、名称和具体事实
- 仅引用返回的来源
- 避免编造 URL 或来源标题

## 提取

用于从 URL 拉取干净的内容或 markdown。

```bash
parallel-cli extract https://example.com --json
parallel-cli extract https://company.com --objective "Find pricing info" --json
parallel-cli extract https://example.com --full-content --json
parallel-cli fetch https://example.com --json
```

当页面内容广泛而您只需要一部分信息时，使用 `--objective`。

## 深度研究

用于可能需要时间的更深层次的多步骤研究任务。

常用处理器层级：
- `lite` / `base` 用于更快、更便宜的通过
- `core` / `pro` 用于更彻底的综合
- `ultra` 用于最重的研究作业

### 同步

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor core \
  --json
```

### 异步启动 + 轮询

```bash
parallel-cli research run \
  "Compare the leading AI coding agents by pricing, model support, and enterprise controls" \
  --processor ultra \
  --no-wait \
  --json

parallel-cli research status trun_xxx --json
parallel-cli research poll trun_xxx --json
parallel-cli research processors --json
```

### 上下文链接 / 跟进

```bash
parallel-cli research run "What are the top AI coding agents?" --json
parallel-cli research run \
  "What enterprise controls does the top-ranked one offer?" \
  --previous-interaction-id trun_xxx \
  --json
```

推荐 Hermes 工作流程：
1. 用 `--no-wait --json` 启动
2. 捕获返回的运行/任务 ID
3. 如果用户想要继续其他工作，继续
4. 稍后调用 `status` 或 `poll`
5. 用引用的来源总结最终报告

## 丰富

当用户有 CSV/JSON/表格输入并希望从网络研究中推断出附加列时使用。

### 建议列

```bash
parallel-cli enrich suggest "Find the CEO and annual revenue" --json
```

### 计划配置

```bash
parallel-cli enrich plan -o config.yaml
```

### 内联数据

```bash
parallel-cli enrich run \
  --data '[{"company": "Anthropic"}, {"company": "Mistral"}]' \
  --intent "Find headquarters and employee count" \
  --json
```

### 非交互式文件运行

```bash
parallel-cli enrich run \
  --source-type csv \
  --source companies.csv \
  --target enriched.csv \
  --source-columns '[{"name": "company", "description": "Company name"}]' \
  --intent "Find the CEO and annual revenue"
```

### YAML 配置运行

```bash
parallel-cli enrich run config.yaml
```

### 状态 / 轮询

```bash
parallel-cli enrich status <task_group_id> --json
parallel-cli enrich poll <task_group_id> --json
```

非交互式操作时，对列定义使用显式 JSON 数组。
在报告成功之前验证输出文件。

## FindAll

当用户想要一个可发现的实体数据集而不是简短答案时，用于网络规模的实体发现。

```bash
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --json
parallel-cli findall run "AI startups in healthcare" -n 25 --json
parallel-cli findall status <run_id> --json
parallel-cli findall poll <run_id> --json
parallel-cli findall result <run_id> --json
parallel-cli findall schema <run_id> --json
```

当用户想要一个可以审查、过滤或稍后丰富的已发现实体集时，这比普通搜索更合适。

## 监控

用于随时间持续检测变化。

```bash
parallel-cli monitor list --json
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor delete <monitor_id> --json
```

创建通常是敏感部分，因为节奏和交付很重要：

```bash
parallel-cli monitor create --help
```

当用户想要对页面或来源进行重复跟踪而不是一次性获取时使用此功能。

## 推荐 Hermes 使用模式

### 带引用的快速答案
1. 运行 `parallel-cli search ... --json`
2. 解析标题、URL、日期、摘要
3. 用仅来自返回 URL 的内联引用进行总结

### URL 调查
1. 运行 `parallel-cli extract URL --json`
2. 如需要，使用 `--objective` 或 `--full-content` 重新运行
3. 引用或总结提取的 markdown

### 长时间研究工作流程
1. 运行 `parallel-cli research run ... --no-wait --json`
2. 存储返回的 ID
3. 继续其他工作或定期轮询
4. 用引用总结最终报告

### 结构化丰富工作流程
1. 检查输入文件和列
2. 使用 `enrich suggest` 或提供显式丰富列
3. 运行 `enrich run`
4. 如需要轮询完成
5. 在报告成功之前验证输出文件

## 错误处理和退出代码

CLI 记录这些退出代码：
- `0` 成功
- `2` 错误输入
- `3` 认证错误
- `4` API 错误
- `5` 超时

如果您遇到认证错误：
1. 检查 `parallel-cli auth`
2. 确认 `PARALLEL_API_KEY` 或运行 `parallel-cli login` / `parallel-cli login --device`
3. 验证 `parallel-cli` 在 `PATH` 上

## 维护

检查当前认证/安装状态：

```bash
parallel-cli auth
parallel-cli --help
```

更新命令：

```bash
parallel-cli update
pip install --upgrade parallel-web-tools
parallel-cli config auto-update-check off
```

## 陷阱

- 不要省略 `--json` 除非用户明确想要人类可读的输出。
- 不要引用 CLI 输出中不存在的来源。
- `login` 可能需要 PTY/浏览器交互。
- 对于短期任务优先使用前台执行；不要过度使用后台进程。
- 对于大型结果集，将 JSON 保存到 `/tmp/*.json` 而不是将所有内容塞入上下文。
- 当 Hermes 原生工具已经足够时，不要默默选择 Parallel。
- 请记住这是一个供应商工作流程，通常需要账户认证和超出免费套餐的付费使用。
