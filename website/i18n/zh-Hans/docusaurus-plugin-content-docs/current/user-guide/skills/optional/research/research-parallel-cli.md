---
title: "Parallel Cli"
sidebar_label: "Parallel Cli"
description: "可选的供应商技能，用于 Parallel CLI — 面向 agent 的网络搜索、提取、深度研究、数据丰富、FindAll 和监控"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Parallel Cli

可选的供应商技能，用于 Parallel CLI — 面向 agent 的网络搜索、提取、深度研究、数据丰富、FindAll 和监控。优先使用 JSON 输出和非交互式流程。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/research/parallel-cli` 安装 |
| 路径 | `optional-skills/research/parallel-cli` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Research`, `Web`, `Search`, `Deep-Research`, `Enrichment`, `CLI` |
| 相关技能 | [`duckduckgo-search`](/user-guide/skills/optional/research/research-duckduckgo-search), [`mcporter`](/user-guide/skills/optional/mcp/mcp-mcporter) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是 agent 在技能激活时所看到的指令内容。
:::

# Parallel CLI

当用户明确要求使用 Parallel，或终端原生工作流能从 Parallel 的供应商专属技术栈中受益时（包括网络搜索、提取、深度研究、数据丰富、实体发现或监控），请使用 `parallel-cli`。

这是一个可选的第三方工作流，不是 Hermes 的核心能力。

重要说明：
- Parallel 是付费服务，提供免费套餐，并非完全免费的本地工具。
- 它与 Hermes 原生的 `web_search` / `web_extract` 存在功能重叠，因此不要在普通查询中优先使用它。
- 当用户明确提及 Parallel，或需要 Parallel 特有的数据丰富、FindAll 或监控工作流时，优先使用此技能。

`parallel-cli` 专为 agent 设计：
- 通过 `--json` 输出 JSON
- 非交互式命令执行
- 使用 `--no-wait`、`status` 和 `poll` 处理异步长时任务
- 通过 `--previous-interaction-id` 进行上下文链式调用
- 在单一 CLI 中集成搜索、提取、研究、数据丰富、实体发现和监控

## 使用时机

在以下情况下优先使用此技能：
- 用户明确提及 Parallel 或 `parallel-cli`
- 任务需要比简单单次搜索/提取更丰富的工作流
- 需要可启动并稍后轮询的异步深度研究任务
- 需要结构化数据丰富、FindAll 实体发现或监控

在未明确要求 Parallel 的情况下进行快速单次查询时，优先使用 Hermes 原生的 `web_search` / `web_extract`。

## 安装

选择当前环境中侵入性最小的安装方式。

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

如果需要隔离的 Python 安装，也可以使用 `pipx`：

```bash
pipx install "parallel-web-tools[cli]"
pipx ensurepath
```

## 认证

交互式登录：

```bash
parallel-cli login
```

无头模式 / SSH / CI：

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

如果认证需要浏览器交互，请使用 `pty=true` 运行。

## 核心规则

1. 需要机器可读输出时，始终优先使用 `--json`。
2. 优先使用显式参数和非交互式流程。
3. 对于长时任务，使用 `--no-wait`，然后调用 `status` / `poll`。
4. 仅引用 CLI 输出中返回的 URL。
5. 当后续可能有追问时，将大型 JSON 输出保存到临时文件。
6. 仅对真正的长时工作流使用后台进程；否则在前台运行。
7. 除非用户明确要求 Parallel 或需要 Parallel 专属工作流，否则优先使用 Hermes 原生工具。

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

## 常用标志与模式

常用标志：
- `--json` 用于结构化输出
- `--no-wait` 用于异步任务
- `--previous-interaction-id <id>` 用于复用早期上下文的后续任务
- `--max-results <n>` 用于限制搜索结果数量
- `--mode one-shot|agentic` 用于控制搜索行为
- `--include-domains domain1.com,domain2.com`
- `--exclude-domains domain1.com,domain2.com`
- `--after-date YYYY-MM-DD`

在方便时从 stdin 读取：

```bash
echo "What is the latest funding for Anthropic?" | parallel-cli search - --json
echo "Research question" | parallel-cli research run - --json
```

## 搜索

用于获取带结构化结果的当前网络查询。

```bash
parallel-cli search "What is Anthropic's latest AI model?" --json
parallel-cli search "SEC filings for Apple" --include-domains sec.gov --json
parallel-cli search "bitcoin price" --after-date 2026-01-01 --max-results 10 --json
parallel-cli search "latest browser benchmarks" --mode one-shot --json
parallel-cli search "AI coding agent enterprise reviews" --mode agentic --json
```

常用约束：
- `--include-domains` 缩小可信来源范围
- `--exclude-domains` 过滤噪声域名
- `--after-date` 按时效性过滤
- `--max-results` 需要更广泛覆盖时使用

如果预计有后续追问，保存输出：

```bash
parallel-cli search "latest React 19 changes" --json -o /tmp/react-19-search.json
```

汇总结果时：
- 以答案开头
- 包含日期、名称和具体事实
- 仅引用返回的来源
- 不得编造 URL 或来源标题

## 提取

用于从 URL 中提取干净内容或 markdown。

```bash
parallel-cli extract https://example.com --json
parallel-cli extract https://company.com --objective "Find pricing info" --json
parallel-cli extract https://example.com --full-content --json
parallel-cli fetch https://example.com --json
```

当页面内容宽泛而只需要其中某一部分信息时，使用 `--objective`。

## 深度研究

用于可能耗时的多步骤深度研究任务。

常用处理器级别：
- `lite` / `base` 用于更快、更经济的处理
- `core` / `pro` 用于更全面的综合分析
- `ultra` 用于最重量级的研究任务

### 同步模式

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

### 上下文链式调用 / 后续追问

```bash
parallel-cli research run "What are the top AI coding agents?" --json
parallel-cli research run \
  "What enterprise controls does the top-ranked one offer?" \
  --previous-interaction-id trun_xxx \
  --json
```

推荐的 Hermes 工作流：
1. 使用 `--no-wait --json` 启动
2. 捕获返回的运行/任务 ID
3. 如果用户希望继续其他工作，继续推进
4. 稍后调用 `status` 或 `poll`
5. 使用返回来源中的引用汇总最终报告

## 数据丰富（Enrichment）

当用户有 CSV/JSON/表格输入并希望通过网络研究推断额外列时使用。

### 建议列

```bash
parallel-cli enrich suggest "Find the CEO and annual revenue" --json
```

### 规划配置

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

在非交互式操作时，使用显式 JSON 数组定义列。
在报告成功前验证输出文件。

## FindAll

当用户需要发现数据集而非简短答案时，用于网络规模的实体发现。

```bash
parallel-cli findall run "Find AI coding agent startups with enterprise offerings" --json
parallel-cli findall run "AI startups in healthcare" -n 25 --json
parallel-cli findall status <run_id> --json
parallel-cli findall poll <run_id> --json
parallel-cli findall result <run_id> --json
parallel-cli findall schema <run_id> --json
```

当用户需要一组可供后续审查、过滤或数据丰富的实体集合时，这比普通搜索更合适。

## 监控（Monitor）

用于随时间推移的持续变更检测。

```bash
parallel-cli monitor list --json
parallel-cli monitor get <monitor_id> --json
parallel-cli monitor events <monitor_id> --json
parallel-cli monitor delete <monitor_id> --json
```

创建通常是敏感环节，因为频率和推送方式很重要：

```bash
parallel-cli monitor create --help
```

当用户希望对某个页面或来源进行周期性跟踪而非一次性抓取时使用。

## 推荐的 Hermes 使用模式

### 快速答案与引用
1. 运行 `parallel-cli search ... --json`
2. 解析标题、URL、日期、摘录
3. 仅使用返回的 URL 进行内联引用并汇总

### URL 调查
1. 运行 `parallel-cli extract URL --json`
2. 如有需要，使用 `--objective` 或 `--full-content` 重新运行
3. 引用或汇总提取的 markdown

### 长时研究工作流
1. 运行 `parallel-cli research run ... --no-wait --json`
2. 存储返回的 ID
3. 继续其他工作或定期轮询
4. 使用引用汇总最终报告

### 结构化数据丰富工作流
1. 检查输入文件和列
2. 使用 `enrich suggest` 或提供显式的丰富列定义
3. 运行 `enrich run`
4. 如有需要，轮询等待完成
5. 在报告成功前验证输出文件

## 错误处理与退出码

CLI 文档中定义的退出码：
- `0` 成功
- `2` 输入错误
- `3` 认证错误
- `4` API 错误
- `5` 超时

遇到认证错误时：
1. 检查 `parallel-cli auth`
2. 确认 `PARALLEL_API_KEY` 已设置，或运行 `parallel-cli login` / `parallel-cli login --device`
3. 验证 `parallel-cli` 在 `PATH` 中

## 维护

检查当前认证 / 安装状态：

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

## 注意事项

- 除非用户明确要求人类可读格式，否则不要省略 `--json`。
- 不要引用 CLI 输出中未出现的来源。
- `login` 可能需要 PTY/浏览器交互。
- 短时任务优先在前台执行；不要过度使用后台进程。
- 对于大型结果集，将 JSON 保存到 `/tmp/*.json`，而不是将所有内容塞入上下文。
- 当 Hermes 原生工具已经足够时，不要静默地选择 Parallel。
- 请记住，这是一个供应商工作流，通常需要账户认证，且超出免费套餐后需要付费使用。