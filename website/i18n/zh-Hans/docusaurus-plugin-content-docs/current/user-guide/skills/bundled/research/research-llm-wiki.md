---
title: "Llm Wiki — Karpathy 的 LLM Wiki：构建/查询互联 Markdown 知识库"
sidebar_label: "Llm Wiki"
description: "Karpathy 的 LLM Wiki：构建/查询互联 Markdown 知识库"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llm Wiki

Karpathy 的 LLM Wiki：构建/查询互联 Markdown 知识库。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/research/llm-wiki` |
| 版本 | `2.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `wiki`, `knowledge-base`, `research`, `notes`, `markdown`, `rag-alternative` |
| 相关 skill | [`obsidian`](/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`arxiv`](/user-guide/skills/bundled/research/research-arxiv) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 看到的指令内容。
:::

# Karpathy 的 LLM Wiki

将知识库构建并维护为互联 Markdown 文件，持续积累、复利增长。
基于 [Andrej Karpathy 的 LLM Wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)。

与传统 RAG（每次查询都从头重新发现知识）不同，wiki 只编译一次知识并保持更新。交叉引用已就位，矛盾已被标记，综合分析反映了所有已摄入的内容。

**分工：** 人类负责筛选来源并指导分析。Agent 负责摘要、交叉引用、归档和维护一致性。

## 此 Skill 的激活时机

当用户执行以下操作时使用此 skill：
- 要求创建、构建或启动 wiki 或知识库
- 要求将某个来源摄入（ingest）、添加或处理到 wiki 中
- 提出问题，且配置路径下已存在 wiki
- 要求对 wiki 进行 lint、审计或健康检查
- 在研究场景中提及其 wiki、知识库或"笔记"

## Wiki 位置

**位置：** 通过 `WIKI_PATH` 环境变量设置（例如在 `~/.hermes/.env` 中）。

未设置时，默认为 `~/wiki`。

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
```

Wiki 只是一个 Markdown 文件目录——可在 Obsidian、VS Code 或任意编辑器中打开。无需数据库，无需特殊工具。

## 架构：三层结构

<!-- ascii-guard-ignore -->
```
wiki/
├── SCHEMA.md           # Conventions, structure rules, domain config
├── index.md            # Sectioned content catalog with one-line summaries
├── log.md              # Chronological action log (append-only, rotated yearly)
├── raw/                # Layer 1: Immutable source material
│   ├── articles/       # Web articles, clippings
│   ├── papers/         # PDFs, arxiv papers
│   ├── transcripts/    # Meeting notes, interviews
│   └── assets/         # Images, diagrams referenced by sources
├── entities/           # Layer 2: Entity pages (people, orgs, products, models)
├── concepts/           # Layer 2: Concept/topic pages
├── comparisons/        # Layer 2: Side-by-side analyses
└── queries/            # Layer 2: Filed query results worth keeping
```
<!-- ascii-guard-ignore-end -->

**第一层——原始来源：** 不可变。Agent 只读，不修改。
**第二层——Wiki 正文：** Agent 拥有的 Markdown 文件，由 Agent 创建、更新和交叉引用。
**第三层——Schema：** `SCHEMA.md` 定义结构、约定和标签分类体系。

## 恢复已有 Wiki（关键——每次会话都必须执行）

当用户已有 wiki 时，**在执行任何操作前务必先定位自身**：

① **读取 `SCHEMA.md`** — 了解领域、约定和标签分类体系。
② **读取 `index.md`** — 了解已有页面及其摘要。
③ **扫描近期 `log.md`** — 读取最后 20-30 条记录，了解近期活动。

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
# Orientation reads at session start
read_file "$WIKI/SCHEMA.md"
read_file "$WIKI/index.md"
read_file "$WIKI/log.md" offset=<last 30 lines>
```

只有完成定位后，才可进行摄入、查询或 lint 操作。这可以防止：
- 为已存在的实体创建重复页面
- 遗漏对已有内容的交叉引用
- 违反 schema 约定
- 重复已记录的工作

对于大型 wiki（100+ 页），在创建任何新内容前，还需针对当前主题快速执行 `search_files`。

## 初始化新 Wiki

当用户要求创建或启动 wiki 时：

1. 确定 wiki 路径（从 `$WIKI_PATH` 环境变量获取，或询问用户；默认 `~/wiki`）
2. 创建上述目录结构
3. 询问用户 wiki 涵盖的领域——要具体
4. 编写针对该领域定制的 `SCHEMA.md`（见下方模板）
5. 编写带分节标题的初始 `index.md`
6. 编写包含创建条目的初始 `log.md`
7. 确认 wiki 已就绪，并建议首批摄入来源

### SCHEMA.md 模板

根据用户领域进行调整。Schema 约束 Agent 行为并确保一致性：

```markdown
# Wiki Schema

## Domain
[What this wiki covers — e.g., "AI/ML research", "personal health", "startup intelligence"]

## Conventions
- File names: lowercase, hyphens, no spaces (e.g., `transformer-architecture.md`)
- Every wiki page starts with YAML frontmatter (see below)
- Use `[[wikilinks]]` to link between pages (minimum 2 outbound links per page)
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md` under the correct section
- Every action must be appended to `log.md`
- **Provenance markers:** On pages that synthesize 3+ sources, append `^[raw/articles/source-file.md]`
  at the end of paragraphs whose claims come from a specific source. This lets a reader trace each
  claim back without re-reading the whole raw file. Optional on single-source pages where the
  `sources:` frontmatter is enough.

## Frontmatter
  ```yaml
  ---
  title: Page Title
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [from taxonomy below]
  sources: [raw/articles/source-name.md]
  # Optional quality signals:
  confidence: high | medium | low        # how well-supported the claims are
  contested: true                        # set when the page has unresolved contradictions
  contradictions: [other-page-slug]      # pages this one conflicts with
  ---
  ```

`confidence` 和 `contested` 是可选字段，但对于观点性强或快速变化的主题建议填写。Lint 会将 `contested: true` 和 `confidence: low` 的页面标记出来供审查，防止薄弱论断悄然固化为公认的 wiki 事实。

### raw/ Frontmatter

原始来源**同样**需要一个小型 frontmatter 块，以便重新摄入时检测内容漂移：

```yaml
---
source_url: https://example.com/article   # original URL, if applicable
ingested: YYYY-MM-DD
sha256: &lt;hex digest of the raw content below the frontmatter>
---
```

`sha256:` 字段允许未来重新摄入同一 URL 时，在内容未变时跳过处理，在内容已变时标记漂移。仅对正文（frontmatter 结束 `---` 之后的所有内容）计算哈希，不含 frontmatter 本身。

## Tag Taxonomy
[Define 10-20 top-level tags for the domain. Add new tags here BEFORE using them.]

Example for AI/ML:
- Models: model, architecture, benchmark, training
- People/Orgs: person, company, lab, open-source
- Techniques: optimization, fine-tuning, inference, alignment, data
- Meta: comparison, timeline, controversy, prediction

Rule: every tag on a page must appear in this taxonomy. If a new tag is needed,
add it here first, then use it. This prevents tag sprawl.

## Page Thresholds
- **Create a page** when an entity/concept appears in 2+ sources OR is central to one source
- **Add to existing page** when a source mentions something already covered
- **DON'T create a page** for passing mentions, minor details, or things outside the domain
- **Split a page** when it exceeds ~200 lines — break into sub-topics with cross-links
- **Archive a page** when its content is fully superseded — move to `_archive/`, remove from index

## Entity Pages
One page per notable entity. Include:
- Overview / what it is
- Key facts and dates
- Relationships to other entities ([[wikilinks]])
- Source references

## Concept Pages
One page per concept or topic. Include:
- Definition / explanation
- Current state of knowledge
- Open questions or debates
- Related concepts ([[wikilinks]])

## Comparison Pages
Side-by-side analyses. Include:
- What is being compared and why
- Dimensions of comparison (table format preferred)
- Verdict or synthesis
- Sources

## Update Policy
When new information conflicts with existing content:
1. Check the dates — newer sources generally supersede older ones
2. If genuinely contradictory, note both positions with dates and sources
3. Mark the contradiction in frontmatter: `contradictions: [page-name]`
4. Flag for user review in the lint report
```

### index.md 模板

索引按类型分节。每条记录为一行：wikilink + 摘要。

```markdown
# Wiki Index

> Content catalog. Every wiki page listed under its type with a one-line summary.
> Read this first to find relevant pages for any query.
> Last updated: YYYY-MM-DD | Total pages: N

## Entities
<!-- Alphabetical within section -->

## Concepts

## Comparisons

## Queries
```

**扩展规则：** 当任意分节超过 50 条时，按首字母或子领域拆分为子节。当索引总条目超过 200 时，创建 `_meta/topic-map.md`，按主题对页面分组，以加快导航速度。

### log.md 模板

```markdown
# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete
> When this file exceeds 500 entries, rotate: rename to log-YYYY.md, start fresh.

## [YYYY-MM-DD] create | Wiki initialized
- Domain: [domain]
- Structure created with SCHEMA.md, index.md, log.md
```

## 核心操作

### 1. 摄入（Ingest）

当用户提供来源（URL、文件、粘贴内容）时，将其整合到 wiki 中：

① **捕获原始来源：**
   - URL → 使用 `web_extract` 获取 Markdown，保存到 `raw/articles/`
   - PDF → 使用 `web_extract`（支持 PDF），保存到 `raw/papers/`
   - 粘贴文本 → 保存到对应的 `raw/` 子目录
   - 文件名应具有描述性：`raw/articles/karpathy-llm-wiki-2026.md`
   - **添加 raw frontmatter**（`source_url`、`ingested`、正文的 `sha256`）。
     重新摄入同一 URL 时：重新计算 sha256，与已存储值比较——相同则跳过，不同则标记漂移并更新。此操作成本极低，每次重新摄入都可执行，能捕获静默的来源变更。

② **与用户讨论要点** — 哪些内容有趣，哪些对领域重要。（自动化/cron 场景下跳过此步，直接继续。）

③ **检查已有内容** — 搜索 index.md，并使用 `search_files` 查找已提及实体/概念的现有页面。这是 wiki 持续增长与变成重复堆砌之间的关键区别。

④ **编写或更新 wiki 页面：**
   - **新实体/概念：** 仅在满足 SCHEMA.md 中页面阈值时创建页面（2+ 来源提及，或在某一来源中处于核心地位）
   - **已有页面：** 添加新信息，更新事实，更新 `updated` 日期。新信息与已有内容矛盾时，遵循更新策略。
   - **交叉引用：** 每个新建或更新的页面必须通过 `[[wikilinks]]` 链接到至少 2 个其他页面。检查已有页面是否有反向链接。
   - **标签：** 只使用 SCHEMA.md 分类体系中的标签
   - **来源溯源：** 在综合 3+ 来源的页面上，在论断可追溯到特定来源的段落末尾添加 `^[raw/articles/source.md]` 标记。
   - **置信度：** 对于观点性强、快速变化或单一来源的论断，在 frontmatter 中设置 `confidence: medium` 或 `low`。除非论断在多个来源中有充分支撑，否则不标记 `high`。

⑤ **更新导航：**
   - 将新页面按字母顺序添加到 `index.md` 对应分节
   - 更新 index 头部的"Total pages"计数和"Last updated"日期
   - 追加到 `log.md`：`## [YYYY-MM-DD] ingest | Source Title`
   - 在日志条目中列出每个创建或更新的文件

⑥ **报告变更内容** — 向用户列出每个创建或更新的文件。

单个来源可能触发 5-15 个 wiki 页面的更新。这是正常且期望的结果——这正是复利效应。

### 2. 查询（Query）

当用户就 wiki 领域提问时：

① **读取 `index.md`** 以识别相关页面。
② **对于 100+ 页的 wiki**，还需对所有 `.md` 文件执行 `search_files` 搜索关键词——仅靠索引可能遗漏相关内容。
③ **读取相关页面**，使用 `read_file`。
④ **从已编译的知识中综合答案**。引用所参考的 wiki 页面："Based on [[page-a]] and [[page-b]]..."
⑤ **将有价值的答案归档** — 如果答案是实质性的比较、深度分析或新颖综合，在 `queries/` 或 `comparisons/` 中创建页面。不要归档琐碎的查询——只归档重新推导代价高昂的答案。
⑥ **更新 log.md**，记录查询内容及是否已归档。

### 3. Lint

当用户要求 lint、健康检查或审计 wiki 时：

① **孤立页面：** 查找没有其他页面通过 `[[wikilinks]]` 指向的页面。
```python
# Use execute_code for this — programmatic scan across all wiki pages
import os, re
from collections import defaultdict
wiki = "<WIKI_PATH>"
# Scan all .md files in entities/, concepts/, comparisons/, queries/
# Extract all [[wikilinks]] — build inbound link map
# Pages with zero inbound links are orphans
```

② **断开的 wikilink：** 查找指向不存在页面的 `[[links]]`。

③ **索引完整性：** 每个 wiki 页面都应出现在 `index.md` 中。对比文件系统与索引条目。

④ **Frontmatter 验证：** 每个 wiki 页面必须包含所有必填字段（title、created、updated、type、tags、sources）。标签必须在分类体系中。

⑤ **过时内容：** `updated` 日期比提及相同实体的最新来源早 90 天以上的页面。

⑥ **矛盾：** 涉及同一主题但论断相互冲突的页面。查找共享标签/实体但陈述不同事实的页面。将所有带有 `contested: true` 或 `contradictions:` frontmatter 的页面标记出来供用户审查。

⑦ **质量信号：** 列出 `confidence: low` 的页面，以及仅引用单一来源但未设置 confidence 字段的页面——这些页面是寻找佐证或降级为 `confidence: medium` 的候选。

⑧ **来源漂移：** 对 `raw/` 中每个带有 `sha256:` frontmatter 的文件，重新计算哈希并标记不匹配项。不匹配表明原始文件被编辑（不应发生——`raw/` 是不可变的）或从已变更的 URL 摄入。不是硬性错误，但值得报告。

⑨ **页面大小：** 标记超过 200 行的页面——拆分候选。

⑩ **标签审计：** 列出所有使用中的标签，标记不在 SCHEMA.md 分类体系中的标签。

⑪ **日志轮转：** 如果 log.md 超过 500 条，进行轮转。

⑫ **报告发现结果**，附具体文件路径和建议操作，按严重程度分组（断开链接 > 孤立页面 > 来源漂移 > 有争议页面 > 过时内容 > 样式问题）。

⑬ **追加到 log.md：** `## [YYYY-MM-DD] lint | N issues found`

## Wiki 使用方法

### 搜索

```bash
# Find pages by content
search_files "transformer" path="$WIKI" file_glob="*.md"

# Find pages by filename
search_files "*.md" target="files" path="$WIKI"

# Find pages by tag
search_files "tags:.*alignment" path="$WIKI" file_glob="*.md"

# Recent activity
read_file "$WIKI/log.md" offset=<last 20 lines>
```

### 批量摄入

同时摄入多个来源时，批量处理更新：
1. 先读取所有来源
2. 识别所有来源中的所有实体和概念
3. 一次性检查所有实体的已有页面（一次搜索，而非 N 次）
4. 一次性创建/更新页面（避免冗余更新）
5. 最后统一更新 index.md
6. 写一条涵盖整批操作的日志条目

### 归档

当内容完全被取代或领域范围发生变化时：
1. 如不存在则创建 `_archive/` 目录
2. 将页面移至 `_archive/`，保留原始路径（例如 `_archive/entities/old-page.md`）
3. 从 `index.md` 中移除
4. 更新所有链接到该页面的页面——将 wikilink 替换为纯文本 + "（已归档）"
5. 记录归档操作

### Obsidian 集成

Wiki 目录开箱即用作为 Obsidian vault：
- `[[wikilinks]]` 渲染为可点击链接
- 图谱视图可视化知识网络
- YAML frontmatter 支持 Dataview 查询
- `raw/assets/` 文件夹存放通过 `![[image.png]]` 引用的图片

最佳实践：
- 将 Obsidian 的附件文件夹设置为 `raw/assets/`
- 在 Obsidian 设置中启用"Wikilinks"（通常默认开启）
- 安装 Dataview 插件，支持如 `TABLE tags FROM "entities" WHERE contains(tags, "company")` 的查询

如果同时使用 Obsidian skill，将 `OBSIDIAN_VAULT_PATH` 设置为与 wiki 路径相同的目录。

### Obsidian 无头模式（服务器和无显示器机器）

在没有显示器的机器上，使用 `obsidian-headless` 代替桌面应用。它通过 Obsidian Sync 同步 vault，无需 GUI——非常适合在服务器上运行、向 wiki 写入内容，同时在另一台设备上用 Obsidian 桌面端读取的 Agent。

**设置：**
```bash
# Requires Node.js 22+
npm install -g obsidian-headless

# Login (requires Obsidian account with Sync subscription)
ob login --email <email> --password '<password>'

# Create a remote vault for the wiki
ob sync-create-remote --name "LLM Wiki"

# Connect the wiki directory to the vault
cd ~/wiki
ob sync-setup --vault "<vault-id>"

# Initial sync
ob sync

# Continuous sync (foreground — use systemd for background)
ob sync --continuous
```

**通过 systemd 实现持续后台同步：**
```ini
# ~/.config/systemd/user/obsidian-wiki-sync.service
[Unit]
Description=Obsidian LLM Wiki Sync
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/path/to/ob sync --continuous
WorkingDirectory=/home/user/wiki
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now obsidian-wiki-sync
# Enable linger so sync survives logout:
sudo loginctl enable-linger $USER
```

这样 Agent 可以在服务器上向 `~/wiki` 写入内容，同时你在笔记本/手机上的 Obsidian 中浏览同一 vault——变更在数秒内即可同步。

## 注意事项

- **永远不要修改 `raw/` 中的文件** — 来源是不可变的。更正内容写入 wiki 页面。
- **始终先定位自身** — 在新会话中执行任何操作前，先读取 SCHEMA + index + 近期日志。跳过此步会导致重复和遗漏交叉引用。
- **始终更新 index.md 和 log.md** — 跳过此步会导致 wiki 退化。这两个文件是导航骨架。
- **不要为一笔带过的提及创建页面** — 遵循 SCHEMA.md 中的页面阈值。某个名称在脚注中出现一次，不足以创建实体页面。
- **不要创建没有交叉引用的页面** — 孤立页面是不可见的。每个页面必须链接到至少 2 个其他页面。
- **Frontmatter 是必填的** — 它支持搜索、过滤和过时检测。
- **标签必须来自分类体系** — 自由形式的标签会退化为噪音。先在 SCHEMA.md 中添加新标签，再使用。
- **保持页面可扫描** — wiki 页面应在 30 秒内可读完。超过 200 行的页面应拆分。将详细分析移至专用深度分析页面。
- **批量更新前先确认** — 如果一次摄入会影响 10+ 个已有页面，先与用户确认范围。
- **轮转日志** — 当 log.md 超过 500 条时，将其重命名为 `log-YYYY.md` 并重新开始。Agent 应在 lint 期间检查日志大小。
- **显式处理矛盾** — 不要静默覆盖。注明两种论断及其日期，在 frontmatter 中标记，标记供用户审查。

## 相关工具

[llm-wiki-compiler](https://github.com/atomicmemory/llm-wiki-compiler) 是一个 Node.js CLI，基于相同的 Karpathy 灵感将来源编译为概念 wiki。它兼容 Obsidian，因此希望使用定时/CLI 驱动编译流水线的用户可以将其指向此 skill 维护的同一 vault。权衡：它拥有页面生成的控制权（取代 Agent 在页面创建上的判断），并针对小型语料库进行了调优。当你希望 Agent 参与策划时使用此 skill；当你希望批量编译来源目录时使用 llmwiki。