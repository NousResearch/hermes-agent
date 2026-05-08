---
title: "Llm Wiki — Karpathy 的 LLM Wiki：构建/查询互联的 Markdown 知识库"
sidebar_label: "Llm Wiki"
description: "Karpathy 的 LLM Wiki：构建/查询互联的 Markdown 知识库"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Llm Wiki

Karpathy 的 LLM Wiki：构建/查询互联的 Markdown 知识库。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/research/llm-wiki` |
| 版本 | `2.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `wiki`, `knowledge-base`, `research`, `notes`, `markdown`, `rag-alternative` |
| 相关技能 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Karpathy 的 LLM Wiki

构建和维护一个持久、复合的知识库，以互联的 Markdown 文件形式存在。
基于 [Andrej Karpathy 的 LLM Wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)。

与传统的 RAG（每次查询都从头重新发现知识）不同，Wiki
一次性编译知识并保持其时效性。交叉引用已经存在。
矛盾已经被标记。综合反映了所有已摄取的内容。

**劳动分工：** 人类策展来源并指导分析。代理
负责摘要、交叉引用、归档和维护一致性。

## 技能激活条件

在以下情况使用此技能：
- 用户要求创建、构建或启动一个 Wiki 或知识库
- 用户要求将来源摄取、添加或处理到其 Wiki 中
- 用户提出问题且配置路径下存在现有 Wiki
- 用户要求对 Wiki 进行 lint、审计或健康检查
- 用户在研究上下文中引用其 Wiki、知识库或"笔记"

## Wiki 位置

**位置：** 通过 `WIKI_PATH` 环境变量设置（例如在 `~/.hermes/.env` 中）。

如未设置，默认为 `~/wiki`。

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
```

Wiki 只是一个 Markdown 文件目录——可以在 Obsidian、VS Code 或任何编辑器中打开。无需数据库，无需特殊工具。

## 架构：三层结构

<!-- ascii-guard-ignore -->
```
wiki/
├── SCHEMA.md           # 约定、结构规则、领域配置
├── index.md            # 分段内容目录，带一行摘要
├── log.md              # 按时间顺序的操作日志（仅追加，每年轮换）
├── raw/                # 第 1 层：不可变的原始材料
│   ├── articles/       # 网络文章、剪藏
│   ├── papers/         # PDF、arXiv 论文
│   ├── transcripts/    # 会议记录、访谈
│   └── assets/         # 来源中引用的图片、图表
├── entities/           # 第 2 层：实体页面（人物、组织、产品、模型）
├── concepts/           # 第 2 层：概念/主题页面
├── comparisons/        # 第 2 层：对比分析
└── queries/            # 第 2 层：值得保存的查询结果
```
<!-- ascii-guard-ignore-end -->

**第 1 层 — 原始来源：** 不可变。代理只能读取，永远不修改这些文件。
**第 2 层 — Wiki：** 代理拥有的 Markdown 文件。由代理创建、更新和交叉引用。
**第 3 层 — 模式：** `SCHEMA.md` 定义结构、约定和标签分类法。

## 恢复现有 Wiki（关键——每次会话都要执行）

当用户有现有 Wiki 时，**在做任何操作之前必须先了解情况**：

① **阅读 `SCHEMA.md`** ——了解领域、约定和标签分类法。
② **阅读 `index.md`** ——了解存在哪些页面及其摘要。
③ **浏览最近的 `log.md`** ——阅读最后 20-30 条条目以了解最近的活动。

```bash
WIKI="${WIKI_PATH:-$HOME/wiki}"
# 会话开始时的定位读取
read_file "$WIKI/SCHEMA.md"
read_file "$WIKI/index.md"
read_file "$WIKI/log.md" offset=<last 30 lines>
```

只有在完成定位后才能进行摄取、查询或 lint。这样可以避免：
- 为已存在的实体创建重复页面
- 遗漏对现有内容的交叉引用
- 与模式的约定产生矛盾
- 重复已记录的工作

对于大型 Wiki（100+ 页面），在创建任何新内容之前，还要针对当前主题运行一次快速的 `search_files`。

## 初始化新 Wiki

当用户要求创建或启动一个 Wiki 时：

1. 确定 Wiki 路径（从 `$WIKI_PATH` 环境变量，或询问用户；默认 `~/wiki`）
2. 创建上述目录结构
3. 询问用户 Wiki 涵盖的领域——要具体
4. 编写针对该领域定制的 `SCHEMA.md`（见下方模板）
5. 编写带分段标题的初始 `index.md`
6. 编写带创建条目的初始 `log.md`
7. 确认 Wiki 准备就绪，并建议首批要摄取的来源

### SCHEMA.md 模板

根据用户的领域进行适配。模式约束代理行为并确保一致性：

```markdown
# Wiki 模式

## 领域
[此 Wiki 涵盖的内容——例如 "AI/ML 研究"、"个人健康"、"创业情报"]

## 约定
- 文件名：小写、连字符、无空格（例如 `transformer-architecture.md`）
- 每个 Wiki 页面以 YAML frontmatter 开头（见下方）
- 使用 `[[wikilinks]]` 在页面之间链接（每页至少 2 个出站链接）
- 更新页面时，始终更新 `updated` 日期
- 每个新页面必须添加到 `index.md` 的正确分类下
- 每个操作必须追加到 `log.md`
- **来源标记：** 在综合了 3+ 来源的页面上，在声明来自特定来源的段落末尾
  附加 `^[raw/articles/source-file.md]`。这使读者无需重新阅读整个原始文件即可
  追溯每个声明。在单一来源页面（`sources:` frontmatter 足够时）为可选。

## Frontmatter
  ```yaml
  ---
  title: 页面标题
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [来自下方的分类法]
  sources: [raw/articles/source-name.md]
  # 可选的质量信号：
  confidence: high | medium | low        # 声明的支持程度
  contested: true                        # 页面存在未解决的矛盾时设置
  contradictions: [other-page-slug]      # 与此页面冲突的页面
  ---
  ```

`confidence` 和 `contested` 是可选的，但对于观点密集或快速变化的
主题建议使用。Lint 会显示 `contested: true` 和 `confidence: low` 的页面
供审查，以防止薄弱的声明悄悄变成公认的 Wiki 事实。

### raw/ Frontmatter

原始来源也需要一个小的 frontmatter 块，以便重新摄取时可以检测变化：

```yaml
---
source_url: https://example.com/article   # 原始 URL（如适用）
ingested: YYYY-MM-DD
sha256: <frontmatter 下方原始内容的十六进制摘要>
---
```

`sha256:` 让未来对同一 URL 的重新摄取在内容未变时跳过处理，
并在内容变化时标记偏差。仅对正文（`---` 之后的所有内容）计算，
不对 frontmatter 本身计算。

## 标签分类法
[为领域定义 10-20 个顶级标签。在使用新标签之前先在此添加。]

AI/ML 示例：
- 模型：model, architecture, benchmark, training
- 人物/组织：person, company, lab, open-source
- 技术：optimization, fine-tuning, inference, alignment, data
- 元信息：comparison, timeline, controversy, prediction

规则：页面上的每个标签都必须出现在此分类法中。如果需要新标签，
先在此添加，然后再使用。这可以防止标签泛滥。

## 页面阈值
- **创建页面**：当实体/概念出现在 2+ 来源中或对某个来源至关重要时
- **添加到现有页面**：当来源提到了已有内容时
- **不创建页面**：对于顺便提及、次要细节或领域之外的内容
- **拆分页面**：当页面超过约 200 行时——拆分为带交叉链接的子主题
- **归档页面**：当内容被完全取代时——移至 `_archive/`，从索引中删除

## 实体页面
每个知名实体一个页面。包括：
- 概述/它是什么
- 关键事实和日期
- 与其他实体的关系（[[wikilinks]]）
- 来源引用

## 概念页面
每个概念或主题一个页面。包括：
- 定义/解释
- 当前知识状态
- 未解决的问题或争议
- 相关概念（[[wikilinks]]）

## 对比页面
并排分析。包括：
- 正在比较什么以及为什么
- 比较维度（首选表格格式）
- 结论或综合
- 来源

## 更新策略
当新信息与现有内容冲突时：
1. 检查日期——较新的来源通常取代较旧的
2. 如果确实矛盾，注明双方立场及日期和来源
3. 在 frontmatter 中标记矛盾：`contradictions: [page-name]`
4. 在 lint 报告中标记供用户审查
```

### index.md 模板

索引按类型分段。每个条目一行：wikilink + 摘要。

```markdown
# Wiki 索引

> 内容目录。每个 Wiki 页面按类型列出并附带一行摘要。
> 阅读此文件以查找与任何查询相关的页面。
> 最后更新：YYYY-MM-DD | 总页面数：N

## 实体
<!-- 分类内按字母顺序排列 -->

## 概念

## 对比

## 查询
```

**扩展规则：** 当任何分类超过 50 个条目时，按首字母或子领域拆分为子分类。当索引总计超过 200 个条目时，创建一个 `_meta/topic-map.md`，按主题分组页面以便更快导航。

### log.md 模板

```markdown
# Wiki 日志

> 所有 Wiki 操作的按时间顺序记录。仅追加。
> 格式：`## [YYYY-MM-DD] 操作 | 主题`
> 操作：ingest, update, query, lint, create, archive, delete
> 当此文件超过 500 条条目时，轮换：重命名为 log-YYYY.md，重新开始。

## [YYYY-MM-DD] create | Wiki 已初始化
- 领域：[领域]
- 已使用 SCHEMA.md、index.md、log.md 创建结构
```

## 核心操作

### 1. 摄取

当用户提供来源（URL、文件、粘贴内容）时，将其整合到 Wiki 中：

① **捕获原始来源：**
   - URL → 使用 `web_extract` 获取 Markdown，保存到 `raw/articles/`
   - PDF → 使用 `web_extract`（支持 PDF），保存到 `raw/papers/`
   - 粘贴文本 → 保存到适当的 `raw/` 子目录
   - 为文件取一个描述性名称：`raw/articles/karpathy-llm-wiki-2026.md`
   - **添加原始 frontmatter**（`source_url`、`ingested`、正文的 `sha256`）。
     重新摄取同一 URL 时：重新计算 sha256，与存储值比较——
     相同则跳过，不同则标记偏差并更新。这足够廉价，
     可以在每次重新摄取时执行，并能捕获静默的来源变化。

② **与用户讨论要点** ——什么有趣，对领域来说什么重要。
  （在自动化/cron 上下文中跳过——直接继续。）

③ **检查现有内容** ——搜索 index.md 并使用 `search_files` 查找
   提及的实体/概念的现有页面。这是区分
   增长的 Wiki 和一堆重复内容的关键。

④ **编写或更新 Wiki 页面：**
   - **新实体/概念：** 仅当满足 SCHEMA.md 中的页面阈值时创建页面
     （2+ 来源提及，或对某个来源至关重要）
   - **现有页面：** 添加新信息，更新事实，更新 `updated` 日期。
     当新信息与现有内容矛盾时，遵循更新策略。
   - **交叉引用：** 每个新建或更新的页面必须通过 `[[wikilinks]]`
     链接到至少 2 个其他页面。检查现有页面是否链接回来。
   - **标签：** 仅使用 SCHEMA.md 分类法中的标签
   - **来源追溯：** 在综合了 3+ 来源的页面上，对声明来自特定来源
     的段落附加 `^[raw/articles/source.md]` 标记。
   - **置信度：** 对于观点密集、快速变化或单一来源的声明，在 frontmatter
     中设置 `confidence: medium` 或 `low`。除非声明在多个来源中得到充分支持，
     否则不要标记为 `high`。

⑤ **更新导航：**
   - 将新页面按字母顺序添加到 `index.md` 的正确分类中
   - 更新索引标题中的"总页面数"和"最后更新"日期
   - 追加到 `log.md`：`## [YYYY-MM-DD] ingest | 来源标题`
   - 在日志条目中列出每个创建或更新的文件

⑥ **报告变更** ——向用户列出每个创建或更新的文件。

单个来源可以触发 5-15 个 Wiki 页面的更新。这是正常且预期的——这就是复合效应。

### 2. 查询

当用户询问关于 Wiki 领域的问题时：

① **阅读 `index.md`** 以识别相关页面。
② **对于 100+ 页面的 Wiki**，还要在所有 `.md` 文件中
   `search_files` 关键词——仅靠索引可能会遗漏相关内容。
③ 使用 `read_file` **阅读相关页面**。
④ **从已编译的知识中综合答案。** 引用你使用的 Wiki 页面：
   "基于 [[page-a]] 和 [[page-b]]..."
⑤ **将有价值的答案归档** ——如果答案是一个重要的对比、
   深入分析或新颖的综合，在 `queries/` 或 `comparisons/` 中创建页面。
   不要归档琐碎的查询——只归档那些难以重新推导的答案。
⑥ **更新 log.md** 记录查询及是否已归档。

### 3. Lint

当用户要求对 Wiki 进行 lint、健康检查或审计时：

① **孤立页面：** 查找没有来自其他页面的入站 `[[wikilinks]]` 的页面。
```python
# 使用 execute_code 执行——对所有 Wiki 页面进行程序化扫描
import os, re
from collections import defaultdict
wiki = "<WIKI_PATH>"
# 扫描 entities/、concepts/、comparisons/、queries/ 中的所有 .md 文件
# 提取所有 [[wikilinks]] ——建立入站链接映射
# 入站链接为零的页面是孤立页面
```

② **断裂的 wikilinks：** 查找指向不存在页面的 `[[links]]`。

③ **索引完整性：** 每个 Wiki 页面都应出现在 `index.md` 中。将文件系统与索引条目进行比较。

④ **Frontmatter 验证：** 每个 Wiki 页面必须具有所有必填字段
   （title、created、updated、type、tags、sources）。标签必须在分类法中。

⑤ **过期内容：** `updated` 日期比提及相同实体的最新来源
   超过 90 天的页面。

⑥ **矛盾：** 同一主题上具有冲突声明的页面。查找
   共享标签/实体但陈述不同事实的页面。列出所有带有
   `contested: true` 或 `contradictions:` frontmatter 的页面供用户审查。

⑦ **质量信号：** 列出 `confidence: low` 的页面以及任何仅引用
   单一来源但未设置 confidence 字段的页面——这些是
   寻找佐证或降级为 `confidence: medium` 的候选。

⑧ **来源偏差：** 对于 `raw/` 中每个带有 `sha256:` frontmatter 的文件，
   重新计算哈希值并标记不匹配。不匹配表示原始文件被编辑
   （不应该发生——raw/ 是不可变的）或从 URL 摄取后该 URL 内容已变。
   这不是硬错误，但值得报告。

⑨ **页面大小：** 标记超过 200 行的页面——拆分的候选。

⑩ **标签审计：** 列出所有使用中的标签，标记不在 SCHEMA.md 分类法中的标签。

⑪ **日志轮换：** 如果 log.md 超过 500 条条目，进行轮换。

⑫ **报告发现**，附带具体的文件路径和建议的操作，
    按严重程度分组（断裂链接 > 孤立页面 > 来源偏差 > 有争议的页面 > 过期内容 > 样式问题）。

⑬ **追加到 log.md：** `## [YYYY-MM-DD] lint | 发现 N 个问题`

## 使用 Wiki

### 搜索

```bash
# 按内容查找页面
search_files "transformer" path="$WIKI" file_glob="*.md"

# 按文件名查找页面
search_files "*.md" target="files" path="$WIKI"

# 按标签查找页面
search_files "tags:.*alignment" path="$WIKI" file_glob="*.md"

# 最近活动
read_file "$WIKI/log.md" offset=<last 20 lines>
```

### 批量摄取

当一次摄取多个来源时，批量处理更新：
1. 先读取所有来源
2. 识别所有来源中的所有实体和概念
3. 一次性检查所有现有页面（一次搜索，而非 N 次）
4. 一次性创建/更新页面（避免冗余更新）
5. 最后一次性更新 index.md
6. 编写一条覆盖整个批次的日志条目

### 归档

当内容被完全取代或领域范围变更时：
1. 如果不存在 `_archive/` 目录则创建
2. 将页面以原始路径移动到 `_archive/`（例如 `_archive/entities/old-page.md`）
3. 从 `index.md` 中删除
4. 更新所有链接到它的页面——将 wikilink 替换为纯文本 + "（已归档）"
5. 记录归档操作

### Obsidian 集成

Wiki 目录可以直接作为 Obsidian 库使用：
- `[[wikilinks]]` 渲染为可点击链接
- 图形视图可视化知识网络
- YAML frontmatter 支持 Dataview 查询
- `raw/assets/` 文件夹存放通过 `![[image.png]]` 引用的图片

最佳实践：
- 将 Obsidian 的附件文件夹设置为 `raw/assets/`
- 在 Obsidian 设置中启用"Wikilinks"（通常默认开启）
- 安装 Dataview 插件，用于类似 `TABLE tags FROM "entities" WHERE contains(tags, "company")` 的查询

如果同时使用 Obsidian 技能和此技能，将 `OBSIDIAN_VAULT_PATH` 设置为与 Wiki 路径相同的目录。

### Obsidian Headless（服务器和无头机器）

在没有显示器的机器上，使用 `obsidian-headless` 代替桌面应用。
它通过 Obsidian Sync 同步库，无需 GUI——非常适合在服务器上运行代理
写入 Wiki 而 Obsidian 桌面版在另一台设备上阅读。

**设置：**
```bash
# 需要 Node.js 22+
npm install -g obsidian-headless

# 登录（需要具有 Sync 订阅的 Obsidian 账户）
ob login --email <email> --password '<password>'

# 创建一个远程库用于 Wiki
ob sync-create-remote --name "LLM Wiki"

# 将 Wiki 目录连接到库
cd ~/wiki
ob sync-setup --vault "<vault-id>"

# 初始同步
ob sync

# 持续同步（前台——使用 systemd 后台运行）
ob sync --continuous
```

**通过 systemd 持续后台同步：**
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
# 启用 linger 以便同步在注销后继续：
sudo loginctl enable-linger $USER
```

这让代理在服务器上写入 `~/wiki`，而你可以在笔记本电脑/手机上
通过 Obsidian 浏览同一个库——更改在几秒内出现。

## 常见陷阱

- **永远不要修改 `raw/` 中的文件** ——来源是不可变的。更正放在 Wiki 页面中。
- **始终先定位** ——在新会话中任何操作之前，先阅读 SCHEMA + index + 最近的日志。
  跳过此步骤会导致重复和遗漏交叉引用。
- **始终更新 index.md 和 log.md** ——跳过此步骤会使 Wiki 退化。这些是
  导航的骨架。
- **不要为顺便提及创建页面** ——遵循 SCHEMA.md 中的页面阈值。脚注中出现一次的
  名字不值得创建实体页面。
- **不要创建没有交叉引用的页面** ——孤立的页面是不可见的。每个页面必须
  链接到至少 2 个其他页面。
- **Frontmatter 是必需的** ——它支持搜索、过滤和过期检测。
- **标签必须来自分类法** ——自由标签会退化为噪声。先在 SCHEMA.md 中
  添加新标签，然后再使用。
- **保持页面可扫描** ——一个 Wiki 页面应该能在 30 秒内读完。超过
  200 行的页面进行拆分。将详细分析移到专门的深入页面。
- **批量更新前先询问** ——如果一个摄取会涉及 10+ 个现有页面，先
  与用户确认范围。
- **轮换日志** ——当 log.md 超过 500 条条目时，将其重命名为 `log-YYYY.md` 并重新开始。
  代理应在 lint 期间检查日志大小。
- **显式处理矛盾** ——不要静默覆盖。同时注明两个声明及日期，
  在 frontmatter 中标记，供用户审查。

## 相关工具

[llm-wiki-compiler](https://github.com/atomicmemory/llm-wiki-compiler) 是一个 Node.js CLI 工具，
将来源编译为具有相同 Karpathy 灵感的概念 Wiki。它与 Obsidian 兼容，
因此想要定时/CLI 驱动的编译流水线的用户可以将其指向
此技能维护的同一个库。权衡：它拥有页面生成的控制权（替代了代理的页面创建判断），
并针对小型语料库进行了优化。当你需要人在环中的策展时使用此技能；
当你想对源目录进行批量编译时使用 llmwiki。
