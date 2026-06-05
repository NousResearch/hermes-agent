---
title: "研究论文写作 — 为 NeurIPS/ICML/ICLR 撰写 ML 论文：设计→投稿"
sidebar_label: "研究论文写作"
description: "为 NeurIPS/ICML/ICLR 撰写 ML 论文：设计→投稿"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 研究论文写作

为 NeurIPS/ICML/ICLR 撰写 ML 论文：设计→投稿。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/research/research-paper-writing` |
| 版本 | `1.1.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `semanticscholar`, `arxiv`, `habanero`, `requests`, `scipy`, `numpy`, `matplotlib`, `SciencePlots` |
| 平台 | linux, macos |
| 标签 | `Research`, `Paper Writing`, `Experiments`, `ML`, `AI`, `NeurIPS`, `ICML`, `ICLR`, `ACL`, `AAAI`, `COLM`, `LaTeX`, `Citations`, `Statistical Analysis` |
| 相关 skill | [`arxiv`](/user-guide/skills/bundled/research/research-arxiv), `ml-paper-writing`, [`subagent-driven-development`](/user-guide/skills/bundled/software-development/software-development-subagent-driven-development), [`plan`](/user-guide/skills/bundled/software-development/software-development-plan) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 研究论文写作流水线

面向 **NeurIPS、ICML、ICLR、ACL、AAAI 和 COLM** 的端到端 ML/AI 研究论文生产流水线，覆盖完整研究生命周期：实验设计、执行、监控、分析、论文撰写、审稿、修改与投稿。

这**不是线性流水线**——它是一个迭代循环。结果会触发新实验，审稿意见会触发新分析。agent 必须处理这些反馈循环。

<!-- ascii-guard-ignore -->
<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH PAPER PIPELINE                  │
│                                                             │
│  Phase 0: Project Setup ──► Phase 1: Literature Review      │
│       │                          │                          │
│       ▼                          ▼                          │
│  Phase 2: Experiment     Phase 5: Paper Drafting ◄──┐      │
│       Design                     │                   │      │
│       │                          ▼                   │      │
│       ▼                    Phase 6: Self-Review      │      │
│  Phase 3: Execution &           & Revision ──────────┘      │
│       Monitoring                 │                          │
│       │                          ▼                          │
│       ▼                    Phase 7: Submission               │
│  Phase 4: Analysis ─────► (feeds back to Phase 2 or 5)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->
<!-- ascii-guard-ignore-end -->

---

## 何时使用本 Skill

在以下情况下使用本 skill：
- **从现有代码库或想法开始撰写新研究论文**
- **设计并运行实验**以支撑论文论点
- **撰写或修改**研究论文的任意章节
- **为特定会议或研讨会准备投稿**
- **根据审稿意见**补充实验或修改论文
- **在不同会议格式之间转换**论文
- **撰写非实证类论文**——理论、综述、基准测试或立场论文（参见[超越实证 ML 的论文类型](#paper-types-beyond-empirical-ml)）
- **为 NLP、HCI 或对齐研究设计人工评估**
- **准备接收后的交付物**——海报、演讲、代码发布

## 核心理念

1. **主动出击。** 交付完整草稿，而非提问。科学家很忙——先产出具体内容供其反应，再迭代。
2. **绝不捏造引用。** AI 生成的引用错误率约 40%。始终以编程方式获取。无法核实的引用标记为 `[CITATION NEEDED]`。
3. **论文是一个故事，而非实验的堆砌。** 每篇论文都需要用一句话清晰陈述贡献。做不到这一点，论文就还没准备好。
4. **实验服务于论点。** 每个实验都必须明确说明它支撑哪个论点。绝不运行与论文叙事无关的实验。
5. **尽早提交，频繁提交。** 每完成一批实验、每次更新论文草稿都要提交，并附上描述性 commit 信息。Git 日志就是实验历史。

### 主动性与协作

**默认：主动出击。先起草，再附草稿提问。**

| 置信度 | 行动 |
|--------|------|
| **高**（代码库清晰，贡献明确） | 写完整草稿，交付，根据反馈迭代 |
| **中**（存在一定歧义） | 写草稿并标注不确定之处，继续推进 |
| **低**（存在重大未知） | 通过 `clarify` 提 1-2 个针对性问题，然后起草 |

| 章节 | 是否自主起草？ | 随草稿标注 |
|------|--------------|-----------|
| 摘要 | 是 | "将贡献框架为 X——如需调整请告知" |
| 引言 | 是 | "强调了问题 Y——如有误请纠正" |
| 方法 | 是 | "包含了细节 A、B、C——请补充遗漏部分" |
| 实验 | 是 | "突出了结果 1、2、3——如需重排请告知" |
| 相关工作 | 是 | "引用了论文 X、Y、Z——如有遗漏请补充" |

**仅在以下情况等待输入**：目标会议不明确、存在多个相互矛盾的框架、结果似乎不完整、明确要求先审阅。

---

## 阶段 0：项目设置

**目标**：建立工作空间，了解现有工作，明确贡献点。

### 步骤 0.1：探索代码库

```bash
# 了解项目结构
ls -la
find . -name "*.py" | head -30
find . -name "*.md" -o -name "*.txt" | xargs grep -l -i "result\|conclusion\|finding"
```

关注：
- `README.md` — 项目概述与论点
- `results/`、`outputs/`、`experiments/` — 现有发现
- `configs/` — 实验配置
- `.bib` 文件 — 现有引用
- 草稿文档或笔记

### 步骤 0.2：组织工作空间

建立一致的工作空间结构：

```
workspace/
  paper/               # LaTeX 源文件、图表、编译后的 PDF
  experiments/         # 实验运行脚本
  code/                # 核心方法实现
  results/             # 原始实验结果（自动生成）
  tasks/               # 任务/基准定义
  human_eval/          # 人工评估材料（如需要）
```

### 步骤 0.3：设置版本控制

```bash
git init  # 如果尚未初始化
git remote add origin <repo-url>
git checkout -b paper-draft  # 或 main
```

**Git 规范**：每完成一批实验都要提交，附上描述性信息。示例：
```
Add Monte Carlo constrained results (5 runs, Sonnet 4.6, policy memo task)
Add Haiku baseline comparison: autoreason vs refinement baselines at cheap model tier
```

### 步骤 0.4：明确贡献点

在撰写任何内容之前，先阐明：
- **是什么**：这篇论文贡献的单一事项是什么？
- **为什么**：有哪些证据支撑？
- **意义何在**：读者为何应该关注？

> 向科学家提议："根据我的理解，主要贡献是：[一句话]。关键结果显示 [Y]。这是您想要的框架吗？"

### 步骤 0.5：创建 TODO 列表

使用 `todo` 工具创建结构化项目计划：

```
Research Paper TODO:
- [ ] Define one-sentence contribution
- [ ] Literature review (related work + baselines)
- [ ] Design core experiments
- [ ] Run experiments
- [ ] Analyze results
- [ ] Write first draft
- [ ] Self-review (simulate reviewers)
- [ ] Revise based on review
- [ ] Submission prep
```

在整个项目过程中持续更新。它是跨会话的持久状态。

### 步骤 0.6：估算计算预算

在运行实验之前，估算总成本和时间：

```
Compute Budget Checklist:
- [ ] API costs: (model price per token) × (estimated tokens per run) × (number of runs)
- [ ] GPU hours: (time per experiment) × (number of experiments) × (number of seeds)
- [ ] Human evaluation costs: (annotators) × (hours) × (hourly rate)
- [ ] Total budget ceiling and contingency (add 30-50% for reruns)
```

随着实验运行跟踪实际支出：
```python
# Simple cost tracker pattern
import json, os
from datetime import datetime

COST_LOG = "results/cost_log.jsonl"

def log_cost(experiment: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    with open(COST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**预算紧张时**：在进行完整扫描之前，先运行试点实验（1-2 个随机种子，任务子集）。调试流水线时使用更便宜的模型，最终运行时再切换到目标模型。

### 步骤 0.7：多作者协调

大多数论文有 3-10 位作者。尽早建立工作流程：

| 工作流 | 工具 | 适用场景 |
|--------|------|----------|
| **Overleaf** | 基于浏览器 | 多作者同时编辑，无 git 经验 |
| **Git + LaTeX** | `git` 配合 `.gitignore` 排除辅助文件 | 技术团队，需要基于分支的审阅 |
| **Overleaf + Git 同步** | Overleaf 高级版 | 两全其美——实时协作加版本历史 |

**章节所有权**：每个章节指定一位主要作者。其他人只评论，不直接编辑。防止合并冲突和风格不一致。

```
Author Coordination Checklist:
- [ ] Agree on section ownership (who writes what)
- [ ] Set up shared workspace (Overleaf or git repo)
- [ ] Establish notation conventions (before anyone writes)
- [ ] Schedule internal review rounds (not just at the end)
- [ ] Designate one person for final formatting pass
- [ ] Agree on figure style (colors, fonts, sizes) before creating figures
```

**需要提前约定的 LaTeX 规范**：
- `\method{}` 宏，用于统一方法命名
- 引用风格：`\citet{}` 与 `\citep{}` 的使用规则
- 数学符号：小写粗体表示向量，大写粗体表示矩阵，等
- 英式拼写与美式拼写

---

## 阶段 1：文献综述

**目标**：查找相关工作，确定基线，收集引用。

### 步骤 1.1：确定种子论文

从代码库中已引用的论文出发：

```bash
# 通过终端：
grep -r "arxiv\|doi\|cite" --include="*.md" --include="*.bib" --include="*.py"
find . -name "*.bib"
```

### 步骤 1.2：搜索相关工作

**加载 `arxiv` skill** 进行结构化论文发现：`skill_view("arxiv")`。它提供 arXiv REST API 搜索、Semantic Scholar 引用图谱、作者档案和 BibTeX 生成。

使用 `web_search` 进行广泛发现，使用 `web_extract` 获取特定论文：

```
# 通过 web_search：
web_search("[main technique] + [application domain] site:arxiv.org")
web_search("[baseline method] comparison ICML NeurIPS 2024")

# 通过 web_extract（针对特定论文）：
web_extract("https://arxiv.org/abs/2303.17651")
```

其他可尝试的搜索查询：

```
Search queries:
- "[main technique] + [application domain]"
- "[baseline method] comparison"
- "[problem name] state-of-the-art"
- Author names from existing citations
```

**推荐**：安装 **Exa MCP** 进行实时学术搜索：
```bash
claude mcp add exa -- npx -y mcp-remote "https://mcp.exa.ai/mcp"
```

### 步骤 1.2b：深化搜索（先广度，后深度）

单轮扁平搜索通常会遗漏重要的相关工作。使用受深度研究流水线启发的迭代**先广后深**模式：

```
Iterative Literature Search:

Round 1 (Breadth): 4-6 parallel queries covering different angles
  - "[method] + [domain]"
  - "[problem name] state-of-the-art 2024 2025"
  - "[baseline method] comparison"
  - "[alternative approach] vs [your approach]"
  → Collect papers, extract key concepts and terminology

Round 2 (Depth): Generate follow-up queries from Round 1 learnings
  - New terminology discovered in Round 1 papers
  - Papers cited by the most relevant Round 1 results
  - Contradictory findings that need investigation
  → Collect papers, identify remaining gaps

Round 3 (Targeted): Fill specific gaps
  - Missing baselines identified in Rounds 1-2
  - Concurrent work (last 6 months, same problem)
  - Key negative results or failed approaches
  → Stop when new queries return mostly papers you've already seen
```

**何时停止**：如果某轮搜索返回的论文中 >80% 已在你的收藏中，则搜索已饱和。通常 2-3 轮即可。综述论文预计需要 4-5 轮。

**基于 agent 的工作流**：通过 `delegate_task` 并行委派每轮查询。收集结果，去重，然后从综合所得中生成下一轮查询。

### 步骤 1.3：核实每条引用

**绝不从记忆中生成 BibTeX。始终以编程方式获取。**

对每条引用，遵循强制性的 5 步流程：

```
Citation Verification (MANDATORY per citation):
1. SEARCH → Query Semantic Scholar or Exa MCP with specific keywords
2. VERIFY → Confirm paper exists in 2+ sources (Semantic Scholar + arXiv/CrossRef)
3. RETRIEVE → Get BibTeX via DOI content negotiation (programmatically, not from memory)
4. VALIDATE → Confirm the claim you're citing actually appears in the paper
5. ADD → Add verified BibTeX to bibliography
If ANY step fails → mark as [CITATION NEEDED], inform scientist
```

```python
# Fetch BibTeX via DOI
import requests

def doi_to_bibtex(doi: str) -> str:
    response = requests.get(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex"}
    )
    response.raise_for_status()
    return response.text
```

如果无法核实某条引用：

```latex
\cite{PLACEHOLDER_author2024_verify_this}  % TODO: Verify this citation exists
```

**务必告知科学家**："我已将 [X] 条引用标记为需要核实的占位符。"

完整 API 文档和 `CitationManager` 类请参见 [references/citation-workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/citation-workflow.md)。

### 步骤 1.4：整理相关工作

按方法论分组，而非逐篇论文列举：

**好的写法**："一类工作使用 X 的假设 [refs]，而我们使用 Y 的假设，因为……"
**不好的写法**："Smith 等人提出了 X。Jones 等人提出了 Y。我们将两者结合。"

---

## 阶段 2：实验设计

**目标**：设计直接支撑论文论点的实验。每个实验都必须回答一个具体问题。

### 步骤 2.1：将论点映射到实验

创建明确的映射关系：

| 论点 | 实验 | 预期证据 |
|------|------|----------|
| "我们的方法优于基线" | 主要对比（表 1） | 胜率、统计显著性 |
| "效果在较弱模型上更显著" | 模型规模研究 | 单调递增曲线 |
| "收敛需要范围约束" | 有约束 vs 无约束 | 收敛速率对比 |

**规则**：如果某个实验无法映射到某个论点，就不要运行它。

### 步骤 2.2：设计基线

强基线是区分被接收论文与被拒绝论文的关键。审稿人会问："他们有没有与 X 进行对比？"

标准基线类别：
- **朴素基线**：最简单的可行方法
- **强基线**：已知最佳的现有方法
- **消融基线**：去掉某一组件的你的方法
- **计算量匹配基线**：相同计算预算，不同分配方式

### 步骤 2.3：定义评估协议

在运行任何实验之前，明确：
- **指标**：测量什么，方向符号（越高/越低越好）
- **聚合方式**：如何跨运行/任务汇总结果
- **统计检验**：用什么检验来确立显著性
- **样本量**：运行/问题/任务的数量

### 步骤 2.4：编写实验脚本

遵循成功研究流水线中的以下模式：

**增量保存**——每步后保存结果，以便崩溃恢复：
```python
# Save after each problem/task
result_path = f"results/{task}/{strategy}/result.json"
if os.path.exists(result_path):
    continue  # Skip already-completed work
# ... run experiment ...
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)
```

**制品保存**——保存所有中间输出：
```
results/<experiment>/
  <task>/
    <strategy>/
      final_output.md          # Final result
      history.json             # Full trajectory
      pass_01/                 # Per-iteration artifacts
        version_a.md
        version_b.md
        critic.md
```

**关注点分离**——将生成、评估和可视化分开：
```
run_experiment.py              # Core experiment runner
run_baselines.py               # Baseline comparison
run_comparison_judge.py        # Blind evaluation
analyze_results.py             # Statistical analysis
make_charts.py                 # Visualization
```

完整设计模式、cron 监控和错误恢复请参见 [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md)。

### 步骤 2.5：设计人工评估（如适用）

许多 NLP、HCI 和对齐研究论文需要人工评估作为主要或补充证据。在运行自动化实验之前先设计好——人工评估通常有更长的准备周期（IRB 审批、招募标注员）。

**何时需要人工评估：**
- 自动化指标无法捕捉你关心的内容（流畅性、有用性、安全性）
- 你的贡献涉及面向人类的质量（可读性、偏好、信任度）
- NLP 会议（ACL、EMNLP）的审稿人对生成任务有此期望

**关键设计决策：**

| 决策 | 选项 | 指导 |
|------|------|------|
| **标注员类型** | 专家、众包工人、终端用户 | 与你的论点要求相匹配 |
| **量表** | Likert（1-5）、成对比较、排序 | 对 LLM 输出而言，成对比较比 Likert 更可靠 |
| **样本量** | 每位标注员及总条目数 | 功效分析，或最少 100 条、3+ 位标注员 |
| **一致性指标** | Cohen's kappa、Krippendorff's alpha、ICC | 2 位以上标注员用 Krippendorff's alpha；同时报告原始一致率 |
| **平台** | Prolific、MTurk、内部团队 | Prolific 质量好；MTurk 规模大；内部团队适合领域专业知识 |

**标注指南清单：**
```
- [ ] Clear task description with examples (good AND bad)
- [ ] Decision criteria for ambiguous cases
- [ ] At least 2 worked examples per category
- [ ] Attention checks / gold standard items (10-15% of total)
- [ ] Qualification task or screening round
- [ ] Estimated time per item and fair compensation (>= local minimum wage)
- [ ] IRB/ethics review if required by your institution
```

**报告要求**（审稿人会逐一核查）：
- 标注员数量及其资质
- 标注员间一致性，含具体指标和数值
- 报酬详情（金额、估计时薪）
- 标注界面描述或截图（附录）
- 总标注时间

完整指南（含人工评估数据的统计检验、众包质量控制模式和 IRB 指导）请参见 [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md)。

---

## 阶段 3：实验执行与监控

**目标**：可靠地运行实验，监控进度，从故障中恢复。

### 步骤 3.1：启动实验

对长时间运行的实验使用 `nohup`：

```bash
nohup python run_experiment.py --config config.yaml > logs/experiment_01.log 2>&1 &
echo $!  # Record the PID
```

**并行执行**：同时运行独立实验，但注意 API 速率限制。在同一 API 上并发 4+ 个实验会使每个实验都变慢。

### 步骤 3.2：设置监控（Cron 模式）

对于长时间运行的实验，设置定期状态检查。Cron prompt（提示词）应遵循以下模板：

```
Monitor Prompt Template:
1. Check if process is still running: ps aux | grep <pattern>
2. Read last 30 lines of log: tail -30 <logfile>
3. Check for completed results: ls <result_dir>
4. If results exist, read and report: cat <result_file>
5. If all done, commit: git add -A && git commit -m "<descriptive message>" && git push
6. Report in structured format (tables with key metrics)
7. Answer the key analytical question for this experiment
```

**静默模式**：如果自上次检查以来没有任何变化，回复 `[SILENT]` 以抑制对用户的通知。仅在有新情况时报告。

### 步骤 3.3：处理故障

常见故障模式及恢复方法：

| 故障 | 检测 | 恢复 |
|------|------|------|
| API 速率限制/额度耗尽 | 日志中出现 402/429 错误 | 等待后重新运行（脚本会跳过已完成的工作） |
| 进程崩溃 | PID 消失，结果不完整 | 从最后一个检查点重新运行 |
| 难题超时 | 进程卡住，日志无进展 | 终止并跳过，在结果中记录 |
| 模型 ID 错误 | 日志中出现引用模型名称的错误 | 修正 ID 后重新运行 |

**关键**：脚本应始终检查现有结果并跳过已完成的工作。这使重新运行安全高效。

### 步骤 3.4：提交已完成的结果

每批实验完成后：

```bash
git add -A
git commit -m "Add <experiment name>: <key finding in 1 line>"
git push
```

### 步骤 3.5：维护实验日志

Git commit 记录发生了什么，但不记录**探索树**——即根据所学内容决定下一步尝试什么。维护一个结构化的实验日志来捕捉这棵树：

```json
// experiment_journal.jsonl — append one entry per experiment attempt
{
  "id": "exp_003",
  "parent": "exp_001",
  "timestamp": "2025-05-10T14:30:00Z",
  "hypothesis": "Adding scope constraints will fix convergence failure from exp_001",
  "plan": "Re-run autoreason with max_tokens=2000 and fixed structure template",
  "config": {"model": "haiku", "strategy": "autoreason", "max_tokens": 2000},
  "status": "completed",
  "result_path": "results/exp_003/",
  "key_metrics": {"win_rate": 0.85, "convergence_rounds": 3},
  "analysis": "Scope constraints fixed convergence. Win rate jumped from 0.42 to 0.85.",
  "next_steps": ["Try same constraints on Sonnet", "Test without structure template"],
  "figures": ["figures/exp003_convergence.pdf"]
}
```

**为什么要日志，而不只是 git？** Git 跟踪文件变更。日志跟踪推理过程：为什么尝试 X，学到了什么，以及这对下一个实验意味着什么。撰写论文时，这棵树对方法章节（"我们观察到 X，这促使我们尝试 Y"）和诚实报告失败至关重要。

**选择最佳路径**：当日志显示分支树（exp_001 → exp_002a、exp_002b、exp_003）时，找出最能支撑论文论点的路径。在附录中将死胡同分支记录为消融实验或负面结果。

**每次实验后快照代码**：
```bash
cp experiment.py results/exp_003/experiment_snapshot.py
```
即使后续代码发生变化，也能精确复现。

---

## 阶段 4：结果分析

**目标**：提取发现，计算统计数据，找出故事主线。

### 步骤 4.1：汇总结果

编写分析脚本，完成以下工作：
1. 从一批结果文件中加载所有数据
2. 计算每个任务和总体指标
3. 生成汇总表格

```python
# Standard analysis pattern
import json, os
from pathlib import Path

results = {}
for result_file in Path("results/").rglob("result.json"):
    data = json.loads(result_file.read_text())
    strategy = result_file.parent.name
    task = result_file.parent.parent.name
    results.setdefault(strategy, {})[task] = data

# Compute aggregate metrics
for strategy, tasks in results.items():
    scores = [t["score"] for t in tasks.values()]
    print(f"{strategy}: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
```

### 步骤 4.2：统计显著性

始终计算：
- **误差棒**：标准差或标准误，注明使用哪种
- **置信区间**：关键结果的 95% CI
- **成对检验**：McNemar 检验用于比较两种方法
- **效应量**：Cohen's d 或 h 用于实际显著性

McNemar 检验、自举 CI 和 Cohen's h 的完整实现请参见 [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md)。

### 步骤 4.3：找出故事主线

分析后，明确回答：
1. **主要发现是什么？** 用一句话陈述。
2. **什么让你感到意外？** 意外结果往往造就最好的论文。
3. **什么失败了？** 失败的实验往往最具信息量。诚实报告失败会增强论文说服力。
4. **需要哪些后续实验？** 结果往往会引发新问题。

#### 处理负面或零结果

当你的假设被证伪或结果不确定时，有三种选择：

| 情况 | 行动 | 适合的会议 |
|------|------|-----------|
| 假设错误，但**原因**有信息量 | 围绕原因分析来框架论文 | NeurIPS、ICML（如果分析严谨） |
| 方法未超越基线，但**揭示了新东西** | 将贡献重新框架为理解/分析 | ICLR（重视理解）、研讨会论文 |
| 对流行论断的干净负面结果 | 写出来——该领域需要知道 | NeurIPS Datasets & Benchmarks、TMLR、研讨会 |
| 结果不确定，没有清晰故事 | 转向——运行不同实验或重新框架 | 不要强行写一篇不成立的论文 |

**如何撰写负面结果论文：**
- 以社区的既有信念及其重要性开篇
- 描述你严谨的方法论（必须无懈可击——审稿人会更严格审查）
- 用统计证据清晰呈现零结果
- 分析**为什么**预期结果没有出现
- 讨论对该领域的影响

**明确欢迎负面结果的会议**：NeurIPS（Datasets & Benchmarks 赛道）、TMLR、ML Reproducibility Challenge、各大会议的研讨会。部分研讨会专门征集负面结果。

### 步骤 4.4：创建图表

**图形**：
- 所有图表使用矢量图（PDF）：`plt.savefig('fig.pdf')`
- 色盲友好调色板（Okabe-Ito 或 Paul Tol）
- 自包含的图注——读者无需阅读正文即可理解
- 图形内部不加标题——图注承担此功能

**表格**：
- 使用 `booktabs` LaTeX 包
- 每个指标的最佳值加粗
- 包含方向符号（越高/越低越好）
- 小数精度一致

```latex
\usepackage{booktabs}
\begin{tabular}{lcc}
\toprule
Method & Accuracy $\uparrow$ & Latency $\downarrow$ \\
\midrule
Baseline & 85.2 & 45ms \\
\textbf{Ours} & \textbf{92.1} & 38ms \\
\bottomrule
\end{tabular}
```

### 步骤 4.5：决策：继续实验还是开始写作？

| 情况 | 行动 |
|------|------|
| 核心论点已支撑，结果显著 | 进入阶段 5（写作） |
| 结果不确定，需要更多数据 | 返回阶段 2（设计） |
| 意外发现提示新方向 | 返回阶段 2（设计） |
| 缺少审稿人会问的某个消融实验 | 运行它，然后进入阶段 5 |
| 所有实验完成但部分失败 | 记录失败，进入阶段 5 |

### 步骤 4.6：撰写实验日志（写作前的桥梁）

在进入论文写作之前，创建一个将结果与文字连接起来的结构化实验日志。这是实验与写作之间最重要的连接纽带——没有它，写作 agent 必须从原始结果文件中重新推导故事。

**创建 `experiment_log.md`**，结构如下：

```markdown
# Experiment Log

## Contribution (one sentence)
[The paper's main claim]

## Experiments Run

### Experiment 1: [Name]
- **Claim tested**: [Which paper claim this supports]
- **Setup**: [Model, dataset, config, number of runs]
- **Key result**: [One sentence with the number]
- **Result files**: results/exp1/final_info.json
- **Figures generated**: figures/exp1_comparison.pdf
- **Surprising findings**: [Anything unexpected]

### Experiment 2: [Name]
...

## Figures
| Filename | Description | Which section it belongs in |
|----------|-------------|---------------------------|
| figures/main_comparison.pdf | Bar chart comparing all methods on benchmark X | Results, Figure 2 |
| figures/ablation.pdf | Ablation removing components A, B, C | Results, Figure 3 |
...

## Failed Experiments (document for honesty)
- [What was tried, why it failed, what it tells us]

## Open Questions
- [Anything the results raised that the paper should address]
```

**为什么重要**：起草时，agent（或委派的子 agent）可以加载 `experiment_log.md` 和 LaTeX 模板，生成基于实际结果的初稿。没有这座桥梁，写作 agent 必须解析原始 JSON/CSV 文件并推断故事——这是捏造或误报数字的常见根源。

**Git 规范**：将此日志与它所描述的结果一起提交。

---

## 迭代精炼：策略选择

本流水线中的任何输出——论文草稿、实验脚本、分析——都可以迭代精炼。autoreason 研究提供了经验证据，说明每种精炼策略何时有效、何时失败。使用本节选择正确的方法。

### 快速决策表

| 你的情况 | 策略 | 原因 |
|----------|------|------|
| 中等模型 + 受约束任务 | **Autoreason** | 最佳甜蜜点。生成-评估差距最大。基线会主动破坏弱模型输出。 |
| 中等模型 + 开放任务 | 添加范围约束的 **Autoreason** | 添加固定事实、结构或可交付物来限定改进空间。 |
| 前沿模型 + 受约束任务 | **Autoreason** | 即使在前沿模型上，2/3 受约束任务也能获胜。 |
| 前沿模型 + 无约束任务 | **批评-修改** 或 **单次通过** | Autoreason 排最后。模型自我评估已足够好。 |
| 具体技术任务（系统设计） | **批评-修改** | 直接的查找-修复循环更高效。 |
| 模板填充任务（只有一种正确结构） | **单次通过** 或 **保守策略** | 决策空间极小。迭代无附加价值。 |
| 带测试用例的代码 | **Autoreason（代码变体）** | 在修复前对*失败原因*进行结构化分析。恢复率 62% vs 43%。 |
| 极弱模型（Llama 8B 级别） | **单次通过** | 模型太弱，无法生成多样候选。投资于生成质量。 |

### 生成-评估差距

**核心洞见**：Autoreason 的价值取决于模型生成能力与自我评估能力之间的差距。

<!-- ascii-guard-ignore -->
```
Model Tier        │ Generation │ Self-Eval │ Gap    │ Autoreason Value
──────────────────┼────────────┼───────────┼────────┼─────────────────
Weak (Llama 8B)   │ Poor       │ Poor      │ Small  │ None — can't generate diverse candidates
Mid (Haiku 3.5)   │ Decent     │ Poor      │ LARGE  │ MAXIMUM — 42/42 perfect Borda
Mid (Gemini Flash)│ Decent     │ Moderate  │ Large  │ High — wins 2/3
Strong (Sonnet 4) │ Good       │ Decent    │ Medium │ Moderate — wins 3/5
Frontier (S4.6)   │ Excellent  │ Good      │ Small  │ Only with constraints
```
<!-- ascii-guard-ignore-end -->

这种差距是结构性的，而非暂时的。随着成本下降，今天的前沿模型会成为明天的中等模型。甜蜜点会移动，但永远不会消失。

### Autoreason 循环（摘要）

每次迭代由来自全新、隔离 agent 的三个候选组成：

1. **批评者** → 找出现有方案 A 的问题（不修复）
2. **作者 B** → 根据批评修改 A
3. **综合者** → 合并 A 和 B（随机化标签）
4. **评判小组** → 3 位盲评 CoT 评判者通过 Borda 计数对 A、B、AB 排名
5. **收敛** → A 连续赢得 k=2 次 → 完成

**关键参数：**
- k=2 收敛（k=1 过早，k=3 太贵，无质量提升）
- 始终使用 CoT 评判者（收敛速度快 3 倍）
- 作者温度 0.8，评判者温度 0.3
- 保守平局处理：现有方案赢得平局
- 每个角色都是无共享上下文的全新 agent

### 应用于论文草稿

通过 autoreason 精炼论文本身时：
- **向批评者提供真实数据**：实际实验数据、结果 JSON、统计输出。没有这些，模型会捏造虚假的消融研究和假置信区间。
- **至少使用 3 位有效评判者**：一个损坏的评判者解析器不会增加噪声——它会完全阻止均衡的达成。
- **范围约束修改**："解决这些具体弱点"，而非"改进论文"。

### 失败模式

| 失败 | 检测 | 修复 |
|------|------|------|
| 不收敛（A 从不获胜） | 20+ 次迭代中 A 获胜率 &lt;15% | 为任务添加范围约束 |
| 综合漂移 | 字数无限增长 | 约束结构和可交付物 |
| 退化至单次通过以下 | 基线得分高于迭代输出 | 切换到单次通过；模型可能太弱 |
| 过拟合（代码） | 公开测试通过率高，私有测试通过率低 | 使用结构化分析，而非仅依赖测试反馈 |
| 评判者损坏 | 解析失败导致小组人数低于 3 | 先修复解析器再继续 |

完整 prompt（提示词）、Borda 计分细节、模型选择指南、范围约束设计模式和计算预算参考请参见 [references/autoreason-methodology.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/autoreason-methodology.md)。

---

## 阶段 5：论文起草

**目标**：撰写完整的、可发表的论文。

### 大型项目的上下文管理

一个包含 50+ 个实验文件、多个结果目录和大量文献笔记的论文项目，很容易超出 agent 的上下文窗口。主动管理这一问题：

**每个起草任务加载到上下文的内容：**

| 起草任务 | 加载到上下文 | 不要加载 |
|----------|------------|---------|
| 撰写引言 | `experiment_log.md`、贡献陈述、5-10 篇最相关论文的摘要 | 原始结果 JSON、完整实验脚本、所有文献笔记 |
| 撰写方法 | 实验配置、伪代码、架构描述 | 原始日志、其他实验的结果 |
| 撰写结果 | `experiment_log.md`、结果汇总表、图表列表 | 完整分析脚本、中间数据 |
| 撰写相关工作 | 整理好的引用笔记（步骤 1.4 的输出）、.bib 文件 | 实验文件、原始 PDF |
| 修改 | 完整论文草稿、具体审稿人意见 | 其他所有内容 |

**原则：**
- **`experiment_log.md` 是主要的上下文桥梁**——它汇总了写作所需的一切，无需加载原始数据文件（参见步骤 4.6）
- **委派时每次只加载一个章节的上下文。** 起草方法章节的子 agent 不需要文献综述笔记。
- **汇总，而非包含原始文件。** 对于 200 行的结果 JSON，加载 10 行汇总表。对于 50 页的相关论文，加载 5 句摘要 + 你关于其相关性的 2 行笔记。
- **对于非常大的项目**：创建 `context/` 目录，存放预压缩的摘要：
  ```
  context/
    contribution.md          # 1 sentence
    experiment_summary.md    # Key results table (from experiment_log.md)
    literature_map.md        # Organized citation notes
    figure_inventory.md      # List of figures with descriptions
  ```

### 叙事原则

**最关键的洞见**：你的论文不是实验的集合——它是一个有一个清晰贡献、由证据支撑的故事。

每篇成功的 ML 论文都围绕 Neel Nanda 所说的"叙事"展开：一个简短、严谨、基于证据的技术故事，读者会关心其结论。

**三大支柱（引言结束时必须清晰）：**

| 支柱 | 描述 | 检验 |
|------|------|------|
| **是什么** | 1-3 个具体的新颖论点 | 能用一句话陈述吗？ |
| **为什么** | 严谨的实证证据 | 实验能将你的假设与其他假设区分开吗？ |
| **意义何在** | 读者为何应该关注 | 这与社区认可的问题相关联吗？ |

**如果你无法用一句话陈述你的贡献，你还没有一篇论文。**

### 本指导的来源

本 skill 综合了在顶级会议上发表过大量论文的研究者的写作理念。写作理念层最初由 [Orchestra Research](https://github.com/orchestra-research) 作为 `ml-paper-writing` skill 编写。

| 来源 | 主要贡献 | 链接 |
|------|----------|------|
| **Neel Nanda**（Google DeepMind） | 叙事原则、是什么/为什么/意义何在框架 | [How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) |
| **Sebastian Farquhar**（DeepMind） | 5 句摘要公式 | [How to Write ML Papers](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/) |
| **Gopen & Swan** | 读者期望的 7 条原则 | [Science of Scientific Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) |
| **Zachary Lipton** | 词语选择，消除模糊表达 | [Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/) |
| **Jacob Steinhardt**（UC Berkeley） | 精确性，术语一致性 | [Writing Tips](https://bounded-regret.ghost.io/) |
| **Ethan Perez**（Anthropic） | 微观层面的清晰度技巧 | [Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/) |
| **Andrej Karpathy** | 单一贡献聚焦 | 各类讲座 |

**深入了解任何一项，请参见：**
- [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md) — 含示例的完整说明
- [references/sources.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/sources.md) — 完整参考书目

### 时间分配

在以下各项上花费大约**相等**的时间：
1. 摘要
2. 引言
3. 图表
4. 其他所有内容的总和

**为什么？** 大多数审稿人在读到方法之前就已形成判断。读者接触论文的顺序是：标题 → 摘要 → 引言 → 图表 → 也许是其余部分。

### 写作工作流

```
Paper Writing Checklist:
- [ ] Step 1: Define the one-sentence contribution
- [ ] Step 2: Draft Figure 1 (core idea or most compelling result)
- [ ] Step 3: Draft abstract (5-sentence formula)
- [ ] Step 4: Draft introduction (1-1.5 pages max)
- [ ] Step 5: Draft methods
- [ ] Step 6: Draft experiments & results
- [ ] Step 7: Draft related work
- [ ] Step 8: Draft conclusion & discussion
- [ ] Step 9: Draft limitations (REQUIRED by all venues)
- [ ] Step 10: Plan appendix (proofs, extra experiments, details)
- [ ] Step 11: Complete paper checklist
- [ ] Step 12: Final review
```

### 两遍精炼模式

使用 AI agent 起草时，采用**两遍**方法（在 SakanaAI 的 AI-Scientist 流水线中经过验证）：

**第一遍——逐章节写作 + 即时精炼：**
对每个章节，先写完整草稿，然后在同一上下文中立即精炼。这能在章节内容还新鲜时发现局部问题（清晰度、流畅性、完整性）。

**第二遍——带完整论文上下文的全局精炼：**
所有章节起草完成后，在了解完整论文的情况下重新审视每个章节。这能发现跨章节问题：冗余、术语不一致、叙事流畅性，以及某章节承诺了另一章节未兑现的内容。

```
Second-pass refinement prompt (per section):
"Review the [SECTION] in the context of the complete paper.
- Does it fit with the rest of the paper? Are there redundancies with other sections?
- Is terminology consistent with Introduction and Methods?
- Can anything be cut without weakening the message?
- Does the narrative flow from the previous section and into the next?
Make minimal, targeted edits. Do not rewrite from scratch."
```

### LaTeX 错误清单

将此清单附加到每个精炼 prompt（提示词）中。这些是 LLM 撰写 LaTeX 时最常见的错误：

```
LaTeX Quality Checklist (verify after every edit):
- [ ] No unenclosed math symbols ($ signs balanced)
- [ ] Only reference figures/tables that exist (\ref matches \label)
- [ ] No fabricated citations (\cite matches entries in .bib)
- [ ] Every \begin{env} has matching \end{env} (especially figure, table, algorithm)
- [ ] No HTML contamination (</end{figure}> instead of \end{figure})
- [ ] No unescaped underscores outside math mode (use \_ in text)
- [ ] No duplicate \label definitions
- [ ] No duplicate section headers
- [ ] Numbers in text match actual experimental results
- [ ] All figures have captions and labels
- [ ] No overly long lines that cause overfull hbox warnings
```

### 步骤 5.0：标题

标题是论文中被阅读次数最多的元素。它决定了是否有人会点击进入摘要。

**好的标题**：
- 陈述贡献或发现："Autoreason: When Iterative LLM Refinement Works and Why It Fails"
- 突出令人惊讶的结果："Scaling Data-Constrained Language Models"（暗示你能做到）
- 命名方法 + 说明其作用："DPO: Direct Preference Optimization of Language Models"

**不好的标题**：
- 过于笼统："An Approach to Improving Language Model Outputs"
- 过长：超过约 15 个词的任何标题
- 纯术语堆砌："Asymptotic Convergence of Iterative Stochastic Policy Refinement"（这是给谁看的？）

**规则**：
- 如果有方法名称，包含进去（便于引用）
- 包含 1-2 个审稿人会搜索的关键词
- 除非冒号两侧都有实质内容，否则避免使用冒号
- 测试：审稿人仅凭标题能否了解领域和贡献？

### 步骤 5.1：摘要（5 句公式）

来自 Sebastian Farquhar（DeepMind）：

```
1. What you achieved: "We introduce...", "We prove...", "We demonstrate..."
2. Why this is hard and important
3. How you do it (with specialist keywords for discoverability)
4. What evidence you have
5. Your most remarkable number/result
```

**删除**"大型语言模型取得了显著成就……"之类的通用开头。

### 步骤 5.2：图 1

图 1 是大多数读者看的第二个内容（仅次于摘要）。在撰写引言之前先起草它——这会迫使你厘清核心思想。

| 图 1 类型 | 适用场景 | 示例 |
|-----------|----------|------|
| **方法图** | 新架构或流水线 | 展示系统的 TikZ 流程图 |
| **结果预告** | 一个引人注目的结果能讲述整个故事 | 柱状图："我们的方法 vs 基线"，差距清晰 |
| **问题说明** | 问题不直观 | 前后对比，展示你解决的失败模式 |
| **概念图** | 抽象贡献需要视觉支撑 | 展示方法属性的 2×2 矩阵 |

**规则**：图 1 必须在不阅读任何文字的情况下可理解。仅凭图注就应能传达核心思想。有目的地使用颜色——不要只是装饰。

### 步骤 5.3：引言（最多 1-1.5 页）

必须包含：
- 清晰的问题陈述
- 简要的方法概述
- 2-4 条贡献要点（双栏格式下每条最多 1-2 行）
- 方法应在第 2-3 页开始

### 步骤 5.4：方法

使复现成为可能：
- 概念性概述或伪代码
- 列出所有超参数
- 足以复现的架构细节
- 呈现最终设计决策；消融实验放在实验章节

### 步骤 5.5：实验与结果

对每个实验，明确陈述：
- **它支撑哪个论点**
- 它如何与主要贡献相关联
- 应观察什么："蓝线显示 X，这证明了 Y"

要求：
- 误差棒及其方法（标准差 vs 标准误）
- 超参数搜索范围
- 计算基础设施（GPU 类型、总小时数）
- 随机种子设置方法

### 步骤 5.6：相关工作

按方法论组织，而非逐篇论文列举。慷慨引用——审稿人很可能是相关论文的作者。

### 步骤 5.7：局限性（必须）

所有主要会议都要求此章节。诚实有益：
- 审稿人被指示不因诚实承认局限性而扣分
- 先于批评者识别弱点
- 解释局限性为何不会削弱核心论点

### 步骤 5.8：结论与讨论

**结论**（必须，0.5-1 页）：
- 用一句话重申贡献（与摘要措辞不同）
- 总结关键发现（2-3 句话，而非列表）
- 影响：这对该领域意味着什么？
- 未来工作：2-3 个具体的后续步骤（不要含糊地说"我们将 X 留给未来工作"）

**讨论**（可选，有时与结论合并）：
- 超出直接结果的更广泛影响
- 与其他子领域的联系
- 对方法何时有效、何时无效的诚实评估
- 实际部署考量

**不要**在结论中引入新结果或新论点。

### 步骤 5.9：附录策略

所有主要会议的附录页数不限，对可复现性至关重要。结构：

| 附录章节 | 内容 |
|----------|------|
| **证明与推导** | 正文太长的完整证明。正文可陈述定理并注明"证明见附录 A"。 |
| **额外实验** | 消融实验、规模曲线、按数据集分解、超参数敏感性 |
| **实现细节** | 完整超参数表、训练细节、硬件规格、随机种子 |
| **数据集文档** | 数据收集过程、标注指南、许可证、预处理 |
| **Prompt 与模板** | 使用的确切 prompt（对基于 LLM 的方法）、评估模板 |
| **人工评估** | 标注界面截图、给标注员的说明、IRB 细节 |
| **额外图表** | 按任务分解、轨迹可视化、失败案例示例 |

**规则**：
- 正文必须自包含——审稿人无义务阅读附录
- 绝不将关键证据仅放在附录中
- 交叉引用："完整结果见表 5（附录 B）"，而非仅说"见附录"
- 使用 `\appendix` 命令，然后 `\section{A: Proofs}` 等

### 页面预算管理

超出页面限制时：

| 削减策略 | 节省 | 风险 |
|----------|------|------|
| 将证明移至附录 | 0.5-2 页 | 低——标准做法 |
| 压缩相关工作 | 0.5-1 页 | 中——可能遗漏关键引用 |
| 将表格与子图合并 | 0.25-0.5 页 | 低——通常提升可读性 |
| 谨慎使用 `\vspace{-Xpt}` | 0.1-0.3 页 | 细微时低，明显时高 |
| 删除定性示例 | 0.5-1 页 | 中——审稿人喜欢示例 |
| 缩小图形尺寸 | 0.25-0.5 页 | 高——图形必须保持可读 |

**不要**：缩小字体、更改页边距、删除必要章节（局限性、更广泛影响），或对正文使用 `\small`/`\footnotesize`。

### 步骤 5.10：伦理与更广泛影响声明

大多数会议现在要求或强烈建议提供伦理/更广泛影响声明。这不是样板文字——审稿人会阅读它，并可能标记导致直接拒稿的伦理问题。

**应包含的内容：**

| 组成部分 | 内容 | 要求方 |
|----------|------|--------|
| **积极的社会影响** | 你的工作如何造福社会 | NeurIPS、ICML |
| **潜在负面影响** | 滥用风险、两用性问题、失败模式 | NeurIPS、ICML |
| **公平性与偏见** | 你的方法/数据是否存在已知偏见？ | 所有会议（隐性要求） |
| **环境影响** | 大规模训练的计算碳足迹 | ICML，NeurIPS 日益要求 |
| **隐私** | 你的工作是否使用或允许处理个人数据？ | ACL、NeurIPS |
| **LLM 披露** | 写作或实验中是否使用了 AI？ | ICLR（强制），ACL |

**撰写声明：**

```latex
\section*{Broader Impact Statement}
% NeurIPS/ICML: after conclusion, does not count toward page limit

% 1. Positive applications (1-2 sentences)
This work enables [specific application] which may benefit [specific group].

% 2. Risks and mitigations (1-3 sentences, be specific)
[Method/model] could potentially be misused for [specific risk]. We mitigate
this by [specific mitigation, e.g., releasing only model weights above size X,
including safety filters, documenting failure modes].

% 3. Limitations of impact claims (1 sentence)
Our evaluation is limited to [specific domain]; broader deployment would
require [specific additional work].
```

**常见错误：**
- 写"我们预见不到负面影响"（几乎从不成立——审稿人不信任这种说法）
- 含糊其辞："这可能被滥用"，但不说明如何
- 对大规模工作忽视计算成本
- 在要求披露的会议上忘记披露 LLM 使用情况

**计算碳足迹**（对训练密集型论文）：
```python
# Estimate using ML CO2 Impact tool methodology
gpu_hours = 1000  # total GPU hours
gpu_tdp_watts = 400  # e.g., A100 = 400W
pue = 1.1  # Power Usage Effectiveness (data center overhead)
carbon_intensity = 0.429  # kg CO2/kWh (US average; varies by region)

energy_kwh = (gpu_hours * gpu_tdp_watts * pue) / 1000
carbon_kg = energy_kwh * carbon_intensity
print(f"Energy: {energy_kwh:.0f} kWh, Carbon: {carbon_kg:.0f} kg CO2eq")
```

### 步骤 5.11：数据集说明书与模型卡（如适用）

如果你的论文引入了**新数据集**或**发布了模型**，请包含结构化文档。审稿人对此的期望日益提高，NeurIPS Datasets & Benchmarks 赛道要求提供。

**数据集说明书**（Gebru 等，2021）——包含在附录中：

```
Dataset Documentation (Appendix):
- Motivation: Why was this dataset created? What task does it support?
- Composition: What are the instances? How many? What data types?
- Collection: How was data collected? What was the source?
- Preprocessing: What cleaning/filtering was applied?
- Distribution: How is the dataset distributed? Under what license?
- Maintenance: Who maintains it? How to report issues?
- Ethical considerations: Contains personal data? Consent obtained?
  Potential for harm? Known biases?
```

**模型卡**（Mitchell 等，2019）——模型发布时包含在附录中：

```
Model Card (Appendix):
- Model details: Architecture, training data, training procedure
- Intended use: Primary use cases, out-of-scope uses
- Metrics: Evaluation metrics and results on benchmarks
- Ethical considerations: Known biases, fairness evaluations
- Limitations: Known failure modes, domains where model underperforms
```

### 写作风格

**句子级清晰度（Gopen & Swan 的 7 条原则）：**

| 原则 | 规则 |
|------|------|
| 主谓接近 | 保持主语和谓语紧密相连 |
| 强调位置 | 将重点放在句末 |
| 主题位置 | 先放上下文，后放新信息 |
| 旧信息在前 | 熟悉信息 → 陌生信息 |
| 一个单元，一个功能 | 每段只表达一个观点 |
| 动作在动词中 | 使用动词，而非名词化 |
| 先铺垫后呈现 | 先设置场景，再呈现内容 |

**词语选择（Lipton、Steinhardt）：**
- 具体："accuracy（准确率）"，而非"performance（性能）"
- 消除模糊：除非真正不确定，否则去掉"may（可能）"
- 全文术语一致
- 避免渐进式词汇："develop（开发）"，而非"combine（结合）"

**含示例的完整写作指南**：参见 [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md)

### 使用 LaTeX 模板

**始终先复制整个模板目录，然后在其中写作。**

```
Template Setup Checklist:
- [ ] Step 1: Copy entire template directory to new project
- [ ] Step 2: Verify template compiles as-is (before any changes)
- [ ] Step 3: Read the template's example content to understand structure
- [ ] Step 4: Replace example content section by section
- [ ] Step 5: Use template macros (check preamble for \newcommand definitions)
- [ ] Step 6: Clean up template artifacts only at the end
```

**第一步：复制完整模板**

```bash
cp -r templates/neurips2025/ ~/papers/my-paper/
cd ~/papers/my-paper/
ls -la  # Should see: main.tex, neurips.sty, Makefile, etc.
```

复制**整个**目录，而非仅复制 .tex 文件。模板包含样式文件（.sty）、参考文献样式（.bst）、示例内容和 Makefile。

**第二步：先验证模板可编译**

在做任何修改之前：
```bash
latexmk -pdf main.tex
# Or manual: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

如果未修改的模板无法编译，先解决这个问题（通常是缺少 TeX 包——通过 `tlmgr install <package>` 安装）。

**第三步：保留模板内容作为参考**

不要立即删除示例内容。注释掉并用作格式参考：
```latex
% Template example (keep for reference):
% \begin{figure}[t]
%   \centering
%   \includegraphics[width=0.8\linewidth]{example-image}
%   \caption{Template shows caption style}
% \end{figure}

% Your actual figure:
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{your-figure.pdf}
  \caption{Your caption following the same style.}
\end{figure}
```

**第四步：逐章节替换内容**

系统地推进：标题/作者 → 摘要 → 引言 → 方法 → 实验 → 相关工作 → 结论 → 参考文献 → 附录。每个章节后编译一次。

**第五步：使用模板宏**

```latex
\newcommand{\method}{YourMethodName}  % Consistent method naming
\newcommand{\eg}{e.g.,\xspace}        % Proper abbreviations
\newcommand{\ie}{i.e.,\xspace}
```

### 模板陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|----------|
| 只复制 `.tex` 文件 | 缺少 `.sty`，无法编译 | 复制整个目录 |
| 修改 `.sty` 文件 | 破坏会议格式 | 绝不编辑样式文件 |
| 随意添加包 | 冲突，破坏模板 | 仅在必要时添加 |
| 过早删除模板内容 | 失去格式参考 | 保留为注释直到完成 |
| 不频繁编译 | 错误积累 | 每个章节后编译 |
| 图形使用光栅 PNG | 论文中模糊 | 始终通过 `savefig('fig.pdf')` 使用矢量 PDF |

### 快速模板参考

| 会议 | 主文件 | 样式文件 | 页面限制 |
|------|--------|----------|----------|
| NeurIPS 2025 | `main.tex` | `neurips.sty` | 9 页 |
| ICML 2026 | `example_paper.tex` | `icml2026.sty` | 8 页 |
| ICLR 2026 | `iclr2026_conference.tex` | `iclr2026_conference.sty` | 9 页 |
| ACL 2025 | `acl_latex.tex` | `acl.sty` | 8 页（长文） |
| AAAI 2026 | `aaai2026-unified-template.tex` | `aaai2026.sty` | 7 页 |
| COLM 2025 | `colm2025_conference.tex` | `colm2025_conference.sty` | 9 页 |

**通用规则**：双盲审稿，参考文献不计入页数，附录不限页数，必须使用 LaTeX。

模板位于 `templates/` 目录。编译设置（VS Code、CLI、Overleaf、其他 IDE）请参见 [templates/README.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/templates/README.md)。

### 表格与图形

**表格**——使用 `booktabs` 实现专业格式：

```latex
\usepackage{booktabs}
\begin{tabular}{lcc}
\toprule
Method & Accuracy $\uparrow$ & Latency $\downarrow$ \\
\midrule
Baseline & 85.2 & 45ms \\
\textbf{Ours} & \textbf{92.1} & 38ms \\
\bottomrule
\end{tabular}
```

规则：
- 每个指标的最佳值加粗
- 包含方向符号（$\uparrow$ 越高越好，$\downarrow$ 越低越好）
- 数值列右对齐
- 小数精度一致

**图形**：
- 所有图表和示意图使用**矢量图**（PDF、EPS）——`plt.savefig('fig.pdf')`
- 照片才使用**光栅图**（PNG 600 DPI）
- **色盲友好调色板**（Okabe-Ito 或 Paul Tol）
- 验证**灰度可读性**（8% 的男性有色觉缺陷）
- **图形内部不加标题**——图注承担此功能
- **自包含的图注**——读者无需阅读正文即可理解

### 会议重投

关于在会议之间转换，请参见阶段 7（投稿准备）——它涵盖完整的转换工作流、页面变化表和被拒后的指导。

### 专业 LaTeX 前言

将以下包添加到任何论文中以获得专业质量。它们与所有主要会议样式文件兼容：

```latex
% --- Professional Packages (add after conference style file) ---

% Typography
\usepackage{microtype}              % Microtypographic improvements (protrusion, expansion)
                                     % Makes text noticeably more polished — always include

% Tables
\usepackage{booktabs}               % Professional table rules (\toprule, \midrule, \bottomrule)
\usepackage{siunitx}                % Consistent number formatting, decimal alignment
                                     % Usage: \num{12345} → 12,345; \SI{3.5}{GHz} → 3.5 GHz
                                     % Table alignment: S column type for decimal-aligned numbers

% Figures
\usepackage{graphicx}               % Include graphics (\includegraphics)
\usepackage{subcaption}             % Subfigures with (a), (b), (c) labels
                                     % Usage: \begin{subfigure}{0.48\textwidth} ... \end{subfigure}

% Diagrams and Algorithms
\usepackage{tikz}                   % Programmable vector diagrams
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc, fit, backgrounds}
\usepackage[ruled,vlined]{algorithm2e}  % Professional pseudocode
                                     % Alternative: \usepackage{algorithmicx} if template bundles it

% Cross-references
\usepackage{cleveref}               % Smart references: \cref{fig:x} → "Figure 1"
                                     % MUST be loaded AFTER hyperref
                                     % Handles: figures, tables, sections, equations, algorithms

% Math (usually included by conference .sty, but verify)
\usepackage{amsmath,amssymb}        % AMS math environments and symbols
\usepackage{mathtools}              % Extends amsmath (dcases, coloneqq, etc.)

% Colors (for figures and diagrams)
\usepackage{xcolor}                 % Color management
% Okabe-Ito colorblind-safe palette:
\definecolor{okblue}{HTML}{0072B2}
\definecolor{okorange}{HTML}{E69F00}
\definecolor{okgreen}{HTML}{009E73}
\definecolor{okred}{HTML}{D55E00}
\definecolor{okpurple}{HTML}{CC79A7}
\definecolor{okcyan}{HTML}{56B4E9}
\definecolor{okyellow}{HTML}{F0E442}
```

**注意：**
- `microtype` 是视觉质量影响最大的单个包。它在亚像素级别调整字符间距。始终包含它。
- `siunitx` 通过 `S` 列类型处理表格中的小数对齐——消除手动间距。
- `cleveref` 必须在 `hyperref` **之后**加载。大多数会议 .sty 文件会加载 hyperref，所以将 cleveref 放在最后。
- 检查会议模板是否已加载其中任何包（尤其是 `algorithm`、`amsmath`、`graphicx`）。不要重复加载。

### siunitx 表格对齐

`siunitx` 使数字密集的表格显著更易读：

```latex
\begin{tabular}{l S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]}
\toprule
Method & {Accuracy $\uparrow$} & {F1 $\uparrow$} & {Latency (ms) $\downarrow$} \\
\midrule
Baseline         & 85.2  & 83.7  & 45.3 \\
Ablation (no X)  & 87.1  & 85.4  & 42.1 \\
\textbf{Ours}    & \textbf{92.1} & \textbf{90.8} & \textbf{38.7} \\
\bottomrule
\end{tabular}
```

`S` 列类型自动按小数点对齐。`{}` 中的表头跳过对齐。

### 子图

并排图形的标准模式：

```latex
\begin{figure}[t]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_results_a.pdf}
    \caption{Results on Dataset A.}
    \label{fig:results-a}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_results_b.pdf}
    \caption{Results on Dataset B.}
    \label{fig:results-b}
  \end{subfigure}
  \caption{Comparison of our method across two datasets. (a) shows the scaling
  behavior and (b) shows the ablation results. Both use 5 random seeds.}
  \label{fig:results}
\end{figure}
```

使用 `\cref{fig:results}` → "Figure 1"，`\cref{fig:results-a}` → "Figure 1a"。

### 使用 algorithm2e 编写伪代码

```latex
\begin{algorithm}[t]
\caption{Iterative Refinement with Judge Panel}
\label{alg:method}
\KwIn{Task $T$, model $M$, judges $J_1 \ldots J_n$, convergence threshold $k$}
\KwOut{Final output $A^*$}
$A \gets M(T)$ \tcp*{Initial generation}
$\text{streak} \gets 0$\;
\While{$\text{streak} < k$}{
  $C \gets \text{Critic}(A, T)$ \tcp*{Identify weaknesses}
  $B \gets M(T, C)$ \tcp*{Revised version addressing critique}
  $AB \gets \text{Synthesize}(A, B)$ \tcp*{Merge best elements}
  \ForEach{judge $J_i$}{
    $\text{rank}_i \gets J_i(\text{shuffle}(A, B, AB))$ \tcp*{Blind ranking}
  }
  $\text{winner} \gets \text{BordaCount}(\text{ranks})$\;
  \eIf{$\text{winner} = A$}{
    $\text{streak} \gets \text{streak} + 1$\;
  }{
    $A \gets \text{winner}$; $\text{streak} \gets 0$\;
  }
}
\Return{$A$}\;
\end{algorithm}
```

### TikZ 图形模式

TikZ 是 ML 论文中方法示意图的标准工具。常见模式：

**流水线/流程图**（ML 论文中最常见）：

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  node distance=1.8cm,
  box/.style={rectangle, draw, rounded corners, minimum height=1cm, 
              minimum width=2cm, align=center, font=\small},
  arrow/.style={-{Stealth[length=3mm]}, thick},
]
  \node[box, fill=okcyan!20] (input) {Input\\$x$};
  \node[box, fill=okblue!20, right of=input] (encoder) {Encoder\\$f_\theta$};
  \node[box, fill=okgreen!20, right of=encoder] (latent) {Latent\\$z$};
  \node[box, fill=okorange!20, right of=latent] (decoder) {Decoder\\$g_\phi$};
  \node[box, fill=okred!20, right of=decoder] (output) {Output\\$\hat{x}$};
  
  \draw[arrow] (input) -- (encoder);
  \draw[arrow] (encoder) -- (latent);
  \draw[arrow] (latent) -- (decoder);
  \draw[arrow] (decoder) -- (output);
\end{tikzpicture}
\caption{Architecture overview. The encoder maps input $x$ to latent 
representation $z$, which the decoder reconstructs.}
\label{fig:architecture}
\end{figure}
```

**对比/矩阵图**（用于展示方法变体）：

```latex
\begin{tikzpicture}[
  cell/.style={rectangle, draw, minimum width=2.5cm, minimum height=1cm, 
               align=center, font=\small},
  header/.style={cell, fill=gray!20, font=\small\bfseries},
]
  % Headers
  \node[header] at (0, 0) {Method};
  \node[header] at (3, 0) {Converges?};
  \node[header] at (6, 0) {Quality?};
  % Rows
  \node[cell] at (0, -1) {Single Pass};
  \node[cell, fill=okgreen!15] at (3, -1) {N/A};
  \node[cell, fill=okorange!15] at (6, -1) {Baseline};
  \node[cell] at (0, -2) {Critique+Revise};
  \node[cell, fill=okred!15] at (3, -2) {No};
  \node[cell, fill=okred!15] at (6, -2) {Degrades};
  \node[cell] at (0, -3) {Ours};
  \node[cell, fill=okgreen!15] at (3, -3) {Yes ($k$=2)};
  \node[cell, fill=okgreen!15] at (6, -3) {Improves};
\end{tikzpicture}
```

**迭代循环图**（用于有反馈的方法）：

```latex
\begin{tikzpicture}[
  node distance=2cm,
  box/.style={rectangle, draw, rounded corners, minimum height=0.8cm, 
              minimum width=1.8cm, align=center, font=\small},
  arrow/.style={-{Stealth[length=3mm]}, thick},
  label/.style={font=\scriptsize, midway, above},
]
  \node[box, fill=okblue!20] (gen) {Generator};
  \node[box, fill=okred!20, right=2.5cm of gen] (critic) {Critic};
  \node[box, fill=okgreen!20, below=1.5cm of $(gen)!0.5!(critic)$] (judge) {Judge Panel};
  
  \draw[arrow] (gen) -- node[label] {output $A$} (critic);
  \draw[arrow] (critic) -- node[label, right] {critique $C$} (judge);
  \draw[arrow] (judge) -| node[label, left, pos=0.3] {winner} (gen);
\end{tikzpicture}
```

### latexdiff 用于修改追踪

对于答辩至关重要——生成带标记的 PDF，显示版本间的变化：

```bash
# Install
# macOS: brew install latexdiff (or comes with TeX Live)
# Linux: sudo apt install latexdiff

# Generate diff
latexdiff paper_v1.tex paper_v2.tex > paper_diff.tex
pdflatex paper_diff.tex

# For multi-file projects (with \input{} or \include{})
latexdiff --flatten paper_v1.tex paper_v2.tex > paper_diff.tex
```

生成的 PDF 中，删除内容显示为红色删除线，新增内容显示为蓝色——这是答辩补充材料的标准格式。

### SciencePlots 用于 matplotlib

安装并使用以获得出版质量的图表：

```bash
pip install SciencePlots
```

```python
import matplotlib.pyplot as plt
import scienceplots  # registers styles

# Use science style (IEEE-like, clean)
with plt.style.context(['science', 'no-latex']):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Single-column width
    ax.plot(x, y, label='Ours', color='#0072B2')
    ax.plot(x, y2, label='Baseline', color='#D55E00', linestyle='--')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig('paper/fig_results.pdf', bbox_inches='tight')

# Available styles: 'science', 'ieee', 'nature', 'science+ieee'
# Add 'no-latex' if LaTeX is not installed on the machine generating plots
```

**标准图形尺寸**（双栏格式）：
- 单栏：`figsize=(3.5, 2.5)` — 适合一栏
- 双栏：`figsize=(7.0, 3.0)` — 跨两栏
- 正方形：`figsize=(3.5, 3.5)` — 用于热力图、混淆矩阵

---

## 阶段 6：自我审阅与修改

**目标**：在投稿前模拟审稿过程。尽早发现弱点。

### 步骤 6.1：模拟审稿（集成模式）

从多个角度生成审稿意见。来自自动化研究流水线（尤其是 SakanaAI 的 AI-Scientist）的关键洞见：**集成审稿加元审稿人产生的反馈比单次审稿通过校准得多。**

**第一步：生成 N 份独立审稿意见**（N=3-5）

使用不同模型或温度设置。每位审稿人只看论文，看不到其他审稿意见。**默认偏向负面**——LLM 在评估中有充分记录的正面偏见。

```
You are an expert reviewer for [VENUE]. You are critical and thorough.
If a paper has weaknesses or you are unsure about a claim, flag it clearly
and reflect that in your scores. Do not give the benefit of the doubt.

Review this paper according to the official reviewer guidelines. Evaluate:

1. Soundness (are claims well-supported? are baselines fair and strong?)
2. Clarity (is the paper well-written? could an expert reproduce it?)
3. Significance (does this matter to the community?)
4. Originality (new insights, not just incremental combination?)

Provide your review as structured JSON:
{
  "summary": "2-3 sentence summary",
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1 (most critical)", "weakness 2", ...],
  "questions": ["question for authors 1", ...],
  "missing_references": ["paper that should be cited", ...],
  "soundness": 1-4,
  "presentation": 1-4,
  "contribution": 1-4,
  "overall": 1-10,
  "confidence": 1-5
}
```

**第二步：元审稿（领域主席汇总）**

将所有 N 份审稿意见提交给元审稿人：

```
You are an Area Chair at [VENUE]. You have received [N] independent reviews
of a paper. Your job is to:

1. Identify consensus strengths and weaknesses across reviewers
2. Resolve disagreements by examining the paper directly
3. Produce a meta-review that represents the aggregate judgment
4. Use AVERAGED numerical scores across all reviews

Be conservative: if reviewers disagree on whether a weakness is serious,
treat it as serious until the authors address it.

Reviews:
[review_1]
[review_2]
...
```

**第三步：反思循环**（可选，2-3 轮）

每位审稿人在看到元审稿后可以完善自己的意见。使用提前终止标志：如果审稿人回复"I am done"（无变化），停止迭代。

**审稿模型选择**：审稿最好使用最强的可用模型，即使你用更便宜的模型写了论文。审稿模型应独立于写作模型选择。

**少样本校准**：如果可用，包含 1-2 份来自目标会议的真实已发表审稿意见作为示例。这会显著提升分数校准。示例审稿意见请参见 [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md)。

### 步骤 6.1b：视觉审阅（VLM）

纯文本审阅会遗漏整类问题：图形质量、排版问题、视觉一致性。如果你有访问视觉能力模型的权限，对编译后的 PDF 运行单独的**视觉审阅**：

```
You are reviewing the visual presentation of this research paper PDF.
Check for:
1. Figure quality: Are plots readable? Labels legible? Colors distinguishable?
2. Figure-caption alignment: Does each caption accurately describe its figure?
3. Layout issues: Orphaned section headers, awkward page breaks, figures far from their references
4. Table formatting: Aligned columns, consistent decimal precision, bold for best results
5. Visual consistency: Same color scheme across all figures, consistent font sizes
6. Grayscale readability: Would the figures be understandable if printed in B&W?

For each issue, specify the page number and exact location.
```

这能发现纯文本审阅无法发现的问题：坐标轴标签难以辨认的图表、距其首次引用 3 页之远的图形、图 2 和图 5 之间不一致的调色板，或明显超出栏宽的表格。

### 步骤 6.1c：论点核实

模拟审稿后，运行单独的核实。这能发现审稿人可能遗漏的事实错误：

```
Claim Verification Protocol:
1. Extract every factual claim from the paper (numbers, comparisons, trends)
2. For each claim, trace it to the specific experiment/result that supports it
3. Verify the number in the paper matches the actual result file
4. Flag any claim without a traceable source as [VERIFY]
```

对于基于 agent 的工作流：将核实委派给**全新的子 agent**，该 agent 只接收论文文本和原始结果文件。全新的上下文防止确认偏见——核实者不会"记得"结果应该是什么。

### 步骤 6.2：优先处理反馈

收集审稿意见后，分类：

| 优先级 | 行动 |
|--------|------|
| **关键**（技术缺陷、缺少基线） | 必须修复。可能需要新实验 → 返回阶段 2 |
| **高**（清晰度问题、缺少消融实验） | 本次修改中应修复 |
| **中**（小的写作问题、额外实验） | 时间允许时修复 |
| **低**（风格偏好、边缘建议） | 记录为未来工作 |

### 步骤 6.3：修改循环

对每个关键/高优先级问题：
1. 确定受影响的具体章节
2. 起草修复方案
3. 验证修复不会破坏其他论点
4. 更新论文
5. 对照审稿人的关切重新检查

### 步骤 6.4：撰写答辩

回应实际审稿意见（投稿后）时，答辩是一项不同于修改的独立技能：

**格式**：逐点回应。对每个审稿人关切：
```
> R1-W1: "The paper lacks comparison with Method X."

We thank the reviewer for this suggestion. We have added a comparison with 
Method X in Table 3 (revised). Our method outperforms X by 3.2pp on [metric] 
(p<0.05). We note that X requires 2x our compute budget.
```

**规则**：
- 回应每一个关切——审稿人会注意到你跳过了哪些
- 以最有力的回应开头
- 简洁直接——审稿人要阅读数十份答辩
- 如果在答辩期间运行了实验，包含新结果
- 即使面对弱批评，也不要防御或轻视
- 使用 `latexdiff` 生成带标记的 PDF 显示变化（参见专业 LaTeX 工具章节）
- 对具体、可操作的反馈表示感谢（不要泛泛称赞）

**不要做的事**：没有证据地说"我们尊重地不同意"。不加解释地说"这超出范围"。只回应优点而忽视弱点。

### 步骤 6.5：论文演变追踪

在关键里程碑处保存快照：
```
paper/
  paper.tex                    # Current working version
  paper_v1_first_draft.tex     # First complete draft
  paper_v2_post_review.tex     # After simulated review
  paper_v3_pre_submission.tex  # Final before submission
  paper_v4_camera_ready.tex    # Post-acceptance final
```

---

## 阶段 7：投稿准备

**目标**：最终检查、格式化和投稿。

### 步骤 7.1：会议清单

每个会议都有强制性清单。仔细完成——清单不完整可能导致直接拒稿。

参见 [references/checklists.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/checklists.md)，包含：
- NeurIPS 16 项论文清单
- ICML 更广泛影响 + 可复现性
- ICLR LLM 披露政策
- ACL 强制局限性章节
- 通用投稿前清单

### 步骤 7.2：匿名化清单

双盲审稿意味着审稿人不能知道论文作者是谁。检查以下**所有**内容：

```
Anonymization Checklist:
- [ ] No author names or affiliations anywhere in the PDF
- [ ] No acknowledgments section (add after acceptance)
- [ ] Self-citations written in third person: "Smith et al. [1] showed..." not "We previously showed [1]..."
- [ ] No GitHub/GitLab URLs pointing to your personal repos
- [ ] Use Anonymous GitHub (https://anonymous.4open.science/) for code links
- [ ] No institutional logos or identifiers in figures
- [ ] No file metadata containing author names (check PDF properties)
- [ ] No "our previous work" or "in our earlier paper" phrasing
- [ ] Dataset names don't reveal institution (rename if needed)
- [ ] Supplementary materials don't contain identifying information
```

**常见错误**：补充代码中可见的 Git commit 信息、机构工具生成的带水印图形、从上一稿遗留的致谢、在匿名期之前发布的 arXiv 预印本。

### 步骤 7.3：格式验证

```
Pre-Submission Format Check:
- [ ] Page limit respected (excluding references and appendix)
- [ ] All figures are vector (PDF) or high-res raster (600 DPI PNG)
- [ ] All figures readable in grayscale
- [ ] All tables use booktabs
- [ ] References compile correctly (no "?" in citations)
- [ ] No overfull hboxes in critical areas
- [ ] Appendix clearly labeled and separated
- [ ] Required sections present (limitations, broader impact, etc.)
```

### 步骤 7.4：编译前验证

在尝试 `pdflatex` **之前**运行这些自动检查。在这里发现错误比调试编译器输出更快。

```bash
# 1. Lint with chktex (catches common LaTeX mistakes)
# Suppress noisy warnings: -n2 (sentence end), -n24 (parens), -n13 (intersentence), -n1 (command terminated)
chktex main.tex -q -n2 -n24 -n13 -n1

# 2. Verify all citations exist in .bib
# Extract \cite{...} from .tex, check each against .bib
python3 -c "
import re
tex = open('main.tex').read()
bib = open('references.bib').read()
cites = set(re.findall(r'\\\\cite[tp]?{([^}]+)}', tex))
for cite_group in cites:
    for cite in cite_group.split(','):
        cite = cite.strip()
        if cite and cite not in bib:
            print(f'WARNING: \\\\cite{{{cite}}} not found in references.bib')
"

# 3. Verify all referenced figures exist on disk
python3 -c "
import re, os
tex = open('main.tex').read()
figs = re.findall(r'\\\\includegraphics(?:\[.*?\])?{([^}]+)}', tex)
for fig in figs:
    if not os.path.exists(fig):
        print(f'WARNING: Figure file not found: {fig}')
"

# 4. Check for duplicate \label definitions
python3 -c "
import re
from collections import Counter
tex = open('main.tex').read()
labels = re.findall(r'\\\\label{([^}]+)}', tex)
dupes = {k: v for k, v in Counter(labels).items() if v > 1}
for label, count in dupes.items():
    print(f'WARNING: Duplicate label: {label} (appears {count} times)')
"
```

在继续之前修复所有警告。对于基于 agent 的工作流：将 chktex 输出反馈给 agent，并指示其进行最小化修复。

### 步骤 7.5：最终编译

```bash
# Clean build
rm -f *.aux *.bbl *.blg *.log *.out *.pdf
latexmk -pdf main.tex

# Or manual (triple pdflatex + bibtex for cross-references)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Verify output exists and has content
ls -la main.pdf
```

**如果编译失败**：解析 `.log` 文件找到第一个错误。常见修复：
- "Undefined control sequence" → 缺少包或命令名拼写错误
- "Missing $ inserted" → 数学符号在数学模式外
- "File not found" → 图形路径错误或缺少 .sty 文件
- "Citation undefined" → .bib 条目缺失或未运行 bibtex

### 步骤 7.6：会议特定要求

| 会议 | 特殊要求 |
|------|----------|
| **NeurIPS** | 附录中的论文清单，接收后提供通俗摘要 |
| **ICML** | 更广泛影响声明（结论后，不计入页数限制） |
| **ICLR** | 必须披露 LLM 使用，互惠审稿协议 |
| **ACL** | 强制局限性章节，负责任 NLP 清单 |
| **AAAI** | 严格的样式文件——绝对不允许任何修改 |
| **COLM** | 为语言模型社区框架贡献 |

### 步骤 7.7：会议重投与格式转换

在会议之间转换时，**绝不在模板之间复制 LaTeX 前言**：

```bash
# 1. Start fresh with target template
cp -r templates/icml2026/ new_submission/

# 2. Copy ONLY content sections (not preamble)
#    - Abstract text, section content, figures, tables, bib entries

# 3. Adjust for page limits
# 4. Add venue-specific required sections
# 5. Update references
```

| 从 → 到 | 页面变化 | 主要调整 |
|---------|----------|----------|
| NeurIPS → ICML | 9 → 8 | 削减 1 页，添加更广泛影响 |
| ICML → ICLR | 8 → 9 | 扩展实验，添加 LLM 披露 |
| NeurIPS → ACL | 9 → 8 | 按 NLP 惯例重构，添加局限性 |
| ICLR → AAAI | 9 → 7 | 大幅削减，严格遵守样式 |
| 任意 → COLM | 不定 → 9 | 重新框架为语言模型焦点 |

削减页面时：将证明移至附录，压缩相关工作，合并表格，使用子图。
扩展页面时：添加消融实验，扩展局限性，包含额外基线，添加定性示例。

**被拒后**：在新版本中解决审稿人关切，但不要包含"变更"章节或引用之前的投稿（盲审）。

### 步骤 7.8：最终版本准备（接收后）

接收后，准备最终版本：

```
Camera-Ready Checklist:
- [ ] De-anonymize: add author names, affiliations, email addresses
- [ ] Add Acknowledgments section (funding, compute grants, helpful reviewers)
- [ ] Add public code/data URL (real GitHub, not anonymous)
- [ ] Address any mandatory revisions from meta-reviewer
- [ ] Switch template to camera-ready mode (if applicable — e.g., AAAI \anon → \camera)
- [ ] Add copyright notice if required by venue
- [ ] Update any "anonymous" placeholders in text
- [ ] Verify final PDF compiles cleanly
- [ ] Check page limit for camera-ready (sometimes differs from submission)
- [ ] Upload supplementary materials (code, data, appendix) to venue portal
```

### 步骤 7.9：arXiv 与预印本策略

在 ML 领域，发布到 arXiv 是标准做法，但有重要的时机和匿名性考量。

**时机决策树：**

| 情况 | 建议 |
|------|------|
| 投稿至双盲会议（NeurIPS、ICML、ACL） | 在投稿截止日期**之后**发布到 arXiv，而非之前。之前发布在技术上可能违反匿名政策，尽管执行力度不一。 |
| 投稿至 ICLR | ICLR 明确允许在投稿前发布到 arXiv。但投稿本身不要写作者姓名。 |
| 论文已在 arXiv，投稿至新会议 | 大多数会议可接受。审稿期间**不要**更新 arXiv 版本以包含回应审稿意见的变化。 |
| 研讨会论文 | arXiv 随时可以发布——研讨会通常不是双盲的。 |
| 想要确立优先权 | 如果担心被抢先，立即发布——但接受匿名性的权衡。 |

**arXiv 类别选择**（ML/AI 论文）：

| 类别 | 代码 | 最适合 |
|------|------|--------|
| Machine Learning | `cs.LG` | 通用 ML 方法 |
| Computation and Language | `cs.CL` | NLP、语言模型 |
| Artificial Intelligence | `cs.AI` | 推理、规划、agent |
| Computer Vision | `cs.CV` | 视觉模型 |
| Information Retrieval | `cs.IR` | 搜索、推荐 |

**列出主要类别 + 1-2 个交叉列出的类别。** 更多类别 = 更高曝光度，但只在真正相关时才交叉列出。

**版本策略：**
- **v1**：初始投稿（与会议投稿版本一致）
- **v2**：接收后附最终版本修正（在摘要中添加"accepted at [Venue]"）
- 审稿期间不要发布 v2，其中包含明显回应审稿意见的变化

```bash
# Check if your paper's title is already taken on arXiv
# (before choosing a title)
pip install arxiv
python -c "
import arxiv
results = list(arxiv.Search(query='ti:\"Your Exact Title\"', max_results=5).results())
print(f'Found {len(results)} matches')
for r in results: print(f'  {r.title} ({r.published.year})')
"
```

### 步骤 7.10：研究代码打包

发布干净、可运行的代码会显著提高引用量和审稿人信任度。与最终版本一起打包代码。

**代码库结构：**

```
your-method/
  README.md              # Setup, usage, reproduction instructions
  requirements.txt       # Or environment.yml for conda
  setup.py               # For pip-installable packages
  LICENSE                # MIT or Apache 2.0 recommended for research
  configs/               # Experiment configurations
  src/                   # Core method implementation
  scripts/               # Training, evaluation, analysis scripts
    train.py
    evaluate.py
    reproduce_table1.sh  # One script per main result
  data/                  # Small data or download scripts
    download_data.sh
  results/               # Expected outputs for verification
```

**研究代码的 README 模板：**

```markdown
# [Paper Title]

Official implementation of "[Paper Title]" (Venue Year).

## Setup
[Exact commands to set up environment]

## Reproduction
To reproduce Table 1: `bash scripts/reproduce_table1.sh`
To reproduce Figure 2: `python scripts/make_figure2.py`

## Citation
[BibTeX entry]
```

**发布前清单：**
```
- [ ] Code runs from a clean clone (test on fresh machine or Docker)
- [ ] All dependencies pinned to specific versions
- [ ] No hardcoded absolute paths
- [ ] No API keys, credentials, or personal data in repo
- [ ] README covers setup, reproduction, and citation
- [ ] LICENSE file present (MIT or Apache 2.0 for max reuse)
- [ ] Results are reproducible within expected variance
- [ ] .gitignore excludes data files, checkpoints, logs
```

**投稿用匿名代码**（接收前）：
```bash
# Use Anonymous GitHub for double-blind review
# https://anonymous.4open.science/
# Upload your repo → get an anonymous URL → put in paper
```

---

## 阶段 8：接收后的交付物

**目标**：通过演示材料和社区参与最大化已接收论文的影响力。

### 步骤 8.1：会议海报

大多数会议要求海报展示。海报设计原则：

| 元素 | 指导 |
|------|------|
| **尺寸** | 查看会议要求（通常为 24"×36" 或 A0 竖版/横版） |
| **内容** | 标题、作者、一句话贡献、方法图、2-3 个关键结果、结论 |
| **流向** | 从左上到右下（Z 形）或分栏 |
| **文字** | 标题在 3 米处可读，正文在 1 米处可读。不要整段文字——只用要点。 |
| **图形** | 复用论文图形，分辨率更高。放大关键结果。 |

**工具**：LaTeX（`beamerposter` 包）、PowerPoint/Keynote、Figma、Canva。

**制作**：在会议前 2 周以上下单。布料海报旅行时更轻便。许多会议现在也支持虚拟/数字海报。

### 步骤 8.2：会议演讲/亮点展示

如果获得口头报告或亮点展示机会：

| 演讲类型 | 时长 | 内容 |
|----------|------|------|
| **亮点展示** | 5 分钟 | 问题、方法、一个关键结果。排练到恰好 5 分钟。 |
| **口头报告** | 15-20 分钟 | 完整故事：问题、方法、关键结果、消融实验、局限性。 |
| **研讨会演讲** | 10-15 分钟 | 根据研讨会受众调整——可能需要更多背景介绍。 |

**幻灯片设计规则：**
- 每张幻灯片一个想法
- 最小化文字——口头讲述细节，不要投影出来
- 逐步动画关键图形以建立理解
- 最后包含一张"要点"幻灯片（单句贡献）
- 为预期问题准备备用幻灯片

### 步骤 8.3：博客文章/社交媒体

易于理解的摘要会显著提升影响力：

- **Twitter/X 帖子**：5-8 条推文。以结果开头，而非方法。包含图 1 和关键结果图。
- **博客文章**：800-1500 字。面向 ML 从业者，而非审稿人。跳过形式化内容，强调直觉和实际影响。
- **项目页面**：包含摘要、图形、演示、代码链接、BibTeX 的 HTML 页面。使用 GitHub Pages。

**时机**：在论文出现在会议论文集或 arXiv 最终版本后 1-2 天内发布。

---

## 研讨会与短文

研讨会论文和短文（如 ACL 短文、Findings 论文）遵循相同的流水线，但有不同的约束和期望。

### 研讨会论文

| 属性 | 研讨会 | 主会议 |
|------|--------|--------|
| **页面限制** | 通常 4-6 页 | 7-9 页 |
| **审稿标准** | 完整性要求较低 | 必须完整、深入 |
| **审稿流程** | 通常单盲或轻度审稿 | 双盲，严格 |
| **重视内容** | 有趣的想法、初步结果、立场文章 | 有强基线的完整实证故事 |
| **arXiv** | 随时发布 | 时机很重要（参见 arXiv 策略） |
| **贡献门槛** | 新方向、有趣的负面结果、进行中的工作 | 有强证据的重大进展 |

**何时投稿研讨会：**
- 在完整论文之前想获得反馈的早期想法
- 不足以支撑 8+ 页的负面结果
- 关于时事话题的立场文章或观点
- 复现研究或可复现性报告

### ACL 短文与 Findings

ACL 系列会议有不同的投稿类型：

| 类型 | 页数 | 期望内容 |
|------|------|----------|
| **长文** | 8 | 完整研究，强基线，消融实验 |
| **短文** | 4 | 聚焦贡献：一个有证据支撑的清晰观点 |
| **Findings** | 8 | 扎实的工作，略未达到主会议标准 |

**短文策略**：选择**一个**论点并充分支撑它。不要试图将长文压缩成 4 页——写一篇不同的、更聚焦的论文。

---

## 超越实证 ML 的论文类型

上述主要流水线针对实证 ML 论文。其他论文类型需要不同的结构和证据标准。每种类型的详细指导请参见 [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md)。

### 理论论文

**结构**：引言 → 预备知识（定义、符号）→ 主要结果（定理）→ 证明草图 → 讨论 → 完整证明（附录）

**与实证论文的主要区别：**
- 贡献是定理、界或不可能性结果——而非实验数字
- 方法章节替换为"预备知识"和"主要结果"
- 证明是证据，而非实验（尽管理论的实证验证受欢迎）
- 正文中的证明草图 + 附录中的完整证明是标准做法
- 实验章节可选，但如果能验证理论预测则会增强论文

**证明写作原则：**
- 明确陈述所有假设的正式定理
- 在正式证明之前提供直觉（"关键洞见是……"）
- 证明草图应在 0.5-1 页内传达主要思想
- 使用 `\begin{proof}...\end{proof}` 环境
- 编号假设并在定理中引用："在假设 1-3 下，……"

### 综述/教程论文

**结构**：引言 → 分类/组织 → 详细覆盖 → 开放问题 → 结论

**主要区别：**
- 贡献是组织、综合和识别开放问题——而非新方法
- 在范围内必须全面（审稿人会检查遗漏的引用）
- 需要清晰的分类或组织框架
- 价值来自单篇论文未建立的工作间联系
- 最佳会议：TMLR（综述赛道）、JMLR、Foundations and Trends in ML、ACM Computing Surveys

### 基准测试论文

**结构**：引言 → 任务定义 → 数据集构建 → 基线评估 → 分析 → 预期用途与局限性

**主要区别：**
- 贡献是基准测试本身——它必须填补真正的评估空白
- 数据集文档是强制性的，而非可选的（参见数据集说明书，步骤 5.11）
- 必须证明基准测试具有挑战性（基线不会使其饱和）
- 必须证明基准测试测量了你声称测量的内容（构建效度）
- 最佳会议：NeurIPS Datasets & Benchmarks 赛道、ACL（资源论文）、LREC-COLING

### 立场论文

**结构**：引言 → 背景 → 论点/主张 → 支撑证据 → 反驳论点 → 影响

**主要区别：**
- 贡献是一个论点，而非结果
- 必须认真对待反驳论点
- 证据可以是实证的、理论的或逻辑分析
- 最佳会议：ICML（立场赛道）、研讨会、TMLR

---

## Hermes Agent 集成

本 skill 专为 Hermes agent 设计。它使用 Hermes 工具、委派、调度和记忆来支撑完整的研究生命周期。

### 相关 Skill

将本 skill 与其他 Hermes skill 组合用于特定阶段：

| Skill | 使用时机 | 加载方式 |
|-------|----------|----------|
| **arxiv** | 阶段 1（文献综述）：搜索 arXiv、生成 BibTeX、通过 Semantic Scholar 查找相关论文 | `skill_view("arxiv")` |
| **subagent-driven-development** | 阶段 5（起草）：并行章节写作，含两阶段审阅（规范合规性，然后质量） | `skill_view("subagent-driven-development")` |
| **plan** | 阶段 0（设置）：执行前创建结构化计划。写入 `.hermes/plans/` | `skill_view("plan")` |
| **qmd** | 阶段 1（文献）：通过混合 BM25+向量搜索查询本地知识库（笔记、转录、文档） | 安装：`skill_manage("install", "qmd")` |
| **diagramming** | 阶段 4-5：创建基于 Excalidraw 的图形和架构示意图 | `skill_view("diagramming")` |
| **data-science** | 阶段 4（分析）：用于交互式分析和可视化的 Jupyter 实时内核 | `skill_view("data-science")` |

**本 skill 取代 `ml-paper-writing`**——它包含 ml-paper-writing 的所有内容，加上完整的实验/分析流水线和 autoreason 方法论。

### Hermes 工具参考

| 工具 | 在本流水线中的用途 |
|------|------------------|
| **`terminal`** | LaTeX 编译（`latexmk -pdf`）、git 操作、启动实验（`nohup python run.py &`）、进程检查 |
| **`process`** | 后台实验管理：`process("start", ...)`、`process("poll", pid)`、`process("log", pid)`、`process("kill", pid)` |
| **`execute_code`** | 运行 Python 进行引用核实、统计分析、数据聚合。通过 RPC 访问工具。 |
| **`read_file`** / **`write_file`** / **`patch`** | 论文编辑、实验脚本、结果文件。对大型 .tex 文件使用 `patch` 进行针对性编辑。 |
| **`web_search`** | 文献发现：`web_search("transformer attention mechanism 2024")` |
| **`web_extract`** | 获取论文内容，核实引用：`web_extract("https://arxiv.org/abs/2303.17651")` |
| **`delegate_task`** | **并行章节起草**——为每个章节生成隔离的子 agent。也用于并发引用核实。 |
| **`todo`** | 跨会话的主要状态追踪器。每次阶段转换后更新。 |
| **`memory`** | 跨会话持久化关键决策：贡献框架、会议选择、审稿反馈。 |
| **`cronjob`** | 调度实验监控、截止日期倒计时、自动 arXiv 检查。 |
| **`clarify`** | 在真正受阻时向用户提出针对性问题（会议选择、贡献框架）。 |
| **`send_message`** | 即使用户不在聊天中，也在实验完成或草稿准备好时通知用户。 |

### 工具使用模式

**实验监控**（最常见）：
```
terminal("ps aux | grep <pattern>")
→ terminal("tail -30 <logfile>")
→ terminal("ls results/")
→ execute_code("analyze results JSON, compute metrics")
→ terminal("git add -A && git commit -m '<descriptive message>' && git push")
→ send_message("Experiment complete: <summary>")
```

**并行章节起草**（使用委派）：
```
delegate_task("Draft the Methods section based on these experiment scripts and configs. 
  Include: pseudocode, all hyperparameters, architectural details sufficient for 
  reproduction. Write in LaTeX using the neurips2025 template conventions.")

delegate_task("Draft the Related Work section. Use web_search and web_extract to 
  find papers. Verify every citation via Semantic Scholar. Group by methodology.")

delegate_task("Draft the Experiments section. Read all result files in results/. 
  State which claim each experiment supports. Include error bars and significance.")
```

每个委派作为**全新子 agent** 运行，无共享上下文——在 prompt（提示词）中提供所有必要信息。收集输出并整合。

**引用核实**（使用 execute_code）：
```python
# In execute_code:
from semanticscholar import SemanticScholar
import requests

sch = SemanticScholar()
results = sch.search_paper("attention mechanism transformers", limit=5)
for paper in results:
    doi = paper.externalIds.get('DOI', 'N/A')
    if doi != 'N/A':
        bibtex = requests.get(f"https://doi.org/{doi}", 
                              headers={"Accept": "application/x-bibtex"}).text
        print(bibtex)
```

### 使用 `memory` 和 `todo` 进行状态管理

**`memory` 工具**——持久化关键决策（有限：MEMORY.md 约 2200 字符）：

```
memory("add", "Paper: autoreason. Venue: NeurIPS 2025 (9 pages). 
  Contribution: structured refinement works when generation-evaluation gap is wide.
  Key results: Haiku 42/42, Sonnet 3/5, S4.6 constrained 2/3.
  Status: Phase 5 — drafting Methods section.")
```

在重大决策或阶段转换后更新记忆。这会跨会话持久化。

**`todo` 工具**——追踪细粒度进度：

```
todo("add", "Design constrained task experiments for Sonnet 4.6")
todo("add", "Run Haiku baseline comparison")
todo("add", "Draft Methods section")
todo("update", id=3, status="in_progress")
todo("update", id=1, status="completed")
```

**会话启动协议：**
```
1. todo("list")                           # Check current task list
2. memory("read")                         # Recall key decisions
3. terminal("git log --oneline -10")      # Check recent commits
4. terminal("ps aux | grep python")       # Check running experiments
5. terminal("ls results/ | tail -20")     # Check for new results
6. Report status to user, ask for direction
```

### 使用 `cronjob` 进行 Cron 监控

使用 `cronjob` 工具调度定期实验检查：

```
cronjob("create", {
  "schedule": "*/30 * * * *",  # Every 30 minutes
  "prompt": "Check experiment status:
    1. ps aux | grep run_experiment
    2. tail -30 logs/experiment_haiku.log
    3. ls results/haiku_baselines/
    4. If complete: read results, compute Borda scores, 
       git add -A && git commit -m 'Add Haiku results' && git push
    5. Report: table of results, key finding, next step
    6. If nothing changed: respond with [SILENT]"
})
```

**[SILENT] 协议**：当自上次检查以来没有任何变化时，精确回复 `[SILENT]`。这会抑制向用户的通知推送。只在有真正值得了解的变化时报告。

**截止日期追踪**：
```
cronjob("create", {
  "schedule": "0 9 * * *",  # Daily at 9am
  "prompt": "NeurIPS 2025 deadline: May 22. Today is {date}. 
    Days remaining: {compute}. 
    Check todo list — are we on track? 
    If <7 days: warn user about remaining tasks."
})
```

### 通信模式

**何时通知用户**（通过 `send_message` 或直接回复）：
- 一批实验完成（附结果表格）
- 意外发现或需要决策的故障
- 草稿章节准备好供审阅
- 截止日期临近但任务未完成

**何时不通知：**
- 实验仍在运行，无新结果 → `[SILENT]`
- 无变化的例行监控 → `[SILENT]`
- 不需要关注的中间步骤

**报告格式**——始终包含结构化数据：
```
## Experiment: <name>
Status: Complete / Running / Failed

| Task | Method A | Method B | Method C |
|------|---------|---------|---------|
| Task 1 | 85.2 | 82.1 | **89.4** |

Key finding: <one sentence>
Next step: <what happens next>
```

### 需要人工输入的决策点

在真正受阻时使用 `clarify` 提出针对性问题：

| 决策 | 何时提问 |
|------|----------|
| 目标会议 | 在开始论文之前（影响页面限制、框架） |
| 贡献框架 | 当存在多个有效框架时 |
| 实验优先级 | 当 TODO 列表中的实验多于时间允许时 |
| 投稿准备情况 | 在最终投稿之前 |

**不要询问**（主动出击，做出选择，标注出来）：
- 措辞选择、章节顺序
- 突出哪些具体结果
- 引用完整性（用你找到的内容起草，记录空缺）

---

## 审稿人评估标准

了解审稿人的关注点有助于集中精力：

| 标准 | 他们检查什么 |
|------|------------|
| **质量** | 技术严谨性、有充分支撑的论点、公平的基线 |
| **清晰度** | 写作清晰、专家可复现、符号一致 |
| **重要性** | 社区影响、推进理解 |
| **原创性** | 新洞见（不要求新方法） |

**评分（NeurIPS 6 分制）：**
- 6：强烈接收——突破性，无懈可击
- 5：接收——技术扎实，高影响力
- 4：边缘接收——扎实，评估有限
- 3：边缘拒绝——弱点超过优点
- 2：拒绝——技术缺陷
- 1：强烈拒绝——已知结果或伦理问题

详细指南、常见关切和答辩策略请参见 [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md)。

---

## 常见问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| 摘要过于笼统 | 如果第一句话可以作为任何 ML 论文的开头，删除它。从你的具体贡献开始。 |
| 引言超过 1.5 页 | 将背景拆分到相关工作中。将贡献要点前置。 |
| 实验缺乏明确论点 | 在每个实验前添加："本实验检验 [具体论点] 是否成立……" |
| 审稿人觉得论文难以理解 | 添加路标语句，使用一致术语，使图注自包含。 |
| 缺少统计显著性 | 添加误差棒、运行次数、统计检验、置信区间。 |
| 实验范围蔓延 | 每个实验必须映射到一个具体论点。删除不映射的实验。 |
| 论文被拒，需要重投 | 参见阶段 7 中的会议重投。解决审稿人关切，不要引用之前的审稿意见。 |
| 缺少更广泛影响声明 | 参见步骤 5.10。大多数会议要求此声明。"无负面影响"几乎从不可信。 |
| 人工评估被批评为薄弱 | 参见步骤 2.5 和 [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md)。报告一致性指标、标注员详情、报酬。 |
| 审稿人质疑可复现性 | 发布代码（步骤 7.9），记录所有超参数，包含随机种子和计算细节。 |
| 理论论文缺乏直觉 | 在正式证明之前添加含通俗语言解释的证明草图。参见 [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md)。 |
| 结果为负面/零结果 | 参见阶段 4.3 关于处理负面结果的内容。考虑研讨会、TMLR 或重新框架为分析。 |

---

## 参考文档

| 文档 | 内容 |
|------|------|
| [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md) | Gopen & Swan 7 条原则、Perez 微观技巧、Lipton 词语选择、Steinhardt 精确性、图形设计 |
| [references/citation-workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/citation-workflow.md) | 引用 API、Python 代码、CitationManager 类、BibTeX 管理 |
| [references/checklists.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/checklists.md) | NeurIPS 16 项、ICML、ICLR、ACL 要求、通用投稿前清单 |
| [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md) | 评估标准、评分、常见关切、答辩模板 |
| [references/sources.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/sources.md) | 所有写作指南、会议指南、API 的完整参考书目 |
| [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md) | 实验设计模式、评估协议、监控、错误恢复 |
| [references/autoreason-methodology.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/autoreason-methodology.md) | Autoreason 循环、策略选择、模型指南、prompt（提示词）、范围约束、Borda 计分 |
| [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md) | 人工评估设计、标注指南、一致性指标、众包质量控制、IRB 指导 |
| [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md) | 理论论文（证明写作、定理结构）、综述论文、基准测试论文、立场论文 |

### LaTeX 模板

`templates/` 中的模板：**NeurIPS 2025**、**ICML 2026**、**ICLR 2026**、**ACL**、**AAAI 2026**、**COLM 2025**。

编译说明请参见 [templates/README.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/templates/README.md)。

### 关键外部资源

**写作理念：**
- [Neel Nanda: How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
- [Sebastian Farquhar: How to Write ML Papers](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/)
- [Gopen & Swan: Science of Scientific Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- [Lipton: Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/)
- [Perez: Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/)

**API：** [Semantic Scholar](https://api.semanticscholar.org/api-docs/) | [CrossRef](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | [arXiv](https://info.arxiv.org/help/api/basics.html)

**会议：** [NeurIPS](https://neurips.cc/Conferences/2025/PaperInformation/StyleFiles) | [ICML](https://icml.cc/Conferences/2025/AuthorInstructions) | [ICLR](https://iclr.cc/Conferences/2026/AuthorGuide) | [ACL](https://github.com/acl-org/acl-style-files)