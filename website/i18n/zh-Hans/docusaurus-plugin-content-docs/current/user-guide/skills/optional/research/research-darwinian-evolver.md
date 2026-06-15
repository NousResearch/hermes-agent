---
title: "Darwinian Evolver — 使用 Imbue 的进化循环来优化 prompt/正则/SQL/代码"
sidebar_label: "Darwinian Evolver"
description: "使用 Imbue 的进化循环来优化 prompt/正则/SQL/代码"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Darwinian Evolver

使用 Imbue 的进化循环来优化 prompt（提示词）/正则/SQL/代码。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/research/darwinian-evolver` 安装 |
| 路径 | `optional-skills/research/darwinian-evolver` |
| 版本 | `0.1.0` |
| 作者 | Bihruze (Asahi0x), Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos |
| 标签 | `evolution`, `optimization`, `prompt-engineering`, `research` |
| 相关 skill | [`arxiv`](/user-guide/skills/bundled/research/research-arxiv), [`jupyter-live-kernel`](/user-guide/skills/bundled/data-science/data-science-jupyter-live-kernel) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Darwinian Evolver

运行 Imbue 的 [darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver) —— 一个
由 LLM 驱动的进化搜索循环 —— 用于针对适应度函数优化 **prompt、正则表达式、SQL 查询
或小型代码片段**。

状态：对上游工具的轻量封装。该 skill 负责安装工具，引导 agent 编写 `Problem` 定义
（organism + evaluator + mutator），并通过上游 CLI 或一个小型自定义 Python 驱动脚本来运行循环。

**许可证：** 上游工具采用 **AGPL-3.0** 授权。该 skill 仅通过上游 CLI 或 `subprocess`/`uv run`
调用来调用它（纯聚合方式）。**不得**将上游类导入 Hermes 本身。

## 使用时机

- 用户说"优化这个 prompt"、"为 X 进化一个正则"、"自动改进这段代码/SQL"、"搜索更好的指令"。
- 你有一个评分器（精确匹配、正则通过率、单元测试、LLM 评判、运行时指标）以及一个起始候选（organism）。如果没有评分器，请先定义一个 —— 这才是难点所在。
- 成本可接受：一次典型运行需要 50–500 次 LLM 调用。使用 gpt-4o-mini 只需几美分；使用 Claude Sonnet 可能需要几美元。

**不适用**的情况：
- 优化目标可微分（请使用梯度下降 / DSPy）。
- 只需尝试 2–3 个变体 —— 直接手写即可。
- 适应度信号纯粹主观，没有可量化的标准。

## 前置条件

- Python ≥3.11
- `git`、`uv`（或 `pip`）
- 以下之一：`OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY`

该 skill 附带一个小型 `parrot_openrouter.py` 驱动脚本，通过 OpenAI SDK 使用 `OPENROUTER_API_KEY`，
因此 OpenRouter 上的任何模型均可使用。上游 CLI 本身硬编码了 Anthropic，需要 `ANTHROPIC_API_KEY`。

## 安装（一次性）

通过 `terminal` 工具运行：

```bash
mkdir -p ~/.hermes/cache/darwinian-evolver && cd ~/.hermes/cache/darwinian-evolver
[ -d darwinian_evolver ] || git clone --depth 1 https://github.com/imbue-ai/darwinian_evolver.git
cd darwinian_evolver && uv sync
```

验证：

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver \
  && uv run darwinian_evolver --help | head -5
```

## 快速开始 —— 内置 Parrot 示例

小型冒烟测试（需要 `ANTHROPIC_API_KEY`）：

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
uv run darwinian_evolver parrot \
  --num_iterations 2 \
  --num_parents_per_iteration 2 \
  --mutator_concurrency 2 --evaluator_concurrency 2 \
  --output_dir /tmp/parrot_demo
```

输出：
- `/tmp/parrot_demo/snapshots/iteration_N.pkl` —— 每次迭代的 pickle 序列化种群
- `/tmp/parrot_demo/<jsonl>` —— 每次迭代的 JSON 日志（路径在结束时打印）

在浏览器中打开 `~/.hermes/cache/darwinian-evolver/darwinian_evolver/darwinian_evolver/lineage_visualizer.html`
并加载 JSON 日志，即可查看进化树。

## 快速开始 —— OpenRouter 驱动（无需 Anthropic Key）

该 skill 附带 `scripts/parrot_openrouter.py` —— 同样的 parrot 问题，但 LLM 调用通过
OpenRouter 进行，因此任何提供商均可使用。

```bash
# From wherever the skill is installed:
SKILL_DIR=~/.hermes/skills/research/darwinian-evolver
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver

cd "$DE_DIR" && \
  EVOLVER_MODEL='openai/gpt-4o-mini' \
  uv run --with openai python "$SKILL_DIR/scripts/parrot_openrouter.py" \
    --num_iterations 3 --num_parents_per_iteration 2 \
    --output_dir /tmp/parrot_or
```

使用 `scripts/show_snapshot.py` 查看结果：

```bash
uv run --with openai python "$SKILL_DIR/scripts/show_snapshot.py" \
  /tmp/parrot_or/snapshots/iteration_3.pkl
```

预期输出：7 个按分数排名的进化 prompt 模板，最佳结果约在 0.6–0.8 之间（初始种子 `Say {{ phrase }}` 得分为 0.000）。

## 定义自定义问题

该 skill 附带 `templates/custom_problem_template.py` —— 复制、编辑、运行。
你必须定义三样东西：

1. **`Organism`** —— 一个 Pydantic `BaseModel` 子类，持有被进化的制品（`prompt_template: str`、`regex_pattern: str`、`sql_query: str`、`code_block: str` 等）。添加一个 `run(*args)` 方法来执行它。

2. **`Evaluator`** —— `.evaluate(organism) -> EvaluationResult(score=..., trainable_failure_cases=[...], holdout_failure_cases=[...], is_viable=True)`。
   - **`score`** 在 `[0, 1]` 范围内，越高越好。
   - **`trainable_failure_cases`** —— mutator 所看到的内容。包含足够的上下文（输入、期望值、实际值），以便 LLM 进行诊断。
   - **`holdout_failure_cases`** —— 对 mutator 隐藏。用于检测过拟合。
   - **`is_viable=True`**，除非 organism 完全损坏（抛出异常、返回 None 等）。得分为 0 的可行 organism 是可以的 —— 它只是在父代选择中权重较低。

3. **`Mutator`** —— `.mutate(organism, failure_cases, learning_log_entries) -> list[Organism]`。
   通常做法：构建一个包含当前 organism + 失败案例 + 修复请求的 LLM prompt；解析 LLM 的响应；返回一个新的 `Organism`。解析失败时返回 `[]` —— 循环会处理这种情况。

然后编写一个驱动脚本，将 `Problem(initial_organism, evaluator, [mutators])` 接入
`EvolveProblemLoop`，并在 `loop.run(num_iterations=N)` 上迭代 —— 附带的
`scripts/parrot_openrouter.py` 是参考实现。

## 实际影响较大的超参数

| 参数 | 默认值 | 何时调整 |
|---|---|---|
| `--num_iterations` | 5 | 一旦信任 evaluator，调高至 10–20 |
| `--num_parents_per_iteration` | 4 | 降至 2 以进行低成本探索 |
| `--mutator_concurrency` | 10 | 降至 2–4 以避免速率限制 |
| `--evaluator_concurrency` | 10 | 同上；evaluator 也会调用 LLM |
| `--batch_size` | 1 | 一旦 mutator 能处理多个失败案例，调高至 3–5 |
| `--verify_mutations` | 关闭 | 一旦 mutator 浪费严重时开启（据 Imbue，后续运行可节省 >10× 成本） |
| `--midpoint_score` | `p75` | 除非分数聚集，否则保持不变 |
| `--sharpness` | 10 | 保持不变 |

## 常见陷阱

1. **`Initial organism must be viable`** —— 即使种子得分为 0，也要在 `EvaluationResult` 中设置 `is_viable=True`。循环拒绝不可行的 organism，因为这意味着循环没有任何可进化的起点。
2. **提供商内容过滤会中断运行。** 基于 Azure 的 OpenRouter 模型会以 HTTP 400 拒绝"ignore previous instructions"等短语。将 LLM 调用包裹在 `try/except` 中，并返回 `f"<LLM_ERROR: {e}>"` —— evolver 会将该 organism 评分为 0 并继续。
3. **`loop.run()` 是一个生成器** —— 调用它不会执行任何操作，直到你对其迭代。使用 `for snap in loop.run(num_iterations=N):`。
4. **快照是嵌套 pickle。** `iteration_N.pkl` 包含一个带有 `population_snapshot`（更多 pickle 字节）的字典。要反序列化，必须让 `Organism` 类在与 pickle 时相同的点分路径下可导入。
5. **并发默认值较激进。** 10/10 会在大多数提供商上触发速率限制。从 2/2 开始。
6. **CLI 硬编码为 Anthropic。** `uv run darwinian_evolver <problem>` 会查找 `ANTHROPIC_API_KEY` 并使用 Claude Sonnet。要使用其他提供商，请编写类似 `parrot_openrouter.py` 的驱动脚本。
7. **AGPL 协议。** 永远不要在 Hermes 核心中使用 `from darwinian_evolver import ...`。`~/.hermes/skills/...` 下的自定义驱动脚本属于用户侧，没有问题。
8. **没有 PyPI 包。** `pip install darwinian-evolver` 会安装错误的东西。始终从 GitHub 仓库安装。

## 验证

安装完成并运行一次 parrot 后，以下命令退出码为 0 即表示验证通过：

```bash
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver
ls "$DE_DIR/darwinian_evolver/lineage_visualizer.html" >/dev/null && \
cd "$DE_DIR" && uv run darwinian_evolver --help >/dev/null && \
echo "darwinian-evolver: OK"
```

## 参考资料

- [Imbue 研究文章](https://imbue.com/research/2026-02-27-darwinian-evolver/)
- [ARC-AGI-2 结果](https://imbue.com/research/2026-02-27-arc-agi-2-evolution/)
- [imbue-ai/darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver)（AGPL-3.0）
- [Darwin Gödel Machines](https://arxiv.org/abs/2505.22954)
- [PromptBreeder](https://arxiv.org/abs/2309.16797)