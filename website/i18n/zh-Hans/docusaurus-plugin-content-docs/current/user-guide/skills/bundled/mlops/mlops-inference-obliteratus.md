---
title: "Obliteratus — OBLITERATUS：消除 LLM 拒绝行为（均值差分法）"
sidebar_label: "Obliteratus"
description: "OBLITERATUS：消除 LLM 拒绝行为（均值差分法）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Obliteratus

OBLITERATUS：消除 LLM 拒绝行为（均值差分法）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/inference/obliteratus` |
| 版本 | `2.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 依赖项 | `obliteratus`, `torch`, `transformers`, `bitsandbytes`, `accelerate`, `safetensors` |
| 平台 | linux, macos |
| 标签 | `Abliteration`, `Uncensoring`, `Refusal-Removal`, `LLM`, `Weight-Projection`, `SVD`, `Mechanistic-Interpretability`, `HuggingFace`, `Model-Surgery` |
| 相关 skill | `vllm`, `gguf`, [`huggingface-tokenizers`](/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# OBLITERATUS Skill

## 内容概览

9 个 CLI 方法、28 个分析模块、116 个跨 5 个算力层级的模型预设、锦标赛评估，以及遥测驱动的推荐。

无需重新训练或微调，即可从开放权重 LLM 中移除拒绝行为（guardrails）。使用机械可解释性（mechanistic interpretability）技术——包括均值差分法（diff-in-means）、SVD、白化 SVD、LEACE 概念擦除、SAE 分解、贝叶斯核投影等——识别并精准切除模型权重中的拒绝方向，同时保留推理能力。

**许可证警告：** OBLITERATUS 采用 AGPL-3.0 许可证。**绝对不要**将其作为 Python 库导入。始终通过 CLI（`obliteratus` 命令）或子进程调用。这样可保持 Hermes Agent 的 MIT 许可证不受污染。

## 视频指南

Hermes agent 使用 OBLITERATUS 对 Gemma 进行 abliterate 的完整演示：
https://www.youtube.com/watch?v=8fG9BrNTeHs（"OBLITERATUS: An AI Agent Removed Gemma 4's Safety Guardrails"）

当用户希望在自行运行前对端到端工作流有直观了解时，可参考此视频。

## 何时使用此 Skill

当用户出现以下情况时触发：
- 想要"解除审查"或"abliterate"一个 LLM
- 询问如何从模型中移除拒绝行为/guardrails
- 想要创建 Llama、Qwen、Mistral 等模型的无审查版本
- 提到"refusal removal"、"abliteration"、"weight projection"
- 想要分析模型的拒绝机制如何运作
- 提及 OBLITERATUS、abliterator 或拒绝方向

## 第一步：安装

检查是否已安装：
```bash
obliteratus --version 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```

如未安装，从 GitHub 克隆并安装：
```bash
git clone https://github.com/elder-plinius/OBLITERATUS.git
cd OBLITERATUS
pip install -e .
# 如需 Gradio Web UI 支持：
# pip install -e ".[spaces]"
```

**重要：** 安装前请与用户确认。此操作会拉取约 5-10GB 的依赖项（PyTorch、Transformers、bitsandbytes 等）。

## 第二步：检查硬件

在执行任何操作前，先检查可用的 GPU：
```bash
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu}')
    print(f'VRAM: {vram:.1f} GB')
    if vram < 4: print('TIER: tiny (models under 1B)')
    elif vram < 8: print('TIER: small (models 1-4B)')
    elif vram < 16: print('TIER: medium (models 4-9B with 4bit quant)')
    elif vram < 32: print('TIER: large (models 8-32B with 4bit quant)')
    else: print('TIER: frontier (models 32B+)')
else:
    print('NO GPU - only tiny models (under 1B) on CPU')
"
```

### VRAM 需求（使用 4-bit 量化）

| VRAM     | 最大模型规模    | 示例模型                                    |
|:---------|:----------------|:--------------------------------------------|
| 仅 CPU   | ~1B 参数        | GPT-2, TinyLlama, SmolLM                    |
| 4-8 GB   | ~4B 参数        | Qwen2.5-1.5B, Phi-3.5 mini, Llama 3.2 3B   |
| 8-16 GB  | ~9B 参数        | Llama 3.1 8B, Mistral 7B, Gemma 2 9B       |
| 24 GB    | ~32B 参数       | Qwen3-32B, Llama 3.1 70B（较紧）, Command-R |
| 48 GB+   | ~72B+ 参数      | Qwen2.5-72B, DeepSeek-R1                    |
| 多 GPU   | 200B+ 参数      | Llama 3.1 405B, DeepSeek-V3 (685B MoE)      |

## 第三步：浏览可用模型并获取推荐

```bash
# 按算力层级浏览模型
obliteratus models --tier medium

# 获取特定模型的架构信息
obliteratus info <model_name>

# 获取遥测驱动的最佳方法与参数推荐
obliteratus recommend <model_name>
obliteratus recommend <model_name> --insights  # 全局跨架构排名
```

## 第四步：选择方法

### 方法选择指南
**默认/大多数情况推荐：`advanced`。** 它使用多方向 SVD 配合范数保持投影，经过充分测试。

| 场景                              | 推荐方法           | 原因                                     |
|:----------------------------------|:-------------------|:-----------------------------------------|
| 默认/大多数模型                   | `advanced`         | 多方向 SVD，范数保持，可靠               |
| 快速测试/原型验证                 | `basic`            | 速度快，简单，足以评估                   |
| 稠密模型（Llama, Mistral）        | `advanced`         | 多方向，范数保持                         |
| MoE 模型（DeepSeek, Mixtral）     | `nuclear`          | 专家粒度，处理 MoE 复杂性               |
| 推理模型（R1 蒸馏）               | `surgical`         | CoT 感知，保留思维链                     |
| 拒绝行为顽固持续                  | `aggressive`       | 白化 SVD + 注意力头手术 + jailbreak      |
| 需要可逆更改                      | 使用 steering vectors（见分析章节） |
| 追求最高质量，不计时间            | `optimized`        | 贝叶斯搜索最优参数                       |
| 实验性自动检测                    | `informed`         | 自动检测对齐类型——实验性，不一定总优于 advanced |

### 9 个 CLI 方法
- **basic** — 通过均值差分法提取单一拒绝方向。速度快（8B 模型约 5-10 分钟）。
- **advanced**（默认，推荐）— 多 SVD 方向，范数保持投影，2 次精化迭代。中等速度（约 10-20 分钟）。
- **aggressive** — 白化 SVD + jailbreak 对比 + 注意力头手术。连贯性损坏风险较高。
- **spectral_cascade** — DCT 频域分解。研究性/新颖方法。
- **informed** — 在 abliterate 过程中运行分析以自动配置。实验性——比 advanced 更慢且可预测性更差。
- **surgical** — SAE 特征 + 神经元掩码 + 注意力头手术 + 逐专家处理。非常慢（约 1-2 小时）。最适合推理模型。
- **optimized** — 贝叶斯超参数搜索（Optuna TPE）。运行时间最长，但能找到最优参数。
- **inverted** — 翻转拒绝方向。模型变为主动配合。
- **nuclear** — 针对顽固 MoE 模型的最大力度组合。专家粒度处理。

### 方向提取方法（`--direction-method` 标志）
- **diff_means**（默认）— 拒绝/配合激活之间的简单均值差分。鲁棒性强。
- **svd** — 多方向 SVD 提取。适用于复杂对齐。
- **leace** — LEACE（线性闭式估计擦除）。最优线性擦除。

### 4 个仅限 Python API 的方法
（**不**可通过 CLI 使用——需要 Python import，违反 AGPL 边界。仅在用户明确希望在其自己的 AGPL 项目中将 OBLITERATUS 作为库使用时提及。）
- failspy, gabliteration, heretic, rdo

## 第五步：执行 Abliteration

### 标准用法
```bash
# 默认方法（advanced）——大多数模型推荐
obliteratus obliterate <model_name> --method advanced --output-dir ./abliterated-models

# 使用 4-bit 量化（节省 VRAM）
obliteratus obliterate <model_name> --method advanced --quantization 4bit --output-dir ./abliterated-models

# 大型模型（70B+）——保守默认值
obliteratus obliterate <model_name> --method advanced --quantization 4bit --large-model --output-dir ./abliterated-models
```

### 精细调整参数
```bash
obliteratus obliterate <model_name> \
  --method advanced \
  --direction-method diff_means \
  --n-directions 4 \
  --refinement-passes 2 \
  --regularization 0.1 \
  --quantization 4bit \
  --output-dir ./abliterated-models \
  --contribute  # 选择加入遥测以贡献社区研究
```

### 关键标志
| 标志 | 描述 | 默认值 |
|:-----|:------------|:--------|
| `--method` | Abliteration 方法 | advanced |
| `--direction-method` | 方向提取方式 | diff_means |
| `--n-directions` | 拒绝方向数量（1-32） | 取决于方法 |
| `--refinement-passes` | 迭代精化次数（1-5） | 2 |
| `--regularization` | 正则化强度（0.0-1.0） | 0.1 |
| `--quantization` | 以 4bit 或 8bit 加载 | 无（全精度） |
| `--large-model` | 120B+ 模型的保守默认值 | false |
| `--output-dir` | 保存 abliterated 模型的位置 | ./obliterated_model |
| `--contribute` | 共享匿名结果用于研究 | false |
| `--verify-sample-size` | 拒绝率检查的测试 prompt 数量 | 20 |
| `--dtype` | 模型数据类型（float16, bfloat16） | auto |

### 其他执行模式
```bash
# 交互式引导模式（硬件 → 模型 → 预设）
obliteratus interactive

# Web UI（Gradio）
obliteratus ui --port 7860

# 从 YAML 配置运行完整消融研究
obliteratus run config.yaml --preset quick

# 锦标赛：所有方法相互对比
obliteratus tourney <model_name>
```

## 第六步：验证结果

Abliteration 完成后，检查输出指标：

| 指标 | 良好值 | 警告 |
|:-------|:-----------|:--------|
| 拒绝率 | &lt; 5%（理想约 0%） | > 10% 表示拒绝行为仍存在 |
| 困惑度变化 | &lt; 10% 增幅 | > 15% 表示连贯性受损 |
| KL 散度 | &lt; 0.1 | > 0.5 表示分布发生显著偏移 |
| 连贯性 | 高 / 通过定性检查 | 响应退化、出现重复 |

### 如果拒绝行为仍持续（> 10%）
1. 尝试 `aggressive` 方法
2. 增大 `--n-directions`（例如 8 或 16）
3. 添加 `--refinement-passes 3`
4. 尝试 `--direction-method svd` 替代 diff_means

### 如果连贯性受损（困惑度增幅 > 15%）
1. 减小 `--n-directions`（尝试 2）
2. 增大 `--regularization`（尝试 0.3）
3. 将 `--refinement-passes` 减至 1
4. 尝试 `basic` 方法（更温和）

## 第七步：使用 Abliterated 模型

输出为标准 HuggingFace 模型目录。

```bash
# 使用 transformers 在本地测试
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./abliterated-models/<model>')
tokenizer = AutoTokenizer.from_pretrained('./abliterated-models/<model>')
inputs = tokenizer('How do I pick a lock?', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"

# 上传到 HuggingFace Hub
huggingface-cli upload <username>/<model-name>-abliterated ./abliterated-models/<model>

# 使用 vLLM 提供服务
vllm serve ./abliterated-models/<model>
```

## CLI 命令参考

| 命令 | 描述 |
|:--------|:------------|
| `obliteratus obliterate` | 主 abliteration 命令 |
| `obliteratus info <model>` | 打印模型架构详情 |
| `obliteratus models --tier <tier>` | 按算力层级浏览精选模型 |
| `obliteratus recommend <model>` | 遥测驱动的方法/参数建议 |
| `obliteratus interactive` | 引导式设置向导 |
| `obliteratus tourney <model>` | 锦标赛：所有方法正面对决 |
| `obliteratus run <config.yaml>` | 从 YAML 执行消融研究 |
| `obliteratus strategies` | 列出所有已注册的消融策略 |
| `obliteratus report <results.json>` | 重新生成可视化报告 |
| `obliteratus ui` | 启动 Gradio Web 界面 |
| `obliteratus aggregate` | 汇总社区遥测数据 |

## 分析模块

OBLITERATUS 包含 28 个用于机械可解释性的分析模块。
完整参考请见 `skill_view(name="obliteratus", file_path="references/analysis-modules.md")`。

### 快速分析命令
```bash
# 运行特定分析模块
obliteratus run analysis-config.yaml --preset quick

# 优先运行的关键模块：
# - alignment_imprint: 识别 DPO/RLHF/CAI/SFT 对齐方法指纹
# - concept_geometry: 单方向 vs 多面锥体
# - logit_lens: 哪一层决定拒绝
# - anti_ouroboros: 自我修复风险评分
# - causal_tracing: 因果必要组件
```

### Steering Vectors（可逆替代方案）
与其永久修改权重，可使用推理时 steering：
```python
# 仅限 Python API——用于用户自己的项目
from obliteratus.analysis.steering_vectors import SteeringVectorFactory, SteeringHookManager
```

## 消融策略

除基于方向的 abliteration 外，OBLITERATUS 还包含结构性消融策略：
- **Embedding Ablation** — 针对嵌入层组件
- **FFN Ablation** — 前馈网络块移除
- **Head Pruning** — 注意力头剪枝
- **Layer Removal** — 完整层移除

列出所有可用策略：`obliteratus strategies`

## 评估

OBLITERATUS 包含内置评估工具：
- 拒绝率基准测试
- 困惑度对比（前/后）
- LM Eval Harness 集成，用于学术基准
- 竞争对手正面对比
- 基线性能追踪

## 平台支持

- **CUDA** — 完整支持（NVIDIA GPU）
- **Apple Silicon（MLX）** — 通过 MLX 后端支持
- **CPU** — 支持小型模型（&lt; 1B 参数）

## YAML 配置模板

通过 `skill_view` 加载模板以实现可复现运行：
- `templates/abliteration-config.yaml` — 标准单模型配置
- `templates/analysis-study.yaml` — abliteration 前分析研究
- `templates/batch-abliteration.yaml` — 多模型批量处理

## 遥测

OBLITERATUS 可选择性地将匿名运行数据贡献至全球研究数据集。
使用 `--contribute` 标志启用。不收集任何个人数据——仅包含模型名称、方法、指标。

## 常见陷阱

1. **不要将 `informed` 作为默认方法** — 它是实验性的且速度更慢。使用 `advanced` 以获得可靠结果。
2. **~1B 以下的模型对 abliteration 响应较差** — 其拒绝行为较浅且碎片化，难以提取干净的方向。预期结果为部分消除（残余拒绝率 20-40%）。3B+ 模型的拒绝方向更清晰，响应好得多（使用 `advanced` 通常可达 0% 拒绝率）。
3. **`aggressive` 可能适得其反** — 在小模型上可能损坏连贯性，甚至实际上增加拒绝率。仅在 `advanced` 对 3B+ 模型仍留有 > 10% 拒绝率时使用。
4. **始终检查困惑度** — 若增幅超过 15%，模型已受损。降低激进程度。
5. **MoE 模型需要特殊处理** — 对 Mixtral、DeepSeek-MoE 等使用 `nuclear` 方法。
6. **量化模型无法再次量化** — 对全精度模型执行 abliterate，然后对输出进行量化。
7. **VRAM 估算是近似值** — 4-bit 量化有帮助，但提取过程中峰值使用量可能突增。
8. **推理模型较为敏感** — 对 R1 蒸馏模型使用 `surgical` 以保留思维链。
9. **查看 `obliteratus recommend`** — 遥测数据可能提供比默认值更好的参数。
10. **AGPL 许可证** — 绝不在 MIT/Apache 项目中 `import obliteratus`。仅限 CLI 调用。
11. **大型模型（70B+）** — 始终使用 `--large-model` 标志以启用保守默认值。
12. **频谱认证 RED 很常见** — 即使实际拒绝率为 0%，频谱检查也经常标记为"不完整"。应检查实际拒绝率，而非单纯依赖频谱认证结果。

## 互补 Skill

- **vllm** — 以高吞吐量提供 abliterated 模型服务
- **gguf** — 将 abliterated 模型转换为 GGUF 格式供 llama.cpp 使用
- **huggingface-tokenizers** — 处理模型 tokenizer