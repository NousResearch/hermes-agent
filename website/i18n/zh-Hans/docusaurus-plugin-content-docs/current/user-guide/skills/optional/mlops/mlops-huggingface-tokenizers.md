---
title: "Huggingface Tokenizers — 为研究和生产优化的快速 tokenizer"
sidebar_label: "Huggingface Tokenizers"
description: "为研究和生产优化的快速 tokenizer"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Huggingface Tokenizers

为研究和生产优化的快速 tokenizer（分词器）。基于 Rust 的实现可在 &lt;20 秒内对 1GB 文本完成分词。支持 BPE、WordPiece 和 Unigram 算法。可训练自定义词表、追踪对齐关系、处理 padding（填充）/truncation（截断）。与 transformers 无缝集成。当需要高性能分词或训练自定义 tokenizer 时使用。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/huggingface-tokenizers` 安装 |
| 路径 | `optional-skills/mlops/huggingface-tokenizers` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `tokenizers`, `transformers`, `datasets` |
| 平台 | linux, macos, windows |
| 标签 | `Tokenization`, `HuggingFace`, `BPE`, `WordPiece`, `Unigram`, `Fast Tokenization`, `Rust`, `Custom Tokenizer`, `Alignment Tracking`, `Production` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# HuggingFace Tokenizers — 高性能 NLP 分词

具备 Rust 性能与 Python 易用性的快速、生产就绪 tokenizer。

## 何时使用 HuggingFace Tokenizers

**在以下情况下使用 HuggingFace Tokenizers：**
- 需要极快的分词速度（每 GB 文本 &lt;20 秒）
- 从头训练自定义 tokenizer
- 需要对齐追踪（token → 原始文本位置）
- 构建生产级 NLP 流水线
- 需要高效地对大型语料库进行分词

**性能**：
- **速度**：CPU 上对 1GB 文本分词 &lt;20 秒
- **实现**：Rust 核心，提供 Python/Node.js 绑定
- **效率**：比纯 Python 实现快 10–100 倍

**改用其他方案的情况**：
- **SentencePiece**：语言无关，被 T5/ALBERT 使用
- **tiktoken**：OpenAI 用于 GPT 模型的 BPE tokenizer
- **transformers AutoTokenizer**：仅加载预训练模型时使用（内部使用本库）

## 快速开始

### 安装

```bash
# 安装 tokenizers
pip install tokenizers

# 与 transformers 集成
pip install tokenizers transformers
```

### 加载预训练 tokenizer

```python
from tokenizers import Tokenizer

# 从 HuggingFace Hub 加载
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 对文本编码
output = tokenizer.encode("Hello, how are you?")
print(output.tokens)  # ['hello', ',', 'how', 'are', 'you', '?']
print(output.ids)     # [7592, 1010, 2129, 2024, 2017, 1029]

# 解码还原
text = tokenizer.decode(output.ids)
print(text)  # "hello, how are you?"
```

### 训练自定义 BPE tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 使用 BPE 模型初始化 tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 配置训练器
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2
)

# 在文件上训练
files = ["train.txt", "validation.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("my-tokenizer.json")
```

**训练时间**：100MB 语料约 1–2 分钟，1GB 语料约 10–20 分钟

### 批量编码与 padding

```python
# 启用 padding
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

# 批量编码
texts = ["Hello world", "This is a longer sentence"]
encodings = tokenizer.encode_batch(texts)

for encoding in encodings:
    print(encoding.ids)
# [101, 7592, 2088, 102, 3, 3, 3]
# [101, 2023, 2003, 1037, 2936, 6251, 102]
```

## 分词算法

### BPE（字节对编码）

**工作原理**：
1. 从字符级词表开始
2. 找出最频繁的字符对
3. 合并为新 token，加入词表
4. 重复直到达到词表大小

**使用者**：GPT-2、GPT-3、RoBERTa、BART、DeBERTa

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=50257,
    special_tokens=["<|endoftext|>"],
    min_frequency=2
)

tokenizer.train(files=["data.txt"], trainer=trainer)
```

**优点**：
- 能较好地处理 OOV 词（拆分为子词）
- 词表大小灵活
- 适合形态丰富的语言

**权衡**：
- 分词结果依赖合并顺序
- 可能意外拆分常见词

### WordPiece

**工作原理**：
1. 从字符词表开始
2. 对合并对打分：`frequency(pair) / (frequency(first) × frequency(second))`
3. 合并得分最高的对
4. 重复直到达到词表大小

**使用者**：BERT、DistilBERT、MobileBERT

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import BertNormalizer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=30522,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    continuing_subword_prefix="##"
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

**优点**：
- 优先进行有意义的合并（高分 = 语义相关）
- 在 BERT 中取得了最优结果

**权衡**：
- 若无子词匹配，未知词变为 `[UNK]`
- 保存词表而非合并规则（文件较大）

### Unigram

**工作原理**：
1. 从大词表（所有子串）开始
2. 用当前词表计算语料损失
3. 移除对损失影响最小的 token
4. 重复直到达到词表大小

**使用者**：ALBERT、T5、mBART、XLNet（通过 SentencePiece）

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

tokenizer = Tokenizer(Unigram())

trainer = UnigramTrainer(
    vocab_size=8000,
    special_tokens=["<unk>", "<s>", "</s>"],
    unk_token="<unk>"
)

tokenizer.train(files=["data.txt"], trainer=trainer)
```

**优点**：
- 概率化（找到最可能的分词方式）
- 适合无词边界的语言
- 能处理多样的语言学上下文

**权衡**：
- 训练计算开销较大
- 需要调整的超参数更多

## 分词流水线

完整流水线：**归一化 → 预分词 → 模型 → 后处理**

### 归一化（Normalization）

清洗并标准化文本：

```python
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Sequence

tokenizer.normalizer = Sequence([
    NFD(),           # Unicode 归一化（分解）
    Lowercase(),     # 转为小写
    StripAccents()   # 去除重音符号
])

# 输入："Héllo WORLD"
# 归一化后："hello world"
```

**常用归一化器**：
- `NFD`, `NFC`, `NFKD`, `NFKC` — Unicode 归一化形式
- `Lowercase()` — 转为小写
- `StripAccents()` — 去除重音（é → e）
- `Strip()` — 去除空白
- `Replace(pattern, content)` — 正则替换

### 预分词（Pre-tokenization）

将文本拆分为类词单元：

```python
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence, ByteLevel

# 按空白和标点拆分
tokenizer.pre_tokenizer = Sequence([
    Whitespace(),
    Punctuation()
])

# 输入："Hello, world!"
# 预分词后：["Hello", ",", "world", "!"]
```

**常用预分词器**：
- `Whitespace()` — 按空格、制表符、换行符拆分
- `ByteLevel()` — GPT-2 风格的字节级拆分
- `Punctuation()` — 隔离标点
- `Digits(individual_digits=True)` — 逐个拆分数字
- `Metaspace()` — 将空格替换为 ▁（SentencePiece 风格）

### 后处理（Post-processing）

为模型输入添加特殊 token：

```python
from tokenizers.processors import TemplateProcessing

# BERT 风格：[CLS] sentence [SEP]
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)
```

**常见模式**：
```python
# GPT-2：sentence <|endoftext|>
TemplateProcessing(
    single="$A <|endoftext|>",
    special_tokens=[("<|endoftext|>", 50256)]
)

# RoBERTa：<s> sentence </s>
TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[("<s>", 0), ("</s>", 2)]
)
```

## 对齐追踪

追踪 token 在原始文本中的位置：

```python
output = tokenizer.encode("Hello, world!")

# 获取 token 偏移量
for token, offset in zip(output.tokens, output.offsets):
    start, end = offset
    print(f"{token:10} → [{start:2}, {end:2}): {text[start:end]!r}")

# 输出：
# hello      → [ 0,  5): 'Hello'
# ,          → [ 5,  6): ','
# world      → [ 7, 12): 'world'
# !          → [12, 13): '!'
```

**使用场景**：
- 命名实体识别（将预测结果映射回文本）
- 问答（提取答案片段）
- Token 分类（将标签对齐到原始位置）

## 与 transformers 集成

### 使用 AutoTokenizer 加载

```python
from transformers import AutoTokenizer

# AutoTokenizer 自动使用快速 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 检查是否使用快速 tokenizer
print(tokenizer.is_fast)  # True

# 访问底层 tokenizers.Tokenizer
fast_tokenizer = tokenizer.backend_tokenizer
print(type(fast_tokenizer))  # <class 'tokenizers.Tokenizer'>
```

### 将自定义 tokenizer 转换为 transformers 格式

```python
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

# 训练自定义 tokenizer
tokenizer = Tokenizer(BPE())
# ... 训练 tokenizer ...
tokenizer.save("my-tokenizer.json")

# 封装为 transformers 格式
transformers_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my-tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# 像使用任何 transformers tokenizer 一样使用
outputs = transformers_tokenizer(
    "Hello world",
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

## 常见模式

### 从迭代器训练（大型数据集）

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# 创建批量迭代器
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]

# 训练 tokenizer
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset)  # 用于进度条
)
```

**性能**：约 10–20 分钟处理 1GB

### 启用 truncation 和 padding

```python
# 启用 truncation
tokenizer.enable_truncation(max_length=512)

# 启用 padding
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]",
    length=512  # 固定长度，或 None 表示批次最大长度
)

# 同时编码
output = tokenizer.encode("This is a long sentence that will be truncated...")
print(len(output.ids))  # 512
```

### 多进程处理

```python
from tokenizers import Tokenizer
from multiprocessing import Pool

# 加载 tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

def encode_batch(texts):
    return tokenizer.encode_batch(texts)

# 并行处理大型语料库
with Pool(8) as pool:
    # 将语料库拆分为块
    chunk_size = 1000
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

    # 并行编码
    results = pool.map(encode_batch, chunks)
```

**加速比**：8 核下约 5–8 倍

## 性能基准

### 训练速度

| 语料大小 | BPE（30k 词表） | WordPiece（30k） | Unigram（8k） |
|----------|----------------|-----------------|--------------|
| 10 MB    | 15 秒          | 18 秒           | 25 秒        |
| 100 MB   | 1.5 分钟       | 2 分钟          | 4 分钟       |
| 1 GB     | 15 分钟        | 20 分钟         | 40 分钟      |

**硬件**：16 核 CPU，在英文 Wikipedia 上测试

### 分词速度

| 实现方式        | 1 GB 语料   | 吞吐量        |
|----------------|-------------|--------------|
| 纯 Python      | ~20 分钟    | ~50 MB/分钟  |
| HF Tokenizers  | ~15 秒      | ~4 GB/分钟   |
| **加速比**     | **80×**     | **80×**      |

**测试**：英文文本，平均句长 20 词

### 内存占用

| 任务                    | 内存     |
|-------------------------|---------|
| 加载 tokenizer          | ~10 MB  |
| 训练 BPE（30k 词表）    | ~200 MB |
| 编码 100 万句           | ~500 MB |

## 支持的模型

可通过 `from_pretrained()` 获取的预训练 tokenizer：

**BERT 系列**：
- `bert-base-uncased`, `bert-large-cased`
- `distilbert-base-uncased`
- `roberta-base`, `roberta-large`

**GPT 系列**：
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `distilgpt2`

**T5 系列**：
- `t5-small`, `t5-base`, `t5-large`
- `google/flan-t5-xxl`

**其他**：
- `facebook/bart-base`, `facebook/mbart-large-cc25`
- `albert-base-v2`, `albert-xlarge-v2`
- `xlm-roberta-base`, `xlm-roberta-large`

浏览全部：https://huggingface.co/models?library=tokenizers

## 参考资料

- **[训练指南](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/training.md)** — 训练自定义 tokenizer、配置训练器、处理大型数据集
- **[算法深度解析](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/algorithms.md)** — BPE、WordPiece、Unigram 详细说明
- **[流水线组件](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/pipeline.md)** — 归一化器、预分词器、后处理器、解码器
- **[Transformers 集成](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/integration.md)** — AutoTokenizer、PreTrainedTokenizerFast、特殊 token

## 资源

- **文档**：https://huggingface.co/docs/tokenizers
- **GitHub**：https://github.com/huggingface/tokenizers ⭐ 9,000+
- **版本**：0.20.0+
- **课程**：https://huggingface.co/learn/nlp-course/chapter6/1
- **论文**：BPE（Sennrich et al., 2016）、WordPiece（Schuster & Nakajima, 2012）