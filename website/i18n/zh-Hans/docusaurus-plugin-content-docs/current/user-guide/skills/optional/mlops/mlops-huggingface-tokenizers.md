---
title: "Huggingface Tokenizers — 为研究和生产优化的快速分词器"
sidebar_label: "Huggingface Tokenizers"
description: "为研究和生产优化的快速分词器"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Huggingface Tokenizers

为研究和生产优化的快速分词器。基于 Rust 实现，1GB 文本分词 <20 秒。支持 BPE、WordPiece 和 Unigram 算法。可训练自定义词表、追踪对齐、处理填充/截断。与 transformers 无缝集成。适用于需要高性能分词或自定义分词器训练的场景。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/huggingface-tokenizers` |
| Path | `optional-skills/mlops/huggingface-tokenizers` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `tokenizers`, `transformers`, `datasets` |
| Tags | `Tokenization`, `HuggingFace`, `BPE`, `WordPiece`, `Unigram`, `Fast Tokenization`, `Rust`, `Custom Tokenizer`, `Alignment Tracking`, `Production` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# HuggingFace Tokenizers - Fast Tokenization for NLP

快速、生产就绪的分词器，兼具 Rust 性能和 Python 易用性。

## 何时使用 HuggingFace Tokenizers

**Use HuggingFace Tokenizers when:**
- Need extremely fast tokenization (&lt;20s per GB of text)
- 训练ing custom tokenizers from scratch
- Want alignment tracking (token → original text position)
- Building production NLP pipelines
- Need to tokenize large corpora efficiently

**Performance**:
- **Speed**: &lt;20 seconds to tokenize 1GB on CPU
- **Implementation**: Rust core with Python/Node.js bindings
- **Efficiency**: 10-100× faster than pure Python implementations

**替代方案**:
- **SentencePiece**: Language-independent, used by T5/ALBERT
- **tiktoken**: OpenAI's BPE tokenizer for GPT models
- **transformers AutoTokenizer**: Loading pretrained only (uses this library internally)

## 快速入门

### 安装

```bash
# Install tokenizers
pip install tokenizers

# With transformers integration
pip install tokenizers transformers
```

### 加载预训练分词器

```python
from tokenizers import Tokenizer

# 从 HuggingFace Hub 加载
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 编码文本
output = tokenizer.encode("Hello, how are you?")
print(output.tokens)  # ['hello', ',', 'how', 'are', 'you', '?']
print(output.ids)     # [7592, 1010, 2129, 2024, 2017, 1029]

# 解码回来
text = tokenizer.decode(output.ids)
print(text)  # "hello, how are you?"
```

### 训练自定义 BPE 分词器

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import Bpe训练er
from tokenizers.pre_tokenizers import Whitespace

# 初始化 tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 配置 trainer
trainer = Bpe训练er(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2
)

# 训练 on files
files = ["train.txt", "validation.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("my-tokenizer.json")
```

**训练ing time**: ~1-2 minutes for 100MB corpus, ~10-20 minutes for 1GB

### 带填充的批量编码

```python
# 启用填充
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

# Encode batch
texts = ["Hello world", "This is a longer sentence"]
encodings = tokenizer.encode_batch(texts)

for encoding in encodings:
    print(encoding.ids)
# [101, 7592, 2088, 102, 3, 3, 3]
# [101, 2023, 2003, 1037, 2936, 6251, 102]
```

## Tokenization algorithms

### BPE（字节对编码）

**How it works**:
1. Start with character-level vocabulary
2. Find most frequent character pair
3. Merge into new token, add to vocabulary
4. Repeat until vocabulary size reached

**Used by**: GPT-2, GPT-3, RoBERTa, BART, DeBERTa

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import Bpe训练er
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
tokenizer.pre_tokenizer = ByteLevel()

trainer = Bpe训练er(
    vocab_size=50257,
    special_tokens=["<|endoftext|>"],
    min_frequency=2
)

tokenizer.train(files=["data.txt"], trainer=trainer)
```

**Advantages**:
- Handles OOV words well (breaks into subwords)
- Flexible vocabulary size
- Good for morphologically rich languages

**Trade-offs**:
- Tokenization depends on merge order
- May split common words unexpectedly

### WordPiece

**How it works**:
1. Start with character vocabulary
2. Score merge pairs: `frequency(pair) / (frequency(first) × frequency(second))`
3. Merge highest scoring pair
4. Repeat until vocabulary size reached

**Used by**: BERT, DistilBERT, MobileBERT

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPiece训练er
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import BertNormalizer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPiece训练er(
    vocab_size=30522,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    continuing_subword_prefix="##"
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

**Advantages**:
- Prioritizes meaningful merges (high score = semantically related)
- Used successfully in BERT (state-of-the-art results)

**Trade-offs**:
- Unknown words become `[UNK]` if no subword match
- 保存s vocabulary, not merge rules (larger files)

### Unigram

**How it works**:
1. Start with large vocabulary (all substrings)
2. Compute loss for corpus with current vocabulary
3. Remove tokens with minimal impact on loss
4. Repeat until vocabulary size reached

**Used by**: ALBERT, T5, mBART, XLNet (via SentencePiece)

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import Unigram训练er

tokenizer = Tokenizer(Unigram())

trainer = Unigram训练er(
    vocab_size=8000,
    special_tokens=["<unk>", "<s>", "</s>"],
    unk_token="<unk>"
)

tokenizer.train(files=["data.txt"], trainer=trainer)
```

**Advantages**:
- Probabilistic (finds most likely tokenization)
- Works well for languages without word boundaries
- Handles diverse linguistic contexts

**Trade-offs**:
- Computationally expensive to train
- More hyperparameters to tune

## Tokenization pipeline

Complete pipeline: **归一化 → 预分词 → Model → 后处理**

### 归一化

清理和标准化文本:

```python
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Sequence

tokenizer.normalizer = Sequence([
    NFD(),           # Unicode normalization (decompose)
    Lowercase(),     # Convert to lowercase
    StripAccents()   # Remove accents
])

# Input: "Héllo WORLD"
# After normalization: "hello world"
```

**Common normalizers**:
- `NFD`, `NFC`, `NFKD`, `NFKC` - Unicode normalization forms
- `Lowercase()` - Convert to lowercase
- `StripAccents()` - Remove accents (é → e)
- `Strip()` - Remove whitespace
- `Replace(pattern, content)` - Regex replacement

### 预分词

将文本分割为类似单词的单元:

```python
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence, ByteLevel

# Split on whitespace and punctuation
tokenizer.pre_tokenizer = Sequence([
    Whitespace(),
    Punctuation()
])

# Input: "Hello, world!"
# After pre-tokenization: ["Hello", ",", "world", "!"]
```

**Common pre-tokenizers**:
- `Whitespace()` - Split on spaces, tabs, newlines
- `ByteLevel()` - GPT-2 style byte-level splitting
- `Punctuation()` - Isolate punctuation
- `Digits(individual_digits=True)` - Split digits individually
- `Metaspace()` - Replace spaces with ▁ (SentencePiece style)

### 后处理

为模型输入添加特殊标记:

```python
from tokenizers.processors import TemplateProcessing

# BERT-style: [CLS] sentence [SEP]
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)
```

**Common patterns**:
```python
# GPT-2: sentence <|endoftext|>
TemplateProcessing(
    single="$A <|endoftext|>",
    special_tokens=[("<|endoftext|>", 50256)]
)

# RoBERTa: <s> sentence </s>
TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[("<s>", 0), ("</s>", 2)]
)
```

## Alignment tracking

追踪令牌在原始文本中的位置:

```python
output = tokenizer.encode("Hello, world!")

# 获取令牌偏移量
for token, offset in zip(output.tokens, output.offsets):
    start, end = offset
    print(f"{token:10} → [{start:2}, {end:2}): {text[start:end]!r}")

# Output:
# hello      → [ 0,  5): 'Hello'
# ,          → [ 5,  6): ','
# world      → [ 7, 12): 'world'
# !          → [12, 13): '!'
```

**用例**:
- 命名实体识别（将预测映射回文本）
- 问答（提取答案范围）
- 令牌分类（将标签对齐到原始位置）

## Integration with transformers

### 使用 AutoTokenizer 加载

```python
from transformers import AutoTokenizer

# AutoTokenizer 自动使用快速分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 检查是否使用快速分词器
print(tokenizer.is_fast)  # True

# 访问底层 tokenizers.Tokenizer
fast_tokenizer = tokenizer.backend_tokenizer
print(type(fast_tokenizer))  # <class 'tokenizers.Tokenizer'>
```

### 将自定义分词器转换为 transformers 格式

```python
from tokenizers import Tokenizer
from transformers import Pre训练edTokenizerFast

# 训练 custom tokenizer
tokenizer = Tokenizer(BPE())
# ... train tokenizer ...
tokenizer.save("my-tokenizer.json")

# 为 transformers 封装
transformers_tokenizer = Pre训练edTokenizerFast(
    tokenizer_file="my-tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# 像使用任何 transformers 分词器一样使用
outputs = transformers_tokenizer(
    "Hello world",
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

## Common patterns

### 从迭代器训练（大数据集）

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# 创建批量迭代器
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]

# 训练分词器
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset)  # 用于进度条
)
```

**Performance**: Processes 1GB in ~10-20 minutes

### 启用截断和填充

```python
# 启用截断
tokenizer.enable_truncation(max_length=512)

# 启用填充
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]",
    length=512  # Fixed length, or None for batch max
)

# 同时使用两者编码
output = tokenizer.encode("This is a long sentence that will be truncated...")
print(len(output.ids))  # 512
```

### Multi-processing

```python
from tokenizers import Tokenizer
from multiprocessing import Pool

# 加载分词器
tokenizer = Tokenizer.from_file("tokenizer.json")

def encode_batch(texts):
    return tokenizer.encode_batch(texts)

# 并行处理大型语料
with Pool(8) as pool:
    # 将语料分成块
    chunk_size = 1000
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

    # 并行编码
    results = pool.map(encode_batch, chunks)
```

**加速比**: 5-8× with 8 cores

## 性能基准

### 训练ing speed

| 语料大小 | BPE（30k 词表） | WordPiece（30k） | Unigram（8k） |
|-------------|-----------------|-----------------|--------------|
| 10 MB       | 15 sec          | 18 sec          | 25 sec       |
| 100 MB      | 1.5 min         | 2 min           | 4 min        |
| 1 GB        | 15 min          | 20 min          | 40 min       |

**硬件**: 16-core CPU, tested on English Wikipedia

### Tokenization speed

| Implementation | 1 GB corpus | Throughput    |
|----------------|-------------|---------------|
| Pure Python    | ~20 minutes | ~50 MB/min    |
| HF Tokenizers  | ~15 seconds | ~4 GB/min     |
| **加速比**    | **80×**     | **80×**       |

**测试**: English text, average sentence length 20 words

### Memory usage

| Task                    | Memory  |
|-------------------------|---------|
| 加载分词器          | ~10 MB  |
| 训练 BPE (30k vocab)   | ~200 MB |
| Encode 1M sentences     | ~500 MB |

## 支持 models

Pre-trained tokenizers available via `from_pretrained()`:

**BERT family**:
- `bert-base-uncased`, `bert-large-cased`
- `distilbert-base-uncased`
- `roberta-base`, `roberta-large`

**GPT family**:
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `distilgpt2`

**T5 family**:
- `t5-small`, `t5-base`, `t5-large`
- `google/flan-t5-xxl`

**Other**:
- `facebook/bart-base`, `facebook/mbart-large-cc25`
- `albert-base-v2`, `albert-xlarge-v2`
- `xlm-roberta-base`, `xlm-roberta-large`

Browse all: https://huggingface.co/models?library=tokenizers

## References

- **[训练ing Guide](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/training.md)** - 训练 custom tokenizers, configure trainers, handle large datasets
- **[Algorithms Deep Dive](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/algorithms.md)** - BPE, WordPiece, Unigram explained in detail
- **[Pipeline Components](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/pipeline.md)** - Normalizers, pre-tokenizers, post-processors, decoders
- **[Transformers Integration](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/integration.md)** - AutoTokenizer, Pre训练edTokenizerFast, special tokens

## Resources

- **Docs**: https://huggingface.co/docs/tokenizers
- **GitHub**: https://github.com/huggingface/tokenizers ⭐ 9,000+
- **Version**: 0.20.0+
- **Course**: https://huggingface.co/learn/nlp-course/chapter6/1
- **Paper**: BPE (Sennrich et al., 2016), WordPiece (Schuster & Nakajima, 2012)
