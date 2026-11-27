---
title: "Huggingface Tokenizers — 빠른 BPE/WordPiece 토큰화 및 사용자 지정 어휘 학습"
sidebar_label: "Huggingface Tokenizers"
description: "빠른 BPE/WordPiece 토큰화 및 사용자 지정 어휘 학습"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 수정하세요. */}

# Huggingface Tokenizers

Rust의 성능과 Python의 편리함을 갖춘 빠르고 프로덕션에 바로 사용할 수 있는 토크나이저입니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/huggingface-tokenizers`로 설치 |
| 경로 | `optional-skills/mlops/huggingface-tokenizers` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `tokenizers`, `transformers`, `datasets` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Tokenization`, `HuggingFace`, `BPE`, `WordPiece`, `Unigram`, `Fast Tokenization`, `Rust`, `Custom Tokenizer`, `Alignment Tracking`, `Production` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보는 내용입니다.
:::

# HuggingFace Tokenizers - NLP를 위한 빠른 토큰화

Rust 성능과 Python의 편리함을 갖춘 빠르고 프로덕션에 바로 사용할 수 있는 토크나이저입니다.

## HuggingFace Tokenizers를 사용할 때

**다음과 같은 경우 HuggingFace Tokenizers를 사용하세요:**
- 매우 빠른 토큰화가 필요한 경우 (텍스트 1GB당 &lt;20초)
- 처음부터 사용자 지정 토크나이저를 학습하는 경우
- 정렬 추적이 필요한 경우 (토큰 → 원본 텍스트 위치)
- 프로덕션 NLP 파이프라인을 구축하는 경우
- 대규모 말뭉치를 효율적으로 토큰화해야 하는 경우

**성능**:
- **속도**: CPU에서 1GB를 토큰화하는 데 &lt;20초
- **구현**: Python/Node.js 바인딩을 사용하는 Rust 코어
- **효율성**: 순수 Python 구현보다 10~100배 빠름

**대신 다음 대안을 사용하세요**:
- **SentencePiece**: 언어 독립적이며 T5/ALBERT에서 사용
- **tiktoken**: GPT 모델용 OpenAI BPE 토크나이저
- **transformers AutoTokenizer**: 사전 학습 모델 로드 전용 (내부적으로 이 라이브러리를 사용)

## 빠른 시작

### 설치

```bash
# Install tokenizers
pip install tokenizers

# With transformers integration
pip install tokenizers transformers
```

### 사전 학습 토크나이저 로드

```python
from tokenizers import Tokenizer

# Load from HuggingFace Hub
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Encode text
output = tokenizer.encode("Hello, how are you?")
print(output.tokens)  # ['hello', ',', 'how', 'are', 'you', '?']
print(output.ids)     # [7592, 1010, 2129, 2024, 2017, 1029]

# Decode back
text = tokenizer.decode(output.ids)
print(text)  # "hello, how are you?"
```

### 사용자 지정 BPE 토크나이저 학습

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Configure trainer
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2
)

# Train on files
files = ["train.txt", "validation.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("my-tokenizer.json")
```

**학습 시간**: 100MB 말뭉치는 약 1~2분, 1GB는 약 10~20분

### 패딩을 사용한 배치 인코딩

```python
# Enable padding
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

# Encode batch
texts = ["Hello world", "This is a longer sentence"]
encodings = tokenizer.encode_batch(texts)

for encoding in encodings:
    print(encoding.ids)
# [101, 7592, 2088, 102, 3, 3, 3]
# [101, 2023, 2003, 1037, 2936, 6251, 102]
```

## 토큰화 알고리즘

### BPE (Byte-Pair Encoding)

**작동 방식**:
1. 문자 수준 어휘로 시작
2. 가장 빈번한 문자 쌍 찾기
3. 새 토큰으로 병합하고 어휘에 추가
4. 어휘 크기에 도달할 때까지 반복

**사용 모델**: GPT-2, GPT-3, RoBERTa, BART, DeBERTa

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

**장점**:
- OOV 단어를 잘 처리함 (서브워드로 분할)
- 유연한 어휘 크기
- 형태적으로 풍부한 언어에 적합

**절충점**:
- 토큰화가 병합 순서에 따라 달라짐
- 일반적인 단어가 예상치 않게 분할될 수 있음

### WordPiece

**작동 방식**:
1. 문자 어휘로 시작
2. 병합 쌍 점수 계산: `frequency(pair) / (frequency(first) × frequency(second))`
3. 점수가 가장 높은 쌍 병합
4. 어휘 크기에 도달할 때까지 반복

**사용 모델**: BERT, DistilBERT, MobileBERT

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

**장점**:
- 의미 있는 병합을 우선함 (높은 점수 = 의미적으로 관련 있음)
- BERT에서 성공적으로 사용됨 (최첨단 결과)

**절충점**:
- 서브워드 일치 항목이 없으면 알 수 없는 단어가 `[UNK]`가 됨
- 병합 규칙이 아닌 어휘를 저장함 (파일이 더 커짐)

### Unigram

**작동 방식**:
1. 큰 어휘로 시작 (모든 부분 문자열)
2. 현재 어휘로 말뭉치의 손실 계산
3. 손실에 미치는 영향이 가장 작은 토큰 제거
4. 어휘 크기에 도달할 때까지 반복

**사용 모델**: ALBERT, T5, mBART, XLNet (SentencePiece를 통해)

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

**장점**:
- 확률 기반 (가장 가능성 높은 토큰화를 찾음)
- 단어 경계가 없는 언어에 적합
- 다양한 언어적 맥락을 처리

**절충점**:
- 학습에 계산 비용이 많이 듦
- 조정할 하이퍼파라미터가 더 많음

## 토큰화 파이프라인

전체 파이프라인: **정규화 → 사전 토큰화 → 모델 → 후처리**

### 정규화

텍스트를 정리하고 표준화합니다.

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

**일반적인 정규화 도구**:
- `NFD`, `NFC`, `NFKD`, `NFKC` - 유니코드 정규화 형식
- `Lowercase()` - 소문자로 변환
- `StripAccents()` - 악센트 제거 (é → e)
- `Strip()` - 공백 제거
- `Replace(pattern, content)` - 정규식 치환

### 사전 토큰화

텍스트를 단어와 유사한 단위로 나눕니다.

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

**일반적인 사전 토큰화 도구**:
- `Whitespace()` - 공백, 탭, 줄바꿈으로 분할
- `ByteLevel()` - GPT-2 방식의 바이트 수준 분할
- `Punctuation()` - 구두점을 분리
- `Digits(individual_digits=True)` - 숫자를 개별적으로 분할
- `Metaspace()` - 공백을 ▁로 치환 (SentencePiece 방식)

### 후처리

모델 입력에 특수 토큰을 추가합니다.

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

**일반적인 패턴**:
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

## 정렬 추적

원본 텍스트에서 토큰의 위치를 추적합니다.

```python
output = tokenizer.encode("Hello, world!")

# Get token offsets
for token, offset in zip(output.tokens, output.offsets):
    start, end = offset
    print(f"{token:10} → [{start:2}, {end:2}): {text[start:end]!r}")

# Output:
# hello      → [ 0,  5): 'Hello'
# ,          → [ 5,  6): ','
# world      → [ 7, 12): 'world'
# !          → [12, 13): '!'
```

**사용 사례**:
- 개체명 인식 (예측 결과를 텍스트로 다시 매핑)
- 질의응답 (답변 범위 추출)
- 토큰 분류 (레이블을 원본 위치에 정렬)

## transformers와 통합

### AutoTokenizer로 로드

```python
from transformers import AutoTokenizer

# AutoTokenizer automatically uses fast tokenizers
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Check if using fast tokenizer
print(tokenizer.is_fast)  # True

# Access underlying tokenizers.Tokenizer
fast_tokenizer = tokenizer.backend_tokenizer
print(type(fast_tokenizer))  # <class 'tokenizers.Tokenizer'>
```

### 사용자 지정 토크나이저를 transformers로 변환

```python
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

# Train custom tokenizer
tokenizer = Tokenizer(BPE())
# ... train tokenizer ...
tokenizer.save("my-tokenizer.json")

# Wrap for transformers
transformers_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my-tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Use like any transformers tokenizer
outputs = transformers_tokenizer(
    "Hello world",
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

## 일반적인 패턴

### 이터레이터에서 학습 (대규모 데이터셋)

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Create batch iterator
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]

# Train tokenizer
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset)  # For progress bar
)
```

**성능**: 1GB를 약 10~20분 내에 처리

### 잘라내기 및 패딩 활성화

```python
# Enable truncation
tokenizer.enable_truncation(max_length=512)

# Enable padding
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]",
    length=512  # Fixed length, or None for batch max
)

# Encode with both
output = tokenizer.encode("This is a long sentence that will be truncated...")
print(len(output.ids))  # 512
```

### 멀티프로세싱

```python
from tokenizers import Tokenizer
from multiprocessing import Pool

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

def encode_batch(texts):
    return tokenizer.encode_batch(texts)

# Process large corpus in parallel
with Pool(8) as pool:
    # Split corpus into chunks
    chunk_size = 1000
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

    # Encode in parallel
    results = pool.map(encode_batch, chunks)
```

**속도 향상**: 코어 8개 사용 시 5~8배

## 성능 벤치마크

### 학습 속도

| 말뭉치 크기 | BPE (어휘 30k) | WordPiece (30k) | Unigram (8k) |
|-------------|----------------|-----------------|--------------|
| 10 MB       | 15초           | 18초            | 25초         |
| 100 MB      | 1.5분          | 2분             | 4분          |
| 1 GB        | 15분           | 20분            | 40분         |

**하드웨어**: 16코어 CPU, 영어 Wikipedia에서 테스트
### 토큰화 속도

| 구현 | 1GB 코퍼스 | 처리량 |
|----------------|-------------|-----------|
| 순수 Python | 약 20분 | 약 50MB/분 |
| HF Tokenizers | 약 15초 | 약 4GB/분 |
| **속도 향상** | **80배** | **80배** |

**테스트**: 영어 텍스트, 평균 문장 길이 20단어

### 메모리 사용량

| 작업 | 메모리 |
|---------|---------|
| 토크나이저 로드 | 약 10MB |
| BPE 학습 (어휘 30k) | 약 200MB |
| 100만 개 문장 인코딩 | 약 500MB |

## 지원 모델

`from_pretrained()`를 통해 사용할 수 있는 사전 학습 토크나이저:

**BERT 계열**:
- `bert-base-uncased`, `bert-large-cased`
- `distilbert-base-uncased`
- `roberta-base`, `roberta-large`

**GPT 계열**:
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `distilgpt2`

**T5 계열**:
- `t5-small`, `t5-base`, `t5-large`
- `google/flan-t5-xxl`

**기타**:
- `facebook/bart-base`, `facebook/mbart-large-cc25`
- `albert-base-v2`, `albert-xlarge-v2`
- `xlm-roberta-base`, `xlm-roberta-large`

전체 목록 보기: https://huggingface.co/models?library=tokenizers

## 참고 자료

- **[학습 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/training.md)** - 사용자 정의 토크나이저 학습, 트레이너 구성, 대규모 데이터셋 처리
- **[알고리즘 심층 분석](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/algorithms.md)** - BPE, WordPiece, Unigram을 자세히 설명
- **[파이프라인 구성 요소](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/pipeline.md)** - 정규화 도구, 사전 토큰화 도구, 후처리기, 디코더
- **[Transformers 통합](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/integration.md)** - AutoTokenizer, PreTrainedTokenizerFast, 특수 토큰

## 리소스

- **문서**: https://huggingface.co/docs/tokenizers
- **GitHub**: https://github.com/huggingface/tokenizers ⭐ 9,000+
- **버전**: 0.20.0+
- **강좌**: https://huggingface.co/learn/nlp-course/chapter6/1
- **논문**: BPE (Sennrich et al., 2016), WordPiece (Schuster & Nakajima, 2012)
