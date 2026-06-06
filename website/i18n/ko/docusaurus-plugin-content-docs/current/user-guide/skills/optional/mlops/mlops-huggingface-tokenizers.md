---
title: "Huggingface Tokenizers — 연구 및 프로덕션에 최적화된 빠른 토크나이저"
sidebar_label: "Huggingface Tokenizers"
description: "연구 및 프로덕션에 최적화된 빠른 토크나이저"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Huggingface Tokenizers

연구 및 프로덕션에 최적화된 빠른 토크나이저. Rust 기반 구현으로 20초 이내에 1GB를 토크나이징합니다. BPE, WordPiece 및 Unigram 알고리즘을 지원합니다. 사용자 지정 어휘집 학습, 정렬(alignment) 추적, 패딩/자르기(truncation) 처리 기능. transformers와 매끄럽게 통합됩니다. 고성능 토크나이징이나 사용자 정의 토크나이저 학습이 필요할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/huggingface-tokenizers`로 설치 |
| 경로 | `optional-skills/mlops/huggingface-tokenizers` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `tokenizers`, `transformers`, `datasets` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Tokenization`, `HuggingFace`, `BPE`, `WordPiece`, `Unigram`, `Fast Tokenization`, `Rust`, `Custom Tokenizer`, `Alignment Tracking`, `Production` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# HuggingFace Tokenizers - NLP를 위한 빠른 토크나이징

Rust의 성능과 Python의 사용 편의성을 결합한 프로덕션 지원이 되는 빠른 토크나이저.

## HuggingFace Tokenizers를 사용하는 경우

**다음과 같은 경우에 사용하세요:**
- 극도로 빠른 토크나이징이 필요할 때 (GB 텍스트 당 20초 이내)
- 토크나이저를 처음부터 학습해야 할 때
- 정렬 추적이 원할 때 (토큰 → 원본 텍스트 위치)
- 프로덕션 수준의 NLP 파이프라인 구축 시
- 방대한 코퍼스를 효율적으로 토크나이징해야 할 때

**성능**:
- **속도**: CPU에서 1GB를 토크나이징하는 데 20초 이내
- **구현**: Python/Node.js 바인딩이 있는 Rust 코어
- **효율성**: 순수 Python 구현에 비해 10-100배 빠름

**다음을 대신 사용해 보세요**:
- **SentencePiece**: 언어에 독립적이며, T5/ALBERT에서 사용됨
- **tiktoken**: GPT 모델을 위한 OpenAI의 BPE 토크나이저
- **transformers AutoTokenizer**: 사전 학습된 토크나이저만 불러올 때 (내부적으로 이 라이브러리를 사용함)

## 빠른 시작

### 설치

```bash
# tokenizers 설치
pip install tokenizers

# transformers 연동과 함께 설치
pip install tokenizers transformers
```

### 사전 학습된 토크나이저 로드

```python
from tokenizers import Tokenizer

# HuggingFace Hub에서 로드
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 텍스트 인코딩
output = tokenizer.encode("Hello, how are you?")
print(output.tokens)  # ['hello', ',', 'how', 'are', 'you', '?']
print(output.ids)     # [7592, 1010, 2129, 2024, 2017, 1029]

# 다시 디코딩
text = tokenizer.decode(output.ids)
print(text)  # "hello, how are you?"
```

### 사용자 지정 BPE 토크나이저 학습

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# BPE 모델로 토크나이저 초기화
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 트레이너 설정
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2
)

# 파일로 학습
files = ["train.txt", "validation.txt"]
tokenizer.train(files, trainer)

# 저장
tokenizer.save("my-tokenizer.json")
```

**학습 시간**: 100MB 코퍼스에서 약 1-2분, 1GB에서 약 10-20분

### 패딩 처리된 일괄(batch) 인코딩

```python
# 패딩 활성화
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

# 일괄 인코딩
texts = ["Hello world", "This is a longer sentence"]
encodings = tokenizer.encode_batch(texts)

for encoding in encodings:
    print(encoding.ids)
# [101, 7592, 2088, 102, 3, 3, 3]
# [101, 2023, 2003, 1037, 2936, 6251, 102]
```

## 토크나이징 알고리즘

### BPE (Byte-Pair Encoding)

**작동 방식**:
1. 문자 수준 어휘집으로 시작
2. 가장 빈번한 문자 쌍 찾기
3. 새 토큰으로 병합하고 어휘집에 추가
4. 목표 어휘집 크기에 도달할 때까지 반복

**사용되는 모델**: GPT-2, GPT-3, RoBERTa, BART, DeBERTa

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

**이점**:
- OOV(미등록 단어)를 잘 처리 (서브워드로 나눔)
- 어휘집 크기를 유연하게 설정
- 형태소가 발달한 언어에 적합

**절충점**:
- 토크나이징이 병합 순서에 종속됨
- 흔한 단어를 예상치 못하게 분리할 수도 있음

### WordPiece

**작동 방식**:
1. 문자 어휘집으로 시작
2. 병합 쌍 점수 매기기: `frequency(pair) / (frequency(first) × frequency(second))`
3. 최고 점수의 쌍을 병합
4. 목표 어휘집 크기에 도달할 때까지 반복

**사용되는 모델**: BERT, DistilBERT, MobileBERT

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

**이점**:
- 의미 있는 병합을 우선순위로 둠 (높은 점수 = 의미적으로 연관됨)
- BERT에서 성공적으로 사용됨 (SOTA 결과)

**절충점**:
- 서브워드 일치가 없는 알 수 없는 단어는 `[UNK]`가 됨
- 병합 규칙이 아닌 어휘집을 저장함 (더 큰 파일 크기)

### Unigram

**작동 방식**:
1. 큰 어휘집으로 시작 (모든 하위 문자열)
2. 현재 어휘집으로 코퍼스의 손실(loss) 계산
3. 손실에 최소한의 영향을 주는 토큰 제거
4. 목표 어휘집 크기에 도달할 때까지 반복

**사용되는 모델**: ALBERT, T5, mBART, XLNet (SentencePiece를 거쳐)

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

**이점**:
- 확률론적 모델 (가장 가능성 있는 토큰화 찾기)
- 띄어쓰기(단어 경계)가 없는 언어에 적합
- 다양한 언어적 문맥 처리

**절충점**:
- 학습 비용이 많이 듦
- 조정해야 할 하이퍼파라미터가 더 많음

## 토크나이징 파이프라인

완전한 파이프라인: **정규화(Normalization) → 사전 토크나이징(Pre-tokenization) → 모델 → 사후 처리(Post-processing)**

### 정규화 (Normalization)

텍스트 정리 및 표준화:

```python
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Sequence

tokenizer.normalizer = Sequence([
    NFD(),           # 유니코드 정규화 (분해)
    Lowercase(),     # 소문자 변환
    StripAccents()   # 강세(accents) 제거
])

# 입력: "Héllo WORLD"
# 정규화 후: "hello world"
```

**일반적인 정규화기**:
- `NFD`, `NFC`, `NFKD`, `NFKC` - 유니코드 정규화 형태
- `Lowercase()` - 소문자 변환
- `StripAccents()` - 강세 제거 (é → e)
- `Strip()` - 공백 제거
- `Replace(pattern, content)` - 정규표현식 변환

### 사전 토크나이징 (Pre-tokenization)

단어와 유사한 단위로 텍스트 분리:

```python
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence, ByteLevel

# 공백과 구두점을 기준으로 분할
tokenizer.pre_tokenizer = Sequence([
    Whitespace(),
    Punctuation()
])

# 입력: "Hello, world!"
# 사전 토크나이징 후: ["Hello", ",", "world", "!"]
```

**일반적인 사전 토크나이저**:
- `Whitespace()` - 스페이스, 탭, 줄바꿈으로 분할
- `ByteLevel()` - GPT-2 스타일의 바이트 단위 분할
- `Punctuation()` - 구두점을 독립적으로 분리
- `Digits(individual_digits=True)` - 숫자들을 개별 분할
- `Metaspace()` - SentencePiece 스타일로 스페이스를 ▁로 교체

### 사후 처리 (Post-processing)

모델 입력을 위한 특수 토큰 추가:

```python
from tokenizers.processors import TemplateProcessing

# BERT 스타일: [CLS] 문장 [SEP]
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
# GPT-2: 문장 <|endoftext|>
TemplateProcessing(
    single="$A <|endoftext|>",
    special_tokens=[("<|endoftext|>", 50256)]
)

# RoBERTa: <s> 문장 </s>
TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[("<s>", 0), ("</s>", 2)]
)
```

## 정렬(Alignment) 추적

원본 텍스트에서의 토큰 위치 추적:

```python
output = tokenizer.encode("Hello, world!")

# 토큰 오프셋 얻기
for token, offset in zip(output.tokens, output.offsets):
    start, end = offset
    print(f"{token:10} → [{start:2}, {end:2}): {text[start:end]!r}")

# 출력:
# hello      → [ 0,  5): 'Hello'
# ,          → [ 5,  6): ','
# world      → [ 7, 12): 'world'
# !          → [12, 13): '!'
```

**사용 사례**:
- 개체명 인식(NER) (예측을 다시 텍스트에 매핑)
- 질의응답(QA) (답변 구간 추출)
- 토큰 분류 (라벨을 원본 위치에 매핑)

## transformers 통합

### AutoTokenizer와 함께 로드

```python
from transformers import AutoTokenizer

# AutoTokenizer는 자동으로 fast tokenizer를 사용
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# fast tokenizer를 사용하고 있는지 확인
print(tokenizer.is_fast)  # True

# 내부적인 tokenizers.Tokenizer 접근
fast_tokenizer = tokenizer.backend_tokenizer
print(type(fast_tokenizer))  # <class 'tokenizers.Tokenizer'>
```

### 사용자 지정 토크나이저를 transformers용으로 변환

```python
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

# 사용자 지정 토크나이저 학습
tokenizer = Tokenizer(BPE())
# ... 토크나이저 학습 ...
tokenizer.save("my-tokenizer.json")

# transformers용 래퍼
transformers_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my-tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# 어떤 transformers 토크나이저처럼 사용 가능
outputs = transformers_tokenizer(
    "Hello world",
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

## 일반적인 패턴

### 이터레이터로부터 학습 (대규모 데이터셋)

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# 일괄 이터레이터 생성
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]

# 토크나이저 학습
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset)  # 진행 표시줄(progress bar)용
)
```

**성능**: 약 10-20분 이내에 1GB 처리

### 자르기(Truncation)와 패딩 활성화

```python
# 자르기 활성화
tokenizer.enable_truncation(max_length=512)

# 패딩 활성화
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]",
    length=512  # 고정 길이, 배치 최대값으로 할 경우 None
)

# 둘 다 적용된 인코딩
output = tokenizer.encode("This is a long sentence that will be truncated...")
print(len(output.ids))  # 512
```

### 멀티 프로세싱

```python
from tokenizers import Tokenizer
from multiprocessing import Pool

# 토크나이저 로드
tokenizer = Tokenizer.from_file("tokenizer.json")

def encode_batch(texts):
    return tokenizer.encode_batch(texts)

# 대규모 코퍼스를 병렬로 처리
with Pool(8) as pool:
    # 코퍼스를 청크로 분할
    chunk_size = 1000
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

    # 병렬 인코딩
    results = pool.map(encode_batch, chunks)
```

**속도 향상**: 8 코어 시 5-8배 향상

## 성능 벤치마크

### 학습 속도

| 코퍼스 크기 | BPE (30k vocab) | WordPiece (30k) | Unigram (8k) |
|-------------|-----------------|-----------------|--------------|
| 10 MB       | 15 초          | 18 초          | 25 초       |
| 100 MB      | 1.5 분         | 2 분           | 4 분        |
| 1 GB        | 15 분          | 20 분          | 40 분       |

**하드웨어**: 16코어 CPU, 영문 위키피디아로 테스트

### 토크나이징 속도

| 구현 | 1 GB 코퍼스 | 처리량 |
|----------------|-------------|---------------|
| 순수 Python | 약 20분 | 약 50 MB/분 |
| HF Tokenizers  | 약 15초 | 약 4 GB/분 |
| **속도 향상**  | **80배**     | **80배**       |

**테스트**: 평균 문장 길이 20단어의 영문 텍스트

### 메모리 사용량

| 작업 | 메모리 |
|-------------------------|---------|
| 토크나이저 로드 | 약 10 MB |
| BPE 학습 (30k 어휘) | 약 200 MB |
| 문장 100만 개 인코딩 | 약 500 MB |

## 지원되는 모델

`from_pretrained()`를 통해 사용 가능한 사전 학습된 토크나이저:

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

전체 찾아보기: https://huggingface.co/models?library=tokenizers

## 참조 항목

- **[학습 가이드](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/training.md)** - 사용자 지정 토크나이저 학습, 트레이너 구성, 대규모 데이터셋 처리
- **[알고리즘 상세 분석](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/algorithms.md)** - BPE, WordPiece, Unigram을 자세하게 설명
- **[파이프라인 컴포넌트](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/pipeline.md)** - 정규화기, 사전 토크나이저, 사후 처리기, 디코더
- **[Transformers 통합](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/huggingface-tokenizers/references/integration.md)** - AutoTokenizer, PreTrainedTokenizerFast, 특수 토큰

## 리소스

- **문서**: https://huggingface.co/docs/tokenizers
- **GitHub**: https://github.com/huggingface/tokenizers ⭐ 9,000+
- **버전**: 0.20.0+
- **강의 코스**: https://huggingface.co/learn/nlp-course/chapter6/1
- **논문**: BPE (Sennrich et al., 2016), WordPiece (Schuster & Nakajima, 2012)
