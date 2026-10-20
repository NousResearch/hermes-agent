---
title: "Clip — 제로샷 이미지 분류 및 이미지-텍스트 검색"
sidebar_label: "Clip"
description: "제로샷 이미지 분류 및 이미지-텍스트 검색"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Clip

제로샷 이미지 분류 및 이미지-텍스트 검색.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/clip`으로 설치 |
| 경로 | `optional-skills/mlops/clip` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `transformers`, `torch`, `pillow` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Multimodal`, `CLIP`, `Vision-Language`, `Zero-Shot`, `Image Classification`, `OpenAI`, `Image Search`, `Cross-Modal Retrieval`, `Content Moderation` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 활성화될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침입니다.
:::

# CLIP - Contrastive Language-Image Pre-Training

자연어로 이미지를 이해하는 OpenAI의 모델입니다.

## CLIP 사용 시점

**다음과 같은 경우에 사용하세요:**
- 제로샷 이미지 분류(학습 데이터가 필요 없음)
- 이미지-텍스트 유사도/매칭
- 의미 기반 이미지 검색
- 콘텐츠 조정(NSFW, 폭력 감지)
- 시각적 질의응답
- 크로스모달 검색(이미지→텍스트, 텍스트→이미지)

**지표**:
- **GitHub 스타 25,300개 이상**
- 4억 개의 이미지-텍스트 쌍으로 학습
- ImageNet에서 ResNet-50과 일치(제로샷)
- MIT 라이선스

**대신 다음 대안을 사용하세요**:
- **BLIP-2**: 더 나은 캡션 생성
- **LLaVA**: 비전-언어 채팅
- **Segment Anything**: 이미지 분할

## 빠른 시작

### 설치

```bash
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision ftfy regex tqdm
```

### 제로샷 분류

```python
import torch
import clip
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load image
image = preprocess(Image.open("photo.jpg")).unsqueeze(0).to(device)

# Define possible labels
text = clip.tokenize(["a dog", "a cat", "a bird", "a car"]).to(device)

# Compute similarity
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Cosine similarity
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Print results
labels = ["a dog", "a cat", "a bird", "a car"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.2%}")
```

## 사용 가능한 모델

```python
# Models (sorted by size)
models = [
    "RN50",           # ResNet-50
    "RN101",          # ResNet-101
    "ViT-B/32",       # Vision Transformer (recommended)
    "ViT-B/16",       # Better quality, slower
    "ViT-L/14",       # Best quality, slowest
]

model, preprocess = clip.load("ViT-B/32")
```

| 모델 | 파라미터 | 속도 | 품질 |
|-------|------------|---------|---------|
| RN50 | 102M | 빠름 | 좋음 |
| ViT-B/32 | 151M | 보통 | 더 좋음 |
| ViT-L/14 | 428M | 느림 | 최고 |

## 이미지-텍스트 유사도

```python
# Compute embeddings
image_features = model.encode_image(image)
text_features = model.encode_text(text)

# Normalize
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Cosine similarity
similarity = (image_features @ text_features.T).item()
print(f"Similarity: {similarity:.4f}")
```

## 의미 기반 이미지 검색

```python
# Index images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
image_embeddings = []

for img_path in image_paths:
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    image_embeddings.append(embedding)

image_embeddings = torch.cat(image_embeddings)

# Search with text query
query = "a sunset over the ocean"
text_input = clip.tokenize([query]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_input)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

# Find most similar images
similarities = (text_embedding @ image_embeddings.T).squeeze(0)
top_k = similarities.topk(3)

for idx, score in zip(top_k.indices, top_k.values):
    print(f"{image_paths[idx]}: {score:.3f}")
```

## 콘텐츠 조정

```python
# Define categories
categories = [
    "safe for work",
    "not safe for work",
    "violent content",
    "graphic content"
]

text = clip.tokenize(categories).to(device)

# Check image
with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1)

# Get classification
max_idx = probs.argmax().item()
max_prob = probs[0, max_idx].item()

print(f"Category: {categories[max_idx]} ({max_prob:.2%})")
```

## 일괄 처리

```python
# Process multiple images
images = [preprocess(Image.open(f"img{i}.jpg")) for i in range(10)]
images = torch.stack(images).to(device)

with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Batch text
texts = ["a dog", "a cat", "a bird"]
text_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Similarity matrix (10 images × 3 texts)
similarities = image_features @ text_features.T
print(similarities.shape)  # (10, 3)
```

## 벡터 데이터베이스와 통합

```python
# Store CLIP embeddings in Chroma/FAISS
import chromadb

client = chromadb.Client()
collection = client.create_collection("image_embeddings")

# Add image embeddings
for img_path, embedding in zip(image_paths, image_embeddings):
    collection.add(
        embeddings=[embedding.cpu().numpy().tolist()],
        metadatas=[{"path": img_path}],
        ids=[img_path]
    )

# Query with text
query = "a sunset"
text_embedding = model.encode_text(clip.tokenize([query]))
results = collection.query(
    query_embeddings=[text_embedding.cpu().numpy().tolist()],
    n_results=5
)
```

## 모범 사례

1. **대부분의 경우 ViT-B/32 사용** - 균형이 좋음
2. **임베딩 정규화** - 코사인 유사도에 필요
3. **일괄 처리** - 더 효율적임
4. **임베딩 캐시** - 다시 계산하는 비용이 큼
5. **설명적인 레이블 사용** - 제로샷 성능 향상
6. **GPU 권장** - 10~50배 빠름
7. **이미지 전처리** - 제공된 preprocess 함수 사용

## 성능

| 작업 | CPU | GPU (V100) |
|-----------|-----|------------|
| 이미지 인코딩 | ~200ms | ~20ms |
| 텍스트 인코딩 | ~50ms | ~5ms |
| 유사도 계산 | &lt;1ms | &lt;1ms |

## 제한 사항

1. **세밀한 작업에는 적합하지 않음** - 광범위한 범주에 가장 적합
2. **설명적인 텍스트 필요** - 모호한 레이블은 성능이 낮음
3. **웹 데이터에 따른 편향** - 데이터셋 편향이 있을 수 있음
4. **바운딩 박스 없음** - 전체 이미지 단위로만 처리
5. **제한적인 공간 이해** - 위치/개수 파악이 약함

## 리소스

- **GitHub**: https://github.com/openai/CLIP ⭐ 25,300+
- **논문**: https://arxiv.org/abs/2103.00020
- **Colab**: https://colab.research.google.com/github/openai/clip/
- **라이선스**: MIT
