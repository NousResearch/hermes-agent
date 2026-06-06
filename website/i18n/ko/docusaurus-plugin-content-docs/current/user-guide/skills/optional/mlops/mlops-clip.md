---
title: "Clip — 시각과 언어를 연결하는 OpenAI의 모델"
sidebar_label: "Clip"
description: "시각과 언어를 연결하는 OpenAI의 모델"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Clip

시각과 언어를 연결하는 OpenAI의 모델. 제로샷(Zero-shot) 이미지 분류, 이미지-텍스트 매칭, 교차 모달(cross-modal) 검색을 가능하게 합니다. 4억 쌍의 이미지-텍스트 데이터로 학습되었습니다. 파인튜닝 없는 이미지 검색, 콘텐츠 필터링이나 시각-언어 태스크에서 사용하세요. 범용 이미지 이해에 적합합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/clip`로 설치 |
| 경로 | `optional-skills/mlops/clip` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `transformers`, `torch`, `pillow` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Multimodal`, `CLIP`, `Vision-Language`, `Zero-Shot`, `Image Classification`, `OpenAI`, `Image Search`, `Cross-Modal Retrieval`, `Content Moderation` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# CLIP - Contrastive Language-Image Pre-Training

자연어를 통해 이미지를 이해하는 OpenAI의 모델.

## CLIP을 사용하는 경우

**다음의 경우에 사용하세요:**
- 제로샷 이미지 분류 (학습 데이터가 필요 없음)
- 이미지-텍스트 유사도/매칭
- 의미 기반 이미지 검색
- 콘텐츠 모더레이션/필터링 (NSFW, 폭력물 탐지)
- 시각적 질의응답 (Visual question answering)
- 교차 모달(Cross-modal) 검색 (이미지→텍스트, 텍스트→이미지)

**지표(Metrics)**:
- **25,300+ GitHub stars**
- 4억 쌍의 이미지-텍스트로 학습
- ImageNet에서 ResNet-50과 맞먹음 (제로샷)
- MIT 라이선스

**다음을 대신 사용해 보세요**:
- **BLIP-2**: 더 나은 캡션 생성
- **LLaVA**: 비전-언어 대화형
- **Segment Anything**: 이미지 세그멘테이션

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

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 로드
image = preprocess(Image.open("photo.jpg")).unsqueeze(0).to(device)

# 가능한 라벨 정의
text = clip.tokenize(["a dog", "a cat", "a bird", "a car"]).to(device)

# 유사도 계산
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 코사인 유사도
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 결과 출력
labels = ["a dog", "a cat", "a bird", "a car"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.2%}")
```

## 사용 가능한 모델들

```python
# 모델 (크기순 정렬)
models = [
    "RN50",           # ResNet-50
    "RN101",          # ResNet-101
    "ViT-B/32",       # Vision Transformer (권장)
    "ViT-B/16",       # 더 높은 품질, 더 느림
    "ViT-L/14",       # 최고 품질, 제일 느림
]

model, preprocess = clip.load("ViT-B/32")
```

| 모델 | 파라미터 수 | 속도 | 품질 |
|-------|------------|-------|---------|
| RN50 | 102M | 빠름 | 좋음 |
| ViT-B/32 | 151M | 보통 | 더 좋음 |
| ViT-L/14 | 428M | 느림 | 최고 |

## 이미지-텍스트 유사도

```python
# 임베딩 계산
image_features = model.encode_image(image)
text_features = model.encode_text(text)

# 정규화
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 코사인 유사도
similarity = (image_features @ text_features.T).item()
print(f"Similarity: {similarity:.4f}")
```

## 의미 기반 이미지 검색

```python
# 이미지 색인
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
image_embeddings = []

for img_path in image_paths:
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    image_embeddings.append(embedding)

image_embeddings = torch.cat(image_embeddings)

# 텍스트 쿼리로 검색
query = "a sunset over the ocean"
text_input = clip.tokenize([query]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_input)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

# 가장 비슷한 이미지 찾기
similarities = (text_embedding @ image_embeddings.T).squeeze(0)
top_k = similarities.topk(3)

for idx, score in zip(top_k.indices, top_k.values):
    print(f"{image_paths[idx]}: {score:.3f}")
```

## 콘텐츠 모더레이션(Content moderation)

```python
# 카테고리 정의
categories = [
    "safe for work",
    "not safe for work",
    "violent content",
    "graphic content"
]

text = clip.tokenize(categories).to(device)

# 이미지 확인
with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1)

# 분류 결과 획득
max_idx = probs.argmax().item()
max_prob = probs[0, max_idx].item()

print(f"Category: {categories[max_idx]} ({max_prob:.2%})")
```

## 일괄(Batch) 처리

```python
# 여러 이미지 한 번에 처리
images = [preprocess(Image.open(f"img{i}.jpg")) for i in range(10)]
images = torch.stack(images).to(device)

with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# 텍스트 배치
texts = ["a dog", "a cat", "a bird"]
text_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 유사도 행렬 (이미지 10개 × 텍스트 3개)
similarities = image_features @ text_features.T
print(similarities.shape)  # (10, 3)
```

## 벡터 데이터베이스와의 연동

```python
# Chroma/FAISS에 CLIP 임베딩 저장
import chromadb

client = chromadb.Client()
collection = client.create_collection("image_embeddings")

# 이미지 임베딩 추가
for img_path, embedding in zip(image_paths, image_embeddings):
    collection.add(
        embeddings=[embedding.cpu().numpy().tolist()],
        metadatas=[{"path": img_path}],
        ids=[img_path]
    )

# 텍스트로 검색
query = "a sunset"
text_embedding = model.encode_text(clip.tokenize([query]))
results = collection.query(
    query_embeddings=[text_embedding.cpu().numpy().tolist()],
    n_results=5
)
```

## 모범 사례

1. **대부분의 경우 ViT-B/32 사용** - 좋은 균형
2. **임베딩 정규화** - 코사인 유사도를 위해 필수
3. **일괄 처리(Batch processing)** - 더 효율적임
4. **임베딩 캐싱** - 다시 계산하려면 비용이 큼
5. **설명적인 라벨 사용** - 더 나은 제로샷 성능 제공
6. **GPU 권장** - 10-50배 빠름
7. **이미지 전처리** - 제공된 preprocess 함수 사용

## 성능

| 작업 | CPU | GPU (V100) |
|-----------|-----|------------|
| 이미지 인코딩 | ~200ms | ~20ms |
| 텍스트 인코딩 | ~50ms | ~5ms |
| 유사도 계산 | &lt;1ms | &lt;1ms |

## 한계점

1. **세밀한(fine-grained) 태스크에는 부적합** - 광범위한 분류/카테고리에 적합
2. **설명적인 텍스트 필요** - 모호한 라벨은 성능이 저하됨
3. **웹 데이터에 대한 편향(Bias)** - 데이터셋 편향이 존재할 수 있음
4. **바운딩 박스(Bounding boxes) 없음** - 전체 이미지만 다룸
5. **제한적인 공간 이해도** - 위치 파악/개수 세기에 취약

## 리소스

- **GitHub**: https://github.com/openai/CLIP ⭐ 25,300+
- **논문**: https://arxiv.org/abs/2103.00020
- **Colab**: https://colab.research.google.com/github/openai/clip/
- **라이선스**: MIT
