---
title: "Clip — OpenAI 连接视觉与语言的模型"
sidebar_label: "Clip"
description: "OpenAI 连接视觉与语言的模型"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Clip

OpenAI 连接视觉与语言的模型。支持零样本图像分类、图文匹配和跨模态检索。在 4 亿图文对上训练而成。可用于图像搜索、内容审核或视觉语言任务，无需微调。最适合通用图像理解场景。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/clip` 安装 |
| 路径 | `optional-skills/mlops/clip` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `transformers`, `torch`, `pillow` |
| 平台 | linux, macos, windows |
| 标签 | `Multimodal`, `CLIP`, `Vision-Language`, `Zero-Shot`, `Image Classification`, `OpenAI`, `Image Search`, `Cross-Modal Retrieval`, `Content Moderation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# CLIP - 对比语言图像预训练（Contrastive Language-Image Pre-Training）

OpenAI 推出的能够通过自然语言理解图像的模型。

## 何时使用 CLIP

**适用场景：**
- 零样本图像分类（无需训练数据）
- 图文相似度/匹配
- 语义图像搜索
- 内容审核（检测 NSFW、暴力内容）
- 视觉问答
- 跨模态检索（图像→文本、文本→图像）

**指标**：
- **GitHub 25,300+ 星**
- 在 4 亿图文对上训练
- 零样本下在 ImageNet 上与 ResNet-50 持平
- MIT 许可证

**以下情况请使用替代方案**：
- **BLIP-2**：更好的图像描述生成
- **LLaVA**：视觉语言对话
- **Segment Anything**：图像分割

## 快速开始

### 安装

```bash
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision ftfy regex tqdm
```

### 零样本分类

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

## 可用模型

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

| 模型 | 参数量 | 速度 | 质量 |
|-------|------------|-------|---------|
| RN50 | 102M | 快 | 良好 |
| ViT-B/32 | 151M | 中等 | 更好 |
| ViT-L/14 | 428M | 慢 | 最佳 |

## 图文相似度

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

## 语义图像搜索

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

## 内容审核

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

## 批量处理

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

## 与向量数据库集成

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

## 最佳实践

1. **大多数场景使用 ViT-B/32** — 性能与速度均衡
2. **归一化 embedding（嵌入向量）** — 余弦相似度计算必须归一化
3. **批量处理** — 效率更高
4. **缓存 embedding** — 重新计算代价较高
5. **使用描述性标签** — 零样本性能更好
6. **推荐使用 GPU** — 速度提升 10–50 倍
7. **预处理图像** — 使用提供的 preprocess 函数

## 性能

| 操作 | CPU | GPU (V100) |
|-----------|-----|------------|
| 图像编码 | ~200ms | ~20ms |
| 文本编码 | ~50ms | ~5ms |
| 相似度计算 | &lt;1ms | &lt;1ms |

## 局限性

1. **不适合细粒度任务** — 最适合宽泛类别
2. **需要描述性文本** — 模糊标签效果差
3. **网络数据偏差** — 可能存在数据集偏差
4. **无边界框** — 仅处理整张图像
5. **空间理解有限** — 位置/计数能力较弱

## 资源

- **GitHub**: https://github.com/openai/CLIP ⭐ 25,300+
- **论文**: https://arxiv.org/abs/2103.00020
- **Colab**: https://colab.research.google.com/github/openai/clip/
- **许可证**: MIT