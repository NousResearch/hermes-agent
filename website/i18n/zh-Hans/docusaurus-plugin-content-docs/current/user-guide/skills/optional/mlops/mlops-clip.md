---
title: "Clip — OpenAI 连接视觉和语言的模型"
sidebar_label: "Clip"
description: "OpenAI 连接视觉和语言的模型"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Clip

OpenAI 连接视觉和语言的模型。支持零样本图像分类、图文匹配和跨模态检索。在 4 亿个图文对上进行训练。适用于图像搜索、内容审核或无需微调的视觉语言任务。最适合通用图像理解。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/clip` |
| Path | `optional-skills/mlops/clip` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `transformers`, `torch`, `pillow` |
| Tags | `Multimodal`, `CLIP`, `Vision-Language`, `Zero-Shot`, `Image 分类`, `OpenAI`, `Image Search`, `Cross-Modal Retrieval`, `Content Moderation` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# CLIP - Contrastive Language-Image Pre-训练ing

OpenAI 的通过自然语言理解图像的模型。

## 何时使用 CLIP

**Use when:**
- Zero-shot image classification (no training data needed)
- 图文相似性/matching
- 语义图像搜索
- 内容审核 (detect NSFW, violence)
- 视觉问答
- Cross-modal retrieval (image→text, text→image)

**Metrics**:
- **25,300+ GitHub stars**
- 训练ed on 400M image-text pairs
- Matches ResNet-50 on ImageNet (zero-shot)
- MIT License

**替代方案**:
- **BLIP-2**: Better captioning
- **LLaVA**: Vision-language chat
- **Segment Anything**: Image segmentation

## 快速入门

### 安装

```bash
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision ftfy regex tqdm
```

### Zero-shot classification

```python
import torch
import clip
from PIL import Image

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图像
image = preprocess(Image.open("photo.jpg")).unsqueeze(0).to(device)

# 定义可能的标签
text = clip.tokenize(["a dog", "a cat", "a bird", "a car"]).to(device)

# 计算相似性
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Cosine similarity
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印结果
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
    "ViT-B/16",       # 质量更好，速度更慢
    "ViT-L/14",       # 质量最好，最慢
]

model, preprocess = clip.load("ViT-B/32")
```

| 模型 | 参数量 | 速度 | 质量 |
|-------|------------|-------|---------|
| RN50 | 102M | Fast | Good |
| ViT-B/32 | 151M | Medium | Better |
| ViT-L/14 | 428M | Slow | Best |

## 图文相似性

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

# 查询 with text
query = "a sunset"
text_embedding = model.encode_text(clip.tokenize([query]))
results = collection.query(
    query_embeddings=[text_embedding.cpu().numpy().tolist()],
    n_results=5
)
```

## Best practices

1. **Use ViT-B/32 for most cases** - 良好的平衡
2. **Normalize embeddings** - Required for cosine similarity
3. **批量处理** - More efficient
4. **Cache embeddings** - Expensive to recompute
5. **Use descriptive labels** - Better zero-shot performance
6. **推荐使用 GPU** - 10-50× faster
7. **Preprocess images** - Use provided preprocess function

## Performance

| Operation | CPU | GPU (V100) |
|-----------|-----|------------|
| Image encoding | ~200ms | ~20ms |
| Text encoding | ~50ms | ~5ms |
| Similarity compute | &lt;1ms | &lt;1ms |

## Limitations

1. **Not for fine-grained tasks** - 最适合 broad categories
2. **Requires descriptive text** - Vague labels perform poorly
3. **Biased on web data** - May have dataset biases
4. **No bounding boxes** - Whole image only
5. **Limited spatial understanding** - Position/counting weak

## Resources

- **GitHub**: https://github.com/openai/CLIP ⭐ 25,300+
- **Paper**: https://arxiv.org/abs/2103.00020
- **Colab**: https://colab.research.google.com/github/openai/clip/
- **License**: MIT
