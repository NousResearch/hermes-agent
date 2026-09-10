---
title: "Segment Anything Model — SAM: 포인트, 상자, 마스크를 통한 제로샷 이미지 분할"
sidebar_label: "Segment Anything Model"
description: "SAM: 포인트, 상자, 마스크를 통한 제로샷 이미지 분할"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Segment Anything Model

SAM: 포인트, 상자, 마스크를 통한 제로샷 이미지 분할.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/segment-anything-model`로 설치 |
| 경로 | `optional-skills/mlops/models/segment-anything-model` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `segment-anything`, `transformers>=4.30.0`, `torch>=1.7.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Multimodal`, `Image Segmentation`, `Computer Vision`, `SAM`, `Zero-Shot` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 확인하는 내용입니다.
:::

# Segment Anything Model (SAM)

Meta AI의 Segment Anything Model을 사용한 제로샷 이미지 분할 가이드입니다.

## SAM을 사용해야 하는 경우

**다음과 같은 경우 SAM을 사용하세요:**
- 작업별 학습 없이 이미지의 모든 객체를 분할해야 하는 경우
- 포인트/상자 프롬프트를 사용하는 대화형 어노테이션 도구를 구축하는 경우
- 다른 비전 모델을 위한 학습 데이터를 생성하는 경우
- 새로운 이미지 도메인으로 제로샷 전이가 필요한 경우
- 객체 탐지/분할 파이프라인을 구축하는 경우
- 의료, 위성 또는 도메인별 이미지를 처리하는 경우

**주요 기능:**
- **제로샷 분할**: 파인튜닝 없이 모든 이미지 도메인에서 작동
- **유연한 프롬프트**: 포인트, 경계 상자 또는 이전 마스크
- **자동 분할**: 모든 객체 마스크를 자동으로 생성
- **높은 품질**: 1,100만 개 이미지의 11억 개 마스크로 학습
- **다양한 모델 크기**: ViT-B(가장 빠름), ViT-L, ViT-H(가장 정확함)
- **ONNX 내보내기**: 브라우저와 엣지 디바이스에 배포

**다음 대안을 대신 사용하세요:**
- **YOLO/Detectron2**: 클래스를 사용하는 실시간 객체 탐지
- **Mask2Former**: 카테고리가 있는 의미론적/파노픽 분할
- **GroundingDINO + SAM**: 텍스트 프롬프트 기반 분할
- **SAM 2**: 동영상 분할 작업

## 빠른 시작

### 설치

```bash
# From GitHub
pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional dependencies
pip install opencv-python pycocotools matplotlib

# Or use HuggingFace transformers
pip install transformers
```

### 체크포인트 다운로드

```bash
# ViT-H (largest, most accurate) - 2.4GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (medium) - 1.2GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (smallest, fastest) - 375MB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### SamPredictor를 사용한 기본 사용법

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Load model
sam = sam_model_registry["vit_h"](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/models/segment-anything-model/checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")

# Create predictor
predictor = SamPredictor(sam)

# Set image (computes embeddings once)
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# Predict with point prompts
input_point = np.array([[500, 375]])  # (x, y) coordinates
input_label = np.array([1])  # 1 = foreground, 0 = background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Returns 3 mask options
)

# Select best mask
best_mask = masks[np.argmax(scores)]
```

### HuggingFace Transformers

```python
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# Load model and processor
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model.to("cuda")

# Process image with point prompt
image = Image.open("image.jpg")
input_points = [[[450, 600]]]  # Batch of points

inputs = processor(image, input_points=input_points, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate masks
with torch.no_grad():
    outputs = model(**inputs)

# Post-process masks to original size
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)
```

## 핵심 개념

### 모델 아키텍처

<!-- ascii-guard-ignore -->
<!-- ascii-guard-ignore -->
```
SAM Architecture:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Image Encoder  │────▶│ Prompt Encoder  │────▶│  Mask Decoder   │
│     (ViT)       │     │ (Points/Boxes)  │     │ (Transformer)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   Image Embeddings      Prompt Embeddings         Masks + IoU
   (computed once)       (per prompt)             predictions
```
<!-- ascii-guard-ignore-end -->
<!-- ascii-guard-ignore-end -->

### 모델 변형

| 모델 | 체크포인트 | 크기 | 속도 | 정확도 |
|-------|------------|------|-------|----------|
| ViT-H | `vit_h` | 2.4 GB | 가장 느림 | 최고 |
| ViT-L | `vit_l` | 1.2 GB | 중간 | 좋음 |
| ViT-B | `vit_b` | 375 MB | 가장 빠름 | 좋음 |

### 프롬프트 유형

| 프롬프트 | 설명 | 사용 사례 |
|--------|-------------|----------|
| 포인트(전경) | 객체를 클릭 | 단일 객체 선택 |
| 포인트(배경) | 객체 바깥을 클릭 | 영역 제외 |
| 경계 상자 | 객체 주위의 직사각형 | 더 큰 객체 |
| 이전 마스크 | 저해상도 마스크 입력 | 반복적 정제 |

## 대화형 분할

### 포인트 프롬프트

```python
# Single foreground point
input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# Multiple points (foreground + background)
input_points = np.array([[500, 375], [600, 400], [450, 300]])
input_labels = np.array([1, 1, 0])  # 2 foreground, 1 background

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False  # Single mask when prompts are clear
)
```

### 상자 프롬프트

```python
# Bounding box [x1, y1, x2, y2]
input_box = np.array([425, 600, 700, 875])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)
```

### 결합 프롬프트

```python
# Box + points for precise control
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    box=np.array([400, 300, 700, 600]),
    multimask_output=False
)
```

### 반복적 정제

```python
# Initial prediction
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True
)

# Refine with additional point using previous mask
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375], [550, 400]]),
    point_labels=np.array([1, 0]),  # Add background point
    mask_input=logits[np.argmax(scores)][None, :, :],  # Use best mask
    multimask_output=False
)
```

## 자동 마스크 생성

### 기본 자동 분할

```python
from segment_anything import SamAutomaticMaskGenerator

# Create generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate all masks
masks = mask_generator.generate(image)

# Each mask contains:
# - segmentation: binary mask
# - bbox: [x, y, w, h]
# - area: pixel count
# - predicted_iou: quality score
# - stability_score: robustness score
# - point_coords: generating point
```

### 사용자 지정 생성

```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,          # Grid density (more = more masks)
    pred_iou_thresh=0.88,        # Quality threshold
    stability_score_thresh=0.95,  # Stability threshold
    crop_n_layers=1,             # Multi-scale crops
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,    # Remove tiny masks
)

masks = mask_generator.generate(image)
```

### 마스크 필터링

```python
# Sort by area (largest first)
masks = sorted(masks, key=lambda x: x['area'], reverse=True)

# Filter by predicted IoU
high_quality = [m for m in masks if m['predicted_iou'] > 0.9]

# Filter by stability score
stable_masks = [m for m in masks if m['stability_score'] > 0.95]
```

## 배치 추론

### 여러 이미지

```python
# Process multiple images efficiently
images = [cv2.imread(f"image_{i}.jpg") for i in range(10)]

all_masks = []
for image in images:
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=np.array([[500, 375]]),
        point_labels=np.array([1]),
        multimask_output=True
    )
    all_masks.append(masks)
```

### 이미지당 여러 프롬프트

```python
# Process multiple prompts efficiently (one image encoding)
predictor.set_image(image)

# Batch of point prompts
points = [
    np.array([[100, 100]]),
    np.array([[200, 200]]),
    np.array([[300, 300]])
]

all_masks = []
for point in points:
    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=np.array([1]),
        multimask_output=True
    )
    all_masks.append(masks[np.argmax(scores)])
```

## ONNX 배포

### 모델 내보내기

```bash
python scripts/export_onnx_model.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --output sam_onnx.onnx \
    --return-single-mask
```

### ONNX 모델 사용

```python
import onnxruntime

# Load ONNX model
ort_session = onnxruntime.InferenceSession("sam_onnx.onnx")

# Run inference (image embeddings computed separately)
masks = ort_session.run(
    None,
    {
        "image_embeddings": image_embeddings,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
        "orig_im_size": np.array([h, w], dtype=np.float32)
    }
)
```

## 일반적인 워크플로

### 워크플로 1: 어노테이션 도구

```python
import cv2

# Load model
predictor = SamPredictor(sam)
predictor.set_image(image)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Foreground point
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        # Display best mask
        display_mask(masks[np.argmax(scores)])
```

### 워크플로 2: 객체 추출

```python
def extract_object(image, point):
    """Extract object at point with transparent background."""
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    # Create RGBA output
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = best_mask * 255

    return rgba
```

### 워크플로 3: 의료 이미지 분할

```python
# Process medical images (grayscale to RGB)
medical_image = cv2.imread("scan.png", cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.cvtColor(medical_image, cv2.COLOR_GRAY2RGB)

predictor.set_image(rgb_image)

# Segment region of interest
masks, scores, _ = predictor.predict(
    box=np.array([x1, y1, x2, y2]),  # ROI bounding box
    multimask_output=True
)
```

## 출력 형식

### 마스크 데이터 구조

```python
# SamAutomaticMaskGenerator output
{
    "segmentation": np.ndarray,  # H×W binary mask
    "bbox": [x, y, w, h],        # Bounding box
    "area": int,                 # Pixel count
    "predicted_iou": float,      # 0-1 quality score
    "stability_score": float,    # 0-1 robustness score
    "crop_box": [x, y, w, h],    # Generation crop region
    "point_coords": [[x, y]],    # Input point
}
```

### COCO RLE 형식

```python
from pycocotools import mask as mask_utils

# Encode mask to RLE
rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
rle["counts"] = rle["counts"].decode("utf-8")

# Decode RLE to mask
decoded_mask = mask_utils.decode(rle)
```

## 성능 최적화

### GPU 메모리

```python
# Use smaller model for limited VRAM
sam = sam_model_registry["vit_b"](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/models/segment-anything-model/checkpoint="sam_vit_b_01ec64.pth")

# Process images in batches
# Clear CUDA cache between large batches
torch.cuda.empty_cache()
```

### 속도 최적화

```python
# Use half precision
sam = sam.half()

# Reduce points for automatic generation
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # Default is 32
)

# Use ONNX for deployment
# Export with --return-single-mask for faster inference
```

## 일반적인 문제

| 문제 | 해결 방법 |
|-------|----------|
| 메모리 부족 | ViT-B 모델 사용, 이미지 크기 축소 |
| 느린 추론 | ViT-B 사용, points_per_side 축소 |
| 낮은 마스크 품질 | 다른 프롬프트 시도, 상자 + 포인트 사용 |
| 가장자리 아티팩트 | stability_score 필터링 사용 |
| 작은 객체 누락 | points_per_side 증가 |

## 참고 자료

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/models/segment-anything-model/references/advanced-usage.md)** - 배칭, 파인튜닝, 통합
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/models/segment-anything-model/references/troubleshooting.md)** - 일반적인 문제와 해결 방법

## 리소스

- **GitHub**: https://github.com/facebookresearch/segment-anything
- **논문**: https://arxiv.org/abs/2304.02643
- **데모**: https://segment-anything.com
- **SAM 2(동영상)**: https://github.com/facebookresearch/segment-anything-2
- **HuggingFace**: https://huggingface.co/facebook/sam-vit-huge
