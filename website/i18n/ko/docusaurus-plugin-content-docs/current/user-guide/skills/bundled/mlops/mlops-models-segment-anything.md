---
title: "Segment Anything Model — SAM: 점, 박스, 마스크를 통한 제로샷(Zero-shot) 이미지 분할"
sidebar_label: "Segment Anything Model"
description: "SAM: 점, 박스, 마스크를 통한 제로샷(Zero-shot) 이미지 분할"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Segment Anything Model

SAM: 점, 박스, 마스크를 통한 제로샷(Zero-shot) 이미지 분할.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/models/segment-anything` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `segment-anything`, `transformers>=4.30.0`, `torch>=1.7.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Multimodal`, `Image Segmentation`, `Computer Vision`, `SAM`, `Zero-Shot` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Segment Anything Model (SAM)

제로샷(Zero-shot) 이미지 분할을 위한 Meta AI의 Segment Anything Model 사용에 대한 포괄적인 가이드입니다.

## SAM 사용 시기

**다음과 같은 경우에 SAM을 사용하세요:**
- 작업 특화(task-specific) 학습 없이 이미지 내 어떤 객체든 분할해야 할 때
- 점/박스 프롬프트를 이용한 대화형 어노테이션 도구를 구축할 때
- 다른 비전 모델을 위한 학습 데이터를 생성할 때
- 새로운 이미지 도메인으로의 제로샷 전이(transfer)가 필요할 때
- 객체 탐지/분할 파이프라인을 구축할 때
- 의료, 위성 또는 도메인 특화 이미지를 처리할 때

**주요 특징:**
- **제로샷 분할(Zero-shot segmentation)**: 미세 조정(fine-tuning) 없이 어떤 이미지 도메인에서도 작동
- **유연한 프롬프트**: 점, 경계 상자(bounding boxes) 또는 이전 마스크 활용
- **자동 분할**: 모든 객체 마스크를 자동으로 생성
- **고품질**: 1,100만 장의 이미지에서 추출한 11억 개의 마스크로 학습됨
- **다양한 모델 크기**: ViT-B (가장 빠름), ViT-L, ViT-H (가장 정확함)
- **ONNX 내보내기**: 브라우저 및 엣지(edge) 디바이스 배포 지원

**대신 다음 대안을 사용하는 것이 좋은 경우:**
- **YOLO/Detectron2**: 클래스(class)를 분류하는 실시간 객체 탐지의 경우
- **Mask2Former**: 카테고리를 포함한 시맨틱(semantic)/팬옵틱(panoptic) 분할의 경우
- **GroundingDINO + SAM**: 텍스트 프롬프트 기반 분할의 경우
- **SAM 2**: 비디오 분할 작업의 경우

## 빠른 시작

### 설치

```bash
# GitHub에서 설치
pip install git+https://github.com/facebookresearch/segment-anything.git

# 선택적 의존성 설치
pip install opencv-python pycocotools matplotlib

# 또는 HuggingFace transformers 사용
pip install transformers
```

### 체크포인트 다운로드

```bash
# ViT-H (가장 큼, 가장 정확함) - 2.4GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (중간) - 1.2GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (가장 작음, 가장 빠름) - 375MB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### SamPredictor 기본 사용법

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# 모델 로드
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")

# Predictor 생성
predictor = SamPredictor(sam)

# 이미지 설정 (임베딩을 한 번 계산함)
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# 점 프롬프트로 예측
input_point = np.array([[500, 375]])  # (x, y) 좌표
input_label = np.array([1])  # 1 = 전경(foreground), 0 = 배경(background)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # 3개의 마스크 옵션 반환
)

# 가장 좋은 마스크 선택
best_mask = masks[np.argmax(scores)]
```

### HuggingFace Transformers

```python
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# 모델과 프로세서 로드
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model.to("cuda")

# 점 프롬프트와 함께 이미지 처리
image = Image.open("image.jpg")
input_points = [[[450, 600]]]  # 점 배치(Batch)

inputs = processor(image, input_points=input_points, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 마스크 생성
with torch.no_grad():
    outputs = model(**inputs)

# 원래 크기로 마스크 후처리
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
SAM 아키텍처:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Image Encoder  │────▶│ Prompt Encoder  │────▶│  Mask Decoder   │
│     (ViT)       │     │ (Points/Boxes)  │     │ (Transformer)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   Image Embeddings      Prompt Embeddings         Masks + IoU
   (계산 1회)            (프롬프트마다)           예측
```
<!-- ascii-guard-ignore-end -->
<!-- ascii-guard-ignore-end -->

### 모델 변형(Variants)

| 모델 | 체크포인트 | 크기 | 속도 | 정확도 |
|-------|------------|------|-------|----------|
| ViT-H | `vit_h` | 2.4 GB | 가장 느림 | 최고 |
| ViT-L | `vit_l` | 1.2 GB | 보통 | 좋음 |
| ViT-B | `vit_b` | 375 MB | 가장 빠름 | 좋음 |

### 프롬프트 유형

| 프롬프트 | 설명 | 사용 사례 |
|--------|-------------|----------|
| 점 (전경) | 객체 위 클릭 | 단일 객체 선택 |
| 점 (배경) | 객체 밖 클릭 | 특정 영역 제외 |
| 경계 상자 | 객체 주변의 사각형 | 큰 객체 |
| 이전 마스크 | 저해상도 마스크 입력 | 반복적인 정제 |

## 대화형 분할(Interactive segmentation)

### 점 프롬프트

```python
# 단일 전경 점
input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# 다중 점 (전경 + 배경)
input_points = np.array([[500, 375], [600, 400], [450, 300]])
input_labels = np.array([1, 1, 0])  # 2 전경, 1 배경

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False  # 프롬프트가 명확할 때 단일 마스크 반환
)
```

### 박스 프롬프트

```python
# 경계 상자 [x1, y1, x2, y2]
input_box = np.array([425, 600, 700, 875])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)
```

### 프롬프트 결합

```python
# 정밀한 제어를 위한 박스 + 점 결합
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    box=np.array([400, 300, 700, 600]),
    multimask_output=False
)
```

### 반복적 정제(Iterative refinement)

```python
# 초기 예측
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True
)

# 이전 마스크를 사용하여 추가적인 점으로 정제
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375], [550, 400]]),
    point_labels=np.array([1, 0]),  # 배경 점 추가
    mask_input=logits[np.argmax(scores)][None, :, :],  # 가장 좋은 마스크 사용
    multimask_output=False
)
```

## 자동 마스크 생성

### 기본 자동 분할

```python
from segment_anything import SamAutomaticMaskGenerator

# 생성기 만들기
mask_generator = SamAutomaticMaskGenerator(sam)

# 모든 마스크 생성
masks = mask_generator.generate(image)

# 각 마스크의 포함 요소:
# - segmentation: 이진 마스크
# - bbox: [x, y, w, h]
# - area: 픽셀 개수
# - predicted_iou: 품질 점수
# - stability_score: 견고성 점수
# - point_coords: 마스크를 생성한 점
```

### 커스텀 생성

```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,          # 그리드 밀도 (클수록 더 많은 마스크)
    pred_iou_thresh=0.88,        # 품질 임계값
    stability_score_thresh=0.95,  # 안정성 임계값
    crop_n_layers=1,             # 멀티스케일 크롭
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,    # 아주 작은 마스크 제거
)

masks = mask_generator.generate(image)
```

### 마스크 필터링

```python
# 면적순으로 정렬 (가장 큰 것부터)
masks = sorted(masks, key=lambda x: x['area'], reverse=True)

# 예측된 IoU로 필터링
high_quality = [m for m in masks if m['predicted_iou'] > 0.9]

# 안정성 점수로 필터링
stable_masks = [m for m in masks if m['stability_score'] > 0.95]
```

## 배치 추론(Batched inference)

### 여러 이미지

```python
# 여러 이미지를 효율적으로 처리
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
# 여러 프롬프트를 효율적으로 처리 (이미지 인코딩은 1회)
predictor.set_image(image)

# 점 프롬프트의 배치
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

# ONNX 모델 로드
ort_session = onnxruntime.InferenceSession("sam_onnx.onnx")

# 추론 실행 (이미지 임베딩은 별도로 계산됨)
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

## 일반적인 워크플로우

### 워크플로우 1: 어노테이션 도구

```python
import cv2

# 모델 로드
predictor = SamPredictor(sam)
predictor.set_image(image)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 전경 점
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        # 가장 좋은 마스크 표시
        display_mask(masks[np.argmax(scores)])
```

### 워크플로우 2: 객체 추출

```python
def extract_object(image, point):
    """해당 점의 객체를 투명한 배경과 함께 추출합니다."""
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    # RGBA 출력 생성
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = best_mask * 255

    return rgba
```

### 워크플로우 3: 의료 이미지 분할

```python
# 의료 이미지 처리 (그레이스케일을 RGB로)
medical_image = cv2.imread("scan.png", cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.cvtColor(medical_image, cv2.COLOR_GRAY2RGB)

predictor.set_image(rgb_image)

# 관심 영역(ROI) 분할
masks, scores, _ = predictor.predict(
    box=np.array([x1, y1, x2, y2]),  # ROI 경계 상자
    multimask_output=True
)
```

## 출력 형식

### 마스크 데이터 구조

```python
# SamAutomaticMaskGenerator 출력
{
    "segmentation": np.ndarray,  # H×W 이진 마스크
    "bbox": [x, y, w, h],        # 경계 상자
    "area": int,                 # 픽셀 개수
    "predicted_iou": float,      # 0-1 품질 점수
    "stability_score": float,    # 0-1 견고성 점수
    "crop_box": [x, y, w, h],    # 생성 자르기 영역
    "point_coords": [[x, y]],    # 입력 점
}
```

### COCO RLE 형식

```python
from pycocotools import mask as mask_utils

# 마스크를 RLE로 인코딩
rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
rle["counts"] = rle["counts"].decode("utf-8")

# RLE를 마스크로 디코딩
decoded_mask = mask_utils.decode(rle)
```

## 성능 최적화

### GPU 메모리

```python
# VRAM이 제한적인 경우 작은 모델 사용
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

# 배치로 이미지 처리
# 큰 배치 사이에 CUDA 캐시 지우기
torch.cuda.empty_cache()
```

### 속도 최적화

```python
# 반정밀도(Half precision) 사용
sam = sam.half()

# 자동 생성 시 점 개수 줄이기
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # 기본값은 32
)

# 배포 시 ONNX 사용
# 더 빠른 추론을 위해 --return-single-mask 로 내보내기
```

## 일반적인 문제

| 문제 | 해결책 |
|-------|----------|
| 메모리 부족 (OOM) | ViT-B 모델 사용, 이미지 크기 줄이기 |
| 느린 추론 속도 | ViT-B 사용, points_per_side 줄이기 |
| 마스크 품질이 나쁨 | 다른 프롬프트 시도, 박스 + 점 결합 사용 |
| 가장자리 아티팩트(Artifacts) | stability_score 필터링 사용 |
| 작은 객체 누락 | points_per_side 늘리기 |

## 참조

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/segment-anything/references/advanced-usage.md)** - 배치 처리, 미세 조정, 통합
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/segment-anything/references/troubleshooting.md)** - 일반적인 문제 및 해결책

## 리소스

- **GitHub**: https://github.com/facebookresearch/segment-anything
- **논문**: https://arxiv.org/abs/2304.02643
- **데모**: https://segment-anything.com
- **SAM 2 (Video)**: https://github.com/facebookresearch/segment-anything-2
- **HuggingFace**: https://huggingface.co/facebook/sam-vit-huge
