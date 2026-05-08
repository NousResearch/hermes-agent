---
title: "Segment Anything Model — SAM：通过点、框、掩码实现零样本图像分割"
sidebar_label: "Segment Anything Model"
description: "SAM：通过点、框、掩码实现零样本图像分割"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Segment Anything Model

SAM：通过点、框、掩码实现零样本图像分割。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/models/segment-anything` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `segment-anything`, `transformers>=4.30.0`, `torch>=1.7.0` |
| 标签 | `多模态`, `图像分割`, `计算机视觉`, `SAM`, `零样本` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Segment Anything Model (SAM)

使用 Meta AI 的 Segment Anything Model 进行零样本图像分割的完整指南。

## 何时使用 SAM

**在以下情况下使用 SAM：**
- 需要在图像中分割任意对象而无需特定任务训练
- 构建带有点/框提示的交互式标注工具
- 为其他视觉模型生成训练数据
- 需要零样本迁移到新的图像领域
- 构建对象检测/分割流水线
- 处理医学、卫星或特定领域的图像

**核心特性：**
- **零样本分割**：无需微调即可在任何图像领域上工作
- **灵活提示**：点、边界框或之前的掩码
- **自动分割**：自动生成所有对象掩码
- **高质量**：在 1100 万张图像上的 11 亿个掩码上训练
- **多种模型规格**：ViT-B（最快）、ViT-L、ViT-H（最准确）
- **ONNX 导出**：在浏览器和边缘设备上部署

**建议使用替代方案：**
- **YOLO/Detectron2**：用于带类别的实时目标检测
- **Mask2Former**：用于带类别的语义/全景分割
- **GroundingDINO + SAM**：用于文本提示的分割
- **SAM 2**：用于视频分割任务

## 快速开始

### 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/facebookresearch/segment-anything.git

# 可选依赖
pip install opencv-python pycocotools matplotlib

# 或使用 HuggingFace transformers
pip install transformers
```

### 下载检查点

```bash
# ViT-H（最大，最准确）- 2.4GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L（中等）- 1.2GB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B（最小，最快）- 375MB
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 使用 SamPredictor 的基本用法

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# 加载模型
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")

# 创建预测器
predictor = SamPredictor(sam)

# 设置图像（仅计算一次嵌入）
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# 使用点提示进行预测
input_point = np.array([[500, 375]])  # (x, y) 坐标
input_label = np.array([1])  # 1 = 前景, 0 = 背景

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # 返回 3 个掩码选项
)

# 选择最佳掩码
best_mask = masks[np.argmax(scores)]
```

### HuggingFace Transformers

```python
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

# 加载模型和处理器
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model.to("cuda")

# 使用点提示处理图像
image = Image.open("image.jpg")
input_points = [[[450, 600]]]  # 点的批次

inputs = processor(image, input_points=input_points, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 生成掩码
with torch.no_grad():
    outputs = model(**inputs)

# 将掩码后处理为原始大小
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)
```

## 核心概念

### 模型架构

<!-- ascii-guard-ignore -->
<!-- ascii-guard-ignore -->
```
SAM 架构：
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   图像编码器    │────▶│  提示编码器     │────▶│   掩码解码器    │
│     (ViT)       │     │  (点/框)        │     │  (Transformer)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   图像嵌入              提示嵌入                掩码 + IoU
   (计算一次)            (每次提示)              预测
```
<!-- ascii-guard-ignore-end -->
<!-- ascii-guard-ignore-end -->

### 模型变体

| 模型 | 检查点 | 大小 | 速度 | 准确度 |
|-------|------------|------|-------|----------|
| ViT-H | `vit_h` | 2.4 GB | 最慢 | 最佳 |
| ViT-L | `vit_l` | 1.2 GB | 中等 | 良好 |
| ViT-B | `vit_b` | 375 MB | 最快 | 良好 |

### 提示类型

| 提示 | 描述 | 使用场景 |
|--------|-------------|----------|
| 点（前景） | 点击对象 | 单个对象选择 |
| 点（背景） | 点击对象外部 | 排除区域 |
| 边界框 | 对象周围的矩形 | 较大对象 |
| 之前的掩码 | 低分辨率掩码输入 | 迭代精炼 |

## 交互式分割

### 点提示

```python
# 单个前景点
input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# 多个点（前景 + 背景）
input_points = np.array([[500, 375], [600, 400], [450, 300]])
input_labels = np.array([1, 1, 0])  # 2 个前景, 1 个背景

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False  # 提示清晰时返回单个掩码
)
```

### 框提示

```python
# 边界框 [x1, y1, x2, y2]
input_box = np.array([425, 600, 700, 875])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)
```

### 组合提示

```python
# 框 + 点用于精确控制
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    box=np.array([400, 300, 700, 600]),
    multimask_output=False
)
```

### 迭代精炼

```python
# 初始预测
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True
)

# 使用之前的掩码通过额外的点进行精炼
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375], [550, 400]]),
    point_labels=np.array([1, 0]),  # 添加背景点
    mask_input=logits[np.argmax(scores)][None, :, :],  # 使用最佳掩码
    multimask_output=False
)
```

## 自动掩码生成

### 基本自动分割

```python
from segment_anything import SamAutomaticMaskGenerator

# 创建生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 生成所有掩码
masks = mask_generator.generate(image)

# 每个掩码包含：
# - segmentation：二值掩码
# - bbox：[x, y, w, h]
# - area：像素数
# - predicted_iou：质量分数
# - stability_score：鲁棒性分数
# - point_coords：生成点
```

### 自定义生成

```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,          # 网格密度（越多 = 更多掩码）
    pred_iou_thresh=0.88,        # 质量阈值
    stability_score_thresh=0.95,  # 稳定性阈值
    crop_n_layers=1,             # 多尺度裁剪
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,    # 移除微小掩码
)

masks = mask_generator.generate(image)
```

### 过滤掩码

```python
# 按面积排序（从大到小）
masks = sorted(masks, key=lambda x: x['area'], reverse=True)

# 按预测 IoU 过滤
high_quality = [m for m in masks if m['predicted_iou'] > 0.9]

# 按稳定性分数过滤
stable_masks = [m for m in masks if m['stability_score'] > 0.95]
```

## 批量推理

### 多张图像

```python
# 高效处理多张图像
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

### 每张图像的多个提示

```python
# 高效处理多个提示（一次图像编码）
predictor.set_image(image)

# 点提示批次
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

## ONNX 部署

### 导出模型

```bash
python scripts/export_onnx_model.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --output sam_onnx.onnx \
    --return-single-mask
```

### 使用 ONNX 模型

```python
import onnxruntime

# 加载 ONNX 模型
ort_session = onnxruntime.InferenceSession("sam_onnx.onnx")

# 运行推理（图像嵌入单独计算）
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

## 常见工作流

### 工作流 1：标注工具

```python
import cv2

# 加载模型
predictor = SamPredictor(sam)
predictor.set_image(image)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 前景点
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        # 显示最佳掩码
        display_mask(masks[np.argmax(scores)])
```

### 工作流 2：对象提取

```python
def extract_object(image, point):
    """提取指定点处的对象，背景透明。"""
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    # 创建 RGBA 输出
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = best_mask * 255

    return rgba
```

### 工作流 3：医学图像分割

```python
# 处理医学图像（灰度转 RGB）
medical_image = cv2.imread("scan.png", cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.cvtColor(medical_image, cv2.COLOR_GRAY2RGB)

predictor.set_image(rgb_image)

# 分割感兴趣区域
masks, scores, _ = predictor.predict(
    box=np.array([x1, y1, x2, y2]),  # ROI 边界框
    multimask_output=True
)
```

## 输出格式

### 掩码数据结构

```python
# SamAutomaticMaskGenerator 输出
{
    "segmentation": np.ndarray,  # H×W 二值掩码
    "bbox": [x, y, w, h],        # 边界框
    "area": int,                 # 像素数
    "predicted_iou": float,      # 0-1 质量分数
    "stability_score": float,    # 0-1 鲁棒性分数
    "crop_box": [x, y, w, h],    # 生成裁剪区域
    "point_coords": [[x, y]],    # 输入点
}
```

### COCO RLE 格式

```python
from pycocotools import mask as mask_utils

# 将掩码编码为 RLE
rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
rle["counts"] = rle["counts"].decode("utf-8")

# 将 RLE 解码为掩码
decoded_mask = mask_utils.decode(rle)
```

## 性能优化

### GPU 内存

```python
# 在 VRAM 有限时使用较小的模型
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

# 批量处理图像
# 在大批次之间清理 CUDA 缓存
torch.cuda.empty_cache()
```

### 速度优化

```python
# 使用半精度
sam = sam.half()

# 减少自动生成的点数
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # 默认为 32
)

# 使用 ONNX 进行部署
# 使用 --return-single-mask 导出以获得更快的推理
```

## 常见问题

| 问题 | 解决方案 |
|-------|----------|
| 内存不足 | 使用 ViT-B 模型，减小图像大小 |
| 推理速度慢 | 使用 ViT-B，减小 points_per_side |
| 掩码质量差 | 尝试不同的提示，使用框 + 点 |
| 边缘伪影 | 使用 stability_score 过滤 |
| 遗漏小对象 | 增加 points_per_side |

## 参考

- **[高级用法](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/segment-anything/references/advanced-usage.md)** - 批处理、微调、集成
- **[故障排除](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/segment-anything/references/troubleshooting.md)** - 常见问题及解决方案

## 资源

- **GitHub**：https://github.com/facebookresearch/segment-anything
- **论文**：https://arxiv.org/abs/2304.02643
- **演示**：https://segment-anything.com
- **SAM 2（视频）**：https://github.com/facebookresearch/segment-anything-2
- **HuggingFace**：https://huggingface.co/facebook/sam-vit-huge
