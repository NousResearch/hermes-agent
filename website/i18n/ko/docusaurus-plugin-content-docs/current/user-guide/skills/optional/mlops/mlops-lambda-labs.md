---
title: "Lambda Labs Gpu Cloud — ML 학습 및 추론을 위한 예약 및 온디맨드 GPU 클라우드 인스턴스"
sidebar_label: "Lambda Labs Gpu Cloud"
description: "ML 학습 및 추론을 위한 예약 및 온디맨드 GPU 클라우드 인스턴스"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Lambda Labs Gpu Cloud

ML 학습 및 추론을 위한 예약 및 온디맨드 GPU 클라우드 인스턴스. 간단한 SSH 액세스가 가능한 전용 GPU 인스턴스, 영구 파일 시스템 또는 대규모 학습을 위한 고성능 다중 노드 클러스터가 필요할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/lambda-labs`로 설치 |
| Path | `optional-skills/mlops/lambda-labs` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `lambda-cloud-client>=1.0.0` |
| Platforms | linux, macos, windows |
| Tags | `Infrastructure`, `GPU Cloud`, `Training`, `Inference`, `Lambda Labs` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Lambda Labs GPU Cloud

온디맨드 인스턴스와 1-Click 클러스터를 사용하여 Lambda Labs GPU 클라우드에서 ML 워크로드를 실행하기 위한 포괄적인 가이드.

## Lambda Labs를 사용해야 할 때

**다음과 같은 경우 Lambda Labs를 사용하세요:**
- 전체 SSH 액세스가 가능한 전용 GPU 인스턴스가 필요할 때
- 긴 학습 작업(시간에서 일 단위)을 실행할 때
- 송신(egress) 수수료가 없는 단순한 요금제를 원할 때
- 세션 전반에 걸쳐 영구 스토리지가 필요할 때
- 고성능 다중 노드 클러스터(16-512 GPU)가 필요할 때
- 사전 설치된 ML 스택(PyTorch, CUDA, NCCL이 포함된 Lambda Stack)을 원할 때

**주요 특징:**
- **다양한 GPU**: B200, H100, GH200, A100, A10, A6000, V100
- **Lambda Stack**: PyTorch, TensorFlow, CUDA, cuDNN, NCCL 사전 설치
- **영구 파일 시스템**: 인스턴스 재시작 시에도 데이터 유지
- **1-Click 클러스터**: InfiniBand가 포함된 16-512 GPU Slurm 클러스터
- **단순한 요금제**: 분당 과금, 송신 수수료 없음
- **글로벌 리전**: 전 세계 12개 이상의 리전

**대신 다른 대안을 사용해야 할 때:**
- **Modal**: 서버리스, 자동 확장 워크로드의 경우
- **SkyPilot**: 다중 클라우드 오케스트레이션 및 비용 최적화의 경우
- **RunPod**: 더 저렴한 스팟 인스턴스 및 서버리스 엔드포인트의 경우
- **Vast.ai**: 최저가의 GPU 마켓플레이스의 경우

## 빠른 시작

### 계정 설정

1. https://lambda.ai 에서 계정 생성
2. 결제 수단 추가
3. 대시보드에서 API 키 생성
4. SSH 키 추가 (인스턴스 실행 전 필수)

### 콘솔을 통한 실행

1. https://cloud.lambda.ai/instances 로 이동
2. "Launch instance" 클릭
3. GPU 유형 및 리전 선택
4. SSH 키 선택
5. 필요시 파일 시스템 연결
6. 실행하고 3-15분 대기

### SSH를 통한 연결

```bash
# 콘솔에서 인스턴스 IP 가져오기
ssh ubuntu@<INSTANCE-IP>

# 또는 특정 키 사용
ssh -i ~/.ssh/lambda_key ubuntu@<INSTANCE-IP>
```

## GPU 인스턴스

### 사용 가능한 GPU

| GPU | VRAM | 가격/GPU/시간 | 권장 용도 |
|-----|------|--------------|----------|
| B200 SXM6 | 180 GB | $4.99 | 가장 큰 모델, 가장 빠른 학습 |
| H100 SXM | 80 GB | $2.99-3.29 | 대규모 모델 학습 |
| H100 PCIe | 80 GB | $2.49 | 비용 효율적인 H100 |
| GH200 | 96 GB | $1.49 | 단일 GPU 대규모 모델 |
| A100 80GB | 80 GB | $1.79 | 프로덕션 학습 |
| A100 40GB | 40 GB | $1.29 | 표준 학습 |
| A10 | 24 GB | $0.75 | 추론, 파인튜닝 |
| A6000 | 48 GB | $0.80 | 좋은 VRAM/가격 비율 |
| V100 | 16 GB | $0.55 | 예산 친화적 학습 |

### 인스턴스 구성

```
8x GPU: 분산 학습에 가장 적합 (DDP, FSDP)
4x GPU: 대규모 모델, 다중 GPU 학습
2x GPU: 중간 규모 워크로드
1x GPU: 파인튜닝, 추론, 개발
```

### 실행 시간

- 단일 GPU: 3-5분
- 다중 GPU: 10-15분

## Lambda Stack

모든 인스턴스에는 Lambda Stack이 사전 설치되어 제공됩니다:

```bash
# 포함된 소프트웨어
- Ubuntu 22.04 LTS
- NVIDIA 드라이버 (최신)
- CUDA 12.x
- cuDNN 8.x
- NCCL (다중 GPU용)
- PyTorch (최신)
- TensorFlow (최신)
- JAX
- JupyterLab
```

### 설치 확인

```bash
# GPU 확인
nvidia-smi

# PyTorch 확인
python -c "import torch; print(torch.cuda.is_available())"

# CUDA 버전 확인
nvcc --version
```

## 파이썬 API

### 설치

```bash
pip install lambda-cloud-client
```

### 인증

```python
import os
import lambda_cloud_client

# API 키로 구성
configuration = lambda_cloud_client.Configuration(
    host="https://cloud.lambdalabs.com/api/v1",
    access_token=os.environ["LAMBDA_API_KEY"]
)
```

### 사용 가능한 인스턴스 나열

```python
with lambda_cloud_client.ApiClient(configuration) as api_client:
    api = lambda_cloud_client.DefaultApi(api_client)

    # 사용 가능한 인스턴스 유형 가져오기
    types = api.instance_types()
    for name, info in types.data.items():
        print(f"{name}: {info.instance_type.description}")
```

### 인스턴스 실행

```python
from lambda_cloud_client.models import LaunchInstanceRequest

request = LaunchInstanceRequest(
    region_name="us-west-1",
    instance_type_name="gpu_1x_h100_sxm5",
    ssh_key_names=["my-ssh-key"],
    file_system_names=["my-filesystem"],  # 선택 사항
    name="training-job"
)

response = api.launch_instance(request)
instance_id = response.data.instance_ids[0]
print(f"Launched: {instance_id}")
```

### 실행 중인 인스턴스 나열

```python
instances = api.list_instances()
for instance in instances.data:
    print(f"{instance.name}: {instance.ip} ({instance.status})")
```

### 인스턴스 종료

```python
from lambda_cloud_client.models import TerminateInstanceRequest

request = TerminateInstanceRequest(
    instance_ids=[instance_id]
)
api.terminate_instance(request)
```

### SSH 키 관리

```python
from lambda_cloud_client.models import AddSshKeyRequest

# SSH 키 추가
request = AddSshKeyRequest(
    name="my-key",
    public_key="ssh-rsa AAAA..."
)
api.add_ssh_key(request)

# 키 나열
keys = api.list_ssh_keys()

# 키 삭제
api.delete_ssh_key(key_id)
```

## curl을 사용한 CLI

### 인스턴스 유형 나열

```bash
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instance-types | jq
```

### 인스턴스 실행

```bash
curl -u $LAMBDA_API_KEY: \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
  -H "Content-Type: application/json" \
  -d '{
    "region_name": "us-west-1",
    "instance_type_name": "gpu_1x_h100_sxm5",
    "ssh_key_names": ["my-key"]
  }' | jq
```

### 인스턴스 종료

```bash
curl -u $LAMBDA_API_KEY: \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" \
  -d '{"instance_ids": ["<INSTANCE-ID>"]}' | jq
```

## 영구 스토리지 (Persistent storage)

### 파일 시스템

파일 시스템은 인스턴스 재시작 간에 데이터를 유지합니다:

```bash
# 마운트 위치
/lambda/nfs/<FILESYSTEM_NAME>

# 예시: 체크포인트 저장
python train.py --checkpoint-dir /lambda/nfs/my-storage/checkpoints
```

### 파일 시스템 생성

1. Lambda 콘솔의 Storage로 이동합니다.
2. "Create filesystem"을 클릭합니다.
3. 리전을 선택합니다 (인스턴스 리전과 일치해야 함).
4. 이름을 지정하고 생성합니다.

### 인스턴스에 연결

파일 시스템은 인스턴스 시작 시에 연결되어야 합니다:
- 콘솔을 통해: 시작 시 파일 시스템을 선택합니다.
- API를 통해: 실행 요청에 `file_system_names`를 포함합니다.

### 모범 사례

<!-- ascii-guard-ignore -->
```bash
# 파일 시스템에 저장 (영구적)
/lambda/nfs/storage/
  ├── datasets/
  ├── checkpoints/
  ├── models/
  └── outputs/

# 로컬 SSD (더 빠름, 임시적)
/home/ubuntu/
  └── working/  # 임시 파일
```
<!-- ascii-guard-ignore-end -->

## SSH 구성

### SSH 키 추가

```bash
# 로컬에서 키 생성
ssh-keygen -t ed25519 -f ~/.ssh/lambda_key

# Lambda 콘솔에 퍼블릭 키 추가
# 또는 API를 통해 추가
```

### 다중 키

```bash
# 인스턴스에서 추가 키 추가
echo 'ssh-rsa AAAA...' >> ~/.ssh/authorized_keys
```

### GitHub에서 가져오기

```bash
# 인스턴스에서
ssh-import-id gh:username
```

### SSH 터널링

```bash
# Jupyter 포워딩
ssh -L 8888:localhost:8888 ubuntu@<IP>

# TensorBoard 포워딩
ssh -L 6006:localhost:6006 ubuntu@<IP>

# 다중 포트
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 ubuntu@<IP>
```

## JupyterLab

### 콘솔에서 실행

1. Instances 페이지로 이동합니다.
2. Cloud IDE 열에서 "Launch"를 클릭합니다.
3. 브라우저에서 JupyterLab이 열립니다.

### 수동 액세스

```bash
# 인스턴스에서
jupyter lab --ip=0.0.0.0 --port=8888

# 터널이 있는 로컬 시스템에서
ssh -L 8888:localhost:8888 ubuntu@<IP>
# http://localhost:8888 열기
```

## 학습 워크플로

### 단일 GPU 학습

```bash
# 인스턴스로 SSH 접속
ssh ubuntu@<IP>

# 리포지토리 클론
git clone https://github.com/user/project
cd project

# 의존성 설치
pip install -r requirements.txt

# 학습
python train.py --epochs 100 --checkpoint-dir /lambda/nfs/storage/checkpoints
```

### 다중 GPU 학습 (단일 노드)

```python
# train_ddp.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    model = MyModel().to(device)
    model = DDP(model, device_ids=[device])

    # 학습 루프...

if __name__ == "__main__":
    main()
```

```bash
# torchrun으로 실행 (8 GPU)
torchrun --nproc_per_node=8 train_ddp.py
```

### 파일 시스템에 체크포인트 저장

```python
import os

checkpoint_dir = "/lambda/nfs/my-storage/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 체크포인트 저장
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f"{checkpoint_dir}/checkpoint_{epoch}.pt")
```

## 1-Click 클러스터

### 개요

다음을 갖춘 고성능 Slurm 클러스터:
- 16-512 NVIDIA H100 또는 B200 GPU
- NVIDIA Quantum-2 400 Gb/s InfiniBand
- 3200 Gb/s의 GPUDirect RDMA
- 사전 설치된 분산 ML 스택

### 포함된 소프트웨어

- Ubuntu 22.04 LTS + Lambda Stack
- NCCL, Open MPI
- DDP 및 FSDP가 포함된 PyTorch
- TensorFlow
- OFED 드라이버

### 스토리지

- 컴퓨팅 노드당 24 TB NVMe (임시)
- 영구 데이터를 위한 Lambda 파일 시스템

### 다중 노드 학습

```bash
# Slurm 클러스터에서
srun --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 \
  torchrun --nnodes=4 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py
```

## 네트워킹

### 대역폭

- 인스턴스 간(동일 리전): 최대 200 Gbps
- 인터넷 아웃바운드: 최대 20 Gbps

### 방화벽

- 기본값: 포트 22(SSH)만 열려 있음
- Lambda 콘솔에서 추가 포트를 구성할 수 있음
- ICMP 트래픽은 기본적으로 허용됨

### 프라이빗 IP

```bash
# 프라이빗 IP 찾기
ip addr show | grep 'inet '
```

## 일반적인 워크플로

### 워크플로 1: LLM 파인튜닝

```bash
# 1. 파일 시스템과 함께 8x H100 인스턴스 실행

# 2. SSH 및 설정
ssh ubuntu@<IP>
pip install transformers accelerate peft

# 3. 모델을 파일 시스템에 다운로드
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model.save_pretrained('/lambda/nfs/storage/models/llama-2-7b')
"

# 4. 파일 시스템에 체크포인트를 두어 파인튜닝
accelerate launch --num_processes 8 train.py \
  --model_path /lambda/nfs/storage/models/llama-2-7b \
  --output_dir /lambda/nfs/storage/outputs \
  --checkpoint_dir /lambda/nfs/storage/checkpoints
```

### 워크플로 2: 배치 추론

```bash
# 1. A10 인스턴스 실행 (추론에 비용 효율적)

# 2. 추론 실행
python inference.py \
  --model /lambda/nfs/storage/models/fine-tuned \
  --input /lambda/nfs/storage/data/inputs.jsonl \
  --output /lambda/nfs/storage/data/outputs.jsonl
```

## 비용 최적화

### 올바른 GPU 선택

| 작업 | 권장 GPU |
|------|-----------------|
| LLM 파인튜닝 (7B) | A100 40GB |
| LLM 파인튜닝 (70B) | 8x H100 |
| 추론 | A10, A6000 |
| 개발 | V100, A10 |
| 최대 성능 | B200 |

### 비용 절감

1. **파일 시스템 사용**: 데이터 다시 다운로드 방지
2. **자주 체크포인트**: 중단된 학습 재개
3. **적절한 크기 조정**: GPU를 과도하게 프로비저닝하지 않음
4. **유휴 인스턴스 종료**: 자동 중지 기능이 없으므로 수동으로 종료

### 사용량 모니터링

- 대시보드에 실시간 GPU 활용도가 표시됩니다.
- 프로그래밍 방식 모니터링을 위한 API 제공.

## 일반적인 문제

| 문제 | 해결책 |
|-------|----------|
| 인스턴스가 실행되지 않음 | 리전 가용성 확인, 다른 GPU 시도 |
| SSH 연결 거부 | 인스턴스가 초기화될 때까지 대기 (3-15분) |
| 종료 후 데이터 손실 | 영구 파일 시스템 사용 |
| 느린 데이터 전송 | 동일한 리전의 파일 시스템 사용 |
| GPU가 감지되지 않음 | 인스턴스 재부팅, 드라이버 확인 |

## 참조

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/lambda-labs/references/advanced-usage.md)** - 다중 노드 학습, API 자동화
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/lambda-labs/references/troubleshooting.md)** - 일반적인 문제와 해결책

## 리소스

- **문서**: https://docs.lambda.ai
- **콘솔**: https://cloud.lambda.ai
- **가격**: https://lambda.ai/instances
- **지원**: https://support.lambdalabs.com
- **블로그**: https://lambda.ai/blog
