---
title: "Lambda Labs — ML 학습을 위한 온디맨드 GPU 클라우드 인스턴스"
sidebar_label: "Lambda Labs"
description: "ML 학습을 위한 온디맨드 GPU 클라우드 인스턴스"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Lambda Labs

ML 학습을 위한 온디맨드 GPU 클라우드 인스턴스입니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/lambda-labs`로 설치 |
| 경로 | `optional-skills/mlops/lambda-labs` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `lambda-cloud-client>=1.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Infrastructure`, `GPU Cloud`, `Training`, `Inference`, `Lambda Labs` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Lambda Labs GPU Cloud

온디맨드 인스턴스와 1-Click Clusters를 사용하여 Lambda Labs GPU 클라우드에서 ML 워크로드를 실행하는 안내서입니다.

## Lambda Labs를 사용하는 경우

**다음과 같은 경우 Lambda Labs를 사용하세요:**
- 완전한 SSH 액세스가 가능한 전용 GPU 인스턴스가 필요한 경우
- 장시간 학습 작업(수 시간에서 수 일)을 실행하는 경우
- 이그레스 요금 없는 단순한 가격 정책을 원하는 경우
- 세션 간에도 유지되는 스토리지가 필요한 경우
- 고성능 멀티노드 클러스터(16-512 GPU)가 필요한 경우
- 사전 설치된 ML 스택(PyTorch, CUDA, NCCL이 포함된 Lambda Stack)을 원하는 경우

**주요 기능:**
- **GPU 종류**: B200, H100, GH200, A100, A10, A6000, V100
- **Lambda Stack**: PyTorch, TensorFlow, CUDA, cuDNN, NCCL 사전 설치
- **영구 파일 시스템**: 인스턴스를 다시 시작해도 데이터 유지
- **1-Click Clusters**: InfiniBand가 포함된 16-512 GPU Slurm 클러스터
- **단순한 가격 정책**: 분당 과금, 이그레스 요금 없음
- **글로벌 리전**: 전 세계 12개 이상의 리전

**대신 다음 대안을 사용하세요:**
- **Modal**: 서버리스, 자동 확장 워크로드
- **SkyPilot**: 멀티클라우드 오케스트레이션 및 비용 최적화
- **RunPod**: 더 저렴한 스팟 인스턴스 및 서버리스 엔드포인트
- **Vast.ai**: 최저 가격의 GPU 마켓플레이스

## 빠른 시작

### 계정 설정

1. https://lambda.ai에서 계정 생성
2. 결제 수단 추가
3. 대시보드에서 API 키 생성
4. SSH 키 추가(인스턴스를 시작하기 전에 필수)

### 콘솔을 통한 시작

1. https://cloud.lambda.ai/instances로 이동
2. "Launch instance" 클릭
3. GPU 유형과 리전 선택
4. SSH 키 선택
5. 필요하면 파일 시스템 연결
6. 시작하고 3-15분 대기

### SSH로 연결

```bash
# Get instance IP from console
ssh ubuntu@<INSTANCE-IP>

# Or with specific key
ssh -i ~/.ssh/lambda_key ubuntu@<INSTANCE-IP>
```

## GPU 인스턴스

### 사용 가능한 GPU

| GPU | VRAM | GPU/시간당 가격 | 적합한 용도 |
|-----|------|--------------|----------|
| B200 SXM6 | 180 GB | $4.99 | 가장 큰 모델, 가장 빠른 학습 |
| H100 SXM | 80 GB | $2.99-3.29 | 대규모 모델 학습 |
| H100 PCIe | 80 GB | $2.49 | 비용 효율적인 H100 |
| GH200 | 96 GB | $1.49 | 단일 GPU 대규모 모델 |
| A100 80GB | 80 GB | $1.79 | 프로덕션 학습 |
| A100 40GB | 40 GB | $1.29 | 표준 학습 |
| A10 | 24 GB | $0.75 | 추론, 파인튜닝 |
| A6000 | 48 GB | $0.80 | 우수한 VRAM/가격 비율 |
| V100 | 16 GB | $0.55 | 저예산 학습 |

### 인스턴스 구성

```
8x GPU: Best for distributed training (DDP, FSDP)
4x GPU: Large models, multi-GPU training
2x GPU: Medium workloads
1x GPU: Fine-tuning, inference, development
```

### 시작 시간

- 단일 GPU: 3-5분
- 멀티 GPU: 10-15분

## Lambda Stack

모든 인스턴스에는 Lambda Stack이 사전 설치되어 제공됩니다:

```bash
# Included software
- Ubuntu 22.04 LTS
- NVIDIA drivers (latest)
- CUDA 12.x
- cuDNN 8.x
- NCCL (for multi-GPU)
- PyTorch (latest)
- TensorFlow (latest)
- JAX
- JupyterLab
```

### 설치 확인

```bash
# Check GPU
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvcc --version
```

## Python API

### 설치

```bash
pip install lambda-cloud-client
```

### 인증

```python
import os
import lambda_cloud_client

# Configure with API key
configuration = lambda_cloud_client.Configuration(
    host="https://cloud.lambdalabs.com/api/v1",
    access_token=os.environ["LAMBDA_API_KEY"]
)
```

### 사용 가능한 인스턴스 나열

```python
with lambda_cloud_client.ApiClient(configuration) as api_client:
    api = lambda_cloud_client.DefaultApi(api_client)

    # Get available instance types
    types = api.instance_types()
    for name, info in types.data.items():
        print(f"{name}: {info.instance_type.description}")
```

### 인스턴스 시작

```python
from lambda_cloud_client.models import LaunchInstanceRequest

request = LaunchInstanceRequest(
    region_name="us-west-1",
    instance_type_name="gpu_1x_h100_sxm5",
    ssh_key_names=["my-ssh-key"],
    file_system_names=["my-filesystem"],  # Optional
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

# Add SSH key
request = AddSshKeyRequest(
    name="my-key",
    public_key="ssh-rsa AAAA..."
)
api.add_ssh_key(request)

# List keys
keys = api.list_ssh_keys()

# Delete key
api.delete_ssh_key(key_id)
```

## curl을 사용한 CLI

### 인스턴스 유형 나열

```bash
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instance-types | jq
```

### 인스턴스 시작

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

## 영구 스토리지

### 파일 시스템

파일 시스템은 인스턴스를 다시 시작해도 데이터를 유지합니다:

```bash
# Mount location
/lambda/nfs/<FILESYSTEM_NAME>

# Example: save checkpoints
python train.py --checkpoint-dir /lambda/nfs/my-storage/checkpoints
```

### 파일 시스템 생성

1. Lambda 콘솔에서 Storage로 이동
2. "Create filesystem" 클릭
3. 리전 선택(인스턴스 리전과 일치해야 함)
4. 이름을 지정하고 생성

### 인스턴스에 연결

파일 시스템은 인스턴스를 시작할 때 연결해야 합니다:
- 콘솔 사용: 시작할 때 파일 시스템 선택
- API 사용: 시작 요청에 `file_system_names` 포함

### 모범 사례

<!-- ascii-guard-ignore -->
```bash
# Store on filesystem (persists)
/lambda/nfs/storage/
  ├── datasets/
  ├── checkpoints/
  ├── models/
  └── outputs/

# Local SSD (faster, ephemeral)
~/ (instance home)
  └── working/  # Temporary files
```
<!-- ascii-guard-ignore-end -->

## SSH 구성

### SSH 키 추가

```bash
# Generate key locally
ssh-keygen -t ed25519 -f ~/.ssh/lambda_key

# Add public key to Lambda console
# Or via API
```

### 여러 키

```bash
# On instance, add more keys
echo 'ssh-rsa AAAA...' >> ~/.ssh/authorized_keys
```

### GitHub에서 가져오기

```bash
# On instance
ssh-import-id gh:username
```

### SSH 터널링

```bash
# Forward Jupyter
ssh -L 8888:localhost:8888 ubuntu@<IP>

# Forward TensorBoard
ssh -L 6006:localhost:6006 ubuntu@<IP>

# Multiple ports
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 ubuntu@<IP>
```

## JupyterLab

### 콘솔에서 시작

1. Instances 페이지로 이동
2. Cloud IDE 열에서 "Launch" 클릭
3. 브라우저에서 JupyterLab 열림

### 수동 액세스

```bash
# On instance
jupyter lab --ip=0.0.0.0 --port=8888

# From local machine with tunnel
ssh -L 8888:localhost:8888 ubuntu@<IP>
# Open http://localhost:8888
```

## 학습 워크플로

### 단일 GPU 학습

```bash
# SSH to instance
ssh ubuntu@<IP>

# Clone repo
git clone https://github.com/user/project
cd project

# Install dependencies
pip install -r requirements.txt

# Train
python train.py --epochs 100 --checkpoint-dir /lambda/nfs/storage/checkpoints
```

### 멀티 GPU 학습(단일 노드)

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

    # Training loop...

if __name__ == "__main__":
    main()
```

```bash
# Launch with torchrun (8 GPUs)
torchrun --nproc_per_node=8 train_ddp.py
```

### 파일 시스템에 체크포인트 저장

```python
import os

checkpoint_dir = "/lambda/nfs/my-storage/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f"{checkpoint_dir}/checkpoint_{epoch}.pt")
```

## 1-Click Clusters

### 개요

다음이 포함된 고성능 Slurm 클러스터:
- NVIDIA H100 또는 B200 GPU 16-512개
- NVIDIA Quantum-2 400 Gb/s InfiniBand
- 3200 Gb/s의 GPUDirect RDMA
- 사전 설치된 분산 ML 스택

### 포함된 소프트웨어

- Ubuntu 22.04 LTS + Lambda Stack
- NCCL, Open MPI
- DDP와 FSDP가 포함된 PyTorch
- TensorFlow
- OFED 드라이버

### 스토리지

- 컴퓨트 노드당 24 TB NVMe(임시)
- 영구 데이터용 Lambda 파일 시스템

### 멀티노드 학습

```bash
# On Slurm cluster
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

- 기본값: 포트 22(SSH)만 열림
- Lambda 콘솔에서 추가 포트 구성
- ICMP 트래픽은 기본적으로 허용

### 프라이빗 IP

```bash
# Find private IP
ip addr show | grep 'inet '
```

## 일반적인 워크플로

### 워크플로 1: LLM 파인튜닝

```bash
# 1. Launch 8x H100 instance with filesystem

# 2. SSH and setup
ssh ubuntu@<IP>
pip install transformers accelerate peft

# 3. Download model to filesystem
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model.save_pretrained('/lambda/nfs/storage/models/llama-2-7b')
"

# 4. Fine-tune with checkpoints on filesystem
accelerate launch --num_processes 8 train.py \
  --model_path /lambda/nfs/storage/models/llama-2-7b \
  --output_dir /lambda/nfs/storage/outputs \
  --checkpoint_dir /lambda/nfs/storage/checkpoints
```

### 워크플로 2: 배치 추론

```bash
# 1. Launch A10 instance (cost-effective for inference)

# 2. Run inference
python inference.py \
  --model /lambda/nfs/storage/models/fine-tuned \
  --input /lambda/nfs/storage/data/inputs.jsonl \
  --output /lambda/nfs/storage/data/outputs.jsonl
```

## 비용 최적화

### 적합한 GPU 선택

| 작업 | 권장 GPU |
|------|---------|
| LLM 파인튜닝(7B) | A100 40GB |
| LLM 파인튜닝(70B) | 8x H100 |
| 추론 | A10, A6000 |
| 개발 | V100, A10 |
| 최대 성능 | B200 |

### 비용 절감

1. **파일 시스템 사용**: 데이터를 다시 다운로드하지 않기
2. **자주 체크포인트 저장**: 중단된 학습 재개
3. **적정 규모 선택**: GPU를 과도하게 프로비저닝하지 않기
4. **유휴 인스턴스 종료**: 자동 중지가 없으므로 수동으로 종료

### 사용량 모니터링

- 대시보드에서 실시간 GPU 사용률 표시
- 프로그래밍 방식의 모니터링을 위한 API

## 일반적인 문제

| 문제 | 해결 방법 |
|-------|----------|
| 인스턴스가 시작되지 않음 | 리전 가용성을 확인하고 다른 GPU 시도 |
| SSH 연결이 거부됨 | 인스턴스 초기화(3-15분)가 끝날 때까지 대기 |
| 종료 후 데이터가 손실됨 | 영구 파일 시스템 사용 |
| 데이터 전송이 느림 | 동일 리전의 파일 시스템 사용 |
| GPU가 감지되지 않음 | 인스턴스를 재부팅하고 드라이버 확인 |

## 참고 자료

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/lambda-labs/references/advanced-usage.md)** - 멀티노드 학습, API 자동화
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/lambda-labs/references/troubleshooting.md)** - 일반적인 문제와 해결 방법

## 리소스

- **문서**: https://docs.lambda.ai
- **콘솔**: https://cloud.lambda.ai
- **가격**: https://lambda.ai/instances
- **지원**: https://support.lambdalabs.com
- **블로그**: https://lambda.ai/blog
