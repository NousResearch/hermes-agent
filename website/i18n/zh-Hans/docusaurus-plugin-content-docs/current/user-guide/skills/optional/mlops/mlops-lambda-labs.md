---
title: "Lambda Labs Gpu Cloud — 用于 ML 训练和推理的预留和按需 GPU 云实例"
sidebar_label: "Lambda Labs Gpu Cloud"
description: "用于 ML 训练和推理的预留和按需 GPU 云实例"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Lambda Labs Gpu Cloud

用于 ML 训练和推理的预留和按需 GPU 云实例。适用于需要专用 GPU 实例（支持简单 SSH 访问、持久文件系统）或高性能多节点集群进行大规模训练的场景。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/lambda-labs` |
| Path | `optional-skills/mlops/lambda-labs` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `lambda-cloud-client>=1.0.0` |
| Tags | `Infrastructure`, `GPU Cloud`, `训练ing`, `Inference`, `Lambda Labs` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Lambda Labs GPU Cloud

Comprehensive guide to running ML workloads on Lambda Labs GPU cloud with on-demand instances and 一键集群.

## 何时使用 Lambda Labs

**Use Lambda Labs when:**
- Need dedicated GPU 实例 with full SSH access
- Running long training jobs (hours to days)
- Want simple pricing with no egress fees
- Need persistent storage across sessions
- Require high-performance multi-node clusters (16-512 GPUs)
- Want pre-installed ML stack (Lambda Stack with PyTorch, CUDA, NCCL)

**Key features:**
- **GPU variety**: B200, H100, GH200, A100, A10, A6000, V100
- **Lambda Stack**: Pre-installed PyTorch, TensorFlow, CUDA, cuDNN, NCCL
- **持久文件系统**: Keep data across instance restarts
- **一键集群**: 16-512 GPU Slurm clusters with InfiniBand
- **简单定价**: Pay-per-minute, no egress fees
- **全球区域**: 12+ regions worldwide

**替代方案:**
- **Modal**: For serverless, auto-scaling workloads
- **SkyPilot**: For multi-cloud orchestration and cost optimization
- **RunPod**: For cheaper spot instances and serverless endpoints
- **Vast.ai**: For GPU marketplace with lowest prices

## 快速入门

### 账户设置

1. Create account at https://lambda.ai
2. Add payment method
3. Generate API key from dashboard
4. 添加 SSH 密钥 (required before launching instances)

### 通过控制台启动

1. 转到 https://cloud.lambda.ai/instances
2. 点击 "启动实例"
3. Select GPU type and region
4. Choose SSH key
5. Optionally attach filesystem
6. 启动 and wait 3-15 minutes

### 通过 SSH 连接

```bash
# 从控制台获取实例 IP
ssh ubuntu@<INSTANCE-IP>

# 或使用指定密钥
ssh -i ~/.ssh/lambda_key ubuntu@<INSTANCE-IP>
```

## GPU 实例

### 可用 GPU

| GPU | 显存 | 价格/GPU/小时 | 最适合 |
|-----|------|--------------|----------|
| B200 SXM6 | 180 GB | $4.99 | Largest models, fastest training |
| H100 SXM | 80 GB | $2.99-3.29 | Large model training |
| H100 PCIe | 80 GB | $2.49 | Cost-effective H100 |
| GH200 | 96 GB | $1.49 | Single-GPU large models |
| A100 80GB | 80 GB | $1.79 | Production training |
| A100 40GB | 40 GB | $1.29 | Standard training |
| A10 | 24 GB | $0.75 | Inference, fine-tuning |
| A6000 | 48 GB | $0.80 | Good VRAM/price ratio |
| V100 | 16 GB | $0.55 | Budget training |

### 实例配置

```
8x GPU: 最适合 distributed training (DDP, FSDP)
4x GPU: Large models, multi-GPU training
2x GPU: Medium workloads
1x GPU: Fine-tuning, inference, development
```

### 启动时间

- Single-GPU: 3-5 minutes
- 多 GPU: 10-15 minutes

## Lambda Stack

All instances come with Lambda Stack pre-installed:

```bash
# 包含的软件
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

### 验证安装

```bash
# 检查 GPU
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 检查 CUDA 版本
nvcc --version
```

## Python API

### 安装

```bash
pip install lambda-cloud-client
```

### 认证

```python
import os
import lambda_cloud_client

# 配置 with API key
configuration = lambda_cloud_client.Configuration(
    host="https://cloud.lambdalabs.com/api/v1",
    access_token=os.environ["LAMBDA_API_KEY"]
)
```

### 列出可用实例

```python
with lambda_cloud_client.ApiClient(configuration) as api_client:
    api = lambda_cloud_client.DefaultApi(api_client)

    # 获取可用实例类型
    types = api.instance_types()
    for name, info in types.data.items():
        print(f"{name}: {info.instance_type.description}")
```

### 启动实例

```python
from lambda_cloud_client.models import 启动InstanceRequest

request = 启动InstanceRequest(
    region_name="us-west-1",
    instance_type_name="gpu_1x_h100_sxm5",
    ssh_key_names=["my-ssh-key"],
    file_system_names=["my-filesystem"],  # Optional
    name="training-job"
)

response = api.launch_instance(request)
instance_id = response.data.instance_ids[0]
print(f"启动ed: {instance_id}")
```

### 列出运行中的实例

```python
instances = api.list_instances()
for instance in instances.data:
    print(f"{instance.name}: {instance.ip} ({instance.status})")
```

### 终止实例

```python
from lambda_cloud_client.models import TerminateInstanceRequest

request = TerminateInstanceRequest(
    instance_ids=[instance_id]
)
api.terminate_instance(request)
```

### SSH 密钥管理

```python
from lambda_cloud_client.models import AddSshKeyRequest

# 添加 SSH 密钥
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

## 使用 curl 的 CLI

### List instance types

```bash
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instance-types | jq
```

### 启动实例

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

### 终止实例

```bash
curl -u $LAMBDA_API_KEY: \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" \
  -d '{"instance_ids": ["<INSTANCE-ID>"]}' | jq
```

## 持久存储

### 文件系统

文件系统 persist data across instance restarts:

```bash
# Mount location
/lambda/nfs/<FILESYSTEM_NAME>

# Example: save checkpoints
python train.py --checkpoint-dir /lambda/nfs/my-storage/checkpoints
```

### 创建文件系统

1. 转到 存储 in Lambda console
2. 点击 "创建文件系统"
3. Select region (must match instance region)
4. Name and create

### 附加到实例

文件系统 must be attached at instance launch time:
- Via console: Select filesystem when launching
- Via API: Include `file_system_names` in launch request

### Best practices

<!-- ascii-guard-ignore -->
```bash
# 存储到文件系统（持久化）
/lambda/nfs/storage/
  ├── datasets/
  ├── checkpoints/
  ├── models/
  └── outputs/

# 本地 SSD（更快，临时）
/home/ubuntu/
  └── working/  # Temporary files
```
<!-- ascii-guard-ignore-end -->

## SSH 配置

### 添加 SSH 密钥

```bash
# Generate key locally
ssh-keygen -t ed25519 -f ~/.ssh/lambda_key

# Add public key to Lambda console
# Or via API
```

### 多个密钥

```bash
# 在实例上, add more keys
echo 'ssh-rsa AAAA...' >> ~/.ssh/authorized_keys
```

### 从 GitHub 导入

```bash
# 在实例上
ssh-import-id gh:username
```

### SSH 隧道

```bash
# 转发 Jupyter
ssh -L 8888:localhost:8888 ubuntu@<IP>

# 转发 TensorBoard
ssh -L 6006:localhost:6006 ubuntu@<IP>

# 多个端口
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 ubuntu@<IP>
```

## JupyterLab

### 从控制台启动

1. 转到 Instances page
2. 点击 "启动" in Cloud IDE column
3. JupyterLab opens in browser

### 手动访问

```bash
# 在实例上
jupyter lab --ip=0.0.0.0 --port=8888

# 来自 local machine with tunnel
ssh -L 8888:localhost:8888 ubuntu@<IP>
# Open http://localhost:8888
```

## 训练工作流

### 单 GPU 训练

```bash
# SSH 到实例
ssh ubuntu@<IP>

# 克隆仓库
git clone https://github.com/user/project
cd project

# 安装依赖
pip install -r requirements.txt

# 训练
python train.py --epochs 100 --checkpoint-dir /lambda/nfs/storage/checkpoints
```

### 多 GPU 训练（单节点）

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

    # 训练ing loop...

if __name__ == "__main__":
    main()
```

```bash
# 启动 with torchrun (8 GPUs)
torchrun --nproc_per_node=8 train_ddp.py
```

### 检查点保存到文件系统

```python
import os

checkpoint_dir = "/lambda/nfs/my-storage/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 保存 checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f"{checkpoint_dir}/checkpoint_{epoch}.pt")
```

## 一键集群

### 概述

High-performance Slurm clusters with:
- 16-512 NVIDIA H100 or B200 GPUs
- NVIDIA Quantum-2 400 Gb/s InfiniBand
- GPUDirect RDMA at 3200 Gb/s
- Pre-installed distributed ML stack

### 包含的软件

- Ubuntu 22.04 LTS + Lambda Stack
- NCCL, Open MPI
- PyTorch with DDP and FSDP
- TensorFlow
- OFED drivers

### 存储

- 24 TB NVMe per compute node (ephemeral)
- Lambda filesystems for persistent data

### 多节点训练

```bash
# On Slurm cluster
srun --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 \
  torchrun --nnodes=4 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py
```

## 网络

### 带宽

- Inter-instance (same region): up to 200 Gbps
- Internet outbound: 20 Gbps max

### 防火墙

- Default: Only port 22 (SSH) open
- 配置 additional ports in Lambda console
- ICMP traffic allowed by default

### 私有 IP

```bash
# 查找私有 IP
ip addr show | grep 'inet '
```

## 常见工作流

### Workflow 1: Fine-tuning LLM

```bash
# 1. 启动 8x H100 instance with filesystem

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

### Workflow 2: Batch inference

```bash
# 1. 启动 A10 instance (cost-effective for inference)

# 2. Run inference
python inference.py \
  --model /lambda/nfs/storage/models/fine-tuned \
  --input /lambda/nfs/storage/data/inputs.jsonl \
  --output /lambda/nfs/storage/data/outputs.jsonl
```

## Cost optimization

### 选择合适的 GPU

| 任务 | 推荐 GPU |
|------|-----------------|
| LLM fine-tuning (7B) | A100 40GB |
| LLM fine-tuning (70B) | 8x H100 |
| Inference | A10, A6000 |
| Development | V100, A10 |
| Maximum performance | B200 |

### 降低成本

1. **Use filesystems**: Avoid re-downloading data
2. **Checkpoint frequently**: Resume interrupted training
3. **Right-size**: Don't over-provision GPUs
4. **Terminate idle**: No auto-stop, manually terminate

### 监控使用情况

- Dashboard shows real-time GPU utilization
- API for programmatic monitoring

## Common issues

| 问题 | 解决方案 |
|-------|----------|
| 实例无法启动 | 检查区域可用性，尝试不同的 GPU |
| SSH 连接被拒绝 | 等待实例初始化（3-15 分钟） |
| 终止后数据丢失 | 使用持久文件系统 |
| 数据传输缓慢 | 使用相同区域的文件系统 |
| GPU 未检测到 | 重启实例，检查驱动程序 |

## References

- **[Advanced Usage](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/lambda-labs/references/advanced-usage.md)** - 多节点训练, API automation
- **[Troubleshooting](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/lambda-labs/references/troubleshooting.md)** - Common issues and solutions

## Resources

- **Documentation**: https://docs.lambda.ai
- **Console**: https://cloud.lambda.ai
- **Pricing**: https://lambda.ai/instances
- **Support**: https://support.lambdalabs.com
- **Blog**: https://lambda.ai/blog
