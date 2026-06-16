---
title: "Comfyui"
sidebar_label: "Comfyui"
description: "使用 ComfyUI 生成图像、视频和音频——安装、启动、管理节点/模型、运行带参数注入的工作流"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Comfyui

使用 ComfyUI 生成图像、视频和音频——安装、启动、管理节点/模型、运行带参数注入的工作流。使用官方 comfy-cli 进行生命周期管理，使用直接 REST/WebSocket API 执行工作流。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/comfyui` |
| 版本 | `5.1.0` |
| 作者 | ['kshitijk4poor', 'alt-glitch', 'purzbeats'] |
| 许可证 | MIT |
| 平台 | macos, linux, windows |
| 标签 | `comfyui`, `image-generation`, `stable-diffusion`, `flux`, `sd3`, `wan-video`, `hunyuan-video`, `creative`, `generative-ai`, `video-generation` |
| 相关 skill | [`stable-diffusion-image-generation`](/user-guide/skills/optional/mlops/mlops-stable-diffusion), `image_gen` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# ComfyUI

通过 ComfyUI 生成图像、视频、音频和 3D 内容，使用官方 `comfy-cli` 进行安装/生命周期管理，使用直接 REST/WebSocket API 执行工作流。

## 此 skill 包含的内容

**参考文档（`references/`）：**

- `official-cli.md` — 所有 `comfy ...` 命令及其标志
- `rest-api.md` — REST + WebSocket 端点（本地 + 云端），payload（载荷）schema
- `workflow-format.md` — API 格式 JSON、常见节点类型、参数映射
- `template-integrity.md` — 将 `comfyui-workflow-templates` 从编辑器格式转换为 API 格式：Reroute bypass、点分动态输入键（`values.a`、`resize_type.width`）、云端特性（302 重定向、免费层 1 个并发任务、1080p VRAM 上限）、Discord 兼容 ffmpeg 拼接。由 [@purzbeats](https://github.com/purzbeats) 撰写。从官方模板开始时请加载此文档。

**脚本（`scripts/`）：**

| 脚本 | 用途 |
|--------|---------|
| `_common.py` | 共享 HTTP、云端路由、节点目录（不要直接运行） |
| `hardware_check.py` | 探测 GPU/VRAM/磁盘 → 推荐本地或 Comfy Cloud |
| `comfyui_setup.sh` | 硬件检查 + comfy-cli + ComfyUI 安装 + 启动 + 验证 |
| `extract_schema.py` | 读取工作流 → 列出可控参数 + 模型依赖 |
| `check_deps.py` | 对比运行中的服务器检查工作流 → 列出缺失节点/模型 |
| `auto_fix_deps.py` | 运行 check_deps 然后执行 `comfy node install` / `comfy model download` |
| `run_workflow.py` | 注入参数、提交、监控、下载输出（HTTP 或 WS） |
| `run_batch.py` | 以 sweep 方式提交工作流 N 次，并行数量受限于你的套餐层级 |
| `ws_monitor.py` | 执行中任务的实时 WebSocket 查看器（实时进度） |
| `health_check.py` | 验证清单运行器——comfy-cli + 服务器 + 模型 + 冒烟测试 |
| `fetch_logs.py` | 拉取指定 prompt_id 的 traceback / 状态消息 |

**示例工作流（`workflows/`）：** SD 1.5、SDXL、Flux Dev、SDXL img2img、SDXL inpaint、ESRGAN 放大、AnimateDiff 视频、Wan T2V。参见 `workflows/README.md`。

## 使用场景

- 用户要求使用 Stable Diffusion、SDXL、Flux、SD3 等生成图像
- 用户想运行特定的 ComfyUI 工作流文件
- 用户想串联生成步骤（txt2img → 放大 → 人脸修复）
- 用户需要 ControlNet、inpainting、img2img 或其他高级 pipeline
- 用户要管理 ComfyUI 队列、检查模型或安装自定义节点
- 用户想通过 AnimateDiff、Hunyuan、Wan、AudioCraft 等进行视频/音频/3D 生成

## 架构：两层

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────┐
│ Layer 1: comfy-cli (official lifecycle tool)        │
│   Setup, server lifecycle, custom nodes, models     │
│   → comfy install / launch / stop / node / model    │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│ Layer 2: REST/WebSocket API + skill scripts         │
│   Workflow execution, param injection, monitoring   │
│   POST /api/prompt, GET /api/view, WS /ws           │
│   → run_workflow.py, run_batch.py, ws_monitor.py    │
└─────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

**为什么要两层？** 官方 CLI 非常适合安装和服务器管理，但对工作流执行的支持极少。REST/WS API 填补了这一空缺——脚本处理 CLI 不具备的参数注入、执行监控和输出下载功能。

## 快速开始

### 检测环境

```bash
# 检查可用内容
command -v comfy >/dev/null 2>&1 && echo "comfy-cli: installed"
curl -s http://127.0.0.1:8188/system_stats 2>/dev/null && echo "server: running"

# 此机器能否在本地运行 ComfyUI？（GPU/VRAM/磁盘检查）
python3 scripts/hardware_check.py
```

如果未安装任何内容，请参阅下方的**安装与引导**——但始终先运行硬件检查。

### 一行健康检查

```bash
python3 scripts/health_check.py
# → JSON: comfy_cli 在 PATH 中？服务器可达？至少有一个 checkpoint？冒烟测试通过？
```

## 核心工作流

### 第一步：获取 API 格式的工作流 JSON

工作流必须为 API 格式（每个节点有 `class_type`）。来源包括：

- ComfyUI Web UI → **Workflow → Export (API)**（新版 UI）或旧版"Save (API Format)"按钮（旧版 UI）
- 此 skill 的 `workflows/` 目录（可直接运行的示例）
- 社区下载（civitai、Reddit、Discord）——通常为编辑器格式，必须加载到 ComfyUI 后重新导出

编辑器格式（顶层含 `nodes` 和 `links` 数组）**不可直接执行**。脚本会检测此情况并提示你重新导出。

### 第二步：查看可控内容

```bash
python3 scripts/extract_schema.py workflow_api.json --summary-only
# → {"parameter_count": 12, "has_negative_prompt": true, "has_seed": true, ...}

python3 scripts/extract_schema.py workflow_api.json
# → 完整 schema，包含参数、模型依赖、embedding 引用
```

### 第三步：带参数运行

```bash
# 本地（默认 http://127.0.0.1:8188）
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset over mountains", "seed": -1, "steps": 30}' \
  --output-dir ./outputs

# 云端（一次性导出 API key；自动使用正确的 /api 路由）
export COMFY_CLOUD_API_KEY="comfyui-..."
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "..."}' \
  --host https://cloud.comfy.org \
  --output-dir ./outputs

# 通过 WebSocket 实时查看进度（需要 `pip install websocket-client`）
python3 scripts/run_workflow.py \
  --workflow flux_dev.json \
  --args '{"prompt": "..."}' \
  --ws

# img2img / inpaint：传入 --input-image 自动上传并引用
python3 scripts/run_workflow.py \
  --workflow sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it watercolor", "denoise": 0.6}'

# 批量 / sweep：8 个随机种子，并行数量受限于云端套餐层级
python3 scripts/run_batch.py \
  --workflow sdxl.json \
  --args '{"prompt": "abstract"}' \
  --count 8 --randomize-seed --parallel 3 \
  --output-dir ./outputs/batch
```

`seed` 传 `-1`（或配合 `--randomize-seed` 省略 seed）可在每次运行时生成新的随机种子。

### 第四步：呈现结果

脚本向 stdout 输出描述每个输出文件的 JSON：

```json
{
  "status": "success",
  "prompt_id": "abc-123",
  "outputs": [
    {"file": "./outputs/sdxl_00001_.png", "node_id": "9",
     "type": "image", "filename": "sdxl_00001_.png"}
  ]
}
```

## 决策树

| 用户说 | 工具 | 命令 |
|-----------|------|---------|
| **生命周期（使用 comfy-cli）** | | |
| "安装 ComfyUI" | comfy-cli | `bash scripts/comfyui_setup.sh` |
| "启动 ComfyUI" | comfy-cli | `comfy launch --background` |
| "停止 ComfyUI" | comfy-cli | `comfy stop` |
| "安装 X 节点" | comfy-cli | `comfy node install <name>` |
| "下载 X 模型" | comfy-cli | `comfy model download --url <url> --relative-path models/checkpoints` |
| "列出已安装模型" | comfy-cli | `comfy model list` |
| "列出已安装节点" | comfy-cli | `comfy node show installed` |
| **执行（使用脚本）** | | |
| "一切准备好了吗？" | 脚本 | `health_check.py`（可选加 `--workflow X --smoke-test`） |
| "这个工作流我能改什么？" | 脚本 | `extract_schema.py W.json` |
| "检查 W 的依赖是否满足" | 脚本 | `check_deps.py W.json` |
| "修复缺失依赖" | 脚本 | `auto_fix_deps.py W.json` |
| "生成一张图片" | 脚本 | `run_workflow.py --workflow W --args '{...}'` |
| "使用这张图片"（img2img） | 脚本 | `run_workflow.py --input-image image=./x.png ...` |
| "8 个随机种子变体" | 脚本 | `run_batch.py --count 8 --randomize-seed ...` |
| "显示实时进度" | 脚本 | `ws_monitor.py --prompt-id <id>` |
| "获取任务 X 的错误" | 脚本 | `fetch_logs.py <prompt_id>` |
| **直接 REST** | | |
| "队列里有什么？" | REST | `curl http://HOST:8188/queue`（本地）或 `--host https://cloud.comfy.org` |
| "取消那个" | REST | `curl -X POST http://HOST:8188/interrupt` |
| "释放 GPU 内存" | REST | `curl -X POST http://HOST:8188/free` |

## 安装与引导

当用户要求安装 ComfyUI 时，**首先要询问他们想要 Comfy Cloud（托管，零安装，API key）还是本地安装（在其机器上安装 ComfyUI）**。在得到答复之前，不要开始运行安装命令或硬件检查。

**官方文档：** https://docs.comfy.org/installation
**CLI 文档：** https://docs.comfy.org/comfy-cli/getting-started
**Cloud 文档：** https://docs.comfy.org/get_started/cloud
**Cloud API：** https://docs.comfy.org/development/cloud/overview

### 第零步：询问本地还是云端（始终优先）

建议话术：

> "您想在本地机器上运行 ComfyUI，还是使用 Comfy Cloud？
>
> - **Comfy Cloud** — 托管于 RTX 6000 Pro GPU，所有常用模型预装，零配置。需要 API key（实际运行工作流需要付费订阅；免费层仅限只读）。如果您没有性能足够的 GPU，推荐此选项。
> - **本地** — 免费，但您的机器必须满足硬件要求：
>   - NVIDIA GPU，**≥6 GB VRAM**（SDXL 需 ≥8 GB，Flux/视频需 ≥12 GB），或
>   - 支持 ROCm 的 AMD GPU（Linux），或
>   - Apple Silicon Mac（M1+），**≥16 GB 统一内存**（推荐 ≥32 GB）。
>   - Intel Mac 和无 GPU 的机器**不可用**——请改用 Cloud。
>
> 您选择哪种？"

路由逻辑：

- **Cloud** → 跳至**路径 A**。
- **本地** → 先运行硬件检查，再根据结果从路径 B–E 中选择。
- **不确定** → 运行硬件检查，由结果决定。

### 第一步：验证硬件（仅当用户选择本地时）

```bash
python3 scripts/hardware_check.py --json
# 可选：同时探测 `torch` 以获取实际 CUDA/MPS 信息：
python3 scripts/hardware_check.py --json --check-pytorch
```

| 结果 | 含义 | 操作 |
|------------|---------------------------------------------------------------|--------|
| `ok` | ≥8 GB VRAM（独立显卡）或 ≥32 GB 统一内存（Apple Silicon） | 本地安装——使用报告中的 `comfy_cli_flag` |
| `marginal` | SD1.5 可用；SDXL 较紧张；Flux/视频不太可能 | 轻量工作流可本地，否则选**路径 A（Cloud）** |
| `cloud` | 无可用 GPU、&lt;6 GB VRAM、&lt;16 GB Apple 统一内存、Intel Mac、Rosetta Python | **切换至 Cloud**，除非用户明确强制本地 |

脚本还会显示 `wsl: true`（带 NVIDIA 直通的 WSL2）和 `rosetta: true`（Apple Silicon 上的 x86_64 Python——必须重新安装为 ARM64）。

如果结果为 `cloud` 但用户想要本地，不要静默继续。逐字显示 `notes` 数组，并询问他们是否要（a）切换至 Cloud 或（b）强制本地安装（在现代模型上会 OOM 或极慢）。

### 选择安装路径

优先使用硬件检查结果。下表适用于用户已告知其硬件的情况：

| 情况 | 推荐路径 |
|-----------|------------------|
| 硬件检查结果为 `verdict: cloud` | **路径 A：Comfy Cloud** |
| 无 GPU / 想先试用 | **路径 A：Comfy Cloud** |
| Windows + NVIDIA + 非技术用户 | **路径 B：ComfyUI Desktop** |
| Windows + NVIDIA + 技术用户 | **路径 C：Portable** 或**路径 D：comfy-cli** |
| Linux + 任意 GPU | **路径 D：comfy-cli**（最简单） |
| macOS + Apple Silicon | **路径 B：Desktop** 或**路径 D：comfy-cli** |
| 无头/服务器/CI/agent | **路径 D：comfy-cli** |

全自动路径（硬件检查 → 安装 → 启动 → 验证）：

```bash
bash scripts/comfyui_setup.sh
# 或带覆盖参数：
bash scripts/comfyui_setup.sh --m-series --port=8190 --workspace=/data/comfy
```

该脚本内部运行 `hardware_check.py`，当结果为 `cloud` 时拒绝本地安装（除非传入 `--force-cloud-override`），选择正确的 `comfy-cli` 标志，并优先使用 `pipx`/`uvx` 而非全局 `pip` 以避免污染系统 Python。

---

### 路径 A：Comfy Cloud（无需本地安装）

适用于没有性能足够 GPU 或想要零配置的用户。托管于 RTX 6000 Pro。

**文档：** https://docs.comfy.org/get_started/cloud

1. 在 https://comfy.org/cloud 注册
2. 在 https://platform.comfy.org/login 生成 API key
3. 设置 key：
   ```bash
   export COMFY_CLOUD_API_KEY="comfyui-xxxxxxxxxxxx"
   ```
4. 运行工作流：
   ```bash
   python3 scripts/run_workflow.py \
     --workflow workflows/flux_dev_txt2img.json \
     --args '{"prompt": "..."}' \
     --host https://cloud.comfy.org \
     --output-dir ./outputs
   ```

**定价：** https://www.comfy.org/cloud/pricing
**并发任务：** 免费/标准版 1 个，Creator 3 个，Pro 5 个。免费层**无法通过 API 运行工作流**——仅可浏览模型。`/api/prompt`、`/api/upload/*`、`/api/view` 等需要付费订阅。

---

### 路径 B：ComfyUI Desktop（Windows / macOS）

面向非技术用户的一键安装程序。目前为 Beta 版。

**文档：** https://docs.comfy.org/installation/desktop
- **Windows（NVIDIA）：** https://download.comfy.org/windows/nsis/x64
- **macOS（Apple Silicon）：** https://comfy.org

Linux **不支持** Desktop——请使用路径 D。

---

### 路径 C：ComfyUI Portable（仅 Windows）

**文档：** https://docs.comfy.org/installation/comfyui_portable_windows

从 https://github.com/comfyanonymous/ComfyUI/releases 下载，解压后运行 `run_nvidia_gpu.bat`。通过 `update/update_comfyui_stable.bat` 更新。

---

### 路径 D：comfy-cli（全平台——推荐用于 Agent）

官方 CLI 是无头/自动化安装的最佳路径。

**文档：** https://docs.comfy.org/comfy-cli/getting-started

#### 安装 comfy-cli

```bash
# 推荐：
pipx install comfy-cli
# 或不安装直接使用 uvx：
uvx --from comfy-cli comfy --help
# 或（如果 pipx/uvx 不可用）：
pip install --user comfy-cli
```

非交互式禁用分析：
```bash
comfy --skip-prompt tracking disable
```

#### 安装 ComfyUI

```bash
comfy --skip-prompt install --nvidia              # NVIDIA（CUDA）
comfy --skip-prompt install --amd                 # AMD（ROCm，Linux）
comfy --skip-prompt install --m-series            # Apple Silicon（MPS）
comfy --skip-prompt install --cpu                 # 仅 CPU（较慢）
comfy --skip-prompt install --nvidia --fast-deps  # 基于 uv 的依赖解析
```

默认位置：`~/comfy/ComfyUI`（Linux），`~/Documents/comfy/ComfyUI`（macOS/Win）。使用 `comfy --workspace /custom/path install` 覆盖。

#### 启动 / 验证

```bash
comfy launch --background                       # 后台守护进程，端口 :8188
comfy launch -- --listen 0.0.0.0 --port 8190    # 局域网可访问的自定义端口
curl -s http://127.0.0.1:8188/system_stats      # 健康检查
```

---

### 路径 E：手动安装（高级 / 不支持的硬件）

适用于昇腾 NPU、寒武纪 MLU、Intel Arc 或其他不支持的硬件。

**文档：** https://docs.comfy.org/installation/manual_install

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python main.py
```

---

### 安装后：下载模型

```bash
# SDXL（通用，约 6.5 GB）
comfy model download \
  --url "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  --relative-path models/checkpoints

# SD 1.5（更轻量，约 4 GB，适合 6 GB 显卡）
comfy model download \
  --url "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
  --relative-path models/checkpoints

# Flux Dev fp8（较小变体，约 12 GB）
comfy model download \
  --url "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
  --relative-path models/checkpoints

# CivitAI（先设置 token）：
comfy model download \
  --url "https://civitai.com/api/download/models/128713" \
  --relative-path models/checkpoints \
  --set-civitai-api-token "YOUR_TOKEN"
```

列出已安装：`comfy model list`。

### 安装后：安装自定义节点

```bash
comfy node install comfyui-impact-pack             # 常用工具包
comfy node install comfyui-animatediff-evolved     # 视频生成
comfy node install comfyui-controlnet-aux          # ControlNet 预处理器
comfy node install comfyui-essentials              # 常用辅助工具
comfy node update all
comfy node install-deps --workflow=workflow.json   # 安装工作流所需的全部内容
```

### 安装后：验证

```bash
python3 scripts/health_check.py
# → comfy_cli 在 PATH 中？服务器可达？有 checkpoint？冒烟测试？

python3 scripts/check_deps.py my_workflow.json
# → 此工作流的节点/模型/embedding 是否已安装？

python3 scripts/run_workflow.py \
  --workflow workflows/sd15_txt2img.json \
  --args '{"prompt": "test", "steps": 4}' \
  --output-dir ./test-outputs
```

## 图像上传（img2img / Inpainting）

最简单的方式是在 `run_workflow.py` 中使用 `--input-image`：

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it cyberpunk", "denoise": 0.6}'
```

该标志上传 `photo.png`，然后将其服务端文件名注入到 schema 中名为 `image` 的参数。对于 inpainting，同时传入：

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_inpaint.json \
  --input-image image=./photo.png \
  --input-image mask_image=./mask.png \
  --args '{"prompt": "fill with flowers"}'
```

通过 REST 手动上传：
```bash
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
# 返回：{"name": "photo.png", "subfolder": "", "type": "input"}

# 云端等效：
curl -X POST "https://cloud.comfy.org/api/upload/image" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
```

## 云端特性

- **Base URL：** `https://cloud.comfy.org`
- **认证：** `X-API-Key` 请求头（WebSocket 使用 `?token=KEY`）
- **API key：** 设置一次 `$COMFY_CLOUD_API_KEY`，脚本自动读取
- **输出下载：** `/api/view` 返回 302 跳转至签名 URL；脚本会跟随跳转并在从存储后端（S3/CloudFront）获取前去除 `X-API-Key`（避免泄露 API key）。
- **与本地 ComfyUI 的端点差异：**
  - `/api/object_info`、`/api/queue`、`/api/userdata` — **免费层返回 403**；仅付费可用。
  - `/history` 在云端重命名为 `/history_v2`（脚本自动路由）。
  - `/models/<folder>` 在云端重命名为 `/experiment/models/<folder>`（脚本自动路由）。
  - WebSocket 中的 `clientId` 目前被忽略——同一用户的所有连接接收相同广播。请在客户端按 `prompt_id` 过滤。
  - 上传时接受 `subfolder` 但会被忽略——云端使用扁平命名空间。
- **并发任务：** 免费/标准版：1，Creator：3，Pro：5。超出部分自动排队。使用 `run_batch.py --parallel N` 充分利用你的套餐层级。

## 队列与系统管理

```bash
# 本地
curl -s http://127.0.0.1:8188/queue | python3 -m json.tool
curl -X POST http://127.0.0.1:8188/queue -d '{"clear": true}'    # 取消待处理任务
curl -X POST http://127.0.0.1:8188/interrupt                      # 取消运行中任务
curl -X POST http://127.0.0.1:8188/free \
  -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'

# 云端——相同路径加 /api/ 前缀，另外：
python3 scripts/fetch_logs.py --tail-queue --host https://cloud.comfy.org
```

## 常见问题

1. **必须使用 API 格式** — 所有脚本和 `/api/prompt` 端点均需要 API 格式的工作流 JSON。脚本会检测编辑器格式（顶层含 `nodes` 和 `links` 数组）并提示通过"Workflow → Export (API)"（新版 UI）或"Save (API Format)"（旧版 UI）重新导出。

2. **服务器必须运行** — 所有执行操作都需要运行中的服务器。`comfy launch --background` 可启动服务器。通过 `curl http://127.0.0.1:8188/system_stats` 验证。

3. **模型名称必须精确** — 区分大小写，包含文件扩展名。`check_deps.py` 会进行模糊匹配（含/不含扩展名和文件夹前缀），但工作流本身必须使用规范名称。使用 `comfy model list` 查看已安装内容。

4. **缺少自定义节点** — "class_type not found" 表示所需节点未安装。`check_deps.py` 会报告需要安装哪个包；`auto_fix_deps.py` 会自动执行安装。

5. **工作目录** — `comfy-cli` 会自动检测 ComfyUI workspace。如果命令报错"no workspace found"，请使用 `comfy --workspace /path/to/ComfyUI <command>` 或 `comfy set-default /path/to/ComfyUI`。

6. **云端免费层 API 限制** — `/api/prompt`、`/api/view`、`/api/upload/*`、`/api/object_info` 在免费账户上均返回 403。`health_check.py` 和 `check_deps.py` 会优雅处理此情况并显示清晰提示。

7. **视频/音频工作流超时** — 当输出节点为 `VHS_VideoCombine`、`SaveVideo` 等时自动检测；默认超时从 300 秒跳至 900 秒。可通过 `--timeout 1800` 显式覆盖。

8. **输出文件名路径遍历** — 服务端提供的文件名会经过 `safe_path_join` 处理，拒绝任何试图逃出 `--output-dir` 的路径。请保留此保护——带自定义保存节点的工作流可能产生任意路径。

9. **工作流 JSON 是任意代码** — 自定义节点运行 Python，因此提交未知工作流的信任风险与 `eval` 相同。运行来自不可信来源的工作流前请先检查。

10. **自动随机化种子** — 在 `--args` 中传入 `seed: -1`（或使用 `--randomize-seed` 并省略 seed）可在每次运行时获得新种子。实际种子会记录到 stderr。

11. **`tracking` 提示** — 首次运行 `comfy` 可能会提示分析选项。使用 `comfy --skip-prompt tracking disable` 非交互式跳过。`comfyui_setup.sh` 会自动处理此问题。

## 验证清单

使用 `python3 scripts/health_check.py` 一次性运行全部检查。手动检查：

- [ ] `hardware_check.py` 结果为 `ok`，或用户明确选择了 Comfy Cloud
- [ ] `comfy --version` 可用（或 `uvx --from comfy-cli comfy --help`）
- [ ] `curl http://HOST:PORT/system_stats` 返回 JSON
- [ ] `comfy model list` 显示至少一个 checkpoint（本地），或 `/api/experiment/models/checkpoints` 返回模型（云端）
- [ ] 工作流 JSON 为 API 格式
- [ ] `check_deps.py` 报告 `is_ready: true`（或云端免费层仅显示 `node_check_skipped`）
- [ ] 用小型工作流测试运行完成；输出文件出现在 `--output-dir` 中