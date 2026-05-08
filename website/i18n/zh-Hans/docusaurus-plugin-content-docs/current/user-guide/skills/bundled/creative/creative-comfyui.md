---
title: "Comfyui"
sidebar_label: "Comfyui"
description: "使用ComfyUI生成图像、视频和音频 — 安装、启动、管理节点/模型、使用参数注入运行工作流"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Comfyui

使用ComfyUI生成图像、视频和音频 — 安装、启动、管理节点/模型、使用参数注入运行工作流。使用官方comfy-cli进行生命周期管理和直接REST/WebSocket API进行执行。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/comfyui` |
| 版本 | `5.0.0` |
| 作者 | ['kshitijk4poor', 'alt-glitch'] |
| 许可证 | MIT |
| 平台 | macos, linux, windows |
| 标签 | `comfyui`, `image-generation`, `stable-diffusion`, `flux`, `sd3`, `wan-video`, `hunyuan-video`, `creative`, `generative-ai`, `video-generation` |
| 相关技能 | [`stable-diffusion-image-generation`](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion), `image_gen` |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# ComfyUI

通过ComfyUI使用官方`comfy-cli`进行设置/生命周期管理和直接REST/WebSocket API进行工作流执行，生成图像、视频、音频和3D内容。

## 此技能包含的内容

**参考文档（`references/`）：**

- `official-cli.md` — 每个`comfy ...`命令及其标志
- `rest-api.md` — REST + WebSocket端点（本地+云），有效载荷模式
- `workflow-format.md` — API格式JSON，常见节点类型，参数映射

**脚本（`scripts/`）：**

| 脚本 | 用途 |
|------|------|
| `_common.py` | 共享HTTP、云路由、节点目录（不直接运行） |
| `hardware_check.py` | 探测GPU/VRAM/磁盘 → 推荐本地vs Comfy Cloud |
| `comfyui_setup.sh` | 硬件检查 + comfy-cli + ComfyUI安装 + 启动 + 验证 |
| `extract_schema.py` | 读取工作流 → 列出可控参数 + 模型依赖 |
| `check_deps.py` | 检查工作流与运行中服务器 → 列出缺失的节点/模型 |
| `auto_fix_deps.py` | 运行check_deps然后`comfy node install` / `comfy model download` |
| `run_workflow.py` | 注入参数、提交、监控、下载输出（HTTP或WS） |
| `run_batch.py` | 提交工作流N次并扫描，并行至您的层级限制 |
| `ws_monitor.py` | 执行作业的实时WebSocket查看器（实时进度） |
| `health_check.py` | 验证检查表运行器 — comfy-cli + 服务器 + 模型 + 冒烟测试 |
| `fetch_logs.py` | 拉取给定prompt_id的追溯/状态消息 |

**示例工作流（`workflows/`）：** SD 1.5、SDXL、Flux Dev、SDXL img2img、SDXL inpaint、ESRGAN upscale、AnimateDiff video、Wan T2V。参见`workflows/README.md`。

## 使用场景

- 用户要求使用Stable Diffusion、SDXL、Flux、SD3等生成图像
- 用户想要运行特定的ComfyUI工作流文件
- 用户想要链接生成步骤（txt2img → upscale → face restore）
- 用户需要ControlNet、inpainting、img2img或其他高级流水线
- 用户要求管理ComfyUI队列、检查模型或安装自定义节点
- 用户想要通过AnimateDiff、Hunyuan、Wan、AudioCraft等生成视频/音频/3D

## 架构：两层

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────┐
│ Layer 1: comfy-cli (官方生命周期工具)              │
│   设置、服务器生命周期、自定义节点、模型              │
│   → comfy install / launch / stop / node / model   │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│ Layer 2: REST/WebSocket API + 技能脚本              │
│   工作流执行、参数注入、监控                         │
│   POST /api/prompt, GET /api/view, WS /ws          │
│   → run_workflow.py, run_batch.py, ws_monitor.py   │
└─────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

**为什么两层？** 官方CLI在安装和服务器管理方面出色，但对工作流执行支持很少。REST/WS API填补了这一空白——脚本处理CLI不做的参数注入、执行监控和输出下载。

## 快速开始

### 检测环境

```bash
# 有什么可用？
command -v comfy >/dev/null 2>&1 && echo "comfy-cli: installed"
curl -s http://127.0.0.1:8188/system_stats 2>/dev/null && echo "server: running"

# 这台机器能本地运行ComfyUI吗？（GPU/VRAM/磁盘检查）
python3 scripts/hardware_check.py
```

如果什么都没有安装，参见下面的**设置与入门**——但始终先运行硬件检查。

### 一键健康检查

```bash
python3 scripts/health_check.py
# → JSON: comfy_cli在PATH上？服务器可达？至少一个checkpoint？冒烟测试通过？
```

## 核心工作流

### 第1步：获取API格式的工作流JSON

工作流必须是API格式（每个节点有`class_type`）。来源：

- ComfyUI web UI → **Workflow → Export (API)**（较新UI）或旧版"Save (API Format)"按钮（旧UI）
- 此技能的`workflows/`目录（可直接运行的示例）
- 社区下载（civitai、Reddit、Discord）——通常为编辑器格式，必须加载到ComfyUI然后重新导出

编辑器格式（顶级`nodes`和`links`数组）**不能直接执行**。脚本检测到这一点并告诉您重新导出。

### 第2步：查看可控制内容

```bash
python3 scripts/extract_schema.py workflow_api.json --summary-only
# → {"parameter_count": 12, "has_negative_prompt": true, "has_seed": true, ...}

python3 scripts/extract_schema.py workflow_api.json
# → 完整模式，包含参数、模型依赖、embedding引用
```

### 第3步：带参数运行

```bash
# 本地（默认为http://127.0.0.1:8188）
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset over mountains", "seed": -1, "steps": 30}' \
  --output-dir ./outputs

# 云（导出一次API密钥；自动使用正确的/api路由）
export COMFY_CLOUD_API_KEY="comfyui-..."
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "..."}' \
  --host https://cloud.comfy.org \
  --output-dir ./outputs

# 通过WebSocket实时进度（需要`pip install websocket-client`）
python3 scripts/run_workflow.py \
  --workflow flux_dev.json \
  --args '{"prompt": "..."}' \
  --ws

# img2img / inpaint：传递--input-image自动上传+引用
python3 scripts/run_workflow.py \
  --workflow sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it watercolor", "denoise": 0.6}'

# 批量/扫描：8个随机种子，并行至云层级限制
python3 scripts/run_batch.py \
  --workflow sdxl.json \
  --args '{"prompt": "abstract"}' \
  --count 8 --randomize-seed --parallel 3 \
  --output-dir ./outputs/batch
```

`-1`表示`seed`（或使用`--randomize-seed`省略）每次运行生成新的随机种子。

### 第4步：呈现结果

脚本向stdout发出描述每个输出文件的JSON：

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
|--------|------|------|
| **生命周期（使用comfy-cli）** | | |
| "安装ComfyUI" | comfy-cli | `bash scripts/comfyui_setup.sh` |
| "启动ComfyUI" | comfy-cli | `comfy launch --background` |
| "停止ComfyUI" | comfy-cli | `comfy stop` |
| "安装X节点" | comfy-cli | `comfy node install <name>` |
| "下载X模型" | comfy-cli | `comfy model download --url <url> --relative-path models/checkpoints` |
| "列出已安装模型" | comfy-cli | `comfy model list` |
| "列出已安装节点" | comfy-cli | `comfy node show installed` |
| **执行（使用脚本）** | | |
| "一切都准备好了吗？" | 脚本 | `health_check.py`（可选带`--workflow X --smoke-test`） |
| "这个工作流我可以改什么？" | 脚本 | `extract_schema.py W.json` |
| "检查W的依赖是否满足" | 脚本 | `check_deps.py W.json` |
| "修复缺失的依赖" | 脚本 | `auto_fix_deps.py W.json` |
| "生成图像" | 脚本 | `run_workflow.py --workflow W --args '{...}'` |
| "使用此图像"（img2img） | 脚本 | `run_workflow.py --input-image image=./x.png ...` |
| "8个随机种子变体" | 脚本 | `run_batch.py --count 8 --randomize-seed ...` |
| "显示实时进度" | 脚本 | `ws_monitor.py --prompt-id <id>` |
| "获取作业X的错误" | 脚本 | `fetch_logs.py <prompt_id>` |
| **直接REST** | | |
| "队列里有什么？" | REST | `curl http://HOST:8188/queue`（本地）或`--host https://cloud.comfy.org` |
| "取消那个" | REST | `curl -X POST http://HOST:8188/interrupt` |
| "释放GPU内存" | REST | `curl -X POST http://HOST:8188/free` |

## 设置与入门

当用户要求设置ComfyUI时，**首先要做的是询问他们想要Comfy Cloud（托管，零安装，需要API密钥）还是本地（在他们的机器上安装ComfyUI）**。在他们回答之前不要开始运行安装命令或硬件检查。

**官方文档：** https://docs.comfy.org/installation
**CLI文档：** https://docs.comfy.org/comfy-cli/getting-started
**云文档：** https://docs.comfy.org/get_started/cloud
**云API：** https://docs.comfy.org/development/cloud/overview

### 第0步：询问本地vs云（始终首先）

建议脚本：

> "您想在本地机器上运行ComfyUI，还是使用Comfy Cloud？
>
> - **Comfy Cloud** — 托管在RTX 6000 Pro GPU上，所有常见模型预装，零设置。需要API密钥（实际运行工作流需要付费订阅；免费层级仅可浏览）。如果您没有高性能GPU，最佳选择。
> - **本地** — 免费，但您的机器**必须**满足硬件要求：
>   - NVIDIA GPU，**≥6 GB VRAM**（SDXL≥8 GB，Flux/视频≥12 GB），或
>   - AMD GPU，支持ROCm（Linux），或
>   - Apple Silicon Mac（M1+），**≥16 GB统一内存**（≥32 GB推荐）。
>   - Intel Mac和无GPU机器**无法工作** — 请使用云端。
>
> 您想要哪个？"

路由：

- **云** → 跳至**路径A**。
- **本地** → 首先运行硬件检查，然后根据判断从路径B-E中选择。
- **不确定** → 运行硬件检查，让判断决定。

### 第1步：验证硬件（仅在用户选择本地时）

```bash
python3 scripts/hardware_check.py --json
# 可选：也探测`torch`以获取实际CUDA/MPS：
python3 scripts/hardware_check.py --json --check-pytorch
```

| 判断 | 含义 | 操作 |
|------|------|------|
| `ok` | ≥8 GB VRAM（独立）或≥32 GB统一内存（Apple Silicon） | 本地安装 — 使用报告中的`comfy_cli_flag` |
| `marginal` | SD1.5可用；SDXL紧张；Flux/视频不太可能 | 轻度工作流可本地，否则**路径A（云）** |
| `cloud` | 无可用GPU、<6 GB VRAM、<16 GB Apple统一内存、Intel Mac、Rosetta Python | **切换到云**，除非用户明确强制本地 |

脚本还显示`wsl: true`（WSL2 + NVIDIA直通）和`rosetta: true`（Apple Silicon上的x86_64 Python — 必须重新安装为ARM64）。

如果判断为`cloud`但用户想要本地，不要沉默地进行。逐字显示`notes`数组，然后询问他们是想(a)切换到云还是(b)强制本地安装（会在现代模型上OOM或慢得无法使用）。

### 选择安装路径

首先运行硬件检查。当用户已经告诉您硬件时，下表作为后备：

| 情况 | 推荐路径 |
|------|----------|
| 硬件检查`verdict: cloud` | **路径A：Comfy Cloud** |
| 无GPU / 想试试不承诺 | **路径A：Comfy Cloud** |
| Windows + NVIDIA + 非技术 | **路径B：ComfyUI Desktop** |
| Windows + NVIDIA + 技术 | **路径C：Portable**或**路径D：comfy-cli** |
| Linux + 任何GPU | **路径D：comfy-cli**（最简单） |
| macOS + Apple Silicon | **路径B：Desktop**或**路径D：comfy-cli** |
| 无头/服务器/CI/代理 | **路径D：comfy-cli** |

对于完全自动化路径（硬件检查 → 安装 → 启动 → 验证）：

```bash
bash scripts/comfyui_setup.sh
# 或带覆盖：
bash scripts/comfyui_setup.sh --m-series --port=8190 --workspace=/data/comfy
```

它在内部运行`hardware_check.py`，当判断为`cloud`时拒绝本地安装（除非`--force-cloud-override`），选择正确的`comfy-cli`标志，并优先使用`pipx`/`uvx`而不是全局`pip`以避免污染系统Python。

---

### 路径A：Comfy Cloud（无本地安装）

适用于没有高性能GPU或希望零设置的用户。托管在RTX 6000 Pro上。

**文档：** https://docs.comfy.org/get_started/cloud

1. 在 https://comfy.org/cloud 注册
2. 在 https://platform.comfy.org/login 生成API密钥
3. 设置密钥：
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
**并发作业：** Free/Standard 1、Creator 3、Pro 5。免费层级**无法通过API运行工作流** — 仅可浏览模型。`/api/prompt`、`/api/upload/*`、`/api/view`等需要付费订阅。

---

### 路径B：ComfyUI Desktop（Windows / macOS）

一键安装程序，面向非技术用户。目前处于Beta。

**文档：** https://docs.comfy.org/installation/desktop
- **Windows（NVIDIA）：** https://download.comfy.org/windows/nsis/x64
- **macOS（Apple Silicon）：** https://comfy.org

Linux**不支持Desktop** — 使用路径D。

---

### 路径C：ComfyUI Portable（仅Windows）

**文档：** https://docs.comfy.org/installation/comfyui_portable_windows

从 https://github.com/comfyanonymous/ComfyUI/releases 下载，解压，运行`run_nvidia_gpu.bat`。通过`update/update_comfyui_stable.bat`更新。

---

### 路径D：comfy-cli（所有平台 — 推荐用于代理）

官方CLI是headless/自动化设置的最佳路径。

**文档：** https://docs.comfy.org/comfy-cli/getting-started

#### 安装comfy-cli

```bash
# 推荐：
pipx install comfy-cli
# 或使用uvx不安装：
uvx --from comfy-cli comfy --help
# 或（如果pipx/uvx不可用）：
pip install --user comfy-cli
```

非交互式禁用分析：
```bash
comfy --skip-prompt tracking disable
```

#### 安装ComfyUI

```bash
comfy --skip-prompt install --nvidia              # NVIDIA (CUDA)
comfy --skip-prompt install --amd                 # AMD (ROCm, Linux)
comfy --skip-prompt install --m-series            # Apple Silicon (MPS)
comfy --skip-prompt install --cpu                 # 仅CPU（慢）
comfy --skip-prompt install --nvidia --fast-deps  # 基于uv的依赖解析
```

默认位置：`~/comfy/ComfyUI`（Linux）、`~/Documents/comfy/ComfyUI`（macOS/Win）。通过`comfy --workspace /custom/path install`覆盖。

#### 启动/验证

```bash
comfy launch --background                       # 后台守护进程在:8188
comfy launch -- --listen 0.0.0.0 --port 8190    # LAN可访问的自定义端口
curl -s http://127.0.0.1:8188/system_stats      # 健康检查
```

---

### 路径E：手动安装（高级/不支持的硬件）

适用于Ascend NPU、Cambricon MLU、Intel Arc或其他不支持的硬件。

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
# SDXL（通用，约6.5 GB）
comfy model download \
  --url "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  --relative-path models/checkpoints

# SD 1.5（较轻，约4 GB，适合6 GB显卡）
comfy model download \
  --url "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
  --relative-path models/checkpoints

# Flux Dev fp8（较小变体，约12 GB）
comfy model download \
  --url "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
  --relative-path models/checkpoints

# CivitAI（先设置token）：
comfy model download \
  --url "https://civitai.com/api/download/models/128713" \
  --relative-path models/checkpoints \
  --set-civitai-api-token "YOUR_TOKEN"
```

列出已安装：`comfy model list`。

### 安装后：安装自定义节点

```bash
comfy node install comfyui-impact-pack             # 流行实用工具包
comfy node install comfyui-animatediff-evolved     # 视频生成
comfy node install comfyui-controlnet-aux          # ControlNet预处理器
comfy node install comfyui-essentials              # 常见辅助函数
comfy node update all
comfy node install-deps --workflow=workflow.json   # 安装工作流所需的一切
```

### 安装后：验证

```bash
python3 scripts/health_check.py
# → comfy_cli在PATH上？服务器可达？checkpoints？冒烟测试？

python3 scripts/check_deps.py my_workflow.json
# → 此工作流的节点/模型/embeddings已安装？

python3 scripts/run_workflow.py \
  --workflow workflows/sd15_txt2img.json \
  --args '{"prompt": "test", "steps": 4}' \
  --output-dir ./test-outputs
```

## 图像上传（img2img / Inpainting）

最简单的方法是使用`run_workflow.py`的`--input-image`：

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it cyberpunk", "denoise": 0.6}'
```

此标志上传`photo.png`，然后将其服务器端文件名注入到名为`image`的模式参数中。对于inpainting，同时传递：

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_inpaint.json \
  --input-image image=./photo.png \
  --input-image mask_image=./mask.png \
  --args '{"prompt": "fill with flowers"}'
```

通过REST手动上传：
```bash
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
# 返回: {"name": "photo.png", "subfolder": "", "type": "input"}

# 云端等效：
curl -X POST "https://cloud.comfy.org/api/upload/image" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
```

## 云端特定

- **基础URL：** `https://cloud.comfy.org`
- **认证：** `X-API-Key`头（或`?token=KEY`用于WebSocket）
- **API密钥：** 设置一次`$COMFY_CLOUD_API_KEY`，脚本自动拾取
- **输出下载：** `/api/view`返回到签名URL的302；脚本跟随它并在从存储后端获取前剥离`X-API-Key`（不要向S3/CloudFront泄露API密钥）。
- **与本地ComfyUI的端点差异：**
  - `/api/object_info`、`/api/queue`、`/api/userdata` — 免费层级**403**；仅付费。
  - `/history`在云上重命名为`/history_v2`（脚本自动路由）。
  - `/models/<folder>`在云上重命名为`/experiment/models/<folder>`（脚本自动路由）。
  - WebSocket中的`clientId`当前被忽略 — 用户的所有连接收到相同的广播。客户端按`prompt_id`过滤。
  - 上传接受`subfolder`但被忽略 — 云有平面命名空间。
- **并发作业：** Free/Standard: 1、Creator: 3、Pro: 5。额外队列自动。使用`run_batch.py --parallel N`饱和您的层级。

## 队列与系统管理

```bash
# 本地
curl -s http://127.0.0.1:8188/queue | python3 -m json.tool
curl -X POST http://127.0.0.1:8188/queue -d '{"clear": true}'    # 取消待处理
curl -X POST http://127.0.0.1:8188/interrupt                      # 取消运行中
curl -X POST http://127.0.0.1:8188/free \
  -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'

# 云 — 在/api/下相同路径，加上：
python3 scripts/fetch_logs.py --tail-queue --host https://cloud.comfy.org
```

## 陷阱

1. **需要API格式** — 每个脚本和`/api/prompt`端点都需要API格式的工作流JSON。脚本检测到编辑器格式（顶级`nodes`和`links`数组）并告诉您通过"Workflow → Export (API)"（较新UI）或"Save (API Format)"（旧UI）重新导出。

2. **服务器必须运行** — 所有执行都需要一个实时服务器。
   `comfy launch --background`启动一个。通过
   `curl http://127.0.0.1:8188/system_stats`验证。

3. **模型名称是精确的** — 区分大小写，包含文件扩展名。
   `check_deps.py`做模糊匹配（有/无扩展名和文件夹前缀），但工作流本身必须使用规范名称。使用`comfy model list`发现已安装的内容。

4. **缺失自定义节点** — "class_type not found"意味着所需节点未安装。`check_deps.py`报告需要安装哪个包；
   `auto_fix_deps.py`为您运行安装。

5. **工作目录** — `comfy-cli`自动检测ComfyUI工作区。
   如果命令失败并显示"no workspace found"，使用
   `comfy --workspace /path/to/ComfyUI <command>`或
   `comfy set-default /path/to/ComfyUI`。

6. **云免费层级API限制** — `/api/prompt`、`/api/view`、`/api/upload/*`、
   `/api/object_info`在免费账户上都返回403。`health_check.py`和
   `check_deps.py`优雅地处理此问题并显示清晰消息。

7. **视频/音频工作流超时** — 当输出节点为`VHS_VideoCombine`、`SaveVideo`等时自动检测；默认值从300秒跳至900秒。用`--timeout 1800`显式覆盖。

8. **输出文件名中的路径遍历** — 服务器提供的文件名通过`safe_path_join`传递，拒绝任何超出`--output-dir`的内容。
   保留此保护 — 带有自定义保存节点的工作流可能产生任意路径。

9. **工作流JSON是任意代码** — 自定义节点运行Python，因此提交未知工作流与`eval`具有相同的信任配置文件。运行前检查来自不信任来源的工作流。

10. **自动随机种子** — 在`--args`中传递`seed: -1`（或使用
    `--randomize-seed`并省略种子）每次运行获取新种子。
    实际种子记录到stderr。

11. **`tracking`提示** — `comfy`首次运行可能提示分析。
    使用`comfy --skip-prompt tracking disable`非交互式跳过。
    `comfyui_setup.sh`为您执行此操作。

## 验证清单

使用`python3 scripts/health_check.py`一次运行整个列表。手动：

- [ ] `hardware_check.py`判断为`ok`或用户明确选择了Comfy Cloud
- [ ] `comfy --version`可用（或`uvx --from comfy-cli comfy --help`）
- [ ] `curl http://HOST:PORT/system_stats`返回JSON
- [ ] `comfy model list`显示至少一个checkpoint（本地）或
      `/api/experiment/models/checkpoints`返回模型（云）
- [ ] 工作流JSON为API格式
- [ ] `check_deps.py`报告`is_ready: true`（或在云免费层级上只有`node_check_skipped`）
- [ ] 使用小型工作流的测试运行完成；输出进入`--output-dir`
