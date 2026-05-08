---
title: "Touchdesigner Mcp"
sidebar_label: "Touchdesigner Mcp"
description: "通过 twozero MCP 控制正在运行的 TouchDesigner 实例 — 创建算子、设置参数、连接线路、执行 Python、构建实时视觉效果"
---

 {/* 此页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Touchdesigner Mcp

通过 twozero MCP 控制正在运行的 TouchDesigner 实例 — 创建算子、设置参数、连接线路、执行 Python、构建实时视觉效果。36 个原生工具。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/touchdesigner-mcp` |
| 版本 | `1.1.0` |
| 作者 | kshitijk4poor |
| 许可证 | MIT |
| 标签 | `TouchDesigner`, `MCP`, `twozero`, `creative-coding`, `real-time-visuals`, `generative-art`, `audio-reactive`, `VJ`, `installation`, `GLSL` |
| 相关技能 | [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp), [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), `hermes-video` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时智能体看到的指令。
:::

# TouchDesigner 集成（twozero MCP）

## 关键规则

1. **绝不猜测参数名。** 先调用 `td_get_par_info` 获取 OP 类型。您的训练数据对于 TD 2025.32 是错误的。
2. **如果触发 `tdAttributeError`，立即停止。** 在继续之前对故障节点调用 `td_get_operator_info`。
3. **绝不硬编码脚本回调中的绝对路径。** 使用 `me.parent()` / `scriptOp.parent()`。
4. **优先使用原生 MCP 工具而非 td_execute_python。** 使用 `td_create_operator`、`td_set_operator_pars`、`td_get_errors` 等。仅在复杂多步逻辑时回退到 `td_execute_python`。
5. **在构建前调用 `td_get_hints`。** 它返回您正在处理的 OP 类型的特定模式。

## 架构

```
Hermes Agent -> MCP (Streamable HTTP) -> twozero.tox (端口 40404) -> TD Python
```

36 个原生工具。免费插件（无需付款/许可 — 已确认 2026 年 4 月）。
上下文感知（了解选定的 OP、当前网络）。
中心健康检查：`GET http://localhost:40404/mcp` 返回包含实例 PID、项目名称、TD 版本的 JSON。

## 设置（自动化）

运行设置脚本处理一切：

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/skills/creative/touchdesigner-mcp/scripts/setup.sh"
```

脚本将：
1. 检查 TD 是否正在运行
2. 如果未缓存，下载 twozero.tox
3. 将 `twozero_td` MCP 服务器添加到 Hermes 配置（如果缺失）
4. 测试端口 40404 上的 MCP 连接
5. 报告剩余的手动步骤（将 .tox 拖入 TD、启用 MCP 切换）

### 手动步骤（一次性，无法自动化）

1. **将 `~/Downloads/twozero.tox` 拖入 TD 网络编辑器** → 点击安装
2. **启用 MCP：** 点击 twozero 图标 → 设置 → mcp → "自动启动 MCP" → 是
3. **重启 Hermes 会话** 以加载新的 MCP 服务器

设置后验证：
```bash
nc -z 127.0.0.1 40404 && echo "twozero MCP: READY"
```

## 环境说明

- **非商业版 TD** 分辨率上限为 1280×1280。使用 `outputresolution = 'custom'` 并显式设置宽度/高度。
- **编解码器：** macOS 上首选 `prores`，备选 `mjpa`。H.264/H.265/AV1 需要商业许可。
- 设置参数前始终调用 `td_get_par_info` — 名称因 TD 版本而异（见关键规则 #1）。

## 工作流程

### 第 0 步：发现（在构建任何东西之前）

```
调用 td_get_par_info 获取您计划使用的每个类型的 op_type。
调用 td_get_hints 获取您正在构建的主题（如 "glsl"、"audio reactive"、"feedback"）。
调用 td_get_focus 查看用户所在位置和已选择的内容。
调用 td_get_network 查看已存在的内容。
```

无需临时节点，无需清理。这完全取代了旧的发现流程。

### 第 1 步：清理 + 构建

**重要：将清理和创建分为单独的两个 MCP 调用。** 在一个 `td_execute_python` 脚本中销毁和重建同名节点会导致"无效 OP 对象"错误。见陷阱 #11b。

使用 `td_create_operator` 创建每个节点（自动处理视口定位）：

```
td_create_operator(type="noiseTOP", parent="/project1", name="bg", parameters={"resolutionw": 1280, "resolutionh": 720})
td_create_operator(type="levelTOP", parent="/project1", name="brightness")
td_create_operator(type="nullTOP", parent="/project1", name="out")
```

对于批量创建或连接，使用 `td_execute_python`：

```python
# td_execute_python 脚本：
root = op('/project1')
nodes = []
for name, optype in [('bg', noiseTOP), ('fx', levelTOP), ('out', nullTOP)]:
    n = root.create(optype, name)
    nodes.append(n.path)
# 连接链
for i in range(len(nodes)-1):
    op(nodes[i]).outputConnectors[0].connect(op(nodes[i+1]).inputConnectors[0])
result = {'created': nodes}
```

### 第 2 步：设置参数

优先使用原生工具（验证参数，不会崩溃）：

```
td_set_operator_pars(path="/project1/bg", parameters={"roughness": 0.6, "monochrome": true})
```

对于表达式或模式，使用 `td_execute_python`：

```python
op('/project1/time_driver').par.colorr.expr = "absTime.seconds % 1000.0"
```

### 第 3 步：连接

使用 `td_execute_python` — 没有原生连接工具：

```python
op('/project1/bg').outputConnectors[0].connect(op('/project1/fx').inputConnectors[0])
```

### 第 4 步：验证

```
td_get_errors(path="/project1", recursive=true)
td_get_perf()
td_get_operator_info(path="/project1/out", detail="full")
```

### 第 5 步：显示/捕获

```
td_get_screenshot(path="/project1/out")
```

或通过脚本打开窗口：

```python
win = op('/project1').create(windowCOMP, 'display')
win.par.winop = op('/project1/out').path
win.par.winw = 1280; win.par.winh = 720
win.par.winopen.pulse()
```

## MCP 工具快速参考

**核心（最常用）：**
| 工具 | 功能 |
|------|------|
| `td_execute_python` | 在 TD 中运行任意 Python。完整 API 访问。 |
| `td_create_operator` | 创建带参数的节点 + 自动定位 |
| `td_set_operator_pars` | 安全设置参数（验证，不会崩溃） |
| `td_get_operator_info` | 检查一个节点：连接、参数、错误 |
| `td_get_operators_info` | 在一次调用中检查多个节点 |
| `td_get_network` | 查看路径处的网络结构 |
| `td_get_errors` | 递归查找错误/警告 |
| `td_get_par_info` | 获取 OP 类型的参数名（取代发现流程） |
| `td_get_hints` | 构建前获取模式/提示 |
| `td_get_focus` | 打开哪个网络，选择了什么 |

**读取/写入：**
| 工具 | 功能 |
|------|------|
| `td_read_dat` | 读取 DAT 文本内容 |
| `td_write_dat` | 写入/修补 DAT 内容 |
| `td_read_chop` | 读取 CHOP 通道值 |
| `td_read_textport` | 读取 TD 控制台输出 |

**视觉：**
| 工具 | 功能 |
|------|------|
| `td_get_screenshot` | 将一个 OP 查看器捕获到文件 |
| `td_get_screenshots` | 一次捕获多个 OP |
| `td_get_screen_screenshot` | 通过 TD 捕获实际屏幕 |
| `td_navigate_to` | 跳转到网络编辑器中的 OP |

**搜索：**
| 工具 | 功能 |
|------|------|
| `td_find_op` | 在项目中按名称/类型查找 op |
| `td_search` | 搜索代码、表达式、字符串参数 |

**系统：**
| 工具 | 功能 |
|------|------|
| `td_get_perf` | 性能分析（FPS、慢速 op） |
| `td_list_instances` | 列出所有运行的 TD 实例 |
| `td_get_docs` | 关于 TD 主题的深入文档 |
| `td_agents_md` | 读取/写入每个 COMP 的 markdown 文档 |
| `td_reinit_extension` | 编辑代码后重新加载扩展 |
| `td_clear_textport` | 在调试会话前清除控制台 |

**输入自动化：**
| 工具 | 功能 |
|------|------|
| `td_input_execute` | 发送鼠标/键盘到 TD |
| `td_input_status` | 轮询输入队列状态 |
| `td_input_clear` | 停止输入自动化 |
| `td_op_screen_rect` | 获取节点的屏幕坐标 |
| `td_click_screen_point` | 点击截图中的一个点 |
| `td_screen_point_to_global` | 将截图像素转换为绝对屏幕坐标 |

上表涵盖了典型创意工作流中使用的 32 个工具。其余 4 个工具（`td_project_quit`、`td_test_session`、`td_dev_log`、`td_clear_dev_log`）是管理/开发模式工具 — 完整的 36 工具参考及完整参数模式见 `references/mcp-tools.md`。

## 关键实现规则

**GLSL 时间：** GLSL TOP 中没有 `uTDCurrentTime`。使用 Values 页面：
```python
# 首先调用 td_get_par_info(op_type="glslTOP") 确认参数名
td_set_operator_pars(path="/project1/shader", parameters={"value0name": "uTime"})
# 然后通过脚本设置表达式：
# op('/project1/shader').par.value0.expr = "absTime.seconds"
# 在 GLSL 中：uniform float uTime;
```

备选方案：使用 `rgba32float` 格式的 Constant TOP（8 位限制为 0-1，会冻结着色器）。

**Feedback TOP：** 使用 `top` 参数引用，而非直接输入线。"源不足"在首次 cook 后解决。"Cook 依赖循环"警告是预期的。

**分辨率：** 非商业版上限为 1280×1280。使用 `outputresolution = 'custom'`。

**大型着色器：** 将 GLSL 写入 `/tmp/file.glsl`，然后使用 `td_write_dat` 或 `td_execute_python` 加载。

**顶点/点访问（TD 2025.32）：** `point.P[0]`, `point.P[1]`, `point.P[2]` — 不是 `.x`, `.y`, `.z`。

**扩展：** 扩展格式为 `"op('./datName').module.ClassName(me)"`（CONSTANT 模式）。使用 `td_write_dat` 编辑扩展代码后，调用 `td_reinit_extension`。

**脚本回调：** 始终使用相对路径通过 `me.parent()` / `scriptOp.parent()`。

**清理节点：** 在遍历之前始终 `list(root.children)` 和 `child.valid` 检查。

## 录制/导出视频

```python
# 通过 td_execute_python：
root = op('/project1')
rec = root.create(moviefileoutTOP, 'recorder')
op('/project1/out').outputConnectors[0].connect(rec.inputConnectors[0])
rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'prores'  # Apple ProRes — macOS 上无需许可
rec.par.record = True   # 开始
# rec.par.record = False  # 停止（稍后单独调用）
```

H.264/H.265/AV1 需要商业许可。macOS 上使用 `prores` 或备选 `mjpa`。
提取帧：`ffmpeg -i /tmp/output.mov -vframes 120 /tmp/frames/frame_%06d.png`

**TOP.save() 对动画无用** — 每次捕获相同的 GPU 纹理。始终使用 MovieFileOut。

### 录制前：检查清单

1. **验证 FPS > 0** 通过 `td_get_perf`。如果 FPS=0，录制将为空。见陷阱 #38-39。
2. **验证着色器输出不是黑色** 通过 `td_get_screenshot`。黑色输出 = 着色器错误或缺少输入。见陷阱 #8, #40。
3. **如果录制带音频：** 首先将音频预热，然后延迟 3 帧开始录制。见陷阱 #19。
4. **在开始录制前设置输出路径** — 在同一脚本中设置两者可能导致竞态。

## 音频反应式 GLSL（经验证配方）

### 正确的信号链（2026 年 4 月测试）

```
AudioFileIn CHOP (playmode=sequential)
  → AudioSpectrum CHOP (FFT=512, outputmenu=setmanually, outlength=256, timeslice=ON)
  → Math CHOP (gain=10)
  → CHOP to TOP (dataformat=r, layout=rowscropped)
  → GLSL TOP 输入 1（频谱纹理，256x2）

Constant TOP (rgba32float, time) → GLSL TOP 输入 0
GLSL TOP → Null TOP → MovieFileOut
```

### 关键音频反应规则（经验证）

1. **TimeSlice 必须保持 ON** 用于 AudioSpectrum。关闭 = 处理整个音频文件 → 24000+ 样本 → CHOP to TOP 溢出。
2. **手动设置输出长度** 为 256，通过 `outputmenu='setmanally'` 和 `outlength=256`。默认输出 22050 样本。
3. **不要使用 Lag CHOP 进行频谱平滑。** Lag CHOP 在时间片模式下运行，将 256 样本扩展到 2400+，将所有值平均到接近零（~1e-06）。着色器接收不到可用数据。这是测试中音频同步失败的首要原因。
4. **也不要使用 Filter CHOP** — 对频谱数据存在相同的时间片扩展问题。
5. **平滑应在 GLSL 着色器中进行**（如需要），通过带反馈纹理的临时 lerp：`mix(prevValue, newValue, 0.3)`。这提供帧精确同步且零管道延迟。
6. **CHOP to TOP dataformat = 'r'**，layout = 'rowscropped'。频谱输出为 256x2（立体声）。在 y=0.25 处采样第一个通道。
7. **Math gain = 10**（不是 5）。原始频谱值在低音范围约为 0.19。增益 10 在着色器中给出可用的约 5.0。
8. **不需要 Resample CHOP。** 通过 AudioSpectrum 的 `outlength` 参数直接控制输出大小。

### GLSL 频谱采样

```glsl
// 输入 0 = 时间 (1x1 rgba32float), 输入 1 = 频谱 (256x2)
float iTime = texture(sTD2DInputs[0], vec2(0.5)).r;

// 对每个频段采样多个点并取平均以保持稳定：
// 注意：y=0.25 为第一个通道（立体声纹理是 256x2，第一行中心是 0.25）
float bass = (texture(sTD2DInputs[1], vec2(0.02, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.05, 0.25)).r) / 2.0;
float mid  = (texture(sTD2DInputs[1], vec2(0.2, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.35, 0.25)).r) / 2.0;
float hi   = (texture(sTD2DInputs[1], vec2(0.6, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.8, 0.25)).r) / 2.0;
```

完整构建脚本和着色器代码见 `references/network-patterns.md`。

## 算子快速参考

| 家族 | 颜色 | Python 类 / MCP 类型 | 后缀 |
|--------|-------|-------------|--------|
| TOP | 紫色 | noiseTOP, glslTOP, compositeTOP, levelTop, blurTOP, textTOP, nullTOP | TOP |
| CHOP | 绿色 | audiofileinCHOP, audiospectrumCHOP, mathCHOP, lfoCHOP, constantCHOP | CHOP |
| SOP | 蓝色 | gridSOP, sphereSOP, transformSOP, noiseSOP | SOP |
| DAT | 白色 | textDAT, tableDAT, scriptDAT, webserverDAT | DAT |
| MAT | 黄色 | phongMAT, pbrMAT, glslMAT, constMAT | MAT |
| COMP | 灰色 | geometryCOMP, containerCOMP, cameraCOMP, lightCOMP, windowCOMP | COMP |

## 安全说明

- MCP 仅在本地运行（端口 40404）。无身份验证 — 任何本地进程都可以发送命令。
- `td_execute_python` 对 TD Python 环境和文件系统（作为 TD 进程用户）具有不受限制的访问权限。
- `setup.sh` 从官方 404zero.com URL 下载 twozero.tox。如有疑虑，请验证下载。
- 此技能从不向外发送数据。所有 MCP 通信都是本地的。

## 参考

| 文件 | 内容 |
|------|------|
| `references/pitfalls.md` | 真实会话中来之不易的教训 |
| `references/operators.md` | 所有算子家族及其参数和用例 |
| `references/network-patterns.md` | 配方：音频反应式、生成式、GLSL、实例化 |
| `references/mcp-tools.md` | 完整 twozero MCP 工具参数模式 |
| `references/python-api.md` | TD Python：op()、脚本编写、扩展 |
| `references/troubleshooting.md` | 连接诊断、调试 |
| `references/glsl.md` | GLSL 统一变量、内置函数、着色器模板 |
| `references/postfx.md` | 后处理特效：泛光、CRT、色差、反馈发光 |
| `references/layout-compositor.md` | HUD 布局模式、面板网格、BSP 风格布局 |
| `references/operator-tips.md` | 线框渲染、反馈 TOP 设置 |
| `references/geometry-comp.md` | Geometry COMP：实例化、POP vs SOP、变形 |
| `references/audio-reactive.md` | 音频频段提取、节拍检测、包络跟随 |
| `references/animation.md` | LFO、定时器、关键帧、缓动、表达式驱动运动 |
| `references/midi-osc.md` | MIDI/OSC 控制器、TouchOSC、多机同步 |
| `references/particles.md` | POP 和传统 particleSOP — 发射、力、碰撞 |
| `references/projection-mapping.md` | 多窗口输出、角针、网格变形、边缘混合 |
| `references/external-data.md` | HTTP、WebSocket、MQTT、串口、TCP、webserverDAT |
| `references/panel-ui.md` | 自定义参数、面板 COMP、按钮/滑块/字段、panelExecuteDAT |
| `references/replicator.md` | replicatorCOMP — 数据驱动克隆、布局、回调 |
| `references/dat-scripting.md` | Execute DAT 家族 — chop/dat/parameter/panel/op/executeDAT |
| `references/3d-scene.md` | 照明设置、阴影、IBL/cubemaps、多摄像头、PBR |
| `scripts/setup.sh` | 自动化设置脚本 |

---

> 你不是在写代码。你是在指挥光。
