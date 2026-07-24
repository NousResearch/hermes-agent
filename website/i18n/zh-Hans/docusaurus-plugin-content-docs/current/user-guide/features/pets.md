---
sidebar_position: 11
title: "宠物（Petdex 吉祥物）"
description: "领养一个能对 Agent 活动作出反应的动画吉祥物，支持 CLI、TUI 和桌面应用"
---

# 宠物

Hermes 可以显示一个动画**宠物** —— 一个小型吉祥物精灵，能对 Agent 正在做的事情（空闲、运行工具、思考、完成、失败）作出反应，跨 **CLI**、**TUI** 和**桌面应用**。宠物来自公共的 [petdex](https://github.com/crafter-station/petdex) 画廊。

宠物纯粹是装饰性的。它们**对提示缓存、token 或 Agent 的行为没有任何影响** —— 精灵仅用于显示。该功能**默认关闭**，在你安装和选择宠物之前保持休眠。

## 工作原理

- 宠物安装到你配置文件的 `pets/` 目录（`<HERMES_HOME>/pets/<slug>/`），因此每个[配置文件](../profiles.md)保持自己的宠物集合。
- 选择宠物会将 `display.pet.slug` 和 `display.pet.enabled` 写入 `config.yaml` —— 没有东西作为密钥或环境变量存储。
- 每个界面监控它已经跟踪的活动，并将其映射到六种动画状态之一。映射集中在一个地方，因此每个界面行为一致：

  | Agent 活动 | 宠物状态 |
  | --- | --- |
  | 工具/轮次刚刚失败 | `failed` |
  | 计划完成（所有待办事项完成） | `jump`（庆祝） |
  | 轮次正常完成 | `wave` |
  | 工具正在执行 | `run` |
  | 模型正在思考/阅读 | `review` |
  | 轮次进行中（未指明） | `run` |
  | 等待你（澄清/审批提示已打开） | `waiting`（旧版 8 行电子表格上回退为 `idle`） |
  | 没有活动发生 | `idle` |

## 渲染

在终端中（CLI/TUI），当你的终端支持图形协议（**kitty**、**Ghostty**、**WezTerm**、**iTerm2** 或 **sixel**）时，Hermes 以全保真度渲染精灵。否则自动回退到真彩色 Unicode **半块**渲染。在管道或重定向内（无 TTY），终端渲染按设计禁用。

桌面应用将宠物绘制为浮动精灵并从**设置 → 外观**中切换。

## 快速开始（CLI）

```bash
# Browse the gallery (filter by substring)
hermes pets list
hermes pets list cat

# Install a pet and make it active in one step
hermes pets install boba --select

# Preview / animate it in your terminal (Ctrl+C to stop)
hermes pets show

# Check your setup
hermes pets doctor
```

## `hermes pets` 命令

| 目标 | 命令 |
| --- | --- |
| 浏览画廊 | `hermes pets list [query] [--limit N]` |
| 列出已安装的宠物 | `hermes pets list --installed` |
| 安装宠物 | `hermes pets install <slug> [--select] [--force]` |
| 设置活动宠物 | `hermes pets select [slug]`（省略 slug 会出现选择器） |
| 调整宠物大小 | `hermes pets scale <factor>`（例如 `0.5`，限制在 0.1–3.0） |
| 预览/动画 | `hermes pets show [slug] [--state <s>] [--cycle] [--once] [--mode <m>] [--scale <f>]` |
| 禁用宠物 | `hermes pets off` |
| 移除已安装的宠物 | `hermes pets remove <slug>` |
| 诊断设置 | `hermes pets doctor` |

`hermes pets show` 标志：

- `--state` —— 播放单个状态（`idle`、`wave`、`run`、`failed`、`review`、`jump`）。
- `--cycle` —— 循环所有状态。
- `--once` —— 播放一次而非循环。
- `--mode` —— 覆盖渲染协议（`kitty`、`iterm`、`sixel`、`unicode`、`auto`）。
- `--scale` —— 覆盖屏幕上的缩放比例（`0` = 使用配置值）。

## `/pet` 斜杠命令

在 CLI 和 TUI 中，你可以在不离开会话的情况下管理宠物：

- `/pet` —— 切换宠物开/关（如果没有活动宠物，则领养第一个已安装的宠物）。
- `/pet list` —— 浏览画廊。
- `/pet scale <factor>` —— 在所有地方调整宠物大小（例如 `/pet scale 0.5`）。
- `/pet <slug>` —— 领养特定宠物。
- `/pet off` —— 禁用宠物。

在 TUI 中，`/pet list` 打开一个交互式选择器覆盖层；在桌面应用中，它打开 Cmd+K 宠物面板。

## 生成宠物（`/hatch`）

除了从画廊安装预制宠物外，Hermes 还可以根据文本描述**生成全新宠物** —— 它自己的 AI 精灵生成管线。

- CLI/TUI：`/hatch <description>`（别名 `/generate-pet`），或 `hermes pets` → 生成流程。
- 桌面应用：类似宝可梦图鉴的**生成** UI —— 动画蛋、孵化特效和草稿选择器。

生成流程（两步、有成本限制的流程）：

1. **基础草稿** —— 生成几个低成本的、仅提示的"这个宠物应该长什么样"变体。你选择一个，或混合/重试进行新一轮。
2. **孵化** —— 将选定的基础图作为参考图像，为每个 Hermes 状态（空闲、思考、工具使用等）生成一个有依据的动画行，它们被确定性地切割成帧并打包为标准 petdex/Codex 图集（192×208 单元格的 8×9 网格）。结果是你保留的有效精灵表 —— 可以 `petdex submit`。

### 图像后端

生成使用活动的[图像生成 Provider](/user-guide/features/image-generation)，但它需要**参考图像依据**，以便每个动画行保持与基础相同的角色。支持参考的后端：**Nous Portal**、**OpenRouter**、**OpenAI**（`gpt-image-2`）和 **Krea**。OpenRouter/Nous 默认运行质量优先的模型链。

- 解析顺序优先 Nous Portal → OpenAI → OpenRouter。
- 如果没有配置支持参考的后端，生成会显示一个可操作的错误，指向 `hermes tools` → 图像生成。（安装/领养现有画廊宠物不需要图像后端。）
- 使用 `HERMES_PET_IMAGE_PROVIDER` 环境变量覆盖后端（例如 `HERMES_PET_IMAGE_PROVIDER=openrouter`）。

## 桌面应用

在桌面应用中，你可以通过两种方式管理宠物：

- **Cmd+K → "Pets…"** —— 在不离开键盘的情况下浏览、搜索、领养和切换宠物（与主题选择器一致）。
- **设置 → 外观** —— 同样的画廊加上一个**大小滑块**，拖动时实时调整浮动吉祥物的大小。

两者都会就地调整浮动吉祥物的大小/切换/领养 —— 大小变更立即生效；领养新宠物会在瞬间点亮它。

### 弹出覆盖层

**Shift 点击**浮动宠物，将其弹出为独立的透明、始终在顶层的桌面窗口。在那里，即使 Hermes 最小化（Codex 风格），它也保持可见，一瞥就能告诉你 Agent 在做什么。

弹出后的手势：

| 手势 | 操作 |
| --- | --- |
| **拖拽** | 将宠物移动到屏幕任何位置，甚至应用外。它的位置和弹入/弹出状态跨重启保持。 |
| **单击** | 打开一个迷你编辑器，向最近的会话发送提示 —— 无需显示应用。 |
| **双击** | 切换应用窗口：如果在前台则最小化，如果隐藏则恢复。 |
| **Shift 点击** | 将宠物弹回窗口内。 |
| **邮件图标** | 仅在你不在时轮次结束时出现；点击将应用提升到最近的线程（并标记为已读）。 |

只有弹出的宠物显示**语音气泡**（`working…`、`thinking…`、`your turn`……） —— 在窗口内应用本身就是界面，宠物在那里保持安静。

覆盖层是应用内宠物的纯粹代理 —— 它不携带独立的网关连接，也不会出现在 Dock 或应用切换器中。

## 配置

所有设置位于 `config.yaml` 中的 `display.pet` 下：

```yaml
display:
  pet:
    enabled: false        # master on/off (true once you select a pet)
    slug: ""              # active pet; empty = first installed
    render_mode: auto      # auto | kitty | iterm | sixel | unicode | off
    scale: 0.33           # master size knob (relative to native 192x208 frames)
    unicode_cols: 0       # hard override for terminal width (0 = derive from scale)
```

- **`scale`** 是唯一的全局大小控制。一个数字缩小所有界面：桌面画布按它缩放像素，CLI/TUI 从它派生终端列宽。半块回退有一个可读性下限 —— 它无法像真正的像素 kitty/GUI 渲染那样缩小到同样程度而不变成糊状，因此相同的 `scale` 在 kitty 下看起来清晰，但在半块模式下被限制。
- **`render_mode: auto`** 检测 kitty/iTerm2/sixel 并回退到 unicode 半块。显式设置它以强制使用特定协议，或设置 `off` 在保持桌面宠物的同时禁用终端渲染。
- **`unicode_cols`** 独立于 `scale` 固定终端列宽；保持 `0` 从 `scale` 派生宽度。

## 故障排除

运行 `hermes pets doctor` —— 它会报告：

- 宠物目录及已安装的宠物，
- `display.pet.enabled`、`display.pet.slug` 和已解析的活动宠物，
- 配置的 `render_mode`、检测到的终端图形协议和 TTY 的有效模式，
- Pillow（用于精灵解码）是否可导入。

当宠物已安装、已选择、已启用且 Pillow 可用时，它会打印 `✓ ready`。

常见问题：

- 宠物只有在**已安装且已选择**时才会显示（`enabled: true`）。
- 在管道/重定向内（无 TTY），终端渲染按设计禁用。
- petdex npm CLI 安装到 `~/.codex/pets`；Hermes 使用自己的配置文件级 `<HERMES_HOME>/pets/` —— 请通过 `hermes pets` 安装。

## 另请参阅

- [`petdex` 技能](../skills/bundled/productivity/productivity-petdex.md)让 Agent 根据你的请求安装和切换宠物。
