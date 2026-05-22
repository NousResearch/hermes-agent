---
title: 视觉与图片粘贴
description: 将剪贴板中的图片粘贴到 Hermes CLI 中，进行多模态视觉分析。
sidebar_label: 视觉与图片粘贴
sidebar_position: 7
---

# 视觉与图片粘贴

Hermes Agent 支持**多模态视觉**——您可以直接从剪贴板粘贴图片到 CLI 中，让智能体分析、描述或处理它们。图片以 base64 编码的内容块形式发送给模型，因此任何支持视觉的模型都可以处理它们。

## 工作原理

1. 将图片复制到剪贴板（截图、浏览器图片等）
2. 使用以下方法之一附加它
3. 输入您的问题并按回车
4. 图片以 `[📎 Image #1]` 徽章形式出现在输入上方
5. 提交时，图片作为视觉内容块发送给模型

您可以在发送前附加多张图片——每张都有自己的徽章。按 `Ctrl+C` 清除所有附加图片。

图片以带时间戳文件名的 PNG 文件保存到 `~/.hermes/images/`。

## 粘贴方法

如何附加图片取决于您的终端环境。并非所有方法在任何地方都有效——以下是完整分解：

### `/paste` 命令

**最可靠的显式图片附加回退。**

```
/paste
```

输入 `/paste` 并按回车。Hermes 检查您的剪贴板是否有图片并附加它。当您的终端重写 `Cmd+V`/`Ctrl+V`，或当您仅复制了图片且没有括号粘贴文本负载可检查时，这是最安全的选项。

### Ctrl+V / Cmd+V

Hermes 现在将粘贴视为分层流程：
- 正常文本粘贴优先
- 如果终端未干净地交付文本，则使用原生剪贴板 / OSC52 文本回退
- 当剪贴板或粘贴负载解析为图片或图片路径时，附加图片

这意味着粘贴的 macOS 截图临时路径和 `file://...` 图片 URI 可以立即附加，而非作为原始文本留在编辑器中。

:::warning
如果您的剪贴板**只有图片**（没有文本），终端仍然无法直接发送二进制图片字节。使用 `/paste` 作为显式图片附加回退。
:::

### VS Code / Cursor / Windsurf 的 `/terminal-setup`

如果您在 macOS 上的本地 VS Code 系列集成终端中运行 TUI，Hermes 可以安装推荐的 `workbench.action.terminal.sendSequence` 绑定，以获得更好的多行和撤销/重做一致性：

```text
/terminal-setup
```

当 `Cmd+Enter`、`Cmd+Z` 或 `Shift+Cmd+Z` 被 IDE 拦截时，这特别有用。仅在本地机器上运行——不要在 SSH 会话中运行。

## 平台兼容性

| 环境 | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | 说明 |
|---|---|:---:|:---:|:---:|---|
| **macOS Terminal / iTerm2** | ✅ | ✅ | 不适用 | 最佳体验——原生剪贴板 + 截图路径恢复 |
| **Apple Terminal** | ✅ | ✅ | 不适用 | 如果 Cmd+←/→/⌫ 被重写，使用 Ctrl+A / Ctrl+E / Ctrl+U 回退 |
| **Linux X11 桌面** | ✅ | ✅ | 不适用 | 需要 `xclip` (`apt install xclip`) |
| **Linux Wayland 桌面** | ✅ | ✅ | 不适用 | 需要 `wl-paste` (`apt install wl-clipboard`) |
| **WSL2 (Windows Terminal)** | ✅ | ✅ | 不适用 | 使用 `powershell.exe` —— 无需额外安装 |
| **VS Code / Cursor / Windsurf (本地)** | ✅ | ✅ | ✅ | 推荐用于更好的 Cmd+Enter / 撤销 / 重做一致性 |
| **VS Code / Cursor / Windsurf (SSH)** | ❌² | ❌² | ❌³ | 改在本地机器上运行 `/terminal-setup` |
| **SSH 终端 (任何)** | ❌² | ❌² | 不适用 | 远程剪贴板不可访问 |

² 请参阅下面的 [SSH 与远程会话](#ssh--远程会话)
³ 该命令写入本地 IDE 键绑定，不应从远程主机运行

## 平台特定设置

### macOS

**无需设置。** Hermes 使用 `osascript`（内置于 macOS）读取剪贴板。为了更快性能，可选安装 `pngpaste`：

```bash
brew install pngpaste
```

### Linux (X11)

安装 `xclip`：

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux (Wayland)

现代 Linux 桌面（Ubuntu 22.04+、Fedora 34+）通常默认使用 Wayland。安装 `wl-clipboard`：

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip 如何检查您是否在使用 Wayland
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland, "x11" = X11, "tty" = 无显示服务器
```
:::

### WSL2

**无需额外设置。** Hermes 通过 `/proc/version` 自动检测 WSL2 并使用 `powershell.exe` 通过 .NET 的 `System.Windows.Forms.Clipboard` 访问 Windows 剪贴板。这内置于 WSL2 的 Windows 互操作中——`powershell.exe` 默认可用。

剪贴板数据以 base64 编码的 PNG 通过 stdout 传输，因此不需要文件路径转换或临时文件。

:::info WSLg 说明
如果您运行 WSLg（带 GUI 支持的 WSL2），Hermes 先尝试 PowerShell 路径，然后回退到 `wl-paste`。WSLg 的剪贴板桥仅支持 BMP 格式的图片——Hermes 使用 Pillow（如果已安装）或 ImageMagick 的 `convert` 命令自动将 BMP 转换为 PNG。
:::

#### 验证 WSL2 剪贴板访问

```bash
# 1. 检查 WSL 检测
grep -i microsoft /proc/version

# 2. 检查 PowerShell 是否可访问
which powershell.exe

# 3. 复制图片，然后检查
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# 应打印 "True"
```

## SSH 与远程会话 {#ssh--远程会话}

**剪贴板图片粘贴无法完全通过 SSH 工作。** 当您 SSH 到远程机器时，Hermes CLI 在远程主机上运行。剪贴板工具（`xclip`、`wl-paste`、`powershell.exe`、`osascript`）读取它们运行的机器的剪贴板——即远程服务器，而非您的本地机器。因此，您的本地剪贴板图片从远程端无法访问。

文本有时仍可通过终端粘贴或 OSC52 桥接，但图片剪贴板访问和本地截图临时路径仍与运行 Hermes 的机器绑定。

### SSH 的变通方法

1. **上传图片文件** —— 本地保存图片，通过 `scp`、VSCode 的文件资源管理器（拖放）或任何文件传输方法上传到远程服务器。然后按路径引用它。*(`/attach <filepath>` 命令计划在未来版本中推出。)*

2. **使用 URL** —— 如果图片可在线访问，只需在消息中粘贴 URL。智能体可以直接使用 `vision_analyze` 查看任何图片 URL。

3. **X11 转发** —— 使用 `ssh -X` 连接以转发 X11。这让远程机器上的 `xclip` 访问您的本地 X11 剪贴板。需要本地运行 X 服务器（macOS 上的 XQuartz，Linux X11 桌面内置）。大图片较慢。

4. **使用消息平台** —— 通过 Telegram、Discord、Slack 或 WhatsApp 向 Hermes 发送图片。这些平台原生处理图片上传，不受剪贴板/终端限制影响。

## 为什么终端无法粘贴图片

这是常见的困惑来源，因此这里是技术解释：

终端是**基于文本**的界面。当您按 Ctrl+V（或 Cmd+V）时，终端模拟器：

1. 读取剪贴板中的**文本内容**
2. 将其包装在[括号粘贴](https://en.wikipedia.org/wiki/Bracketed-paste)转义序列中
3. 通过终端的文本流将其发送给应用程序

如果剪贴板仅包含图片（没有文本），终端没有可发送的内容。没有用于二进制图片数据的标准终端转义序列。终端简单地什么都不做。

这就是 Hermes 使用单独剪贴板检查的原因——不是通过终端粘贴事件接收图片数据，而是通过子进程直接调用操作系统级工具（`osascript`、`powershell.exe`、`xclip`、`wl-paste`）独立读取剪贴板。

## 支持的模型

图片粘贴适用于任何支持视觉的模型。图片以 OpenAI 视觉内容格式的 base64 编码数据 URL 发送：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

大多数现代模型支持此格式，包括 GPT-4 Vision、Claude（带视觉）、Gemini，以及通过 OpenRouter 提供的开源多模态模型。

## 图片路由（支持视觉 vs 仅文本模型）

当用户附加图片时——从 CLI 剪贴板、网关（Telegram/Discord 照片）或任何其他入口点——Hermes 根据您当前的模型是否实际支持视觉来路由它：

| 您的模型 | 图片会发生什么 |
|---|---|
| **支持视觉** (GPT-4V、Claude 带视觉、Gemini、Qwen-VL、MiMo-VL 等) | 使用提供商的原生图片内容格式作为**真实像素**发送。无文本摘要层。 |
| **仅文本** (DeepSeek V3、较小的开源模型、较旧的仅聊天端点) | 通过 `vision_analyze` 辅助工具路由——辅助视觉模型描述图片，文本描述被注入对话中。 |

您无需配置——Hermes 在提供商元数据中查找您当前模型的能力，并自动选择正确的路径。实际效果：您可以在会话中在视觉和非视觉模型之间切换，图片处理"正常工作"，无需更改工作流。仅文本模型获得关于图片的连贯上下文，而非它们必须拒绝的损坏多模态负载。

哪个辅助模型处理文本描述路径可在 `auxiliary.vision` 下配置——请参阅 [辅助模型](/user-guide/configuration#辅助模型)。
