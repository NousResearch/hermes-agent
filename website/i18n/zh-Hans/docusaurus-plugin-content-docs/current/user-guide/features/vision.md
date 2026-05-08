---
title: 视觉与图像粘贴
description: "将剪贴板中的图像粘贴到 Hermes CLI 中进行多模态视觉分析。"
sidebar_label: "视觉与图像粘贴"
sidebar_position: 7
---

# 视觉与图像粘贴

Hermes Agent 支持**多模态视觉** — 您可以直接将剪贴板中的图像粘贴到 CLI 中，并让代理分析、描述或处理它们。图像作为 base64 编码的内容块发送到模型，因此任何支持视觉的模型都可以处理它们。

## 工作原理

1. 将图像复制到剪贴板（截图、浏览器图像等）
2. 使用以下方法之一附加它
3. 输入您的问题并按回车
4. 图像显示为 `[📎 Image #1]` 徽章在输入上方
5. 提交时，图像作为视觉内容块发送到模型

在发送之前，您可以附加多张图像 — 每张获得自己的徽章。按 `Ctrl+C` 清除所有附加的图像。

图像作为带时间戳文件名的 PNG 文件保存到 `~/.hermes/images/`。

## 粘贴方法

如何附加图像取决于您的终端环境。并非所有方法都适用于所有地方 — 以下是完整说明：

### `/paste` 命令

**最可靠的显式图像附加后备。**

```
/paste
```

输入 `/paste` 并按回车。Hermes 检查剪贴板中是否有图像并附加它。当您的终端重写 `Cmd+V`/`Ctrl+V`，或者当您只复制了图像且没有要检查的括号粘贴文本负载时，这是最安全的选择。

### Ctrl+V / Cmd+V

Hermes 现在将粘贴视为分层流程：
- 首先是正常的文本粘贴
- 如果终端没有干净地传递文本，则使用原生剪贴板 / OSC52 文本后备
- 当剪贴板或粘贴负载解析为图像或图像路径时附加图像

这意味着粘贴的 macOS 截图临时路径和 `file://...` 图像 URI 可以立即附加，而不是作为原始文本留在输入框中。

:::warning
如果您的剪贴板**只有图像**（没有文本），终端仍然无法直接发送二进制图像字节。使用 `/paste` 作为显式图像附加后备。
:::

### `/terminal-setup` 用于 VS Code / Cursor / Windsurf

如果您在 macOS 上的本地 VS Code 系列集成终端中运行 TUI，Hermes 可以安装推荐的 `workbench.action.terminal.sendSequence` 绑定，以获得更好的多行和撤销/重做体验：

```text
/terminal-setup
```

当 `Cmd+Enter`、`Cmd+Z` 或 `Shift+Cmd+Z` 被 IDE 拦截时，这尤其有用。仅在本地机器上运行它 — 不在 SSH 会话中。

## 平台兼容性

| 环境 | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | 备注 |
|---|:---:|:---:|:---:|---|
| **macOS 终端 / iTerm2** | ✅ | ✅ | n/a | 最佳体验 — 原生剪贴板 + 截图路径恢复 |
| **Apple 终端** | ✅ | ✅ | n/a | 如果 Cmd+←/→/⌫ 被重写，使用 Ctrl+A / Ctrl+E / Ctrl+U 后备 |
| **Linux X11 桌面** | ✅ | ✅ | n/a | 需要 `xclip`（`apt install xclip`） |
| **Linux Wayland 桌面** | ✅ | ✅ | n/a | 需要 `wl-paste`（`apt install wl-clipboard`） |
| **WSL2 (Windows Terminal)** | ✅ | ✅ | n/a | 使用 `powershell.exe` — 无需额外安装 |
| **VS Code / Cursor / Windsurf（本地）** | ✅ | ✅ | ✅ | 推荐使用以获得更好的 Cmd+Enter / 撤销 / 重做 体验 |
| **VS Code / Cursor / Windsurf（SSH）** | ❌² | ❌² | ❌³ | 改为在本地机器上运行 `/terminal-setup` |
| **SSH 终端（任意）** | ❌² | ❌² | n/a | 远程剪贴板不可访问 |

² 参见下面的 [SSH 和远程会话](#ssh-和远程会话)
³ 该命令写入本地 IDE 键绑定，不应从远程主机运行

## 平台特定设置

### macOS

**无需设置。** Hermes 使用 `osascript`（内置于 macOS）读取剪贴板。为获得更快性能，可选择安装 `pngpaste`：

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

**无需额外设置。** Hermes 自动检测 WSL2（通过 `/proc/version`）并使用 `powershell.exe` 通过 .NET 的 `System.Windows.Forms.Clipboard` 访问 Windows 剪贴板。这是 WSL2 Windows 互操作的内置功能 — `powershell.exe` 默认可用。

剪贴板数据作为 base64 编码的 PNG 通过 stdout 传输，因此不需要文件路径转换或临时文件。

:::info WSLg 说明
如果您正在运行 WSLg（带 GUI 支持的 WSL2），Hermes 首先尝试 PowerShell 路径，然后回退到 `wl-paste`。WSLg 的剪贴板桥仅支持 BMP 格式的图像 — Hermes 使用 Pillow（如果已安装）或 ImageMagick 的 `convert` 命令自动将 BMP 转换为 PNG。
:::

#### 验证 WSL2 剪贴板访问

```bash
# 1. 检查 WSL 检测
grep -i microsoft /proc/version

# 2. 检查 PowerShell 是否可访问
which powershell.exe

# 3. 复制一个图像，然后检查
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# 应该打印 "True"
```

## SSH 和远程会话

**剪贴板图像粘贴在 SSH 上不能完全工作。** 当您 SSH 到远程机器时，Hermes CLI 在远程主机上运行。剪贴板工具（`xclip`、`wl-paste`、`powershell.exe`、`osascript`）读取它们运行所在机器的剪贴板 — 即远程服务器，而不是您的本地机器。因此，您的本地剪贴板图像从远程端无法访问。

文本有时仍然可以通过终端粘贴或 OSC52 传输，但图像剪贴板访问和本地截图临时路径仍绑定到运行 Hermes 的机器。

### SSH 的变通方法

1. **上传图像文件** — 在本地保存图像，通过 `scp`、VSCode 的文件资源管理器（拖放）或任何文件传输方法上传到远程服务器。然后按路径引用。（`/attach <filepath>` 命令计划在未来版本中添加。）

2. **使用 URL** — 如果图像可以在线访问，只需在消息中粘贴 URL。代理可以使用 `vision_analyze` 直接查看任何图像 URL。

3. **X11 转发** — 使用 `ssh -X` 连接进行转发。这让远程机器上的 `xclip` 访问您的本地 X11 剪贴板。需要在本地运行 X 服务器（macOS 上的 XQuartz、Linux X11 桌面上的内置）。大图像速度较慢。

4. **使用消息平台** — 通过 Telegram、Discord、Slack 或 WhatsApp 向 Hermes 发送图像。这些平台原生处理图像上传，不受剪贴板/终端限制的影响。

## 为什么终端不能粘贴图像

这是一个常见困惑来源，所以这里是技术解释：

终端是**基于文本的**接口。当您按 Ctrl+V（或 Cmd+V）时，终端模拟器：

1. 读取剪贴板的**文本内容**
2. 将其包装在[括号粘贴](https://en.wikipedia.org/wiki/Bracketed-paste)转义序列中
3. 通过终端的文本流将其发送到应用程序

如果剪贴板只包含图像（没有文本），终端就没有要发送的内容。没有用于二进制图像数据的标准终端转义序列。终端简单地不执行任何操作。

这就是为什么 Hermes 使用单独的剪贴板检查 — 它不是通过终端粘贴事件接收图像数据，而是通过子进程直接调用 OS 级工具（`osascript`、`powershell.exe`、`xclip`、`wl-paste`）来独立读取剪贴板。

## 支持的模型

图像粘贴适用于任何支持视觉的模型。图像以 OpenAI 视觉内容格式的 base64 编码数据 URL 发送：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

大多数现代模型都支持此格式，包括 GPT-4 Vision、Claude（带视觉）、Gemini，以及通过 OpenRouter 服务的开源多模态模型。

## 图像路由（支持视觉 vs. 仅文本模型）

当用户附加图像时 — 从 CLI 剪贴板、网关（Telegram/Discord 照片）或任何其他入口点 — Hermes 根据您当前模型是否实际支持视觉来路由：

| 您的模型 | 图像处理方式 |
|---|---|
| **支持视觉**（GPT-4V、Claude 带视觉、Gemini、Qwen-VL、MiMo-VL 等） | 使用上述提供商的原生图像内容格式作为**真实像素**发送。没有文本摘要层。 |
| **仅文本**（DeepSeek V3、更小的开源模型、旧聊天专用端点） | 通过 `vision_analyze` 辅助工具路由 — 辅助视觉模型描述图像，文本描述被注入对话。 |

您不需要配置这个功能 — Hermes 在提供商元数据中查找您当前模型的能力并自动选择正确路径。实际效果：您可以在会话中间在视觉和非视觉模型之间切换，图像处理"正常工作"，而无需更改您的工作流程。仅文本模型获得关于图像的连贯上下文，而不是它们不得不拒绝的损坏的多模态负载。

哪个辅助模型处理文本描述路径可以通过 `auxiliary.vision` 配置 — 参见[辅助模型](/docs/user-guide/configuration#auxiliary-models)。
