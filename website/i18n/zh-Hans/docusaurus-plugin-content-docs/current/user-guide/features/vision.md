---
title: 视觉与图像粘贴
description: 将剪贴板中的图像粘贴到 Hermes CLI，进行多模态视觉分析。
sidebar_label: 视觉与图像粘贴
sidebar_position: 7
---

# 视觉与图像粘贴

Hermes Agent 支持**多模态视觉**——你可以直接将剪贴板中的图像粘贴到 CLI，让 Agent 对其进行分析、描述或处理。图像以 base64 编码的内容块形式发送给模型，因此任何支持视觉的模型均可处理。

## 工作原理

1. 将图像复制到剪贴板（截图、浏览器图片等）
2. 使用以下任一方式附加图像
3. 输入问题并按 Enter
4. 图像以 `[📎 Image #1]` 徽章形式显示在输入框上方
5. 提交时，图像作为视觉内容块发送给模型

发送前可附加多张图像，每张图像都有独立徽章。按 `Ctrl+C` 可清除所有已附加图像。

图像以带时间戳的 PNG 文件名保存至 `~/.hermes/images/`。

## 粘贴方式

附加图像的方式取决于你的终端环境。并非所有方式在所有环境下均可用——以下是完整说明：

### `/paste` 命令

**最可靠的显式图像附加备用方案。**

```
/paste
```

输入 `/paste` 并按 Enter。Hermes 会检查剪贴板中是否有图像并附加。当你的终端重写了 `Cmd+V`/`Ctrl+V`，或剪贴板中只有图像而没有 bracketed-paste（括号粘贴）文本载荷可供检查时，这是最安全的选项。

### Ctrl+V / Cmd+V

Hermes 现在将粘贴处理为分层流程：
- 优先进行普通文本粘贴
- 若终端未能正常传递文本，则回退到原生剪贴板 / OSC52 文本
- 当剪贴板或粘贴内容解析为图像或图像路径时，附加图像

这意味着粘贴的 macOS 截图临时路径和 `file://...` 图像 URI 可以立即附加，而不是以原始文本形式留在编辑器中。

:::warning
如果剪贴板中**只有图像**（无文本），终端仍无法直接发送二进制图像字节。请使用 `/paste` 作为显式图像附加的备用方案。
:::

### `/terminal-setup`（适用于 VS Code / Cursor / Windsurf）

如果你在 macOS 上的 VS Code 系列集成终端中运行 TUI，Hermes 可以安装推荐的 `workbench.action.terminal.sendSequence` 绑定，以获得更好的多行输入及撤销/重做一致性：

```text
/terminal-setup
```

当 `Cmd+Enter`、`Cmd+Z` 或 `Shift+Cmd+Z` 被 IDE 拦截时，此命令尤为有用。仅在本地机器上运行——不要在 SSH 会话中使用。

## 平台兼容性

| 环境 | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | 备注 |
|---|:---:|:---:|:---:|---|
| **macOS Terminal / iTerm2** | ✅ | ✅ | n/a | 最佳体验——原生剪贴板 + 截图路径恢复 |
| **Apple Terminal** | ✅ | ✅ | n/a | 若 Cmd+←/→/⌫ 被重写，使用 Ctrl+A / Ctrl+E / Ctrl+U 备用方案 |
| **Linux X11 桌面** | ✅ | ✅ | n/a | 需要 `xclip`（`apt install xclip`） |
| **Linux Wayland 桌面** | ✅ | ✅ | n/a | 需要 `wl-paste`（`apt install wl-clipboard`） |
| **WSL2（Windows Terminal）** | ✅ | ✅ | n/a | 使用 `powershell.exe`——无需额外安装 |
| **VS Code / Cursor / Windsurf（本地）** | ✅ | ✅ | ✅ | 推荐，以获得更好的 Cmd+Enter / 撤销 / 重做一致性 |
| **VS Code / Cursor / Windsurf（SSH）** | ❌² | ❌² | ❌³ | 请在本地机器上运行 `/terminal-setup` |
| **SSH 终端（任意）** | ❌² | ❌² | n/a | 无法访问远程剪贴板 |

² 参见下方 [SSH 与远程会话](#ssh--remote-sessions)
³ 该命令写入本地 IDE 快捷键绑定，不应从远程主机运行

## 各平台配置说明

### macOS

**无需任何配置。** Hermes 使用 `osascript`（macOS 内置）读取剪贴板。如需更快的性能，可选择安装 `pngpaste`：

```bash
brew install pngpaste
```

### Linux（X11）

安装 `xclip`：

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux（Wayland）

现代 Linux 桌面（Ubuntu 22.04+、Fedora 34+）通常默认使用 Wayland。安装 `wl-clipboard`：

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip 如何检查是否在使用 Wayland
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland，"x11" = X11，"tty" = 无显示服务器
```
:::

### WSL2

**无需额外配置。** Hermes 通过 `/proc/version` 自动检测 WSL2，并使用 `powershell.exe` 通过 .NET 的 `System.Windows.Forms.Clipboard` 访问 Windows 剪贴板。这是 WSL2 Windows 互操作的内置功能——`powershell.exe` 默认可用。

剪贴板数据通过 stdout 以 base64 编码的 PNG 格式传输，无需路径转换或临时文件。

:::info WSLg 说明
如果你使用的是 WSLg（带 GUI 支持的 WSL2），Hermes 会优先尝试 PowerShell 路径，然后回退到 `wl-paste`。WSLg 的剪贴板桥接仅支持 BMP 格式的图像——Hermes 会使用 Pillow（如已安装）或 ImageMagick 的 `convert` 命令自动将 BMP 转换为 PNG。
:::

#### 验证 WSL2 剪贴板访问

```bash
# 1. 检查 WSL 检测
grep -i microsoft /proc/version

# 2. 检查 PowerShell 是否可访问
which powershell.exe

# 3. 复制一张图像，然后检查
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# 应输出 "True"
```

## SSH 与远程会话

**通过 SSH 进行剪贴板图像粘贴无法完全正常工作。** 当你 SSH 到远程机器时，Hermes CLI 运行在远程主机上。剪贴板工具（`xclip`、`wl-paste`、`powershell.exe`、`osascript`）读取的是其所在机器的剪贴板——即远程服务器，而非你的本地机器。因此，本地剪贴板中的图像在远程端无法访问。

文本有时仍可通过终端粘贴或 OSC52 传输，但图像剪贴板访问和本地截图临时路径始终绑定于运行 Hermes 的机器。

### SSH 的变通方案

1. **上传图像文件**——在本地保存图像，通过 `scp`、VSCode 文件浏览器（拖放）或任何文件传输方式上传到远程服务器，然后通过路径引用。*（计划在未来版本中提供 `/attach <filepath>` 命令。）*

2. **使用 URL**——如果图像可在线访问，直接在消息中粘贴 URL。Agent 可使用 `vision_analyze` 直接查看任意图像 URL。

3. **X11 转发**——使用 `ssh -X` 连接以转发 X11。这允许远程机器上的 `xclip` 访问你本地的 X11 剪贴板。需要本地运行 X 服务器（macOS 上为 XQuartz，Linux X11 桌面内置）。大图像传输较慢。

4. **使用消息平台**——通过 Telegram、Discord、Slack 或 WhatsApp 向 Hermes 发送图像。这些平台原生支持图像上传，不受剪贴板/终端限制的影响。

## 为什么终端无法粘贴图像

这是一个常见的困惑来源，以下是技术说明：

终端是**基于文本**的界面。当你按下 Ctrl+V（或 Cmd+V）时，终端模拟器会：

1. 从剪贴板读取**文本内容**
2. 将其包裹在 [bracketed paste](https://en.wikipedia.org/wiki/Bracketed-paste)（括号粘贴）转义序列中
3. 通过终端的文本流将其发送给应用程序

如果剪贴板中只有图像（无文本），终端没有任何内容可发送。目前没有标准的终端转义序列用于传输二进制图像数据，终端会直接忽略。

这就是为什么 Hermes 使用独立的剪贴板检查——它不通过终端粘贴事件接收图像数据，而是直接通过子进程调用操作系统级工具（`osascript`、`powershell.exe`、`xclip`、`wl-paste`）独立读取剪贴板。

## 支持的模型

图像粘贴适用于任何支持视觉的模型。图像以 base64 编码的 data URL 形式，按 OpenAI 视觉内容格式发送：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

大多数现代模型支持此格式，包括 GPT-4 Vision、Claude（带视觉）、Gemini，以及通过 OpenRouter 提供服务的开源多模态模型。

## 图像路由（视觉模型 vs 纯文本模型）

当用户附加图像时——无论来自 CLI 剪贴板、gateway（Telegram/Discord 图片）还是其他入口——Hermes 会根据当前模型是否支持视觉进行路由：

| 你的模型 | 图像处理方式 |
|---|---|
| **支持视觉的模型**（GPT-4V、Claude with vision、Gemini、Qwen-VL、MiMo-VL 等） | 使用上述提供商原生图像内容格式，以**真实像素**发送。无文本摘要层。 |
| **纯文本模型**（DeepSeek V3、较小的开源模型、旧版纯对话端点） | 通过 `vision_analyze` 辅助工具路由——辅助视觉模型描述图像，文本描述注入对话。 |

无需手动配置——Hermes 在提供商元数据中查找当前模型的能力并自动选择正确路径。实际效果：你可以在会话中途切换视觉模型与非视觉模型，图像处理"开箱即用"，无需更改工作流。纯文本模型会获得关于图像的连贯上下文，而不是一个会被拒绝的损坏多模态载荷。

处理文本描述路径的辅助模型可在 `auxiliary.vision` 下配置——参见[辅助模型](/user-guide/configuration#auxiliary-models)。

### `vision_analyze` 具有相同的双重行为

`vision_analyze` 工具本身遵循相同的路由逻辑。当当前主模型支持视觉，**且**其提供商支持在工具结果中包含图像内容（目前为 Anthropic、OpenAI、Azure-OpenAI 和 Gemini 3.x 技术栈），`vision_analyze` 会跳过辅助描述器，直接将原始图像像素作为多模态工具结果信封返回。主模型在下一轮会原生看到图像——无辅助调用、无文本摘要信息损失、无额外延迟。

对于纯文本主模型（或工具结果通道不支持图像的提供商），`vision_analyze` 回退到旧路径：请求已配置的辅助视觉模型描述图像，并以纯文本形式返回描述。无论哪种情况，调用工具的签名相同——工具在运行时根据当前模型决定采用哪条路径。