---
title: "Ascii Art — ASCII art: pyfiglet, cowsay, boxes, image-to-ascii"
sidebar_label: "Ascii Art"
description: "ASCII art：pyfiglet、cowsay、boxes、image-to-ascii"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ascii Art

ASCII art：pyfiglet、cowsay、boxes、image-to-ascii。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/ascii-art` |
| 版本 | `4.0.0` |
| 作者 | 0xbyt4, Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `ASCII`, `Art`, `Banners`, `Creative`, `Unicode`, `Text-Art`, `pyfiglet`, `figlet`, `cowsay`, `boxes` |
| 相关 skill | [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# ASCII Art Skill

多种工具，满足不同的 ASCII art 需求。所有工具均为本地 CLI 程序或免费 REST API——无需 API 密钥。

## 工具 1：文字横幅（pyfiglet——本地）

将文本渲染为大型 ASCII art 横幅。内置 571 种字体。

### 安装

```bash
pip install pyfiglet --break-system-packages -q
```

### 用法

```bash
python3 -m pyfiglet "YOUR TEXT" -f slant
python3 -m pyfiglet "TEXT" -f doom -w 80    # Set width
python3 -m pyfiglet --list_fonts             # List all 571 fonts
```

### 推荐字体

| 风格 | 字体 | 适用场景 |
|-------|------|----------|
| 简洁现代 | `slant` | 项目名称、标题 |
| 粗体块状 | `doom` | 标题、Logo |
| 大而易读 | `big` | 横幅 |
| 经典横幅 | `banner3` | 宽屏显示 |
| 紧凑 | `small` | 副标题 |
| 赛博朋克 | `cyberlarge` | 科技主题 |
| 3D 效果 | `3-d` | 启动画面 |
| 哥特风 | `gothic` | 戏剧性文字 |

### 提示

- 预览 2-3 种字体，让用户选择喜欢的
- 短文本（1-8 个字符）与 `doom` 或 `block` 等精细字体搭配效果最佳
- 长文本更适合 `small` 或 `mini` 等紧凑字体

## 工具 2：文字横幅（asciified API——远程，无需安装）

将文本转换为 ASCII art 的免费 REST API。支持 250+ 种 FIGlet 字体。直接返回纯文本——无需解析。当 pyfiglet 未安装时使用，或作为快速替代方案。

### 用法（通过终端 curl）

```bash
# Basic text banner (default font)
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello+World"

# With a specific font
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Slant"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Doom"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Star+Wars"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=3-D"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Banner3"

# List all available fonts (returns JSON array)
curl -s "https://asciified.thelicato.io/api/v2/fonts"
```

### 提示

- 在 text 参数中将空格 URL 编码为 `+`
- 响应为纯文本 ASCII art——无 JSON 包装，可直接显示
- 字体名称区分大小写；使用 fonts 端点获取精确名称
- 在任何带有 curl 的终端中均可使用——无需 Python 或 pip

## 工具 3：Cowsay（消息艺术）

经典工具，将文本包裹在带有 ASCII 角色的对话气泡中。

### 安装

```bash
sudo apt install cowsay -y    # Debian/Ubuntu
# brew install cowsay         # macOS
```

### 用法

```bash
cowsay "Hello World"
cowsay -f tux "Linux rules"       # Tux the penguin
cowsay -f dragon "Rawr!"          # Dragon
cowsay -f stegosaurus "Roar!"     # Stegosaurus
cowthink "Hmm..."                  # Thought bubble
cowsay -l                          # List all characters
```

### 可用角色（50+）

`beavis.zen`, `bong`, `bunny`, `cheese`, `daemon`, `default`, `dragon`,
`dragon-and-cow`, `elephant`, `eyes`, `flaming-skull`, `ghostbusters`,
`hellokitty`, `kiss`, `kitty`, `koala`, `luke-koala`, `mech-and-cow`,
`meow`, `moofasa`, `moose`, `ren`, `sheep`, `skeleton`, `small`,
`stegosaurus`, `stimpy`, `supermilker`, `surgery`, `three-eyes`,
`turkey`, `turtle`, `tux`, `udder`, `vader`, `vader-koala`, `www`

### 眼睛/舌头修饰符

```bash
cowsay -b "Borg"       # =_= eyes
cowsay -d "Dead"       # x_x eyes
cowsay -g "Greedy"     # $_$ eyes
cowsay -p "Paranoid"   # @_@ eyes
cowsay -s "Stoned"     # *_* eyes
cowsay -w "Wired"      # O_O eyes
cowsay -e "OO" "Msg"   # Custom eyes
cowsay -T "U " "Msg"   # Custom tongue
```

## 工具 4：Boxes（装饰性边框）

在任意文本周围绘制装饰性 ASCII art 边框/框架。内置 70+ 种设计。

### 安装

```bash
sudo apt install boxes -y    # Debian/Ubuntu
# brew install boxes         # macOS
```

### 用法

```bash
echo "Hello World" | boxes                    # Default box
echo "Hello World" | boxes -d stone           # Stone border
echo "Hello World" | boxes -d parchment       # Parchment scroll
echo "Hello World" | boxes -d cat             # Cat border
echo "Hello World" | boxes -d dog             # Dog border
echo "Hello World" | boxes -d unicornsay      # Unicorn
echo "Hello World" | boxes -d diamonds        # Diamond pattern
echo "Hello World" | boxes -d c-cmt           # C-style comment
echo "Hello World" | boxes -d html-cmt        # HTML comment
echo "Hello World" | boxes -a c               # Center text
boxes -l                                       # List all 70+ designs
```

### 与 pyfiglet 或 asciified 组合使用

```bash
python3 -m pyfiglet "HERMES" -f slant | boxes -d stone
# Or without pyfiglet installed:
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=HERMES&font=Slant" | boxes -d stone
```

## 工具 5：TOIlet（彩色文字艺术）

类似 pyfiglet，但支持 ANSI 颜色效果和视觉滤镜。非常适合终端视觉效果。

### 安装

```bash
sudo apt install toilet toilet-fonts -y    # Debian/Ubuntu
# brew install toilet                      # macOS
```

### 用法

```bash
toilet "Hello World"                    # Basic text art
toilet -f bigmono12 "Hello"            # Specific font
toilet --gay "Rainbow!"                 # Rainbow coloring
toilet --metal "Metal!"                 # Metallic effect
toilet -F border "Bordered"             # Add border
toilet -F border --gay "Fancy!"         # Combined effects
toilet -f pagga "Block"                 # Block-style font (unique to toilet)
toilet -F list                          # List available filters
```

### 滤镜

`crop`、`gay`（彩虹）、`metal`、`flip`、`flop`、`180`、`left`、`right`、`border`

**注意**：toilet 输出带颜色的 ANSI 转义码——在终端中正常显示，但在某些场景下可能无法渲染（例如纯文本文件、部分聊天平台）。

## 工具 6：图片转 ASCII Art

将图片（PNG、JPEG、GIF、WEBP）转换为 ASCII art。

### 方案 A：ascii-image-converter（推荐，现代化）

```bash
# Install
sudo snap install ascii-image-converter
# OR: go install github.com/TheZoraiz/ascii-image-converter@latest
```

```bash
ascii-image-converter image.png                  # Basic
ascii-image-converter image.png -C               # Color output
ascii-image-converter image.png -d 60,30         # Set dimensions
ascii-image-converter image.png -b               # Braille characters
ascii-image-converter image.png -n               # Negative/inverted
ascii-image-converter https://url/image.jpg      # Direct URL
ascii-image-converter image.png --save-txt out   # Save as text
```

### 方案 B：jp2a（轻量级，仅支持 JPEG）

```bash
sudo apt install jp2a -y
jp2a --width=80 image.jpg
jp2a --colors image.jpg              # Colorized
```

## 工具 7：搜索预制 ASCII Art

从网络搜索精选 ASCII art。使用 `terminal` 配合 `curl`。

### 来源 A：ascii.co.uk（推荐用于预制艺术）

大量按主题分类的经典 ASCII art 合集。艺术内容位于 HTML `<pre>` 标签内。使用 curl 获取页面，再用简短的 Python 代码提取艺术内容。

**URL 格式：** `https://ascii.co.uk/art/{subject}`

**第一步——获取页面：**

```bash
curl -s 'https://ascii.co.uk/art/cat' -o /tmp/ascii_art.html
```

**第二步——从 pre 标签中提取艺术内容：**

```python
import re, html
with open('/tmp/ascii_art.html') as f:
    text = f.read()
arts = re.findall(r'<pre[^>]*>(.*?)</pre>', text, re.DOTALL)
for art in arts:
    clean = re.sub(r'<[^>]+>', '', art)
    clean = html.unescape(clean).strip()
    if len(clean) > 30:
        print(clean)
        print('\n---\n')
```

**可用主题**（用作 URL 路径）：
- 动物：`cat`、`dog`、`horse`、`bird`、`fish`、`dragon`、`snake`、`rabbit`、`elephant`、`dolphin`、`butterfly`、`owl`、`wolf`、`bear`、`penguin`、`turtle`
- 物品：`car`、`ship`、`airplane`、`rocket`、`guitar`、`computer`、`coffee`、`beer`、`cake`、`house`、`castle`、`sword`、`crown`、`key`
- 自然：`tree`、`flower`、`sun`、`moon`、`star`、`mountain`、`ocean`、`rainbow`
- 角色：`skull`、`robot`、`angel`、`wizard`、`pirate`、`ninja`、`alien`
- 节日：`christmas`、`halloween`、`valentine`

**提示：**
- 保留艺术家签名/缩写——这是重要的礼仪
- 每个页面包含多件艺术作品——为用户挑选最合适的
- 通过 curl 可靠运行，无需 JavaScript

### 来源 B：GitHub Octocat API（有趣的彩蛋）

返回一个带有智慧语录的随机 GitHub Octocat。无需认证。

```bash
curl -s https://api.github.com/octocat
```

## 工具 8：有趣的 ASCII 实用工具（通过 curl）

这些免费服务直接返回 ASCII art——非常适合作为有趣的附加内容。

### QR 码转 ASCII Art

```bash
curl -s "qrenco.de/Hello+World"
curl -s "qrenco.de/https://example.com"
```

### 天气转 ASCII Art

```bash
curl -s "wttr.in/London"          # Full weather report with ASCII graphics
curl -s "wttr.in/Moon"            # Moon phase in ASCII art
curl -s "v2.wttr.in/London"       # Detailed version
```

## 工具 9：LLM 生成自定义艺术（兜底方案）

当上述工具无法满足需求时，直接使用以下 Unicode 字符生成 ASCII art：

### 字符调色板

**方框绘制：** `╔ ╗ ╚ ╝ ║ ═ ╠ ╣ ╦ ╩ ╬ ┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ╭ ╮ ╰ ╯`

**块元素：** `░ ▒ ▓ █ ▄ ▀ ▌ ▐ ▖ ▗ ▘ ▝ ▚ ▞`

**几何与符号：** `◆ ◇ ◈ ● ○ ◉ ■ □ ▲ △ ▼ ▽ ★ ☆ ✦ ✧ ◀ ▶ ◁ ▷ ⬡ ⬢ ⌂`

### 规则

- 最大宽度：每行 60 个字符（终端安全）
- 最大高度：横幅 15 行，场景 25 行
- 仅限等宽字体：输出必须在等宽字体下正确渲染

## 决策流程

1. **将文本作为横幅** → 若已安装 pyfiglet 则使用，否则通过 curl 调用 asciified API
2. **将消息包裹在有趣的角色艺术中** → cowsay
3. **添加装饰性边框/框架** → boxes（可与 pyfiglet/asciified 组合使用）
4. **特定事物的艺术**（猫、火箭、龙）→ 通过 curl + 解析使用 ascii.co.uk
5. **将图片转换为 ASCII** → ascii-image-converter 或 jp2a
6. **QR 码** → 通过 curl 使用 qrenco.de
7. **天气/月相艺术** → 通过 curl 使用 wttr.in
8. **自定义/创意内容** → 使用 Unicode 调色板进行 LLM 生成
9. **任何工具未安装** → 安装它，或回退到下一个选项