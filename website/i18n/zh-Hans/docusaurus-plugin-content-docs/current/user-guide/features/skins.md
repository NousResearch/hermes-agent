---
sidebar_position: 10
title: "皮肤与主题"
description: "使用内置和用户自定义皮肤定制 Hermes CLI 的外观"
---

# 皮肤与主题

皮肤控制 Hermes CLI 的**视觉呈现**：横幅颜色、spinner（加载动画）面孔与动词、响应框标签、品牌文本以及工具活动前缀。

对话风格与视觉风格是两个独立的概念：

- **Personality（个性）** 改变 agent 的语气和措辞。
- **Skin（皮肤）** 改变 CLI 的外观。

## 切换皮肤

```bash
/skin                # show the current skin and list available skins
/skin ares           # switch to a built-in skin
/skin mytheme        # switch to a custom skin from ~/.hermes/skins/mytheme.yaml
```

或在 `~/.hermes/config.yaml` 中设置默认皮肤：

```yaml
display:
  skin: default
```

## 内置皮肤

| 皮肤 | 描述 | Agent 品牌 | 视觉特征 |
|------|------|-----------|---------|
| `default` | 经典 Hermes — 金色与 kawaii 风格 | `Hermes Agent` | 暖金色边框，cornsilk 文字，spinner 中的 kawaii 面孔。熟悉的双蛇杖横幅。简洁亲切。 |
| `ares` | 战神主题 — 深红与青铜 | `Ares Agent` | 深红色边框配青铜点缀。激进的 spinner 动词（"forging"、"marching"、"tempering steel"）。自定义剑盾 ASCII 艺术横幅。 |
| `mono` | 单色 — 简洁灰度 | `Hermes Agent` | 全灰色，无彩色。边框为 `#555555`，文字为 `#c9d1d9`。适合极简终端或录屏场景。 |
| `slate` | 冷蓝色 — 面向开发者 | `Hermes Agent` | 皇家蓝边框（`#4169e1`），柔和蓝色文字。沉稳专业。无自定义 spinner，使用默认面孔。 |
| `daylight` | 适用于亮色终端的浅色主题，深色文字配冷蓝点缀 | `Hermes Agent` | 专为白色或亮色终端设计。深石板色文字配蓝色边框，浅色状态面板，补全菜单在亮色终端配置下保持清晰可读。 |
| `warm-lightmode` | 适用于浅色终端背景的暖棕/金色文字 | `Hermes Agent` | 适合浅色终端的暖羊皮纸色调。深棕色文字配马鞍棕点缀，奶油色状态面板。比 daylight 主题更温暖的大地色系选择。 |
| `poseidon` | 海神主题 — 深蓝与海沫绿 | `Poseidon Agent` | 深蓝到海沫绿渐变。海洋主题 spinner（"charting currents"、"sounding the depth"）。三叉戟 ASCII 艺术横幅。 |
| `sisyphus` | 西西弗斯主题 — 朴素灰度，彰显坚韧 | `Sisyphus Agent` | 浅灰色配强烈对比。巨石主题 spinner（"pushing uphill"、"resetting the boulder"、"enduring the loop"）。巨石与山丘 ASCII 艺术横幅。 |
| `charizard` | 火山主题 — 焦橙与余烬色 | `Charizard Agent` | 暖焦橙到余烬色渐变。火焰主题 spinner（"banking into the draft"、"measuring burn"）。龙剪影 ASCII 艺术横幅。 |

## 可配置键完整列表

### 颜色（`colors:`）

控制 CLI 中所有颜色值。值为十六进制颜色字符串。

| 键 | 描述 | 默认值（`default` 皮肤） |
|----|------|------------------------|
| `banner_border` | 启动横幅周围的面板边框 | `#CD7F32`（青铜色） |
| `banner_title` | 横幅中的标题文字颜色 | `#FFD700`（金色） |
| `banner_accent` | 横幅中的区块标题（Available Tools 等） | `#FFBF00`（琥珀色） |
| `banner_dim` | 横幅中的弱化文字（分隔符、次要标签） | `#B8860B`（暗金菊色） |
| `banner_text` | 横幅中的正文文字（工具名、技能名） | `#FFF8DC`（玉米丝色） |
| `ui_accent` | 通用 UI 强调色（高亮、活动元素） | `#FFBF00` |
| `ui_label` | UI 标签与标记 | `#4dd0e1`（青色） |
| `ui_ok` | 成功指示器（对勾、完成） | `#4caf50`（绿色） |
| `ui_error` | 错误指示器（失败、阻断） | `#ef5350`（红色） |
| `ui_warn` | 警告指示器（注意、审批提示） | `#ffa726`（橙色） |
| `prompt` | 交互式 prompt（提示符）文字颜色 | `#FFF8DC` |
| `input_rule` | 输入区域上方的水平分隔线 | `#CD7F32` |
| `response_border` | agent 响应框边框（ANSI 转义） | `#FFD700` |
| `session_label` | 会话标签颜色 | `#DAA520` |
| `session_border` | 会话 ID 弱化边框颜色 | `#8B8682` |
| `status_bar_bg` | TUI 状态/用量栏的背景色 | `#1a1a2e` |
| `voice_status_bg` | 语音模式状态徽章的背景色 | `#1a1a2e` |
| `selection_bg` | TUI 鼠标选区高亮的背景色。未设置时回退到 `completion_menu_current_bg`。 | `#333355` |
| `completion_menu_bg` | 补全菜单列表的背景色 | `#1a1a2e` |
| `completion_menu_current_bg` | 当前活动补全行的背景色 | `#333355` |
| `completion_menu_meta_bg` | 补全元信息列的背景色 | `#1a1a2e` |
| `completion_menu_meta_current_bg` | 当前活动补全元信息列的背景色 | `#333355` |

### Spinner（`spinner:`）

控制等待 API 响应时显示的动画 spinner。

| 键 | 类型 | 描述 | 示例 |
|----|------|------|------|
| `waiting_faces` | 字符串列表 | 等待 API 响应时循环显示的面孔 | `["(⚔)", "(⛨)", "(▲)"]` |
| `thinking_faces` | 字符串列表 | 模型推理期间循环显示的面孔 | `["(⚔)", "(⌁)", "(<>)"]` |
| `thinking_verbs` | 字符串列表 | spinner 消息中显示的动词 | `["forging", "plotting", "hammering plans"]` |
| `wings` | [左, 右] 对的列表 | spinner 周围的装饰括号 | `[["⟪⚔", "⚔⟫"], ["⟪▲", "▲⟫"]]` |

当 spinner 值为空时（如 `default` 和 `mono`），将使用 `display.py` 中的硬编码默认值。

### 品牌（`branding:`）

CLI 界面中使用的文字字符串。

| 键 | 描述 | 默认值 |
|----|------|--------|
| `agent_name` | 横幅标题和状态显示中的名称 | `Hermes Agent` |
| `welcome` | CLI 启动时显示的欢迎消息 | `Welcome to Hermes Agent! Type your message or /help for commands.` |
| `goodbye` | 退出时显示的消息 | `Goodbye! ⚕` |
| `response_label` | 响应框标题上的标签 | ` ⚕ Hermes ` |
| `prompt_symbol` | 用户输入 prompt 前的符号（裸 token，渲染器会在后面添加空格） | `❯` |
| `help_header` | `/help` 命令输出的标题文字 | `(^_^)? Available Commands` |

### 其他顶级键

| 键 | 类型 | 描述 | 默认值 |
|----|------|------|--------|
| `tool_prefix` | 字符串 | CLI 中工具输出行的前缀字符 | `┊` |
| `tool_emojis` | 字典 | 各工具的 emoji 覆盖，用于 spinner 和进度显示（`{tool_name: emoji}`） | `{}` |
| `banner_logo` | 字符串 | Rich 标记 ASCII 艺术 logo（替换默认的 HERMES_AGENT 横幅） | `""` |
| `banner_hero` | 字符串 | Rich 标记英雄艺术图（替换默认的双蛇杖图案） | `""` |

## 自定义皮肤

在 `~/.hermes/skins/` 下创建 YAML 文件。用户皮肤会从内置 `default` 皮肤继承缺失的值，因此只需指定要更改的键。

### 完整自定义皮肤 YAML 模板

```yaml
# ~/.hermes/skins/mytheme.yaml
# Complete skin template — all keys shown. Delete any you don't need;
# missing values automatically inherit from the 'default' skin.

name: mytheme
description: My custom theme

colors:
  banner_border: "#CD7F32"
  banner_title: "#FFD700"
  banner_accent: "#FFBF00"
  banner_dim: "#B8860B"
  banner_text: "#FFF8DC"
  ui_accent: "#FFBF00"
  ui_label: "#4dd0e1"
  ui_ok: "#4caf50"
  ui_error: "#ef5350"
  ui_warn: "#ffa726"
  prompt: "#FFF8DC"
  input_rule: "#CD7F32"
  response_border: "#FFD700"
  session_label: "#DAA520"
  session_border: "#8B8682"
  status_bar_bg: "#1a1a2e"
  voice_status_bg: "#1a1a2e"
  selection_bg: "#333355"
  completion_menu_bg: "#1a1a2e"
  completion_menu_current_bg: "#333355"
  completion_menu_meta_bg: "#1a1a2e"
  completion_menu_meta_current_bg: "#333355"

spinner:
  waiting_faces:
    - "(⚔)"
    - "(⛨)"
    - "(▲)"
  thinking_faces:
    - "(⚔)"
    - "(⌁)"
    - "(<>)"
  thinking_verbs:
    - "processing"
    - "analyzing"
    - "computing"
    - "evaluating"
  wings:
    - ["⟪⚡", "⚡⟫"]
    - ["⟪●", "●⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome to My Agent! Type your message or /help for commands."
  goodbye: "See you later! ⚡"
  response_label: " ⚡ My Agent "
  prompt_symbol: "⚡"
  help_header: "(⚡) Available Commands"

tool_prefix: "┊"

# Per-tool emoji overrides (optional)
tool_emojis:
  terminal: "⚔"
  web_search: "🔮"
  read_file: "📄"

# Custom ASCII art banners (optional, Rich markup supported)
# banner_logo: |
#   [bold #FFD700] MY AGENT [/]
# banner_hero: |
#   [#FFD700]  Custom art here  [/]
```

### 最简自定义皮肤示例

由于所有值都继承自 `default`，最简皮肤只需指定要更改的部分：

```yaml
name: cyberpunk
description: Neon terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

## Hermes Mod — 可视化皮肤编辑器

[Hermes Mod](https://github.com/cocktailpeanut/hermes-mod) 是一个社区构建的 Web UI，用于可视化创建和管理皮肤。无需手写 YAML，提供带实时预览的点击式编辑器。

![Hermes Mod skin editor](https://raw.githubusercontent.com/cocktailpeanut/hermes-mod/master/nous.png)

**功能说明：**

- 列出所有内置和自定义皮肤
- 将任意皮肤在可视化编辑器中打开，涵盖所有 Hermes 皮肤字段（颜色、spinner、品牌、工具前缀、工具 emoji）
- 根据文字 prompt 生成 `banner_logo` 文字艺术
- 将上传的图片（PNG、JPG、GIF、WEBP）转换为 `banner_hero` ASCII 艺术，支持多种渲染风格（盲文点阵、ASCII 字符渐变、方块、点阵）
- 直接保存到 `~/.hermes/skins/`
- 通过更新 `~/.hermes/config.yaml` 激活皮肤
- 显示生成的 YAML 及实时预览

### 安装

**方式一 — Pinokio（一键安装）：**

在 [pinokio.computer](https://pinokio.computer) 上找到并一键安装。

**方式二 — npx（终端最快方式）：**

```bash
npx -y hermes-mod
```

**方式三 — 手动安装：**

```bash
git clone https://github.com/cocktailpeanut/hermes-mod.git
cd hermes-mod/app
npm install
npm start
```

### 使用方法

1. 启动应用（通过 Pinokio 或终端）。
2. 打开 **Skin Studio**。
3. 选择要编辑的内置或自定义皮肤。
4. 从文字生成 logo，和/或上传图片作为英雄艺术图。选择渲染风格和宽度。
5. 编辑颜色、spinner、品牌及其他字段。
6. 点击 **Save** 将皮肤 YAML 写入 `~/.hermes/skins/`。
7. 点击 **Activate** 将其设为当前皮肤（更新 `config.yaml` 中的 `display.skin`）。

Hermes Mod 遵循 `HERMES_HOME` 环境变量，因此也适用于[配置文件](/user-guide/profiles)。

## 操作说明

- 内置皮肤从 `hermes_cli/skin_engine.py` 加载。
- 未知皮肤自动回退到 `default`。
- `/skin` 立即更新当前会话的活动 CLI 主题。
- `~/.hermes/skins/` 中的用户皮肤优先于同名内置皮肤。
- 通过 `/skin` 切换皮肤仅对当前会话有效。如需永久设为默认皮肤，请在 `config.yaml` 中配置。
- `banner_logo` 和 `banner_hero` 字段支持 Rich 控制台标记（例如 `[bold #FF0000]text[/]`），可用于彩色 ASCII 艺术。