---
title: "Pixel Art — 像素画：怀旧主机调色板（NES、Game Boy、PICO-8）"
sidebar_label: "Pixel Art"
description: "像素画：怀旧主机调色板（NES、Game Boy、PICO-8）"
---

 {/* 此页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Pixel Art

像素画：怀旧主机调色板（NES、Game Boy、PICO-8）。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/pixel-art` |
| 版本 | `2.0.0` |
| 作者 | dodo-reach |
| 许可证 | MIT |
| 标签 | `creative`, `pixel-art`, `arcade`, `snes`, `nes`, `gameboy`, `retro`, `image`, `video` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时智能体看到的指令。
:::

# Pixel Art

将任意图片转换为复古像素画，然后可选地将其制作为带时代特效的短视频 MP4 或 GIF（雨、萤火虫、雪、余烬）。

此技能附带两个脚本：

- `scripts/pixel_art.py` — 图片 → 像素画 PNG（Floyd-Steinberg 抖动）
- `scripts/pixel_art_video.py` — 像素画 PNG → 动态 MP4（+ 可选 GIF）

每个脚本都可导入或直接运行。当您想要时代精确的颜色（NES、Game Boy、PICO-8 等）时，预设会锁定到硬件调色板，或者使用自适应 N 色量化来实现街机/SNES 风格效果。

## 使用场景

- 用户想要从源图片生成复古像素画
- 用户要求 NES / Game Boy / PICO-8 / C64 / 街机 / SNES 风格
- 用户想要循环短视频（雨景、夜空、雪景等）
- 海报、专辑封面、社交帖子、精灵图、角色、头像

## 工作流程

生成前，先与用户确认风格。不同的预设会产生截然不同的输出，重新生成成本较高。

### 第一步 — 提供风格选择

调用 `clarify` 提供 4 个代表性预设。根据用户需求选择预设组合 — 不要一次性展示全部 14 个。

当用户意图不明确时的默认菜单：

```python
clarify(
    question="Which pixel-art style do you want?",
    choices=[
        "arcade — bold, chunky 80s cabinet feel (16 colors, 8px)",
        "nes — Nintendo 8-bit hardware palette (54 colors, 8px)",
        "gameboy — 4-shade green Game Boy DMG",
        "snes — cleaner 16-bit look (32 colors, 4px)",
    ],
)
```

当用户已指定时代（如"80年代街机"、"Gameboy"）时，跳过 `clarify` 直接使用匹配的预设。

### 第二步 — 提供动画选择（可选）

如果用户要求视频/GIF，或输出可能受益于动态效果，询问场景选择：

```python
clarify(
    question="Want to animate it? Pick a scene or skip.",
    choices=[
        "night — stars + fireflies + leaves",
        "urban — rain + neon pulse",
        "snow — falling snowflakes",
        "skip — just the image",
    ],
)
```

不要连续调用 `clarify` 超过两次。风格一次，场景一次（如需动画）。如果用户在消息中已明确要求特定风格和场景，完全跳过 `clarify`。

### 第三步 — 生成

先运行 `pixel_art()`；如需动画，将结果链式传入 `pixel_art_video()`。

## 预设目录

| 预设 | 时代 | 调色板 | 像素块 | 最适合 |
|--------|-----|---------|-------|----------|
| `arcade` | 80年代街机 | 自适应 16 色 | 8px | 大胆的海报、英雄画面 |
| `snes` | 16位 | 自适应 32 色 | 4px | 角色、细节丰富的场景 |
| `nes` | 8位 | NES (54) | 8px | 真正的 NES 外观 |
| `gameboy` | DMG 手持 | 4 色绿色 | 8px | 单色 Game Boy |
| `gameboy_pocket` | Pocket 手持 | 4 色灰色 | 8px | 单色 GB Pocket |
| `pico8` | PICO-8 | 16 固定 | 6px | 幻想主机外观 |
| `c64` | Commodore 64 | 16 固定 | 8px | 8位家用电脑 |
| `apple2` | Apple II 高分辨率 | 6 固定 | 10px | 极致复古，6 色 |
| `teletext` | BBC Teletext | 8 纯色 | 10px | 厚重的原色 |
| `mspaint` | Windows 画图 | 24 固定 | 8px | 怀旧的桌面 |
| `mono_green` | CRT 磷光 | 2 绿色 | 6px | 终端/CRT 美学 |
| `mono_amber` | CRT 琥珀色 | 2 琥珀色 | 6px | 琥珀色显示器外观 |
| `neon` | 赛博朋克 | 10 霓虹色 | 6px | 蒸汽波/赛博 |
| `pastel` | 柔和粉彩 | 10 粉彩色 | 6px | 卡哇伊/柔和 |

命名调色板位于 `scripts/palettes.py`（完整列表见 `references/palettes.md` — 共 28 个命名调色板）。任何预设都可以覆盖：

```python
pixel_art("in.png", "out.png", preset="snes", palette="PICO_8", block=6)
```

## 场景目录（用于视频）

| 场景 | 特效 |
|-------|---------|
| `night` | 闪烁星星 + 萤火虫 + 飘落的树叶 |
| `dusk` | 萤火虫 + 闪光 |
| `tavern` | 尘埃微粒 + 温暖的闪光 |
| `indoor` | 尘埃微粒 |
| `urban` | 雨 + 霓虹跳动 |
| `nature` | 树叶 + 萤火虫 |
| `magic` | 闪光 + 萤火虫 |
| `storm` | 雨 + 闪电 |
| `underwater` | 气泡 + 灯光闪烁 |
| `fire` | 余烬 + 闪光 |
| `snow` | 雪花 + 闪光 |
| `desert` | 热浪 + 尘埃 |

## 调用模式

### Python（导入）

```python
import sys
sys.path.insert(0, "/home/teknium/.hermes/skills/creative/pixel-art/scripts")
from pixel_art import pixel_art
from pixel_art_video import pixel_art_video

# 1. 转换为像素画
pixel_art("/path/to/photo.jpg", "/tmp/pixel.png", preset="nes")

# 2. 动画（可选）
pixel_art_video(
    "/tmp/pixel.png",
    "/tmp/pixel.mp4",
    scene="night",
    duration=6,
    fps=15,
    seed=42,
    export_gif=True,
)
```

### CLI

```bash
cd /home/teknium/.hermes/skills/creative/pixel-art/scripts

python pixel_art.py in.jpg out.png --preset gameboy
python pixel_art.py in.jpg out.png --preset snes --palette PICO_8 --block 6

python pixel_art_video.py out.png out.mp4 --scene night --duration 6 --gif
```

## 流水线原理

**像素转换：**
1. 增强对比度/颜色/锐度（对于较少颜色的调色板效果更强）
2. 色调分离以简化量化前的色调区域
3. 使用 `Image.NEAREST` 缩小 `block` 倍（硬像素，无插值）
4. 使用 Floyd-Steinberg 抖动进行量化 — 适配色板或命名硬件调色板的自适应 N 色
5. 使用 `Image.NEAREST` 放大回原尺寸

在缩小后进行量化可保持抖动与最终像素网格对齐。在缩小前量化会浪费误差扩散在消失的细节上。

**视频叠加：**
- 每帧复制基础帧（静态背景）
- 叠加无状态粒子绘制（每个特效一个函数）
- 通过 ffmpeg `libx264 -pix_fmt yuv420p -crf 18` 编码
- 可选 GIF 通过 `palettegen` + `paletteuse`

## 依赖

- Python 3.9+
- Pillow（`pip install Pillow`）
- PATH 中的 ffmpeg（仅视频需要 — Hermes 安装时会安装此包）

## 陷阱

- 调色板键名区分大小写（`"NES"`, `"PICO_8"`, `"GAMEBOY_ORIGINAL"`）。
- 非常小的源图（<100px 宽）在 8-10px 像素块下会崩溃。如果源图太小，先放大。
- 分数的 `block` 或 `palette` 会破坏量化 — 保持为正整数。
- 动画粒子数量针对约 640x480 画布调整。对于非常大的图片，可能需要用不同的 seed 进行第二次处理以调整密度。
- `mono_green` / `mono_amber` 强制 `color=0.0`（去饱和）。如果覆盖并保留色度，2 色调色板可能在平滑区域产生条纹。
- `clarify` 循环：每轮最多调用两次（风格，然后是场景）。不要用更多的选择来打扰用户。

## 验证

- PNG 在输出路径创建
- 在预设的像素块大小下可见清晰的方形像素块
- 颜色数量与预设匹配（目测或运行 `Image.open(p).getcolors()`）
- 视频是有效的 MP4（`ffprobe` 可打开）且大小非零

## 归属

命名硬件调色板和 `pixel_art_video.py` 中的程序化动画循环改编自 [pixel-art-studio](https://github.com/Synero/pixel-art-studio)（MIT）。详见此技能目录中的 `ATTRIBUTION.md`。
