---
title: "Pixel Art — 像素艺术（NES、Game Boy、PICO-8 时代配色）"
sidebar_label: "Pixel Art"
description: "像素艺术（NES、Game Boy、PICO-8 时代配色）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pixel Art

将任何图像转换为复古像素艺术，然后可选地将其动画化为带时代适当效果（雨、萤火虫、雪花、火星）的短 MP4 或 GIF。

此技能附带两个脚本：
- `scripts/pixel_art.py` — 照片 → 像素艺术 PNG（Floyd-Steinberg 抖动）
- `scripts/pixel_art_video.py` — 像素艺术 PNG → 动画 MP4（+ 可选 GIF）

每个都可导入或直接运行。预设在需要时代精确颜色时对齐到硬件调色板（NES、Game Boy、PICO-8 等），或使用自适应 N 色量化获得街机/SNES 风格外观。

## 何时使用

- 用户想要从源图像获得复古像素艺术
- 用户请求 NES / Game Boy / PICO-8 / C64 / 街机 / SNES 风格
- 用户想要短循环动画（雨景、星空、雪花等）
- 海报、专辑封面、社交帖子、精灵图、角色、头像

## 工作流程

### 步骤 1 — 提供风格

使用 4 个代表性预设调用 `clarify`。

### 步骤 2 — 提供动画（可选）

如果用户请求视频/GIF 或输出可能受益于运动，询问场景。

### 步骤 3 — 生成

先运行 `pixel_art()`；如请求了动画，链接到 `pixel_art_video()`。

## 预设目录

| 预设 | 时代 | 调色板 | 块大小 | 最适合 |
|------|------|--------|--------|--------|
| `arcade` | 80年代街机 | 自适应 16 | 8px | 大胆海报、主角艺术 |
| `snes` | 16位 | 自适应 32 | 4px | 角色、详细场景 |
| `nes` | 8位 | NES (54) | 8px | 真实 NES 外观 |
| `gameboy` | DMG掌机 | 4种绿色 | 8px | 单色 Game Boy |
| `gameboy_pocket` | Pocket掌机 | 4种灰色 | 8px | 单色 GB Pocket |
| `pico8` | PICO-8 | 16固定 | 6px | 幻想主机外观 |
| `c64` | Commodore 64 | 16固定 | 8px | 8位家用电脑 |
| `apple2` | Apple II 高分辨率 | 6固定 | 10px | 极致复古，6色 |
| `teletext` | BBC Teletext | 8纯色 | 10px | 粗犷原色 |
| `mspaint` | Windows MS Paint | 24固定 | 8px | 怀旧桌面 |
| `mono_green` | CRT磷光 | 2绿色 | 6px | 终端/CRT美学 |
| `mono_amber` | CRT琥珀 | 2琥珀 | 6px | 琥珀显示器外观 |
| `neon` | 赛博朋克 | 10霓虹 | 6px | 蒸汽波/赛博 |
| `pastel` | 柔和粉彩 | 10粉彩 | 6px | 可爱/温柔 |

## 场景目录（用于视频）

| 场景 | 效果 |
|------|------|
| `night` | 闪烁星星 + 萤火虫 + 飘落树叶 |
| `dusk` | 萤火虫 + 闪光 |
| `tavern` | 灰尘颗粒 + 温暖闪光 |
| `indoor` | 灰尘颗粒 |
| `urban` | 雨 + 霓虹脉冲 |
| `nature` | 树叶 + 萤火虫 |
| `magic` | 闪光 + 萤火虫 |
| `storm` | 雨 + 闪电 |
| `underwater` | 气泡 + 光闪光 |
| `fire` | 火星 + 闪光 |
| `snow` | 雪花 + 闪光 |
| `desert` | 热浪 + 灰尘 |

## 调用模式

### Python（导入）

```python
import sys
sys.path.insert(0, "/home/teknium/.hermes/skills/creative/pixel-art/scripts")
from pixel_art import pixel_art
from pixel_art_video import pixel_art_video

# 1. 转换为像素艺术
pixel_art("/path/to/photo.jpg", "/tmp/pixel.png", preset="nes")

# 2. 动画化（可选）
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

## 陷阱

- 调色板键区分大小写（`"NES"`、`"PICO_8"`、`"GAMEBOY_ORIGINAL"`）。
- 非常小的源图像（&lt;100px 宽）在 8-10px 块下会崩溃。先放大源图像。
- 分数 `block` 或 `palette` 会破坏量化——保持正整数。
- `clarify` 循环：每回合最多调用两次（风格，然后场景）。
