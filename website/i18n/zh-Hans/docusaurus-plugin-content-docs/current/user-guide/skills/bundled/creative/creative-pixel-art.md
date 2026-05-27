---
title: "Pixel Art — 像素艺术（NES、Game Boy、PICO-8 时代调色板）"
sidebar_label: "Pixel Art"
description: "像素艺术（NES、Game Boy、PICO-8 时代调色板）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pixel Art

像素艺术（NES、Game Boy、PICO-8 时代调色板）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/pixel-art` |
| 版本 | `2.0.0` |
| 作者 | dodo-reach |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `creative`, `pixel-art`, `arcade`, `snes`, `nes`, `gameboy`, `retro`, `image`, `video` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Pixel Art

将任意图像转换为复古像素艺术，并可选地将其制作成带有时代感特效（雨、萤火虫、雪、余烬）的短 MP4 或 GIF 动画。

此 skill 附带两个脚本：

- `scripts/pixel_art.py` — 照片 → 像素艺术 PNG（Floyd-Steinberg 抖动算法）
- `scripts/pixel_art_video.py` — 像素艺术 PNG → 动画 MP4（+ 可选 GIF）

每个脚本均可作为模块导入或直接运行。预设可对齐硬件调色板以获得时代准确的色彩（NES、Game Boy、PICO-8 等），或使用自适应 N 色量化实现街机/SNES 风格。

## 使用场景

- 用户希望从源图像生成复古像素艺术
- 用户要求 NES / Game Boy / PICO-8 / C64 / 街机 / SNES 风格
- 用户需要短循环动画（雨景、夜空、雪景等）
- 海报、专辑封面、社交帖子、精灵图、角色、头像

## 工作流程

生成前，先与用户确认风格。不同预设产生的效果差异很大，重新生成代价较高。

### 第一步 — 提供风格选项

使用 `clarify` 提供 4 个代表性预设。根据用户的需求选择组合——不要一次性列出全部 14 个。

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

当用户已指定时代（如"80 年代街机"、"Gameboy"）时，跳过 `clarify`，直接使用对应预设。

### 第二步 — 提供动画选项（可选）

如果用户要求视频/GIF，或输出内容适合加入动效，询问选择哪个场景：

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

每轮最多调用 `clarify` 两次：一次选风格，一次选场景（如涉及动画）。若用户在消息中已明确指定风格和场景，则完全跳过 `clarify`。

### 第三步 — 生成

先运行 `pixel_art()`；若用户要求动画，则将结果传入 `pixel_art_video()`。

## 预设目录

| 预设 | 时代 | 调色板 | 像素块 | 适用场景 |
|--------|-----|---------|-------|----------|
| `arcade` | 80 年代街机 | 自适应 16 色 | 8px | 粗犷海报、主角艺术 |
| `snes` | 16 位 | 自适应 32 色 | 4px | 角色、细节场景 |
| `nes` | 8 位 | NES（54 色） | 8px | 真实 NES 风格 |
| `gameboy` | DMG 掌机 | 4 阶绿色 | 8px | 单色 Game Boy |
| `gameboy_pocket` | Pocket 掌机 | 4 阶灰色 | 8px | 单色 GB Pocket |
| `pico8` | PICO-8 | 16 固定色 | 6px | 幻想主机风格 |
| `c64` | Commodore 64 | 16 固定色 | 8px | 8 位家用电脑 |
| `apple2` | Apple II 高分辨率 | 6 固定色 | 10px | 极致复古，6 色 |
| `teletext` | BBC Teletext | 8 纯色 | 10px | 粗犷原色块 |
| `mspaint` | Windows MS Paint | 24 固定色 | 8px | 怀旧桌面风格 |
| `mono_green` | CRT 荧光绿 | 2 绿色 | 6px | 终端/CRT 美学 |
| `mono_amber` | CRT 琥珀色 | 2 琥珀色 | 6px | 琥珀显示器风格 |
| `neon` | 赛博朋克 | 10 霓虹色 | 6px | 蒸汽波/赛博风 |
| `pastel` | 柔和粉彩 | 10 粉彩色 | 6px | 可爱风 / 温柔风 |

命名调色板位于 `scripts/palettes.py`（完整列表见 `references/palettes.md`，共 28 个命名调色板）。任何预设均可覆盖：

```python
pixel_art("in.png", "out.png", preset="snes", palette="PICO_8", block=6)
```

## 场景目录（用于视频）

| 场景 | 特效 |
|-------|---------|
| `night` | 闪烁星星 + 萤火虫 + 飘落树叶 |
| `dusk` | 萤火虫 + 闪光 |
| `tavern` | 尘埃粒子 + 暖色闪光 |
| `indoor` | 尘埃粒子 |
| `urban` | 雨 + 霓虹脉冲 |
| `nature` | 树叶 + 萤火虫 |
| `magic` | 闪光 + 萤火虫 |
| `storm` | 雨 + 闪电 |
| `underwater` | 气泡 + 光斑 |
| `fire` | 余烬 + 闪光 |
| `snow` | 雪花 + 闪光 |
| `desert` | 热浪扭曲 + 尘埃 |

## 调用方式

### Python（导入）

```python
import sys
sys.path.insert(0, "/home/teknium/.hermes/skills/creative/pixel-art/scripts")
from pixel_art import pixel_art
from pixel_art_video import pixel_art_video

# 1. 转换为像素艺术
pixel_art("/path/to/photo.jpg", "/tmp/pixel.png", preset="nes")

# 2. 制作动画（可选）
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
1. 增强对比度/色彩/锐度（调色板越小，增强越强）
2. 色调分离，在量化前简化色调区域
3. 以 `block` 为步长使用 `Image.NEAREST` 缩小（硬像素，无插值）
4. 使用 Floyd-Steinberg 抖动进行量化——针对自适应 N 色调色板或命名硬件调色板
5. 使用 `Image.NEAREST` 放大还原

在缩小后再量化，可使抖动与最终像素网格对齐。若先量化再缩小，会将误差扩散浪费在最终消失的细节上。

**视频叠加：**
- 每帧复制基础帧（静态背景）
- 叠加无状态的逐帧粒子绘制（每种特效一个函数）
- 通过 ffmpeg `libx264 -pix_fmt yuv420p -crf 18` 编码
- 可选 GIF，通过 `palettegen` + `paletteuse` 生成

## 依赖项

- Python 3.9+
- Pillow（`pip install Pillow`）
- PATH 中的 ffmpeg（仅视频需要——Hermes 会安装此包）

## 注意事项

- 调色板键名区分大小写（`"NES"`、`"PICO_8"`、`"GAMEBOY_ORIGINAL"`）。
- 非常小的源图像（宽度 &lt;100px）在 8-10px 像素块下会崩溃。若源图太小，请先放大。
- `block` 或 `palette` 为小数时会破坏量化——保持为正整数。
- 动画粒子数量针对约 640x480 画布调优。对于非常大的图像，可能需要用不同 seed 进行第二次处理以调整密度。
- `mono_green` / `mono_amber` 强制 `color=0.0`（去饱和）。若覆盖并保留色度，2 色调色板在平滑区域可能产生条纹。
- `clarify` 循环：每轮最多调用两次（风格，然后是场景）。不要反复向用户询问选项。

## 验证

- PNG 已在输出路径创建
- 在预设像素块大小下可见清晰的方形像素块
- 色彩数量与预设匹配（目视检查图像或运行 `Image.open(p).getcolors()`）
- 视频为有效 MP4（`ffprobe` 可打开）且大小非零

## 致谢

命名硬件调色板及 `pixel_art_video.py` 中的程序化动画循环移植自 [pixel-art-studio](https://github.com/Synero/pixel-art-studio)（MIT 许可证）。详见此 skill 目录中的 `ATTRIBUTION.md`。