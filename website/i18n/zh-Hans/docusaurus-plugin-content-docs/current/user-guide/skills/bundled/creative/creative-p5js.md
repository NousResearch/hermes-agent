---
title: "P5Js — p5"
sidebar_label: "P5Js"
description: "p5"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# P5Js

p5.js 草图：生成艺术、着色器、交互、3D。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/p5js` |
| 版本 | `1.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `creative-coding`, `generative-art`, `p5js`, `canvas`, `interactive`, `visualization`, `webgl`, `shaders`, `animation` |
| 相关 skill | [`ascii-video`](/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/user-guide/skills/bundled/creative/creative-manim-video), [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# p5.js 生产流水线

## 适用场景

当用户请求以下内容时使用：p5.js 草图、创意编程、生成艺术、交互式可视化、canvas 动画、基于浏览器的视觉艺术、数据可视化、着色器效果，或任何 p5.js 项目。

## 内容概览

用于交互式和生成式视觉艺术的生产流水线，基于 p5.js。可创建基于浏览器的草图、生成艺术、数据可视化、交互体验、3D 场景、音频响应式视觉效果和动态图形——导出格式支持 HTML、PNG、GIF、MP4 或 SVG。涵盖：2D/3D 渲染、噪声与粒子系统、流场、着色器（GLSL）、像素操作、动态排版、WebGL 场景、音频分析、鼠标/键盘交互，以及无头高分辨率导出。

## 创意标准

这是在浏览器中渲染的视觉艺术。canvas 是媒介，算法是画笔。

**在写下第一行代码之前**，先阐明创意概念。这件作品传达什么？什么能让观者停止滑动屏幕？什么使它区别于一个代码教程示例？用户的 prompt（提示词）只是起点——以创意野心去诠释它。

**首次渲染必须出色。** 输出在首次加载时必须在视觉上令人印象深刻。如果它看起来像 p5.js 教程练习、默认配置或"AI 生成的创意编程"，那就是错的。在交付前重新思考。

**超越参考词汇。** 参考资料中的噪声函数、粒子系统、色彩调色板和着色器效果只是起始词汇。每个项目都要组合、叠加和创造。目录是颜料的调色板——你来写这幅画。

**主动发挥创意。** 如果用户要求"一个粒子系统"，就交付一个具有涌现群集行为、拖尾幽灵回声、调色板偏移深度雾，以及会呼吸的背景噪声场的粒子系统。至少包含一个用户没有要求但会欣赏的视觉细节。

**密集、分层、深思熟虑。** 每一帧都应值得细看。绝不使用纯白背景。始终保持构图层次。始终使用有意图的色彩。始终有只在近距离观察时才会出现的微观细节。

**统一美学优于功能数量。** 所有元素必须服务于统一的视觉语言——共享的色温、一致的描边粗细词汇、和谐的运动速度。一个有十种不相关效果的草图，不如一个有三种相互呼应效果的草图。

## 模式

| 模式 | 输入 | 输出 | 参考 |
|------|-------|--------|-----------|
| **生成艺术** | 种子 / 参数 | 程序化视觉构图（静态或动态） | `references/visual-effects.md` |
| **数据可视化** | 数据集 / API | 交互式图表、图形、自定义数据展示 | `references/interaction.md` |
| **交互体验** | 无（用户驱动） | 鼠标/键盘/触控驱动的草图 | `references/interaction.md` |
| **动画 / 动态图形** | 时间轴 / 故事板 | 定时序列、动态排版、过渡效果 | `references/animation.md` |
| **3D 场景** | 概念描述 | WebGL 几何体、光照、摄像机、材质 | `references/webgl-and-3d.md` |
| **图像处理** | 图像文件 | 像素操作、滤镜、马赛克、点彩 | `references/visual-effects.md` § Pixel Manipulation |
| **音频响应式** | 音频文件 / 麦克风 | 声音驱动的生成视觉效果 | `references/interaction.md` § Audio Input |

## 技术栈

每个项目为单个自包含 HTML 文件，无需构建步骤。

| 层级 | 工具 | 用途 |
|-------|------|---------|
| 核心 | p5.js 1.11.3（CDN） | Canvas 渲染、数学运算、变换、事件处理 |
| 3D | p5.js WebGL 模式 | 3D 几何体、摄像机、光照、GLSL 着色器 |
| 音频 | p5.sound.js（CDN） | FFT 分析、振幅、麦克风输入、振荡器 |
| 导出 | 内置 `saveCanvas()` / `saveGif()` / `saveFrames()` | PNG、GIF、帧序列输出 |
| 捕获 | CCapture.js（可选） | 确定性帧率视频捕获（WebM、GIF） |
| 无头渲染 | Puppeteer + Node.js（可选） | 自动化高分辨率渲染，通过 ffmpeg 生成 MP4 |
| SVG | p5.js-svg 1.6.0（可选） | 用于印刷的矢量输出——需要 p5.js 1.x |
| 自然媒介 | p5.brush（可选） | 水彩、炭笔、钢笔——需要 p5.js 2.x + WEBGL |
| 纹理 | p5.grain（可选） | 胶片颗粒、纹理叠加 |
| 字体 | Google Fonts / `loadFont()` | 通过 OTF/TTF/WOFF2 使用自定义字体 |

### 版本说明

**p5.js 1.x**（1.11.3）是默认版本——稳定、文档完善、库兼容性最广。除非项目需要 2.x 特性，否则使用此版本。

**p5.js 2.x**（2.2+）新增：`async setup()` 替代 `preload()`、OKLCH/OKLAB 色彩模式、`splineVertex()`、着色器 `.modify()` API、可变字体、`textToContours()`、pointer 事件。p5.brush 需要此版本。参见 `references/core-api.md` § p5.js 2.0。

## 流水线

每个项目遵循相同的 6 阶段路径：

```
概念 → 设计 → 编码 → 预览 → 导出 → 验证
```

1. **概念** — 阐明创意愿景：氛围、色彩世界、运动词汇、使其独特的要素
2. **设计** — 选择模式、canvas 尺寸、交互模型、色彩系统、导出格式。将概念映射到技术决策
3. **编码** — 编写内联 p5.js 的单一 HTML 文件。结构：全局变量 → `preload()` → `setup()` → `draw()` → 辅助函数 → 类 → 事件处理器
4. **预览** — 在浏览器中打开，验证视觉质量。在目标分辨率下测试。检查性能
5. **导出** — 捕获输出：PNG 用 `saveCanvas()`，GIF 用 `saveGif()`，MP4 用 `saveFrames()` + ffmpeg，无头批量用 Puppeteer
6. **验证** — 输出是否符合概念？在预期显示尺寸下是否视觉震撼？你会把它裱起来吗？

## 创意方向

### 美学维度

| 维度 | 选项 | 参考 |
|-----------|---------|-----------|
| **色彩系统** | HSB/HSL、RGB、命名调色板、程序化和声、渐变插值 | `references/color-systems.md` |
| **噪声词汇** | Perlin 噪声、simplex、分形（多倍频）、域扭曲、curl 噪声 | `references/visual-effects.md` § Noise |
| **粒子系统** | 基于物理、群集、轨迹绘制、吸引子驱动、流场跟随 | `references/visual-effects.md` § Particles |
| **形状语言** | 几何基元、自定义顶点、贝塞尔曲线、SVG 路径 | `references/shapes-and-geometry.md` |
| **运动风格** | 缓动、弹簧物理、噪声驱动、物理模拟、线性插值、步进 | `references/animation.md` |
| **排版** | 系统字体、加载的 OTF、`textToPoints()` 粒子文字、动态排版 | `references/typography.md` |
| **着色器效果** | GLSL 片段/顶点着色器、滤镜着色器、后处理、反馈循环 | `references/webgl-and-3d.md` § Shaders |
| **构图** | 网格、放射状、黄金比例、三分法、有机散布、平铺 | `references/core-api.md` § Composition |
| **交互模型** | 鼠标跟随、点击生成、拖拽、键盘状态、滚动驱动、麦克风输入 | `references/interaction.md` |
| **混合模式** | `BLEND`、`ADD`、`MULTIPLY`、`SCREEN`、`DIFFERENCE`、`EXCLUSION`、`OVERLAY` | `references/color-systems.md` § Blend Modes |
| **分层** | `createGraphics()` 离屏缓冲区、alpha 合成、遮罩 | `references/core-api.md` § Offscreen Buffers |
| **纹理** | Perlin 表面、点画、排线、半调、像素排序 | `references/visual-effects.md` § Texture Generation |

### 每个项目的变化规则

绝不使用默认配置。每个项目必须：
- **自定义色彩调色板** — 绝不使用原始的 `fill(255, 0, 0)`。始终使用包含 3-7 种颜色的精心设计调色板
- **自定义描边粗细词汇** — 细线强调（0.5）、中等结构（1-2）、粗体重点（3-5）
- **背景处理** — 绝不使用纯 `background(0)` 或 `background(255)`。始终使用纹理、渐变或分层背景
- **运动多样性** — 不同元素使用不同速度。主要元素 1x，次要元素 0.3x，环境元素 0.1x
- **至少一个创造性元素** — 自定义粒子行为、新颖的噪声应用、独特的交互响应

### 项目专属创造

每个项目至少创造以下之一：
- 符合氛围的自定义色彩调色板（非预设）
- 新颖的噪声场组合（例如 curl 噪声 + 域扭曲 + 反馈）
- 独特的粒子行为（自定义力、自定义轨迹、自定义生成方式）
- 用户未要求但能提升作品的交互机制
- 创造视觉层次的构图技巧

### 参数设计哲学

参数应从算法中涌现，而非来自通用菜单。问自己："*这个*系统的哪些属性应该可调？"

**好的参数**揭示算法的特性：
- **数量** — 粒子、分支、单元格的数量（控制密度）
- **尺度** — 噪声频率、元素大小、间距（控制纹理）
- **速率** — 速度、增长率、衰减（控制能量）
- **阈值** — 行为何时改变？（控制戏剧性）
- **比率** — 比例、力之间的平衡（控制和谐）

**坏的参数**是与算法无关的通用控件：
- "color1"、"color2"、"size"——脱离上下文毫无意义
- 不相关效果的开关
- 只改变外观而不改变行为的参数

每个参数都应改变算法*思考*的方式，而不仅仅是*看起来*的样子。改变噪声倍频的"turbulence"参数是好的。只改变 `ellipse()` 半径的"particle size"滑块是浅薄的。

## 工作流程

### 第一步：创意愿景

在任何代码之前，先阐明：

- **氛围 / 情绪**：观者应该感受到什么？沉思？充满活力？不安？愉悦？
- **视觉故事**：随时间（或交互）发生什么？构建？衰减？变换？振荡？
- **色彩世界**：暖色/冷色？单色？互补色？主色调是什么？强调色是什么？
- **形状语言**：有机曲线？锐利几何？点？线？混合？
- **运动词汇**：缓慢漂移？爆炸性迸发？呼吸脉冲？机械精准？
- **这件作品的独特之处**：使这个草图独一无二的一件事是什么？

将用户的 prompt 映射到美学选择。"放松的生成背景"与"故障数据可视化"在各方面都要求截然不同的处理。

### 第二步：技术设计

- **模式** — 上表中 7 种模式中的哪一种
- **Canvas 尺寸** — 横向 1920x1080、纵向 1080x1920、正方形 1080x1080，或响应式 `windowWidth/windowHeight`
- **渲染器** — `P2D`（默认）或 `WEBGL`（用于 3D、着色器、高级混合模式）
- **帧率** — 60fps（交互式）、30fps（环境动画），或 `noLoop()`（静态生成）
- **导出目标** — 浏览器显示、PNG 静图、GIF 循环、MP4 视频、SVG 矢量
- **交互模型** — 被动（无输入）、鼠标驱动、键盘驱动、音频响应式、滚动驱动
- **查看器 UI** — 对于交互式生成艺术（种子探索、参数调整），从 `templates/viewer.html` 开始，它提供种子导航、参数滑块和下载功能。对于简单草图或视频导出，使用裸 HTML

### 第三步：编写草图代码

对于**交互式生成艺术**（种子探索、参数调整）：从 `templates/viewer.html` 开始。先阅读模板，保留固定部分（种子导航、操作按钮），替换算法和参数控件。这为用户提供种子上一个/下一个/随机/跳转、带实时更新的参数滑块，以及 PNG 下载——全部已连接好。

对于**动画、视频导出或简单草图**：使用裸 HTML：

单一 HTML 文件。结构：

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Name</title>
  <script>p5.disableFriendlyErrors = true;</script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
  <!-- <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/addons/p5.sound.min.js"></script> -->
  <!-- <script src="https://unpkg.com/p5.js-svg@1.6.0"></script> -->  <!-- SVG export -->
  <!-- <script src="https://cdn.jsdelivr.net/npm/ccapture.js-npmfixed/build/CCapture.all.min.js"></script> -->  <!-- video capture -->
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
// === Configuration ===
const CONFIG = {
  seed: 42,
  // ... project-specific params
};

// === Color Palette ===
const PALETTE = {
  bg: '#0a0a0f',
  primary: '#e8d5b7',
  // ...
};

// === Global State ===
let particles = [];

// === Preload (fonts, images, data) ===
function preload() {
  // font = loadFont('...');
}

// === Setup ===
function setup() {
  createCanvas(1920, 1080);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  colorMode(HSB, 360, 100, 100, 100);
  // Initialize state...
}

// === Draw Loop ===
function draw() {
  // Render frame...
}

// === Helper Functions ===
// ...

// === Classes ===
class Particle {
  // ...
}

// === Event Handlers ===
function mousePressed() { /* ... */ }
function keyPressed() { /* ... */ }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
</script>
</body>
</html>
```

关键实现模式：
- **种子随机性**：始终使用 `randomSeed()` + `noiseSeed()` 以确保可复现性
- **色彩模式**：使用 `colorMode(HSB, 360, 100, 100, 100)` 以获得直观的色彩控制
- **状态分离**：CONFIG 用于参数，PALETTE 用于颜色，全局变量用于可变状态
- **基于类的实体**：粒子、代理、形状作为具有 `update()` + `display()` 方法的类
- **离屏缓冲区**：`createGraphics()` 用于分层合成、轨迹、遮罩

### 第四步：预览与迭代

- 直接在浏览器中打开 HTML 文件——基本草图无需服务器
- 对于从本地文件加载 `loadImage()`/`loadFont()`：使用 `scripts/serve.sh` 或 `python3 -m http.server`
- 使用 Chrome DevTools 性能面板验证 60fps
- 在目标导出分辨率下测试，而不仅仅是窗口大小
- 调整参数直到视觉效果符合第一步的概念

### 第五步：导出

| 格式 | 方法 | 命令 |
|--------|--------|---------|
| **PNG** | 在 `keyPressed()` 中使用 `saveCanvas('output', 'png')` | 按 's' 保存 |
| **高分辨率 PNG** | Puppeteer 无头捕获 | `node scripts/export-frames.js sketch.html --width 3840 --height 2160 --frames 1` |
| **GIF** | `saveGif('output', 5)` — 捕获 N 秒 | 按 'g' 保存 |
| **帧序列** | `saveFrames('frame', 'png', 10, 30)` — 10 秒 30fps | 然后 `ffmpeg -i frame-%04d.png -c:v libx264 output.mp4` |
| **MP4** | Puppeteer 帧捕获 + ffmpeg | `bash scripts/render.sh sketch.html output.mp4 --duration 30 --fps 30` |
| **SVG** | 使用 p5.js-svg 的 `createCanvas(w, h, SVG)` | `save('output.svg')` |

### 第六步：质量验证

- **是否符合愿景？** 将输出与创意概念对比。如果看起来很普通，回到第一步
- **分辨率检查**：在目标显示尺寸下是否清晰？是否有锯齿伪影？
- **性能检查**：在浏览器中是否保持 60fps？（动画最低 30fps）
- **色彩检查**：颜色是否协调？在亮色和暗色显示器上都测试
- **边界情况**：canvas 边缘会发生什么？调整大小时？运行 10 分钟后？

## 关键实现注意事项

### 性能——首先禁用 FES

友好错误系统（FES）会增加高达 10 倍的开销。在每个生产草图中禁用它：

```javascript
p5.disableFriendlyErrors = true;  // BEFORE setup()

function setup() {
  pixelDensity(1);  // prevent 2x-4x overdraw on retina
  createCanvas(1920, 1080);
}
```

在热循环（粒子、像素操作）中，使用 `Math.*` 而非 p5 包装函数——速度明显更快：

```javascript
// In draw() or update() hot paths:
let a = Math.sin(t);          // not sin(t)
let r = Math.sqrt(dx*dx+dy*dy); // not dist() — or better: skip sqrt, compare magSq
let v = Math.random();        // not random() — when seed not needed
let m = Math.min(a, b);       // not min(a, b)
```

绝不在 `draw()` 内使用 `console.log()`。绝不在 `draw()` 中操作 DOM。参见 `references/troubleshooting.md` § Performance。

### 种子随机性——始终使用

每个生成草图必须可复现。相同种子，相同输出。

```javascript
function setup() {
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  // All random() and noise() calls now deterministic
}
```

绝不对生成内容使用 `Math.random()`——仅用于性能关键的非视觉代码。视觉元素始终使用 `random()`。如果需要随机种子：`CONFIG.seed = floor(random(99999))`。

### 生成艺术平台支持（fxhash / Art Blocks）

对于生成艺术平台，用平台的确定性随机替换 p5 的 PRNG：

```javascript
// fxhash convention
const SEED = $fx.hash;              // unique per mint
const rng = $fx.rand;               // deterministic PRNG
$fx.features({ palette: 'warm', complexity: 'high' });

// In setup():
randomSeed(SEED);   // for p5's noise()
noiseSeed(SEED);

// Replace random() with rng() for platform determinism
let x = rng() * width;  // instead of random(width)
```

参见 `references/export-pipeline.md` § Platform Export。

### 色彩模式——使用 HSB

HSB（色相、饱和度、亮度）在生成艺术中比 RGB 更易于使用：

```javascript
colorMode(HSB, 360, 100, 100, 100);
// Now: fill(hue, sat, bri, alpha)
// Rotate hue: fill((baseHue + offset) % 360, 80, 90)
// Desaturate: fill(hue, sat * 0.3, bri)
// Darken: fill(hue, sat, bri * 0.5)
```

绝不硬编码原始 RGB 值。定义调色板对象，以程序化方式派生变体。参见 `references/color-systems.md`。

### 噪声——多倍频，而非原始噪声

原始 `noise(x, y)` 看起来像平滑的斑点。叠加倍频以获得自然纹理：

```javascript
function fbm(x, y, octaves = 4) {
  let val = 0, amp = 1, freq = 1, sum = 0;
  for (let i = 0; i < octaves; i++) {
    val += noise(x * freq, y * freq) * amp;
    sum += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return val / sum;
}
```

对于流动的有机形态，使用**域扭曲**：将噪声输出作为噪声输入坐标反馈回去。参见 `references/visual-effects.md`。

### createGraphics() 分层——不可省略

单通道平面渲染看起来很平。使用离屏缓冲区进行合成：

```javascript
let bgLayer, fgLayer, trailLayer;
function setup() {
  createCanvas(1920, 1080);
  bgLayer = createGraphics(width, height);
  fgLayer = createGraphics(width, height);
  trailLayer = createGraphics(width, height);
}
function draw() {
  renderBackground(bgLayer);
  renderTrails(trailLayer);   // persistent, fading
  renderForeground(fgLayer);  // cleared each frame
  image(bgLayer, 0, 0);
  image(trailLayer, 0, 0);
  image(fgLayer, 0, 0);
}
```

### 性能——尽可能向量化

p5.js 绘制调用开销较大。对于数千个粒子：

```javascript
// SLOW: individual shapes
for (let p of particles) {
  ellipse(p.x, p.y, p.size);
}

// FAST: single shape with beginShape()
beginShape(POINTS);
for (let p of particles) {
  vertex(p.x, p.y);
}
endShape();

// FASTEST: pixel buffer for massive counts
loadPixels();
for (let p of particles) {
  let idx = 4 * (floor(p.y) * width + floor(p.x));
  pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b; pixels[idx+3] = 255;
}
updatePixels();
```

参见 `references/troubleshooting.md` § Performance。

### 多草图使用实例模式

全局模式会污染 `window`。生产环境中使用实例模式：

```javascript
const sketch = (p) => {
  p.setup = function() {
    p.createCanvas(800, 800);
  };
  p.draw = function() {
    p.background(0);
    p.ellipse(p.mouseX, p.mouseY, 50);
  };
};
new p5(sketch, 'canvas-container');
```

在同一页面嵌入多个草图或与框架集成时必须使用。

### WebGL 模式注意事项

- `createCanvas(w, h, WEBGL)` — 原点在中心，而非左上角
- Y 轴反转（WEBGL 中正 Y 向上，P2D 中向下）
- 使用 `translate(-width/2, -height/2)` 获得类似 P2D 的坐标
- 每次变换前后都要使用 `push()`/`pop()` — 矩阵栈会静默溢出
- `texture()` 在 `rect()`/`plane()` 之前调用——而非之后
- 自定义着色器：`createShader(vert, frag)` — 在多个浏览器上测试

### 导出——按键绑定约定

每个草图的 `keyPressed()` 中都应包含以下内容：

```javascript
function keyPressed() {
  if (key === 's' || key === 'S') saveCanvas('output', 'png');
  if (key === 'g' || key === 'G') saveGif('output', 5);
  if (key === 'r' || key === 'R') { randomSeed(millis()); noiseSeed(millis()); }
  if (key === ' ') CONFIG.paused = !CONFIG.paused;
}
```

### 无头视频导出——使用 noLoop()

对于通过 Puppeteer 进行无头渲染，草图**必须**在 setup 中使用 `noLoop()`。否则，p5 的绘制循环会自由运行，而截图速度较慢——草图会超前运行，导致帧跳过或重复。

```javascript
function setup() {
  createCanvas(1920, 1080);
  pixelDensity(1);
  noLoop();                    // capture script controls frame advance
  window._p5Ready = true;      // signal readiness to capture script
}
```

内置的 `scripts/export-frames.js` 检测 `_p5Ready` 并在每次捕获时调用一次 `redraw()`，实现精确的 1:1 帧对应。参见 `references/export-pipeline.md` § Deterministic Capture。

对于多场景视频，使用每片段架构：每个场景一个 HTML，独立渲染，用 `ffmpeg -f concat` 拼接。参见 `references/export-pipeline.md` § Per-Clip Architecture。

### Agent 工作流程

构建 p5.js 草图时：

1. **编写 HTML 文件** — 单一自包含文件，所有代码内联
2. **在浏览器中打开** — macOS 用 `open sketch.html`，Linux 用 `xdg-open sketch.html`
3. **本地资源**（字体、图像）需要服务器：在项目目录中运行 `python3 -m http.server 8080`，然后打开 `http://localhost:8080/sketch.html`
4. **导出 PNG/GIF** — 如上所示添加 `keyPressed()` 快捷键，告知用户按哪个键
5. **无头导出** — `node scripts/export-frames.js sketch.html --frames 300` 用于自动化帧捕获（草图必须使用 `noLoop()` + `_p5Ready`）
6. **MP4 渲染** — `bash scripts/render.sh sketch.html output.mp4 --duration 30`
7. **迭代优化** — 编辑 HTML 文件，用户刷新浏览器查看变化
8. **按需加载参考资料** — 在实现过程中使用 `skill_view(name="p5js", file_path="references/...")` 加载特定参考文件

## 性能目标

| 指标 | 目标 |
|--------|--------|
| 帧率（交互式） | 持续 60fps |
| 帧率（动画导出） | 最低 30fps |
| 粒子数量（P2D 形状） | 60fps 下 5,000-10,000 |
| 粒子数量（像素缓冲区） | 60fps 下 50,000-100,000 |
| Canvas 分辨率 | 最高 3840x2160（导出），1920x1080（交互式） |
| 文件大小（HTML） | &lt; 100KB（不含 CDN 库） |
| 加载时间 | &lt; 2 秒到首帧 |

## 参考资料

| 文件 | 内容 |
|------|----------|
| `references/core-api.md` | Canvas 设置、坐标系、绘制循环、`push()`/`pop()`、离屏缓冲区、构图模式、`pixelDensity()`、响应式设计 |
| `references/shapes-and-geometry.md` | 2D 基元、`beginShape()`/`endShape()`、贝塞尔/Catmull-Rom 曲线、`vertex()` 系统、自定义形状、`p5.Vector`、有符号距离场、SVG 路径转换 |
| `references/visual-effects.md` | 噪声（Perlin、分形、域扭曲、curl）、流场、粒子系统（物理、群集、轨迹）、像素操作、纹理生成（点画、排线、半调）、反馈循环、反应扩散 |
| `references/animation.md` | 基于帧的动画、缓动函数、`lerp()`/`map()`、弹簧物理、状态机、时间轴排序、基于 `millis()` 的计时、过渡模式 |
| `references/typography.md` | `text()`、`loadFont()`、`textToPoints()`、动态排版、文字遮罩、字体度量、响应式文字大小 |
| `references/color-systems.md` | `colorMode()`、HSB/HSL/RGB、`lerpColor()`、`paletteLerp()`、程序化调色板、色彩和声、`blendMode()`、渐变渲染、精选调色板库 |
| `references/webgl-and-3d.md` | WEBGL 渲染器、3D 基元、摄像机、光照、材质、自定义几何体、GLSL 着色器（`createShader()`、`createFilterShader()`）、帧缓冲区、后处理 |
| `references/interaction.md` | 鼠标事件、键盘状态、触控输入、DOM 元素、`createSlider()`/`createButton()`、音频输入（p5.sound FFT/振幅）、滚动驱动动画、响应式事件 |
| `references/export-pipeline.md` | `saveCanvas()`、`saveGif()`、`saveFrames()`、确定性无头捕获、ffmpeg 帧转视频、CCapture.js、SVG 导出、每片段架构、平台导出（fxhash）、视频注意事项 |
| `references/troubleshooting.md` | 性能分析、每像素预算、常见错误、浏览器兼容性、WebGL 调试、字体加载问题、像素密度陷阱、内存泄漏、CORS |
| `templates/viewer.html` | 交互式查看器模板：种子导航（上一个/下一个/随机/跳转）、参数滑块、下载 PNG、响应式 canvas。可探索生成艺术从此开始 |

---

## 创意发散（仅在用户请求实验性/创意性/独特输出时使用）

如果用户要求创意性、实验性、令人惊喜或非常规的输出，在生成代码**之前**选择最合适的策略并推演其步骤。

- **概念混合** — 当用户命名两件要组合的事物或想要混合美学时
- **SCAMPER** — 当用户想要对已知生成艺术模式进行变体时
- **距离联想** — 当用户给出单一概念并想要探索时（"做一些关于时间的东西"）

### 概念混合
1. 命名两个不同的视觉系统（例如粒子物理 + 手写）
2. 映射对应关系（粒子 = 墨滴，力 = 笔压，场 = 字形）
3. 选择性混合——保留能产生有趣涌现视觉效果的映射
4. 将混合编码为统一系统，而非两个并排的系统

### SCAMPER 变换
取一个已知的生成模式（流场、粒子系统、L 系统、元胞自动机）并系统性地变换它：
- **替换（Substitute）**：用文字字符替换圆形，用渐变替换线条
- **组合（Combine）**：合并两种模式（流场 + Voronoi）
- **适配（Adapt）**：将 2D 模式应用于 3D 投影
- **修改（Modify）**：夸大比例，扭曲坐标空间
- **用途（Purpose）**：用物理模拟做排版，用排序算法做色彩
- **消除（Eliminate）**：去掉网格，去掉颜色，去掉对称性
- **反转（Reverse）**：反向运行模拟，反转参数空间

### 距离联想
1. 锚定用户的概念（例如"孤独"）
2. 在三个距离上生成联想：
   - 近（显而易见）：空房间、单独的人物、寂静
   - 中（有趣）：一条鱼在鱼群中逆向游动、没有通知的手机、地铁车厢之间的间隙
   - 远（抽象）：质数、渐近曲线、凌晨三点的颜色
3. 发展中距离的联想——它们足够具体可以可视化，又足够出人意料而有趣