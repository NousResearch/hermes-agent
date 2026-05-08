---
title: "P5Js — p5"
sidebar_label: "P5Js"
description: "p5"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# P5Js

p5.js sketches: gen art, shaders, interactive, 3D.

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/p5js` |
| 版本 | `1.0.0` |
| 标签 | `creative-coding`, `generative-art`, `p5js`, `canvas`, `interactive`, `visualization`, `webgl`, `shaders`, `animation` |
| 相关技能 | [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# p5.js 制作流水线

## 使用场景

当用户请求以下内容时使用：p5.js sketches、创意编码、生成艺术、交互式可视化、画布动画、基于浏览器的视觉艺术、数据可视化、着色器效果，或任何p5.js项目。

## 内容简介

使用p5.js的生成艺术和交互式视觉艺术制作流水线。创建基于浏览器的sketches、生成艺术、数据可视化、交互式体验、3D场景、音频响应可视化、动态图形——导出为HTML、PNG、GIF、MP4或SVG。涵盖：2D/3D渲染、噪声和粒子系统、流场、着色器（GLSL）、像素操作、动态排版、WebGL场景、音频分析、鼠标/键盘交互，以及无头高分辨率导出。

## 创意标准

这是在浏览器中渲染的视觉艺术。画布是媒介；算法是画笔。

**在写任何代码之前**，阐述创意概念。这件作品传达什么？是什么让用户停止滚动？是什么将它从代码教程示例中区分出来？用户的提示是起点——用创意野心来诠释它。

**首次渲染的卓越性是不可妥协的。** 输出必须在首次加载时就具有视觉冲击力。如果它看起来像p5.js教程练习、默认配置，或"AI生成的创意编码"，那就是错误的。在交付之前重新思考。

**超越参考词汇。** 参考资料中的噪声函数、粒子系统、调色板和着色器效果是起点词汇。对于每个项目，要组合、分层和发明。目录是一组颜料——你来创作这幅画。

**主动发挥创意。** 如果用户要求"一个粒子系统"，就交付一个具有涌现集群行为、拖尾鬼魅回声、调色板偏移深度雾，以及会呼吸的背景噪声场的粒子系统。至少包含一个用户没有要求但他们会欣赏的视觉细节。

**密集、分层、精心考虑。** 每一帧都应该值得观看。永远不要纯白背景。始终要有构图层级。始终要故意的色彩。始终要有只在仔细检查时出现的微细节。

**统一的审美优先于功能数量。** 所有元素必须服务于统一的视觉语言——共享的色温、一致的描边权重词汇、和谐的运动速度。一个有十个不相关效果的sketch比三个属于一起的效果更差。

## 模式

| 模式 | 输入 | 输出 | 参考 |
|------|-------|--------|-----------|
| **生成艺术** | 种子/参数 | 程序化视觉构图（静态或动画） | `references/visual-effects.md` |
| **数据可视化** | 数据集/API | 交互式图表、图形、自定义数据显示 | `references/interaction.md` |
| **交互式体验** | 无（用户驱动） | 鼠标/键盘/触摸驱动的sketch | `references/interaction.md` |
| **动画/动态图形** | 时间线/故事板 | 定时序列、动态排版、过渡 | `references/animation.md` |
| **3D场景** | 概念描述 | WebGL几何、光照、相机、材质 | `references/webgl-and-3d.md` |
| **图像处理** | 图像文件 | 像素操作、滤镜、马赛克、点彩画 | `references/visual-effects.md` § 像素操作 |
| **音频响应** | 音频文件/麦克风 | 声音驱动的生成式视觉 | `references/interaction.md` § 音频输入 |

## 技术栈

每个项目一个独立的HTML文件。无需构建步骤。

| 层级 | 工具 | 用途 |
|-------|------|---------|
| 核心 | p5.js 1.11.3 (CDN) | 画布渲染、数学、变换、事件处理 |
| 3D | p5.js WebGL 模式 | 3D几何、相机、光照、GLSL着色器 |
| 音频 | p5.sound.js (CDN) | FFT分析、振幅、麦克风输入、振荡器 |
| 导出 | 内置`saveCanvas()` / `saveGif()` / `saveFrames()` | PNG、GIF、帧序列输出 |
| 捕获 | CCapture.js（可选） | 确定性帧率视频捕获（WebM、GIF） |
| 无头 | Puppeteer + Node.js（可选） | 自动化高分辨率渲染、通过ffmpeg生成MP4 |
| SVG | p5.js-svg 1.6.0（可选） | 用于打印的矢量输出——需要p5.js 1.x |
| 自然媒材 | p5.brush（可选） | 水彩、炭笔、钢笔——需要p5.js 2.x + WebGL |
| 纹理 | p5.rain（可选） | 胶片颗粒、纹理叠加 |
| 字体 | Google Fonts / `loadFont()` | 通过OTF/TTF/WOFF2的自定义排版 |

### 版本说明

**p5.js 1.x**（1.11.3）是默认版本——稳定、文档完善、最广泛的库兼容性。除非项目需要2.x功能，否则请使用此版本。

**p5.js 2.x**（2.2+）增加了：`async setup()`替代`preload()`、OKLCH/OKLAB颜色模式、`splineVertex()`、着色器`.modify()` API、可变字体、`textToContours()`、指针事件。p5.brush需要。参见`references/core-api.md` § p5.js 2.0。

## 流水线

每个项目都遵循相同的6阶段路径：

```
CONCEPT → DESIGN → CODE → PREVIEW → EXPORT → VERIFY
```

1. **CONCEPT** — 阐述创意愿景：氛围、色彩世界、运动词汇、是什么让它独特
2. **DESIGN** — 选择模式、画布尺寸、交互模型、颜色系统、导出格式。将概念映射到技术决策
3. **CODE** — 编写单个HTML文件，内含内联p5.js。结构：全局变量 → `preload()` → `setup()` → `draw()` → 辅助函数 → 类 → 事件处理器
4. **PREVIEW** — 在浏览器中打开，验证视觉质量。在目标分辨率下测试。检查性能
5. **EXPORT** — 捕获输出：`saveCanvas()`用于PNG、`saveGif()`用于GIF、`saveFrames()` + ffmpeg用于MP4、Puppeteer用于无头批处理
6. **VERIFY** — 输出与概念匹配吗？在预期显示尺寸下看起来有视觉冲击力吗？你会装裱它吗？

## 创意方向

### 审美维度

| 维度 | 选项 | 参考 |
|-----------|---------|-----------|
| **色彩系统** | HSB/HSL、RGB、命名调色板、程序化和谐、渐变插值 | `references/color-systems.md` |
| **噪声词汇** | Perlin噪声、simplex、分形（八度）、域扭曲、旋度噪声 | `references/visual-effects.md` § 噪声 |
| **粒子系统** | 基于物理、集群、拖尾绘制、吸引子驱动、流场跟随 | `references/visual-effects.md` § 粒子 |
| **形状语言** | 几何图元、自定义顶点、贝塞尔曲线、SVG路径 | `references/shapes-and-geometry.md` |
| **运动风格** | 缓动、基于弹簧、噪声驱动、物理模拟、线性插值、步进 | `references/animation.md` |
| **排版** | 系统字体、加载的OTF、`textToPoints()`粒子文本、动态 | `references/typography.md` |
| **着色器效果** | GLSL片段/顶点、滤镜着色器、后处理、反馈循环 | `references/webgl-and-3d.md` § 着色器 |
| **构图** | 网格、放射状、黄金比例、三分法则、有机散点、平铺 | `references/core-api.md` § 构图 |
| **交互模型** | 鼠标跟随、点击生成、拖动、键盘状态、滚动驱动、麦克风输入 | `references/interaction.md` |
| **混合模式** | `BLEND`、`ADD`、`MULTIPLY`、`SCREEN`、`DIFFERENCE`、`EXCLUSION`、`OVERLAY` | `references/color-systems.md` § 混合模式 |
| **分层** | `createGraphics()`离屏缓冲区、Alpha合成、遮罩 | `references/core-api.md` § 离屏缓冲区 |
| **纹理** | Perlin表面、点画、孵化、半调、像素排序 | `references/visual-effects.md` § 纹理生成 |

### 每个项目的变体规则

永远不要使用默认配置。对于每个项目：

- **自定义调色板** — 永远不要`fill(255, 0, 0)`。总是设计一个有3-7种颜色的调色板
- **自定义描边权重词汇** — 细强调（0.5）、中等结构（1-2）、粗强调（3-5）
- **背景处理** — 永远不要纯`background(0)`或`background(255)`。总是有纹理、渐变或分层的
- **运动变化** — 不同元素的不同速度。主要1x，次要0.3x，环境0.1x
- **至少一个发明的元素** — 自定义粒子行为、新颖的噪声应用、独特的交互响应

### 项目特定发明

对于每个项目，至少发明以下之一：

- 匹配氛围的自定义调色板（不是预设）
- 新颖的噪声场组合（例如，旋度噪声+域扭曲+反馈）
- 独特的粒子行为（自定义力、自定义拖尾、自定义生成）
- 用户没有要求但会提升作品的交互机制
- 创建视觉层级的构图技术

### 参数设计哲学

参数应该从算法中涌现，而不是来自通用菜单。问："*这个*系统的哪些属性应该是可调优的？"

**好的参数**展现算法的特性：
- **数量** — 多少粒子、分支、单元格（控制密度）
- **缩放** — 噪声频率、元素尺寸、间距（控制纹理）
- **速率** — 速度、增长率、衰减（控制能量）
- **阈值** — 行为何时改变？（控制戏剧性）
- **比率** — 比例、力之间的平衡（控制和谐）

**坏参数**是与算法无关通用控件：
- "color1"、"color2"、"size" — 没有上下文就没有意义
- 不相关效果的切换开关
- 只改变外观、不改变行为的参数

每个参数都应该改变算法*思考*的方式，而不仅仅是它*看*起来的方式。"湍流"参数改变噪声八度是好的。"粒子大小"滑块只改变`ellipse()`半径是肤浅的。

## 工作流程

### 第1步：创意愿景

在编写任何代码之前，请阐述：

- **氛围/气氛**：用户应该感受到什么？沉思的？充满活力的？不安的？俏皮的？
- **视觉故事**：随着时间（或交互）发生了什么？构建？衰减？转换？振荡？
- **色彩世界**：暖/冷？单色？互补？主色调是什么？强调色？
- **形状语言**：有机曲线？锐利几何？点？线？混合？
- **运动词汇**：慢漂移？爆炸性爆发？呼吸脉冲？机械精度？
- **是什么让这个与众不同**：使这个sketch独特的一件事是什么？

将用户的提示映射到审美选择。"放松的生成式背景"与"故障数据可视化"需要完全不同的所有东西。

### 第2步：技术设计

- **模式** — 上表中7种模式中的哪一种
- **画布尺寸** — 横屏1920x1080、竖屏1080x1920、正方形1080x1080、或响应式`windowWidth/windowHeight`
- **渲染器** — `P2D`（默认）或`WEBGL`（用于3D、着色器、高级混合模式）
- **帧率** — 60fps（交互式）、30fps（环境动画）、或`noLoop()`（静态生成式）
- **导出目标** — 浏览器显示、PNG静态、GIF循环、MP4视频、SVG矢量
- **交互模型** — 被动（无输入）、鼠标驱动、键盘驱动、音频响应、滚动驱动
- **查看器UI** — 对于交互式生成艺术，从`templates/viewer.html`开始，它提供种子导航、参数滑块和下载。对于简单的sketches或视频导出，使用纯HTML

### 第3步：编写Sketch代码

对于**交互式生成艺术**（种子探索、参数调优）：从`templates/viewer.html`开始。先阅读模板，保留固定部分（种子导航、操作），替换算法和参数控件。这给用户提供了种子上一步/下一步/随机/跳转、参数滑块实时更新，以及PNG下载——所有这些都连接好了。

对于**动画、视频导出或简单sketches**：使用纯HTML：

单个HTML文件。结构：

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>项目名称</title>
  <script>p5.disableFriendlyErrors = true;</script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
  <!-- <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/addons/p5.sound.min.js"></script> -->
  <!-- <script src="https://unpkg.com/p5.js-svg@1.6.0"></script> -->  <!-- SVG导出 -->
  <!-- <script src="https://cdn.jsdelivr.net/npm/ccapture.js-npmfixed/build/CCapture.all.min.js"></script> -->  <!-- 视频捕获 -->
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
// === 配置 ===
const CONFIG = {
  seed: 42,
  // ... 项目特定参数
};

// === 调色板 ===
const PALETTE = {
  bg: '#0a0a0f',
  primary: '#e8d5b7',
  // ...
};

// === 全局状态 ===
let particles = [];

// === 预加载（字体、图像、数据） ===
function preload() {
  // font = loadFont('...');
}

// === 设置 ===
function setup() {
  createCanvas(1920, 1080);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  colorMode(HSB, 360, 100, 100, 100);
  // 初始化状态...
}

// === 绘制循环 ===
function draw() {
  // 渲染帧...
}

// === 辅助函数 ===
// ...

// === 类 ===
class Particle {
  // ...
}

// === 事件处理器 ===
function mousePressed() { /* ... */ }
function keyPressed() { /* ... */ }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
</script>
</body>
</html>
```

关键实现模式：
- ** seeded随机性**：总是`randomSeed()` + `noiseSeed()`用于可重现性
- **颜色模式**：使用`colorMode(HSB, 360, 100, 100, 100)`用于直观颜色控制
- **状态分离**：CONFIG用于参数、PALETTE用于颜色、全局变量用于可变状态
- **基于类的实体**：粒子、代理、形状作为具有`update()` + `display()`方法的类
- **离屏缓冲区**：`createGraphics()`用于分层构图、拖尾、遮罩

### 第4步：预览和迭代

- 直接在浏览器中打开HTML文件——基本sketches无需服务器
- 对于`loadImage()`/`loadFont()`来自本地文件：使用`scripts/serve.sh`或`python3 -m http.server`
- Chrome DevTools Performance标签验证60fps
- 在目标导出分辨率下测试，不仅仅是窗口尺寸
- 调整参数直到视觉与第1步的概念匹配

### 第5步：导出

| 格式 | 方法 | 命令 |
|--------|--------|---------|
| **PNG** | `saveCanvas('output', 'png')`在`keyPressed()`中 | 按's'保存 |
| **高分辨率PNG** | Puppeteer无头捕获 | `node scripts/export-frames.js sketch.html --width 3840 --height 2160 --frames 1` |
| **GIF** | `saveGif('output', 5)` — 捕获N秒 | 按'g'保存 |
| **帧序列** | `saveFrames('frame', 'png', 10, 30)` — 10秒@30fps | 然后`ffmpeg -i frame-%04d.png -c:v libx264 output.mp4` |
| **MP4** | Puppeteer帧捕获 + ffmpeg | `bash scripts/render.sh sketch.html output.mp4 --duration 30 --fps 30` |
| **SVG** | 使用p5.js-svg的`createCanvas(w, h, SVG)` | `save('output.svg')` |

### 第6步：质量验证

- **它与愿景匹配吗？** 将输出与创意概念进行比较。如果看起来普通，返回第1步
- **分辨率检查**：在目标显示尺寸下清晰吗？没有锯齿伪影？
- **性能检查**：在浏览器中保持60fps吗？（动画至少30fps）
- **颜色检查**：颜色在一起协调吗？在明亮和黑暗监视器上都测试
- **边缘情况**：画布边缘会发生什么？调整大小时呢？运行10分钟后呢？

## 关键实现注意事项

### 性能 — 首先禁用FES

友好错误系统（FES）最多增加10倍的额外开销。在每个生产sketch中禁用它：

```javascript
p5.disableFriendlyErrors = true;  // 在setup()之前

function setup() {
  pixelDensity(1);  // 防止在retina显示屏上2x-4倍过度绘制
  createCanvas(1920, 1080);
}
```

在热循环（粒子、像素操作）中，使用`Math.*`而不是p5包装器——可测量地更快：

```javascript
// 在draw()或update()热路径中：
let a = Math.sin(t);          // 不是sin(t)
let r = Math.sqrt(dx*dx+dy*dy); // 不是dist() —— 或更好：跳过sqrt，比较magSq
let v = Math.random();        // 不是random() —— 当不需要种子时
let m = Math.min(a, b);       // 不是min(a, b)
```

永远不要在`draw()`中`console.log()`。永远不要在`draw()`中操作DOM。参见`references/troubleshooting.md` § 性能。

###  seeded随机性——总是

每个生成式sketch必须是可重现的。相同的种子，相同的输出。

```javascript
function setup() {
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  // 所有random()和noise()调用现在是确定性的
}
```

永远不要对生成式内容使用`Math.random()` —— 仅用于性能关键的非视觉代码。总是对视觉元素使用`random()`。如果需要随机种子：`CONFIG.seed = floor(random(99999))`。

### 生成艺术平台支持（fxhash / Art Blocks）

对于生成艺术平台，用平台的确定性随机数替换p5的PRNG：

```javascript
// fxhash惯例
const SEED = $fx.hash;              // 每次铸造唯一
const rng = $fx.rand;               // 确定性PRNG
$fx.features({ palette: 'warm', complexity: 'high' });

// 在setup()中：
randomSeed(SEED);   // 用于p5的noise()
noiseSeed(SEED);

// 用rng()替换random()以获得平台确定性
let x = rng() * width;  // 而不是random(width)
```

参见`references/export-pipeline.md` § 平台导出。

### 颜色模式 — 使用HSB

HSB（色相、饱和度、亮度）比RGB用于生成艺术明显更容易使用：

```javascript
colorMode(HSB, 360, 100, 100, 100);
// 现在：fill(hue, sat, bri, alpha)
// 旋转色相：fill((baseHue + offset) % 360, 80, 90)
// 去饱和：fill(hue, sat * 0.3, bri)
// 变暗：fill(hue, sat, bri * 0.5)
```

永远不要硬编码原始RGB值。定义调色板对象，程序化地派生变体。参见`references/color-systems.md`。

### 噪声 — 多八度，不是原始的

原始`noise(x, y)`看起来像光滑斑点。为自然纹理分层八度：

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

对于流动有机形式，使用**域扭曲**：将噪声输出反馈为噪声输入坐标。参见`references/visual-effects.md`。

### createGraphics()用于层级 —— 不是可选的

平面单次传递渲染看起来很平。使用离屏缓冲区进行构图：

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
  renderTrails(trailLayer);   // 持久的、淡出的
  renderForeground(fgLayer); // 每帧清除
  image(bgLayer, 0, 0);
  image(trailLayer, 0, 0);
  image(fgLayer, 0, 0);
}
```

### 性能 — 尽可能向量化

p5.js绘制调用是昂贵的。对于成千上万的粒子：

```javascript
// 慢：单独形状
for (let p of particles) {
  ellipse(p.x, p.y, p.size);
}

// 快：使用beginShape()的单个形状
beginShape(POINTS);
for (let p of particles) {
  vertex(p.x, p.y);
}
endShape();

// 最快：用于大量计数的像素缓冲区
loadPixels();
for (let p of particles) {
  let idx = 4 * (floor(p.y) * width + floor(p.x));
  pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b; pixels[idx+3] = 255;
}
updatePixels();
```

参见`references/troubleshooting.md` § 性能。

### 实例模式用于多个Sketches

全局模式污染`window`。对于生产，使用实例模式：

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

当在一个页面上嵌入多个sketches或与框架集成时需要。

### WebGL模式陷阱

- `createCanvas(w, h, WEBGL)` — 原点是中心，不是左上角
- Y轴反转（WebGL中正Y向上，P2D中向下）
- `translate(-width/2, -height/2)`获得类似P2D的坐标
- 每个变换周围的`push()`/`pop()` —— 矩阵堆栈静默溢出
- `texture()`在`rect()`/`plane()`之前 —— 不是之后
- 自定义着色器：`createShader(vert, frag)` —— 在多个浏览器上测试

### 导出 — 键绑定惯例

每个sketch应该在`keyPressed()`中包含这些：

```javascript
function keyPressed() {
  if (key === 's' || key === 'S') saveCanvas('output', 'png');
  if (key === 'g' || key === 'G') saveGif('output', 5);
  if (key === 'r' || key === 'R') { randomSeed(millis()); noiseSeed(millis()); }
  if (key === ' ') CONFIG.paused = !CONFIG.paused;
}
```

### 无头视频导出 — 使用noLoop()

对于通过Puppeteer进行无头渲染，sketch**必须**在setup中使用`noLoop()`。没有它，p5的draw循环在截图缓慢时自由运行——sketch超前，您得到跳过的/重复的帧。

```javascript
function setup() {
  createCanvas(1920, 1080);
  pixelDensity(1);
  noLoop();                    // 捕获脚本控制帧前进
  window._p5Ready = true;      // 向捕获脚本发出就绪信号
}
```

捆绑的`scripts/export-frames.js`检测`_p5Ready`并精确调用`redraw()`一次每个捕获，以实现精确的1:1帧对应。参见`references/export-pipeline.md` § 确定性捕获。

对于多场景视频，使用每剪辑架构：每个场景一个HTML，独立渲染，用`ffmpeg -f concat`拼接。参见`references/export-pipeline.md` § 每剪辑架构。

### 代理工作流程

构建p5.js sketches时：

1. **编写HTML文件** — 单个独立文件，所有代码内联
2. **在浏览器中打开** — `open sketch.html`（macOS）或`xdg-open sketch.html`（Linux）
3. **本地资产**（字体、图像）需要服务器：`python3 -m http.server 8080`在项目目录中，然后打开`http://localhost:8080/sketch.html`
4. **导出PNG/GIF** — 如上所示添加`keyPressed()`快捷方式，告诉用户按哪个键
5. **无头导出** — `node scripts/export-frames.js sketch.html --frames 300`用于自动化帧捕获（sketch必须使用`noLoop()` + `_p5Ready`）
6. **MP4渲染** — `bash scripts/render.sh sketch.html output.mp4 --duration 30`
7. **迭代优化** — 编辑HTML文件，用户刷新浏览器查看更改
8. **按需加载引用** — 使用`skill_view(name="p5js", file_path="references/...")`在实施期间根据需要加载特定引用文件

## 性能目标

| 指标 | 目标 |
|--------|--------|
| 帧率（交互式） | 60fps持续 |
| 帧率（动画导出） | 至少30fps |
| 粒子计数（P2D形状） | 5,000-10,000 @ 60fps |
| 粒子计数（像素缓冲区） | 50,000-100,000 @ 60fps |
| 画布分辨率 | 高达3840x2160（导出），1920x1080（交互式） |
| 文件大小（HTML） | < 100KB（不包括CDN库） |
| 加载时间 | < 2秒到第一帧 |

## 引用

| 文件 | 内容 |
|------|----------|
| `references/core-api.md` | 画布设置、坐标系统、绘制循环、`push()`/`pop()`、离屏缓冲区、构图模式、`pixelDensity()`、响应式设计 |
| `references/shapes-and-geometry.md` | 2D图元、`beginShape()`/`endShape()`、贝塞尔/Catmull-Rom曲线、`vertex()`系统、自定义形状、`p5.Vector`、带符号距离场、SVG路径转换 |
| `references/visual-effects.md` | 噪声（Perlin、分形、域扭曲、旋度）、流场、粒子系统（物理、集群、拖尾）、像素操作、纹理生成（点画、孵化、半调）、反馈循环、反应扩散 |
| `references/animation.md` | 基于帧的动画、缓动函数、`lerp()`/`map()`、弹簧物理、状态机、时间线排序、`millis()`基于的计时、过渡模式 |
| `references/typography.md` | `text()`、`loadFont()`、`textToPoints()`、动态排版、文本遮罩、字体指标、响应式文字大小 |
| `references/color-systems.md` | `colorMode()`、HSB/HSL/RGB、`lerpColor()`、`paletteLerp()`、程序化调色板、色彩和谐、`blendMode()`、渐变渲染、精选调色板库 |
| `references/webgl-and-3d.md` | WebGL渲染器、3D图元、相机、光照、材质、自定义几何、GLSL着色器（`createShader()`、`createFilterShader()`）、帧缓冲区、后处理 |
| `references/interaction.md` | 鼠标事件、键盘状态、触摸输入、DOM元素、`createSlider()`/`createButton()`、音频输入（p5.sound FFT/振幅）、滚动驱动动画、响应式事件 |
| `references/export-pipeline.md` | `saveCanvas()`、`saveGif()`、`saveFrames()`、确定性无头捕获、ffmpeg帧到视频、CCapture.js、SVG导出、每剪辑架构、平台导出（fxhash）、视频陷阱 |
| `references/troubleshooting.md` | 性能分析、每像素预算、常见错误、浏览器兼容性、WebGL调试、字体加载问题、像素密度陷阱、内存泄漏、CORS |
| `templates/viewer.html` | 交互式查看器模板：种子导航（上一步/下一步/随机/跳转）、参数滑块、下载PNG、响应式画布。从这儿开始用于可探索的生成艺术 |

---

## 创意分化（仅在用户请求实验性/创意/独特输出时使用）

如果用户要求创意、实验性或非传统的解释方法，选择策略并在生成代码之前推理其步骤。

- **概念混合** — 当用户命名两个要组合的事物或想要混合审美时
- **SCAMPER** — 当用户想要已知的生成艺术模式的转折时
- **距离关联** — 当用户给出单个概念并想要探索时（"制作一些关于时间的东西"）

### 概念混合
1. 命名两个不同的视觉系统（例如，粒子物理+手写）
2. 映射对应关系（粒子=墨滴、力=笔压、场=字母形式）
3. 选择性地混合 — 保留产生有趣涌现视觉的映射
4. 将混合编写为统一系统，而不是并排的两个系统

### SCAMPER转换
采用已知的生成模式（流场、粒子系统、L系统、元胞自动机）并系统地转换它：
- **替换**：用文字字符替换圆形、用渐变替换线条
- **组合**：合并两个模式（流场+沃罗诺伊）
- **适配**：将2D模式应用于3D投影
- **修改**：夸大比例、扭曲坐标空间
- **目的**：使用物理模拟进行排版、使用排序算法进行颜色
- **消除**：移除网格、移除颜色、移除对称性
- **反转**：反向运行模拟、反转参数空间

### 距离关联
1. 锚定在用户的概念上（例如，"孤独"）
2. 在三个距离上生成关联：
   - 近（明显）：空房间、单个人物、寂静
   - 中（有趣）：在学校中一条鱼游向错误方向、带有无通知的手机、地铁车厢之间的间隙
   - 远（抽象）：质数、渐近曲线、凌晨3点的颜色
3. 开发中距离关联 — 它们足够具体以可视化，但又足够意外以令人感兴趣
