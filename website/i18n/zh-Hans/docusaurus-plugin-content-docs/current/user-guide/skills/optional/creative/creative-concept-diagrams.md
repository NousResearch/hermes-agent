---
title: "概念图"
sidebar_label: "概念图"
description: "生成扁平、简洁、支持亮/暗主题的 SVG 图表，作为独立的 HTML 文件，使用统一的教学视觉语言，包含 9 个语义色阶、句子大小写排版和自动深色模式。"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# 概念图

生成扁平、简洁、支持亮/暗主题的 SVG 图表，作为独立的 HTML 文件，使用统一的教学视觉语言，包含 9 个语义色阶、句子大小写排版和自动深色模式。最适合教育和非软件视觉 — 物理设置、化学机制、数学曲线、物理对象（飞机、涡轮、手机、机械手表）、解剖图、平面图、剖面图、叙事旅程（X 的生命周期、Y 的过程）、中心辐射系统集成（智慧城市、IoT）和爆炸分层视图。如果存在更专业的技能来处理该主题（专用软件/云架构、手绘草图、动画解释器等），优先使用那些 — 否则此技能也可以作为通用 SVG 图表后备方案，具有干净的教育外观。附带 15 个示例图表。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/creative/concept-diagrams` 安装 |
| 路径 | `optional-skills/creative/concept-diagrams` |
| 版本 | `0.1.0` |
| 作者 | v1k22（原始 PR），移植到 hermes-agent |
| 许可证 | MIT |
| 标签 | `diagrams`、`svg`、`visualization`、`education`、`physics`、`chemistry`、`engineering` |
| 相关技能 | [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram)、[`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw)、`generative-widgets` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# 概念图

使用统一的扁平、简洁设计系统生成生产级 SVG 图表。输出是单个自包含 HTML 文件，可在任何现代浏览器中一致渲染，支持自动亮/暗模式。

## 范围

**最适合：**
- 物理设置、化学机制、数学曲线、生物学
- 物理对象（飞机、涡轮、手机、机械手表、细胞）
- 解剖图、剖面图、爆炸分层视图
- 平面图、建筑转换
- 叙事旅程（X 的生命周期、Y 的过程）
- 中心辐射系统集成（智慧城市、IoT 网络、电网）
- 任何领域的教育/教科书风格视觉
- 定量图表（分组条形图、能量曲线）

**首选其他：**
- 具有深色技术美学的专用软件/云基础设施架构（考虑 `architecture-diagram`）
- 手绘白板草图（考虑 `excalidraw`）
- 动画解释器或视频输出（考虑动画技能）

如果该主题有更专业的技能可用，优先使用。如果没有合适的，此技能可作为通用 SVG 图表后备方案 — 输出将携带下面描述的干净教育美学，这对几乎任何主题都是合理的默认值。

## 工作流程

1. 确定图表类型（见下面的图表类型）。
2. 使用设计系统规则布局组件。
3. 使用 `templates/template.html` 作为包装器编写完整 HTML 页面 — 将您的 SVG 粘贴到模板中 `<!-- PASTE SVG HERE -->` 的位置。
4. 保存为独立的 `.html` 文件（例如 `~/my-diagram.html` 或 `./my-diagram.html`）。
5. 用户直接在浏览器中打开 — 无需服务器，无依赖。

可选：如果用户想要多个图表的可浏览图库，请参阅底部的"本地预览服务器"。

加载 HTML 模板：
```
skill_view(name="concept-diagrams", file_path="templates/template.html")
```

模板嵌入完整 CSS 设计系统（`c-*` 颜色类、文本类、亮/暗变量、箭头标记样式）。您生成的 SVG 依赖于托管页面上存在这些类。

---

## 设计系统

### 理念

- **扁平**：无渐变、投影、模糊、发光或霓虹效果。
- **简洁**：只展示必要的。框内无装饰图标。
- **一致**：每个图表使用相同的颜色、间距、排版和描边宽度。
- **支持深色模式**：所有颜色通过 CSS 类自动适配 — 无需每个模式单独的 SVG。

### 颜色调板

9 个色阶，每个有 7 个色度。在 `<g>` 或形状元素上放置类名；模板 CSS 处理两种模式。

| Class      | 50（最浅） | 100     | 200     | 400     | 600     | 800     | 900（最深） |
|------------|------------|---------|---------|---------|---------|---------|---------------|
| `c-purple` | #EEEDFE | #CECBF6 | #AFA9EC | #7F77DD | #534AB7 | #3C3489 | #26215C |
| `c-teal`   | #E1F5EE | #9FE1CB | #5DCAA5 | #1D9E75 | #0F6E56 | #085041 | #04342C |
| `c-coral`  | #FAECE7 | #F5C4B3 | #F0997B | #D85A30 | #993C1D | #712B13 | #4A1B0C |
| `c-pink`   | #FBEAF0 | #F4C0D1 | #ED93B1 | #D4537E | #993556 | #72243E | #4B1528 |
| `c-gray`   | #F1EFE8 | #D3D1C7 | #B4B2A9 | #888780 | #5F5E5A | #444441 | #2C2C2A |
| `c-blue`   | #E6F1FB | #B5D4F4 | #85B7EB | #378ADD | #185FA5 | #0C447C | #042C53 |
| `c-green`  | #EAF3DE | #C0DD97 | #97C459 | #639922 | #3B6D11 | #27500A | #173404 |
| `c-amber`  | #FAEEDA | #FAC775 | #EF9F27 | #BA7517 | #854F0B | #633806 | #412402 |
| `c-red`    | #FCEBEB | #F7C1C1 | #F09595 | #E24B4A | #A32D2D | #791F1F | #501313 |

#### 颜色分配规则

颜色编码**含义**，而非序列。永远不要像彩虹一样循环使用颜色。

- 按**类别**对节点分组 — 相同类型的所有节点共享一种颜色。
- 使用 `c-gray` 表示中性/结构节点（开始、结束、通用步骤、用户）。
- 每个图表使用 **2-3 种颜色**，而非 6+ 种。
- 优先使用 `c-purple`、`c-teal`、`c-coral`、`c-pink` 作为通用类别。
- 将 `c-blue`、`c-green`、`c-amber`、`c-red` 保留用于语义含义（信息、成功、警告、错误）。

亮/暗色度映射（由模板 CSS 处理 — 只需使用类）：
- 亮色模式：50 填充 + 600 描边 + 800 标题 / 600 副标题
- 深色模式： 800 填充 + 200 描边 + 100 标题 / 200 副标题

### 排版

只有两种字体大小。没有例外。

| Class | Size | Weight | Use |
|-------|------|--------|-----|
| `th`  | 14px | 500    | 节点标题、区域标签 |
| `ts`  | 12px | 400    | 副标题、描述、箭头标签 |
| `t`   | 14px | 400    | 通用文本 |

- **始终使用句子大小写**。永远不要标题大小写，不要全大写。
- 每个 `<text>` 必须有类（`t`、`ts` 或 `th`）。无未分类文本。
- 框内所有文本使用 `dominant-baseline="central"`。
- 框内居中文本使用 `text-anchor="middle"`。

**宽度估算（约）：**
- 14px 500 权重：每字符约 8px
- 12px 400 权重：每字符约 6.5px
- 始终验证：`box_width >= (char_count × px_per_char) + 48`（每边 24px 内边距）

### 间距与布局

- **视口**：`viewBox="0 0 680 H"`，其中 H = 内容高度 + 40px 缓冲。
- **安全区域**：x=40 到 x=640，y=40 到 y=(H-40)。
- **框之间**：最小 60px 间隙。
- **框内**：水平内边距 24px，垂直内边距 12px。
- **箭头间隙**：箭头和框边缘之间 10px。
- **单行框**：高度 44px。
- **双行框**：高度 56px，标题和副标题基线之间 18px。
- **容器内边距**：每个容器内最小 20px。
- **最大嵌套深度**：2-3 层。在 680px 宽度下过深会难以阅读。

### 描边与形状

- **描边宽度**：所有节点边框 0.5px。不是 1px，不是 2px。
- **矩形圆角**：`rx="8"` 用于节点，`rx="12"` 用于内部容器，`rx="16"` 到 `rx="20"` 用于外部容器。
- **连接器路径**：必须设置 `fill="none"`。否则 SVG 默认为 `fill: black`。

### 箭头标记

在**每个** SVG 开头包含此 `<defs>` 块：

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

在线上使用 `marker-end="url(#arrow)"`。箭头通过 `context-stroke` 继承线条颜色。

### CSS 类（由模板提供）

模板页面提供：

- 文本：`.t`、`.ts`、`.th`
- 中性：`.box`、`.arr`、`.leader`、`.node`
- 颜色色阶：`.c-purple`、`.c-teal`、`.c-coral`、`.c-pink`、`.c-gray`、`.c-blue`、`.c-green`、`.c-amber`、`.c-red`（全部自动亮/暗模式）

您**不需要**重新定义这些 — 只需在 SVG 中应用它们。模板文件包含完整 CSS 定义。

---

## SVG 模板

模板页面内每个 SVG 以此确结构开始：

```xml
<svg width="100%" viewBox="0 0 680 {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Diagram content here -->

</svg>
```

将 `{HEIGHT}` 替换为实际计算的高度（最后一个元素底部 + 40px）。

### 节点模式

**单行节点（44px）：**
```xml
<g class="node c-blue">
  <rect x="100" y="20" width="180" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="190" y="42" text-anchor="middle" dominant-baseline="central">Service name</text>
</g>
```

**双行节点（56px）：**
```xml
<g class="node c-teal">
  <rect x="100" y="20" width="200" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="200" y="38" text-anchor="middle" dominant-baseline="central">Service name</text>
  <text class="ts" x="200" y="56" text-anchor="middle" dominant-baseline="central">Short description</text>
</g>
```

**连接器（无标签）：**
```xml
<line x1="200" y1="76" x2="200" y2="120" class="arr" marker-end="url(#arrow)"/>
```

**容器（虚线或实线）：**
```xml
<g class="c-purple">
  <rect x="40" y="92" width="600" height="300" rx="16" stroke-width="0.5"/>
  <text class="th" x="66" y="116">Container label</text>
  <text class="ts" x="66" y="134">Subtitle info</text>
</g>
```

---

## 图表类型

选择适合主题的布局：

1. **流程图** — CI/CD 管道、请求生命周期、审批工作流、数据处理。单向流（从上到下或从左到右）。每行最多 4-5 个节点。
2. **结构/包含** — 云基础设施嵌套、分层系统架构。带有内部区域的大型外部容器。虚线矩形用于逻辑分组。
3. **API/端点映射** — REST 路由、GraphQL 模式。从根开始的树，分支到资源组，每个包含端点节点。
4. **微服务拓扑** — 服务网格、事件驱动系统。服务作为节点，箭头表示通信模式，消息队列介于中间。
5. **数据流** — ETL 管道、流架构。从源通过处理到接收器的从左到右流。
6. **物理/结构** — 车辆、建筑、硬件、解剖。使用匹配物理形式的形状 — `<path>` 用于曲线主体，`<polygon>` 用于锥形形状，`<ellipse>`/`<circle>` 用于圆柱形零件，嵌套 `<rect>` 用于隔间。参见 `references/physical-shape-cookbook.md`。
7. **基础设施/系统集成** — 智慧城市、IoT 网络、多域系统。中心辐射布局，中心平台连接子系统。语义线样式（`.data-line`、`.power-line`、`.water-pipe`、`.road`）。参见 `references/infrastructure-patterns.md`。
8. **UI/仪表盘模拟** — 管理面板、监控仪表盘。带有嵌套图表/仪表/指示器元素的屏幕框架。参见 `references/dashboard-patterns.md`。

对于物理、基础设施和仪表盘图表，在生成前加载匹配的参考文件 — 每个提供现成的 CSS 类和形状原语。

---

## 验证清单

在完成任何 SVG 之前，验证以下所有项：

1. 每个 `<text>` 都有类 `t`、`ts` 或 `th`。
2. 框内每个 `<text>` 都有 `dominant-baseline="central"`。
3. 用作箭头的每个连接器 `<path>` 或 `<line>` 都有 `fill="none"`。
4. 没有箭头线穿过无关的框。
5. `box_width >= (longest_label_chars × 8) + 48` 用于 14px 文本。
6. `box_width >= (longest_label_chars × 6.5) + 48` 用于 12px 文本。
7. 视口高度 = 最底部元素 + 40px。
8. 所有内容保持在 x=40 到 x=640 内。
9. 颜色类（`c-*`）在 `<g>` 或形状元素上，从不在 `<path>` 连接器上。
10. 箭头 `<defs>` 块存在。
11. 无渐变、投影、模糊或发光效果。
12. 所有节点边框描边宽度为 0.5px。

---

## 输出与预览

### 默认：独立 HTML 文件

写入用户可直接打开的单个 `.html` 文件。无服务器，无依赖，离线可用。模式：

```python
# 1. 加载模板
template = skill_view("concept-diagrams", "templates/template.html")

# 2. 填写标题、副标题并粘贴您的 SVG
html = template.replace(
    "<!-- DIAGRAM TITLE HERE -->", "SN2 reaction mechanism"
).replace(
    "<!-- OPTIONAL SUBTITLE HERE -->", "Bimolecular nucleophilic substitution"
).replace(
    "<!-- PASTE SVG HERE -->", svg_content
)

# 3. 写入用户选择的路径（或默认 ./）
write_file("./sn2-mechanism.html", html)
```

告诉用户如何打开：

```
# macOS
open ./sn2-mechanism.html
# Linux
xdg-open ./sn2-mechanism.html
```

### 可选：本地预览服务器（多图表图库）

仅在用户明确想要多个图表的可浏览图库时使用。

**规则：**
- 仅绑定到 `127.0.0.1`。永远不要 `0.0.0.0`。在共享网络上将图表暴露在所有网络接口是安全隐患。
- 选择空闲端口（不要硬编码一个）并告诉用户选择的 URL。
- 服务器是可选的且需要选择加入 — 优先使用独立 HTML 文件。

推荐模式（让操作系统选择空闲临时端口）：

```bash
# 将每个图表放在 .diagrams/ 下的各自文件夹中
mkdir -p .diagrams/sn2-mechanism
# ...写入 .diagrams/sn2-mechanism/index.html...

# 仅在回环接口上服务，空闲端口
cd .diagrams && python3 -c "
import http.server, socketserver
with socketserver.TCPServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler) as s:
    print(f'Serving at http://127.0.0.1:{s.server_address[1]}/')
    s.serve_forever()
" &
```

如果用户坚持固定端口，使用 `127.0.0.1:<port>` — 仍然永远不要 `0.0.0.0`。记录如何停止服务器（`kill %1` 或 `pkill -f "http.server"`）。

---

## 示例参考

`examples/` 目录附带 15 个完整、经测试的图表。在编写同类型新图表之前浏览它们以获取工作模式：

| File | Type | Demonstrates |
|------|------|--------------|
| `hospital-emergency-department-flow.md` | Flowchart | Priority routing with semantic colors |
| `feature-film-production-pipeline.md` | Flowchart | Phased workflow, horizontal sub-flows |
| `automated-password-reset-flow.md` | Flowchart | Auth flow with error branches |
| `autonomous-llm-research-agent-flow.md` | Flowchart | Loop-back arrows, decision branches |
| `place-order-uml-sequence.md` | Sequence | UML sequence diagram style |
| `commercial-aircraft-structure.md` | Physical | Paths, polygons, ellipses for realistic shapes |
| `wind-turbine-structure.md` | Physical cross-section | Underground/above-ground separation, color coding |
| `smartphone-layer-anatomy.md` | Exploded view | Alternating left/right labels, layered components |
| `apartment-floor-plan-conversion.md` | Floor plan | Walls, doors, proposed changes in dotted red |
| `banana-journey-tree-to-smoothie.md` | Narrative journey | Winding path, progressive state changes |
| `cpu-ooo-microarchitecture.md` | Hardware pipeline | Fan-out, memory hierarchy sidebar |
| `sn2-reaction-mechanism.md` | Chemistry | Molecules, curved arrows, energy profile |
| `smart-city-infrastructure.md` | Hub-spoke | Semantic line styles per system |
| `electricity-grid-flow.md` | Multi-stage flow | Voltage hierarchy, flow markers |
| `ml-benchmark-grouped-bar-chart.md` | Chart | Grouped bars, dual axis |

使用以下方式加载任何示例：
```
skill_view(name="concept-diagrams", file_path="examples/<filename>")
```

---

## 快速参考：何时使用什么

| User says | Diagram type | Suggested colors |
|-----------|--------------|------------------|
| "show the pipeline" | Flowchart | gray start/end, purple steps, red errors, teal deploy |
| "draw the data flow" | Data pipeline (left-right) | gray sources, purple processing, teal sinks |
| "visualize the system" | Structural (containment) | purple container, teal services, coral data |
| "map the endpoints" | API tree | purple root, one ramp per resource group |
| "show the services" | Microservice topology | gray ingress, teal services, purple bus, coral workers |
| "draw the aircraft/vehicle" | Physical | paths, polygons, ellipses for realistic shapes |
| "smart city / IoT" | Hub-spoke integration | semantic line styles per subsystem |
| "show the dashboard" | UI mockup | dark screen, chart colors: teal, purple, coral for alerts |
| "power grid / electricity" | Multi-stage flow | voltage hierarchy (HV/MV/LV line weights) |
| "wind turbine / turbine" | Physical cross-section | foundation + tower cutaway + nacelle color-coded |
| "journey of X / lifecycle" | Narrative journey | winding path, progressive state changes |
| "layers of X / exploded" | Exploded layer view | vertical stack, alternating labels |
| "CPU / pipeline" | Hardware pipeline | vertical stages, fan-out to execution ports |
| "floor plan / apartment" | Floor plan | walls, doors, proposed changes in dotted red |
| "reaction mechanism" | Chemistry | atoms, bonds, curved arrows, transition state, energy profile |
