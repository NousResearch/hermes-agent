---
title: "概念图"
sidebar_label: "概念图"
description: "以统一的教育视觉语言生成扁平、简约、支持明暗模式的 SVG 图表，输出为独立 HTML 文件，包含 9 种语义色阶、句首大写排版及自动暗色模式。..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 概念图

以统一的教育视觉语言生成扁平、简约、支持明暗模式的 SVG 图表，输出为独立 HTML 文件，包含 9 种语义色阶、句首大写排版及自动暗色模式。最适合教育类和非软件类视觉内容——物理装置、化学机制、数学曲线、实物（飞机、涡轮机、智能手机、机械表）、解剖图、平面图、截面图、叙事流程（X 的生命周期、Y 的过程）、中心辐射型系统集成（智慧城市、IoT）以及爆炸分层视图。若已有更专业的 skill 适用于该主题（专用软件/云架构、手绘草图、动画说明等），优先使用那些 skill——否则本 skill 也可作为通用 SVG 图表的备选方案，具备简洁的教育风格外观。内置 15 个示例图表。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/creative/concept-diagrams` 安装 |
| 路径 | `optional-skills/creative/concept-diagrams` |
| 版本 | `0.1.0` |
| 作者 | v1k22（原始 PR），移植至 hermes-agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `diagrams`, `svg`, `visualization`, `education`, `physics`, `chemistry`, `engineering` |
| 相关 skills | [`architecture-diagram`](/user-guide/skills/bundled/creative/creative-architecture-diagram), [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发本 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 概念图

使用统一的扁平、简约设计系统生成生产级 SVG 图表。输出为单个自包含 HTML 文件，可在任何现代浏览器中一致渲染，并自动支持明暗模式。

## 适用范围

**最适合：**
- 物理装置、化学机制、数学曲线、生物学
- 实物（飞机、涡轮机、智能手机、机械表、细胞）
- 解剖图、截面图、爆炸分层视图
- 平面图、建筑改造图
- 叙事流程（X 的生命周期、Y 的过程）
- 中心辐射型系统集成（智慧城市、IoT 网络、电网）
- 任何领域的教育/教科书风格视觉内容
- 定量图表（分组柱状图、能量曲线）

**优先考虑其他方案：**
- 具有深色科技风格的专用软件/云基础设施架构（如有 `architecture-diagram` 可用，优先使用）
- 手绘白板草图（如有 `excalidraw` 可用，优先使用）
- 动画说明或视频输出（考虑动画 skill）

若已有更专业的 skill 适用于该主题，优先使用。若无合适选项，本 skill 可作为通用 SVG 图表备选方案——输出将呈现下文描述的简洁教育风格，适用于几乎任何主题。

## 工作流程

1. 确定图表类型（见下方"图表类型"）。
2. 使用设计系统规则布局组件。
3. 使用 `templates/template.html` 作为包装器编写完整 HTML 页面——将 SVG 粘贴到模板中 `<!-- PASTE SVG HERE -->` 的位置。
4. 保存为独立 `.html` 文件（例如 `~/my-diagram.html` 或 `./my-diagram.html`）。
5. 用户直接在浏览器中打开——无需服务器，无需依赖。

可选：若用户需要可浏览的多图表画廊，参见底部"本地预览服务器"。

加载 HTML 模板：
```
skill_view(name="concept-diagrams", file_path="templates/template.html")
```

模板内嵌完整 CSS 设计系统（`c-*` 颜色类、文本类、明暗变量、箭头标记样式）。你生成的 SVG 依赖这些类存在于宿主页面中。

---

## 设计系统

### 设计理念

- **扁平**：无渐变、无投影、无模糊、无发光、无霓虹效果。
- **简约**：只展示核心内容，框内无装饰性图标。
- **一致**：每张图表使用相同的颜色、间距、排版和描边宽度。
- **暗色模式就绪**：所有颜色通过 CSS 类自动适配——无需为每种模式单独编写 SVG。

### 调色板

9 种色阶，每种 7 个色阶值。将类名放在 `<g>` 或形状元素上；模板 CSS 自动处理明暗两种模式。

| 类名 | 50（最浅） | 100 | 200 | 400 | 600 | 800 | 900（最深） |
|------------|---------------|---------|---------|---------|---------|---------|---------------|
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

颜色编码**语义**，而非顺序。切勿像彩虹一样循环使用颜色。

- 按**类别**对节点分组——同类型的所有节点共用一种颜色。
- 对中性/结构性节点（起点、终点、通用步骤、用户）使用 `c-gray`。
- 每张图表使用 **2-3 种颜色**，而非 6 种以上。
- 通用类别优先使用 `c-purple`、`c-teal`、`c-coral`、`c-pink`。
- 将 `c-blue`、`c-green`、`c-amber`、`c-red` 保留用于语义含义（信息、成功、警告、错误）。

明暗色阶映射（由模板 CSS 处理——直接使用类名即可）：
- 亮色模式：50 填充 + 600 描边 + 800 标题 / 600 副标题
- 暗色模式：800 填充 + 200 描边 + 100 标题 / 200 副标题

### 排版

只有两种字体大小，不得例外。

| 类名 | 大小 | 字重 | 用途 |
|-------|------|--------|-----|
| `th`  | 14px | 500    | 节点标题、区域标签 |
| `ts`  | 12px | 400    | 副标题、描述、箭头标签 |
| `t`   | 14px | 400    | 通用文本 |

- **始终使用句首大写。** 禁止首字母大写（Title Case），禁止全大写（ALL CAPS）。
- 每个 `<text>` 必须带有类名（`t`、`ts` 或 `th`），不得有无类名的文本。
- 框内所有文本使用 `dominant-baseline="central"`。
- 框内居中文本使用 `text-anchor="middle"`。

**宽度估算（近似值）：**
- 14px 字重 500：每字符约 8px
- 12px 字重 400：每字符约 6.5px
- 始终验证：`box_width >= (字符数 × px/字符) + 48`（每侧 24px 内边距）

### 间距与布局

- **ViewBox**：`viewBox="0 0 680 H"`，其中 H = 内容高度 + 40px 缓冲。
- **安全区域**：x=40 至 x=640，y=40 至 y=(H-40)。
- **框间距**：最小 60px。
- **框内边距**：水平 24px，垂直 12px。
- **箭头间隙**：箭头与框边缘之间 10px。
- **单行框**：高度 44px。
- **双行框**：高度 56px，标题与副标题基线间距 18px。
- **容器内边距**：每个容器内部最小 20px。
- **最大嵌套层级**：2-3 层。在 680px 宽度下更深的嵌套会难以阅读。

### 描边与形状

- **描边宽度**：所有节点边框 0.5px，不得使用 1px 或 2px。
- **矩形圆角**：节点使用 `rx="8"`，内层容器使用 `rx="12"`，外层容器使用 `rx="16"` 至 `rx="20"`。
- **连接路径**：必须设置 `fill="none"`，否则 SVG 默认填充为黑色。

### 箭头标记

在**每个** SVG 开头包含以下 `<defs>` 块：

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

在线条上使用 `marker-end="url(#arrow)"`。箭头通过 `context-stroke` 继承线条颜色。

### CSS 类（由模板提供）

模板页面提供：

- 文本：`.t`、`.ts`、`.th`
- 中性：`.box`、`.arr`、`.leader`、`.node`
- 色阶：`.c-purple`、`.c-teal`、`.c-coral`、`.c-pink`、`.c-gray`、`.c-blue`、`.c-green`、`.c-amber`、`.c-red`（均自动支持明暗模式）

你**无需**重新定义这些类——直接在 SVG 中应用即可。模板文件包含完整的 CSS 定义。

---

## SVG 样板代码

模板页面中的每个 SVG 均以如下结构开头：

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

将 `{HEIGHT}` 替换为实际计算高度（最后一个元素底部 + 40px）。

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

**连接线（无标签）：**
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

根据主题选择合适的布局：

1. **流程图** — CI/CD 流水线、请求生命周期、审批工作流、数据处理。单向流（从上到下或从左到右），每行最多 4-5 个节点。
2. **结构/包含图** — 云基础设施嵌套、分层系统架构。大型外层容器包含内层区域，虚线矩形表示逻辑分组。
3. **API/端点映射** — REST 路由、GraphQL schema。从根节点树状展开，分支到资源组，每组包含端点节点。
4. **微服务拓扑** — 服务网格、事件驱动系统。服务作为节点，箭头表示通信模式，消息队列位于服务之间。
5. **数据流图** — ETL 流水线、流式架构。从数据源经处理流向数据汇，方向从左到右。
6. **实物/结构图** — 交通工具、建筑、硬件、解剖图。使用与实物形态匹配的形状——弯曲体用 `<path>`，锥形用 `<polygon>`，圆柱部件用 `<ellipse>`/`<circle>`，隔间用嵌套 `<rect>`。参见 `references/physical-shape-cookbook.md`。
7. **基础设施/系统集成图** — 智慧城市、IoT 网络、多域系统。中心辐射布局，中央平台连接各子系统。按系统使用语义线型（`.data-line`、`.power-line`、`.water-pipe`、`.road`）。参见 `references/infrastructure-patterns.md`。
8. **UI/仪表盘原型** — 管理面板、监控仪表盘。屏幕框架内嵌套图表/仪表/指示器元素。参见 `references/dashboard-patterns.md`。

对于实物图、基础设施图和仪表盘图，生成前请先加载对应的参考文件——每个文件提供现成的 CSS 类和形状原语。

---

## 验证清单

在最终确定任何 SVG 之前，验证以下**所有**项目：

1. 每个 `<text>` 都有类名 `t`、`ts` 或 `th`。
2. 框内每个 `<text>` 都有 `dominant-baseline="central"`。
3. 用作箭头的每个连接 `<path>` 或 `<line>` 都有 `fill="none"`。
4. 没有箭头线穿过无关的框。
5. 14px 文本：`box_width >= (最长标签字符数 × 8) + 48`。
6. 12px 文本：`box_width >= (最长标签字符数 × 6.5) + 48`。
7. ViewBox 高度 = 最底部元素 + 40px。
8. 所有内容在 x=40 至 x=640 范围内。
9. 颜色类（`c-*`）放在 `<g>` 或形状元素上，不得放在 `<path>` 连接线上。
10. 箭头 `<defs>` 块存在。
11. 无渐变、投影、模糊或发光效果。
12. 所有节点边框描边宽度为 0.5px。

---

## 输出与预览

### 默认：独立 HTML 文件

写入单个 `.html` 文件，用户可直接打开。无需服务器，无需依赖，离线可用。模式：

```python
# 1. Load the template
template = skill_view("concept-diagrams", "templates/template.html")

# 2. Fill in title, subtitle, and paste your SVG
html = template.replace(
    "<!-- DIAGRAM TITLE HERE -->", "SN2 reaction mechanism"
).replace(
    "<!-- OPTIONAL SUBTITLE HERE -->", "Bimolecular nucleophilic substitution"
).replace(
    "<!-- PASTE SVG HERE -->", svg_content
)

# 3. Write to a user-chosen path (or ./ by default)
write_file("./sn2-mechanism.html", html)
```

告知用户如何打开：

```
# macOS
open ./sn2-mechanism.html
# Linux
xdg-open ./sn2-mechanism.html
```

### 可选：本地预览服务器（多图表画廊）

仅在用户明确需要可浏览的多图表画廊时使用。

**规则：**
- 仅绑定到 `127.0.0.1`，绝不使用 `0.0.0.0`。在共享网络上将图表暴露在所有网络接口上存在安全风险。
- 选择空闲端口（不得硬编码），并告知用户所选 URL。
- 服务器是可选的、需用户主动选择的——优先使用独立 HTML 文件。

推荐模式（让操作系统选择空闲的临时端口）：

```bash
# Put each diagram in its own folder under .diagrams/
mkdir -p .diagrams/sn2-mechanism
# ...write .diagrams/sn2-mechanism/index.html...

# Serve on loopback only, free port
cd .diagrams && python3 -c "
import http.server, socketserver
with socketserver.TCPServer(('127.0.0.1', 0), http.server.SimpleHTTPRequestHandler) as s:
    print(f'Serving at http://127.0.0.1:{s.server_address[1]}/')
    s.serve_forever()
" &
```

若用户坚持使用固定端口，使用 `127.0.0.1:<port>`——仍然不得使用 `0.0.0.0`。说明如何停止服务器（`kill %1` 或 `pkill -f "http.server"`）。

---

## 示例参考

`examples/` 目录内置 15 个完整、经过测试的图表。在编写同类型新图表之前，先浏览这些示例以获取可用模式：

| 文件 | 类型 | 演示内容 |
|------|------|--------------|
| `hospital-emergency-department-flow.md` | 流程图 | 带语义颜色的优先级路由 |
| `feature-film-production-pipeline.md` | 流程图 | 分阶段工作流、水平子流程 |
| `automated-password-reset-flow.md` | 流程图 | 带错误分支的认证流程 |
| `autonomous-llm-research-agent-flow.md` | 流程图 | 回环箭头、决策分支 |
| `place-order-uml-sequence.md` | 时序图 | UML 时序图风格 |
| `commercial-aircraft-structure.md` | 实物图 | 使用路径、多边形、椭圆绘制真实形状 |
| `wind-turbine-structure.md` | 实物截面图 | 地下/地上分离、颜色编码 |
| `smartphone-layer-anatomy.md` | 爆炸视图 | 左右交替标签、分层组件 |
| `apartment-floor-plan-conversion.md` | 平面图 | 墙体、门、虚线红色标注改造方案 |
| `banana-journey-tree-to-smoothie.md` | 叙事流程 | 蜿蜒路径、渐进状态变化 |
| `cpu-ooo-microarchitecture.md` | 硬件流水线 | 扇出、内存层次侧边栏 |
| `sn2-reaction-mechanism.md` | 化学图 | 分子、弯曲箭头、能量曲线 |
| `smart-city-infrastructure.md` | 中心辐射图 | 每个系统使用语义线型 |
| `electricity-grid-flow.md` | 多阶段流程图 | 电压层次、流向标记 |
| `ml-benchmark-grouped-bar-chart.md` | 图表 | 分组柱状图、双轴 |

使用以下命令加载任意示例：
```
skill_view(name="concept-diagrams", file_path="examples/<filename>")
```

---

## 快速参考：何时使用何种图表

| 用户说 | 图表类型 | 建议颜色 |
|-----------|--------------|------------------|
| "展示流水线" | 流程图 | 灰色起止点，紫色步骤，红色错误，青色部署 |
| "画数据流" | 数据流水线（从左到右） | 灰色数据源，紫色处理，青色数据汇 |
| "可视化系统" | 结构图（包含关系） | 紫色容器，青色服务，珊瑚色数据 |
| "映射端点" | API 树状图 | 紫色根节点，每个资源组一种色阶 |
| "展示服务" | 微服务拓扑 | 灰色入口，青色服务，紫色总线，珊瑚色 worker |
| "画飞机/交通工具" | 实物图 | 路径、多边形、椭圆绘制真实形状 |
| "智慧城市/IoT" | 中心辐射集成图 | 每个子系统使用语义线型 |
| "展示仪表盘" | UI 原型 | 深色屏幕，图表颜色：青色、紫色、珊瑚色告警 |
| "电网/电力" | 多阶段流程图 | 电压层次（高/中/低压线宽） |
| "风力涡轮机/涡轮机" | 实物截面图 | 基础 + 塔筒截面 + 机舱颜色编码 |
| "X 的旅程/生命周期" | 叙事流程 | 蜿蜒路径，渐进状态变化 |
| "X 的层次/爆炸图" | 爆炸分层视图 | 垂直堆叠，交替标签 |
| "CPU/流水线" | 硬件流水线 | 垂直阶段，扇出到执行端口 |
| "平面图/公寓" | 平面图 | 墙体、门，虚线红色标注改造方案 |
| "反应机制" | 化学图 | 原子、化学键、弯曲箭头、过渡态、能量曲线 |