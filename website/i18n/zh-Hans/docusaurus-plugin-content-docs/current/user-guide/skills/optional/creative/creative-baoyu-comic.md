---
title: "宝玉知识漫画 — 知识漫画：教育、传记、教程"
sidebar_label: "宝玉知识漫画"
description: "知识漫画：教育、传记、教程"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 宝玉知识漫画

知识漫画：教育、传记、教程。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/creative/baoyu-comic` 安装 |
| 路径 | `optional-skills/creative/baoyu-comic` |
| 版本 | `1.56.1` |
| 作者 | 宝玉 (JimLiu) |
| 许可证 | MIT |
| 平台 | linux、macos、windows |
| 标签 | `comic`、`knowledge-comic`、`creative`、`image-generation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# 知识漫画创建器

改编自 [baoyu-comic](https://github.com/JimLiu/baoyu-skills)，适配 Hermes Agent 的工具生态。

使用灵活的画风 × 调性组合创建原创知识漫画。

## 何时使用

当用户要求创建知识/教育漫画、传记漫画、教程漫画，或使用"知识漫画"、"教育漫画"、"Logicomix-style"等术语时触发。用户提供内容（文本、文件路径、URL 或主题），可选指定画风、调性、布局、宽高比或语言。

## 参考图像

Hermes 的 `image_generate` 工具**仅接受提示**——它接受文本提示和宽高比，并返回图像 URL。它**不**接受参考图像。当用户提供参考图像时，用于**提取文本特征**嵌入到每页提示中。

**接收**：接受用户提供的文件路径。
- 文件路径 → 复制到 `refs/NN-ref-{slug}.{ext}` 以备溯源
- 无路径的粘贴图像 → 通过 `clarify` 向用户请求路径，或以文本回退方式口头提取风格特征
- 无参考 → 跳过此部分

**使用模式**（每个参考）：

| 使用 | 效果 |
|------|------|
| `style` | 提取风格特征（线条处理、纹理、氛围）并追加到每页提示主体 |
| `palette` | 提取十六进制颜色并追加到每页提示主体 |
| `scene` | 提取场景构图或主题注释并追加到相关页 |

角色一致性通过 `characters/characters.md`（在步骤 3 编写）中的**文本描述**驱动，这些描述在每页提示中内联嵌入（步骤 5）。

## 选项

### 视觉维度

| 选项 | 值 | 描述 |
|------|-----|------|
| 画风 | ligne-claire（默认）、manga、realistic、ink-brush、chalk、minimalist | 画风/渲染技术 |
| 调性 | neutral（默认）、warm、dramatic、romantic、energetic、vintage、action | 氛围/情绪 |
| 布局 | standard（默认）、cinematic、dense、splash、mixed、webtoon、four-panel | 面板排列 |
| 宽高比 | 3:4（默认，竖版）、4:3（横版）、16:9（宽屏） | 页面宽高比 |
| 语言 | auto（默认）、zh、en、ja 等 | 输出语言 |

### 部分工作流选项

| 选项 | 描述 |
|------|------|
| 仅分镜 | 仅生成分镜，跳过提示和图像 |
| 仅提示 | 生成分镜+提示，跳过图像 |
| 仅图像 | 从现有提示目录生成图像 |
| 重新生成 N | 仅重新生成特定页（如 `3` 或 `2,5,8`） |

## 工作流程

### 进度清单

```
漫画进度：
- [ ] 步骤 1：设置与分析
  - [ ] 1.1 分析内容
  - [ ] 1.2 检查现有目录
- [ ] 步骤 2：确认 - 风格与选项 ⚠️ 必需
- [ ] 步骤 3：生成分镜 + 角色
- [ ] 步骤 4：审核大纲（条件性）
- [ ] 步骤 5：生成提示
- [ ] 步骤 6：审核提示（条件性）
- [ ] 步骤 7：生成图像
  - [ ] 7.1 生成角色表（如需）
  - [ ] 7.2 生成页面（角色描述嵌入提示）
- [ ] 步骤 8：完成报告
```

### 流程

```
输入 → 分析 → [检查现有？] → [确认：风格+审核] → 分镜 → [审核？] → 提示 → [审核？] → 图像 → 完成
```

### 步骤 7：图像生成

使用 Hermes 内置的 `image_generate` 工具进行所有图像渲染。其 schema 仅接受 `prompt` 和 `aspect_ratio`（`landscape` | `portrait` | `square`）；它**返回 URL** 而非本地文件。因此每个生成的页面或角色表必须下载到输出目录。

**提示文件要求（硬性）**：在调用 `image_generate` 之前，将每张图像的完整最终提示写入 `prompts/` 下的独立文件（命名：`NN-{type}-[slug].md`）。提示文件是可复现性记录。

**宽高比映射**：

| 分镜比例 | `image_generate` 格式 |
|---------|----------------------|
| `3:4`、`9:16`、`2:3` | `portrait` |
| `4:3`、`16:9`、`3:2` | `landscape` |
| `1:1` | `square` |

**下载步骤** — 每次 `image_generate` 调用后：
1. 从工具结果读取 URL
2. 使用**绝对**输出路径获取图像字节
3. 在继续下一页前验证文件存在且非空

**永远不要依赖 shell CWD 持久性来设置 `-o` 路径。**

## 陷阱

- 图像生成：每页 10-30 秒；失败时自动重试一次
- **始终下载** `image_generate` 返回的 URL 到本地 PNG
- **对 `curl -o` 使用绝对路径**
- 对敏感公众人物使用风格化替代
- **步骤 2 确认必需** — 不要跳过
- **步骤 4/6 条件性** — 仅在步骤 2 中用户请求时
- **步骤 7.1 角色表** — 推荐用于多页漫画
- **清除密钥** — 在写入任何输出文件之前扫描源内容
