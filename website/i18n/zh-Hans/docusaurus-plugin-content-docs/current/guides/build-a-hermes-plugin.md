---
sidebar_position: 9
sidebar_label: "构建插件"
title: "构建 Hermes 插件"
description: "一步一步搭建一个完整的 Hermes 插件，包含工具、钩子、数据文件和 bundled skill"
---

# 构建 Hermes 插件

这篇指南会带你从零搭建一个完整的 Hermes 插件。最后你会得到一个包含多个工具、生命周期钩子、随插件分发的数据文件，以及 bundled skill 的插件——也就是插件系统支持的完整形态。

:::info 不确定该读哪篇指南？
Hermes 有几种不同的可插拔接口。有些走 Python 的 `register_*` API，有些是配置驱动，有些是直接放目录。先看这个映射：

| 如果你想添加… | 应该看 |
|---|---|
| 自定义工具、钩子、斜杠命令、skills 或 CLI 子命令 | **本指南** |
| 一个 **LLM / inference backend**（新 provider） | [Model Provider Plugins](/developer-guide/model-provider-plugin) |
| 一个 **gateway channel**（Discord/Telegram/IRC/Teams 等） | [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| 一个 **memory backend**（Honcho/Mem0/Supermemory 等） | [Memory Provider Plugins](/developer-guide/memory-provider-plugin) |
| 一个 **context-compression engine** | [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| 一个 **image-generation backend** | [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| 一个 **TTS backend**（任意 CLI） | [TTS custom command providers](/user-guide/features/tts#custom-command-providers) |
| 一个 **STT backend**（自定义 whisper / ASR CLI） | [Voice Message Transcription](/user-guide/features/tts#voice-message-transcription-stt) |
| 通过 MCP 暴露的 **外部工具** | [MCP](/user-guide/features/mcp) |
| **Gateway event hooks** | [Event Hooks](/user-guide/features/hooks#gateway-event-hooks) |
| **Shell hooks** | [Shell Hooks](/user-guide/features/hooks#shell-hooks) |
| **额外 skill 来源** | [Skills](/user-guide/features/skills) |
| 第一类 **core** inference provider（不是插件） | [Adding Providers](/developer-guide/adding-providers) |
:::

## 你要构建什么

一个 **calculator** 插件，带两个工具：
- `calculate` - 计算表达式
- `unit_convert` - 单位转换

再加一个记录每次工具调用的 hook，和一个 bundled skill 文件。

## 第 1 步：创建插件目录

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

## 第 2 步：编写 manifest

创建 `plugin.yaml`：

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides_tools:
  - calculate
  - unit_convert
provides_hooks:
  - post_tool_call
```

## 第 3 步：编写工具 schema

创建 `schemas.py`，给 LLM 看工具描述。

## 第 4 步：编写工具 handler

创建 `tools.py`，这里是真正执行逻辑的地方。

## 第 5 步：写注册逻辑

创建 `__init__.py`，把 schema 和 handler 接起来。

## 第 6 步：测试

启动 Hermes 后，你应该能在 banner 里看到 `calculator: calculate, unit_convert`。

## 插件还可以做什么？

### 发送数据文件

插件目录里的任何文件都可以在导入时读取。

### 打包技能 {#bundle-skills}

插件可以带自己的 skill 文件，由 agent 通过 `skill_view("plugin:skill")` 加载。

```text
~/.hermes/plugins/my-plugin/
├── __init__.py
├── plugin.yaml
└── skills/
    ├── my-workflow/
    │   └── SKILL.md
    └── my-checklist/
        └── SKILL.md
```

```python
def register(ctx):
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
```

### 按环境变量门控 {#gate-on-environment-variables}

如果插件需要 API key，可以在 `requires_env` 里声明。

### 条件式工具可用性

依赖可选库的工具可以在缺失时直接隐藏。

### 注册多个 hooks

你可以同时注册多个生命周期钩子。

### Hook 参考

完整 Hook 说明见 [事件钩子参考](/user-guide/features/hooks#plugin-hooks)。
