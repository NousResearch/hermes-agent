---
sidebar_position: 16
title: "LSP — 语义诊断"
description: "真实语言服务器（pyright、gopls、rust-analyzer 等）接入 write_file 和 patch 所使用的写后 lint 检查。"
---

# 语言服务器协议（LSP）

Hermes 以后台子进程方式运行完整的语言服务器——pyright、gopls、rust-analyzer、
typescript-language-server、clangd 以及约 20 个其他服务器——并将其语义诊断结果
接入 `write_file` 和 `patch` 所使用的写后 lint 检查。当 agent 编辑文件时，
它能精确看到该次编辑引入的错误——不仅是语法错误，还包括语言服务器检测到的
**类型错误、未定义名称、缺失导入以及全项目范围的语义问题**。

这与顶级编码 agent 所采用的架构相同。Hermes 将其作为自包含组件提供：
无需编辑器宿主，无需安装插件，无需管理独立守护进程。

## LSP 的触发时机

LSP 以 **git 工作区检测**为前提条件。当 agent 的工作目录（或正在编辑的文件）
位于 git 仓库内时，LSP 针对该工作区运行。若两者均不在 git 仓库中，LSP 保持
休眠——这对消息网关（gateway）场景很有用，此时 cwd 为用户主目录，没有可诊断的项目。

检查分层进行：首先进行进程内语法检查（微秒级），语法通过后再进行 LSP 语义诊断。
不稳定或缺失的语言服务器永远不会导致写入失败——所有 LSP 失败路径均静默回退至
仅语法检查的结果。

具体而言，每次成功执行 `write_file` 或 `patch` 时：

1. Hermes 捕获该文件当前诊断的基线快照。
2. 执行写入。
3. 重新查询语言服务器，过滤掉基线中已存在的诊断，仅呈现新引入的诊断。

agent 看到的输出如下：

```
{
  "bytes_written": 42,
  "dirs_created": false,
  "lint": {"status": "ok", "output": ""},
  "lsp_diagnostics": "LSP diagnostics introduced by this edit:\n<diagnostics file=\"/path/to/foo.py\">\nERROR [42:5] Cannot find name 'foo' [reportUndefinedVariable] (Pyright)\nERROR [50:1] Argument of type \"str\" is not assignable to \"int\" [reportArgumentType] (Pyright)\n</diagnostics>"
}
```

`lint` 字段承载语法检查结果（通过 `ast.parse`、`json.loads` 等进行微秒级进程内解析）；
`lsp_diagnostics` 字段承载来自真实语言服务器的语义诊断。两个通道，独立信号——
agent 对于语法正确但存在语义问题的文件，会看到 ``lint: ok`` 加上已填充的 ``lsp_diagnostics``。

## 支持的语言

| 语言 | 服务器 | 自动安装 |
|----------|--------|--------------|
| Python | `pyright-langserver` | npm |
| TypeScript / JavaScript / JSX / TSX | `typescript-language-server` | npm |
| Vue | `@vue/language-server` | npm |
| Svelte | `svelte-language-server` | npm |
| Astro | `@astrojs/language-server` | npm |
| Go | `gopls` | `go install` |
| Rust | `rust-analyzer` | 手动（rustup） |
| C / C++ | `clangd` | 手动（LLVM） |
| Bash / Zsh | `bash-language-server` | npm |
| YAML | `yaml-language-server` | npm |
| Lua | `lua-language-server` | 手动（GitHub releases） |
| PHP | `intelephense` | npm |
| OCaml | `ocaml-lsp` | 手动（opam） |
| Dockerfile | `dockerfile-language-server-nodejs` | npm |
| Terraform | `terraform-ls` | 手动 |
| Dart | `dart language-server` | 手动（dart sdk） |
| Haskell | `haskell-language-server` | 手动（ghcup） |
| Julia | `julia` + LanguageServer.jl | 手动 |
| Clojure | `clojure-lsp` | 手动 |
| Nix | `nixd` | 手动 |
| Zig | `zls` | 手动 |
| Gleam | `gleam lsp` | 手动（gleam install） |
| Elixir | `elixir-ls` | 手动 |
| Prisma | `prisma language-server` | 手动 |
| Kotlin | `kotlin-language-server` | 手动 |
| Java | `jdtls` | 手动 |

对于"手动"条目，请通过该语言对应的工具链管理器安装服务器（rustup、ghcup、opam、brew 等）。
Hermes 会自动检测 PATH 上或 `<HERMES_HOME>/lsp/bin/` 中的二进制文件。

部分服务器需要与 npm 不会自动拉取的对等依赖一同安装。当前的典型情况是
`typescript-language-server`，它要求 `typescript` SDK 可从同一 `node_modules`
目录树中导入——当你运行 `hermes lsp install typescript` 或首次使用时触发自动安装时，
Hermes 会同时安装这两个包。

## CLI

```
hermes lsp status          # 服务状态 + 各服务器安装状态
hermes lsp list            # 注册表，可选 --installed-only
hermes lsp install <id>    # 主动安装单个服务器
hermes lsp install-all     # 尝试安装所有已知安装方式的服务器
hermes lsp restart         # 关闭正在运行的客户端
hermes lsp which <id>      # 打印解析后的二进制路径
```

`hermes lsp status` 是最佳起点——它显示哪些语言当前可获得语义诊断，
哪些语言还需要安装二进制文件。

## 配置

默认配置适用于典型场景；若二进制文件已在 PATH 上，无需任何设置。

```yaml
# config.yaml
lsp:
  # 主开关。禁用后跳过整个子系统——不会启动任何服务器，不会运行后台事件循环。
  enabled: true

  # 每次写入后等待诊断结果的方式。
  wait_mode: document      # "document" 或 "full"
  wait_timeout: 5.0

  # 处理缺失服务器二进制文件的策略。
  #   auto    — 通过 npm/pip/go install 安装到 <HERMES_HOME>/lsp/bin
  #   manual  — 仅使用已在 PATH 上的二进制文件
  install_strategy: auto

  # 各服务器覆盖配置（均为可选）。
  servers:
    pyright:
      disabled: false
      command: ["/abs/path/to/pyright-langserver", "--stdio"]
      env: { PYRIGHT_LOG_LEVEL: "info" }
      initialization_options:
        python:
          analysis:
            typeCheckingMode: "strict"
    typescript:
      disabled: true       # 即使扩展名匹配也跳过 TS
```

### 各服务器配置键

* `disabled: true` — 即使扩展名与文件匹配，也完全跳过该服务器。
* `command: [bin, ...args]` — 指定自定义二进制路径，绕过自动安装。
* `env: {KEY: value}` — 传递给启动进程的额外环境变量。
* `initialization_options: {...}` — 合并到 LSP `initialize` 握手时发送的
  `initializationOptions` 载荷中。具体内容因服务器而异，请参阅对应语言服务器的文档。

## 安装位置

当 `install_strategy: auto` 时，Hermes 将二进制文件安装到 `<HERMES_HOME>/lsp/bin/`。
NPM 包安装到 `<HERMES_HOME>/lsp/node_modules/`，bin 符号链接位于上一级目录。
Go 二进制文件通过 `go install` 安装，`GOBIN` 指向暂存目录。

任何内容都不会安装到 `/usr/local/`、`~/.local/` 或其他共享位置——暂存目录完全由
Hermes 管理，重置 profile 时会被删除。

## 性能特性

LSP 服务器在**首次使用时懒启动**。在从未处理过 `.py` 文件的项目中编辑 Python 文件
会启动 pyright；大多数服务器的启动耗时为 1-3 秒（rust-analyzer 在冷启动项目时可能
超过 10 秒）。同一工作区内的后续编辑会复用已运行的服务器。

在没有诊断结果输出时，LSP 层对干净写入仅增加数毫秒延迟。有诊断结果时，等待预算为
`wait_timeout` 秒——pyright/tsserver 通常在数十毫秒内响应，rust-analyzer 在索引
过程中可能需要数秒。

服务器在 Hermes 进程的整个生命周期内保持运行。没有空闲超时回收机制——每次写入都
重启服务器索引的代价远高于保持守护进程运行。

## 禁用

在 `config.yaml` 中设置 `lsp.enabled: false` 可禁用整个子系统。写后检查将回退至
进程内语法检查（Python 使用 `ast.parse`，JSON 使用 `json.loads` 等），与早期版本
保持一致。

若要禁用单个语言而不禁用整个层：

```yaml
lsp:
  servers:
    rust-analyzer:
      disabled: true
```

## 故障排查

**`hermes lsp status` 显示某服务器为"missing"**

该二进制文件不在 PATH 上，也不在 `<HERMES_HOME>/lsp/bin/` 中。运行
`hermes lsp install <server_id>` 尝试自动安装，或通过该语言的常规工具链手动安装。

**`hermes lsp status` 中出现 `Backend warnings` 部分**

部分服务器以薄包装层的形式调用外部 CLI 进行实际诊断——它们能正常启动并接受请求，
但在辅助二进制文件缺失时不会报错。最常见的情况是 `bash-language-server`，
它将诊断委托给 `shellcheck`。当 `hermes lsp status` 显示 `Backend warnings` 部分时，
请通过系统包管理器安装对应工具：

```
apt install shellcheck      # Debian / Ubuntu
brew install shellcheck     # macOS
scoop install shellcheck    # Windows
```

同样的警告会在服务器启动时记录一次到 `~/.hermes/logs/agent.log`。

**服务器已启动但从不返回诊断结果**

检查 `~/.hermes/logs/agent.log` 中的 `[agent.lsp.client]` 条目——语言服务器的
stderr 输出和协议错误均记录于此。部分服务器（尤其是 rust-analyzer）需要完成
全项目索引后才会输出单文件诊断；服务器启动后的第一次编辑可能没有诊断结果，
后续编辑才会获取到。

**服务器崩溃**

崩溃的服务器会被加入损坏集合，在本次会话剩余时间内不再重试。运行
`hermes lsp restart` 清除该集合；下次编辑时会重新启动。

**编辑位于任何 git 仓库之外的文件**

按设计，LSP 仅在 git 仓库内运行。若项目尚未初始化，运行 `git init` 以启用
LSP 诊断。否则将使用进程内仅语法检查的回退方案。