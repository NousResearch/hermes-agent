---
sidebar_position: 4
title: "贡献指南"
description: "如何为 Hermes Agent 做贡献 — 开发环境配置、代码风格、PR 流程"
---

# 贡献指南

感谢您为 Hermes Agent 做贡献！本指南涵盖开发环境配置、代码库结构说明以及 PR 合并流程。

## 贡献优先级

我们按以下顺序评估贡献价值：

1. **Bug 修复** — 崩溃、错误行为、数据丢失
2. **跨平台兼容性** — macOS、不同 Linux 发行版、WSL2
3. **安全加固** — shell 注入、prompt（提示词）注入、路径穿越
4. **性能与健壮性** — 重试逻辑、错误处理、优雅降级
5. **新 skill** — 具有广泛用途的 skill（参见 [创建 Skill](creating-skills.md)）
6. **新工具** — 极少需要；大多数能力应以 skill 形式实现
7. **文档** — 修正、说明、新示例

## 常见贡献路径

- 构建自定义/本地工具而不修改 Hermes 核心？从 [构建 Hermes 插件](../guides/build-a-hermes-plugin.md) 开始
- 为 Hermes 本身构建新的内置核心工具？从 [添加工具](./adding-tools.md) 开始
- 构建新的 skill？从 [创建 Skill](./creating-skills.md) 开始
- 构建新的推理提供商？从 [添加提供商](./adding-providers.md) 开始

## 开发环境配置

### 前置要求

| 要求 | 说明 |
|-------------|-------|
| **Git** | 需支持 `--recurse-submodules`，并安装 `git-lfs` 扩展 |
| **Python 3.11+** | 若未安装，uv 会自动安装 |
| **uv** | 高速 Python 包管理器（[安装](https://docs.astral.sh/uv/)） |
| **Node.js 20+** | 可选 — 浏览器工具和 WhatsApp bridge 需要（与根目录 `package.json` engines 字段一致） |

### 克隆与安装

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# 使用 Python 3.11 创建虚拟环境
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# 安装所有扩展（messaging、cron、CLI 菜单、开发工具）
uv pip install -e ".[all,dev]"

# 可选：浏览器工具
npm install
```

### 配置开发环境

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# 至少添加一个 LLM 提供商密钥：
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' >> ~/.hermes/.env
```

### 运行

```bash
# 创建全局访问的符号链接
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# 验证
hermes doctor
hermes chat -q "Hello"
```

### 运行测试

```bash
pytest tests/ -v
```

## 代码风格

- **PEP 8**，允许合理例外（不强制限制行长度）
- **注释**：仅在解释非显而易见的意图、权衡取舍或 API 特殊行为时添加
- **错误处理**：捕获具体异常。对于意外错误，使用 `logger.warning()`/`logger.error()` 并设置 `exc_info=True`
- **跨平台**：不得假设 Unix 环境（见下文）
- **Profile 安全路径**：不得硬编码 `~/.hermes` — 代码路径使用 `hermes_constants` 中的 `get_hermes_home()`，面向用户的消息使用 `display_hermes_home()`。完整规则参见 [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#profiles-multi-instance-support)。

## 跨平台兼容性

Hermes 官方支持 **Linux、macOS、WSL2 以及原生 Windows（早期 beta — 通过 PowerShell 安装）**。原生 Windows 使用 [Git for Windows](https://git-scm.com/download/win) 提供的 Git Bash 执行 shell 命令。部分功能依赖 POSIX 内核原语，已做条件限制：dashboard 内嵌的 PTY 终端面板（`/chat` 标签页）仅支持 WSL2。原生 Windows 路径较新且迭代较快 — 如果您主要在 Windows 上开发，请做好遇到并修复粗糙边缘的准备。

贡献代码时，请遵守以下规则：

- **不得添加未加保护的 `signal.SIGKILL` 引用。** Windows 上未定义该信号。请通过 `gateway.status.terminate_pid(pid, force=True)`（集中式原语，Windows 上执行 `taskkill /T /F`，POSIX 上发送 SIGKILL）路由，或使用 `getattr(signal, "SIGKILL", signal.SIGTERM)` 回退。
- **在 `os.kill(pid, 0)` 探测时同时捕获 `OSError` 和 `ProcessLookupError`。** Windows 对已消失的 PID 抛出 `OSError`（WinError 87，"参数不正确"），而非 `ProcessLookupError`。
- **不得强制终端使用 POSIX 语义。** `os.setsid`、`os.killpg`、`os.getpgid`、`os.fork` 在 Windows 上均会抛出异常 — 使用 `if sys.platform != "win32":` 或 `if os.name != "nt":` 进行条件判断。
- **打开文件时显式指定 `encoding="utf-8"`。** Windows 上 Python 默认使用系统区域设置（通常为 cp1252），处理非拉丁字符时会出现乱码或崩溃。
- **使用 `pathlib.Path` / `os.path.join`，不得手动用 `/` 拼接路径。** 这对我们构造后传给子进程的字符串尤为重要，而非 OS 返回给我们的字符串。

关键模式：

### 1. `termios` 和 `fcntl` 仅适用于 Unix

始终同时捕获 `ImportError` 和 `NotImplementedError`：

```python
try:
    from simple_term_menu import TerminalMenu
    menu = TerminalMenu(options)
    idx = menu.show()
except (ImportError, NotImplementedError):
    # 回退：编号菜单
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    idx = int(input("Choice: ")) - 1
```

### 2. 文件编码

某些环境可能以非 UTF-8 编码保存 `.env` 文件：

```python
try:
    load_dotenv(env_path)
except UnicodeDecodeError:
    load_dotenv(env_path, encoding="latin-1")
```

### 3. 进程管理

`os.setsid()`、`os.killpg()` 以及信号处理在各平台间存在差异：

```python
import platform
if platform.system() != "Windows":
    kwargs["preexec_fn"] = os.setsid
```

### 4. 路径分隔符

使用 `pathlib.Path` 代替用 `/` 进行字符串拼接。

## 安全注意事项

Hermes 拥有终端访问权限，安全至关重要。

### 现有保护措施

| 层级 | 实现方式 |
|-------|---------------|
| **sudo 密码管道** | 使用 `shlex.quote()` 防止 shell 注入 |
| **危险命令检测** | `tools/approval.py` 中的正则表达式模式，配合用户审批流程 |
| **Cron prompt 注入** | 扫描器阻断指令覆盖模式 |
| **写入拒绝列表** | 受保护路径通过 `os.path.realpath()` 解析，防止符号链接绕过 |
| **Skill 守卫** | 对 hub 安装的 skill 进行安全扫描 |
| **代码执行沙箱** | 子进程运行时剥离 API 密钥 |
| **容器加固** | Docker：删除所有 capability，禁止权限提升，限制 PID 数量 |

### 贡献安全敏感代码

- 将用户输入插入 shell 命令时，始终使用 `shlex.quote()`
- 访问控制检查前，使用 `os.path.realpath()` 解析符号链接
- 不得记录密钥信息
- 在工具执行周围捕获宽泛异常
- 若您的变更涉及文件路径或进程，请在所有平台上测试

## Pull Request 流程

### 分支命名

```
fix/description        # Bug 修复
feat/description       # 新功能
docs/description       # 文档
test/description       # 测试
refactor/description   # 代码重构
```

### 提交前检查

1. **运行测试**：`pytest tests/ -v`
2. **手动测试**：运行 `hermes` 并验证您修改的代码路径
3. **检查跨平台影响**：考虑 macOS 和不同 Linux 发行版
4. **保持 PR 聚焦**：每个 PR 只包含一个逻辑变更

### PR 描述

请包含：
- **变更内容**及**变更原因**
- **测试方法**
- **测试平台**
- 关联 issue 引用

### Commit 消息

我们使用 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <description>
```

| 类型 | 适用场景 |
|------|---------|
| `fix` | Bug 修复 |
| `feat` | 新功能 |
| `docs` | 文档 |
| `test` | 测试 |
| `refactor` | 代码重构 |
| `chore` | 构建、CI、依赖更新 |

Scope 范围：`cli`、`gateway`、`tools`、`skills`、`agent`、`install`、`whatsapp`、`security`

示例：
```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
```

## 报告问题

- 使用 [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- 请包含：操作系统、Python 版本、Hermes 版本（`hermes version`）、完整错误堆栈
- 包含复现步骤
- 创建前请检查是否已有重复 issue
- 安全漏洞请私下报告

## 社区

- **Discord**：[discord.gg/NousResearch](https://discord.gg/NousResearch)
- **GitHub Discussions**：用于设计提案和架构讨论
- **Skills Hub**：上传专业 skill 并与社区共享

## 许可证

提交贡献即表示您同意您的贡献将以 [MIT 许可证](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE) 授权。