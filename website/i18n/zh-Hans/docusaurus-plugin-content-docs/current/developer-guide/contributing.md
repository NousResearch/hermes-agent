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

- 构建自定义/本地工具而不修改 Hermes 核心？从 [构建 Hermes 插件](../developer-guide/plugins/index.md) 开始
- 为 Hermes 本身构建新的内置核心工具？从 [添加工具](./adding-tools.md) 开始
- 构建新的 skill？从 [创建 Skill](./creating-skills.md) 开始
- 构建新的推理提供商？从 [添加提供商](./adding-providers.md) 开始

## 开发环境配置 {#development-setup}

### 前置要求

项目要求 Python 3.14（`>=3.14,<3.15`）。PM 提供固定版本的解释器和工具。
准备 Git 和 git-lfs。JS 构建使用 PM 的 Node/npm，或满足相应 `package.json` engines 的版本。

### PM 开发环境

[PM 开发工作流](/reference/package-management#developer-workflow) 包含首次准备、激活、日常使用、依赖更新和当前 bootstrap 限制。
请在准备环境前选择独立的开发 `HERMES_HOME`，避免实验代码迁移生产数据。

成功准备后，每次在仓库根目录的新 shell 中激活已有环境。

Bash：

```bash
source ./activate
python hermes --version
```

PowerShell：

```powershell
. .\activate.ps1
python hermes --version
```

PowerShell 开头的点和空格用于 dot-source，不能省略。
激活通过 PM 准备工具并同步依赖，但不创建 JS workspaces，也不设置常规 venv 提示符。
使用 `python hermes` 明确运行当前 checkout，避免命中全局命令或 MSIX 别名。
`deactivate` 恢复激活前的环境，不卸载依赖或停止已启动的进程。

### 独立开发和测试环境 {#manual-development-and-test-environment}

先按 [PM 开发工作流](/reference/package-management#developer-workflow) 准备 Python 3.14。
在该 checkout 中使用准备好的 Python，并保持相同的开发 `HERMES_HOME`。
PM 必须能够启动，才能构建独立测试环境：

```bash
python -m pm.build_env --source . --out .venv --extra dev --group test
```

此命令使用提交的锁文件，创建新环境并检查依赖一致性。输出路径必须不存在。
如需重新生成，请先停止使用该环境的进程，再明确删除该可丢弃的环境。
PM 不会自动删除已有目录。不要通过原始 pip 或 uv 命令修改 Hermes 环境。

测试 runner 自动发现仓库的 `.venv`。它会清除 `PYTHONPATH`，因此 pytest 必须安装在解释器自身的环境中。
也可将 `--out` 指向仓库外的新路径，再将 `HERMES_PYTHON` 设为该环境的解释器。
Windows 上通过 Bash 运行 `scripts/run_tests.sh`，并预先准备本机 C++ 编译环境。

独立测试环境不替代 PM 工具存储或应用的依赖选择。不要修改签名应用的载荷。
运行开发实例前，选择临时的 `HERMES_HOME`，再使用 `python hermes setup` 配置它。
不要把生产凭据复制到 checkout。

从仓库根目录运行 `npm ci` 安装 JS workspaces。网站单独使用：

```bash
npm ci --prefix website
npm run build:fast --prefix website
```

图标从 `assets/nous-girl-*.svg` 和 `assets/backgrounds/` 生成。
`node scripts/generate-icons.mjs` 使用隔离的 `icon-build` 依赖组，不应将这些构建依赖加入生产包。

### 运行测试

```bash
scripts/run_tests.sh
```

该脚本清除凭据环境、设置 UTC 和临时 `HERMES_HOME`，并使用独立子进程运行各测试文件。
不同文件可并行，单个文件内的测试串行执行。不要绕过脚本直接运行 pytest。

## 代码风格

- **PEP 8**，允许合理例外（不强制限制行长度）
- **注释**：仅在解释非显而易见的意图、权衡取舍或 API 特殊行为时添加
- **错误处理**：捕获具体异常。对于意外错误，使用 `logger.warning()`/`logger.error()` 并设置 `exc_info=True`
- **跨平台**：不得假设 Unix 环境（见下文）
- **Profile 安全路径**：不得硬编码 `~/.hermes` — 代码路径使用 `hermes_constants` 中的 `get_hermes_home()`，面向用户的消息使用 `display_hermes_home()`。完整规则参见 [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#profiles-multi-instance-support)。

## 跨平台兼容性

Hermes 支持 Linux、macOS、WSL2 和原生 Windows。Windows shell 由 PM 解析 Git Bash。Dashboard 聊天通过 pywinpty/ConPTY 支持原生 Windows，并非仅限 WSL2。平台和依赖限制见[平台支持](/getting-started/platform-support)。

贡献代码时，请遵守以下规则：

- **不得添加未加保护的 `signal.SIGKILL` 引用。** Windows 上未定义该信号。请通过 `gateway.status.terminate_pid(pid, force=True)`（集中式原语，Windows 上执行 `taskkill /T /F`，POSIX 上发送 SIGKILL）路由，或使用 `getattr(signal, "SIGKILL", signal.SIGTERM)` 回退。
- **不要在 Windows 上用 `os.kill(pid, 0)` 检查存活。** 使用 `psutil.pid_exists()`；信号调用不是安全的只读检查。
- **不得强制终端使用 POSIX 语义。** `os.setsid`、`os.killpg`、`os.getpgid`、`os.fork` 在 Windows 上均会抛出异常 — 使用 `if sys.platform != "win32":` 或 `if os.name != "nt":` 进行条件判断。
- **打开文件时显式指定 `encoding="utf-8"`。** Windows 上 Python 默认使用系统区域设置（通常为 cp1252），处理非拉丁字符时会出现乱码或崩溃。
- **使用 `pathlib.Path` / `os.path.join`，不得手动用 `/` 拼接路径。** 这对我们构造后传给子进程的字符串尤为重要，而非 OS 返回给我们的字符串。

关键模式：
### 1. 文件编码

某些环境可能以非 UTF-8 编码保存 `.env` 文件：

```python
try:
    load_dotenv(env_path)
except UnicodeDecodeError:
    load_dotenv(env_path, encoding="latin-1")
```

### 2. 进程管理

`os.setsid()`、`os.killpg()` 以及信号处理在各平台间存在差异：

```python
import platform
if platform.system() != "Windows":
    kwargs["preexec_fn"] = os.setsid
```

### 3. 路径分隔符

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

1. **运行测试**：`scripts/run_tests.sh` 以确保 CI 一致性。仅当 wrapper 不可用或您有意在 wrapper 之外调试时，才使用直接 `python -m pytest ...`。
2. **手动测试**：运行 `hermes` 并验证您修改的代码路径
3. **检查跨平台影响**：考虑 macOS、Linux、WSL2 和原生 Windows。如果您修改了文件 I/O、进程管理、终端处理、子进程或信号相关代码，请运行 `scripts/check-windows-footguns.py`。
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
- 请包含：操作系统、Python 版本、Hermes 版本（`hermes --version`）、完整错误堆栈
- 包含复现步骤
- 创建前请检查是否已有重复 issue
- 安全漏洞请私下报告

## 社区

- **Discord**：[discord.gg/NousResearch](https://discord.gg/NousResearch)
- **GitHub Discussions**：用于设计提案和架构讨论
- **Skills Hub**：上传专业 skill 并与社区共享

## 许可证

提交贡献即表示您同意您的贡献将以 [MIT 许可证](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE) 授权。