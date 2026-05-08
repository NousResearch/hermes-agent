---
sidebar_position: 4
title: "贡献"
description: "如何为 Hermes Agent 做贡献 — 开发设置、代码风格、PR 流程"
---

# 贡献

感谢您为 Hermes Agent 做贡献！本指南涵盖设置开发环境、理解代码库以及让您的 PR 合并。

## 贡献优先级

我们按此顺序重视贡献：

1. **Bug 修复** — 崩溃、错误行为、数据丢失
2. **跨平台兼容性** — macOS、不同 Linux 发行版、WSL2
3. **安全加固** — shell 注入、提示词注入、路径遍历
4. **性能和健壮性** — 重试逻辑、错误处理、优雅降级
5. **新技能** — 广泛有用的技能（参见[创建技能](creating-skills.md)）
6. **新工具** — 很少需要；大多数功能应该是技能
7. **文档** — 修复、澄清、新示例

## 常见贡献路径

- 想在不修改 Hermes 核心的情况下构建自定义/本地工具？从[构建 Hermes 插件](../guides/build-a-hermes-plugin.md)开始
- 想为 Hermes 本身构建新的内置核心工具？从[添加工具](./adding-tools.md)开始
- 想构建新技能？从[创建技能](./creating-skills.md)开始
- 想构建新的推理 provider？从[添加 Provider](./adding-providers.md)开始

## 开发设置

### 前置条件

| 要求 | 备注 |
|-------------|-------|
| **Git** | 支持 `--recurse-submodules`，并已安装 `git-lfs` 扩展 |
| **Python 3.11+** | uv 会在缺失时安装 |
| **uv** | 快速 Python 包管理器（[安装](https://docs.astral.sh/uv/)） |
| **Node.js 20+** | 可选 — 浏览器工具和 WhatsApp 桥接需要（匹配根 `package.json` engines） |

### 克隆和安装

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# Install with all extras (messaging, cron, CLI menus, dev tools)
uv pip install -e ".[all,dev]"
uv pip install -e "./tinker-atropos"

# Optional: browser tools
npm install
```

### 为开发配置

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# Add at minimum an LLM provider key:
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' >> ~/.hermes/.env
```

### 运行

```bash
# Symlink for global access
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# Verify
hermes doctor
hermes chat -q "Hello"
```

### 运行测试

```bash
pytest tests/ -v
```

## 代码风格

- **PEP 8**，有实际例外（不严格限制行长度）
- **注释**：仅在解释非显而易见的意图、权衡或 API 怪癖时
- **错误处理**：捕获特定异常。对意外错误使用 `logger.warning()`/`logger.error()` 加上 `exc_info=True`
- **跨平台**：绝不假设 Unix（见下文）
- **配置文件安全路径**：绝不硬编码 `~/.hermes` — 对代码路径使用 `hermes_constants` 中的 `get_hermes_home()`，对用户面向消息使用 `display_hermes_home()`。参见 [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#profiles-multi-instance-support) 获取完整规则。

## 跨平台兼容性

Hermes 正式支持 Linux、macOS 和 WSL2。原生日 Windows **不支持**，但代码库包含一些防御性编码模式以避免边缘情况下的硬崩溃。关键规则：

### 1. `termios` 和 `fcntl` 仅 Unix

始终捕获 `ImportError` 和 `NotImplementedError`：

```python
try:
    from simple_term_menu import TerminalMenu
    menu = TerminalMenu(options)
    idx = menu.show()
except (ImportError, NotImplementedError):
    # Fallback: numbered menu
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

`os.setsid()`、`os.killpg()` 和信号处理因平台而异：

```python
import platform
if platform.system() != "Windows":
    kwargs["preexec_fn"] = os.setsid
```

### 4. 路径分隔符

使用 `pathlib.Path` 而不是用 `/` 进行字符串连接。

## 安全考虑

Hermes 有终端访问权限。安全很重要。

### 现有保护

| 层 | 实现 |
|-------|---------------|
| **Sudo 密码管道** | 使用 `shlex.quote()` 防止 shell 注入 |
| **危险命令检测** | `tools/approval.py` 中的正则模式，带用户审批流程 |
| **Cron 提示词注入** | 扫描器阻止指令覆盖模式 |
| **写入拒绝列表** | 通过 `os.path.realpath()` 解析受保护路径以防止符号链接绕过 |
| **技能保护** | Hub 安装技能的安全扫描器 |
| **代码执行沙盒** | 子进程在剥离 API 密钥的情况下运行 |
| **容器加固** | Docker：丢弃所有能力，无特权提升，PID 限制 |

### 贡献安全敏感代码

- 在将用户输入插入 shell 命令时始终使用 `shlex.quote()`
- 在访问控制检查之前使用 `os.path.realpath()` 解析符号链接
- 不记录 secrets
- 在工具执行周围捕获广泛异常
- 如果更改涉及文件路径或进程，在所有平台上测试

## Pull Request 流程

### 分支命名

```
fix/description        # Bug fixes
feat/description       # New features
docs/description       # Documentation
test/description       # Tests
refactor/description   # Code restructuring
```

### 提交前

1. **运行测试**：`pytest tests/ -v`
2. **手动测试**：运行 `hermes` 并测试您更改的代码路径
3. **检查跨平台影响**：考虑 macOS 和不同 Linux 发行版
4. **保持 PR 聚焦**：每个 PR 一个逻辑更改

### PR 描述

包括：
- **什么**改变了以及**为什么**
- **如何**测试它
- **在哪些平台**上测试
- 引用任何相关问题

### 提交消息

我们使用 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <description>
```

| 类型 | 用于 |
|------|-------|
| `fix` | Bug 修复 |
| `feat` | 新功能 |
| `docs` | 文档 |
| `test` | 测试 |
| `refactor` | 代码重构 |
| `chore` | 构建、CI、依赖更新 |

作用域：`cli`、`gateway`、`tools`、`skills`、`agent`、`install`、`whatsapp`、`security`

示例：
```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
```

## 报告问题

- 使用 [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- 包括：OS、Python 版本、Hermes 版本（`hermes version`）、完整错误回溯
- 包括重现步骤
- 创建前检查现有问题
- 对于安全漏洞，请私下报告

## 社区

- **Discord**：[discord.gg/NousResearch](https://discord.gg/NousResearch)
- **GitHub Discussions**：用于设计提案和架构讨论
- **技能中心**：上传专业技能并与社区分享

## 许可证

通过贡献，您同意您的贡献将根据 [MIT 许可证](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE) 获得许可。
