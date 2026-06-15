---
sidebar_position: 8
title: "代码执行"
description: "通过 RPC 工具访问实现程序化 Python 执行——将多步骤工作流压缩至单次对话轮次"
---

# 代码执行（程序化工具调用）

`execute_code` 工具允许 agent 编写调用 Hermes 工具的 Python 脚本，将多步骤工作流压缩至单次 LLM 对话轮次。脚本在 agent 宿主机的子进程中运行，通过 Unix 域套接字 RPC 与 Hermes 通信。

## 工作原理

1. Agent 编写使用 `from hermes_tools import ...` 的 Python 脚本
2. Hermes 生成带有 RPC 函数的 `hermes_tools.py` 存根模块
3. Hermes 打开 Unix 域套接字并启动 RPC 监听线程
4. 脚本在子进程中运行——工具调用通过套接字传回 Hermes
5. 只有脚本的 `print()` 输出会返回给 LLM；中间工具结果不会进入上下文窗口

```python
# The agent can write scripts like:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... filter and process ...
print(summary)
```

**脚本内可用工具：** `web_search`、`web_extract`、`read_file`、`write_file`、`search_files`、`patch`、`terminal`（仅前台模式）。

## Agent 何时使用此功能

当存在以下情况时，agent 会使用 `execute_code`：

- **3 次及以上工具调用**，且调用之间包含处理逻辑
- 批量数据过滤或条件分支
- 对结果进行循环处理

核心优势：中间工具结果不会进入上下文窗口——只有最终的 `print()` 输出会返回，大幅降低 token 用量。

## 实际示例

### 数据处理流水线

```python
from hermes_tools import search_files, read_file
import json

# Find all config files and extract database settings
matches = search_files("database", path=".", file_glob="*.yaml", limit=20)
configs = []
for match in matches.get("matches", []):
    content = read_file(match["path"])
    configs.append({"file": match["path"], "preview": content["content"][:200]})

print(json.dumps(configs, indent=2))
```

### 多步骤网络调研

```python
from hermes_tools import web_search, web_extract
import json

# Search, extract, and summarize in one turn
results = web_search("Rust async runtime comparison 2025", limit=5)
summaries = []
for r in results["data"]["web"]:
    page = web_extract([r["url"]])
    for p in page.get("results", []):
        if p.get("content"):
            summaries.append({
                "title": r["title"],
                "url": r["url"],
                "excerpt": p["content"][:500]
            })

print(json.dumps(summaries, indent=2))
```

### 批量文件重构

```python
from hermes_tools import search_files, read_file, patch

# Find all Python files using deprecated API and fix them
matches = search_files("old_api_call", path="src/", file_glob="*.py")
fixed = 0
for match in matches.get("matches", []):
    result = patch(
        path=match["path"],
        old_string="old_api_call(",
        new_string="new_api_call(",
        replace_all=True
    )
    if "error" not in str(result):
        fixed += 1

print(f"Fixed {fixed} files out of {len(matches.get('matches', []))} matches")
```

### 构建与测试流水线

```python
from hermes_tools import terminal, read_file
import json

# Run tests, parse results, and report
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# Parse test output
passed = output.count(" passed")
failed = output.count(" failed")
errors = output.count(" error")

report = {
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "exit_code": result.get("exit_code", -1),
    "summary": output[-500:] if len(output) > 500 else output
}

print(json.dumps(report, indent=2))
```

## 执行模式

`execute_code` 有两种执行模式，通过 `~/.hermes/config.yaml` 中的 `code_execution.mode` 控制：

| 模式 | 工作目录 | Python 解释器 |
|------|----------|---------------|
| **`project`**（默认） | 会话的工作目录（与 `terminal()` 相同） | 活跃的 `VIRTUAL_ENV` / `CONDA_PREFIX` python，回退至 Hermes 自身的 python |
| `strict` | 与用户项目隔离的临时暂存目录 | `sys.executable`（Hermes 自身的 python） |

**何时保持 `project` 模式：** 当你希望 `import pandas`、`from my_project import foo` 或 `open(".env")` 等相对路径与 `terminal()` 中的行为一致时。这几乎是你始终想要的模式。

**何时切换至 `strict` 模式：** 当你需要最大可复现性时——希望无论用户激活哪个 venv，每次会话都使用相同的解释器，并且希望脚本与项目目录隔离（避免通过相对路径意外读取项目文件）。

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # or "strict"
```

`project` 模式的回退行为：若 `VIRTUAL_ENV` / `CONDA_PREFIX` 未设置、已损坏或指向低于 3.8 的 Python，解析器会干净地回退至 `sys.executable`——agent 始终有可用的解释器。

两种模式的安全关键不变量完全相同：

- 环境变量清理（API key、token、凭据默认被剥离）
- 工具白名单（脚本不能递归调用 `execute_code`、`delegate_task` 或 MCP 工具）
- 资源限制（超时、stdout 上限、工具调用上限）

切换模式只改变脚本的运行位置和使用的解释器，不改变脚本可见的凭据或可调用的工具。

## 资源限制

| 资源 | 限制 | 说明 |
|------|------|------|
| **超时** | 5 分钟（300 秒） | 脚本先收到 SIGTERM，5 秒宽限期后收到 SIGKILL |
| **Stdout** | 50 KB | 输出截断并附加 `[output truncated at 50KB]` 提示 |
| **Stderr** | 10 KB | 非零退出时包含在输出中，用于调试 |
| **工具调用** | 每次执行 50 次 | 达到上限时返回错误 |

所有限制均可通过 `config.yaml` 配置：

```yaml
# In ~/.hermes/config.yaml
code_execution:
  mode: project      # project (default) | strict
  timeout: 300       # Max seconds per script (default: 300)
  max_tool_calls: 50 # Max tool calls per execution (default: 50)
```

## 脚本内工具调用的工作方式

当脚本调用 `web_search("query")` 等函数时：

1. 调用被序列化为 JSON，通过 Unix 域套接字发送至父进程
2. 父进程通过标准 `handle_function_call` 处理器进行分发
3. 结果通过套接字发回
4. 函数返回解析后的结果

这意味着脚本内的工具调用与普通工具调用行为完全一致——相同的速率限制、相同的错误处理、相同的能力。唯一的限制是 `terminal()` 仅支持前台模式（不支持 `background` 或 `pty` 参数）。

## 错误处理

脚本失败时，agent 会收到结构化的错误信息：

- **非零退出码**：stderr 包含在输出中，agent 可看到完整的 traceback
- **超时**：脚本被终止，agent 看到 `"Script timed out after 300s and was killed."`
- **中断**：若用户在执行期间发送新消息，脚本被终止，agent 看到 `[execution interrupted — user sent a new message]`
- **工具调用上限**：达到 50 次调用上限后，后续工具调用返回错误消息

响应始终包含 `status`（success/error/timeout/interrupted）、`output`、`tool_calls_made` 和 `duration_seconds`。

## 安全性

:::danger 安全模型
子进程在**最小化环境**中运行。API key、token 和凭据默认被剥离。脚本只能通过 RPC 通道访问工具——除非显式允许，否则无法从环境变量中读取密钥。
:::

名称中包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD` 或 `AUTH` 的环境变量会被排除。只有安全的系统变量（`PATH`、`HOME`、`LANG`、`SHELL`、`PYTHONPATH`、`VIRTUAL_ENV` 等）会被传递。

### Skill 环境变量透传

当 skill 在其 frontmatter 中声明 `required_environment_variables` 时，这些变量会在 skill 加载后**自动透传**至 `execute_code` 和 `terminal` 子进程。这使 skill 可以使用其声明的 API key，而不会削弱任意代码的安全态势。

对于非 skill 场景，可在 `config.yaml` 中显式添加变量白名单：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

详情参见[安全指南](/user-guide/security#environment-variable-passthrough)。

Hermes 始终将脚本和自动生成的 `hermes_tools.py` RPC 存根写入临时暂存目录，执行完成后清理。在 `strict` 模式下，脚本也在该目录中*运行*；在 `project` 模式下，脚本在会话的工作目录中运行（暂存目录保留在 `PYTHONPATH` 中以确保导入正常解析）。子进程在独立的进程组中运行，以便在超时或中断时干净地终止。

## execute_code 与 terminal 对比

| 使用场景 | execute_code | terminal |
|----------|-------------|----------|
| 调用之间含逻辑的多步骤工作流 | ✅ | ❌ |
| 简单 shell 命令 | ❌ | ✅ |
| 过滤/处理大量工具输出 | ✅ | ❌ |
| 运行构建或测试套件 | ❌ | ✅ |
| 对搜索结果进行循环处理 | ✅ | ❌ |
| 交互式/后台进程 | ❌ | ✅ |
| 需要环境变量中的 API key | ⚠️ 仅通过[透传](/user-guide/security#environment-variable-passthrough) | ✅（大多数可透传） |

**经验法则：** 需要在调用之间含逻辑地程序化调用 Hermes 工具时，使用 `execute_code`。运行 shell 命令、构建和进程时，使用 `terminal`。

## 平台支持

代码执行依赖 Unix 域套接字，仅在 **Linux 和 macOS** 上可用。在 Windows 上会自动禁用——agent 回退至常规的顺序工具调用。