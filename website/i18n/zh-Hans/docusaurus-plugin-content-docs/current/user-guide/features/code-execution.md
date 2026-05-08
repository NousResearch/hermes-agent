---
sidebar_position: 8
title: "代码执行"
description: "通过 RPC 工具访问进行程序化 Python 执行——将多步骤工作流压缩为单个 LLM 调用"
---

# 代码执行（程序化工具调用）

`execute_code` 工具让 agent 编写可以程序化调用 Hermes 工具的 Python 脚本，将多步骤工作流压缩为单个 LLM 调用。脚本在 agent 主机上的子进程中运行，通过 Unix 域套接字 RPC 与 Hermes 通信。

## 工作原理

1. Agent 编写使用 `from hermes_tools import ...` 的 Python 脚本
2. Hermes 生成带有 RPC 函数的 `hermes_tools.py` 存根模块
3. Hermes 打开 Unix 域套接字并启动 RPC 监听线程
4. 脚本在子进程中运行——工具调用通过套接字返回 Hermes
5. 只有脚本的 `print()` 输出返回到 LLM；中间工具结果永远不会进入上下文窗口

```python
# Agent 可以编写如下脚本：
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... 过滤和处理 ...
print(summary)
```

**脚本中可用的工具：** `web_search`、`web_extract`、`read_file`、`write_file`、`search_files`、`patch`、`terminal`（仅前台）。

## Agent 何时使用此功能

Agent 在以下情况下使用 `execute_code`：

- **3+ 个工具调用**，中间有处理逻辑
- 批量数据过滤或条件分支
- 循环遍历结果

关键好处：中间工具结果永远不会进入上下文窗口——只有最终的 `print()` 输出返回，大大减少令牌使用。

## 实用示例

### 数据处理管道

```python
from hermes_tools import search_files, read_file
import json

# 查找所有配置文件并提取数据库设置
matches = search_files("database", path=".", file_glob="*.yaml", limit=20)
configs = []
for match in matches.get("matches", []):
    content = read_file(match["path"])
    configs.append({"file": match["path"], "preview": content["content"][:200]})

print(json.dumps(configs, indent=2))
```

### 多步骤网络研究

```python
from hermes_tools import web_search, web_extract
import json

# 在一个调用中搜索、提取和总结
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

# 查找所有使用已弃用 API 的 Python 文件并修复
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

print(f"修复了 {fixed} 个文件，共 {len(matches.get('matches', []))} 个匹配")
```

### 构建和测试管道

```python
from hermes_tools import terminal, read_file
import json

# 运行测试、解析结果并报告
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# 解析测试输出
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
|------|-------------------|--------------------|
| **`project`**（默认） | 会话的工作目录（与 `terminal()` 相同） | 活动的 `VIRTUAL_ENV` / `CONDA_PREFIX` python，回退到 Hermes 自己的 python |
| `strict` | 与用户项目隔离的临时目录 | `sys.executable`（Hermes 自己的 python） |

**何时保持 `project`：** 你希望 `import pandas`、`from my_project import foo` 或相对路径如 `open(".env")` 的工作方式与在 `terminal()` 中相同。这几乎总是你想要的。

**何时切换到 `strict`：** 你需要最大可重复性——无论用户激活了哪个 venv，你都希望每次会话都使用相同的解释器，并且你希望脚本与项目树隔离（没有通过相对路径意外读取项目文件的风险）。

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # 或 "strict"
```

`project` 模式中的回退行为：如果 `VIRTUAL_ENV` / `CONDA_PREFIX` 未设置、损坏或指向早于 3.8 的 Python，解析器会干净地回退到 `sys.executable`——它永远不会让 agent 没有可用的解释器。

两种模式的安全关键不变量相同：

- 环境清理（API 密钥、令牌、凭据被剥离）
- 工具白名单（脚本不能递归调用 `execute_code`、`delegate_task` 或 MCP 工具）
- 资源限制（超时、stdout 上限、工具调用上限）

切换模式会改变脚本运行的位置和运行它们的解释器，而不是它们可以查看哪些凭据或可以调用哪些工具。

## 资源限制

| 资源 | 限制 | 说明 |
|----------|-------|-------|
| **超时** | 5 分钟（300秒） | 脚本被 SIGTERM 杀死，然后 5 秒后 SIGKILL |
| **Stdout** | 50 KB | 输出被截断并带有 `[output truncated at 50KB]` 通知 |
| **Stderr** | 10 KB | 在非零退出时包含在输出中以便调试 |
| **工具调用** | 每次执行 50 次 | 达到限制时返回错误 |

所有限制都可以通过 `config.yaml` 配置：

```yaml
# 在 ~/.hermes/config.yaml 中
code_execution:
  mode: project      # project（默认）| strict
  timeout: 300       # 每个脚本的最大秒数（默认：300）
  max_tool_calls: 50 # 每次执行的最大工具调用数（默认：50）
```

## 脚本中工具调用如何工作

当你的脚本调用如 `web_search("query")` 的函数时：

1. 调用被序列化为 JSON 并通过 Unix 域套接字发送到父进程
2. 父进程通过标准的 `handle_function_call` 处理程序调度
3. 结果通过套接字发回
4. 函数返回解析后的结果

这意味着脚本中的工具调用与正常工具调用的行为完全相同——相同的速率限制、相同的错误处理、相同的功能。唯一的限制是 `terminal()` 仅限前台（无 `background` 或 `pty` 参数）。

## 错误处理

当脚本失败时，agent 收到结构化错误信息：

- **非零退出码**：stderr 包含在输出中，以便 agent 看到完整的回溯
- **超时**：脚本被杀死，agent 看到 `"Script timed out after 300s and was killed."`
- **中断**：如果用户在执行期间发送新消息，脚本被终止，agent 看到 `[execution interrupted — user sent a new message]`
- **工具调用限制**：当达到 50 次调用上限时，后续工具调用返回错误消息

响应始终包含 `status`（success/error/timeout/interrupted）、`output`、`tool_calls_made` 和 `duration_seconds`。

## 安全性

:::danger 安全模型
子进程使用**最小环境**运行。默认情况下，API 密钥、令牌和凭据会被剥离。脚本仅通过 RPC 通道访问工具——除非明确允许，否则无法从环境变量中读取密钥。
:::

包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD` 或 `AUTH` 的环境变量名称被排除。只有安全的系统变量（`PATH`、`HOME`、`LANG`、`SHELL`、`PYTHONPATH`、`VIRTUAL_ENV` 等）被传递。

### 技能环境变量传递

当技能在其 frontmatter 中声明 `required_environment_variables` 时，这些变量在技能加载后**自动传递**到 `execute_code` 和 `terminal` 子进程。这让技能使用其声明的 API 密钥，而不会削弱任意代码的安全态势。

对于非技能用例，你可以在 `config.yaml` 中明确允许列出变量：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

有关完整详情，请参阅[安全指南](/docs/user-guide/security#environment-variable-passthrough)。

Hermes 始终将脚本和自动生成的 `hermes_tools.py` RPC 存根写入在执行后清理的临时目录。在 `strict` 模式下，脚本也在那里*运行*；在 `project` 模式下，它在会话的工作目录中运行（临时目录保留在 `PYTHONPATH` 上，以便导入仍然解析）。子进程在自己的进程组中运行，因此可以在超时或中断时被干净地杀死。

## execute_code 与 terminal 对比

| 用例 | execute_code | terminal |
|----------|-------------|----------|
| 工具调用之间有多步骤工作流 | ✅ | ❌ |
| 简单 shell 命令 | ❌ | ✅ |
| 过滤/处理大型工具输出 | ✅ | ❌ |
| 运行构建或测试套件 | ❌ | ✅ |
| 循环遍历搜索结果 | ✅ | ❌ |
| 交互式/后台进程 | ❌ | ✅ |
| 需要环境中的 API 密钥 | ⚠️ 仅通过[传递](/docs/user-guide/security#environment-variable-passthrough) | ✅（大多数都会传递） |

**经验法则：** 当你需要程序化调用 Hermes 工具且调用之间有逻辑时使用 `execute_code`。当运行 shell 命令、构建和进程时使用 `terminal`。

## 平台支持

代码执行需要 Unix 域套接字，仅在**Linux 和 macOS**上可用。在 Windows 上自动禁用——agent 回退到常规顺序工具调用。
