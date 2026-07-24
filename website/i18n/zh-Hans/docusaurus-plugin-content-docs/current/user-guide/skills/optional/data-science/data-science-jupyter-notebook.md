---
title: "Jupyter Notebook — 通过实时 Jupyter 内核进行迭代式 Python 开发（hamelnb）"
sidebar_label: "Jupyter Notebook"
description: "通过实时 Jupyter 内核进行迭代式 Python 开发（hamelnb）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Jupyter Notebook

通过实时 Jupyter 内核进行迭代式 Python 开发（hamelnb）。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/data-science/jupyter-notebook` 安装 |
| 路径 | `optional-skills/data-science/jupyter-notebook` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `jupyter`, `notebook`, `repl`, `data-science`, `exploration`, `iterative` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Jupyter Notebook（hamelnb 实时内核）

通过实时 Jupyter 内核为你提供一个**有状态的 Python REPL**（交互式解释器）。变量在多次执行之间持久保留。当你需要逐步构建状态、探索 API、检查 DataFrame 或迭代复杂代码时，请使用此工具而非 `execute_code`。

## 何时使用本 Skill 与其他工具

| 工具 | 使用场景 |
|------|----------|
| **本 skill** | 迭代式探索、跨步骤保持状态、数据科学、机器学习、"试试看再检查" |
| `execute_code` | 需要访问 Hermes 工具（web_search、文件操作）的一次性脚本。无状态。 |
| `terminal` | Shell 命令、构建、安装、git、进程管理 |

**经验法则：** 如果你会为某个任务打开 Jupyter notebook，就使用本 skill。

## 前置条件

1. 必须安装 **uv**（检查：`which uv`）
2. 必须安装 **JupyterLab**：`uv tool install jupyterlab`
3. 必须有一个正在运行的 Jupyter 服务器（参见下方"设置"部分）

## 设置

hamelnb 脚本位置：
```
SCRIPT="$HOME/.agent-skills/hamelnb/skills/jupyter-live-kernel/scripts/jupyter_live_kernel.py"
```

如果尚未克隆：
```
git clone https://github.com/hamelsmu/hamelnb.git ~/.agent-skills/hamelnb
```

### 启动 JupyterLab

检查是否已有服务器在运行：
```
uv run "$SCRIPT" servers
```

如果未找到服务器，启动一个：
```
jupyter-lab --no-browser --port=8888 --notebook-dir=$HOME/notebooks \
  --IdentityProvider.token='' --ServerApp.password='' > /tmp/jupyter.log 2>&1 &
sleep 3
```

注意：已禁用 token/password 以供本地 agent 访问。服务器以无头模式运行。

### 为 REPL 使用创建 Notebook

如果你只需要一个 REPL（无需现有 notebook），创建一个最小化的 notebook 文件：
```
mkdir -p ~/notebooks
```
写入一个包含一个空代码单元格的最小 .ipynb JSON 文件，然后通过 Jupyter REST API 启动一个内核会话：
```
curl -s -X POST http://127.0.0.1:8888/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"path":"scratch.ipynb","type":"notebook","name":"scratch.ipynb","kernel":{"name":"python3"}}'
```

## 核心工作流

所有命令均返回结构化 JSON。始终使用 `--compact` 以节省 token。

### 1. 发现服务器和 notebook

```
uv run "$SCRIPT" servers --compact
uv run "$SCRIPT" notebooks --compact
```

### 2. 执行代码（主要操作）

```
uv run "$SCRIPT" execute --path <notebook.ipynb> --code '<python code>' --compact
```

状态在多次 execute 调用之间持久保留。变量、导入、对象均会保留。

多行代码可使用 `$'...'` 引号语法：
```
uv run "$SCRIPT" execute --path scratch.ipynb --code $'import os\nfiles = os.listdir(".")\nprint(f"Found {len(files)} files")' --compact
```

### 3. 检查实时变量

```
uv run "$SCRIPT" variables --path <notebook.ipynb> list --compact
uv run "$SCRIPT" variables --path <notebook.ipynb> preview --name <varname> --compact
```

### 4. 编辑 notebook 单元格

```
# 查看当前单元格
uv run "$SCRIPT" contents --path <notebook.ipynb> --compact

# 插入新单元格
uv run "$SCRIPT" edit --path <notebook.ipynb> insert \
  --at-index <N> --cell-type code --source '<code>' --compact

# 替换单元格源码（使用 contents 输出中的 cell-id）
uv run "$SCRIPT" edit --path <notebook.ipynb> replace-source \
  --cell-id <id> --source '<new code>' --compact

# 删除单元格
uv run "$SCRIPT" edit --path <notebook.ipynb> delete --cell-id <id> --compact
```

### 5. 验证（重启并全部运行）

仅在用户要求进行干净验证，或你需要确认 notebook 能从头到尾运行时使用：

```
uv run "$SCRIPT" restart-run-all --path <notebook.ipynb> --save-outputs --compact
```

## 实践经验提示

1. **服务器启动后首次执行可能超时** —— 内核需要片刻时间初始化。如果超时，重试即可。

2. **内核 Python 是 JupyterLab 的 Python** —— 包必须安装在该环境中。如需额外的包，请先将其安装到 JupyterLab 工具环境中。

3. **`--compact` 标志可显著节省 token** —— 始终使用它。不加此标志时 JSON 输出可能非常冗长。

4. **纯 REPL 使用时**，创建一个 scratch.ipynb，无需关心单元格编辑。反复使用 `execute` 即可。

5. **参数顺序很重要** —— 子命令标志（如 `--path`）必须放在子子命令**之前**。例如：`variables --path nb.ipynb list`，而非 `variables list --path nb.ipynb`。

6. **如果会话尚不存在**，需要通过 REST API 启动一个（参见"设置"部分）。没有实时内核会话，工具无法执行代码。

7. **错误以 JSON 形式返回**，包含 traceback —— 读取 `ename` 和 `evalue` 字段以了解出错原因。

8. **偶发的 websocket 超时** —— 某些操作（尤其是内核重启后）首次尝试可能超时。在上报问题前先重试一次。

## 超时默认值

脚本每次执行的默认超时为 30 秒。对于长时间运行的操作，传入 `--timeout 120`。初始设置或大量计算时，建议使用较宽松的超时值（60 秒以上）。