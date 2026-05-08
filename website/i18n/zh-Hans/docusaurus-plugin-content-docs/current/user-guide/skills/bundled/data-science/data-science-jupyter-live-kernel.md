---
title: "Jupyter 实时内核 — 通过实时 Jupyter 内核进行迭代式 Python 开发 (hamelnb)"
sidebar_label: "Jupyter 实时内核"
description: "通过实时 Jupyter 内核进行迭代式 Python 开发 (hamelnb)"
---

{/* 此页面由网站脚本 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Jupyter 实时内核

通过实时 Jupyter 内核进行迭代式 Python 开发 (hamelnb)。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑安装（默认安装） |
| 路径 | `skills/data-science/jupyter-live-kernel` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `jupyter`, `notebook`, `repl`, `data-science`, `exploration`, `iterative` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes Agent 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Jupyter 实时内核 (hamelnb)

通过实时 Jupyter 内核为您提供**有状态的 Python REPL**。变量在执行之间保持不变。当您需要逐步建立状态、探索 API、检查 DataFrame 或迭代复杂代码时，请使用此工具而不是 `execute_code`。

## 何时使用此工具与其他工具

| 工具 | 使用场景 |
|------|----------|
| **此技能** | 迭代式探索、跨步骤保持状态、数据科学、机器学习、"让我试试这个并检查" |
| `execute_code` | 需要 hermes 工具访问（web_search、文件操作）的一次性脚本。无状态。 |
| `terminal` | Shell 命令、构建、安装、git、进程管理 |

**经验法则：** 如果您为此任务想要一个 Jupyter notebook，请使用此技能。

## 前置条件

1. **uv** 必须已安装（检查：`which uv`）
2. **JupyterLab** 必须已安装：`uv tool install jupyterlab`
3. Jupyter 服务器必须正在运行（见下面的设置）

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

检查服务器是否已运行：
```
uv run "$SCRIPT" servers
```

如果未找到服务器，启动一个：
```
jupyter-lab --no-browser --port=8888 --notebook-dir=$HOME/notebooks \
  --IdentityProvider.token='' --ServerApp.password='' > /tmp/jupyter.log 2>&1 &
sleep 3
```

注意：已禁用令牌/密码以供本地代理访问。服务器在无头模式下运行。

### 创建用于 REPL 使用的笔记本

如果您只需要 REPL（没有现有笔记本），创建一个最小的笔记本文件：
```
mkdir -p ~/notebooks
```
编写一个包含一个空代码单元的最小 .ipynb JSON 文件，然后通过 Jupyter REST API 启动内核会话：
```
curl -s -X POST http://127.0.0.1:8888/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"path":"scratch.ipynb","type":"notebook","name":"scratch.ipynb","kernel":{"name":"python3"}}'
```

## 核心工作流程

所有命令返回结构化 JSON。始终使用 `--compact` 以节省 token。

### 1. 发现服务器和笔记本

```
uv run "$SCRIPT" servers --compact
uv run "$SCRIPT" notebooks --compact
```

### 2. 执行代码（主要操作）

```
uv run "$SCRIPT" execute --path <notebook.ipynb> --code '<python code>' --compact
```

状态在执行调用之间保持不变。变量、导入、对象都会保留。

多行代码使用 $'...' 引号：
```
uv run "$SCRIPT" execute --path scratch.ipynb --code $'import os\nfiles = os.listdir(".")\nprint(f"Found {len(files)} files")' --compact
```

### 3. 检查实时变量

```
uv run "$SCRIPT" variables --path <notebook.ipynb> list --compact
uv run "$SCRIPT" variables --path <notebook.ipynb> preview --name <varname> --compact
```

### 4. 编辑笔记本单元格

```
# 查看当前单元格
uv run "$SCRIPT" contents --path <notebook.ipynb> --compact

# 插入新单元格
uv run "$SCRIPT" edit --path <notebook.ipynb> insert \
  --at-index <N> --cell-type code --source '<code>' --compact

# 替换单元格内容（使用 contents 输出中的 cell-id）
uv run "$SCRIPT" edit --path <notebook.ipynb> replace-source \
  --cell-id <id> --source '<new code>' --compact

# 删除单元格
uv run "$SCRIPT" edit --path <notebook.ipynb> delete --cell-id <id> --compact
```

### 5. 验证（重启 + 运行全部）

仅在用户要求干净验证或需要确认笔记本从顶到底可以运行时使用：

```
uv run "$SCRIPT" restart-run-all --path <notebook.ipynb> --save-outputs --compact
```

## 实践经验提示

1. **服务器启动后的首次执行可能会超时** — 内核需要片刻初始化。如果您超时，只需重试。

2. **内核 Python 是 JupyterLab 的 Python** — 包必须安装在该环境中。如果您需要额外的包，首先将它们安装到 JupyterLab 工具环境中。

3. **--compact 标志节省大量 token** — 始终使用它。没有它，JSON 输出可能非常冗长。

4. **对于纯 REPL 使用**，创建一个 scratch.ipynb，不必费心编辑单元格。只需重复使用 `execute`。

5. **参数顺序很重要** — 子命令标志如 `--path` 放在子子命令之前。例如：`variables --path nb.ipynb list` 而不是 `variables list --path nb.ipynb`。

6. **如果会话尚不存在**，您需要通过 REST API 启动一个（见设置部分）。工具没有实时内核会话无法执行。

7. **错误作为 JSON 返回**，包含 traceback — 阅读 `ename` 和 `evalue` 字段以了解出了什么问题。

8. **偶尔的 websocket 超时** — 某些操作可能会在首次尝试时超时，尤其是在内核重启后。重试一次后再升级。

## 超时默认值

脚本默认每次执行超时 30 秒。对于长时间运行的操作，传递 `--timeout 120`。对于初始设置或重型计算，使用较大的超时时间（60+）。
