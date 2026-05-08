---
title: "GitHub 代码审查 — 审查 PR：差异、行内评论（通过 gh 或 REST）"
sidebar_label: "GitHub 代码审查"
description: "审查 PR：差异、行内评论（通过 gh 或 REST）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# GitHub 代码审查

审查 PR：差异、行内评论（通过 gh 或 REST）。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/github/github-code-review` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `GitHub`、`代码审查`、`Pull Request`、`Git`、`质量` |
| 相关技能 | [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth)、[`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# GitHub 代码审查

在推送前审查本地更改，或在 GitHub 上审查开放的 PR。此技能的大部分内容使用纯 `git` — `gh`/`curl` 的区分仅在 PR 级别的交互中有意义。

## 前置条件

- 已通过 GitHub 认证（参见 `github-auth` 技能）
- 在 git 仓库中

### 设置（用于 PR 交互）

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 1. 审查本地更改（推送前）

这是纯 `git` — 在任何地方都能工作，不需要 API。

### 获取差异

```bash
# 暂存的更改（将被提交的内容）
git diff --staged

# 与 main 的所有更改（PR 将包含的内容）
git diff main...HEAD

# 仅文件名
git diff main...HEAD --name-only

# 统计摘要（每个文件的插入/删除）
git diff main...HEAD --stat
```

### 审查策略

1. **先看全局：**

```bash
git diff main...HEAD --stat
git log main..HEAD --oneline
```

2. **逐文件审查** — 对更改的文件使用 `read_file` 获取完整上下文，并查看差异了解更改了什么：

```bash
git diff main...HEAD -- src/auth/login.py
```

3. **检查常见问题：**

```bash
# 调试语句、TODO、console.log 残留
git diff main...HEAD | grep -n "print(\|console\.log\|TODO\|FIXME\|HACK\|XXX\|debugger"

# 意外暂存的大文件
git diff main...HEAD --stat | sort -t'|' -k2 -rn | head -10

# 密钥或凭证模式
git diff main...HEAD | grep -in "password\|secret\|api_key\|token.*=\|private_key"

# 合并冲突标记
git diff main...HEAD | grep -n "<<<<<<\|>>>>>>\|======="
```

4. **向用户展示结构化的反馈。**

### 审查输出格式

审查本地更改时，按以下结构展示发现：

```
## 代码审查摘要

### 严重
- **src/auth.py:45** — SQL 注入：用户输入直接传递给查询。
  建议：使用参数化查询。

### 警告
- **src/models/user.py:23** — 密码以明文存储。使用 bcrypt 或 argon2。
- **src/api/routes.py:112** — 登录端点无速率限制。

### 建议
- **src/utils/helpers.py:8** — 与 `src/core/utils.py:34` 的逻辑重复。合并。
- **tests/test_auth.py** — 缺少边缘情况：过期令牌测试。

### 看起来不错
- 中间件层的关注点分离清晰
- 正常路径的测试覆盖良好
```

---

## 2. 审查 GitHub 上的 Pull Request

### 查看 PR 详情

**使用 gh：**

```bash
gh pr view 123
gh pr diff 123
gh pr diff 123 --name-only
```

**使用 git + curl：**

```bash
PR_NUMBER=123

# 获取 PR 详情
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "
import sys, json
pr = json.load(sys.stdin)
print(f\"Title: {pr['title']}\")
print(f\"Author: {pr['user']['login']}\")
print(f\"Branch: {pr['head']['ref']} -> {pr['base']['ref']}\")
print(f\"State: {pr['state']}\")
print(f\"Body:\n{pr['body']}\")"

# 列出更改的文件
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/files \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin):
    print(f\"{f['status']:10} +{f['additions']:-4} -{f['deletions']:-4}  {f['filename']}\")"
```

### 本地检出 PR 进行完整审查

这使用纯 `git` — 不需要 `gh`：

```bash
# 获取 PR 分支并检出
git fetch origin pull/123/head:pr-123
git checkout pr-123

# 现在你可以使用 read_file、search_files、运行测试等

# 查看与基础分支的差异
git diff main...pr-123
```

**使用 gh（快捷方式）：**

```bash
gh pr checkout 123
```

### 在 PR 上发表评论

**一般 PR 评论 — 使用 gh：**

```bash
gh pr comment 123 --body "整体看起来不错，以下是一些建议。"
```

**一般 PR 评论 — 使用 curl：**

```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/issues/$PR_NUMBER/comments \
  -d '{"body": "整体看起来不错，以下是一些建议。"}'
```

### 发表行内审查评论

**单条行内评论 — 使用 gh（通过 API）：**

```bash
HEAD_SHA=$(gh pr view 123 --json headRefOid --jq '.headRefOid')

gh api repos/$OWNER/$REPO/pulls/123/comments \
  --method POST \
  -f body="可以用列表推导式简化这里。" \
  -f path="src/auth/login.py" \
  -f commit_id="$HEAD_SHA" \
  -f line=45 \
  -f side="RIGHT"
```

**单条行内评论 — 使用 curl：**

```bash
# 获取 head commit SHA
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/comments \
  -d "{
    \"body\": \"可以用列表推导式简化这里。\",
    \"path\": \"src/auth/login.py\",
    \"commit_id\": \"$HEAD_SHA\",
    \"line\": 45,
    \"side\": \"RIGHT\"
  }"
```

### 提交正式审查（批准/请求更改）

**使用 gh：**

```bash
gh pr review 123 --approve --body "LGTM！"
gh pr review 123 --request-changes --body "见行内评论。"
gh pr review 123 --comment --body "一些建议，没有阻塞问题。"
```

**使用 curl — 多评论审查以原子方式提交：**

```bash
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"COMMENT\",
    \"body\": \"来自 Hermes Agent 的代码审查\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"使用参数化查询防止 SQL 注入。\"},
      {\"path\": \"src/models/user.py\", \"line\": 23, \"body\": \"存储前用 bcrypt 哈希密码。\"},
      {\"path\": \"tests/test_auth.py\", \"line\": 1, \"body\": \"添加过期令牌边缘情况的测试。\"}
    ]
  }"
```

事件值：`"APPROVE"`、`"REQUEST_CHANGES"`、`"COMMENT"`

`line` 字段指的是文件*新*版本中的行号。对于已删除的行，使用 `"side": "LEFT"`。

---

## 3. 审查清单

执行代码审查时（本地或 PR），系统地检查：

### 正确性
- 代码是否实现了它声称的功能？
- 边缘情况是否处理（空输入、空值、大数据、并发访问）？
- 错误路径是否优雅地处理？

### 安全性
- 没有硬编码的密钥、凭证或 API 密钥
- 面向用户的输入有输入验证
- 没有 SQL 注入、XSS 或路径遍历
- 需要的地方有认证/授权检查

### 代码质量
- 清晰的命名（变量、函数、类）
- 没有不必要的复杂性或过早抽象
- DRY — 没有应该提取的重复逻辑
- 函数职责聚焦（单一职责）

### 测试
- 新的代码路径是否已测试？
- 正常路径和错误情况是否覆盖？
- 测试是否可读且可维护？

### 性能
- 没有 N+1 查询或不必要的循环
- 有益的地方有适当的缓存
- 异步代码路径中没有阻塞操作

### 文档
- 公共 API 已记录
- 不明显的逻辑有注释解释"为什么"
- 如果行为发生变化，README 是否已更新

---

## 4. 推送前审查工作流

当用户要求你"审查代码"或"推送前检查"时：

1. `git diff main...HEAD --stat` — 查看更改范围
2. `git diff main...HEAD` — 阅读完整差异
3. 对每个更改的文件，如果需要更多上下文使用 `read_file`
4. 应用上述检查清单
5. 以结构化格式展示发现（严重/警告/建议/看起来不错）
6. 如果发现严重问题，在用户推送前主动提出修复

---

## 5. PR 审查工作流（端到端）

当用户要求你"审查 PR #N"、"看一下这个 PR"或给你一个 PR URL 时，按以下步骤：

### 步骤 1：设置环境

```bash
source "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"
# 或运行此技能顶部的内联设置块
```

### 步骤 2：收集 PR 上下文

获取 PR 元数据、描述和更改文件列表，在深入代码之前了解范围。

**使用 gh：**
```bash
gh pr view 123
gh pr diff 123 --name-only
gh pr checks 123
```

**使用 curl：**
```bash
PR_NUMBER=123

# PR 详情（标题、作者、描述、分支）
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER

# 带行数统计的更改文件
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/files
```

### 步骤 3：本地检出 PR

这让你完全访问 `read_file`、`search_files` 和运行测试的能力。

```bash
git fetch origin pull/$PR_NUMBER/head:pr-$PR_NUMBER
git checkout pr-$PR_NUMBER
```

### 步骤 4：阅读差异并理解更改

```bash
# 与基础分支的完整差异
git diff main...HEAD

# 或对于大型 PR 逐文件查看
git diff main...HEAD --name-only
# 然后对每个文件：
git diff main...HEAD -- path/to/file.py
```

对每个更改的文件，使用 `read_file` 查看更改周围的完整上下文 — 仅看差异可能会遗漏只有周围代码才能看到的问题。

### 步骤 5：本地运行自动化检查（如适用）

```bash
# 如果有测试套件则运行
python -m pytest 2>&1 | tail -20
# 或：npm test, cargo test, go test ./..., 等

# 如果配置了 linter 则运行
ruff check . 2>&1 | head -30
# 或：eslint, clippy, 等
```

### 步骤 6：应用审查清单（第 3 节）

逐一检查每个类别：正确性、安全性、代码质量、测试、性能、文档。

### 步骤 7：将审查发布到 GitHub

收集你的发现并以正式审查+行内评论的形式提交。

**使用 gh：**
```bash
# 如果没有问题 — 批准
gh pr review $PR_NUMBER --approve --body "由 Hermes Agent 审查。代码看起来干净 — 测试覆盖良好，没有安全问题。"

# 如果发现问题 — 请求更改并附行内评论
gh pr review $PR_NUMBER --request-changes --body "发现了几个问题 — 见行内评论。"
```

**使用 curl — 带多条行内评论的原子审查：**
```bash
HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

# 构建审查 JSON — event 为 APPROVE、REQUEST_CHANGES 或 COMMENT
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"REQUEST_CHANGES\",
    \"body\": \"## Hermes Agent 审查\n\n发现 2 个问题，1 个建议。见行内评论。\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"🔴 **严重：** 用户输入直接传入 SQL 查询 — 使用参数化查询。\"},
      {\"path\": \"src/models.py\", \"line\": 23, \"body\": \"⚠️ **警告：** 密码未哈希存储。\"},
      {\"path\": \"src/utils.py\", \"line\": 8, \"body\": \"💡 **建议：** 这与 core/utils.py:34 的逻辑重复。\"}
    ]
  }"
```

### 步骤 8：同时发布摘要评论

除了行内评论，还要留下一个顶层摘要，让 PR 作者一目了然。使用 `references/review-output-template.md` 中的审查输出格式。

**使用 gh：**
```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
## 代码审查摘要

**结论：请求更改**（2 个问题，1 个建议）

### 🔴 严重
- **src/auth.py:45** — SQL 注入漏洞

### ⚠️ 警告
- **src/models.py:23** — 明文密码存储

### 💡 建议
- **src/utils.py:8** — 逻辑重复，考虑合并

### ✅ 看起来不错
- API 设计清晰
- 中间件层的错误处理良好

---
*由 Hermes Agent 审查*
EOF
)"
```

### 步骤 9：清理

```bash
git checkout main
git branch -D pr-$PR_NUMBER
```

### 决策：批准 vs 请求更改 vs 评论

- **批准** — 没有严重或警告级别的问题，只有小建议或一切正常
- **请求更改** — 任何应该在合并前修复的严重或警告级别问题
- **评论** — 观察和建议，没有阻塞问题（当你不确定或 PR 是草稿时使用）
